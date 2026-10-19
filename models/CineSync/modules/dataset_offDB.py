# file: dataset_offDB.py
# libraries: numpy, torch, av (only when video is selected)
# Standalone CineSync dataset reading the supplied signals and original clips.
# to use this, amend imports in ./models/CineSync/modules/runtime.py
from __future__ import annotations

# [SET UP INFO] ==========================
SUBJECT_IDS = (1, 2, 3, 4, 5, 6)
SUBJECT_DIRECTORY = 'sub-{subject_id:04d}'
EEG_DIRECTORY = 'eeg_02'
FMRI_DIRECTORY = 'visual_audio'
VIDEO_DIRECTORY = 'clips'
BEHAVIOUR_DIRECTORY = 'sub{subject_id:02d}_beh'
BEHAVIOUR_FILE = 'sub{subject_id:02d}_{run}.csv'
NEURAL_FILE = '{index}.npy'
VIDEO_FILE = '{index:06d}.mp4'
RUNS_PER_SUBJECT = 20
FRAMES_PER_RUN = 1350
FRAMES_PER_TRIAL = 5
CLIPS_PER_RUN = 270
FMRI_TR_SECONDS = 0.8
CLIP_DURATION_SECONDS = 4.0
SOURCE_FRAMES_PER_CLIP = 33
TRAIN_RUNS = tuple(range(1, 19))
VAL_RUNS = ()
TEST_RUNS = (19, 20)
TRIAL_STATE = 'task'
EEG_CHANNEL_COUNT = 64
EEG_SAMPLES_PER_FRAME = 800
FMRI_VOXEL_COUNT = 18946
# As in gen_database.py, numbering is inferred because official annotations are absent.
VIDEO_EPISODE_ORDER = tuple((season, episode) for season in (7, 9, 11) for episode in range(1, 11))
SUBJECT_SEASONS = {1: (7, 9), 2: (7, 9), 3: (7, 11), 4: (7, 11), 5: (7, 11), 6: (7, 9)}
EPISODES_PER_SEASON = 10
NEURAL_FIRST_INDEX = 0
VIDEO_FIRST_INDEX = 0
NEURAL_RUN_STARTS: dict[tuple[int, int], int] = {}
VIDEO_EPISODE_STARTS: dict[tuple[int, int], int] = {}
ALLOW_SCHEDULE_FALLBACK = True
# =======================================

import copy
import csv
import json
import pickle
import random
import re
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


class Dataset_CineSync:
    AVAILABLE_MODALITIES = ('fmri', 'eeg', 'video')
    ROI_RANGES = {'visual': (0, 8405), 'auditory': (8405, 18946), 'both': (0, 18946)}

    def __init__(self, db_path: str = './data/CineBrain_database/lmdb', # place holder
                 captions_path: str = './data/CineBrain/captions-qwen-2.5-vl-7b.json',
                 modalities: Sequence[str] = ('fmri', 'video'), fmri_roi: str = 'visual',
                 augment_captions: bool = False, video_cache_mb: int = 512,
                 *, trial_indices: Sequence[int] | None = None, subject_id: int | None = None,
                 phase: str | None = None, state: str | None = None,
                 return_trial_idx: bool = True, return_meta: bool = True, preload: bool = False,
                 transform: Callable[[dict[str, Any]], Any] | None = None) -> None:
        
        self.root = Path(captions_path).resolve().parent
        self.subject_id, self.phase, self.state = subject_id, phase, state
        self.modalities = tuple(modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities) or any(name not in self.AVAILABLE_MODALITIES for name in self.modalities):
            raise ValueError(f'Select unique modalities from {self.AVAILABLE_MODALITIES}.')
        if fmri_roi not in self.ROI_RANGES or video_cache_mb < 0:
            raise ValueError('Select a valid fMRI ROI and a non-negative video cache size.')
        self.fmri_roi, self.augment_captions = fmri_roi, augment_captions
        self.return_trial_idx, self.return_meta = return_trial_idx, return_meta
        self.preload, self.transform = preload, transform
        self._closed = False
        self._cache: dict[int, dict[str, Any]] = {}
        self._video_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._video_cache_limit = int(video_cache_mb * 1024 * 1024) if 'video' in self.modalities else 0
        self._video_cache_bytes = 0
        self.embeddings: dict[str, np.ndarray] = {}
        self._embedding_rows: dict[int, int] = {}
        self._trials: dict[int, dict[str, Any]] = {}
        self.captions: dict[int, str] = {}
        for entry in json.loads(Path(captions_path).read_text(encoding = 'utf-8')):
            clip_id = int(Path(entry['video']).stem)
            if clip_id in self.captions:
                raise ValueError(f'Duplicate caption for clip {clip_id:06d}.')
            self.captions[clip_id] = str(entry['text'])
        self.trial_indices = self._resolve_trial_indices(trial_indices = trial_indices,
            subject_id = subject_id, phase = phase, state = state)
        if self.preload:
            self._preload_samples()
        self.print_brief()

    def _episode_identity(self, subject_id: int, run: int) -> tuple[int, int]:
        # behaviour determines episode identity; only missing identities use the schedule
        path = self.root / BEHAVIOUR_DIRECTORY.format(subject_id = subject_id)
        path /= BEHAVIOUR_FILE.format(subject_id = subject_id, run = run)
        episodes = set()
        if path.is_file():
            with path.open(newline = '', encoding = 'utf-8-sig') as stream:
                for row in csv.DictReader(stream):
                    stimulus = row.get('stim_path') or ''
                    match = re.search(r'S(\d+)E(\d+)', stimulus, flags = re.IGNORECASE)
                    if match:
                        episodes.add((int(match[1]), int(match[2])))
                    elif stimulus.strip():
                        raise ValueError(f"Cannot interpret stimulus in '{path}': {stimulus}")
        if len(episodes) > 1:
            raise ValueError(f"Conflicting episode identities in '{path}'.")
        if episodes:
            return episodes.pop()
        if not ALLOW_SCHEDULE_FALLBACK:
            raise ValueError(f"Missing episode identity in '{path}'.")
        season_index, episode_index = divmod(run - 1, EPISODES_PER_SEASON)
        episode = (SUBJECT_SEASONS[subject_id][season_index], episode_index + 1)
        print(f'\033[1;33m[CineSync offDB] WARN\033[0m: Subject {subject_id}, run {run}: inferred S{episode[0]:02d}E{episode[1]:02d} from the schedule.')
        return episode

    def _resolve_trial_indices(self, *, trial_indices: Sequence[int] | None,
                               subject_id: int | None, phase: str | None, state: str | None) -> list[int]:
        # fixed source-grid IDs remain stable across filtering and missing files
        if sorted((*TRAIN_RUNS, *VAL_RUNS, *TEST_RUNS)) != list(range(1, RUNS_PER_SUBJECT + 1)):
            raise ValueError('Phase configuration must partition the subject runs.')
        if FRAMES_PER_RUN != CLIPS_PER_RUN * FRAMES_PER_TRIAL or FRAMES_PER_TRIAL != 5:
            raise ValueError('This dataset requires five neural frames per four-second clip.')
        if subject_id is not None and subject_id not in SUBJECT_IDS:
            raise ValueError(f'Unknown subject {subject_id}.')
        if phase not in (None, 'train', 'val', 'test'):
            raise ValueError(f'Unknown phase {phase!r}.')
        requested = None if trial_indices is None else {int(index) for index in trial_indices}
        starts = {episode: VIDEO_FIRST_INDEX + i * CLIPS_PER_RUN for i, episode in enumerate(VIDEO_EPISODE_ORDER)}
        starts.update(VIDEO_EPISODE_STARTS)
        self._trials = {}
        if state not in (None, TRIAL_STATE):
            return []
        # scan filenames once per subject; never close gaps by renumbering source files
        videos = {int(p.stem) for p in (self.root / VIDEO_DIRECTORY).glob('*.mp4') if p.stem.isdigit() and p.name == VIDEO_FILE.format(index = int(p.stem))}
        missing = 0
        for subject in SUBJECT_IDS:
            if subject_id is not None and subject != subject_id:
                continue
            subject_path = self.root / SUBJECT_DIRECTORY.format(subject_id = subject)
            neural_files = {}
            for name, directory in (('fmri', FMRI_DIRECTORY), ('eeg', EEG_DIRECTORY)):
                neural_files[name] = {int(p.stem) for p in (subject_path / directory).glob('*.npy')
                                      if p.stem.isdigit() and p.name == NEURAL_FILE.format(index = int(p.stem))}
            for run in range(1, RUNS_PER_SUBJECT + 1):
                split = 'train' if run in TRAIN_RUNS else 'val' if run in VAL_RUNS else 'test'
                if phase is not None and split != phase:
                    continue
                first_id = ((subject - 1) * RUNS_PER_SUBJECT + run - 1) * CLIPS_PER_RUN
                candidates = [offset for offset in range(CLIPS_PER_RUN)
                              if requested is None or first_id + offset in requested]
                if not candidates:
                    continue
                episode = self._episode_identity(subject, run)
                neural_start = NEURAL_RUN_STARTS.get((subject, run), NEURAL_FIRST_INDEX + (run - 1) * FRAMES_PER_RUN)
                for offset in candidates:
                    first = neural_start + offset * FRAMES_PER_TRIAL
                    clip_id = starts[episode] + offset
                    if clip_id not in videos or any(i not in neural_files[name] for name in ('fmri', 'eeg') for i in range(first, first + FRAMES_PER_TRIAL)):
                        missing += 1
                        continue
                    if clip_id not in self.captions:
                        raise ValueError(f'Missing caption for clip {clip_id:06d}.')
                    # cache full paths once, so item access only loads the ten relevant arrays
                    paths = {name: tuple(str(subject_path / directory / NEURAL_FILE.format(index = i))
                                        for i in range(first, first + FRAMES_PER_TRIAL))
                             for name, directory in (('fmri', FMRI_DIRECTORY), ('eeg', EEG_DIRECTORY))}
                    self._trials[first_id + offset] = {
                        'subject_id': subject, 'phase': split, 'state': TRIAL_STATE,
                        'run': run, 'season': episode[0], 'episode': episode[1], 'clip_id': clip_id,
                        'neural_start': first, 'duration_seconds': CLIP_DURATION_SECONDS,
                        'data': {**{name: {'paths': value} for name, value in paths.items()},
                                 'video': {'path': str(self.root / VIDEO_DIRECTORY / VIDEO_FILE.format(index = clip_id))}}}
        if missing:
            print(f'\033[1;33m[CineSync offDB] WARN\033[0m: Omitted {missing} incomplete source-grid trials; later indices are unchanged.')
        return sorted(self._trials)

    def __len__(self) -> int:
        return len(self.trial_indices)

    def _resolve_trial_idx(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f'Dataset index {index} is out of range.')
        return int(self.trial_indices[index])

    def _resolve_modalities(self, trial_meta: dict[str, Any]) -> list[str]:
        return list(self.modalities)

    def __getitem__(self, index: int) -> Any:
        if self._closed:
            raise RuntimeError('Dataset is closed.')
        trial_idx = self._resolve_trial_idx(index)
        if trial_idx in self._cache:
            sample = self._clone_sample(self._cache[trial_idx])
        else:
            sample = self.load_trial(trial_idx)
            if self.preload:
                self._cache[trial_idx] = self._clone_sample(sample)
        sample = self.process_sample(sample, trial_idx = trial_idx)
        return self.transform(sample) if self.transform is not None else sample

    def load_trial(self, trial_idx: int) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError('Dataset is closed.')
        meta = self._trials[trial_idx]
        data: dict[str, Any] = {}
        for name in self._resolve_modalities(meta):
            if name == 'video':
                import av
                path = meta['data']['video']['path']
                if path in self._video_cache:
                    self._video_cache.move_to_end(path)
                    data[name] = self._video_cache[path].copy()
                    continue
                with av.open(path) as container:
                    # the official loader selects the first 33 frames from each original clip
                    frames = []
                    for frame in container.decode(video = 0):
                        frames.append(frame.to_ndarray(format = 'rgb24'))
                        if len(frames) == SOURCE_FRAMES_PER_CLIP:
                            break
                if len(frames) != SOURCE_FRAMES_PER_CLIP:
                    raise ValueError(f"Expected 33 frames in '{path}'.")
                video = np.stack(frames)
                data[name] = video
                if not self.preload and video.nbytes <= self._video_cache_limit:
                    while self._video_cache_bytes + video.nbytes > self._video_cache_limit:
                        _, evicted = self._video_cache.popitem(last = False)
                        self._video_cache_bytes -= evicted.nbytes
                    self._video_cache[path] = video
                    self._video_cache_bytes += video.nbytes
                    data[name] = video.copy()
            else:
                # stack raw values like the official loader; only select ROI/EEG channels
                values = [np.load(path, allow_pickle = False) for path in meta['data'][name]['paths']]
                if name == 'fmri':
                    if any(value.shape != (FMRI_VOXEL_COUNT,) for value in values):
                        raise ValueError('Expected one voxel vector per fMRI source file.')
                    start, stop = self.ROI_RANGES[self.fmri_roi]
                    data[name] = np.stack([value[start:stop] for value in values])
                else:
                    if any(value.ndim != 2 or value.shape[0] < EEG_CHANNEL_COUNT or value.shape[1] != EEG_SAMPLES_PER_FRAME for value in values):
                        raise ValueError('Expected (at least 64 channels, 800 samples) per EEG source file.')
                    data[name] = np.stack([value[:EEG_CHANNEL_COUNT] for value in values])
        sample = {'data': data, 'caption': self.captions[meta['clip_id']], 'clip_id': meta['clip_id']}
        if self.return_trial_idx:
            sample['trial_idx'] = trial_idx
        if self.return_meta:
            sample['meta'] = copy.deepcopy(meta)
        return sample

    def __getstate__(self) -> dict[str, Any]:
        # workers keep mappings and embeddings, but start with an empty decoded-video cache
        state = self.__dict__.copy()
        state['_video_cache'] = OrderedDict()
        state['_video_cache_bytes'] = 0
        return state

    def _preload_samples(self) -> None:
        for index in self.trial_indices:
            self._cache[index] = self.load_trial(index)

    def _collect_selected_meta(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self._trials[index]) for index in self.trial_indices]

    def refresh(self) -> None:
        if self._closed:
            raise RuntimeError('Dataset is closed.')
        # rebuild from the active filters, matching the base dataset's refresh semantics
        self._cache.clear()
        self._video_cache.clear()
        self._video_cache_bytes = 0
        self.embeddings = {}
        self._embedding_rows = {}
        self.trial_indices = self._resolve_trial_indices(trial_indices = None,
            subject_id = self.subject_id, phase = self.phase, state = self.state)
        if self.preload:
            self._preload_samples()

    def print_brief(self) -> None:
        print(f"\033[1;34m[CineSync offDB] INFO\033[0m: {len(self)} source trials at '{self.root}'; modalities {self.modalities}; fMRI ROI '{self.fmri_roi}'.")

    def close(self) -> None:
        self._closed = True
        self._cache.clear()
        self._video_cache.clear()
        self._video_cache_bytes = 0
        self.embeddings = {}
        self._embedding_rows = {}

    def __enter__(self) -> Dataset_CineSync:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(root = {str(self.root)!r}, trials = {len(self)}, modalities = {self.modalities})'

    def process_sample(self, sample: dict[str, Any], trial_idx: int | None = None) -> dict[str, Any]:
        # sample the same adjacent sentence pairs with or without cached embeddings
        selection = 0
        if self.augment_captions:
            sentences = sample['caption'].split('.')
            if len(sentences) >= 2:
                start = random.randrange(len(sentences) - 1)
                sample['caption'] = '.'.join(sentences[start:start + 2]) + '.'
                selection = start + 1
        if self.embeddings:
            row = self._embedding_rows[trial_idx]
            text_row = int(self.embeddings['textual_offsets'][row]) + selection
            sample['embeddings'] = {'visual': self.embeddings['visual'][row].copy(),
                                    'textual': self.embeddings['textual'][text_row].copy()}
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}
        # video may retain differing source sizes until the model-specific processor runs
        data = {}
        for name in batch[0]['data']:
            values = [sample['data'][name] for sample in batch]
            # retain source dtypes in samples; match model precision only when collating
            data[name] = values if name == 'video' else torch.from_numpy(np.stack(values)).float()
        result = {'data': data, 'caption': [s['caption'] for s in batch],
                  'clip_id': torch.tensor([s['clip_id'] for s in batch])}
        if 'embeddings' in batch[0]:
            result['embeddings'] = {name: torch.from_numpy(np.stack([s['embeddings'][name] for s in batch]))
                                    for name in ('visual', 'textual')}
        if 'trial_idx' in batch[0]:
            result['trial_idx'] = torch.tensor([s['trial_idx'] for s in batch])
        if 'meta' in batch[0]:
            result['meta'] = [s['meta'] for s in batch]
        return result

    def _clone_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        # isolate optional cached arrays from user transforms
        return copy.deepcopy(sample)

    # added for embedding caching
    def _embedding_cache_path(self, base_path: str | Path) -> Path:
        # share embeddings across CineSync models, keeping each selection separate
        subject = self.subject_id if self.subject_id is not None else 'all'
        phase = self.phase if self.phase is not None else 'all'
        state = f'_state_{self.state}' if self.state is not None else ''
        return Path(base_path) / f'cached_emb_offDB_subj{subject}_phase_{phase}{state}.pkl'

    def generate_cache_embeddings(self, base_path: str | Path,
                                  encoder: Callable[[list[np.ndarray], list[str]], tuple[torch.Tensor, torch.Tensor]],
                                  batch_size: int = 1) -> None:
        # Encode only selected trials; the caller supplies the teacher and device.
        if batch_size < 1 or not self.trial_indices:
            raise ValueError('Embedding generation requires a positive batch size and a non-empty selection.')
        path = self._embedding_cache_path(base_path)
        path.parent.mkdir(parents = True, exist_ok = True)
        indices = list(self.trial_indices)
        # count full captions plus every adjacent pair before allocating contiguous arrays
        if self._closed:
            raise RuntimeError('Dataset is closed.')
        captions = [self.captions[self._trials[index]['clip_id']] for index in indices]
        # map each flat textual row to a trial; preserve upstream's empty split fragments
        selections: list[str] = []
        offsets = [0]
        for caption in captions:
            sentences = caption.split('.')
            selections.append(caption)
            selections.extend('.'.join(sentences[i:i + 2]) + '.' for i in range(len(sentences) - 1))
            offsets.append(len(selections))
        offsets = np.asarray(offsets, dtype = np.int64)
        trial_rows = np.repeat(np.arange(len(indices)), np.diff(offsets))
        cached = {'trial_indices': np.asarray(indices, dtype = np.int64), 'textual_offsets': offsets}
        modalities = self.modalities
        temporary = None
        print(f"\033[1;34m[CineSync offDB] INFO\033[0m: Encoding {len(indices)} full captions and {int(offsets[-1]) - len(indices)} adjacent sentence pairs.")
        try:
            # batch mapped selections across trials, retaining only the current batch's videos
            self.modalities = ('video',)
            previous_row, video = -1, None
            with torch.inference_mode():
                for start in range(0, len(selections), batch_size):
                    stop = min(start + batch_size, len(selections))
                    videos = []
                    for row in trial_rows[start:stop]:
                        # mappings are ordered, so even a trial crossing batches is decoded once
                        if row != previous_row:
                            video = self.load_trial(indices[row])['data']['video']
                            previous_row = row
                        videos.append(video)
                    visual, textual = encoder(videos, selections[start:stop])
                    visual = visual.detach().float().cpu().numpy()
                    textual = textual.detach().float().cpu().numpy()
                    if visual.ndim != 3 or textual.ndim != 2 or visual.shape[0] != stop - start or textual.shape[0] != stop - start:
                        raise ValueError('Invalid visual or textual embedding batch shape.')
                    if start == 0:
                        cached['visual'] = np.empty((len(indices), *visual.shape[1:]), dtype = visual.dtype)
                        cached['textual'] = np.empty((len(selections), textual.shape[1]), dtype = textual.dtype)
                    cached['textual'][start:stop] = textual
                    # store the visual result only at each trial's first mapped selection
                    for position, row in enumerate(trial_rows[start:stop]):
                        if start + position == offsets[row]:
                            cached['visual'][row] = visual[position]
                    print(f"\033[1;34m[CineSync offDB] INFO\033[0m: Cached {stop}/{len(selections)} selections ({100 * stop / len(selections):.1f}%).", flush = True)
                    del videos
            # publish the complete pickle atomically after all selected trials are encoded
            with tempfile.NamedTemporaryFile(dir = path.parent, suffix = '.tmp', delete = False) as stream:
                temporary = Path(stream.name)
                pickle.dump(cached, stream, protocol = pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
            print(f"\033[1;32m[CineSync offDB] OK\033[0m: Saved embeddings to '{path}'.")
        finally:
            # restore the requested modalities and remove any unfinished temporary file
            self.modalities = modalities
            if temporary is not None:
                temporary.unlink(missing_ok = True)

    def load_cache_embeddings(self, base_path: str | Path) -> bool:
        # Load the selected trials' embeddings, returning False if unavailable or invalid.
        print("\033[1;33m[CineSync offDB] WARN\033[0m: You are responsible for selecting the correct embedding cache; the generating model is not recorded or checked.")
        self.embeddings = {}
        self._embedding_rows = {}
        path = self._embedding_cache_path(base_path)
        try:
            # verify row alignment, without inspecting the teacher or its configuration
            with path.open('rb') as stream:
                cached = pickle.load(stream)
            if not np.array_equal(cached['trial_indices'], self.trial_indices):
                raise ValueError('Cached trial indices do not match the current selection.')
            offsets = cached['textual_offsets']
            if (not isinstance(offsets, np.ndarray) or offsets.shape != (len(self) + 1,)
                    or not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0
                    or np.any(np.diff(offsets) < 1)):
                raise ValueError('Invalid textual embedding offsets.')
            for name, count, dimensions in (('visual', len(self), 3), ('textual', int(offsets[-1]), 2)):
                value = cached[name]
                if not isinstance(value, np.ndarray) or value.ndim != dimensions or value.shape[0] != count:
                    raise ValueError(f"Invalid '{name}' embedding matrix.")
            self.embeddings = {name: cached[name] for name in ('visual', 'textual', 'textual_offsets')}
            self._embedding_rows = {index: row for row, index in enumerate(self.trial_indices)}
        except FileNotFoundError:
            print(f"\033[1;34m[CineSync offDB] INFO\033[0m: No embedding cache at '{path}'.")
            return False
        except Exception as error:
            print(f"\033[1;33m[CineSync offDB] WARN\033[0m: Cannot load embedding cache '{path}': {error}")
            return False
        print(f"\033[1;32m[CineSync offDB] OK\033[0m: Loaded embeddings for {len(self)} trials from '{path}'.")
        return True


# Entry point (testing)
# run with `python -m models.CineSync.modules.dataset_offDB`
if __name__ == '__main__':
    import av

    print('\033[1;34m[CineSync offDB] INFO\033[0m: Starting dataset smoke test.')
    # create one source trial in an isolated directory, removed automatically afterwards
    with tempfile.TemporaryDirectory(prefix = 'cinesync_offDB_') as temporary:
        root = Path(temporary)
        print(f"\033[1;34m[CineSync offDB] INFO\033[0m: Temporary dataset: '{root}'.")
        subject = root / SUBJECT_DIRECTORY.format(subject_id = 1)
        for directory in (subject / FMRI_DIRECTORY, subject / EEG_DIRECTORY,
                          root / VIDEO_DIRECTORY, root / BEHAVIOUR_DIRECTORY.format(subject_id = 1)):
            directory.mkdir(parents = True)
        captions_path = root / 'captions.json'
        captions_path.write_text(json.dumps([{'video': '000000.mp4', 'text': 'A person enters. They sit down.'}]), encoding = 'utf-8')
        behaviour = root / BEHAVIOUR_DIRECTORY.format(subject_id = 1) / BEHAVIOUR_FILE.format(subject_id = 1, run = 1)
        behaviour.write_text('stim_path\nS07E01.mp4\n', encoding = 'utf-8')

        # distinct frame values expose incorrect ordering; the extra EEG channel is ECG
        fmri = np.stack([np.full(FMRI_VOXEL_COUNT, i + 0.25, dtype = np.float64) for i in range(5)])
        eeg = np.stack([np.full((EEG_CHANNEL_COUNT + 1, EEG_SAMPLES_PER_FRAME), i + 0.5, dtype = np.float64) for i in range(5)])
        for i in range(5):
            np.save(subject / FMRI_DIRECTORY / NEURAL_FILE.format(index = i), fmri[i])
            np.save(subject / EEG_DIRECTORY / NEURAL_FILE.format(index = i), eeg[i])
        with av.open(str(root / VIDEO_DIRECTORY / '000000.mp4'), mode = 'w') as container:
            stream = container.add_stream('mpeg4', rate = 8)
            stream.width, stream.height, stream.pix_fmt = 16, 16, 'yuv420p'
            for i in range(SOURCE_FRAMES_PER_CLIP):
                frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 7, dtype = np.uint8), format = 'rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        # exercise source mapping, exact neural loading, transforms and collation
        with Dataset_CineSync(captions_path = str(captions_path), subject_id = 1, phase = 'train',
                             trial_indices = [0], modalities = ('fmri', 'eeg', 'video'), fmri_roi = 'both',
                             preload = True, transform = lambda sample: {**sample, 'transformed': True}) as dataset:
            assert len(dataset) == 1, 'Expected one selected trial.'
            sample = dataset[0]
            assert sample['trial_idx'] == 0 and sample['clip_id'] == 0
            assert sample['transformed'], 'The sample transform was not applied.'
            np.testing.assert_array_equal(sample['data']['fmri'], fmri)
            np.testing.assert_array_equal(sample['data']['eeg'], eeg[:, :EEG_CHANNEL_COUNT])
            assert sample['data']['video'].shape == (33, 16, 16, 3)
            sample['data']['fmri'][:] = -1
            np.testing.assert_array_equal(dataset[0]['data']['fmri'], fmri)
            batch = dataset.collate_fn([dataset[0]])
            assert batch['data']['fmri'].shape == (1, 5, FMRI_VOXEL_COUNT)
            assert batch['data']['eeg'].shape == (1, 5, EEG_CHANNEL_COUNT, EEG_SAMPLES_PER_FRAME)
            assert batch['data']['fmri'].dtype == torch.float32
            assert len(batch['data']['video']) == 1

    print('\033[1;32m[CineSync offDB] OK\033[0m: Test completed successfully.')
