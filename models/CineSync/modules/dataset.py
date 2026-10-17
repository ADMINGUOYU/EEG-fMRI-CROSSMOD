# file: dataset.py
# libraries: numpy, torch, lmdb, av (only when video is selected)
# Caption-aware CineSync samples; the existing database remains authoritative.
from __future__ import annotations

import copy
import json
import os
import pickle
import tempfile
import random
import re
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dataset.database import Database
from dataset.dataset import Dataset


class Dataset_CineSync(Dataset):
    AVAILABLE_MODALITIES = ('fmri', 'eeg', 'video')
    ROI_RANGES = {'visual': (0, 8405), 'auditory': (8405, 18946), 'both': (0, 18946)}

    def __init__(self, db_path: str = './data/CineBrain_database/lmdb',
                 captions_path: str = './data/CineBrain/captions-qwen-2.5-vl-7b.json',
                 modalities: Sequence[str] = ('fmri', 'video'), fmri_roi: str = 'visual',
                 augment_captions: bool = False, video_cache_mb: int = 512,
                 **kwargs: Any) -> None:
        # index captions by filename, never by JSON position (the release is unordered)
        entries = json.loads(Path(captions_path).read_text(encoding = 'utf-8'))
        self.captions: dict[int, str] = {}
        for entry in entries:
            clip_id = int(Path(entry['video']).stem)
            if clip_id in self.captions:
                raise ValueError(f"Duplicate caption for clip {clip_id:06d}.")
            self.captions[clip_id] = str(entry['text'])
        if not modalities or len(set(modalities)) != len(modalities) or any(x not in self.AVAILABLE_MODALITIES for x in modalities):
            raise ValueError(f"Select unique modalities from {self.AVAILABLE_MODALITIES}.")
        if fmri_roi not in self.ROI_RANGES:
            raise ValueError(f"Unknown fMRI ROI '{fmri_roi}'.")
        self.fmri_roi = fmri_roi
        self.augment_captions = augment_captions
        self._pid = os.getpid()
        self._closed = False

        # lazily retain decoded clips, bounded per worker; zero disables video caching
        if video_cache_mb < 0:
            raise ValueError('Video cache size must be non-negative.')
        self._video_cache_limit = int(video_cache_mb * 1024 * 1024) if 'video' in modalities else 0
        self._video_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._video_cache_bytes = 0

        # added for embedding caching
        self.embeddings: dict[str, np.ndarray] = {}
        self._embedding_rows: dict[int, int] = {}

        try:
            super().__init__(db_path = db_path, modalities = modalities, **kwargs)
            if self.trial_indices:
                self._validate_trial(self.db.get_db_attribute(f'meta.trial_{self.trial_indices[0]}'))
        except BaseException:
            if getattr(self, 'db', None) is not None:
                self.db.close()
            raise
        # each DataLoader worker opens its own environment on first access
        self.db.close()
        self.db = None

    def _ensure_database(self) -> None:
        # discard an inherited handle after fork and reopen lazily after spawn
        if self._closed:
            raise RuntimeError('Dataset is closed.')
        if self.db is not None and self._pid != os.getpid():
            self.db.close()
            self.db = None
        if self.db is None:
            self.db = Database(self.db_path, mode = 'r')
            self._pid = os.getpid()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state['db'] = None
        # spawned workers build their own caches without copying decoded parent videos
        state['_video_cache'] = OrderedDict()
        state['_video_cache_bytes'] = 0
        return state

    def _validate_trial(self, meta: dict[str, Any]) -> int:
        # accept the existing database contract; only constrain the replica's duration
        shape = meta['data'].get('fmri', {}).get('shape')
        duration = meta.get('cinebrain', {}).get('duration_seconds')
        if not ((shape is not None and shape[0] == 5) or (duration is not None and abs(duration - 4.0) < 1e-6)):
            raise ValueError('CineSync requires five fMRI frames or a four-second trial.')
        # standard segments identify one original stimulus and therefore one caption
        name = Path(meta['data']['video']['path']).stem
        match = re.fullmatch(r'(\d+)_(\d+)_(\d+)_(\d+)', name)
        if match is None:
            raise ValueError(f"Cannot extract a caption clip ID from '{name}'.")
        first, _, last, _ = map(int, match.groups())
        if first != last:
            raise ValueError(f"CineSync needs one standard source clip per trial, got '{name}'.")
        return first

    def load_trial(self, trial_idx: int) -> dict[str, Any]:
        self._ensure_database()
        meta = self.db.get_db_attribute(f'meta.trial_{trial_idx}')
        clip_id = self._validate_trial(meta)
        # load only requested neural modalities; use metadata directly for the video path
        data: dict[str, Any] = {}
        for name in self._resolve_modalities(meta):
            if name == 'video':
                import av
                path = meta['data']['video']['path']
                # reuse recent clips by path, including trials referring to the same video
                if path in self._video_cache:
                    self._video_cache.move_to_end(path)
                    data[name] = self._video_cache[path].copy()
                    continue
                with av.open(str(path)) as container:
                    data[name] = np.stack([f.to_ndarray(format = 'rgb24') for f in container.decode(video = 0)])
                if data[name].shape[0] != 33:
                    raise ValueError(f"Expected 33 video frames in '{path}'.")
                # full-sample preloading already caches videos; avoid storing them twice
                video = data[name]
                if not self.preload and video.nbytes <= self._video_cache_limit:
                    while self._video_cache_bytes + video.nbytes > self._video_cache_limit:
                        _, evicted = self._video_cache.popitem(last = False)
                        self._video_cache_bytes -= evicted.nbytes
                    self._video_cache[path] = video
                    self._video_cache_bytes += video.nbytes
                    # caller transforms must not mutate the reusable decoded clip
                    data[name] = video.copy()
            else:
                value = self.db.read_trial_modality(trial_idx, name)
                if name == 'fmri':
                    start, stop = self.ROI_RANGES[self.fmri_roi]
                    if value.shape != (5, 18946):
                        raise ValueError(f'Expected fMRI (5, 18946), got {value.shape}.')
                    value = value[:, start:stop]
                data[name] = np.asarray(value, dtype = np.float32)
        sample = {'data': data, 'caption': self.captions[clip_id], 'clip_id': clip_id}
        if self.return_trial_idx:
            sample['trial_idx'] = trial_idx
        if self.return_meta:
            sample['meta'] = meta
        return sample

    def process_sample(self, sample: dict[str, Any], trial_idx: int | None = None) -> dict[str, Any]:
        # cached text represents the full caption, so never augment it after encoding
        if self.embeddings:
            row = self._embedding_rows[trial_idx]
            sample['embeddings'] = {name: value[row].copy() for name, value in self.embeddings.items()}
        elif self.augment_captions:
            sentences = sample['caption'].split('.')
            if len(sentences) >= 2:
                start = random.randrange(len(sentences) - 1)
                sample['caption'] = '.'.join(sentences[start:start + 2]) + '.'
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError('Dataset is closed.')
        return super().__getitem__(index)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        if not batch:
            return {}
        # video may retain differing source sizes until the model-specific processor runs
        data = {}
        for name in batch[0]['data']:
            values = [sample['data'][name] for sample in batch]
            data[name] = values if name == 'video' else torch.from_numpy(np.stack(values))
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

    def refresh(self) -> None:
        self._video_cache.clear()
        self._video_cache_bytes = 0
        self.embeddings = {}
        self._embedding_rows = {}
        self._ensure_database()
        super().refresh()

    def print_brief(self) -> None:
        print(f"\033[1;34m[CineSync] INFO\033[0m: {len(self)} trials; modalities {self.modalities}; fMRI ROI '{self.fmri_roi}'.")
        if self._video_cache_limit and not self.preload:
            print(f"\033[1;34m[CineSync] INFO\033[0m: Lazy video cache: {self._video_cache_limit // (1024 * 1024)} MiB per worker.")

    def close(self) -> None:
        self._closed = True
        self._video_cache.clear()
        self._video_cache_bytes = 0
        self.embeddings = {}
        self._embedding_rows = {}
        self._cache.clear()
        super().close()

    # added for embedding caching
    def _embedding_cache_path(self, base_path: str | Path) -> Path:
        # share embeddings across CineSync models, keeping each selection separate
        subject = self.subject_id if self.subject_id is not None else 'all'
        phase = self.phase if self.phase is not None else 'all'
        state = f'_state_{self.state}' if self.state is not None else ''
        return Path(base_path) / f'cached_emb_subj{subject}_phase_{phase}{state}.pkl'

    def generate_cache_embeddings(self, base_path: str | Path,
                                  encoder: Callable[[list[np.ndarray], list[str]], tuple[torch.Tensor, torch.Tensor]],
                                  batch_size: int = 1) -> None:
        # Encode only selected trials; the caller supplies the teacher and device.
        if batch_size < 1 or not self.trial_indices:
            raise ValueError('Embedding generation requires a positive batch size and a non-empty selection.')
        path = self._embedding_cache_path(base_path)
        path.parent.mkdir(parents = True, exist_ok = True)
        indices = list(self.trial_indices)
        cached = {'trial_indices': np.asarray(indices, dtype = np.int64)}
        modalities = self.modalities
        temporary = None
        try:
            # load videos and full captions directly, bypassing augmentation and sample preloading
            self.modalities = ('video',)
            with torch.inference_mode():
                for start in range(0, len(indices), batch_size):
                    samples = [self.load_trial(index) for index in indices[start:start + batch_size]]
                    visual, textual = encoder([s['data']['video'] for s in samples], [s['caption'] for s in samples])
                    for name, value in (('visual', visual), ('textual', textual)):
                        array = value.detach().float().cpu().numpy()
                        if array.ndim < 2 or array.shape[0] != len(samples):
                            raise ValueError(f"Invalid '{name}' embedding batch shape: {array.shape}.")
                        if start == 0:
                            cached[name] = np.empty((len(indices), *array.shape[1:]), dtype = array.dtype)
                        cached[name][start:start + len(samples)] = array
                    print(f"\033[1;34m[CineSync] INFO\033[0m: Cached embeddings for {start + len(samples)}/{len(indices)} trials.", flush = True)
            # publish the complete pickle atomically after all selected trials are encoded
            with tempfile.NamedTemporaryFile(dir = path.parent, suffix = '.tmp', delete = False) as stream:
                temporary = Path(stream.name)
                pickle.dump(cached, stream, protocol = pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
            print(f"\033[1;32m[CineSync] OK\033[0m: Saved embeddings to '{path}'.")
        finally:
            self.modalities = modalities
            if temporary is not None:
                temporary.unlink(missing_ok = True)

    def load_cache_embeddings(self, base_path: str | Path) -> bool:
        # Load the selected trials' embeddings, returning False if unavailable or invalid.
        print("\033[1;33m[CineSync] WARN\033[0m: You are responsible for selecting the correct embedding cache; the generating model is not recorded or checked.")
        self.embeddings = {}
        self._embedding_rows = {}
        path = self._embedding_cache_path(base_path)
        try:
            # verify row alignment, without inspecting the teacher or its configuration
            with path.open('rb') as stream:
                cached = pickle.load(stream)
            if not np.array_equal(cached['trial_indices'], self.trial_indices):
                raise ValueError('Cached trial indices do not match the current selection.')
            for name in ('visual', 'textual'):
                value = cached[name]
                if not isinstance(value, np.ndarray) or value.ndim < 2 or value.shape[0] != len(self):
                    raise ValueError(f"Invalid '{name}' embedding matrix.")
            self.embeddings = {name: cached[name] for name in ('visual', 'textual')}
            self._embedding_rows = {index: row for row, index in enumerate(self.trial_indices)}
        except FileNotFoundError:
            print(f"\033[1;34m[CineSync] INFO\033[0m: No embedding cache at '{path}'.")
            return False
        except Exception as error:
            print(f"\033[1;33m[CineSync] WARN\033[0m: Cannot load embedding cache '{path}': {error}")
            return False
        print(f"\033[1;32m[CineSync] OK\033[0m: Loaded embeddings for {len(self)} trials from '{path}'.")
        return True
