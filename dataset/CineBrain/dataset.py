# file: dataset.py
# CineBrain extension of the standardised dataset wrapper.
# we implemented code to support multiple workers
# (the base class only spawns one shared LMDB database)

from __future__ import annotations

# import libraries
import copy
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

from dataset.database import Database
from dataset.dataset import Dataset


class Dataset_CineBrain(Dataset):
    # extend this tuple when additional modalities are supported by the database
    AVAILABLE_MODALITIES = ('fmri', 'eeg', 'video')

    # The paper (sections 3.1 and 7.1) confirms the two voxel counts:
    # https://arxiv.org/html/2503.06940v2
    # Upstream encoders use 8405 voxels; the visual/auditory configuration uses 18946:
    # https://github.com/yanweifu-sii/CineBrain/blob/main/configs/cogvideox_5b_lora_brain_va.yaml
    # The public loader does not specify concatenation order. These ranges follow
    # our fMRI_Helper.py's visual-first convention; override for a different export.
    FMRI_ROI_RANGES = {'visual': (0, 8405), 'auditory': (8405, 18946), 'both': (0, 18946)}
    FMRI_VOXEL_COUNT = 18946

    def __init__(
        self,
        db_path: str = './data/CineBrain_database/lmdb',
        trial_indices: Sequence[int] | None = None,
        subject_id: int | None = None,
        phase: str | None = None,
        state: str | None = None,
        modalities: Sequence[str] | None = None,
        return_trial_idx: bool = True,
        return_meta: bool = True,
        preload: bool = False,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        *,
        fmri_roi: str = 'both',
    ) -> None:
        """
        Initialise CineBrain Dataset

        Arguments:
            db_path (str): path to the CineBrain LMDB database directory
            trial_indices (Sequence[int] | None): optional subset of trial indices to load (only if you know)
            subject_id (int | None): optional subject ID to filter trials (only if your packed dataset has that subject)
            phase (str | None): optional phase to filter trials ('train', 'val', 'test', or None for all)
            state (str | None): optional state to filter trials (NOTE: not used for now)
            modalities (Sequence[str] | None): list of modalities to load ('fmri', 'eeg', 'video'); None selects all
            return_trial_idx (bool): whether to include the trial index in the returned sample
            return_meta (bool): whether to include the trial metadata in the returned sample
            preload (bool): whether to preload all selected trials into memory
            transform (Callable[[dict[str, Any]], Any] | None): optional function to apply to each sample, read base class for more
            fmri_roi (str): which fMRI ROI to load ('visual', 'auditory', or 'both'); only relevant if 'fmri' is in modalities
        
        NOTE:
        None selects AVAILABLE_MODALITIES; use ('eeg', 'fmri') to avoid video I/O.
        Samples retain the base {'data': ..., 'trial_idx': ..., 'meta': ...} layout.
        EEG is (channels, samples), fMRI is (frames, voxels), and video is
        (frames, height, width, RGB), uint8. No resizing or normalisation is applied.
        Use collate_fn with DataLoader to stack data and preserve metadata as a list.
        Preloading caches decoded samples; transforms still run on every access.
        """
        # validate selection before opening files or importing the video decoder
        selected = tuple(self.AVAILABLE_MODALITIES if modalities is None else modalities)
        if not selected or any(name not in self.AVAILABLE_MODALITIES for name in selected) or len(set(selected)) != len(selected):
            raise ValueError(f"\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Select unique modalities from {self.AVAILABLE_MODALITIES}.")
        if fmri_roi not in self.FMRI_ROI_RANGES:
            raise ValueError(f"\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Unknown fMRI ROI '{fmri_roi}'; choose from {tuple(self.FMRI_ROI_RANGES)}.")
        start, stop = self.FMRI_ROI_RANGES[fmri_roi]
        if not 0 <= start < stop <= self.FMRI_VOXEL_COUNT:
            raise ValueError("\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Invalid configured fMRI ROI range.")
        self.fmri_roi = fmri_roi
        self._fmri_slice = slice(start, stop)
        self._db_pid = os.getpid()
        self._closed = False
        try:
            super().__init__(
                db_path = db_path, trial_indices = trial_indices, subject_id = subject_id,
                phase = phase, state = state, modalities = selected,
                return_trial_idx = return_trial_idx, return_meta = return_meta,
                preload = preload, transform = transform,
            )
        except BaseException:
            if getattr(self, 'db', None) is not None:
                self.db.close()
            raise
        # workers open their own LMDB handles lazily; no environment is pickled
        # i.e. the override of get item will call _ensure_database() to reopen the
        # database in the worker process
        print(f"\033[1;34m[Dataset_CineBrain] INFO\033[0m: We will reopen the database in each worker process. (don't panic)")
        self.db.close()
        self.db = None

    def _ensure_database(self) -> None:
        # reopen after spawning or forking, never using another process's environment
        if self._closed:
            raise RuntimeError("\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Dataset is closed.")
        pid = os.getpid()
        if self.db is not None and self._db_pid != pid:
            self.db.close()
            self.db = None
        if self.db is None:
            self.db = Database(self.db_path, mode = 'r')
            self._db_pid = pid

    def __getstate__(self) -> dict[str, Any]:
        # remove the unpicklable LMDB handle from the state sent to spawned workers
        state = self.__dict__.copy()
        state['db'] = None
        return state

    def __getitem__(self, index: int) -> Any:
        # reject access after close even when every sample is already cached
        if self._closed:
            raise RuntimeError("\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Dataset is closed.")
        return super().__getitem__(index)

    def load_trial(self, trial_idx: int) -> dict[str, Any]:
        # the base loader reads only the selected modalities and retains their provenance
        self._ensure_database()
        sample = super().load_trial(trial_idx)
        data = sample['data']
        if 'fmri' in data:
            fmri = np.asarray(data['fmri'])
            if fmri.ndim != 2 or fmri.shape[-1] != self.FMRI_VOXEL_COUNT:
                raise ValueError(f"\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Trial {trial_idx} requires fMRI shape (frames, {self.FMRI_VOXEL_COUNT}), got {fmri.shape}.")
            # slice the voxel axis without changing the trial's time axis
            data['fmri'] = fmri[:, self._fmri_slice]
        if 'eeg' in data and np.ndim(data['eeg']) != 2:
            raise ValueError(f"\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Trial {trial_idx} requires EEG shape (channels, samples).")
        if 'video' in data:
            video = data['video']
            if isinstance(video, (str, os.PathLike)):
                # PyAV is only required for experiments that request encoded video
                try:
                    import av
                except ImportError as exc:
                    raise ImportError("\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Install PyAV with 'pip install av' to load MP4 clips.") from exc
                with av.open(os.fspath(video)) as container:
                    frames = [frame.to_ndarray(format = 'rgb24') for frame in container.decode(video = 0)]
                if not frames:
                    raise ValueError(f"\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Video '{video}' contains no frames.")
                video = np.stack(frames, axis = 0)
            # retain compatibility with earlier databases containing RGB arrays
            if not isinstance(video, np.ndarray) or video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
                raise ValueError(f"\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Trial {trial_idx} requires uint8 RGB video with shape (frames, height, width, 3).")
            data['video'] = video
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any] | list[Any]:
        """Stack equal-shaped modalities; keep arbitrary metadata outside tensor collation."""
        if not batch:
            return []
        # reject inconsistent selections instead of silently dropping modalities or samples
        keys = set(batch[0])
        modalities = set(batch[0]['data'])
        if any(set(sample) != keys or set(sample['data']) != modalities for sample in batch):
            raise ValueError("\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Batch samples must have matching keys and modalities.")
        result: dict[str, Any] = {'data': {}}
        for name in batch[0]['data']:
            values = [sample['data'][name] for sample in batch]
            shapes = [tuple(value.shape) for value in values]
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError(f"\033[1;31m[Dataset_CineBrain] ERROR\033[0m: Cannot stack '{name}' shapes {shapes}; use a transform to align sizes.")
            # stacking NumPy buffers first avoids exposing read-only LMDB arrays to PyTorch
            result['data'][name] = torch.from_numpy(np.stack(values, axis = 0)) if all(isinstance(value, np.ndarray) for value in values) else default_collate(values)
        for key in batch[0]:
            if key != 'data':
                values = [sample[key] for sample in batch]
                result[key] = values if key == 'meta' else default_collate(values)
        return result

    def _clone_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        # isolate cached arrays and metadata from in-place training transforms
        return copy.deepcopy(sample)

    def refresh(self) -> None:
        # restore the lazy connection before the base class rebuilds its selection
        self._ensure_database()
        super().refresh()

    def print_brief(self) -> None:
        # avoid scanning all trial metadata merely to print the initial configuration
        self._ensure_database()
        print(f"\033[1;34m[Dataset_CineBrain] INFO\033[0m: Dataset at '{self.db_path}'; {len(self)} trials.")
        print(f"  Modalities: {', '.join(self.modalities)}")
        if 'fmri' in self.modalities:
            print(f"  fMRI ROI: {self.fmri_roi} [{self._fmri_slice.start}:{self._fmri_slice.stop}]")
        print(f"  Preload: {'enabled' if self.preload else 'disabled'}")

    def close(self) -> None:
        # release cached video frames as well as the database handle
        self._closed = True
        self._cache.clear()
        super().close()


# Entry point
# run with `python -m dataset.CineBrain.dataset`
if __name__ == '__main__':

    # database path
    db_path = './data/CineBrain_database/lmdb'

    # create a dataset instance
    dataset = Dataset_CineBrain(
        db_path = db_path,
        modalities = ('fmri', 'eeg', 'video'),
        fmri_roi = 'both',
        preload = False)
    print(f"\n\033[1;34m[Dataset_CineBrain] INFO\033[0m: Dataset Loaded (length: {len(dataset)})")

    # load a sample and inspect its contents
    sample = dataset[0]
    # print keys and type of each item in the sample
    # print shape if it has and print the value if it is a scalar or a string
    # print every nested dict's key
    def print_sample_structure(d, prefix = ''): # recursive function
        for key, value in d.items():
            key_str = f"{prefix}\033[1m{key}\033[0m"
            if isinstance(value, dict):
                print(f"{key_str}: dict")
                print_sample_structure(value, prefix + '  ')
            elif isinstance(value, np.ndarray):
                print(f"{key_str}: ndarray, shape {value.shape}, dtype {value.dtype}")
            elif isinstance(value, torch.Tensor):
                print(f"{key_str}: tensor, shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"{key_str}: {type(value).__name__}, value: {value}")
    print_sample_structure(sample)

    # tidy up
    dataset.close()
