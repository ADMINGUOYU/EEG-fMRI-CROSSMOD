# file: dataset.py
# libraries: numpy, torch, lmdb, av (only when video is selected)
# Caption-aware CineSync samples; the existing database remains authoritative.
from __future__ import annotations

import copy
import json
import os
import random
import re
from collections.abc import Sequence
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
                 augment_captions: bool = False,
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
                with av.open(str(path)) as container:
                    data[name] = np.stack([f.to_ndarray(format = 'rgb24') for f in container.decode(video = 0)])
                if data[name].shape[0] != 33:
                    raise ValueError(f"Expected 33 video frames in '{path}'.")
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
        # reproduce the released loader's two-adjacent-sentence augmentation in training only
        if self.augment_captions:
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
        if 'trial_idx' in batch[0]:
            result['trial_idx'] = torch.tensor([s['trial_idx'] for s in batch])
        if 'meta' in batch[0]:
            result['meta'] = [s['meta'] for s in batch]
        return result

    def _clone_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        # isolate optional cached arrays from user transforms
        return copy.deepcopy(sample)

    def refresh(self) -> None:
        self._ensure_database()
        super().refresh()

    def print_brief(self) -> None:
        print(f"\033[1;34m[CineSync] INFO\033[0m: {len(self)} trials; modalities {self.modalities}; fMRI ROI '{self.fmri_roi}'.")

    def close(self) -> None:
        self._closed = True
        self._cache.clear()
        super().close()
