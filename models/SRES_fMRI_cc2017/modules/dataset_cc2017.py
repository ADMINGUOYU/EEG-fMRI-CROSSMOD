# file: dataset_cc2017.py
# Chronological windows from the published cc2017 voxel tensors;
# NOTE: Assume signal normalisation / Assume adjacent frames / Assume +x shift baked in (if applicable and configured)
# NOTE: Train/Val get from normal train / test is test
#       NEURONS uses full train, but this calibration setup should not cause data leaking

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from .runtime import log

# import as type checking
# NOTE: version specific Config refactored to version specific directory
if TYPE_CHECKING:
    from ..models.v1.config import Config as ConfigV1
    from ..models.v2.config import Config as ConfigV2

    Config = ConfigV1 | ConfigV2


class DatasetCC2017(Dataset):

    """
    Class to Yield a 'Window' of a few frames of continuous data
    We don't do reconstruction here, so just getting fMRI
    """

    def __init__(self, config: Config, phase: str, selection: str = 'all',
                 average_repeats: bool = False, source: torch.Tensor | None = None) -> None:
        # Validate numerical window settings before reading potentially large files.
        config.validate()
        # NOTE: assume adjacent frames <-> adjacent fMRI
        log('Assuming chronological samples within the configured runs and the configured stored delay', 'WARN')
        # validate phase
        if phase not in ('train', 'test') or selection not in ('all', 'training', 'validation'):
            raise ValueError('Invalid dataset phase or run selection.')
        self.config = config  # Retain the window length and stride for subsequent slicing.
        self.phase = phase    # Select the train/test file; validation uses held-out train runs.
        # Subject IDs are saved with two digits, for example subj01_train_fmri.pt.
        self.path = Path(config.data_root) / f'subj0{config.subject}_{phase}_fmri.pt'
        # Load once; a validation view may share the training tensor without duplication.
        data = source if source is not None else torch.load(self.path, map_location = 'cpu', weights_only = False)
        # Axis 0 is chronological movie TR, axis 1 (if present) is repetition, and the last axis is voxel.
        if not isinstance(data, torch.Tensor) or data.ndim not in (2, 3):
            raise ValueError('Expected [trial, repetition, voxel] or [trial, voxel]; no layout is inferred.')
        if data.ndim == 2:
            # A tensor without repetitions is represented internally as one presentation.
            data = data.unsqueeze(1)
        # Convert precision only: preserve the original means, variances and negative values.
        self.data = data.float()
        # The launcher declares how many chronological TRs belong to each movie run.
        self.run_lengths = [int(n) for n in (config.train_run_lengths if phase == 'train' else config.test_run_lengths).split(',')]
        # Exact coverage prevents dropped rows or accidental inference of a flattened repeat layout.
        if min(self.run_lengths) < config.window_trs or sum(self.run_lengths) != len(self.data):
            raise ValueError(f'Configured run lengths do not match tensor shape {tuple(self.data.shape)} or window size.')
        # Reject NaN/Inf values rather than allowing them to corrupt calibration and training.
        if not torch.isfinite(self.data).all():
            raise ValueError('Non-finite fMRI samples found.')
        self.voxels = self.data.shape[-1]       # Number of selected cortical voxels, not a 3D grid.
        self.repetitions = self.data.shape[1]   # Independent presentations of the same movie.
        self.average_repeats = average_repeats  # False during training/calibration; optional at inference.
        # Convert user-facing one-based IDs into a set for the train/validation run selection.
        validation = {int(n) for n in config.validation_runs.split(',') if n.strip()}
        if phase == 'train' and (not validation or min(validation) < 1 or max(validation) > len(self.run_lengths)):
            raise ValueError('Specify valid one-based validation run IDs from the training set.')
        # Test recordings must never enter the fitting or validation branches.
        if phase == 'test' and selection != 'all':
            raise ValueError('Test runs are only available for inference, not fitting or validation.')
        # Keep complete movie runs on one side of the train/validation split.
        self.run_ids = [i for i in range(len(self.run_lengths)) if selection == 'all'
                        or (selection == 'validation') == (i + 1 in validation)]
        if not self.run_ids:
            raise ValueError('Run selection is empty.')
        # Cumulative offsets map each run's local time index into the saved tensor's row index.
        self.offsets = [0]
        for length in self.run_lengths:
            self.offsets.append(self.offsets[-1] + length)
        # Cache lightweight (run, repetition, local start) triples, not copies of signal windows.
        self.windows: list[tuple[int, int, int]] = []
        for run in self.run_ids:
            # The last legal start keeps the complete window inside this run.
            final = self.run_lengths[run] - config.window_trs
            # Slice before any DataLoader shuffling; shuffling later reorders whole windows only.
            starts = list(range(0, final + 1, config.window_stride))
            if starts[-1] != final:
                # Include the final legal window even when the stride does not divide the run length.
                starts.append(final)
            # Averaging produces one series; otherwise index each presentation independently.
            for repetition in range(1 if average_repeats else self.repetitions):
                self.windows.extend((run, repetition, start) for start in starts)
        log(f'{phase}/{selection}: {tuple(self.data.shape)}, {len(self.run_ids)} runs, {len(self.windows)} windows.')

    def __len__(self) -> int:
        return len(self.windows)

    def run(self, run: int, repetition: int | None = None) -> torch.Tensor:
        # Return [time, voxel]; repeated presentations remain separate unless explicitly averaged
        values = self.data[self.offsets[run]:self.offsets[run + 1]]
        return values.mean(1) if repetition is None else values[:, repetition]

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Recover the selected window's movie run, presentation and local acquisition index.
        run, repetition, start = self.windows[index]
        # Convert local run time into the absolute row of the saved tensor.
        absolute = self.offsets[run] + start
        # Slice consecutive TRs only; this retains [window, repetition, voxel] order.
        values = self.data[absolute:absolute + self.config.window_trs]
        # Average repeated presentations only when requested; never average neighbouring TRs.
        values = values.mean(1) if self.average_repeats else values[:, repetition]
        # The network expects [voxel, time]; DataLoader adds the leading batch axis.
        # Retain run/start metadata for overlap averaging and explicit stimulus-time export.
        return {'bold': values.T.contiguous(), 'run': run, 'repetition': repetition, 'start': start}

    def provenance(self) -> dict[str, Any]:
        # A changed source or run selection invalidates its offline noise calibration.
        stat = self.path.stat()
        return {'path': str(self.path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'shape': list(self.data.shape), 'run_lengths': self.run_lengths,
                'training_runs': [i + 1 for i in self.run_ids]}

    def audit(self) -> dict[str, Any]:
        # Measure actual saved values without assuming that they were z-scored upstream
        rows = []
        for run in self.run_ids:
            values = self.data[self.offsets[run]:self.offsets[run + 1]]
            rows.append({'run': run + 1, 'mean': values.mean().item(),
                         'std': values.std().item(), 'negative_fraction': (values < 0).float().mean().item(),
                         'mean_absolute_voxel_mean': values.mean(0).abs().mean().item(),
                         'mean_voxel_std': values.std(0).mean().item()})
        # Average the run means and run standard deviations; this is not the pooled standard deviation
        overall_mean = sum(r['mean'] for r in rows) / len(rows)
        overall_std = sum(r['std'] for r in rows) / len(rows)
        # return dict
        return {'shape': list(self.data.shape),
                'runs': rows,  # list of dicts with run-level statistics
                'mean': overall_mean, 'std': overall_std,  # averages of run-level statistics
                'stored_delay_sec': self.config.stored_delay_sec,
                'timing': 'Stored index i measures acquisition time i * TR + stored_delay_sec relative to stimulus (shifted +x)'}
