# file: config.py
# Typed configuration for v2 deconvolution training and inference

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class Config:

    """
    Launchers supply experiment settings;
    derived timing is evaluated on demand
    """

    # Model version recorded in checkpoints and used for result directories
    # NOTE: This is version 2 (without de-noise-r) variant
    version: str = 'v2'
    # Run action: train or infer
    action: str = 'train'
    # Directory containing the subject train/test fMRI tensors
    data_root: str = './data/cc2017'
    # Subject ID: 1, 2 or 3
    subject: int = 1
    # Comma-separated volume counts for each training movie run
    train_run_lengths: str = ','.join(['240'] * 18)
    # Comma-separated volume counts for each test movie run
    test_run_lengths: str = ','.join(['240'] * 5)
    # One-based training run IDs held out from fitting
    validation_runs: str = '17,18'
    # BOLD delay already included in the saved stimulus pairing, in seconds
    stored_delay_sec: float = 4.0
    # Scanner repetition time: seconds between consecutive fMRI volumes
    tr_sec: float = 2.0
    # Consecutive volumes per input window; duration is window_trs * tr_sec
    window_trs: int = 32
    # Step between neighbouring window starts, in volumes
    window_stride: int = 4
    # Leading volumes excluded from losses and export to allow a full HRF history
    burnin_trs: int = 15
    # Trailing volumes excluded from losses and export, not a causal receptive-field limit
    lookahead_trs: int = 4
    # Activity samples per TR; 1 retains the original temporal grid
    tsr_factor: int = 4
    # Causal HRF kernel duration in seconds
    hrf_length_sec: float = 30.0
    # Canonical for fixed timing or dilated_learnable for bounded time stretching
    hrf_mode: str = 'canonical'
    # Lower bound on learned HRF dilation, below 1
    hrf_dilation_min: float = 0.8
    # Upper bound on learned HRF dilation, above 1
    hrf_dilation_max: float = 1.2
    # Weight of the temporal total-variation penalty on activity
    lambda_tv: float = 0.01
    # Allow negative activity; false constrains activity to be non-negative
    signed_activations: bool = True
    # Learn a constant additive BOLD offset per voxel, initialised to zero
    learn_baseline: bool = True
    # Apply the supplied spatial blur matrix during reblurring
    enable_spatial_spsf: bool = False
    # Torch spatial operator with matrix, fwhm_mm and voxel_order_confirmed metadata
    spatial_matrix_path: str = ''
    # Expected spatial blur full width at half maximum in millimetres
    spsf_fwhm_mm: float = 2.5
    # Feature channels at the first temporal U-Net level
    width: int = 16
    # Number of temporal downsampling levels in each U-Net
    depth: int = 2
    # Temporal convolution width in samples; odd and at least 3
    kernel_size: int = 3
    # Voxels per temporal processing chunk to limit activation memory
    voxel_batch_size: int = 256
    # Fresh training: initialise the deconvolver on synthetic activity
    pretrain_synthetic: bool = True
    # Number of optimiser steps during synthetic pre-training
    synthetic_steps: int = 500
    # Shortest synthetic constant-activity block, in seconds
    synthetic_min_shot_sec: float = 2.0
    # Longest synthetic constant-activity block, in seconds
    synthetic_max_shot_sec: float = 8.0
    # Standard deviation of Gaussian block amplitudes before any absolute-value constraint
    synthetic_amplitude: float = 1.0
    # Inference only: trained checkpoint supplying model weights
    checkpoint: str = ''
    # Training: restore checkpoint weights, optimiser, scheduler and RNG states
    resume: str = ''
    # Inference only: source split, train or test
    phase: str = 'test'
    # Inference only: average repeated presentations before processing
    average_repeats: bool = True
    # Base directory for run outputs, logs and checkpoints
    results_root: str = './results'
    # Launcher path copied into run outputs to record the experiment settings
    launch_script: str = ''
    # PyTorch compute device, for example cpu or cuda
    device: str = 'cuda'
    # Random seed for initialisation and sampling
    seed: int = 42
    # DataLoader worker processes; 0 loads in the main process
    workers: int = 0
    # Number of temporal windows per DataLoader batch
    batch_size: int = 2
    # Total training epochs, including epochs completed before resuming
    epochs: int = 30
    # Peak AdamW learning rate
    lr: float = 0.0001
    # AdamW weight-decay coefficient
    weight_decay: float = 0.01
    # Fraction of optimiser steps allocated to learning-rate warm-up
    warmup_fraction: float = 0.05
    # Final cosine-schedule learning rate as a fraction of lr
    minimum_lr_ratio: float = 0.05
    # Gradient-norm clipping threshold; 0 disables clipping
    max_grad_norm: float = 1.0
    # Progress reporting interval in batches or synthetic steps
    log_every: int = 20
    # Epoch interval for numbered checkpoints; last and best are saved separately
    save_every: int = 5

    @property
    def dt_sec(self) -> float:
        # Seconds between inferred activity samples
        return self.tr_sec / self.tsr_factor

    @property
    def hrf_samples(self) -> int:
        # Number of HRF weights needed to cover the configured duration
        return math.ceil(self.hrf_length_sec / self.dt_sec)

    @property
    def delay_trs(self) -> int:
        # Stored stimulus-to-BOLD delay expressed in whole scanner volumes
        return round(self.stored_delay_sec / self.tr_sec)

    def validate(self) -> None:
        # Validate numerical settings and window coverage; chronology is assumed by the loader
        if self.version != 'v2':
            raise ValueError('This configuration belongs to v2.')
        if self.action not in ('train', 'infer'):
            raise ValueError('Choose train or infer.')
        if self.subject not in (1, 2, 3) or self.phase not in ('train', 'test'):
            raise ValueError('Invalid subject or inference phase.')
        positive = ('tr_sec', 'window_trs', 'window_stride', 'tsr_factor', 'hrf_length_sec',
                    'width', 'depth', 'voxel_batch_size',
                    'batch_size', 'epochs', 'lr', 'log_every', 'save_every', 'synthetic_amplitude')
        if any(not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0 for name in positive):
            raise ValueError(f'These settings must be positive: {positive}')
        if self.burnin_trs < 0 or self.lookahead_trs < 0 or self.burnin_trs + self.lookahead_trs >= self.window_trs:
            raise ValueError('Window must contain a non-empty valid interior.')
        if self.window_stride > self.window_trs - self.burnin_trs - self.lookahead_trs:
            raise ValueError('Window stride would leave gaps in the valid output.')
        if self.window_trs < 2 ** self.depth or self.kernel_size < 3 or self.kernel_size % 2 != 1:
            raise ValueError('Window must suit the U-Net depth; kernel size must be odd and at least three.')
        if self.hrf_samples < 2 or (self.window_trs - self.burnin_trs - self.lookahead_trs) * self.tsr_factor < 2:
            raise ValueError('HRF and valid high-resolution interior must each contain at least two samples.')
        if self.burnin_trs * self.tsr_factor < self.hrf_samples - 1:
            raise ValueError('Deconvolution burn-in must cover the full causal HRF history; increase burnin_trs/window_trs.')
        if self.hrf_mode not in ('canonical', 'dilated_learnable'):
            raise ValueError('Unknown HRF mode.')
        if not 0 < self.hrf_dilation_min < 1 < self.hrf_dilation_max:
            raise ValueError('HRF dilation bounds must bracket one.')
        if not 0 <= self.warmup_fraction < 1 or not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError('Invalid scheduling fraction.')
        nonnegative = ('lambda_tv', 'weight_decay', 'synthetic_steps', 'workers', 'max_grad_norm')
        if any(not math.isfinite(getattr(self, name)) or getattr(self, name) < 0 for name in nonnegative):
            raise ValueError('Regularisation, worker count and synthetic steps cannot be negative.')
        if not 0 < self.synthetic_min_shot_sec <= self.synthetic_max_shot_sec:
            raise ValueError('Invalid synthetic shot durations.')
        if not math.isfinite(self.stored_delay_sec) or self.stored_delay_sec < 0 or not math.isclose(self.delay_trs * self.tr_sec, self.stored_delay_sec):
            raise ValueError('Stored lag must be a non-negative whole number of TRs.')
        if self.enable_spatial_spsf and (not self.spatial_matrix_path or self.spsf_fwhm_mm <= 0):
            raise ValueError('Spatial sPSF requires a precomputed matrix with verified voxel order and FWHM.')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_config() -> Config:
    """
    Parse all v2 settings from the launcher
    """
    parser = argparse.ArgumentParser(description = 'HRF-constrained deconvolution for cc2017 (v2).')
    defaults = Config()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        name = '--' + field.name.replace('_', '-')
        if isinstance(value, bool):
            parser.add_argument(name, action = argparse.BooleanOptionalAction, default = value)
        else:
            parser.add_argument(name, type = type(value), default = value)
    config = Config(**vars(parser.parse_args()))
    config.validate()
    return config
