# file: noise.py
# [For training]
#  - robust noise calibration 
#  - stationary AR(1) reciprocal re-corruption

from __future__ import annotations

# [SETUP] ===============================================================
MAD_GAUSSIAN_DIFFERENCE = 0.9539  # Approximately Phi^-1(0.75) * sqrt(2).
CALIBRATION_VOXEL_CHUNK = 256     # Number of voxels to process at once (batch size)
RHO_LIMIT = 0.94                  # Keep stationary estimates away from singular endpoints.
# =======================================================================

from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .dataset_cc2017 import DatasetCC2017
from .runtime import atomic_save, log, progress

# import as type checking
# NOTE: version specific Config refactored to version specific directory
if TYPE_CHECKING:
    from ..models.v1.config import Config as ConfigV1
    from ..models.v2.config import Config as ConfigV2

    Config = ConfigV1 | ConfigV2


def calibrate(dataset: DatasetCC2017, config: Config) -> dict[str, Any]:
    """
    Estimate voxel noise from training runs only,
    without crossing run boundaries (a fun should be a full movie session)
    """
    # train only (no leak)
    if dataset.phase != 'train' or dataset.average_repeats:
        raise ValueError('Calibrate on individual training presentations, never averaged or test data')
    # validation runs cannot contribute to calibration
    # NOTE: val is a subset of normally train split
    validation = {int(n) - 1 for n in config.validation_runs.split(',') if n.strip()}
    if validation.intersection(dataset.run_ids):
        raise ValueError('Validation runs cannot contribute to calibration.')
    if config.rho_ar1 == 'auto' and dataset.repetitions < 2:
        raise ValueError('Auto rho requires repeated training presentations to isolate noise from stimulus dynamics.')
    # containers
    sigma = np.empty(dataset.voxels, dtype = np.float32)  # per selected cortical voxels
    rho = np.empty_like(sigma)                            # per selected cortical voxels (TOO)
    started = monotonic()
    for first in range(0, dataset.voxels, CALIBRATION_VOXEL_CHUNK):
        last = min(first + CALIBRATION_VOXEL_CHUNK, dataset.voxels)
        differences = []
        # numerator and denominator for AR(1) estimation, summed across runs
        numerator = np.zeros(last - first, dtype = np.float64)
        denominator = np.zeros_like(numerator)
        for run in dataset.run_ids:
            values = dataset.data[dataset.offsets[run]:dataset.offsets[run + 1], :, first:last].numpy().astype(np.float64)
            differences.append(np.diff(values, axis = 0).reshape(-1, last - first))
            if config.rho_ar1 == 'auto':
                # Paired-repeat differences remove the shared movie response under repeat stability
                for repetition in range(0, dataset.repetitions - 1, 2):
                    residual = (values[:, repetition] - values[:, repetition + 1]) / np.sqrt(2)
                    residual -= residual.mean(axis = 0)
                    numerator += (residual[1:] * residual[:-1]).sum(axis = 0)
                    denominator += (residual[:-1] ** 2).sum(axis = 0)
        chunk_rho = np.clip(numerator / np.maximum(denominator, 1e-12), -RHO_LIMIT, RHO_LIMIT) if config.rho_ar1 == 'auto' else np.full(last - first, float(config.rho_ar1))
        delta = np.concatenate(differences, axis = 0)
        mad = np.median(np.abs(delta - np.median(delta, axis = 0)), axis = 0)
        # rho is used consistently here and in the stationary noise generator below.
        sigma[first:last] = mad / (MAD_GAUSSIAN_DIFFERENCE * np.sqrt(1 - chunk_rho))
        rho[first:last] = chunk_rho
        if (first // CALIBRATION_VOXEL_CHUNK + 1) % 4 == 0 or last == dataset.voxels:
            progress('Noise calibration', last, dataset.voxels, started)
    log('MAD differences may include neural changes; R2R independence depends on the noise model being accurate.', 'WARN')
    result = {'sigma': torch.from_numpy(sigma), 'rho': torch.from_numpy(rho),
              'provenance': dataset.provenance(), 'rho_setting': config.rho_ar1,
              'method': 'median absolute deviation of within-run first differences',
              'sigma_mean': float(sigma.mean()), 'rho_mean': float(rho.mean())}
    # save calibration
    atomic_save(result, Path(config.calibration_path))
    log(f'Calibrated {dataset.voxels} voxels: mean sigma = {sigma.mean():.5g}, rho = {rho.mean():.4f}.', 'OK')
    return result


def load_calibration(dataset: DatasetCC2017, config: Config) -> dict[str, Any]:
    """
    Load / Generate calibration (generate if cache not exists)
    """
    path = Path(config.calibration_path)
    if not path.exists():
        return calibrate(dataset, config)
    result = torch.load(path, map_location = 'cpu', weights_only = False)
    if result['provenance'] != dataset.provenance() or result['rho_setting'] != config.rho_ar1:
        raise ValueError('Calibration settings/source changed; rerun ACTION=calibrate or choose a new calibration path.')
    return result


# Actual noise generation
def ar1_noise(shape: tuple[int, ...], sigma: torch.Tensor, rho: torch.Tensor,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Draw [batch, voxel, time] noise with stationary variance sigma squared
    """
    eps = torch.randn(shape, device = sigma.device, dtype = sigma.dtype, generator = generator)
    coloured = torch.empty_like(eps)
    coloured[..., 0] = eps[..., 0]  # Stationary initial state, rather than a zero transient.
    innovation = torch.sqrt(1 - rho.square())
    for time in range(1, shape[-1]):
        coloured[..., time] = rho * coloured[..., time - 1] + innovation * eps[..., time]
    return coloured * sigma[None, :, None]


# [IMPORTANT] Reciprocal corruption decorrelates the two noise terms if their covariances match
# Conditional noise independence additionally requires the Gaussian model
def reciprocal_pair(values: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor,
                    alpha: float, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Keep alpha and calibrated noise fixed while drawing fresh noise each call
    """
    noise = ar1_noise(tuple(values.shape), sigma, rho, generator)
    return values + alpha * noise, values - noise / alpha
