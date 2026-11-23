# file: physics.py
# - Differentiable causal HRF observation model;
# - optional spatial vascular blur
#
# [module usage]
# Measured BOLD
#      |
# De-noise-r -> De-convolver
#                   |
#            Inferred activity (final activity output)
#                   |
#           -> (this file) physics.py
#            - HRF convolution
#            - Optional spatial blur (sPSF)
#            - Sampling at original TR
#            - Add voxel baseline (a global offset map, if learned)
#                   |
#           Reconstructed BOLD
#                   |
#        Compare with training target

from __future__ import annotations

# [SETUP] ===============================================================
# At unit dilation and a 1-second scale, each gamma component peaks at (shape - 1) seconds
# Dilation scales these times; the combined HRF extrema differ because the components overlap
HRF_PEAK_SHAPE = 6.0         # Positive gamma shape; component peak at 5 seconds
HRF_UNDERSHOOT_SHAPE = 16.0  # Subtracted gamma shape; component peak at 15 seconds
HRF_UNDERSHOOT_RATIO = 6.0   # Divides the undershoot by 6; area ratio is 1/6 before truncation, not peak-height ratio
# =======================================================================

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

# import as type checking
# NOTE: version specific Config refactored to version specific directory
if TYPE_CHECKING:
    from ..models.v1.config import Config as ConfigV1
    from ..models.v2.config import Config as ConfigV2

    Config = ConfigV1 | ConfigV2


class ObservationModel(nn.Module):

    """
    Map latent activity to observed BOLD using time-domain convolution only
    """

    def __init__(self, config: Config, voxels: int) -> None:
        super().__init__()
        self.config = config

        # Fraction locates dilation = 1 within the configured bounds, so the HRF starts un-stretched
        fraction = (1 - config.hrf_dilation_min) / (config.hrf_dilation_max - config.hrf_dilation_min)
        # Dilation stretches HRF timing: above 1 is slower response, below 1 is faster response
        # Store its unconstrained logit; kernel() uses sigmoid to recover the bounded dilation
        # The inverse sigmoid initialises dilation to 1, with learning enabled only when requested
        self.dilation_logit = nn.Parameter(torch.tensor(math.log(fraction / (1 - fraction))),
                                          requires_grad = config.hrf_mode == 'dilated_learnable')

        # sample_times is count from [0, hrf_samples) * dt_sec
        # hrf_samples is sample count over the truncated HRF duration (30s by default)
        self.register_buffer('sample_times', torch.arange(config.hrf_samples, dtype = torch.float32) * config.dt_sec)
        
        # just register, used if sPSF is enabled
        self.register_buffer('spatial', None)

        # load the spatial PSF matrix if requested
        if config.enable_spatial_spsf:
            # The supplied sparse [V,V] operator must already use the saved tensor's voxel order.
            payload = torch.load(config.spatial_matrix_path, map_location = 'cpu', weights_only = False)
            matrix = payload['matrix'].to_sparse_coo().coalesce().float()
            if tuple(matrix.shape) != (voxels, voxels) or not math.isclose(payload['fwhm_mm'], config.spsf_fwhm_mm):
                raise ValueError('Spatial matrix shape/FWHM does not match configuration.')
            if not payload.get('voxel_order_confirmed', False):
                raise ValueError('Spatial matrix must declare voxel_order_confirmed = True.')
            sums = torch.sparse.sum(matrix, dim = 1).to_dense()
            if not torch.isfinite(matrix.values()).all() or (matrix.values() < 0).any() or not torch.allclose(sums, torch.ones_like(sums), atol = 1e-4):
                raise ValueError('Spatial PSF must have finite non-negative weights and rows summing to one.')
            self.spatial = matrix

    # construct the HRF kernel (called by forward())
    # HRF = positive response - smaller, later undershoot
    # Canonical means the standard reference HRF
    def kernel(self, canonical: bool = False) -> torch.Tensor:
        # A bounded dilation changes HRF timing (inherits sample_times's dtype, device, ...etc.)
        dilation = self.sample_times.new_tensor(1.0)
        # Map the learned logit to the configured dilation bounds;
        # canonical = True keeps dilation at 1
        if self.config.hrf_mode == 'dilated_learnable' and not canonical:
            dilation = self.config.hrf_dilation_min + torch.sigmoid(self.dilation_logit) * (self.config.hrf_dilation_max - self.config.hrf_dilation_min)
        # Evaluate the original curve at rescaled times: dilation above 1 stretches the response in time
        time = self.sample_times / dilation
        # Evaluate the unit-scale gamma density in log space, then exponentiate
        # lgamma supplies the log normalising constant; clamping avoids log(0)
        # Dividing by dilation preserves the continuous area when stretching time
        def gamma_density(shape: float) -> torch.Tensor:
            return torch.exp((shape - 1) * torch.log(time.clamp_min(1e-8)) - time - math.lgamma(shape)) / dilation
        # Build the double-gamma HRF: positive response minus the smaller, later undershoot
        # Set the response at time zero to exactly zero before normalising the discrete weights
        kernel = gamma_density(HRF_PEAK_SHAPE) - gamma_density(HRF_UNDERSHOOT_SHAPE) / HRF_UNDERSHOOT_RATIO
        kernel = torch.where(time > 0, kernel, torch.zeros_like(kernel))
        r"""
        1D over each voxel's time sequence
        HRF weight
            ^
            |           /\
            |          /  \
            |         /    \
            |        /      \
            |       /        \
          0 +------/----------\--------------------------------> time
            |                  \                 ________
            |                   \_______________/
            |
            0      ~5 s       ~12 s     ~16 s               30 s
                   positive             negative
                   peak                 undershoot
        """
        # normalisation fixes the DC gain (scaling)
        return kernel / kernel.sum().clamp_min(1e-8)

    # convert activity into BOLD
    def forward(self, activation: torch.Tensor, baseline: torch.Tensor | None = None,
                canonical: bool = False) -> torch.Tensor:
        # Activity:  ____|____________________
        # BOLD:      _____/‾‾\____\___/_______
        #                later response ->
        # PyTorch performs cross-correlation,
        # use kernel.flip to make sure convolution is okay:
        # Input window:       [x[t-2], x[t-1], x[t]]
        # Required weights:   [  h[2],   h[1], h[0]]
        # because h[k] describes the response k time steps AFTER an activity event,
        # past activity contributes in future (compared to past) at current time t
        batch, voxels, length = activation.shape
        # get 1D convolution kernel
        kernel = self.kernel(canonical)
        # pad with zero past activity to preserve output length and make the convolution causal
        padded = F.pad(activation.reshape(batch * voxels, 1, length), (len(kernel) - 1, 0))
        # do convolution
        bold = F.conv1d(padded, kernel.flip(0).reshape(1, 1, -1)).reshape(batch, voxels, length)
        # do sPSF blur if needed
        if self.spatial is not None:
            if voxels != self.spatial.shape[0]:
                raise ValueError('Spatial PSF requires all voxels together.')
            flattened = bold.permute(1, 0, 2).reshape(voxels, -1)
            bold = torch.sparse.mm(self.spatial, flattened).reshape(voxels, batch, length).permute(1, 0, 2)
        # downsample to the original TR (basic slicing with a step size)
        bold = bold[..., ::self.config.tsr_factor]
        return bold if baseline is None else bold + baseline[None, :, None]


# make activity for pre-training (my bling bling idea)
# i.e. Activity:  ____████████____███________████____
def synthetic_shots(batch: int, voxels: int, config: Config, device: torch.device) -> torch.Tensor:
    """
    Generate signed or non-negative block activity with a full leading HRF history.
    """
    # history supplies past activity for synthetic BOLD generation; it is cropped before the U-Net input
    history = math.ceil((config.hrf_samples - 1) / config.tsr_factor) * config.tsr_factor
    # length, the total number of time points including history
    length = config.window_trs * config.tsr_factor + history

    activity = torch.empty(batch, voxels, length, device = device)
    minimum = max(1, round(config.synthetic_min_shot_sec / config.dt_sec))
    maximum = max(minimum, round(config.synthetic_max_shot_sec / config.dt_sec))
    # Within each sample, voxels share block boundaries and have independent amplitudes
    for sample in range(batch):
        position = 0
        while position < length:
            duration = int(torch.randint(minimum, maximum + 1, ()).item())
            amplitude = torch.randn(voxels, 1, device = device) * config.synthetic_amplitude
            if not config.signed_activations:
                amplitude = amplitude.abs()
            activity[sample, :, position:min(length, position + duration)] = amplitude
            position += duration
    return activity
