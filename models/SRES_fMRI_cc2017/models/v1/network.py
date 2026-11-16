# file: network.py
# Cascaded temporal U-Nets
# self supervised by HRF temporal modelling
# (ZS-DeconvNet uses PSF spatial modelling instead)
# de-noise/deconvolve design

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ...modules.config import Config
from ...modules.physics import ObservationModel


class TemporalUNet(nn.Module):

    """
    Share temporal filters between voxels without assuming spatial voxel adjacency
    """

    def __init__(self,
                 width: int,        # number of channels in the first encoder layer
                 depth: int,        # number of downsampling levels; depth + 1 encoders and depth decoders
                 kernel_size: int,  # convolution kernel size (must be odd to preserve temporal length)
                 factor: int = 1,   # temporal upsampling factor (1 = no upsampling)
                 residual: bool = True
    ) -> None:
        super().__init__()
        self.factor = factor
        self.residual = residual
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        sizes = [width * 2 ** i for i in range(depth + 1)]
        for first, last in zip([1] + sizes[:-1], sizes):
            self.encoders.append(nn.Sequential(nn.Conv1d(first, last, kernel_size, padding = kernel_size // 2),
                                               nn.GELU(), nn.Conv1d(last, last, kernel_size, padding = kernel_size // 2), nn.GELU()))
        for index in range(depth - 1, -1, -1):
            self.decoders.append(nn.Sequential(nn.Conv1d(sizes[index + 1] + sizes[index], sizes[index], kernel_size, padding = kernel_size // 2),
                                               nn.GELU(), nn.Conv1d(sizes[index], sizes[index], kernel_size, padding = kernel_size // 2), nn.GELU()))
        self.output = nn.Conv1d(width, factor, 1)
        # Begin as an identity de-noise-r;
        # the de-convolver must learn the inverse mapping
        # s.t. the output layer will output 0 for prediction
        # so upon initialisation: Initial output = input + 0 = input
        # this will let the output linear layer to learn first
        # earlier layers initially receive zero loss gradients because the output weights are 0
        if residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, voxels, length = values.shape
        current = values.reshape(batch * voxels, 1, length)
        # skip is for concatenated encoder features, not additive residuals
        skips = []
        for index, encoder in enumerate(self.encoders):
            current = encoder(current)
            if index < len(self.encoders) - 1:
                skips.append(current)
                # F.avg_pool1d(current, 2) averages adjacent time samples,
                # halving the temporal length before the next encoder level
                current = F.avg_pool1d(current, 2)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            current = F.interpolate(current, size = skip.shape[-1], mode = 'linear', align_corners = False)
            current = decoder(torch.cat((current, skip), dim = 1))
        # Temporal subpixel rearrangement produces exactly factor samples per original TR.
        output = self.output(current).transpose(1, 2).reshape(batch, voxels, length * self.factor)
        return output + values if self.residual else output


class SuperResolutionHead(nn.Module):

    """
    Stage 1 predicts de-noised BOLD; stage 2 predicts latent activity at TR / tsr_factor.
    """

    def __init__(self, config: Config, voxels: int) -> None:
        super().__init__()
        self.config = config
        self.voxels = voxels
        # de-noise-r
        self.denoiser = TemporalUNet(config.width, config.depth, config.kernel_size)
        # de-convolver (with temporal upsampling)
        self.deconvolver = TemporalUNet(config.width, config.depth, config.kernel_size,
                                       factor = config.tsr_factor, residual = False)
        # baseline signal/amplitude (learned per voxel, not per time sample)
        self.baseline = nn.Parameter(torch.zeros(voxels), requires_grad = config.learn_baseline)
        self.physics = ObservationModel(config, voxels)

        # mixing: interaction between voxels, before temporal denoising
        # All voxel values at one time point
        #           [13,447]
        #               |
        #       Linear projection
        #               |
        #    [mixing rank shared rep.]
        #               |
        #              GELU
        #               |
        #       Linear projection
        #               |
        #           [13,447]
        #               |
        #      Add to original values (residual)
        self.mixing = None
        if config.mixing_rank:
            # Optional global channel mixing requires the full voxel set, not arbitrary chunks
            self.mixing = nn.Sequential(nn.Conv1d(voxels, config.mixing_rank, 1), nn.GELU(),
                                        nn.Conv1d(config.mixing_rank, voxels, 1))
            nn.init.zeros_(self.mixing[-1].weight)
            nn.init.zeros_(self.mixing[-1].bias)

    def forward(self, values: torch.Tensor, voxel_ids: torch.Tensor | None = None,
                stage: int = 2) -> dict[str, torch.Tensor]:
        # Chunk temporal inference to bound activation memory while preserving
        # shared weights

        # Pre-mixing
        if self.mixing is not None:
            if values.shape[1] != self.voxels:
                raise ValueError('Low-rank mixing requires all voxels together.')
            values = values + self.mixing(values)  # residual

        # Stage 1: de-noise BOLD
        bold = torch.cat([self.denoiser(chunk) for chunk in values.split(self.config.voxel_batch_size, dim = 1)], dim = 1)
        result = {'bold': bold}

        # Stage 2: de-convolve to latent activity
        if stage == 2:
            activation = torch.cat([self.deconvolver(chunk) for chunk in bold.split(self.config.voxel_batch_size, dim = 1)], dim = 1)
            if not self.config.signed_activations:
                activation = F.relu(activation)
            baseline = self.baseline if voxel_ids is None else self.baseline[voxel_ids]
            result.update(activation = activation, reblurred = self.physics(activation, baseline))
        
        # return as dict
        return result
