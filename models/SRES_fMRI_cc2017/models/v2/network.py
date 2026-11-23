# file: network.py
# Temporal activity estimation with HRF and optional spatial observation modelling

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from ...modules.physics import ObservationModel


class TemporalUNet(nn.Module):

    """
    Share temporal filters between voxels without assuming spatial voxel adjacency
    """

    def __init__(self,
                 width: int,        # number of channels in the first encoder layer
                 depth: int,        # number of downsampling levels; depth + 1 encoders and depth decoders
                 kernel_size: int,  # convolution kernel size (must be odd to preserve temporal length)
                 factor: int = 1    # activity samples per original TR
    ) -> None:
        super().__init__()
        self.factor = factor
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
        return output


class SuperResolutionHead(nn.Module):

    """
    Infer latent activity directly from BOLD and reconstruct its observation
    """

    def __init__(self, config: Config, voxels: int) -> None:
        super().__init__()
        self.config = config
        self.voxels = voxels
        self.deconvolver = TemporalUNet(config.width, config.depth, config.kernel_size, config.tsr_factor)
        # The observation baseline is constant in time and learned separately for each voxel
        self.baseline = nn.Parameter(torch.zeros(voxels), requires_grad = config.learn_baseline)
        self.physics = ObservationModel(config, voxels)

    def activity(self, values: torch.Tensor) -> torch.Tensor:
        # Share temporal weights across voxel chunks; export needs only this inverse mapping
        activation = torch.cat([self.deconvolver(chunk) for chunk in values.split(self.config.voxel_batch_size, dim = 1)], dim = 1)
        return activation if self.config.signed_activations else F.relu(activation)

    def forward(self, values: torch.Tensor, voxel_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        # Compare reconstructed BOLD with the unchanged input during training
        activation = self.activity(values)
        baseline = self.baseline if voxel_ids is None else self.baseline[voxel_ids]
        return {'activation': activation, 'reblurred': self.physics(activation, baseline)}
