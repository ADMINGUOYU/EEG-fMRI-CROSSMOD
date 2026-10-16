# file: recon.py
# libraries: torch; the injected video backend uses diffusers
from __future__ import annotations
import torch
from torch import nn
from ...modules.encoders.fmri import FMRIEncoder
from ...modules.decoders.projector import BrainProjector


class CineSyncFMRIRecon(nn.Module):
    def __init__(self, encoder: FMRIEncoder, backend: nn.Module, condition_dropout: float = 0.1) -> None:
        super().__init__()
        # NOTE: load the stage 1 aligned encoder
        self.encoder = encoder
        self.encoder.requires_grad_(True)
        # the alignment class head is unused during diffusion fine-tuning
        self.encoder.cls_proj.requires_grad_(False)
        self.projector = BrainProjector(encoder.config.output_dim, backend.transformer.config.text_embed_dim)
        self.backend = backend
        self.condition_dropout = condition_dropout

    def condition(self, fmri: torch.Tensor) -> torch.Tensor:
        _, tokens = self.encoder(fmri)
        return self.projector(tokens)

    def forward(self, fmri: torch.Tensor, video: torch.Tensor) -> dict[str, torch.Tensor]:
        # NOTE: just to encode and project (single fMRI model would not need neural feature fusion)
        condition = self.condition(fmri)
        if self.training:
            # drop the entire condition per sample as in the upstream conditioner
            keep = torch.rand(len(condition), 1, 1, device = condition.device) >= self.condition_dropout
            condition = condition * keep
        return {'loss': self.backend.diffusion_loss(video, condition)}

    @torch.no_grad()
    def generate(self, fmri: torch.Tensor, **kwargs) -> torch.Tensor:
        self.eval()
        return self.backend.generate(self.condition(fmri), **kwargs)
