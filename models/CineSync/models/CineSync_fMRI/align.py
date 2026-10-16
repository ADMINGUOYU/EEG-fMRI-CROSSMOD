# file: align.py
# libraries: torch
from __future__ import annotations
import math
import torch
from torch import nn
from ...modules.encoders.fmri import FMRIConfig, FMRIEncoder
from ...modules.losses.contrastive import contrastive_loss


class CineSyncFMRIAlign(nn.Module):
    def __init__(self, config: FMRIConfig, normalise: bool = False) -> None:
        super().__init__()
        self.config = config
        self.normalise = normalise
        self.encoder = FMRIEncoder(config)
        self.video_pool = nn.Conv1d(33, 1, 1)
        self.video_log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.text_log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        # the token output projection has no alignment objective; stage two trains it
        self.encoder.out_proj.requires_grad_(False)

    def forward(self, fmri: torch.Tensor, video_features: torch.Tensor,
                text_features: torch.Tensor) -> dict[str, torch.Tensor]:
        brain, _ = self.encoder(fmri, return_tokens = False)
        video = self.video_pool(video_features).squeeze(1)
        visual = contrastive_loss(brain, video, self.video_log_scale, self.normalise)
        textual = contrastive_loss(brain, text_features, self.text_log_scale, self.normalise)
        return {'loss': visual + textual, 'visual_loss': visual, 'text_loss': textual,
                'brain': brain, 'video': video, 'text': text_features}
