# file: fmri.py
# libraries: torch
# Architecture follows CineBrain/sgm/modules/encoders/fmri_encoder_custom.py.
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FMRIConfig:
    frames: int = 5
    voxels: int = 8405
    width: int = 2048
    spatial_tokens: int = 226
    layers: int = 12
    heads: int = 32
    embedding_dim: int = 1152
    output_dim: int = 2048
    dropout: float = 0.1


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fused scaled-dot-product attention preserves upstream projection parameters
        b, t, d = x.shape
        q, k, v = [layer(x).view(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for layer in (self.q_proj, self.k_proj, self.v_proj)]
        y = F.scaled_dot_product_attention(q, k, v, dropout_p = self.dropout if self.training else 0.0)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg: FMRIConfig) -> None:
        super().__init__()
        self.self_attn = Attention(cfg.width, cfg.heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.feed_forward = nn.Sequential(nn.Linear(cfg.width, cfg.width * 4), nn.GELU(),
                                          nn.Dropout(cfg.dropout), nn.Linear(cfg.width * 4, cfg.width))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class FMRIEncoder(nn.Module):
    def __init__(self, config: FMRIConfig) -> None:
        super().__init__()
        if config.width % config.heads or config.layers < 1:
            raise ValueError('Encoder width must be divisible by heads; layers must be positive.')
        self.config = config
        self.channel_proj = nn.Linear(config.frames, config.width)
        self.spatial_proj = nn.Linear(config.voxels, config.spatial_tokens)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.width))
        self.pos_embed = nn.Parameter(torch.randn(1, config.spatial_tokens + 1, config.width))
        self.layers = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.cls_proj = nn.Linear(config.width, config.embedding_dim)
        self.out_proj = nn.Linear(config.width, config.output_dim)

    def forward(self, fmri: torch.Tensor, return_tokens: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tuple(fmri.shape[1:]) != (self.config.frames, self.config.voxels):
            raise ValueError(f'Unexpected fMRI shape {tuple(fmri.shape)}.')
        # commute the two affine projections to avoid a (B, 8405, 2048) intermediate;
        # include the spatial projection's row sums to preserve the channel bias exactly
        x = F.linear(fmri, self.spatial_proj.weight, None).transpose(1, 2)
        x = F.linear(x, self.channel_proj.weight, None)
        x = x + self.spatial_proj.weight.sum(-1)[None, :, None] * self.channel_proj.bias
        x = x + self.spatial_proj.bias[None, :, None]
        # cat class token and add positional embedding
        x = torch.cat((self.cls_token.expand(fmri.shape[0], -1, -1), x), dim = 1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        # alignment only needs CLS, so avoid the unused spatial output projection there
        return self.cls_proj(x[:, 0]), self.out_proj(x[:, 1:]) if return_tokens else None
