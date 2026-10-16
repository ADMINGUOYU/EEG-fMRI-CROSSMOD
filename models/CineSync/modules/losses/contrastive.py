# file: contrastive.py
# libraries: torch
from __future__ import annotations
import torch
from torch.nn import functional as F


def contrastive_loss(a: torch.Tensor, b: torch.Tensor, log_scale: torch.Tensor,
                     normalise: bool = False) -> torch.Tensor:
    # upstream uses raw dot products; cosine normalisation is an explicit experiment option
    a, b = a.float(), b.float()
    if normalise:
        a, b = F.normalize(a, dim = -1), F.normalize(b, dim = -1)
    logits = log_scale.float().clamp(0, 4.605170186).exp() * a @ b.T
    labels = torch.arange(len(a), device = a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
