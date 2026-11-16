# file: scheduling.py
# Cosine learning-rate decay with linear warm-up;
# measured in optimiser updates.

from __future__ import annotations

import math
import torch


def make_scheduler(optimiser: torch.optim.Optimizer, total_steps: int,
                   warmup_fraction: float, minimum_ratio: float) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Set the rate for each upcoming update; weight decay belongs to AdamW.
    """
    warmup = round(total_steps * warmup_fraction)

    def multiplier(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        fraction = min(1.0, max(0.0, (step - warmup) / max(total_steps - warmup - 1, 1)))
        return minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * fraction)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimiser, multiplier)
