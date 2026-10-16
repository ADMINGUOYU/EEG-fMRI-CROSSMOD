# file: scheduling.py
# libraries: torch
from __future__ import annotations
import math
import torch


class CineBrainLR(torch.optim.lr_scheduler.LRScheduler):
    
    """
    SAT-style warm-up and decay, measured in optimiser updates.
    """

    def __init__(self, optimiser: torch.optim.Optimizer, decay_steps: int,
                 style: str = 'linear', warmup: float = 0.01, decay_ratio: float = 0.1,
                 auto_warmup_steps: int = 100, auto_warmup_rate: float = 0.05) -> None:
        if decay_steps < 1 or not 0 <= warmup <= 1 or not 0 <= decay_ratio <= 1:
            raise ValueError('Decay steps must be positive; warm-up and decay ratio must be in [0, 1].')
        if style not in ('constant', 'linear', 'cosine'):
            raise ValueError(f"Unknown learning-rate schedule '{style}'.")
        if auto_warmup_steps < 0 or not 0 <= auto_warmup_rate <= 1:
            raise ValueError('Automatic warm-up steps must be non-negative and its rate must be in [0, 1].')
        self.decay_steps = decay_steps
        self.style = style
        self.warmup_steps = warmup * decay_steps
        self.decay_ratio = decay_ratio
        self.auto_warmup_steps = auto_warmup_steps
        self.auto_warmup_rate = auto_warmup_rate
        super().__init__(optimiser)

    def get_lr(self) -> list[float]:
        # SAT starts at iteration one; the stored LR is for the next optimiser update.
        iteration = self.last_epoch + 1
        warmup_factor = iteration / self.warmup_steps if self.warmup_steps > 0 else 1.0
        if iteration <= self.auto_warmup_steps:
            factor = min(self.auto_warmup_rate, warmup_factor)
        elif self.warmup_steps > 0 and iteration <= self.warmup_steps:
            factor = warmup_factor
        else:
            # SAT uses the full decay horizon after warm-up, rather than subtracting it.
            progress = min(1.0, max(0.0, (iteration - self.warmup_steps) / self.decay_steps))
            if self.style == 'linear':
                factor = 1.0 - progress
            elif self.style == 'cosine':
                factor = self.decay_ratio + (1.0 - self.decay_ratio) * (1.0 + math.cos(math.pi * progress)) / 2
            else:
                factor = 1.0
        return [base_lr * factor for base_lr in self.base_lrs]
