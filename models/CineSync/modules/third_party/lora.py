# file: lora.py
# libraries: torch
# Native PyTorch LoRA on attention Q/K/V/output; no PEFT or training framework required.
from __future__ import annotations
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 64, alpha: float = 64) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError('LoRA rank must be positive.')
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device = base.weight.device, dtype = torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device = base.weight.device, dtype = torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a = math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        residual = (x.to(self.lora_a.dtype) @ self.lora_a.T @ self.lora_b.T) * self.scale
        return base + residual.to(base.dtype)


def attach_lora(transformer: nn.Module, rank: int, alpha: float) -> list[str]:
    # match the upstream attention-only LoRA scope, with separate Q/K/V adapters
    transformer.requires_grad_(False)
    targets = [name for name, layer in transformer.named_modules()
               if isinstance(layer, nn.Linear) and name.endswith(('to_q', 'to_k', 'to_v', 'to_out.0'))]
    if not targets:
        raise ValueError('No CogVideoX attention projections found for LoRA.')
    for name in targets:
        parent_name, child = name.rsplit('.', 1)
        parent = transformer.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha))
    return targets
