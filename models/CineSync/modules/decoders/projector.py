# file: projector.py
# libraries: torch
from torch import nn


class BrainProjector(nn.Linear):
    # the fMRI-only branch maps 2048-wide spatial tokens to CogVideoX's 4096-wide condition
    def __init__(self, input_dim: int = 2048, output_dim: int = 4096) -> None:
        super().__init__(input_dim, output_dim)
