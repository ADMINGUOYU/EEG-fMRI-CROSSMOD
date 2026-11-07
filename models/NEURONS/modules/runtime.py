"""
shared path and logging utilities for the NEURONS port
"""

from __future__ import annotations

import argparse
import builtins
import os
from typing import Any


def log(*values: Any, sep: str = ' ', end: str = '\n', **kwargs: Any) -> None:
    # Match the project's coloured, immediately flushed console output.
    if int(os.environ.get('RANK', '0')) != 0:
        return
    kwargs.pop('flush', None)
    builtins.print('\033[1;34m[NEURONS] INFO\033[0m:', *values,
                   sep = sep, end = end, flush = True, **kwargs)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    # Preserve stage-specific arguments while sharing output and cache locations.
    if '--cache_dir' not in parser._option_string_actions:
        parser.add_argument('--cache_dir', type = str, default = './data/cache/huggingface/hub')
    if '--exp_dir' not in parser._option_string_actions:
        parser.add_argument('--exp_dir', type = str, default = None,
                            help = 'Full subject output directory; overrides --exp.')


def experiment_dir(args: argparse.Namespace) -> str:
    # The explicit directory lets future variants choose their own results root.
    return args.exp_dir or f'./results/NEURONS/NEURONS/exp_{args.exp}/subj_{args.subj}'


class TensorBoardLogger:
    def __init__(self, directory: str, enabled: bool = True) -> None:
        # Only the main process creates event files.
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(directory) if enabled and int(os.environ.get('RANK', '0')) == 0 else None

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.writer is not None:
            for name, value in values.items():
                self.writer.add_scalar(name, value, global_step = step)

    def image(self, name: str, value: Any, step: int, caption: str) -> None:
        if self.writer is not None:
            self.writer.add_image(name, value, global_step = step, dataformats = 'HWC')
            self.writer.add_text(f'{name}/caption', caption, global_step = step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
