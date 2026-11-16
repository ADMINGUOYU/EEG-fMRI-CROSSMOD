# file: runtime.py
# utility script (same as usual)
# Run artefacts, reproducibility and coloured console output

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import Config


def log(message: str, level: str = 'INFO') -> None:
    # Use the same severity colours as CineSync.
    colour = {'INFO': 34, 'WARN': 33, 'OK': 32, 'ERROR': 31}[level]
    print(f'\033[1;{colour}m[SRES_fMRI_cc2017] {level}\033[0m: {message}', flush = True)


def setup(config: Config) -> torch.device:
    # Keep this initial implementation single-process; the launcher selects the GPU.
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; configure DEVICE_TYPE in the launcher.')
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config.seed)
    return device


def save_json(path: Path, value: Any) -> None:
    # Reject non-finite statistics instead of writing non-standard JSON.
    path.write_text(json.dumps(value, indent = 2, allow_nan = False) + '\n', encoding = 'utf-8')


def create_run(config: Config) -> Path:
    # Preserve the literal launcher and resolved settings alongside each new run.
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    folder = Path(config.results_root) / 'SRES_fMRI_cc2017' / 'v1' / f'{config.action}_stage{config.stage}' / f'run_{name}'
    for part in ('meta', 'checkpoints', 'logs', 'artifacts'):
        (folder / part).mkdir(parents = True, exist_ok = False)
    save_json(folder / 'meta' / 'arguments.json', config.to_dict())
    if config.launch_script:
        shutil.copy2(config.launch_script, folder / Path(config.launch_script).name)
    log(f'Results: {folder}')
    return folder


def atomic_save(value: Any, path: Path) -> None:
    # Keep the previous complete checkpoint if serialisation is interrupted.
    path.parent.mkdir(parents = True, exist_ok = True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def make_loader(dataset: Dataset, config: Config, training: bool) -> DataLoader:
    # Windows are already contiguous; shuffle whole windows only during training.
    return DataLoader(dataset, batch_size = config.batch_size, shuffle = training,
                      num_workers = config.workers, pin_memory = config.device.startswith('cuda'),
                      persistent_workers = config.workers > 0)


def progress(label: str, completed: int, total: int, started: float) -> None:
    # Estimate remaining wall time from completed work.
    remaining = (monotonic() - started) * (total - completed) / max(completed, 1)
    log(f'{label}: {completed}/{total} ({100 * completed / max(total, 1):.1f}%); '
        f'remaining {int(remaining // 60)}m {int(remaining % 60)}s.')
