# file: runtime.py
# libraries: torch, numpy, tensorboard
# Shared plain-PyTorch run management, checkpointing and reproducibility helpers.
from __future__ import annotations
import argparse
import json
import os
import random
import shutil
from time import monotonic
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import numpy as np
import torch
import torch.distributed as dist


def log(message: str, level: str = 'INFO') -> None:
    colour = {'INFO': 34, 'WARN': 33, 'OK': 32, 'ERROR': 31}[level]
    print(f'\033[1;{colour}m[CineSync] {level}\033[0m: {message}', flush = True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(args: argparse.Namespace) -> tuple[torch.device, int, int]:
    # torchrun supplies ranks; ordinary python remains a single-process entry point
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if args.device.startswith('cuda'):
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    seed_everything(args.seed + rank)
    return device, rank, world


def create_run(args: argparse.Namespace, stage: str, rank: int = 0, world: int = 1, *, model_name: str) -> Path:
    # validate model name
    if not model_name.strip() or model_name in ('.', '..') or any(char in model_name for char in ('/', '\\', '\0')):
        raise ValueError('Model name must be a non-empty folder name without path separators.')
    # the launcher path is explicitly supplied; preserve the exact script alongside settings
    folder = None
    if rank == 0:
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        folder = Path(args.results_root) / 'CineSync' / model_name / stage / f'run_{name}'
        for subdir in ('logs', 'checkpoints', 'meta'):
            (folder / subdir).mkdir(parents = True, exist_ok = False)
        if args.launch_script:
            shutil.copy2(args.launch_script, folder / Path(args.launch_script).name)
        (folder / 'meta' / 'arguments.json').write_text(json.dumps(vars(args), indent = 2), encoding = 'utf-8')
        log(f"Results: '{folder}'.")
    if world > 1:
        values = [str(folder) if folder else None]
        dist.broadcast_object_list(values, src = 0)
        folder = Path(values[0])
    return folder


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    # only replace the last complete checkpoint after torch.save succeeds
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def model_state(model: torch.nn.Module, stage: str) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    # reconstruction checkpoints contain brain weights and adapters, not the frozen 5B base
    return {key: value.cpu() for key, value in state.items()
            if stage == 'align' or not key.startswith('backend.') or 'lora_' in key}


def restore_model(model: torch.nn.Module, checkpoint: dict[str, Any]) -> None:
    expected = set(model_state(model, checkpoint['stage']))
    if expected != set(checkpoint['model']):
        raise ValueError('Checkpoint parameter names do not match the configured model.')
    model.load_state_dict(checkpoint['model'], strict = checkpoint['stage'] == 'align')


def load_checkpoint(path: str) -> dict[str, Any]:
    # run checkpoints include optimiser and RNG states; load only trusted local checkpoints
    return torch.load(path, map_location = 'cpu', weights_only = False)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--db-path', default = './data/CineBrain_database/lmdb')
    parser.add_argument('--captions-path', default = './data/CineBrain/captions-qwen-2.5-vl-7b.json')
    parser.add_argument('--subject-id', type = int, default = 1)
    parser.add_argument('--phase', default = 'train', choices = ['train', 'val', 'test'])
    parser.add_argument('--fmri-roi', default = 'visual', choices = ['visual', 'auditory', 'both'])
    parser.add_argument('--cache-dir', default = './data/cache/huggingface')
    parser.add_argument('--hf-endpoint', default = None)
    parser.add_argument('--local-files-only', action = 'store_true')
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--workers', type = int, default = 4)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--batch-size', type = int, default = 16)
    parser.add_argument('--results-root', default = './results')
    parser.add_argument('--launch-script', default = None)
    parser.add_argument('--max-samples', type = int, default = None)


def configure_huggingface(args: argparse.Namespace) -> None:
    # call before importing Transformers/Diffusers so endpoint selection takes effect
    os.environ['HF_HOME'] = str(Path(args.cache_dir).resolve())
    if args.hf_endpoint:
        os.environ['HF_ENDPOINT'] = args.hf_endpoint


def make_loader(args: argparse.Namespace, training: bool, rank: int = 0, world: int = 1,
                *, modalities: tuple[str, ...]):
    from torch.utils.data import DataLoader, DistributedSampler
    from .dataset import Dataset_CineSync
    dataset = Dataset_CineSync(args.db_path, args.captions_path, subject_id = args.subject_id,
                              phase = args.phase, fmri_roi = args.fmri_roi, modalities = modalities,
                              augment_captions = training, return_meta = False)
    if args.max_samples is not None:
        dataset.trial_indices = dataset.trial_indices[:args.max_samples]
    sampler = DistributedSampler(dataset, num_replicas = world, rank = rank, shuffle = training, seed = args.seed) if world > 1 else None
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = training and sampler is None,
                        sampler = sampler, num_workers = args.workers, collate_fn = dataset.collate_fn,
                        pin_memory = args.device.startswith('cuda'), drop_last = training,
                        multiprocessing_context = 'spawn' if args.workers else None,
                        persistent_workers = args.workers > 0, generator = generator)
    if len(loader) == 0:
        dataset.close()
        raise ValueError('No batches remain; check subject/phase selection and batch size.')
    return dataset, loader, sampler

# added progress printing assistant
def progress_text(completed: int, total: int, started: float, initial: int = 0) -> str:
    # estimate remaining loop time using only work completed in this invocation
    processed = completed - initial
    percentage = 100 * completed / total if total > 0 else 100.0
    remaining = max(0, total - completed)
    if remaining == 0:
        eta = '0m 00s'
    elif processed > 0:
        seconds = round((monotonic() - started) / processed * remaining)
        minutes, seconds = divmod(seconds, 60)
        eta = f'{minutes}m {seconds:02d}s'
    else:
        eta = 'estimating'
    return f'{completed}/{total} ({percentage:.1f}%), estimated remaining {eta}'
