# file: pack_single_frame.py
# Pack v2 activity at the original TR for NEURONS, preserving individual repetitions
# Run with `python -m models.SRES_fMRI_cc2017.models.v2.NEURONS_adapter.pack_single_frame`
# Unsupported boundaries retain original BOLD; the configured stored delay maps activity to stimulus rows

from __future__ import annotations

# [SETUP] ===============================================================
CHECKPOINT = ''                                # Trained v2 checkpoint
DATA_ROOT = './data/cc2017'                    # Original tensors and supporting assets
OUTPUT_DIR = './data/cc2017_SRES_v2_single_frame'
COLLAPSE_MODE_T = 'average_pool'               # Average subframes or select a zero-based index, e.g. '0'
DEVICE_TYPE = 'cuda'                           # cuda or cpu
BATCH_SIZE = 2                                 # Temporal windows per inference batch
WORKERS = 0                                    # DataLoader processes
SEED = 42                                      # Random seed
LOG_EVERY = 20                                 # Progress interval in batches
# =======================================================================

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from time import monotonic

import torch

from ....modules.dataset_cc2017 import DatasetCC2017
from ....modules.runtime import atomic_save, log, make_loader, progress, setup
from ..config import Config
from ..network import SuperResolutionHead


def main(args: argparse.Namespace) -> None:
    """
    Preserve stimulus indices and fill unsupported boundaries with original BOLD
    """
    # Recover the subject, temporal windows, architecture and observation settings from training
    checkpoint = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint, map_location = 'cpu', weights_only = False)
    config = replace(Config(**saved['config']), action = 'infer',
                     checkpoint = str(checkpoint), data_root = args.data_root,
                     device = args.device, batch_size = args.batch_size,
                     workers = args.workers, seed = args.seed, log_every = args.log_every)
    # A fixed position is zero-based within the checkpoint's original subframe grid
    collapse = args.collapse_mode_t
    if collapse != 'average_pool':
        try:
            collapse = int(collapse)
        except ValueError as exc:
            raise ValueError('COLLAPSE_MODE_T must be average_pool or a zero-based subframe index') from exc
        if not 0 <= collapse < config.tsr_factor:
            raise ValueError(f'Subframe index must lie in [0, {config.tsr_factor - 1}]')
    device = setup(config)
    log(f'V2 activity: factor = {config.tsr_factor}, collapse = {collapse}, stored delay = {config.stored_delay_sec:g}s')
    log('Packed values are inferred activity, not BOLD; compatibility with a pretrained BOLD encoder is experimental', 'WARN')
    phases = ('train', 'test')
    output_dir = Path(args.output_dir).resolve()
    source_dir = Path(args.data_root).resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(f'Original cc2017 root does not exist: {source_dir}')
    if output_dir == source_dir:
        raise ValueError('Choose a separate output directory to preserve the original dataset')
    # Refuse existing outputs before doing inference, including links to original tensors
    reserved = {f'subj{config.subject:02d}_{phase}_fmri.pt' for phase in phases}
    for name in reserved:
        destination = output_dir / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Output already exists; choose another output directory: {destination}')
    # Link supporting assets without copying large videos or weights
    links: list[tuple[Path, Path]] = []
    for source in sorted(source_dir.iterdir()):
        if source.name in reserved or source.name.startswith('.'):
            continue
        target = source.resolve()
        # Avoid a recursive link if the output directory is nested inside the original root
        if target == output_dir or target in output_dir.parents:
            continue
        destination = output_dir / source.name
        if destination.is_symlink() and destination.resolve() == target:
            continue
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f'Asset destination already exists with different contents: {destination}')
        links.append((destination, target))
    output_dir.mkdir(parents = True, exist_ok = True)

    # Export activity without reblurring; no external spatial PSF file is needed
    model = SuperResolutionHead(replace(config, enable_spatial_spsf = False), saved['voxels']).to(device)
    state = dict(saved['model'])
    state.pop('physics.spatial', None)
    model.load_state_dict(state, strict = True)
    model.eval().requires_grad_(False)
    del saved, state

    for phase in phases:
        # Keep all trials in their original order; validation run selection does not remove rows
        current = replace(config, phase = phase, average_repeats = False)
        # Keep shared dataset status messages quiet in this packing-only script
        with redirect_stdout(StringIO()):
            data = DatasetCC2017(current, phase, average_repeats = False)
        if min(data.run_lengths) <= current.burnin_trs + current.delay_trs:
            raise ValueError('Stored delay leaves no valid activity rows within a run.')
        if data.voxels != model.voxels:
            raise ValueError('Voxel count differs from the checkpoint')
        if phase == 'train' and data.repetitions != 2:
            raise ValueError('NEURONS training expects two repeated presentations')
        # Preserve individual repetitions; NEURONS handles test averaging itself
        total = torch.zeros_like(data.data)
        counts = torch.zeros(len(data.data), data.repetitions, dtype = torch.int32)
        local = torch.arange(current.burnin_trs, current.window_trs - current.lookahead_trs)
        loader = make_loader(data, current, False)
        log(f'Packing {phase}: {len(data)} windows, {data.repetitions} repetitions')
        started = monotonic()
        with torch.inference_mode():
            for iteration, batch in enumerate(loader, start = 1):
                values = batch['bold'].to(device)
                # Collapse activity within each original TR without changing the trained temporal grid
                activity = model.activity(values).reshape(len(values), data.voxels, current.window_trs, current.tsr_factor)
                prediction = activity.mean(dim = -1) if collapse == 'average_pool' else activity[..., collapse]
                prediction = prediction.transpose(1, 2).cpu()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError('Non-finite activity prediction')
                # Map acquisition-time activity to stimulus rows, matching v2 inference
                # Subframe j represents j * dt_sec within that TR; averaging summarises the whole interval
                for sample in range(len(values)):
                    run = int(batch['run'][sample])
                    repetition = int(batch['repetition'][sample])
                    stimulus = int(batch['start'][sample]) + local + current.delay_trs
                    inside = stimulus < data.run_lengths[run]
                    rows = data.offsets[run] + stimulus[inside]
                    total[rows, repetition] += prediction[sample, local[inside]]
                    counts[rows, repetition] += 1
                # Report console progress and remaining time separately for each split
                if iteration % current.log_every == 0 or iteration == len(loader):
                    progress(f'Packing {phase}', iteration, len(loader), started)

        # Preserve boundary rows with original signals so NEURONS receives no NaNs or shifted pairings
        valid = counts > 0
        total /= counts.clamp_min(1).unsqueeze(-1)
        total[~valid] = data.data[~valid]
        if not torch.isfinite(total).all():
            raise FloatingPointError('Non-finite packed activity')
        path = output_dir / f'subj{current.subject:02d}_{phase}_fmri.pt'
        atomic_save(total, path)
        log(f'Saved {path}; mixed output retains original BOLD for {int((~valid).sum())}/{valid.numel()} boundary trial-repetitions', 'WARN')

    # Publish supporting assets only after all requested fMRI tensors have been generated
    for destination, target in links:
        destination.symlink_to(target, target_is_directory = target.is_dir())
    log(f'Packing complete: {output_dir}; created {len(links)} supporting symlinks', 'OK')


if __name__ == '__main__':
    # Configure the launcher; the checkpoint supplies the subject, architecture and HRF settings
    parser = argparse.ArgumentParser(description = 'Pack collapsed v2 activity for NEURONS')
    parser.add_argument('--checkpoint', default = CHECKPOINT)
    parser.add_argument('--data-root', default = DATA_ROOT)
    parser.add_argument('--output-dir', default = OUTPUT_DIR)
    parser.add_argument('--collapse-mode-t', default = COLLAPSE_MODE_T)
    parser.add_argument('--device', default = DEVICE_TYPE)
    parser.add_argument('--batch-size', type = int, default = BATCH_SIZE)
    parser.add_argument('--workers', type = int, default = WORKERS)
    parser.add_argument('--seed', type = int, default = SEED)
    parser.add_argument('--log-every', type = int, default = LOG_EVERY)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error('Specify --checkpoint with a v2 checkpoint')
    main(args)
