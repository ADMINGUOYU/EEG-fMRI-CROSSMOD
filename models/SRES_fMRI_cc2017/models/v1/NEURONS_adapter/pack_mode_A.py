# file: pack_mode_A.py
# NEURONS data adapter (In mode A: use de-noised BOLD not activation maps)
# Infer de-noised BOLD and save tensors in the format expected by NEURONS

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from time import monotonic

import torch

from ..config import Config
from ....modules.dataset_cc2017 import DatasetCC2017
from ....modules.noise import ar1_noise
from ....modules.runtime import atomic_save, log, make_loader, progress, setup
from ..network import SuperResolutionHead


def main(args: argparse.Namespace) -> None:
    """
    Preserve stimulus indices and fill unsupported boundaries with original BOLD
    """
    # Recover the subject, temporal windows, architecture and noise settings from training
    checkpoint = Path(args.checkpoint).resolve()
    saved = torch.load(checkpoint, map_location = 'cpu', weights_only = False)
    config = replace(Config(**saved['config']), action = 'infer', mode = 'A',
                     checkpoint = str(checkpoint), data_root = args.data_root,
                     device = args.device, batch_size = args.batch_size,
                     workers = args.workers, mc_test_samples = args.mc_test_samples,
                     seed = args.seed, log_every = args.log_every)
    device = setup(config)
    phases = ('train', 'test')
    output_dir = Path(args.output_dir).resolve()
    source_dir = Path(args.data_root).resolve()
    real_root = source_dir
    if not real_root.is_dir():
        raise NotADirectoryError(f'Original cc2017 root does not exist: {real_root}')
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
    for source in sorted(real_root.iterdir()):
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

    # Mode A only uses the denoiser and mixing; no external spatial PSF file is needed
    model = SuperResolutionHead(replace(config, enable_spatial_spsf = False), saved['voxels']).to(device)
    state = dict(saved['model'])
    state.pop('physics.spatial', None)
    model.load_state_dict(state, strict = True)
    model.eval().requires_grad_(False)
    sigma = saved['calibration']['sigma'].to(device)
    rho = saved['calibration']['rho'].to(device)
    del saved, state

    for phase in phases:
        # Keep all trials in their original order; validation run selection does not remove rows
        current = replace(config, phase = phase, average_repeats = False)
        # Keep shared dataset status messages quiet in this packing-only script
        with redirect_stdout(StringIO()):
            data = DatasetCC2017(current, phase, average_repeats = False)
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
                prediction = torch.zeros_like(values)
                # Match standard mode-A inference: fresh recorruption followed by Monte-Carlo averaging
                for _ in range(current.mc_test_samples):
                    noisy = values + current.r2r_alpha * ar1_noise(tuple(values.shape), sigma, rho)
                    prediction += model(noisy, stage = 1)['bold'] / current.mc_test_samples
                prediction = prediction.transpose(1, 2).cpu()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError('Non-finite denoising prediction')
                # Average overlapping valid interiors without crossing run boundaries or shifting time
                for sample in range(len(values)):
                    run = int(batch['run'][sample])
                    repetition = int(batch['repetition'][sample])
                    rows = data.offsets[run] + int(batch['start'][sample]) + local
                    total[rows, repetition] += prediction[sample, local]
                    counts[rows, repetition] += 1
                # Report console progress and remaining time separately for each split
                if iteration % current.log_every == 0 or iteration == len(loader):
                    progress(f'Packing {phase}', iteration, len(loader), started)

        # Preserve boundary rows with original signals so NEURONS receives no NaNs or shifted pairings
        valid = counts > 0
        total /= counts.clamp_min(1).unsqueeze(-1)
        total[~valid] = data.data[~valid]
        if not torch.isfinite(total).all():
            raise FloatingPointError('Non-finite packed BOLD')
        path = output_dir / f'subj{current.subject:02d}_{phase}_fmri.pt'
        atomic_save(total, path)
        log(f'Saved {path}; retained original BOLD for {int((~valid).sum())}/{valid.numel()} boundary trial-repetitions', 'OK')

    # Publish supporting assets only after all requested fMRI tensors have been generated
    for destination, target in links:
        destination.symlink_to(target, target_is_directory = target.is_dir())
    log(f'Packing complete: {output_dir}; created {len(links)} supporting symlinks', 'OK')


if __name__ == '__main__':
    # Configure runs in pack_mode_A.sh; model and physics settings come from the checkpoint
    parser = argparse.ArgumentParser(description = 'Pack mode-A denoised fMRI for NEURONS')
    parser.add_argument('--checkpoint', required = True)
    parser.add_argument('--data-root', required = True)
    parser.add_argument('--output-dir', required = True)
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--batch-size', type = int, default = 2)
    parser.add_argument('--workers', type = int, default = 0)
    parser.add_argument('--mc-test-samples', type = int, default = 8)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--log-every', type = int, default = 20)
    main(parser.parse_args())
