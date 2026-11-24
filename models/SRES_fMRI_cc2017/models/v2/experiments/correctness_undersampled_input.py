# file: correctness_undersampled_input.py
# Train a fresh v2 inverse on retained BOLD frames and evaluate the omitted frames
# Run with `python -m models.SRES_fMRI_cc2017.models.v2.experiments.correctness_undersampled_input`

from __future__ import annotations

# [SETUP] ===============================================================
# Times and window lengths below refer to the original recording unless stated otherwise
DEFAULTS = {
    'data_root': './data/cc2017',                 # Original fMRI tensors, never a packed model output
    'subject': 1,                                 # Subject ID: 1, 2 or 3
    'train_run_lengths': ','.join(['240'] * 18),  # Original TR counts per training run
    'test_run_lengths': ','.join(['240'] * 5),    # Original TR counts per testing run
    'validation_runs': '17,18',                   # Held-out training runs; test data never selects weights
    'tr_sec': 2.0,                                # Original sampling interval: 2 seconds = 0.5 Hz
    'input_downsample_factor': 2,                 # Retain run-local rows 0, D, 2D, ... without filtering
    'tsr_factor': 4,                              # Activity samples per retained input; divisible by D
    'window_trs': 32,                             # Original TRs per window; divisible by D
    'window_stride': 4,                           # Original TRs between windows; divisible by D
    'burnin_trs': 15,                             # Original leading TRs excluded; rounded upwards to a retained TR
    'lookahead_trs': 4,                           # Original trailing TRs excluded; rounded upwards to a retained TR
    'average_repeats': False,                     # Apply the same repetition policy to fitting and evaluation
    'hrf_length_sec': 30.0,                       # Causal HRF support in physical seconds
    'hrf_mode': 'canonical',                      # canonical or dilated_learnable
    'hrf_dilation_min': 0.8,                      # Lower bound on optional learned dilation
    'hrf_dilation_max': 1.2,                      # Upper bound on optional learned dilation
    'lambda_tv': 0.01,                            # Activity total-variation weight
    'signed_activations': True,                   # Allow negative activity
    'learn_baseline': True,                       # Learn a time-constant BOLD baseline per voxel
    'enable_spatial_spsf': False,                 # Apply the optional spatial observation operator
    'spatial_matrix_path': '',                    # Verified voxel-order sPSF matrix
    'spsf_fwhm_mm': 2.5,                          # Expected sPSF width
    'width': 16,                                  # Initial U-Net channel count
    'depth': 2,                                   # Temporal pooling levels
    'kernel_size': 3,                             # Odd temporal convolution width
    'voxel_batch_size': 256,                      # Sampled training voxels and inference/metric chunk size
    'pretrain_synthetic': True,                   # Initialise from noise-free synthetic BOLD, not omitted data
    'synthetic_steps': 500,                       # Synthetic inverse updates before real-data fitting
    'synthetic_min_shot_sec': 2.0,                # Shortest synthetic activity block
    'synthetic_max_shot_sec': 8.0,                # Longest synthetic activity block
    'synthetic_amplitude': 1.0,                   # Synthetic activity amplitude standard deviation
    'epochs': 30,                                 # Real-data training epochs
    'lr': 0.0001,                                 # AdamW peak learning rate
    'weight_decay': 0.01,                         # AdamW regularisation
    'warmup_fraction': 0.05,                      # Fraction of updates used for warm-up
    'minimum_lr_ratio': 0.05,                     # Final cosine learning-rate fraction
    'max_grad_norm': 1.0,                         # Gradient clipping; zero disables it
    'batch_size': 2,                              # Windows per batch
    'workers': 0,                                 # DataLoader workers
    'device': 'cuda',                             # Logical CUDA device or cpu
    'seed': 42,                                   # Initialisation and training sampling seed
    'log_every': 20,                              # Progress interval in batches
    'save_every': 5,                              # Numbered checkpoint interval
    'evaluation_checkpoint': 'last',              # last or best, from this experiment's own training
    'output_root': './results/SRES_fMRI_cc2017/v2/experiments',
    'save_reconstructions': True,                 # Save per-run reconstructed BOLD and validity masks
    'plot_run': 1,                                # One-based run to illustrate in each split
    'plot_voxels': '0,1,2',                       # Zero-based voxel indices for example traces
    'plot_max_trs': 120,                          # Original TRs shown per example
    'launch_script': '',                          # launch script path (COPY)
}
# =======================================================================

# import libraries
import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
from time import monotonic
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from ....modules.dataset_cc2017 import DatasetCC2017
from ....modules.physics import ObservationModel
from ....modules.runtime import atomic_save, log, make_loader, progress, save_json, setup
from ....modules.scheduling import make_scheduler
from ..config import Config
from ..network import SuperResolutionHead
from ..train import losses, pretrain


def paired_sums(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    # Accumulate errors and temporal correlations for [time, repetition, voxel] series
    # Correlate within each run and repetition; never join unrelated time sequences
    x, y = target.double(), prediction.double()
    difference = y - x
    x = x - x.mean(dim = 0, keepdim = True)
    y = y - y.mean(dim = 0, keepdim = True)
    denominator = (x.square().sum(0) * y.square().sum(0)).sqrt()
    valid = denominator > 0
    correlations = ((x * y).sum(0)[valid] / denominator[valid]).clamp(-1, 1)
    return {'elements': target.numel(), 'squared_error': difference.square().sum().item(),
            'absolute_error': difference.abs().sum().item(), 'correlation_sum': correlations.sum().item(),
            'correlation_count': valid.sum().item(), 'series_count': denominator.numel()}


def main(args: argparse.Namespace) -> Path:
    # Keep one retained phase across all windows, including the loader's final window
    factor, upsample = args.input_downsample_factor, args.tsr_factor
    if factor < 2 or upsample < 1 or upsample % factor:
        raise ValueError('INPUT_DOWNSAMPLE_FACTOR must be at least 2 and divide TSR_FACTOR.')
    lengths = {phase: [int(n) for n in getattr(args, f'{phase}_run_lengths').split(',')]
               for phase in ('train', 'test')}
    if any(n <= 0 or n % factor for runs in lengths.values() for n in runs):
        raise ValueError('Every run length must be positive and divisible by INPUT_DOWNSAMPLE_FACTOR.')
    if args.window_trs % factor or args.window_stride % factor:
        raise ValueError('Window length and stride must be divisible by INPUT_DOWNSAMPLE_FACTOR to prevent phase leakage.')
    if args.burnin_trs < 0 or args.lookahead_trs < 0:
        raise ValueError('Window exclusions cannot be negative.')
    if args.evaluation_checkpoint not in ('last', 'best'):
        raise ValueError('Select last or best from this experiment; pretrained v2 weights are not accepted.')
    plot_voxels = [int(n) for n in args.plot_voxels.split(',')]
    if not plot_voxels or min(plot_voxels) < 0 or args.plot_max_trs < 1 or not 1 <= args.plot_run <= min(map(len, lengths.values())):
        raise ValueError('Choose non-negative plot voxel indices, a positive plot length and a run present in both splits.')

    # The loader retains the original grid; only strided observations reach the inverse and loss
    names = {field.name for field in fields(Config)}
    common = {key: value for key, value in vars(args).items() if key in names}
    burnin = math.ceil(args.burnin_trs / factor)
    trailing = math.ceil(args.lookahead_trs / factor)
    # No stimulus remapping: this experiment reconstructs BOLD on its stored acquisition axis
    data_config = Config(**(common | {'stored_delay_sec': 0.0, 'tsr_factor': upsample // factor,
                                     'burnin_trs': burnin * factor, 'lookahead_trs': trailing * factor}))
    model_config = replace(data_config, tr_sec = args.tr_sec * factor, tsr_factor = upsample,
                           window_trs = args.window_trs // factor, window_stride = args.window_stride // factor,
                           burnin_trs = burnin, lookahead_trs = trailing)
    data_config.validate()
    device = setup(model_config)
    log(f'Original TR = {args.tr_sec:g}s; input TR = {model_config.tr_sec:g}s; activity dt = {model_config.dt_sec:g}s')
    log(f'Retained phase = 0 per run; effective exclusions = {burnin * factor}/{trailing * factor} original TRs')
    log('Omitted-frame reconstruction tests BOLD interpolation, not recovery of ground-truth neural activity', 'WARN')

    training = DatasetCC2017(data_config, 'train', 'training', average_repeats = args.average_repeats)
    validation = DatasetCC2017(data_config, 'train', 'validation', average_repeats = args.average_repeats, source = training.data)
    if max(plot_voxels) >= training.voxels:
        raise ValueError('A plot voxel lies outside the saved voxel array.')
    train_loader = make_loader(training, model_config, True)
    validation_loader = make_loader(validation, model_config, False)
    # Create a distinct experiment directory and record both physical time grids
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = Path(args.output_root) / f'correctness_undersampled_input_subj{args.subject}_{stamp}'
    folder.mkdir(parents = True, exist_ok = False)
    for part in ('checkpoints', 'artifacts', 'logs'):
        (folder / part).mkdir()
    save_json(folder / 'arguments.json', {'experiment': vars(args), 'data_config': data_config.to_dict(),
                                         'model_config': model_config.to_dict(), 'training_source': training.provenance()})
    if args.launch_script:
        shutil.copy2(args.launch_script, folder / Path(args.launch_script).name)
    log(f'Experiment: {folder}')

    # Always initialise fresh weights; validation uses retained frames only
    model = SuperResolutionHead(model_config, training.voxels).to(device)
    optimiser = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr = model_config.lr, weight_decay = model_config.weight_decay)
    scheduler = make_scheduler(optimiser, args.epochs * len(train_loader), args.warmup_fraction, args.minimum_lr_ratio)
    writer = SummaryWriter(str(folder / 'logs'))
    history: list[dict[str, Any]] = []
    best, global_step = float('inf'), 0
    try:
        if args.pretrain_synthetic:
            pretrain(model, model_config, writer)
        started = monotonic()
        for epoch in range(args.epochs):
            epoch_result: dict[str, Any] = {'epoch': epoch + 1}
            for label, loader in (('train', train_loader), ('validation', validation_loader)):
                model.train(label == 'train')
                sums: dict[str, float] = {}
                samples = 0
                generator = torch.Generator(device = device).manual_seed(args.seed + 100000) if label == 'validation' else None
                for batch in loader:
                    if (batch['start'] % factor).any():
                        raise ValueError('Window start changes the retained-frame phase.')
                    count = training.voxels if args.enable_spatial_spsf else min(args.voxel_batch_size, training.voxels)
                    ids = torch.arange(training.voxels, device = device) if count == training.voxels else torch.randperm(training.voxels, device = device, generator = generator)[:count]
                    observed = batch['bold'][..., ::factor].to(device)[:, ids]
                    with torch.set_grad_enabled(label == 'train'):
                        # Only retained BOLD supplies the input and self-supervised reconstruction target
                        terms = losses(model(observed, ids), observed, model_config)
                        if not torch.isfinite(terms['total']):
                            raise FloatingPointError('Non-finite training or validation loss.')
                        if label == 'train':
                            optimiser.zero_grad(set_to_none = True)
                            terms['total'].backward()
                            if args.max_grad_norm:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite = True)
                            optimiser.step()
                            scheduler.step()
                            global_step += 1
                            writer.add_scalar('train/lr', optimiser.param_groups[0]['lr'], global_step)
                            if global_step % args.log_every == 0 or global_step == args.epochs * len(train_loader):
                                progress(f'Epoch {epoch + 1}: loss = {terms["total"].item():.5g}', global_step,
                                         args.epochs * len(train_loader), started)
                    for name, value in terms.items():
                        sums[name] = sums.get(name, 0.0) + value.detach().item() * len(observed)
                    samples += len(observed)
                epoch_result[label] = {name: value / samples for name, value in sums.items()}
                for name, value in epoch_result[label].items():
                    writer.add_scalar(f'{label}/{name}', value, epoch + 1)
            improved = epoch_result['validation']['total'] < best
            best = min(best, epoch_result['validation']['total'])
            payload = {'experiment': vars(args), 'config': model_config.to_dict(), 'data_config': data_config.to_dict(),
                       'voxels': training.voxels, 'model': model.state_dict(), 'optimiser': optimiser.state_dict(),
                       'scheduler': scheduler.state_dict(), 'epoch': epoch + 1, 'global_step': global_step, 'best': best,
                       'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
            atomic_save(payload, folder / 'checkpoints/last.pt')
            if improved:
                atomic_save(payload, folder / 'checkpoints/best.pt')
            if (epoch + 1) % args.save_every == 0:
                atomic_save(payload, folder / 'checkpoints' / f'epoch_{epoch + 1}.pt')
            history.append(epoch_result)
            save_json(folder / 'training_metrics.json', history)
            log(f'Epoch {epoch + 1}: train = {epoch_result["train"]["total"]:.5g}, validation = {epoch_result["validation"]["total"]:.5g}')
    finally:
        writer.close()

    # Restore only this experiment's selected weights before inspecting omitted-frame targets
    checkpoint = folder / 'checkpoints' / f'{args.evaluation_checkpoint}.pt'
    saved = torch.load(checkpoint, map_location = 'cpu', weights_only = False)
    model.load_state_dict(saved['model'], strict = True)
    model.eval().requires_grad_(False)
    del saved, optimiser, scheduler, train_loader, validation_loader, validation
    # Same kernel samples, learned dilation, sPSF and baseline; only the observation sampling stride changes
    original_physics = ObservationModel(data_config, model.voxels).to(device)
    original_physics.load_state_dict(model.physics.state_dict(), strict = True)
    original_physics.eval().requires_grad_(False)
    report: dict[str, Any] = {'checkpoint': str(checkpoint), 'splits': {}, 'interpretation': {
        'training': 'Fresh model; only retained rows from fitting runs enter optimisation; retained validation rows select best weights',
        'omitted': 'Run-local rows with index modulo INPUT_DOWNSAMPLE_FACTOR != 0; excluded from all real-data training targets',
        'metrics': 'Final overlap-averaged BOLD at original TR; invalid boundaries excluded; correlations are per run/repetition/voxel',
        'baseline': 'Linear interpolation using retained observations only, without joining runs or repetitions',
        'timing': 'Stored BOLD acquisition axis; no stimulus-delay remapping or original-value boundary fallback',
        'scope': 'Missing BOLD recovery is not proof of neural activity recovery; direct decimation can alias frequencies',
    }}
    local = torch.arange(burnin * factor, args.window_trs - trailing * factor)
    fig, axes = plt.subplots(len(plot_voxels), 2, figsize = (13, 3 * len(plot_voxels)), squeeze = False, constrained_layout = True)
    for column, phase in enumerate(('train', 'test')):
        data = DatasetCC2017(data_config, phase, average_repeats = args.average_repeats,
                             source = training.data if phase == 'train' else None)
        if data.voxels != model.voxels:
            raise ValueError('Train and test voxel counts differ.')
        # Group cached window indices once; aggregate only one run at a time to bound memory
        run_windows: dict[int, list[int]] = {run: [] for run in data.run_ids}
        for index, (run, _, _) in enumerate(data.windows):
            run_windows[run].append(index)
        groups: dict[str, dict[str, Any]] = {}
        repetitions = 1 if args.average_repeats else data.repetitions
        completed, started = 0, monotonic()
        total_batches = sum(math.ceil(len(indices) / args.batch_size) for indices in run_windows.values())
        for run in data.run_ids:
            length = data.run_lengths[run]
            total = torch.zeros(length, repetitions, data.voxels)
            counts = torch.zeros(length, repetitions, dtype = torch.int32)
            loader = make_loader(Subset(data, run_windows[run]), model_config, False)
            with torch.inference_mode():
                for batch in loader:
                    if (batch['start'] % factor).any():
                        raise ValueError('Window start changes the retained-frame phase.')
                    activity = model.activity(batch['bold'][..., ::factor].to(device))
                    prediction = original_physics(activity, model.baseline).transpose(1, 2).cpu()
                    if not torch.isfinite(prediction).all():
                        raise FloatingPointError('Non-finite reconstructed BOLD.')
                    for sample in range(len(prediction)):
                        rows = int(batch['start'][sample]) + local
                        repetition = int(batch['repetition'][sample])
                        total[rows, repetition] += prediction[sample, local]
                        counts[rows, repetition] += 1
                    completed += 1
                    if completed % args.log_every == 0 or completed == total_batches:
                        progress(f'Evaluating {phase}', completed, total_batches, started)
            valid = counts > 0
            total /= counts.clamp_min(1).unsqueeze(-1)
            total[~valid] = float('nan')
            observed_rows = torch.arange(length) % factor == 0
            # Match supported windows across methods; forbid extrapolation by the baseline
            bracketed = torch.arange(length) <= ((length - 1) // factor) * factor
            valid &= bracketed[:, None]
            total[~valid] = float('nan')
            scope = ('validation_runs' if run + 1 in {int(n) for n in args.validation_runs.split(',') if n.strip()} else 'fitting_runs') if phase == 'train' else 'test_runs'
            target = data.data[data.offsets[run]:data.offsets[run + 1]]
            target = target.mean(1, keepdim = True) if args.average_repeats else target
            # Compute both metrics and an interpolation control in small voxel chunks
            positions = torch.arange(length)
            left = (positions // factor) * factor
            right = (left + factor).clamp_max(((length - 1) // factor) * factor)
            weight = ((positions - left) / factor)[:, None, None]
            for first in range(0, data.voxels, args.voxel_batch_size):
                last = min(first + args.voxel_batch_size, data.voxels)
                truth = target[..., first:last]
                interpolated = truth[left] * (1 - weight) + truth[right] * weight
                for selection, row_mask in (('all', torch.ones(length, dtype = torch.bool)), ('observed', observed_rows), ('omitted', ~observed_rows)):
                    # Every repetition has identical window coverage; retain the time axis for correlation
                    selected = valid[:, 0] & row_mask
                    if not selected.any():
                        raise ValueError(f'No valid {selection} rows in {phase} run {run + 1}.')
                    for method, prediction in (('model', total[..., first:last]), ('linear_interpolation', interpolated)):
                        partial = paired_sums(truth[selected], prediction[selected])
                        for group in ('all_runs', scope):
                            entry = groups.setdefault(group, {}).setdefault(selection, {}).setdefault(method, {})
                            for key, value in partial.items():
                                entry[key] = entry.get(key, 0.0) + value
            if args.save_reconstructions:
                atomic_save({'reconstructed_bold': total, 'valid': valid, 'overlap_counts': counts,
                             'observed_rows': observed_rows, 'source': str(data.path.resolve()),
                             'original_row_indices': torch.arange(length) + data.offsets[run],
                             'tr_sec': args.tr_sec, 'average_repeats': args.average_repeats,
                             'input_downsample_factor': factor, 'checkpoint': str(checkpoint)},
                            folder / 'artifacts' / f'{phase}_run{run + 1:02d}.pt')
            if run + 1 == args.plot_run:
                # Show repetition zero (or the repetition mean) without averaging unrelated voxels
                stop = min(length, args.plot_max_trs)
                times = np.arange(stop) * args.tr_sec
                for index, voxel in enumerate(plot_voxels):
                    ax = axes[index, column]
                    truth = target[:stop, 0, voxel].numpy()
                    baseline = (target[left, 0, voxel] * (1 - weight[:, 0, 0]) + target[right, 0, voxel] * weight[:, 0, 0]).numpy()
                    ax.plot(times, truth, color = 'black', linewidth = 1, label = 'Original BOLD')
                    ax.plot(times, total[:stop, 0, voxel].numpy(), color = 'tab:orange', linewidth = 1, label = 'HRF reconstruction')
                    ax.plot(times, np.where(valid[:stop, 0].numpy(), baseline[:stop], np.nan), color = 'tab:blue', linestyle = '--', linewidth = 1, label = 'Linear interpolation')
                    retained = observed_rows[:stop].numpy()
                    ax.scatter(times[retained], truth[retained], s = 12, color = 'black', label = 'Retained input', zorder = 3)
                    ax.set(title = f'{phase}, run {run + 1}, voxel {voxel}', xlabel = 'Run time (s)', ylabel = 'BOLD')
                    ax.grid(alpha = 0.2)
                    if index == 0:
                        ax.legend(fontsize = 7)
            del total, counts, target, loader
        # Convert sufficient statistics after accumulation; constant series have undefined correlation
        for selections in groups.values():
            for methods in selections.values():
                for method, sums in methods.items():
                    mse = sums['squared_error'] / sums['elements']
                    methods[method] = {'mse': mse, 'rmse': math.sqrt(mse), 'mae': sums['absolute_error'] / sums['elements'],
                                       'mean_temporal_pearson': sums['correlation_sum'] / sums['correlation_count'] if sums['correlation_count'] else None,
                                       'elements': int(sums['elements']), 'nonconstant_series': int(sums['correlation_count']),
                                       'series_count': int(sums['series_count'])}
        report['splits'][phase] = {'shape': [len(data.data), repetitions, data.voxels], 'groups': groups}
        for group, selections in groups.items():
            missing = selections['omitted']
            log(f'{phase}/{group}: omitted-frame MSE = {missing["model"]["mse"]:.6g}; linear baseline = {missing["linear_interpolation"]["mse"]:.6g}; '
                f'Pearson = {missing["model"]["mean_temporal_pearson"]}')
        save_json(folder / 'metrics.json', report)
        del data
    for extension in ('png', 'pdf'):
        fig.savefig(folder / f'reconstruction_examples.{extension}', dpi = 180)
    plt.close(fig)
    log(f'Experiment complete: {folder}', 'OK')
    return folder


if __name__ == '__main__':
    # All defaults live in the setup zone; the Bash launcher passes settings explicitly
    parser = argparse.ArgumentParser(description = 'Train fresh v2 weights on decimated BOLD and evaluate omitted frames.')
    for name, value in DEFAULTS.items():
        option = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            parser.add_argument(option, action = argparse.BooleanOptionalAction, default = value)
        else:
            parser.add_argument(option, type = type(value), default = value)
    main(parser.parse_args())
