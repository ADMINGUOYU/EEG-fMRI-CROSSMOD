# file: hidden_retained_prediction.py
# Select activity TV by predicting concealed retained validation measurements
# Original:                A B C D E F G H I
# 
# correctness_undersampled_input exp. with factor 4 (default 2):
# Training input:          A       E       I
# Validation input:        A       E       I
# Validation targets:        B C D   F G H
# 
# This exp.:
# Training input:          A   C   E   G   I
# Validation input:        A   c*  E   g*  I
# Validation targets:          C       G
# 
# c* = interpolation from A and E
# g* = interpolation from E and I
# Run with `python -m models.SRES_fMRI_cc2017.models.v2.experiments.hidden_retained_prediction`

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
    'seeds': '42',                                # Comma-separated paired initialisation seeds
    'log_every': 20,                              # Progress interval in batches
    'save_every': 5,                              # Numbered checkpoint interval
    'evaluation_checkpoint': 'best',              # best validation prediction or last epoch
    'output_root': './results/SRES_fMRI_cc2017/v2/experiments',
    'save_reconstructions': True,                 # Save per-run reconstructed BOLD and validity masks
    'plot_run': 1,                                # One-based run to illustrate in each split
    'plot_voxels': '0,1,2',                       # Zero-based voxel indices for example traces
    'plot_max_trs': 120,                          # Original TRs shown per example
    'tv_strengths': '0,0.001,0.01,0.1',           # Activity TV weights, including the un-regularised control
    'hidden_stride': 2,                           # Conceal one retained position in each group, validation only
    'hidden_phase': 1,                            # Run-local retained index modulo HIDDEN_STRIDE to conceal
    'launch_script': '',                          # launch script path (COPY)
}
# =======================================================================

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


def parse_arguments(defaults: dict[str, Any], description: str) -> argparse.Namespace:
    # Keep every launcher setting in its experiment's setup zone
    parser = argparse.ArgumentParser(description = description)
    for name, value in defaults.items():
        option = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            parser.add_argument(option, action = argparse.BooleanOptionalAction, default = value)
        else:
            parser.add_argument(option, type = type(value), default = value)
    return parser.parse_args()


def interpolation_plan(length: int, factor: int, hidden_stride: int = 0,
                       hidden_phase: int = 1) -> tuple[torch.Tensor, ...]:
    # Conceal the same run-local retained rows in every overlapping window
    positions = torch.arange(length)
    retained = positions % factor == 0
    hidden = retained & ((positions // factor) % hidden_stride == hidden_phase) if hidden_stride else torch.zeros_like(retained)
    # Preserve the first and last retained values so hidden targets are bracketed
    last = ((length - 1) // factor) * factor
    hidden[0] = hidden[last] = False
    visible = positions[retained & ~hidden]
    if len(visible) < 2:
        raise ValueError('Each run needs at least two visible retained measurements.')
    right_index = torch.searchsorted(visible, positions).clamp(max = len(visible) - 1)
    left_index = (right_index - 1).clamp(min = 0)
    exact = visible[right_index] == positions
    left_index[exact] = right_index[exact]
    left, right = visible[left_index], visible[right_index]
    weight = (positions - left) / (right - left).clamp_min(1)
    return retained, hidden, left, right, weight, positions <= last


def finish_metrics(sums: dict[str, float]) -> dict[str, Any]:
    # Convert sufficient statistics only after pooling runs and voxel chunks
    if not sums or sums['elements'] == 0:
        raise ValueError('No supported samples remain for a requested metric.')
    mse = sums['squared_error'] / sums['elements']
    return {'mse': mse, 'rmse': math.sqrt(mse), 'mae': sums['absolute_error'] / sums['elements'],
            'mean_temporal_pearson': sums['correlation_sum'] / sums['correlation_count'] if sums['correlation_count'] else None,
            'mean_absolute_prediction': sums['prediction_magnitude'] / sums['elements'],
            'mean_absolute_target': sums['target_magnitude'] / sums['elements'],
            'elements': int(sums['elements']), 'nonconstant_series': int(sums['correlation_count']),
            'series_count': int(sums['series_count'])}


def evaluate(model: SuperResolutionHead, data: DatasetCC2017, data_config: Config,
             args: argparse.Namespace, hidden: bool = False, folder: Path | None = None) -> dict[str, Any]:
    # Reconstruct on the original grid without changing the learned HRF or baseline
    model.eval()
    device = next(model.parameters()).device
    physics = ObservationModel(data_config, model.voxels).to(device)
    physics.load_state_dict(model.physics.state_dict(), strict = True)
    factor = args.input_downsample_factor
    local = torch.arange(data_config.burnin_trs, data_config.window_trs - data_config.lookahead_trs)
    run_windows: dict[int, list[int]] = {run: [] for run in data.run_ids}
    for index, (run, _, _) in enumerate(data.windows):
        run_windows[run].append(index)
    repetitions = 1 if args.average_repeats else data.repetitions
    accumulated: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    plot_voxels = [int(n) for n in args.plot_voxels.split(',')]
    validation_runs = {int(n) for n in args.validation_runs.split(',')}
    completed, started = 0, monotonic()
    total_batches = sum(math.ceil(len(indices) / args.batch_size) for indices in run_windows.values())
    activity_tv_sum, activity_tv_elements = 0.0, 0
    for run in data.run_ids:
        length = data.run_lengths[run]
        target = data.data[data.offsets[run]:data.offsets[run + 1]]
        target = target.mean(1, keepdim = True) if args.average_repeats else target
        retained, concealed, left, right, weight, bracketed = interpolation_plan(
            length, factor, args.hidden_stride if hidden else 0, args.hidden_phase if hidden else 1)
        total = torch.zeros(length, repetitions, data.voxels)
        counts = torch.zeros(length, repetitions, dtype = torch.int32)
        loader = make_loader(Subset(data, run_windows[run]), model.config, False)
        with torch.inference_mode():
            for batch in loader:
                if (batch['start'] % factor).any():
                    raise ValueError('Window start changes the retained-frame phase.')
                observed = batch['bold'][..., ::factor].clone()
                if hidden:
                    # Fill from visible neighbours only; never feed a concealed value in another window
                    # The mask is used for scoring; no extra input channel or model change is introduced
                    for sample, start in enumerate(batch['start'].tolist()):
                        rows = start + torch.arange(observed.shape[-1]) * factor
                        rows = rows[concealed[rows]]
                        repeat = int(batch['repetition'][sample])
                        filled = target[left[rows], repeat] * (1 - weight[rows, None]) + target[right[rows], repeat] * weight[rows, None]
                        observed[sample, :, (rows - start) // factor] = filled.T
                activity = model.activity(observed.to(device))
                first = model.config.burnin_trs * model.config.tsr_factor
                last = (model.config.window_trs - model.config.lookahead_trs) * model.config.tsr_factor
                changes = activity[..., first:last].diff(dim = -1).abs() / model.config.dt_sec
                activity_tv_sum += changes.sum().item()
                activity_tv_elements += changes.numel()
                prediction = physics(activity, model.baseline).transpose(1, 2).cpu()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError('Non-finite reconstructed BOLD.')
                for sample in range(len(prediction)):
                    rows = int(batch['start'][sample]) + local
                    repeat = int(batch['repetition'][sample])
                    total[rows, repeat] += prediction[sample, local]
                    counts[rows, repeat] += 1
                completed += 1
                if completed % args.log_every == 0 or completed == total_batches:
                    progress(f'Evaluating {data.phase}' + (' hidden retained' if hidden else ''), completed, total_batches, started)
        valid = (counts > 0) & bracketed[:, None]
        total /= counts.clamp_min(1).unsqueeze(-1)
        total[~valid] = float('nan')
        if not torch.equal(valid, valid[:, :1].expand_as(valid)):
            raise ValueError('Repetitions have different evaluation coverage.')
        scope = ('validation_runs' if run + 1 in validation_runs else 'fitting_runs') if data.phase == 'train' else 'test_runs'
        selections = {'hidden_retained': concealed} if hidden else {
            'all': torch.ones(length, dtype = torch.bool), 'observed': retained, 'omitted': ~retained}
        run_sums: dict[str, Any] = {}
        for first in range(0, data.voxels, args.voxel_batch_size):
            last = min(first + args.voxel_batch_size, data.voxels)
            truth = target[..., first:last]
            baseline = truth[left] * (1 - weight[:, None, None]) + truth[right] * weight[:, None, None]
            predictions = {'model': total[..., first:last], 'linear_interpolation': baseline}
            # Differences use consecutive original-grid rows; never bridge gaps, runs or repetitions
            targets = {'bold': truth}
            masks = {name: valid[:, 0] & mask for name, mask in selections.items()}
            if not hidden:
                for order, name in ((1, 'first_difference'), (2, 'second_difference')):
                    targets[name] = truth.diff(n = order, dim = 0) / args.tr_sec ** order
                    support = torch.stack([valid[offset:length - order + offset, 0] for offset in range(order + 1)]).all(0)
                    masks[name] = support
            for method, prediction in predictions.items():
                for name, selected in masks.items():
                    if name in selections:
                        x, y = truth, prediction
                    else:
                        order = 1 if name == 'first_difference' else 2
                        x, y = targets[name], prediction.diff(n = order, dim = 0) / args.tr_sec ** order
                    if not selected.any():
                        raise ValueError(f'No supported {name} rows in {data.phase} run {run + 1}.')
                    partial = paired_sums(x[selected], y[selected])
                    partial['prediction_magnitude'] = y[selected].double().abs().sum().item()
                    partial['target_magnitude'] = x[selected].double().abs().sum().item()
                    for destination in (run_sums, accumulated.setdefault('all_runs', {}), accumulated.setdefault(scope, {})):
                        entry = destination.setdefault(name, {}).setdefault(method, {})
                        for key, value in partial.items():
                            entry[key] = entry.get(key, 0.0) + value
        runs[str(run + 1)] = {name: {method: finish_metrics(sums) for method, sums in methods.items()}
                             for name, methods in run_sums.items()}
        if folder is not None:
            folder.mkdir(parents = True, exist_ok = True)
            if args.save_reconstructions:
                atomic_save({'reconstructed_bold': total, 'valid': valid, 'overlap_counts': counts,
                             'retained_rows': retained, 'hidden_rows': concealed,
                             'source': str(data.path.resolve()), 'original_row_indices': torch.arange(length) + data.offsets[run],
                             'tr_sec': args.tr_sec, 'average_repeats': args.average_repeats},
                            folder / f'{data.phase}_run{run + 1:02d}.pt')
            if run + 1 == args.plot_run:
                fig, axes = plt.subplots(len(plot_voxels), 1, figsize = (11, 3 * len(plot_voxels)), squeeze = False, constrained_layout = True)
                stop = min(length, args.plot_max_trs)
                times = np.arange(stop) * args.tr_sec
                for axis, voxel in zip(axes[:, 0], plot_voxels):
                    baseline = target[left, 0, voxel] * (1 - weight) + target[right, 0, voxel] * weight
                    axis.plot(times, target[:stop, 0, voxel], color = 'black', linewidth = 1, label = 'Original BOLD')
                    axis.plot(times, total[:stop, 0, voxel], color = 'tab:orange', linewidth = 1, label = 'HRF reconstruction')
                    axis.plot(times, np.where(valid[:stop, 0], baseline[:stop], np.nan), '--', linewidth = 1, label = 'Linear interpolation')
                    visible = (retained & ~concealed)[:stop]
                    axis.scatter(times[visible], target[:stop, 0, voxel][visible], s = 10, color = 'black', label = 'Visible input')
                    axis.set(title = f'{data.phase}, run {run + 1}, voxel {voxel}', xlabel = 'Run time (s)', ylabel = 'BOLD')
                    axis.grid(alpha = 0.2)
                axes[0, 0].legend(fontsize = 8)
                for extension in ('png', 'pdf'):
                    fig.savefig(folder / f'{data.phase}_examples.{extension}', dpi = 180)
                plt.close(fig)
    groups = {group: {name: {method: finish_metrics(sums) for method, sums in methods.items()}
                      for name, methods in selections.items()} for group, selections in accumulated.items()}
    return {'shape': [len(data.data), repetitions, data.voxels], 'groups': groups, 'runs': runs,
            'activity_tv': activity_tv_sum / activity_tv_elements,
            'activity_tv_scope': 'Mean absolute activity derivative over valid window interiors; overlaps are counted per window'}


def fit_variant(args: argparse.Namespace, data_config: Config, model_config: Config,
                training: DatasetCC2017, validation: DatasetCC2017, folder: Path,
                variant: str, hidden: bool, pretraining: bool, fine_tune: bool) -> dict[str, Any]:
    # Reset initialisation per variant; pretraining consumes the same random stream for each paired seed
    device = setup(model_config)
    model = SuperResolutionHead(model_config, training.voxels).to(device)
    folder.mkdir(parents = True, exist_ok = False)
    writer = SummaryWriter(str(folder / 'logs'))
    save_json(folder / 'arguments.json', {'experiment': vars(args), 'variant': variant,
                                         'data_config': data_config.to_dict(), 'model_config': model_config.to_dict(),
                                         'pretraining': pretraining, 'fine_tune': fine_tune})
    history: list[dict[str, Any]] = []
    score_name = 'hidden_retained' if hidden else 'omitted'
    best, best_epoch, global_step = float('inf'), 0, 0
    try:
        if pretraining:
            pretrain(model, model_config, writer)
        # Data order and voxel choices must not change merely because pretraining consumed random numbers
        setup(model_config)
        train_loader = make_loader(training, model_config, True)
        train_loader.generator = torch.Generator().manual_seed(args.seed + 1)
        train_loader.sampler.generator = torch.Generator().manual_seed(args.seed + 2)
        voxel_generator = torch.Generator(device = device).manual_seed(args.seed + 3)
        optimiser = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr = args.lr, weight_decay = args.weight_decay)
        scheduler = make_scheduler(optimiser, max(args.epochs * len(train_loader), 1), args.warmup_fraction, args.minimum_lr_ratio)
        started = monotonic()
        # Epoch zero records whether real-data adaptation improves on the initial inverse
        for epoch in range((args.epochs if fine_tune else 0) + 1):
            epoch_result: dict[str, Any] = {'epoch': epoch}
            if epoch:
                model.train()
                sums: dict[str, float] = {}
                samples = 0
                for batch in train_loader:
                    if (batch['start'] % args.input_downsample_factor).any():
                        raise ValueError('Window start changes the retained-frame phase.')
                    count = training.voxels if args.enable_spatial_spsf else min(args.voxel_batch_size, training.voxels)
                    ids = torch.arange(training.voxels, device = device) if count == training.voxels else torch.randperm(training.voxels, device = device, generator = voxel_generator)[:count]
                    observed = batch['bold'][..., ::args.input_downsample_factor].to(device)[:, ids]
                    terms = losses(model(observed, ids), observed, model_config)
                    if not all(torch.isfinite(value).item() for value in terms.values()):
                        raise FloatingPointError('Non-finite fitting loss.')
                    optimiser.zero_grad(set_to_none = True)
                    terms['total'].backward()
                    if args.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite = True)
                    optimiser.step()
                    scheduler.step()
                    global_step += 1
                    for name, value in terms.items():
                        sums[name] = sums.get(name, 0.0) + value.item() * len(observed)
                    samples += len(observed)
                    writer.add_scalar('train/lr', optimiser.param_groups[0]['lr'], global_step)
                    if global_step % args.log_every == 0 or global_step == args.epochs * len(train_loader):
                        progress(f'{variant}, epoch {epoch}: loss = {terms["total"].item():.5g}', global_step,
                                 args.epochs * len(train_loader), started)
                epoch_result['train'] = {name: value / samples for name, value in sums.items()}
                for name, value in epoch_result['train'].items():
                    writer.add_scalar(f'train/{name}', value, epoch)
            # Full validation runs, overlap averaged: no official test samples enter checkpoint selection
            result = evaluate(model, validation, data_config, args, hidden = hidden)
            score = result['groups']['all_runs'][score_name]['model']['mse']
            epoch_result['validation'] = result
            writer.add_scalar(f'validation/{score_name}_mse', score, epoch)
            writer.add_scalar('validation/activity_tv', result['activity_tv'], epoch)
            # Epoch zero is diagnostic; fine-tuned variants select trained checkpoints only
            eligible = epoch > 0 or not fine_tune
            improved = eligible and score < best
            if improved:
                best, best_epoch = score, epoch
            payload = {'experiment': vars(args), 'variant': variant, 'config': model_config.to_dict(),
                       'data_config': data_config.to_dict(), 'voxels': training.voxels, 'model': model.state_dict(),
                       'epoch': epoch, 'global_step': global_step, 'selection_metric': score_name + '_mse',
                       'validation_score': score, 'best_epoch': best_epoch,
                       'optimiser': optimiser.state_dict(), 'scheduler': scheduler.state_dict()}
            atomic_save(payload, folder / 'checkpoints/last.pt')
            if improved:
                atomic_save(payload, folder / 'checkpoints/best.pt')
            if epoch and epoch % args.save_every == 0:
                atomic_save(payload, folder / 'checkpoints' / f'epoch_{epoch}.pt')
            history.append(epoch_result)
            save_json(folder / 'training_metrics.json', history)
            log(f'{variant}, epoch {epoch}: validation {score_name} MSE = {score:.6g}; activity TV = {result["activity_tv"]:.6g}')
    finally:
        writer.close()
    # Show whether fine-tuning trades prediction accuracy for a smoother latent sequence
    fig, axes = plt.subplots(1, 2, figsize = (10, 4), constrained_layout = True)
    epochs = [entry['epoch'] for entry in history]
    for axis, values, title in (
            (axes[0], [entry['validation']['groups']['all_runs'][score_name]['model']['mse'] for entry in history],
             f'Validation {score_name.replace("_", " ")} MSE'),
            (axes[1], [entry['validation']['activity_tv'] for entry in history], 'Validation activity TV')):
        axis.plot(epochs, values, marker = 'o', markersize = 3)
        axis.set(title = title, xlabel = 'Real-data epoch; zero is before fine-tuning')
        axis.grid(alpha = 0.2)
    for extension in ('png', 'pdf'):
        fig.savefig(folder / f'learning_curves.{extension}', dpi = 180)
    plt.close(fig)
    chosen = best_epoch if args.evaluation_checkpoint == 'best' else history[-1]['epoch']
    return {'variant': variant, 'seed': args.seed, 'lambda_tv': args.lambda_tv,
            'folder': str(folder), 'checkpoint': str(folder / 'checkpoints' / f'{args.evaluation_checkpoint}.pt'),
            'epoch': chosen, 'selection_metric': score_name + '_mse',
            'selection_score': history[chosen]['validation']['groups']['all_runs'][score_name]['model']['mse']}


def main(args: argparse.Namespace) -> Path:
    # Keep the original BOLD grid separate from the retained input grid
    purpose = 'hidden_retained_prediction'
    factor, upsample = args.input_downsample_factor, args.tsr_factor
    if factor < 2 or upsample < 1 or upsample % factor:
        raise ValueError('INPUT_DOWNSAMPLE_FACTOR must be at least 2 and divide TSR_FACTOR.')
    lengths = {phase: [int(n) for n in getattr(args, f'{phase}_run_lengths').split(',')] for phase in ('train', 'test')}
    if any(n <= 0 or n % factor for runs in lengths.values() for n in runs) or args.window_trs % factor or args.window_stride % factor:
        raise ValueError('Run lengths, window length and stride must be positive multiples of INPUT_DOWNSAMPLE_FACTOR.')
    if args.burnin_trs < 0 or args.lookahead_trs < 0 or args.evaluation_checkpoint not in ('best', 'last'):
        raise ValueError('Choose non-negative exclusions and evaluation checkpoint best or last.')
    seeds = [int(value) for value in args.seeds.split(',')]
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError('Choose distinct non-negative seeds.')
    hidden = True
    if hidden and (args.hidden_stride < 2 or not 0 <= args.hidden_phase < args.hidden_stride):
        raise ValueError('HIDDEN_STRIDE must be at least two and HIDDEN_PHASE must lie within it.')
    strengths = [float(value) for value in args.tv_strengths.split(',')]
    if not strengths or len(set(strengths)) != len(strengths) or any(not math.isfinite(value) or value < 0 for value in strengths):
        raise ValueError('TV strengths must be distinct finite non-negative values.')
    if args.pretrain_synthetic and args.synthetic_steps < 1:
        raise ValueError('Enabled synthetic pretraining requires positive synthetic steps.')
    cases = [(f'tv_{index:02d}_{value:g}', value, args.pretrain_synthetic, True) for index, value in enumerate(strengths)]
    plot_voxels = [int(n) for n in args.plot_voxels.split(',')]
    if min(plot_voxels) < 0 or args.plot_max_trs < 1 or not 1 <= args.plot_run <= min(map(len, lengths.values())):
        raise ValueError('Choose valid plot voxel indices, a positive trace length and a run present in both splits.')
    names = {field.name for field in fields(Config)}
    common = {key: value for key, value in vars(args).items() if key in names}
    burnin, trailing = math.ceil(args.burnin_trs / factor), math.ceil(args.lookahead_trs / factor)
    data_config = Config(**(common | {'seed': seeds[0], 'stored_delay_sec': 0.0, 'tsr_factor': upsample // factor,
                                     'burnin_trs': burnin * factor, 'lookahead_trs': trailing * factor}))
    model_config = replace(data_config, tr_sec = args.tr_sec * factor, tsr_factor = upsample,
                           window_trs = args.window_trs // factor, window_stride = args.window_stride // factor,
                           burnin_trs = burnin, lookahead_trs = trailing)
    data_config.validate()
    setup(model_config)
    training = DatasetCC2017(data_config, 'train', 'training', average_repeats = args.average_repeats)
    validation = DatasetCC2017(data_config, 'train', 'validation', average_repeats = args.average_repeats, source = training.data)
    if max(plot_voxels) >= training.voxels:
        raise ValueError('A plot voxel lies outside the saved voxel array.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = Path(args.output_root) / f'{purpose}_subj{args.subject}_{stamp}'
    folder.mkdir(parents = True, exist_ok = False)
    save_json(folder / 'arguments.json', {'experiment': vars(args), 'purpose': purpose,
                                         'data_config': data_config.to_dict(), 'model_config': model_config.to_dict(),
                                         'training_source': training.provenance()})
    if args.launch_script:
        shutil.copy2(args.launch_script, folder / Path(args.launch_script).name)
    log(f'Experiment: {folder}')
    log(f'Original/input/activity intervals = {args.tr_sec:g}/{model_config.tr_sec:g}/{model_config.dt_sec:g}s')
    log('All fitting uses retained inputs only; validation selects checkpoints and settings; official test is evaluated afterwards')
    if hidden:
        log('Hidden validation uses run-consistent linear filling on the unchanged grid; its larger gaps are a prediction proxy', 'WARN')
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for variant, strength, pretraining, fine_tune in cases:
            current = argparse.Namespace(**(vars(args) | {'seed': seed, 'lambda_tv': strength}))
            config = replace(model_config, seed = seed, lambda_tv = strength, pretrain_synthetic = pretraining)
            record = fit_variant(current, data_config, config, training, validation,
                                 folder / f'seed_{seed}' / variant, variant, hidden, pretraining, fine_tune)
            records.append(record)
            save_json(folder / 'progress.json', records)
    # Freeze the cross-seed variant choice before loading any official test recording
    scores = {variant: float(np.mean([record['selection_score'] for record in records if record['variant'] == variant]))
              for variant, *_ in cases}
    selected_variant = min(scores, key = scores.get)
    selection = {'metric': records[0]['selection_metric'], 'mean_validation_scores': scores,
                 'selected_variant': selected_variant, 'records': records,
                 'test_used_for_selection': False}
    save_json(folder / 'selection.json', selection)
    log(f'Selected {selected_variant} using validation only; evaluating fixed checkpoints on train and test')
    all_training = DatasetCC2017(data_config, 'train', average_repeats = args.average_repeats, source = training.data)
    testing = DatasetCC2017(data_config, 'test', average_repeats = args.average_repeats)
    if testing.voxels != training.voxels:
        raise ValueError('Train and test voxel counts differ.')
    report: dict[str, Any] = {'purpose': purpose, 'selection': selection, 'results': [], 'interpretation': {
        'fitting': 'Only retained rows of fitting runs enter gradients; validation and official test never enter optimisation',
        'selection': 'Hidden retained validation MSE with linear-filled inputs' if hidden else 'Original omitted-frame validation MSE',
        'final_evaluation': 'Every fixed candidate uses all retained inputs; original omitted BOLD is used only for scoring',
        'differences': 'First and second differences use consecutive valid original rows divided by TR and TR squared',
        'scope': 'Activity TV is a prior; original BOLD derivatives are diagnostics, not ground-truth activity',
        'uncertainty': 'Seed standard deviations describe optimisation variability; per-run paired metrics are also retained',
    }}
    for record in records:
        saved = torch.load(record['checkpoint'], map_location = 'cpu', weights_only = False)
        config = Config(**saved['config'])
        device = setup(config)
        model = SuperResolutionHead(config, training.voxels).to(device)
        model.load_state_dict(saved['model'], strict = True)
        del saved
        model.eval().requires_grad_(False)
        current = argparse.Namespace(**(vars(args) | {'seed': record['seed'], 'lambda_tv': record['lambda_tv']}))
        result = dict(record) | {'splits': {}}
        for phase, data in (('train', all_training), ('test', testing)):
            result['splits'][phase] = evaluate(model, data, data_config, current, folder = Path(record['folder']) / 'artifacts')
            score = result['splits'][phase]['groups']['all_runs']['omitted']['model']['mse']
            log(f'{record["variant"]}, seed {record["seed"]}, {phase}: omitted MSE = {score:.6g}')
        report['results'].append(result)
        save_json(Path(record['folder']) / 'metrics.json', result)
        save_json(folder / 'metrics.json', report)
        del model
    # Compare interpolation error and temporal-shape error, keeping optimisation seeds paired
    fig, axes = plt.subplots(1, 4, figsize = (19, 4), constrained_layout = True)
    labels = [variant for variant, *_ in cases]
    display = [f'TV = {strength:g}' if variant.startswith('tv_') else variant.replace('_', '\n')
               for variant, strength, *_ in cases]
    summary: dict[str, Any] = {}
    for index, name in enumerate(('omitted', 'first_difference', 'second_difference')):
        for phase, group, colour in (('train', 'validation_runs', 'tab:blue'), ('test', 'test_runs', 'tab:orange')):
            values = [[result['splits'][phase]['groups'][group][name]['model']['mse'] for result in report['results'] if result['variant'] == variant] for variant in labels]
            means = [float(np.mean(value)) for value in values]
            deviations = [float(np.std(value, ddof = 1)) if len(value) > 1 else 0.0 for value in values]
            axes[index].errorbar(range(len(labels)), means, yerr = deviations, marker = 'o', color = colour, label = group)
            baseline = report['results'][0]['splits'][phase]['groups'][group][name]['linear_interpolation']['mse']
            axes[index].axhline(baseline, color = colour, linestyle = '--', alpha = 0.6, label = f'{group}: linear')
            summary.setdefault(phase, {})[name] = {'variants': labels, 'mean_mse': means, 'seed_std': deviations, 'linear_mse': baseline}
        axes[index].set(title = name.replace('_', ' '), ylabel = 'MSE', xticks = range(len(labels)), xticklabels = display)
        axes[index].tick_params(axis = 'x', labelrotation = 25)
        axes[index].grid(alpha = 0.2)
    for phase, colour in (('train', 'tab:blue'), ('test', 'tab:orange')):
        values = [[result['splits'][phase]['activity_tv'] for result in report['results'] if result['variant'] == variant]
                  for variant in labels]
        means = [float(np.mean(value)) for value in values]
        deviations = [float(np.std(value, ddof = 1)) if len(value) > 1 else 0.0 for value in values]
        axes[3].errorbar(range(len(labels)), means, yerr = deviations, marker = 'o', color = colour, label = phase)
        summary[phase]['activity_tv'] = {'variants': labels, 'mean': means, 'seed_std': deviations}
    axes[3].set(title = 'Inferred activity TV', ylabel = 'Mean absolute derivative',
                xticks = range(len(labels)), xticklabels = display)
    axes[3].tick_params(axis = 'x', labelrotation = 25)
    axes[3].grid(alpha = 0.2)
    axes[3].legend(fontsize = 7)
    axes[0].legend(fontsize = 7)
    for extension in ('png', 'pdf'):
        fig.savefig(folder / f'comparison.{extension}', dpi = 180)
    plt.close(fig)
    report['summary'] = summary
    save_json(folder / 'metrics.json', report)
    log(f'Experiment complete: {folder}', 'OK')
    return folder


if __name__ == '__main__':
    # All experiment settings are local to this module and its Bash launcher
    main(parse_arguments(DEFAULTS, 'Select activity TV by predicting concealed retained validation measurements.'))
