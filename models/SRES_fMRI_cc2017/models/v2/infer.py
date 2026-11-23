# file: infer.py
# Infer activity on the trained temporal grid and map it to stimulus indices

from __future__ import annotations

from pathlib import Path
from time import monotonic

import torch

from ...modules.dataset_cc2017 import DatasetCC2017
from .config import Config, parse_config
from .network import SuperResolutionHead
from ...modules.runtime import atomic_save, create_run, log, make_loader, progress, save_json, setup


def main(config: Config) -> Path:
    device = setup(config)
    if config.action != 'infer' or not config.checkpoint:
        raise ValueError('Inference requires ACTION=infer and CHECKPOINT.')
    saved = torch.load(config.checkpoint, map_location = 'cpu', weights_only = False)
    trained = Config(**saved['config'])
    if trained.subject != config.subject:
        raise ValueError('Inference subject does not match the trained model.')
    # Check architecture and physics against the checkpoint; allow a longer inference window
    for name in ('tr_sec', 'stored_delay_sec', 'tsr_factor', 'hrf_length_sec', 'hrf_mode',
                 'hrf_dilation_min', 'hrf_dilation_max', 'width', 'depth', 'kernel_size',
                 'signed_activations', 'learn_baseline', 'enable_spatial_spsf',
                 'spatial_matrix_path', 'spsf_fwhm_mm'):
        if getattr(config, name) != getattr(trained, name):
            raise ValueError(f'{name} differs from the trained checkpoint; use its recorded setting.')
    config.validate()
    data = DatasetCC2017(config, config.phase, average_repeats = config.average_repeats)
    if min(data.run_lengths) <= config.burnin_trs + config.delay_trs:
        raise ValueError('Stored delay leaves no valid activity rows within a run.')
    if data.voxels != saved['voxels']:
        raise ValueError('Voxel count differs from checkpoint.')
    model = SuperResolutionHead(config, data.voxels).to(device)
    model.load_state_dict(saved['model'], strict = True)
    model.eval().requires_grad_(False)
    del saved
    folder = create_run(config)
    save_json(folder / 'meta' / 'data_audit.json', data.audit())
    factor = config.tsr_factor
    repetitions = 1 if config.average_repeats else data.repetitions
    # Keep the full global index and mark unsupported boundary samples explicitly.
    total = torch.zeros(len(data.data), repetitions, factor, data.voxels)
    counts = torch.zeros(len(data.data), repetitions, dtype = torch.int32)
    loader = make_loader(data, config, False)
    started = monotonic()
    local = torch.arange(config.burnin_trs, config.window_trs - config.lookahead_trs)
    interior = slice(config.burnin_trs, config.window_trs - config.lookahead_trs)
    diagnostics = {'reblur_squared_error_sum': 0.0, 'elements': 0}
    with torch.inference_mode():
        for iteration, batch in enumerate(loader, start = 1):
            values = batch['bold'].to(device)
            # One direct inverse pass; reconstruction error is measured before overlap averaging
            estimates = model(values)
            if not torch.isfinite(estimates['reblurred']).all():
                raise FloatingPointError('Non-finite reconstructed BOLD.')
            diagnostics['reblur_squared_error_sum'] += (estimates['reblurred'][..., interior] - values[..., interior]).square().sum().item()
            diagnostics['elements'] += values[..., interior].numel()
            prediction = estimates['activation']
            prediction = prediction.reshape(len(values), data.voxels, config.window_trs, factor).permute(0, 2, 3, 1).cpu()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Non-finite inference output.')
            for sample in range(len(values)):
                run = int(batch['run'][sample])
                repetition = int(batch['repetition'][sample])
                start = int(batch['start'][sample])
                # Restore stimulus timestamps from the assumed pre-shifted BOLD storage
                stimulus = start + local + config.delay_trs
                valid = stimulus < data.run_lengths[run]
                rows = stimulus[valid] + data.offsets[run]
                total[rows, repetition] += prediction[sample, local[valid]]
                counts[rows, repetition] += 1
            if iteration % config.log_every == 0 or iteration == len(loader):
                progress('Inference windows', iteration, len(loader), started)
    mask = counts > 0
    total /= counts.clamp_min(1)[..., None, None]
    total[~mask] = float('nan')
    artefact = {'values': total,
                'valid': mask, 'overlap_counts': counts, 'run_lengths': data.run_lengths,
                'subject': config.subject, 'phase': config.phase,
                'tr_sec': config.tr_sec, 'subframe_dt_sec': config.tr_sec / factor,
                'stored_delay_sec': config.stored_delay_sec,
                'layout': '[stimulus_trial, repetition, subframe, voxel]',
                'timing': 'Rows refer to original GT clip indices. Invalid run edges contain NaN and must be excluded.',
                'checkpoint': str(Path(config.checkpoint).resolve()), 'config': config.to_dict()}
    path = folder / 'artifacts' / f'subj{config.subject:02d}_{config.phase}_activity.pt'
    atomic_save(artefact, path)
    summary = {'valid_trial_repetitions': int(mask.sum()), 'total_trial_repetitions': mask.numel(),
               'reblur_input_mse': diagnostics['reblur_squared_error_sum'] / diagnostics['elements'],
               'interpretation': 'Window-interior BOLD reconstruction MSE before overlap averaging; not activity ground-truth error.'}
    save_json(folder / 'meta' / 'metrics.json', summary)
    log(f'Saved enhanced signals and valid-boundary masks: {path}', 'OK')
    return path


if __name__ == '__main__':
    main(parse_config())
