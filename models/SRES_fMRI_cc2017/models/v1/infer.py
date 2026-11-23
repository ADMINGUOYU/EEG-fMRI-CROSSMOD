# file: infer.py
# Monte-Carlo de-noising, try different noise's results
# check 'mc_test_samples' ~

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import monotonic

import torch

from .config import Config, parse_config
from ...modules.dataset_cc2017 import DatasetCC2017
from ...modules.noise import ar1_noise
from ...modules.runtime import atomic_save, create_run, log, make_loader, progress, save_json, setup
from .network import SuperResolutionHead


def main(config: Config) -> Path:
    device = setup(config)
    if config.action != 'infer' or not config.checkpoint:
        raise ValueError('Inference requires ACTION=infer and CHECKPOINT.')
    saved = torch.load(config.checkpoint, map_location = 'cpu', weights_only = False)
    trained = Config(**saved['config'])
    config = replace(config, stage = trained.stage)
    if trained.subject != config.subject:
        raise ValueError('Inference subject does not match the trained model.')
    if config.mode == 'B' and trained.stage != 2:
        raise ValueError('Mode B requires a stage-2 checkpoint.')
    # Check architecture and physics against the checkpoint; allow a longer inference window
    for name in ('tr_sec', 'stored_delay_sec', 'tsr_factor', 'hrf_length_sec', 'hrf_mode',
                 'hrf_dilation_min', 'hrf_dilation_max', 'width', 'depth', 'kernel_size',
                 'mixing_rank', 'signed_activations', 'learn_baseline', 'enable_spatial_spsf',
                 'spatial_matrix_path', 'spsf_fwhm_mm', 'r2r_alpha', 'rho_ar1'):
        if getattr(config, name) != getattr(trained, name):
            raise ValueError(f'{name} differs from the trained checkpoint; use its recorded setting.')
    config.validate()
    data = DatasetCC2017(config, config.phase, average_repeats = config.average_repeats)
    if data.voxels != saved['voxels']:
        raise ValueError('Voxel count differs from checkpoint.')
    model = SuperResolutionHead(config, data.voxels).to(device)
    model.load_state_dict(saved['model'], strict = True)
    model.eval().requires_grad_(False)
    calibration = saved['calibration']
    sigma, rho = calibration['sigma'].to(device), calibration['rho'].to(device)
    if config.average_repeats:
        # Repeated scanner noise is assumed independent; correlated repeats need another model.
        sigma = sigma / data.repetitions ** 0.5
    del saved
    folder = create_run(config)
    save_json(folder / 'meta' / 'data_audit.json', data.audit())
    factor = config.tsr_factor if config.mode == 'B' else 1
    repetitions = 1 if config.average_repeats else data.repetitions
    # Keep the full global index and mark unsupported boundary samples explicitly.
    total = torch.zeros(len(data.data), repetitions, factor, data.voxels)
    counts = torch.zeros(len(data.data), repetitions, dtype = torch.int32)
    loader = make_loader(data, config, False)
    started = monotonic()
    interior = slice(config.burnin_trs, config.window_trs - config.lookahead_trs)
    diagnostics = {'squared_change_sum': 0.0, 'reblur_squared_error_sum': 0.0, 'elements': 0}
    with torch.inference_mode():
        for iteration, batch in enumerate(loader, start = 1):
            values = batch['bold'].to(device)
            estimates = {}
            for draw in range(config.mc_test_samples):
                noisy = values + config.r2r_alpha * ar1_noise(tuple(values.shape), sigma, rho)
                result = model(noisy, stage = 1 if config.mode == 'A' else 2)
                for name, prediction in result.items():
                    estimates[name] = estimates.get(name, 0) + prediction / config.mc_test_samples
            diagnostics['squared_change_sum'] += (estimates['bold'][..., interior] - values[..., interior]).square().sum().item()
            diagnostics['elements'] += values[..., interior].numel()
            if config.mode == 'B':
                diagnostics['reblur_squared_error_sum'] += (estimates['reblurred'][..., interior] - values[..., interior]).square().sum().item()
            prediction = estimates['activation'] if config.mode == 'B' else estimates['bold']
            prediction = prediction.reshape(len(values), data.voxels, config.window_trs, factor).permute(0, 2, 3, 1).cpu()
            if not torch.isfinite(prediction).all():
                raise FloatingPointError('Non-finite inference output.')
            for sample in range(len(values)):
                run = int(batch['run'][sample])
                repetition = int(batch['repetition'][sample])
                start = int(batch['start'][sample])
                local = torch.arange(config.burnin_trs, config.window_trs - config.lookahead_trs)
                # Mode A preserves the configured stored delay; mode B assigns Z at index i to stimulus i + delay_trs
                stimulus = start + local + (config.delay_trs if config.mode == 'B' else 0)
                valid = stimulus < data.run_lengths[run]
                rows = stimulus[valid] + data.offsets[run]
                total[rows, repetition] += prediction[sample, local[valid]]
                counts[rows, repetition] += 1
            if iteration % config.log_every == 0 or iteration == len(loader):
                progress('Inference windows', iteration, len(loader), started)
    mask = counts > 0
    total /= counts.clamp_min(1)[..., None, None]
    total[~mask] = float('nan')
    artefact = {'values': total.squeeze(2) if config.mode == 'A' else total,
                'valid': mask, 'overlap_counts': counts, 'run_lengths': data.run_lengths,
                'mode': config.mode, 'subject': config.subject, 'phase': config.phase,
                'tr_sec': config.tr_sec, 'subframe_dt_sec': config.tr_sec / factor,
                'stored_delay_sec': config.stored_delay_sec,
                'layout': '[stimulus_trial, repetition, voxel]' if config.mode == 'A' else '[stimulus_trial, repetition, subframe, voxel]',
                'timing': 'Rows refer to original GT clip indices. Invalid run edges contain NaN and must be excluded.',
                'checkpoint': str(Path(config.checkpoint).resolve()), 'config': config.to_dict()}
    path = folder / 'artifacts' / f'subj0{config.subject}_{config.phase}_mode{config.mode}.pt'
    atomic_save(artefact, path)
    summary = {'valid_trial_repetitions': int(mask.sum()), 'total_trial_repetitions': mask.numel(),
               'denoising_change_mse': diagnostics['squared_change_sum'] / diagnostics['elements'],
               'reblur_input_mse': diagnostics['reblur_squared_error_sum'] / diagnostics['elements'] if config.mode == 'B' else None,
               'interpretation': 'Input consistency diagnostics, not clean-signal reconstruction accuracy.'}
    save_json(folder / 'meta' / 'metrics.json', summary)
    log(f'Saved enhanced signals and valid-boundary masks: {path}', 'OK')
    return path


if __name__ == '__main__':
    main(parse_config())
