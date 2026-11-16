# file: train.py
# Calibration + two-stage training script

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from ...modules.config import Config, parse_config
from ...modules.dataset_cc2017 import DatasetCC2017
from ...modules.noise import ar1_noise, calibrate, load_calibration, reciprocal_pair
from ...modules.physics import synthetic_shots
from ...modules.runtime import atomic_save, create_run, log, make_loader, progress, save_json, setup
from ...modules.scheduling import make_scheduler
from .network import SuperResolutionHead


# NOTE: this is the loss function
# TODO: might want to refactor out if needed
def losses(outputs: dict[str, torch.Tensor], target: torch.Tensor, config: Config) -> dict[str, torch.Tensor]:
    """
    Compare only the valid window interior and penalise temporal activity variation
    """
    valid = slice(config.burnin_trs, config.window_trs - config.lookahead_trs)

    # Stage 1: mse de-noising loss
    den = F.mse_loss(outputs['bold'][..., valid], target[..., valid])
    if config.stage == 1:
        return {'total': den, 'denoising': den}

    # Stage 2: mse deconvolution loss + temporal total variation (tv) regularisation
    dec = F.mse_loss(outputs['reblurred'][..., valid], target[..., valid])
    activation = outputs['activation'][..., config.burnin_trs * config.tsr_factor:(config.window_trs - config.lookahead_trs) * config.tsr_factor]
    tv = activation.diff(dim = -1).abs().mean() / config.dt_sec  # subtracts neighbouring activity samples along time (smoothing)
    total = config.mu_loss_weight * den + (1 - config.mu_loss_weight) * dec + config.lambda_tv * tv
    return {'total': total, 'denoising': den, 'deconvolution': dec, 'temporal_tv': tv}


def pretrain(model: SuperResolutionHead, sigma: torch.Tensor, rho: torch.Tensor,
             config: Config, writer: SummaryWriter) -> None:
    """
    Initialise the temporal inverse on known block activity before real-data adaptation.
    """
    optimiser = torch.optim.AdamW(model.deconvolver.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    scheduler = make_scheduler(optimiser, max(config.synthetic_steps, 1), config.warmup_fraction, config.minimum_lr_ratio)
    started = monotonic()
    for step in range(config.synthetic_steps):
        count = model.voxels if config.enable_spatial_spsf else min(config.voxel_batch_size, model.voxels)
        ids = torch.randperm(model.voxels, device = sigma.device)[:count]
        if config.enable_spatial_spsf:
            ids = torch.arange(model.voxels, device = sigma.device)
        target = synthetic_shots(config.batch_size, count, config, sigma.device)
        # Synthesis has extra history, so the cropped target is not contaminated by zero padding.
        with torch.no_grad():
            clean = model.physics(target, canonical = True)
            history = target.shape[-1] - config.window_trs * config.tsr_factor
            clean = clean[..., history // config.tsr_factor:]
            target = target[..., history:]
            noisy = clean + ar1_noise(tuple(clean.shape), sigma[ids], rho[ids])
        prediction = model.deconvolver(noisy)
        if not config.signed_activations:
            prediction = F.relu(prediction)
        valid = slice(config.burnin_trs * config.tsr_factor, (config.window_trs - config.lookahead_trs) * config.tsr_factor)
        loss = F.mse_loss(prediction[..., valid], target[..., valid])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite synthetic loss.')
        optimiser.zero_grad(set_to_none = True)
        loss.backward()
        if config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.deconvolver.parameters(), config.max_grad_norm, error_if_nonfinite = True)
        optimiser.step()
        scheduler.step()
        writer.add_scalar('synthetic/activation_mse', loss.item(), step)
        if (step + 1) % config.log_every == 0 or step + 1 == config.synthetic_steps:
            progress('Synthetic pre-training', step + 1, config.synthetic_steps, started)


def main(config: Config) -> None:
    device = setup(config)
    if config.action not in ('calibrate', 'train'):
        raise ValueError('Use the infer module for ACTION=infer.')
    training = DatasetCC2017(config, 'train', 'training')
    folder = create_run(config)
    save_json(folder / 'meta' / 'data_audit.json', training.audit())
    calibration = calibrate(training, config) if config.action == 'calibrate' else load_calibration(training, config)
    save_json(folder / 'meta' / 'calibration.json', {key: value for key, value in calibration.items() if not isinstance(value, torch.Tensor)})
    if config.action == 'calibrate':
        return
    validation = DatasetCC2017(config, 'train', 'validation', source = training.data)
    train_loader = make_loader(training, config, True)
    validation_loader = make_loader(validation, config, False)
    model = SuperResolutionHead(config, training.voxels).to(device)
    sigma, rho = calibration['sigma'].to(device), calibration['rho'].to(device)
    if config.stage == 1:
        # Stage 1 learns only the denoising branch and optional voxel mixing.
        model.deconvolver.requires_grad_(False)
        model.physics.requires_grad_(False)
        model.baseline.requires_grad_(False)
    elif not config.resume:
        if not config.stage1_checkpoint:
            raise ValueError('Stage 2 requires STAGE1_CHECKPOINT or RESUME.')
        previous = torch.load(config.stage1_checkpoint, map_location = 'cpu', weights_only = False)
        if previous['config']['stage'] != 1 or previous['voxels'] != training.voxels or previous['config']['subject'] != config.subject:
            raise ValueError('Stage 1 checkpoint belongs to a different stage, subject or voxel layout.')
        # Copy only the denoiser and optional mixing; stage-2 physics/upsampling follow this run.
        for name in ('denoiser', 'mixing'):
            component = getattr(model, name)
            saved = {key[len(name) + 1:]: value for key, value in previous['model'].items() if key.startswith(name + '.')}
            if component is not None:
                component.load_state_dict(saved, strict = True)
            elif saved:
                raise ValueError('Stage 1 voxel mixing must match the stage 2 architecture.')
        del previous
    if config.stage == 2 and config.freeze_denoiser:
        model.denoiser.requires_grad_(False)
        if model.mixing is not None:
            model.mixing.requires_grad_(False)
    optimiser = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr = config.lr, weight_decay = config.weight_decay)
    scheduler = make_scheduler(optimiser, config.epochs * len(train_loader), config.warmup_fraction, config.minimum_lr_ratio)
    start_epoch, global_step, best = 0, 0, float('inf')
    if config.resume:
        saved = torch.load(config.resume, map_location = 'cpu', weights_only = False)
        # Resuming is exact: structural and training settings cannot change under an old scheduler.
        mutable = {'resume', 'launch_script', 'results_root', 'device', 'workers', 'log_every', 'save_every', 'checkpoint'}
        changed = [key for key, value in config.to_dict().items() if key not in mutable and saved['config'][key] != value]
        if changed or saved['calibration']['provenance'] != calibration['provenance']:
            raise ValueError(f'Resume settings/source differ: {changed}')
        model.load_state_dict(saved['model'], strict = True)
        optimiser.load_state_dict(saved['optimiser'])
        scheduler.load_state_dict(saved['scheduler'])
        start_epoch, global_step, best = saved['epoch'], saved['global_step'], saved['best']
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    if start_epoch >= config.epochs:
        raise ValueError('Checkpoint has already completed the configured epochs.')
    writer = SummaryWriter(str(folder / 'logs'))
    history: list[dict[str, Any]] = []
    try:
        if config.stage == 2 and config.pretrain_synthetic and not config.resume:
            pretrain(model, sigma, rho, config, writer)
        started = monotonic()
        total_updates = (config.epochs - start_epoch) * len(train_loader)
        completed = 0
        for epoch in range(start_epoch, config.epochs):
            epoch_result: dict[str, Any] = {'epoch': epoch + 1}
            for label, loader in (('train', train_loader), ('validation', validation_loader)):
                model.train(label == 'train')
                sums: dict[str, float] = {}
                samples = 0
                # Fixed validation corruption makes epoch comparisons reproducible.
                generator = torch.Generator(device = device).manual_seed(config.seed + 100000) if label == 'validation' else None
                for batch in loader:
                    count = training.voxels if config.mixing_rank or config.enable_spatial_spsf else min(config.voxel_batch_size, training.voxels)
                    ids = torch.arange(training.voxels, device = device) if count == training.voxels else torch.randperm(training.voxels, device = device, generator = generator)[:count]
                    values = batch['bold'].to(device)[:, ids]
                    inputs, target = reciprocal_pair(values, sigma[ids], rho[ids], config.r2r_alpha, generator)
                    with torch.set_grad_enabled(label == 'train'):
                        outputs = model(inputs, ids, config.stage)
                        terms = losses(outputs, target, config)
                        if not torch.isfinite(terms['total']):
                            raise FloatingPointError('Non-finite training/validation loss.')
                        if label == 'train':
                            optimiser.zero_grad(set_to_none = True)
                            terms['total'].backward()
                            if config.max_grad_norm:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite = True)
                            optimiser.step()
                            scheduler.step()
                            global_step += 1
                            completed += 1
                            if global_step % config.log_every == 0:
                                progress(f'Epoch {epoch + 1}: loss = {terms["total"].item():.5g}', completed, total_updates, started)
                                writer.add_scalar('train/lr', optimiser.param_groups[0]['lr'], global_step)
                    for name, value in terms.items():
                        sums[name] = sums.get(name, 0.0) + value.detach().item() * len(values)
                    samples += len(values)
                epoch_result[label] = {name: value / samples for name, value in sums.items()}
                for name, value in epoch_result[label].items():
                    writer.add_scalar(f'{label}/{name}', value, epoch + 1)
            improved = epoch_result['validation']['total'] < best
            best = min(best, epoch_result['validation']['total'])
            payload = {'config': config.to_dict(), 'voxels': training.voxels, 'model': model.state_dict(),
                       'optimiser': optimiser.state_dict(), 'scheduler': scheduler.state_dict(),
                       'calibration': calibration, 'epoch': epoch + 1, 'global_step': global_step, 'best': best,
                       'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
            atomic_save(payload, folder / 'checkpoints' / 'last.pt')
            if improved:
                atomic_save(payload, folder / 'checkpoints' / 'best.pt')
            if (epoch + 1) % config.save_every == 0:
                atomic_save(payload, folder / 'checkpoints' / f'epoch_{epoch + 1}.pt')
            history.append(epoch_result)
            save_json(folder / 'meta' / 'metrics.json', history)
            log(f'Epoch {epoch + 1}: train = {epoch_result["train"]["total"]:.5g}, validation = {epoch_result["validation"]["total"]:.5g}.')
    finally:
        writer.close()
    log(f'Training complete: {folder / "checkpoints" / "last.pt"}', 'OK')


if __name__ == '__main__':
    main(parse_config())
