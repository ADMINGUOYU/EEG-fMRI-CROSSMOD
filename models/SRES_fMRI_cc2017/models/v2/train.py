# file: train.py
# Direct BOLD reconstruction through HRF modelling and temporal regularisation

from __future__ import annotations

from time import monotonic
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from ...modules.dataset_cc2017 import DatasetCC2017
from ...modules.physics import synthetic_shots
from ...modules.scheduling import make_scheduler
from .config import Config, parse_config
from .network import SuperResolutionHead
from ...modules.runtime import atomic_save, create_run, log, make_loader, progress, save_json, setup


def losses(outputs: dict[str, torch.Tensor], target: torch.Tensor, config: Config) -> dict[str, torch.Tensor]:
    """
    Reconstruct measured BOLD over the valid interior and regularise latent activity
    """
    valid = slice(config.burnin_trs, config.window_trs - config.lookahead_trs)
    # HRF convolution, optional sPSF and baseline jointly form the observation prediction
    dec = F.mse_loss(outputs['reblurred'][..., valid], target[..., valid])
    activation = outputs['activation'][..., config.burnin_trs * config.tsr_factor:(config.window_trs - config.lookahead_trs) * config.tsr_factor]
    tv = activation.diff(dim = -1).abs().mean() / config.dt_sec
    return {'total': dec + config.lambda_tv * tv, 'deconvolution': dec, 'temporal_tv': tv}


def pretrain(model: SuperResolutionHead, config: Config, writer: SummaryWriter) -> None:
    """
    Initialise the temporal inverse on known block activity before real-data adaptation.
    """
    optimiser = torch.optim.AdamW(model.deconvolver.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    scheduler = make_scheduler(optimiser, max(config.synthetic_steps, 1), config.warmup_fraction, config.minimum_lr_ratio)
    device = next(model.parameters()).device
    started = monotonic()
    for step in range(config.synthetic_steps):
        count = model.voxels if config.enable_spatial_spsf else min(config.voxel_batch_size, model.voxels)
        target = synthetic_shots(config.batch_size, count, config, device)
        # Synthesis has extra history, so the cropped target is not contaminated by zero padding.
        with torch.no_grad():
            clean = model.physics(target, canonical = True)
            history = target.shape[-1] - config.window_trs * config.tsr_factor
            clean = clean[..., history // config.tsr_factor:]
            target = target[..., history:]
        # Initialise the inverse using noise-free synthetic BOLD and its known activity
        prediction = model.activity(clean)
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
    if config.action != 'train':
        raise ValueError('Use the infer module for ACTION=infer.')
    training = DatasetCC2017(config, 'train', 'training')
    folder = create_run(config)
    save_json(folder / 'meta' / 'data_audit.json', training.audit())
    validation = DatasetCC2017(config, 'train', 'validation', source = training.data)
    train_loader = make_loader(training, config, True)
    validation_loader = make_loader(validation, config, False)
    model = SuperResolutionHead(config, training.voxels).to(device)
    optimiser = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr = config.lr, weight_decay = config.weight_decay)
    scheduler = make_scheduler(optimiser, config.epochs * len(train_loader), config.warmup_fraction, config.minimum_lr_ratio)
    start_epoch, global_step, best = 0, 0, float('inf')
    if config.resume:
        saved = torch.load(config.resume, map_location = 'cpu', weights_only = False)
        # Resuming is exact: structural and training settings cannot change under an old scheduler.
        mutable = {'resume', 'launch_script', 'results_root', 'device', 'workers', 'log_every', 'save_every', 'checkpoint'}
        saved_config = Config(**saved['config']).to_dict()
        changed = [key for key, value in config.to_dict().items() if key not in mutable and saved_config[key] != value]
        if changed or saved['provenance'] != training.provenance():
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
        if config.pretrain_synthetic and not config.resume:
            pretrain(model, config, writer)
        started = monotonic()
        total_updates = (config.epochs - start_epoch) * len(train_loader)
        completed = 0
        for epoch in range(start_epoch, config.epochs):
            epoch_result: dict[str, Any] = {'epoch': epoch + 1}
            for label, loader in (('train', train_loader), ('validation', validation_loader)):
                model.train(label == 'train')
                sums: dict[str, float] = {}
                samples = 0
                # Use the same validation voxel selection each epoch for comparable losses
                generator = torch.Generator(device = device).manual_seed(config.seed + 100000) if label == 'validation' else None
                for batch in loader:
                    count = training.voxels if config.enable_spatial_spsf else min(config.voxel_batch_size, training.voxels)
                    ids = torch.arange(training.voxels, device = device) if count == training.voxels else torch.randperm(training.voxels, device = device, generator = generator)[:count]
                    values = batch['bold'].to(device)[:, ids]
                    with torch.set_grad_enabled(label == 'train'):
                        outputs = model(values, ids)
                        terms = losses(outputs, values, config)
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
                       'provenance': training.provenance(), 'epoch': epoch + 1, 'global_step': global_step, 'best': best,
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
