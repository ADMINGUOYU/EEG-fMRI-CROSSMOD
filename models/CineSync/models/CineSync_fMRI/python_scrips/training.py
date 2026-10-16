# file: training.py
# libraries: torch, numpy, tensorboard; stage one uses transformers, stage two diffusers
from __future__ import annotations
import argparse
import json
import math
from contextlib import nullcontext
from dataclasses import asdict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from ....modules.runtime import (add_common_arguments, configure_huggingface, setup_device,
    create_run, make_loader, log, atomic_checkpoint, model_state, restore_model, load_checkpoint)
from ....modules.encoders.fmri import FMRIConfig
from ....modules.scheduling import CineBrainLR
from ..align import CineSyncFMRIAlign


def parser_for(stage: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description = f'Train CineSync-fMRI {stage}.')
    add_common_arguments(parser)
    parser.set_defaults(batch_size = 16 if stage == 'align' else 2)
    parser.add_argument('--epochs', type = int, default = 50 if stage == 'align' else 100000)
    parser.add_argument('--max-steps', type = int, default = None if stage == 'align' else 5000)
    parser.add_argument('--lr', type = float, default = 1e-4)
    parser.add_argument('--lr-decay-style', choices = ['constant', 'linear', 'cosine'], default = 'linear')
    parser.add_argument('--lr-decay-steps', type = int, default = None)
    parser.add_argument('--lr-warmup', type = float, default = 0.01)
    parser.add_argument('--lr-decay-ratio', type = float, default = 0.1)
    parser.add_argument('--lr-auto-warmup-steps', type = int, default = 100)
    parser.add_argument('--lr-auto-warmup-rate', type = float, default = 0.05)
    parser.add_argument('--weight-decay', type = float, default = 0.01)
    parser.add_argument('--gradient-accumulation', type = int, default = 1)
    parser.add_argument('--max-grad-norm', type = float, default = 1.0)
    parser.add_argument('--save-every', type = int, default = 500)
    parser.add_argument('--precision', choices = ['fp32', 'bf16'], default = 'bf16')
    parser.add_argument('--resume', default = None)
    if stage == 'align':
        parser.add_argument('--siglip-model', default = 'google/siglip2-so400m-patch14-384')
        parser.add_argument('--teacher-frame-batch-size', type = int, default = 16)
        parser.add_argument('--normalise-contrastive', action = 'store_true')
        parser.add_argument('--encoder-width', type = int, default = 2048)
        parser.add_argument('--encoder-layers', type = int, default = 12)
        parser.add_argument('--encoder-heads', type = int, default = 32)
    else:
        parser.add_argument('--align-checkpoint', default = None)
        parser.add_argument('--cogvideo-model', default = 'THUDM/CogVideoX-5b')
        parser.add_argument('--lora-rank', type = int, default = 64)
        parser.add_argument('--lora-alpha', type = float, default = 64)
        parser.add_argument('--condition-dropout', type = float, default = 0.1)
        parser.add_argument('--height', type = int, default = 480)
        parser.add_argument('--width', type = int, default = 720)
    return parser


def main(stage: str) -> None:
    args = parser_for(stage).parse_args()
    configure_huggingface(args)
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.save_every) < 1:
        raise ValueError('Epochs, batch size, accumulation and checkpoint interval must be positive.')
    device, rank, world = setup_device(args)
    if stage == 'align' and world > 1:
        raise ValueError('Alignment uses the paper\'s single-GPU batch; launch without torchrun.')
    if stage == 'align' and args.batch_size < 2:
        raise ValueError('Contrastive alignment needs at least two samples per batch.')
    if args.phase == 'test':
        raise ValueError('Training on the test split is not allowed.')
    resumed = load_checkpoint(args.resume) if args.resume else None
    if resumed and resumed['stage'] != stage:
        raise ValueError('Resume checkpoint belongs to a different stage.')
    if resumed:
        # retain the saved schedule when continuing an existing run
        for key in ('lr', 'lr_decay_style', 'lr_decay_steps', 'lr_warmup', 'lr_decay_ratio',
                    'lr_auto_warmup_steps', 'lr_auto_warmup_rate'):
            setattr(args, key, resumed['arguments'][key])
        config = FMRIConfig(**resumed['encoder_config'])
        # architecture and teacher/base settings must come from the saved run
        for key in ('siglip_model', 'normalise_contrastive', 'cogvideo_model', 'lora_rank',
                    'lora_alpha', 'condition_dropout', 'height', 'width', 'fmri_roi', 'subject_id'):
            if key in resumed['arguments']:
                setattr(args, key, resumed['arguments'][key])
    elif stage == 'align':
        config = FMRIConfig(voxels = {'visual': 8405, 'auditory': 10541, 'both': 18946}[args.fmri_roi],
                            width = args.encoder_width, layers = args.encoder_layers, heads = args.encoder_heads)
    else:
        if not args.align_checkpoint:
            raise ValueError('--align-checkpoint is required unless resuming reconstruction.')
        aligned = load_checkpoint(args.align_checkpoint)
        if aligned['stage'] != 'align':
            raise ValueError('Stage two requires a stage-one alignment checkpoint.')
        config = FMRIConfig(**aligned['encoder_config'])
        args.fmri_roi = aligned['arguments']['fmri_roi']
        args.subject_id = aligned['arguments']['subject_id']
        if rank == 0:
            log(f"Stage two inherits subject {args.subject_id} and ROI '{args.fmri_roi}' from alignment.")
    aligned_model = CineSyncFMRIAlign(config, normalise = getattr(args, 'normalise_contrastive', False))
    teacher = None
    if stage == 'align':
        from ....modules.encoders.semantic import SemanticEncoder
        model = aligned_model
        teacher = SemanticEncoder(args.siglip_model, args.cache_dir, args.local_files_only,
                                  args.teacher_frame_batch_size).to(device)
    else:
        # NOTE: reconstruction (stage 2)
        from ..recon import CineSyncFMRIRecon
        from ....modules.third_party.cogvideo import CogVideoBackend
        if not resumed:
            restore_model(aligned_model, aligned)
        dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float32
        backend = CogVideoBackend(args.cogvideo_model, args.cache_dir, args.local_files_only,
                                 dtype, args.lora_rank, args.lora_alpha)
        model = CineSyncFMRIRecon(aligned_model.encoder, backend, args.condition_dropout)
    model.to(device)
    if resumed:
        restore_model(model, resumed)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = args.lr,
                                 betas = (0.9, 0.95), weight_decay = args.weight_decay)
    dataset, loader, sampler = make_loader(args, True, rank, world)
    # schedule optimiser updates, accounting for the final partial accumulation group
    total_steps = args.epochs * math.ceil(len(loader) / args.gradient_accumulation)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    if args.lr_decay_steps is None:
        args.lr_decay_steps = total_steps
    scheduler = CineBrainLR(optimiser, args.lr_decay_steps, args.lr_decay_style, args.lr_warmup,
                            args.lr_decay_ratio, args.lr_auto_warmup_steps, args.lr_auto_warmup_rate)
    step, first_epoch = 0, 0
    if resumed:
        optimiser.load_state_dict(resumed['optimiser'])
        step, first_epoch = resumed['step'], resumed['epoch']
        scheduler.load_state_dict(resumed['scheduler'])
        if rank == 0:
            log('Restoring weights and optimiser; the saved epoch restarts with a fresh data order.', 'WARN')
    if rank == 0:
        log(f"Learning-rate schedule: {args.lr_decay_style}, {args.lr_decay_steps} updates, warm-up {args.lr_warmup:.1%}.")
    run = create_run(args, stage, rank, world)
    writer = SummaryWriter(str(run / 'logs')) if rank == 0 else None
    if rank == 0:
        (run / 'meta' / 'model.json').write_text(json.dumps({'stage': stage, 'encoder': asdict(config),
            'backend': 'diffusers-cogvideox-v1.0', 'source_alignment': getattr(args, 'align_checkpoint', None)}, indent = 2))
    wrapped = DistributedDataParallel(model, device_ids = [device.index] if device.type == 'cuda' else None) if world > 1 else model
    model.train()
    optimiser.zero_grad(set_to_none = True)
    last_epoch = first_epoch
    try:
        for epoch in range(first_epoch, args.epochs):
            last_epoch = epoch
            if sampler:
                sampler.set_epoch(epoch)
            for batch_index, batch in enumerate(loader):
                if args.max_steps is not None and step >= args.max_steps:
                    break
                fmri = batch['data']['fmri'].to(device, non_blocking = True)
                if stage == 'align':
                    with torch.no_grad():
                        video_features, text_features = teacher(batch['data']['video'], batch['caption'], device)
                else:
                    # NOTE: reconstruction (stage 2)
                    from ....modules.encoders.video import prepare_video
                    video = prepare_video(batch['data']['video'], args.height, args.width).to(device)
                # accumulate gradients, including the final shorter accumulation group
                group_start = batch_index // args.gradient_accumulation * args.gradient_accumulation
                group_size = min(args.gradient_accumulation, len(loader) - group_start)
                synchronise = (batch_index + 1) % args.gradient_accumulation == 0 or batch_index + 1 == len(loader)
                context = wrapped.no_sync() if world > 1 and not synchronise else nullcontext()
                with context, torch.autocast(device.type, dtype = torch.bfloat16, enabled = args.precision == 'bf16'):
                    output = wrapped(fmri, video_features, text_features) if stage == 'align' else wrapped(fmri, video)
                    loss = output['loss'] / group_size
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss; checkpoint not overwritten.')
                    loss.backward()
                if synchronise:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite = True)
                    learning_rate = optimiser.param_groups[0]['lr']
                    optimiser.step()
                    scheduler.step()
                    optimiser.zero_grad(set_to_none = True)
                    step += 1
                    if rank == 0:
                        writer.add_scalar('train/loss', output['loss'].item(), step)
                        writer.add_scalar('train/lr', learning_rate, step)
                        if stage == 'align':
                            for key in ('visual_loss', 'text_loss'):
                                writer.add_scalar(f'train/{key}', output[key].item(), step)
                        if step == 1 or step % 10 == 0:
                            log(f'{stage}: epoch {epoch + 1}, step {step}, loss {output["loss"].item():.6f}.')
                        if step % args.save_every == 0:
                            atomic_checkpoint(run / 'checkpoints' / f'step_{step:07d}.pt', {
                                'stage': stage, 'encoder_config': asdict(config), 'arguments': vars(args),
                                'model': model_state(model, stage), 'optimiser': optimiser.state_dict(), 'scheduler': scheduler.state_dict(),
                                'step': step, 'epoch': epoch})
            if args.max_steps is not None and step >= args.max_steps:
                break
        if rank == 0:
            atomic_checkpoint(run / 'checkpoints' / 'last.pt', {'stage': stage, 'encoder_config': asdict(config),
                'arguments': vars(args), 'model': model_state(model, stage), 'optimiser': optimiser.state_dict(), 'scheduler': scheduler.state_dict(),
                'step': step, 'epoch': last_epoch + 1})
            log(f'{stage} complete at step {step}.', 'OK')
    finally:
        dataset.close()
        if writer:
            writer.close()
        if world > 1:
            dist.destroy_process_group()
