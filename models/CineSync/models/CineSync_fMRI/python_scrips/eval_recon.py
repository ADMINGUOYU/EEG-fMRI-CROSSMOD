# file: eval_recon.py
# libraries: torch, diffusers, transformers, tensorboard, scipy, scikit-image, av
from __future__ import annotations
import argparse
import json
from time import monotonic
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from ....modules.runtime import (add_common_arguments, configure_huggingface, setup_device,
    make_loader, create_run, load_checkpoint, restore_model, log, progress_text)
from ....modules.encoders.fmri import FMRIConfig, FMRIEncoder
from ....modules.encoders.video import prepare_video, save_video
from ....modules.evaluation.metrics import pixel_metrics, n_way_accuracy, temporal_consistency, frechet_distance
from ..recon import CineSyncFMRIRecon


def main() -> None:
    parser = argparse.ArgumentParser(description = 'Reconstruct and evaluate standard four-second CineBrain trials.')
    add_common_arguments(parser)
    parser.set_defaults(phase = 'test', batch_size = 1)
    parser.add_argument('--checkpoint', required = True)
    parser.add_argument('--cogvideo-model', default = None, help = 'Optional offline path override for the saved base model.')
    parser.add_argument('--steps', type = int, default = 50)
    parser.add_argument('--guidance-scale', type = float, default = 6)
    parser.add_argument('--semantic-metrics', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--clip-model', default = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    parser.add_argument('--dino-model', default = 'facebook/dinov2-base')
    parser.add_argument('--image-metric-model', default = 'google/vit-base-patch16-224')
    parser.add_argument('--video-metric-model', default = 'MCG-NJU/videomae-base-finetuned-kinetics')
    parser.add_argument('--i3d-torchscript', default = None, help = 'Feature-only I3D TorchScript; required to report FVD.')
    parser.add_argument('--save-arrays', action = 'store_true', help = 'Save float reconstruction arrays for exact later evaluation.')
    args = parser.parse_args()
    configure_huggingface(args)
    from ....modules.third_party.cogvideo import CogVideoBackend
    device, rank, world = setup_device(args)
    if world != 1:
        raise ValueError('Evaluation uses one process.')
    saved = load_checkpoint(args.checkpoint)
    if saved['stage'] != 'recon':
        raise ValueError('Expected a reconstruction checkpoint.')
    previous = saved['arguments']
    args.fmri_roi = previous['fmri_roi']
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    backend = CogVideoBackend(args.cogvideo_model or previous['cogvideo_model'], args.cache_dir, args.local_files_only,
                             dtype, previous['lora_rank'], previous['lora_alpha'], gradient_checkpointing = False)
    model = CineSyncFMRIRecon(FMRIEncoder(FMRIConfig(**saved['encoder_config'])), backend).to(device)
    restore_model(model, saved)
    model.eval()
    extractor, i3d = None, None
    if args.semantic_metrics:
        from ....modules.evaluation.features import EvaluationFeatures
        extractor = EvaluationFeatures(args.clip_model, args.dino_model, args.video_metric_model,
                                       args.image_metric_model, args.cache_dir, args.local_files_only).to(device)
        log('N-way classification reports standard GT top-one scores and CineBrain issue #3 GT top-three scores.')
    if args.i3d_torchscript:
        from ....modules.evaluation.features import I3DFeatures
        i3d = I3DFeatures(args.i3d_torchscript).to(device)
    dataset, loader, _ = make_loader(args, False)
    run = create_run(args, 'eval_recon')
    (run / 'videos').mkdir()
    predicted, targets = [], []
    totals = {'psnr': 0.0, 'ssim': 0.0}
    count, ids = 0, []
    pred_i3d, gt_i3d = [], []
    generator = torch.Generator(device = device).manual_seed(args.seed)
    # estimate trial-processing time; final aggregate metrics are reported separately
    started = monotonic()
    log(f'Reconstruction evaluation: trials {progress_text(count, len(dataset), started)}.')
    try:
        with torch.inference_mode():
            for batch in loader:
                fmri = batch['data']['fmri'].to(device)
                generated = model.generate(fmri, height = previous['height'], width = previous['width'],
                                           steps = args.steps, guidance_scale = args.guidance_scale, generator = generator)
                # Diffusers returns B,T,C,H,W in [0,1]
                videos = batch['data']['video']
                gt = (prepare_video(videos, previous['height'], previous['width']).to(device) + 1) / 2
                scores = pixel_metrics(generated, gt)
                for key, value in scores.items():
                    totals[key] += value * len(fmri)
                if extractor:
                    predicted.append(extractor(generated))
                    targets.append(extractor(gt))
                if i3d:
                    pred_i3d.append(i3d(generated))
                    gt_i3d.append(i3d(gt))
                for i, trial_idx in enumerate(batch['trial_idx'].tolist()):
                    name = f'trial_{trial_idx:06d}_clip_{int(batch["clip_id"][i]):06d}'
                    save_video(str(run / 'videos' / f'{name}.mp4'), generated[i])
                    if args.save_arrays:
                        np.save(run / 'videos' / f'{name}.npy', generated[i].cpu().numpy())
                count += len(fmri)
                ids.extend(batch['clip_id'].tolist())
                log(f'Reconstruction evaluation: trials {progress_text(count, len(dataset), started)}.')
        log('Calculating aggregate reconstruction metrics.')
        scores = {key: value / count for key, value in totals.items()}
        if extractor:
            p = {key: torch.cat([item[key] for item in predicted]) for key in predicted[0]}
            g = {key: torch.cat([item[key] for item in targets]) for key in targets[0]}
            # classify each trial independently; negatives are classes, not other clips
            for n in (2, 50):
                # calculate both top-1 and top-3
                for gt_top_k, suffix in ((1, ''), (3, '_gt_top3')):
                    scores[f'video_{n}way{suffix}'] = n_way_accuracy(
                        p['video_logits'], g['video_logits'], n, seed = args.seed, gt_top_k = gt_top_k)
                    scores[f'frame_{n}way{suffix}'] = float(np.mean([
                        n_way_accuracy(p['image_logits'][:, frame], g['image_logits'][:, frame],
                                                      n, seed = args.seed + frame, gt_top_k = gt_top_k)
                        for frame in range(p['image_logits'].shape[1])]))
            scores['dtc'] = temporal_consistency(p['dino'])
            scores['ctc'] = temporal_consistency(p['clip'])
            torch.save({'prediction': p, 'target': g, 'clip_ids': ids}, run / 'meta' / 'features.pt')
        if i3d:
            scores['fvd'] = frechet_distance(torch.cat(pred_i3d).numpy(), torch.cat(gt_i3d).numpy())
        else:
            log('FVD omitted: supply --i3d-torchscript with a validated feature extractor.', 'WARN')
        (run / 'meta' / 'metrics.json').write_text(json.dumps(scores, indent = 2))
        with SummaryWriter(str(run / 'logs')) as writer:
            for key, value in scores.items():
                writer.add_scalar(key, value, saved['step'])
        log(str(scores), 'OK')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
