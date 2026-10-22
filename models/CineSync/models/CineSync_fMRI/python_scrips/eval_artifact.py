# file: eval_artifact.py
# libraries: torch, numpy, av, transformers, scipy, scikit-image

# import libraries
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
import av
import numpy as np
import torch
from ....modules.runtime import configure_huggingface, setup_device, make_loader, log, progress_text
from ....modules.encoders.video import prepare_video
from ....modules.evaluation.metrics import pixel_metrics, n_way_accuracy, temporal_consistency, frechet_distance


def index_artifacts(run: Path) -> dict[int, tuple[int, Path]]:
    # pair by trial identity; repeated stimuli from different trials remain separate
    artifacts = {}
    for path in sorted((run / 'videos').glob('*.mp4')):
        match = re.fullmatch(r'trial_(\d+)_clip_(\d+)', path.stem)
        if match is None:
            raise ValueError(f"Unrecognised reconstruction filename: '{path.name}'.")
        trial_idx, clip_id = map(int, match.groups())
        if trial_idx in artifacts:
            raise ValueError(f'Duplicate reconstruction for trial {trial_idx}.')
        artifacts[trial_idx] = (clip_id, path)
    if not artifacts:
        raise ValueError(f"No generated MP4 clips found in '{run / 'videos'}'.")
    return artifacts


def load_video(path: Path) -> torch.Tensor:
    # score decoded saved pixels, without resizing or re-encoding the prediction
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format = 'rgb24') for frame in container.decode(video = 0)]
    if len(frames) != 33:
        raise ValueError(f"Expected 33 generated frames in '{path}', got {len(frames)}.")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255


def main() -> None:
    parser = argparse.ArgumentParser(description = 'Recalculate reconstruction metrics from saved CineSync videos.')
    parser.add_argument('--run-dir', required = True, help = 'Existing eval_recon run containing videos/ and meta/arguments.json.')
    parser.add_argument('--cache-dir', default = './data/cache/huggingface')
    parser.add_argument('--hf-endpoint', default = None)
    parser.add_argument('--local-files-only', action = 'store_true')
    parser.add_argument('--device', default = 'cuda')
    parser.add_argument('--workers', type = int, default = 4)
    parser.add_argument('--batch-size', type = int, default = 1)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--semantic-metrics', action = argparse.BooleanOptionalAction, default = True)
    parser.add_argument('--clip-model', default = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    parser.add_argument('--dino-model', default = 'facebook/dinov2-base')
    parser.add_argument('--image-metric-model', default = 'google/vit-base-patch16-224')
    parser.add_argument('--video-metric-model', default = 'MCG-NJU/videomae-base-finetuned-kinetics')
    parser.add_argument('--i3d-torchscript', default = None, help = 'Feature-only I3D TorchScript; required to report FVD.')
    args = parser.parse_args()
    configure_huggingface(args)
    device, _, world = setup_device(args)
    if world != 1:
        raise ValueError('Evaluation uses one process.')
    run = Path(args.run_dir)
    artifacts = index_artifacts(run)

    # recover GT selection from the original run; filenames select the available trials
    previous = json.loads((run / 'meta' / 'arguments.json').read_text(encoding = 'utf-8'))
    selection = {key: previous[key] for key in ('db_path', 'captions_path', 'subject_id', 'phase', 'fmri_roi')}
    loader_args = argparse.Namespace(**vars(args), **selection, max_samples = None)
    dataset, loader, _ = make_loader(loader_args, False, modalities = ('video',))
    try:
        missing = set(artifacts).difference(dataset.trial_indices)
        if missing:
            raise ValueError(f'Generated trials are absent from the saved dataset selection: {sorted(missing)}.')
        # preserve dataset order so sampled classification negatives match eval_recon
        dataset.trial_indices = [index for index in dataset.trial_indices if index in artifacts]
        extractor, i3d = None, None
        if args.semantic_metrics:
            from ....modules.evaluation.features import EvaluationFeatures
            extractor = EvaluationFeatures(args.clip_model, args.dino_model, args.video_metric_model,
                                           args.image_metric_model, args.cache_dir, args.local_files_only).to(device)
            log('N-way classification reports standard GT top-one and GT top-three scores.')
        if args.i3d_torchscript:
            from ....modules.evaluation.features import I3DFeatures
            i3d = I3DFeatures(args.i3d_torchscript).to(device)
        predicted, targets, pred_i3d, gt_i3d = [], [], [], []
        totals = {'psnr': 0.0, 'ssim': 0.0}
        count, started = 0, monotonic()
        log('Scoring saved MP4 pixels; compression may change scores from the original in-memory evaluation.')
        log(f'Artefact evaluation: trials {progress_text(count, len(dataset), started)}.')
        with torch.inference_mode():
            for batch in loader:
                generated = []
                for trial_idx, clip_id in zip(batch['trial_idx'].tolist(), batch['clip_id'].tolist()):
                    expected_clip, path = artifacts[trial_idx]
                    if clip_id != expected_clip:
                        raise ValueError(f'Clip ID mismatch for trial {trial_idx}: dataset {clip_id}, filename {expected_clip}.')
                    generated.append(load_video(path))
                generated = torch.stack(generated).to(device)
                # infer reconstruction size from the artefact; use the original GT preparation
                height, width = generated.shape[-2:]
                gt = (prepare_video(batch['data']['video'], height, width).to(device) + 1) / 2
                for key, value in pixel_metrics(generated, gt).items():
                    totals[key] += value * len(generated)
                if extractor is not None:
                    predicted.append(extractor(generated))
                    targets.append(extractor(gt))
                if i3d is not None:
                    pred_i3d.append(i3d(generated))
                    gt_i3d.append(i3d(gt))
                count += len(generated)
                log(f'Artefact evaluation: trials {progress_text(count, len(dataset), started)}.')

        # recompute all requested metrics; existing feature files and scores are left intact
        log('Calculating aggregate reconstruction metrics.')
        scores = {key: value / count for key, value in totals.items()}
        if extractor is not None:
            p = {key: torch.cat([item[key] for item in predicted]) for key in predicted[0]}
            g = {key: torch.cat([item[key] for item in targets]) for key in targets[0]}
            # match eval_recon: class distractors, both GT acceptance rules, all selected trials
            for n in (2, 50):
                for gt_top_k, suffix in ((1, ''), (3, '_gt_top3')):
                    scores[f'video_{n}way{suffix}'] = n_way_accuracy(
                        p['video_logits'], g['video_logits'], n, seed = args.seed, gt_top_k = gt_top_k)
                    scores[f'frame_{n}way{suffix}'] = float(np.mean([
                        n_way_accuracy(p['image_logits'][:, frame], g['image_logits'][:, frame],
                                       n, seed = args.seed + frame, gt_top_k = gt_top_k)
                        for frame in range(p['image_logits'].shape[1])]))
            scores['dtc'] = temporal_consistency(p['dino'])
            scores['ctc'] = temporal_consistency(p['clip'])
        if i3d is not None:
            scores['fvd'] = frechet_distance(torch.cat(pred_i3d).numpy(), torch.cat(gt_i3d).numpy())
        else:
            log('FVD omitted: supply --i3d-torchscript with a validated feature extractor.', 'WARN')

        # keep timestamped scores and their settings beside the original metrics.json
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        destination = run / 'meta' / f'metrics_{stamp}.json'
        settings = {**vars(args), **selection, 'trial_indices': dataset.trial_indices, 'prediction_format': 'mp4'}
        (run / 'meta' / f'arguments_eval_artifact_{stamp}.json').write_text(
            json.dumps(settings, indent = 2), encoding = 'utf-8')
        destination.write_text(json.dumps(scores, indent = 2), encoding = 'utf-8')
        log(str(scores), 'OK')
        log(f"Metrics saved to '{destination}'.", 'OK')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
