# file: eval_align.py
# libraries: torch, transformers, tensorboard
from __future__ import annotations
import argparse
import json
from time import monotonic
from functools import partial
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from ....modules.runtime import (add_common_arguments, configure_huggingface, setup_device,
    make_loader, create_run, load_checkpoint, restore_model, log, progress_text)
from ....modules.encoders.fmri import FMRIConfig
from ....modules.evaluation.metrics import retrieval_metrics
from ..align import CineSyncFMRIAlign


def main() -> None:
    parser = argparse.ArgumentParser(description = 'Evaluate fMRI/video/text alignment against the full selected gallery.')
    add_common_arguments(parser)
    parser.set_defaults(phase = 'test')
    parser.add_argument('--checkpoint', required = True)
    parser.add_argument('--siglip-model', default = None, help = 'Optional offline path override for the saved teacher.')
    args = parser.parse_args()
    configure_huggingface(args)
    device, rank, world = setup_device(args)
    if world != 1:
        raise ValueError('Evaluation uses one process to avoid duplicate gallery entries.')
    saved = load_checkpoint(args.checkpoint)
    if saved['stage'] != 'align':
        raise ValueError('Expected an alignment checkpoint.')
    args.fmri_roi = saved['arguments']['fmri_roi']
    model = CineSyncFMRIAlign(FMRIConfig(**saved['encoder_config']), saved['arguments']['normalise_contrastive']).to(device)
    restore_model(model, saved)
    model.eval()
    dataset, loader, _ = make_loader(args, False, modalities = ('fmri',))
    # reuse the selected gallery's embeddings before starting DataLoader workers
    cache_path = Path(args.results_root) / 'CineSync'
    if not dataset.load_cache_embeddings(cache_path):
        from ....modules.encoders.semantic import SemanticEncoder
        teacher = SemanticEncoder(args.siglip_model or saved['arguments']['siglip_model'],
                                  args.cache_dir, args.local_files_only).to(device)
        try:
            dataset.generate_cache_embeddings(cache_path, partial(teacher, device = device), args.batch_size)
        finally:
            del teacher
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        if not dataset.load_cache_embeddings(cache_path):
            raise RuntimeError('Cannot load the generated embedding cache.')
    run = create_run(args, 'eval_align', model_name = 'CineSync_fMRI')
    features = {key: [] for key in ('brain', 'video', 'text', 'clip_id')}
    count = 0
    # time selected-trial processing separately from cache preparation and final metrics
    started = monotonic()
    log(f'Alignment evaluation: trials {progress_text(count, len(dataset), started)}.')
    try:
        with torch.inference_mode():
            for batch in loader:
                visual = batch['embeddings']['visual'].to(device, non_blocking = True)
                textual = batch['embeddings']['textual'].to(device, non_blocking = True)
                output = model(batch['data']['fmri'].to(device), visual, textual)
                for key in ('brain', 'video', 'text'):
                    features[key].append(output[key].cpu())
                features['clip_id'].append(batch['clip_id'])
                count += len(batch['clip_id'])
                log(f'Alignment evaluation: trials {progress_text(count, len(dataset), started)}.')
        log('Calculating retrieval metrics for the complete gallery.')
        features = {key: torch.cat(value) for key, value in features.items()}
        scores = {}
        for key in ('video', 'text'):
            for name, value in retrieval_metrics(features['brain'], features[key], features['clip_id']).items():
                scores[f'brain_to_{key}/{name}'] = value
        torch.save(features, run / 'meta' / 'features.pt')
        (run / 'meta' / 'metrics.json').write_text(json.dumps(scores, indent = 2))
        with SummaryWriter(str(run / 'logs')) as writer:
            for key, value in scores.items():
                writer.add_scalar(key, value, saved['step'])
        log(str(scores), 'OK')
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
