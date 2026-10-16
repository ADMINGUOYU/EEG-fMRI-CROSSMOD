# file: eval_align.py
# libraries: torch, transformers, tensorboard
from __future__ import annotations
import argparse
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from ....modules.runtime import (add_common_arguments, configure_huggingface, setup_device,
    make_loader, create_run, load_checkpoint, restore_model, log)
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
    from ....modules.encoders.semantic import SemanticEncoder
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
    teacher = SemanticEncoder(args.siglip_model or saved['arguments']['siglip_model'], args.cache_dir, args.local_files_only).to(device)
    dataset, loader, _ = make_loader(args, False)
    run = create_run(args, 'eval_align')
    features = {key: [] for key in ('brain', 'video', 'text', 'clip_id')}
    try:
        with torch.inference_mode():
            for batch in loader:
                visual, textual = teacher(batch['data']['video'], batch['caption'], device)
                output = model(batch['data']['fmri'].to(device), visual, textual)
                for key in ('brain', 'video', 'text'):
                    features[key].append(output[key].cpu())
                features['clip_id'].append(batch['clip_id'])
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
