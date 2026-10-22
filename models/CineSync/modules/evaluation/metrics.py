# file: metrics.py
# libraries: numpy, torch, scipy, scikit-image
# NOTE: Explicit metric definitions: the release's models.eval_metrics helper is absent.
# NOTE: updated to reference CineBrain issue #3
from __future__ import annotations
import numpy as np
import torch
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity
from torch.nn import functional as F


def pixel_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    # compare decoded RGB frames in [0,1], before output MP4 compression
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError('Pixel metrics require matching B,T,C,H,W videos.')
    p = prediction.detach().cpu().float().clamp(0, 1).numpy()
    t = target.detach().cpu().float().clamp(0, 1).numpy()
    mse = ((p - t) ** 2).mean(axis = (2, 3, 4))
    psnr = -10 * np.log10(np.maximum(mse, 1e-12))
    scores = [structural_similarity(a.transpose(1, 2, 0), b.transpose(1, 2, 0),
                                    channel_axis = -1, data_range = 1.0)
              for a, b in zip(p.reshape(-1, *p.shape[2:]), t.reshape(-1, *t.shape[2:]))]
    return {'psnr': float(psnr.mean()), 'ssim': float(np.mean(scores))}


def retrieval_metrics(prediction: torch.Tensor, target: torch.Tensor, clip_ids: torch.Tensor,
                      chunk_size: int = 256) -> dict[str, float]:
    # exact full-gallery cosine retrieval; shared stimuli count as positives
    target = F.normalize(target.float(), dim = -1)
    total, top1, top5 = 0, 0, 0
    for start in range(0, len(prediction), chunk_size):
        logits = F.normalize(prediction[start:start + chunk_size].float(), dim = -1) @ target.T
        indices = logits.topk(min(5, len(target)), dim = -1).indices
        matches = clip_ids[indices] == clip_ids[start:start + chunk_size, None]
        top1 += matches[:, 0].sum().item()
        top5 += matches.any(-1).sum().item()
        total += len(logits)
    return {'top1': top1 / total, 'top5': top5 / total}


def n_way_accuracy(prediction: torch.Tensor, target: torch.Tensor, n_way: int,
                   trials: int = 100, seed: int = 42, gt_top_k: int = 1) -> float:
    # BY DEFAULT: each query competes against n-1 distinct distractors from a unique-stimulus gallery
    if prediction.shape != target.shape or prediction.ndim != 2 or len(prediction) == 0:
        raise ValueError('Classification metrics require matching non-empty N,C logits.')
    if not 2 <= n_way <= prediction.shape[1] - gt_top_k + 1 or trials < 1:
        raise ValueError('N-way evaluation needs N-1 distractor classes outside the accepted GT classes.')
    # softmax and sorting reproduce the reference, including its treatment of tied scores
    probabilities = prediction.softmax(-1).cpu()
    positives = target.argsort(-1, descending = False)[:, -gt_top_k:].cpu()
    rng = np.random.RandomState(seed)
    correct = 0
    for pred, class_ids in zip(probabilities, positives):
        candidates = np.setdiff1d(np.arange(prediction.shape[1]), class_ids.numpy())
        for _ in range(trials):
            distractors = rng.choice(candidates, n_way - 1, replace = False)
            # reuse one sampled negative set for all acceptable classes
            for class_id in class_ids:
                scores = torch.cat((pred[class_id].unsqueeze(0), pred[distractors]))
                if scores.argsort(descending = False)[-1].item() == 0:
                    correct += 1
                    break
    return correct / (len(prediction) * trials)


def temporal_consistency(features: torch.Tensor) -> float:
    # adjacent-frame cosine similarity: use DINO features for DTC and CLIP for CTC
    if features.ndim != 3 or features.shape[1] < 2:
        raise ValueError('Temporal consistency requires N,T,D features with T >= 2.')
    features = F.normalize(features.float(), dim = -1)
    return float((features[:, 1:] * features[:, :-1]).sum(-1).mean())


def frechet_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    # FVD requires matched I3D feature extraction; arbitrary embeddings are not FVD
    if min(len(prediction), len(target)) < 2:
        raise ValueError('Fréchet distance needs at least two samples per distribution.')
    a, b = np.asarray(prediction, dtype = np.float64), np.asarray(target, dtype = np.float64)
    c1, c2 = np.cov(a, rowvar = False), np.cov(b, rowvar = False)
    root = sqrtm(c1 @ c2)
    if np.iscomplexobj(root):
        if np.abs(root.imag).max() > 1e-5:
            raise ValueError('Covariance square root has a substantial imaginary component.')
        root = root.real
    return float(max(0, np.square(a.mean(0) - b.mean(0)).sum() + np.trace(c1 + c2 - 2 * root)))
