# file: prior_hrf.py
# compare encoder and sampled prior representations with stimulus and haemodynamic references
# run from the project root with `python -m models.NEURONS.models.NEURONS.experiments.prior_hrf`
from __future__ import annotations

# [SETUP] ================================================================
DATA_ROOT = './data/cc2017'
SUBJECT = 1
CHECKPOINT_PATH = '../../brain_model_prior_last.pth'
OUTPUT_ROOT = './results/NEURONS/experiments'
CACHE_DIR = './data/cache/huggingface/hub'
DEVICE = 'cuda:0'
USE_MIXED_PRECISION = True         # Match evaluation's float16 inference on CUDA.
BATCH_SIZE = 8
HIDDEN_DIM = 4096
N_BLOCKS = 4
PRIOR_DEPTH = 6
PRIOR_TIMESTEPS = 100              # Match the existing evaluation sampler.

# Verify that the test tensors contain these runs in chronological order.
# Repeated presentations are averaged, never treated as consecutive time points.
RUN_LENGTHS = (240, 240, 240, 240, 240)
TR_SECONDS = 2.0
BOLD_DELAY_SECONDS = 4.0
REFERENCE_FRAME = 2                 # Same image target as decoupler training.
HRF_DURATION_SECONDS = 32.0
HRF_PEAK = 6.0                      # Gamma shapes; scale is one second.
HRF_UNDERSHOOT = 16.0
HRF_UNDERSHOOT_RATIO = 6.0
DISPLACEMENT_SECONDS = 60.0         # Later real volumes; wrapped samples are excluded.
LAG_TRS = tuple(range(-10, 11))     # Positive offset compares with a later stimulus.

# Subsample coordinates only for analysis, after full model inference.
FEATURE_COORDINATES = 4096          # Set to 256 * 1664 to retain every coordinate.
RSA_TRIALS = 128                    # Maximum trials per run, shared by all conditions.
CONTROL_SEEDS = (11, 22, 33)
ANALYSIS_SEED = 42
BOOTSTRAP_SAMPLES = 2000            # Resample runs after averaging seed results.
SAVE_FEATURES = False              # Optional sampled features, not full model tokens.
# =======================================================================

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import lfilter
from scipy.stats import gamma, spearmanr

from ....modules.runtime import log


def response_kernels() -> tuple[np.ndarray, np.ndarray]:
    # Integrate the HRF over TR bins: references are piecewise-constant clip features.
    edges = np.arange(0.0, HRF_DURATION_SECONDS + TR_SECONDS, TR_SECONDS)
    hrf = np.diff(gamma.cdf(edges, HRF_PEAK))
    hrf -= np.diff(gamma.cdf(edges, HRF_UNDERSHOOT)) / HRF_UNDERSHOOT_RATIO
    hrf /= hrf.sum()

    # A generic positive smoother matches the positive HRF lobe's temporal moments.
    times = (edges[:-1] + edges[1:]) / 2
    positive = np.maximum(hrf, 0)
    positive /= positive.sum()
    centre = np.sum(times * positive)
    width = np.sqrt(np.sum((times - centre) ** 2 * positive))
    smoother = np.exp(-0.5 * ((times - centre) / width) ** 2)
    smoother /= smoother.sum()
    return hrf, smoother


def make_control(real: np.ndarray, condition: str, seed: int,
                 hrf: np.ndarray, displacement: int) -> np.ndarray:
    # Keep controls reproducible and independent of the stimulus and prior noise.
    if condition == 'real':
        return real
    if condition == 'displaced_real':
        # Analysis excludes the final displaced samples; there is no circular wrap.
        shifted = np.empty_like(real)
        shifted[:-displacement] = real[displacement:]
        shifted[-displacement:] = real[-1]
        return shifted
    rng = np.random.default_rng(seed)
    warmup = len(hrf) if condition == 'synthetic_bold' else 0
    values = rng.standard_normal((len(real) + warmup, real.shape[1]))
    if condition == 'synthetic_bold':
        values = lfilter(hrf, [1.0], values, axis = 0)[warmup:]
    elif condition != 'gaussian_noise':
        raise ValueError(f'Unknown control: {condition}')

    # Match empirical voxel-wise mean and variance, including constant voxels.
    values -= values.mean(axis = 0)
    values /= np.maximum(values.std(axis = 0), 1e-12)
    values *= real.std(axis = 0)
    values += real.mean(axis = 0)
    return values.astype(np.float32)


def measure(features: np.ndarray, references: dict[str, np.ndarray],
            valid: np.ndarray, rsa_indices: np.ndarray) -> dict[str, Any]:
    # Remove the run mean to measure changing content rather than a common CLIP bias.
    centred = features - features[valid].mean(axis = 0)
    norm = np.linalg.norm(centred, axis = 1, keepdims = True)
    unit = centred / np.maximum(norm, 1e-12)
    upper = np.triu_indices(len(rsa_indices), k = 1)
    feature_rsm = (unit[rsa_indices] @ unit[rsa_indices].T)[upper]
    result: dict[str, Any] = {'rsa': {}, 'lag_cosine': {}}
    for name, reference in references.items():
        ref = reference - reference[valid].mean(axis = 0)
        ref /= np.maximum(np.linalg.norm(ref, axis = 1, keepdims = True), 1e-12)
        ref_rsm = (ref[rsa_indices] @ ref[rsa_indices].T)[upper]
        # Constant representations have undefined RSA; report null rather than zero.
        score = None
        if np.ptp(feature_rsm) > 1e-12 and np.ptp(ref_rsm) > 1e-12:
            score = float(spearmanr(feature_rsm, ref_rsm).statistic)
        result['rsa'][name] = score
        result['lag_cosine'][name] = [
            float(np.mean(np.sum(unit[valid] * ref[valid + lag], axis = 1)))
            for lag in LAG_TRS
        ]
    result['mean_coordinate_variance'] = float(features[valid].var(axis = 0).mean())
    result['lag1_temporal_cosine'] = float(np.mean(np.sum(unit[valid] * unit[valid + 1], axis = 1)))
    result['near_zero_centred_fraction'] = float(np.mean(norm[valid] < 1e-8))
    return result


if __name__ == '__main__':
    # Import the heavyweight models only when running the experiment.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from ..model_variants.BrainModel_neurons import BrainModel, RidgeRegression, PriorNetwork, BrainDiffusionPrior
    from ....modules.generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

    config = {key: value for key, value in globals().copy().items() if key.isupper()}
    if TR_SECONDS <= 0 or HRF_DURATION_SECONDS <= 0 or BOLD_DELAY_SECONDS < 0:
        raise ValueError('TR and HRF duration must be positive; BOLD delay must be non-negative.')
    if HRF_PEAK <= 0 or HRF_UNDERSHOOT <= HRF_PEAK or HRF_UNDERSHOOT_RATIO <= 1:
        raise ValueError('Use positive gamma shapes, a later undershoot and an undershoot ratio above one.')
    if not Path(CHECKPOINT_PATH).is_file():
        raise FileNotFoundError(f'Configure CHECKPOINT_PATH for an existing checkpoint: {CHECKPOINT_PATH}')
    if not 0 < FEATURE_COORDINATES <= 256 * 1664 or BATCH_SIZE < 1 or RSA_TRIALS < 3:
        raise ValueError('Check feature-coordinate count, batch size and RSA trial count.')
    if not CONTROL_SEEDS or len(set(CONTROL_SEEDS)) != len(CONTROL_SEEDS):
        raise ValueError('Supply distinct control seeds.')
    if not LAG_TRS or 0 not in LAG_TRS or any(not isinstance(lag, int) for lag in LAG_TRS):
        raise ValueError('LAG_TRS must contain integer offsets including zero.')
    if BOOTSTRAP_SAMPLES < 1 or any(n < 1 for n in RUN_LENGTHS) or not RUN_LENGTHS:
        raise ValueError('Supply positive run lengths and a positive bootstrap count.')
    delay = int(round(BOLD_DELAY_SECONDS / TR_SECONDS))
    displacement = int(round(DISPLACEMENT_SECONDS / TR_SECONDS))
    if not np.isclose(delay * TR_SECONDS, BOLD_DELAY_SECONDS) or not np.isclose(displacement * TR_SECONDS, DISPLACEMENT_SECONDS):
        raise ValueError('BOLD delay and displacement must be whole multiples of TR.')
    if DISPLACEMENT_SECONDS < HRF_DURATION_SECONDS or displacement < 1:
        raise ValueError('Use a displacement at least as long as the HRF support.')
    hrf, smoother = response_kernels()
    # The aligned reference at trial t is the convolution at acquisition time t + delay.
    lower = max(len(hrf) - 1 - delay, 0) + max(0, -min(LAG_TRS))
    upper_margin = max(delay + max(0, max(LAG_TRS)), displacement + 1)
    if any(n - upper_margin - lower < 3 for n in RUN_LENGTHS):
        raise ValueError('Runs are too short for the HRF, lag range and displacement margins.')

    device = torch.device(DEVICE)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; configure DEVICE explicitly for this machine.')
    torch.manual_seed(ANALYSIS_SEED)
    np.random.seed(ANALYSIS_SEED)
    random.seed(ANALYSIS_SEED)
    # cuBLAS/cuDNN are kept in their standard evaluation configuration.
    root = Path(DATA_ROOT)
    voxels = torch.load(root / f'subj0{SUBJECT}_test_fmri.pt', map_location = 'cpu', weights_only = False)
    voxels = torch.as_tensor(voxels).float()
    if voxels.ndim == 3:
        voxels = voxels.mean(dim = 1)
    if voxels.ndim != 2 or len(voxels) != sum(RUN_LENGTHS) or not torch.isfinite(voxels).all():
        raise ValueError('Expected finite [trial, voxel] data matching RUN_LENGTHS after repeat averaging.')
    videos = torch.load(root / 'GT_test_3fps.pt', map_location = 'cpu', weights_only = False)
    if videos.ndim != 5 or len(videos) != len(voxels) or not 0 <= REFERENCE_FRAME < videos.shape[1]:
        raise ValueError('Expected matching [trial, frame, channel, height, width] videos.')
    # Release unused video frames before constructing the large visual encoder.
    videos = videos[:, REFERENCE_FRAME].clone()
    log('Using declared chronological run lengths:', RUN_LENGTHS)
    log('Tensor shapes cannot verify chronology or the existing four-second alignment.', level = 'WARN')
    log('Synthetic BOLD omits realistic spatial covariance; displaced real fMRI is the complementary control.')

    # Use identical sampled token coordinates for the teacher, encoder and all controls.
    coordinates = np.sort(np.random.default_rng(ANALYSIS_SEED).choice(
        256 * 1664, size = FEATURE_COORDINATES, replace = False))
    coordinate_tensor = torch.as_tensor(coordinates, device = device)
    reference_features = np.empty((len(voxels), FEATURE_COORDINATES), dtype = np.float32)
    teacher = FrozenOpenCLIPImageEmbedder(
        arch = 'ViT-bigG-14', version = 'laion2b_s39b_b160k',
        output_tokens = True, only_tokens = True, cache_dir = CACHE_DIR,
    ).to(device).eval().requires_grad_(False)
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type = device.type, dtype = torch.float16,
                                               enabled = USE_MIXED_PRECISION and device.type == 'cuda'):
        for begin in range(0, len(videos), BATCH_SIZE):
            stop = min(begin + BATCH_SIZE, len(videos))
            # Match NEURONS's direct image input, including its existing preprocessing.
            images = videos[begin:stop].float().to(device)
            tokens = teacher(images)
            reference_features[begin:stop] = tokens.flatten(1).index_select(1, coordinate_tensor).float().cpu().numpy()
            if begin == 0 or stop == len(videos) or (begin // BATCH_SIZE) % 20 == 0:
                remaining = (time.perf_counter() - start) * (len(videos) - stop) / stop
                log(f'Reference features: {100 * stop / len(videos):.1f}%; remaining {int(remaining // 60)}m {int(remaining % 60)}s.')
    del teacher, videos, images, tokens
    if not np.isfinite(reference_features).all():
        raise ValueError('Non-finite stimulus features; check the videos and inference precision.')
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Load only the required components strictly; downstream decoders are unnecessary.
    ridge = RidgeRegression([voxels.shape[1]], out_features = HIDDEN_DIM, seq_len = 1)
    backbone = BrainModel(h = HIDDEN_DIM, in_dim = HIDDEN_DIM, seq_len = 1,
                          n_blocks = N_BLOCKS, clip_size = 1664, out_dim = 256 * 1664)
    prior = BrainDiffusionPrior(
        net = PriorNetwork(dim = 1664, depth = PRIOR_DEPTH, dim_head = 52, heads = 32,
                           causal = False, num_tokens = 256, learned_query_mode = 'pos_emb'),
        image_embed_dim = 1664, condition_on_text_encodings = False,
        timesteps = PRIOR_TIMESTEPS, cond_drop_prob = 0.2, image_embed_scale = None,
    )
    components = {'ridge': ridge, 'backbone': backbone, 'diffusion_prior': prior}
    checkpoint = torch.load(CHECKPOINT_PATH, map_location = 'cpu', weights_only = False)
    state = checkpoint['model_state_dict']
    for name, component in components.items():
        component.load_state_dict({key[len(name) + 1:]: value for key, value in state.items()
                                   if key.startswith(name + '.')}, strict = True)
        component.to(device).eval().requires_grad_(False)
    del state, checkpoint
    log('Loaded checkpoint:', CHECKPOINT_PATH)

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(OUTPUT_ROOT) / f'prior_hrf_subj{SUBJECT}_{timestamp}'
    output.mkdir(parents = True, exist_ok = False)
    report: dict[str, Any] = {
        'configuration': config, 'checkpoint': str(Path(CHECKPOINT_PATH).resolve()),
        'analysis': 'Exploratory within-run centred cosine RSA, with Spearman correlation of upper triangles.',
        'limitations': ['Chronology and delay are declared, not inferred.',
                       'Visual HRF references are surrogates, not measured neural activity.',
                       'A single selected image represents each two-second clip.',
                       'Coordinate subsampling approximates full-token geometry.',
                       'Run bootstrap intervals are descriptive with few independent runs.'],
        'kernels': {'hrf': hrf.tolist(), 'generic_smoother': smoother.tolist()},
        'feature_coordinates': coordinates.tolist(), 'runs': [], 'records': [],
    }
    conditions = ('real', 'synthetic_bold', 'gaussian_noise', 'displaced_real')
    offset = 0
    completed = 0
    total = len(RUN_LENGTHS) * len(conditions) * len(CONTROL_SEEDS)
    start = time.perf_counter()
    for run, length in enumerate(RUN_LENGTHS):
        real = voxels[offset:offset + length].numpy()
        stimulus = reference_features[offset:offset + length]
        references = {'stimulus': stimulus}
        for name, kernel in [('hrf', hrf), ('generic_smoother', smoother)]:
            convolved = lfilter(kernel, [1.0], stimulus, axis = 0).astype(np.float32)
            # Tail placeholders are excluded from every measurement by the margins above.
            aligned = np.zeros_like(convolved)
            aligned[:length - delay] = convolved[delay:]
            references[name] = aligned
        valid = np.arange(lower, length - upper_margin)
        rng = np.random.default_rng(ANALYSIS_SEED + run)
        rsa_indices = np.sort(rng.choice(valid, size = min(RSA_TRIALS, len(valid)), replace = False))
        report['runs'].append({'run': run + 1, 'start_trial': offset, 'length': length,
                               'analysed_trial_indices': valid.tolist(), 'rsa_trial_indices': rsa_indices.tolist()})
        if SAVE_FEATURES:
            np.savez_compressed(output / f'run{run + 1}_references.npz', **references)
        for condition in conditions:
            for seed in CONTROL_SEEDS:
                inputs = make_control(real, condition, seed + 10000 * run, hrf, displacement)
                stages = {stage: np.empty((length, FEATURE_COORDINATES), dtype = np.float32)
                          for stage in ('encoder', 'prior')}
                with torch.inference_mode(), torch.autocast(device_type = device.type, dtype = torch.float16,
                                                           enabled = USE_MIXED_PRECISION and device.type == 'cuda'):
                    for begin in range(0, length, BATCH_SIZE):
                        stop = min(begin + BATCH_SIZE, length)
                        batch = torch.from_numpy(inputs[begin:stop]).to(device).unsqueeze(1)
                        _, encoded = backbone(ridge(batch, 0))
                        stages['encoder'][begin:stop] = encoded.flatten(1).index_select(1, coordinate_tensor).float().cpu().numpy()
                        # Couple sampling noise across conditions, independently of control generation.
                        # All DDPM draws use the global generator in the existing sampler.
                        torch.manual_seed(seed + 10000 * run + begin)
                        generated = prior.p_sample_loop(
                            encoded.shape, text_cond = dict(text_embed = encoded),
                            cond_scale = 1.0, timesteps = PRIOR_TIMESTEPS,
                        )
                        stages['prior'][begin:stop] = generated.flatten(1).index_select(1, coordinate_tensor).float().cpu().numpy()
                        log(f'Run {run + 1}, {condition}, seed {seed}: sampled {stop}/{length} trials.')
                for stage, features in stages.items():
                    if not np.isfinite(features).all():
                        raise ValueError(f'Non-finite {stage} features in run {run + 1}, {condition}.')
                    record = {'run': run + 1, 'condition': condition, 'seed': seed, 'stage': stage,
                              **measure(features, references, valid, rsa_indices)}
                    report['records'].append(record)
                    if SAVE_FEATURES:
                        np.savez_compressed(output / f'run{run + 1}_{condition}_seed{seed}_{stage}.npz', features = features)
                # Paired differences use identical trials, coordinates and control realisations.
                before, after = report['records'][-2:]
                delta = {'run': run + 1, 'condition': condition, 'seed': seed, 'stage': 'prior_minus_encoder',
                         'rsa': {name: after['rsa'][name] - before['rsa'][name]
                                 if after['rsa'][name] is not None and before['rsa'][name] is not None else None
                                 for name in references},
                         'lag_cosine': {name: (np.array(after['lag_cosine'][name]) - np.array(before['lag_cosine'][name])).tolist()
                                        for name in references}}
                for name in ('mean_coordinate_variance', 'lag1_temporal_cosine', 'near_zero_centred_fraction'):
                    delta[name] = after[name] - before[name]
                report['records'].append(delta)
                completed += 1
                remaining = (time.perf_counter() - start) * (total - completed) / completed
                log(f'Run {run + 1}, {condition}, seed {seed}: {100 * completed / total:.1f}%; remaining {int(remaining // 60)}m {int(remaining % 60)}s.')
        offset += length

    # Average seeds within each run; resample runs rather than dependent trial pairs.
    report['summary'] = []
    rng = np.random.default_rng(ANALYSIS_SEED)
    bootstrap = rng.integers(0, len(RUN_LENGTHS), size = (BOOTSTRAP_SAMPLES, len(RUN_LENGTHS)))
    stage_names = ('encoder', 'prior', 'prior_minus_encoder')
    fig, axes = plt.subplots(2, len(stage_names), figsize = (8 * len(stage_names), 9), squeeze = False)
    for column, stage in enumerate(stage_names):
        for condition in conditions:
            selected = [r for r in report['records'] if r['stage'] == stage and r['condition'] == condition]
            for reference in references:
                values = np.array([np.mean([r['rsa'][reference] for r in selected if r['run'] == run + 1])
                                   if all(r['rsa'][reference] is not None for r in selected if r['run'] == run + 1)
                                   else np.nan for run in range(len(RUN_LENGTHS))])
                finite = bool(np.isfinite(values).all())
                interval = np.quantile(values[bootstrap].mean(axis = 1), [0.025, 0.975]).tolist() if finite and len(values) > 1 else None
                mean = float(values.mean()) if finite else None
                report['summary'].append({'stage': stage, 'condition': condition, 'reference': reference,
                                          'mean_rsa': mean, 'run_bootstrap_95_interval': interval})
                log(f'{stage}, {condition}, {reference}: RSA = {mean}; run bootstrap 95% interval = {interval}.')
            # Plot original-stimulus temporal specificity; all reference curves remain in JSON.
            curves = np.array([np.mean([r['lag_cosine']['stimulus'] for r in selected if r['run'] == run + 1], axis = 0)
                               for run in range(len(RUN_LENGTHS))])
            line, = axes[1, column].plot(np.array(LAG_TRS) * TR_SECONDS, curves.mean(axis = 0), label = condition)
            if len(RUN_LENGTHS) > 1:
                lo, hi = np.quantile(curves[bootstrap].mean(axis = 1), [0.025, 0.975], axis = 0)
                axes[1, column].fill_between(np.array(LAG_TRS) * TR_SECONDS, lo, hi, color = line.get_color(), alpha = 0.12)
        for index, reference in enumerate(references):
            rows = [next(r for r in report['summary'] if r['stage'] == stage and r['condition'] == condition and r['reference'] == reference)
                    for condition in conditions]
            heights = [r['mean_rsa'] if r['mean_rsa'] is not None else np.nan for r in rows]
            axes[0, column].bar(np.arange(len(conditions)) + (index - 1) * 0.25, heights, width = 0.25, label = reference)
        axes[0, column].set_xticks(np.arange(len(conditions)), [c.replace('_', '\n') for c in conditions])
        axes[0, column].set(title = f'{stage}: representational similarity', ylabel = 'Mean within-run Spearman RSA')
        axes[1, column].axvline(0, color = 'grey', linestyle = '--')
        axes[1, column].set(title = f'{stage}: stimulus specificity', xlabel = 'Stimulus offset (seconds; positive = later)', ylabel = 'Centred cosine similarity')
        for row in range(2):
            axes[row, column].legend(fontsize = 'small')
            axes[row, column].grid(alpha = 0.2, axis = 'y')
    fig.suptitle(f'NEURONS subject {SUBJECT}: frozen representation survey')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'overview.{extension}', dpi = 180)
    plt.close(fig)
    (output / 'metrics.json').write_text(json.dumps(report, indent = 2, allow_nan = False) + '\n')
    log('Saved statistics and figures to', output, level = 'OK')
