# file: data_similarity.py
# Survey one subject and one raw modality; no cross-modality comparisons.
# Run from the project root: python -m dataset.CineBrain.dataset_survey.data_similarity
# dependencies: numpy, matplotlib, scikit-learn, threadpoolctl

# for typing
from __future__ import annotations

# [SET UP INFO] ==========================
# Shared: input/output selection and trial duration.
DATASET_PATH = './data/CineBrain'
OUTPUT_PATH = './results/CineBrain/data_similarity'
MODALITY = 'fmri' # 'eeg' or 'fmri'; run separately for each
SUBJECT_ID = 1
SUBJECT_DIRECTORY = 'sub-{subject_id:04d}'
SEGMENT_SECONDS = 0.8    # duration of one neural file; also the fMRI repetition time
FRAMES_PER_TRIAL = 5     # neural segments, how many segments to group, i.e. 5 segments × 0.8 s = 4 s trials

# EEG only: describe the supplied signal;
# changing the rate does not resample it.
EEG_DIRECTORY = 'eeg_02'
EEG_CHANNEL_COUNT = 64       # retain the first 64 EEG channels; exclude remaining channels
EEG_SAMPLING_RATE_HZ = 1000  # samples per second per channel

# fMRI only: use all supplied voxels;
# vector length is inferred from the files.
FMRI_DIRECTORY = 'visual_audio'

# Shared: subject-local numbering and run split
RUNS_PER_SUBJECT = 20
FRAMES_PER_RUN = 1350
TRAIN_RUNS = tuple(range(1, 19))
VAL_RUNS = ()
TEST_RUNS = (19, 20)
SPLITS = ('train', 'test')  # any non-empty selection of train/val/test
NEURAL_FIRST_INDEX = 0
NEURAL_RUN_STARTS: dict[tuple[int, int], int] = {}  # (subject_id, run): first file index

# Shared: full statistics stream over all trials;
# visuals use a bounded sample (only applies to plots, not general stats. i.e. mean, variance, cosine, PCA).
MAX_PLOT_TRIALS_PER_SPLIT = 128
PLOT_MEMORY_MB = 256         # raw sample buffer only; PCA needs extra workspace
PAIRS_PER_COMPARISON = 1000  # sampled with replacement; no self-pairs
PCA_ENABLED = True           # PCA is the only dimensionality reduction used
PCA_FIT_SPLIT = 'train'      # fit this split only, then project all selected splits
SEED = 42
COMPUTE_THREADS = 4
PROGRESS_EVERY_TRIALS = 100
FIGURE_DPI = 180
# =======================================

import json
from datetime import datetime, timezone
from itertools import combinations_with_replacement
from pathlib import Path
from time import monotonic
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def log(message: str, level: str = 'INFO') -> None:
    # use the same coloured, flushed messages as the CineBrain generation script
    colours = {'INFO': 34, 'WARN': 33, 'ERROR': 31, 'OK': 32}
    print(f'\033[1;{colours[level]}m[CineBrain survey] {level}\033[0m: {message}', flush = True)


# helper to adopt the variance / sim and else from running statistics
def summarise(state: dict[str, Any], dimensions: int) -> dict[str, Any]:
    # Welford's M2 is the sum of squared distances to the group's vector centroid
    count, nonzero = state['count'], state['nonzero']
    variance = max(0.0, state['m2']) / ((count - 1) * dimensions) if count > 1 else None
    cosine = ((float(state['unit_sum'] @ state['unit_sum']) - nonzero) / (nonzero * (nonzero - 1))
              if nonzero > 1 else None)
    return {'trials': count, 'zero_norm_trials': count - nonzero,
            'mean_feature_variance': variance,
            'rms_pairwise_difference': float(np.sqrt(2 * variance)) if variance is not None else None,
            'mean_pairwise_cosine': float(np.clip(cosine, -1, 1)) if cosine is not None else None,
            'centroid_rms': float(np.linalg.norm(state['mean']) / np.sqrt(dimensions))}
    # NOTE: adding two independent and identically distributed (i.i.d.) variables, their covariance would be 0
    #       thus Var(X - Y) = Var(X) + Var(Y) = 2 * Var(X) for i.i.d. X and Y (rms_pairwise_difference)


# Entry point
if __name__ == '__main__':
    started = monotonic()
    # validate split and timing configuration before touching output files
    if MODALITY not in ('eeg', 'fmri') or SUBJECT_ID < 1:
        raise ValueError('Choose eeg or fmri and a positive subject ID.')
    for name, value in (('FRAMES_PER_TRIAL', FRAMES_PER_TRIAL), ('FRAMES_PER_RUN', FRAMES_PER_RUN),
                        ('RUNS_PER_SUBJECT', RUNS_PER_SUBJECT), ('COMPUTE_THREADS', COMPUTE_THREADS),
                        ('MAX_PLOT_TRIALS_PER_SPLIT', MAX_PLOT_TRIALS_PER_SPLIT),
                        ('PAIRS_PER_COMPARISON', PAIRS_PER_COMPARISON), ('PROGRESS_EVERY_TRIALS', PROGRESS_EVERY_TRIALS)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if FRAMES_PER_TRIAL > FRAMES_PER_RUN or not np.isfinite(SEGMENT_SECONDS) or SEGMENT_SECONDS <= 0:
        raise ValueError('Use a positive segment duration and trials no longer than a run.')
    # derive EEG file length from the rate and duration without rounding fractional samples
    eeg_samples_per_frame = None
    if MODALITY == 'eeg':
        if isinstance(EEG_CHANNEL_COUNT, bool) or not isinstance(EEG_CHANNEL_COUNT, int) or EEG_CHANNEL_COUNT < 1:
            raise ValueError('EEG_CHANNEL_COUNT must be a positive integer.')
        if isinstance(EEG_SAMPLING_RATE_HZ, bool) or not np.isfinite(EEG_SAMPLING_RATE_HZ) or EEG_SAMPLING_RATE_HZ <= 0:
            raise ValueError('EEG_SAMPLING_RATE_HZ must be positive and finite.')
        samples = EEG_SAMPLING_RATE_HZ * SEGMENT_SECONDS
        if not np.isfinite(samples) or samples < 1 or not np.isclose(samples, round(samples), rtol = 0, atol = 1e-8):
            raise ValueError('EEG sampling rate multiplied by segment duration must give a positive whole number of samples.')
        eeg_samples_per_frame = int(round(samples))
    if not np.isfinite(PLOT_MEMORY_MB) or PLOT_MEMORY_MB <= 0 or not str(OUTPUT_PATH).strip():
        raise ValueError('Specify OUTPUT_PATH and a positive sample memory budget.')
    split_runs = {'train': TRAIN_RUNS, 'val': VAL_RUNS, 'test': TEST_RUNS}
    if sorted((*TRAIN_RUNS, *VAL_RUNS, *TEST_RUNS)) != list(range(1, RUNS_PER_SUBJECT + 1)):
        raise ValueError('TRAIN_RUNS, VAL_RUNS and TEST_RUNS must partition all runs exactly once.')
    if not SPLITS or len(set(SPLITS)) != len(SPLITS) or any(split not in split_runs for split in SPLITS):
        raise ValueError('SPLITS must contain unique train/val/test entries.')
    if PCA_ENABLED and PCA_FIT_SPLIT not in SPLITS:
        raise ValueError('PCA_FIT_SPLIT must be included in SPLITS.')
    source = Path(DATASET_PATH) / SUBJECT_DIRECTORY.format(subject_id = SUBJECT_ID)
    source /= EEG_DIRECTORY if MODALITY == 'eeg' else FMRI_DIRECTORY
    if not source.is_dir():
        raise FileNotFoundError(f"Missing modality directory '{source}'.")

    # scan filenames once; run offsets never change when source files are missing
    files: dict[int, Path] = {}
    for path in source.glob('*.npy'):
        if path.stem.isdigit():
            index = int(path.stem)
            if index in files:
                raise ValueError(f'Duplicate numerical file index {index} in {source}.')
            files[index] = path
    run_starts = {run: NEURAL_RUN_STARTS.get((SUBJECT_ID, run), NEURAL_FIRST_INDEX + (run - 1) * FRAMES_PER_RUN)
                  for run in range(1, RUNS_PER_SUBJECT + 1)}
    ordered_starts = sorted(run_starts.values())
    if any(not isinstance(start, int) or start < 0 for start in ordered_starts):
        raise ValueError('Neural run starts must be non-negative integers.')
    if any(right - left < FRAMES_PER_RUN for left, right in zip(ordered_starts, ordered_starts[1:])):
        raise ValueError('Neural run ranges must not overlap.')
    plan: list[dict[str, Any]] = []
    coverage = []
    for run, start in run_starts.items():
        split = next(split for split, runs in split_runs.items() if run in runs)
        if split not in SPLITS:
            continue
        complete_end = FRAMES_PER_RUN - FRAMES_PER_RUN % FRAMES_PER_TRIAL
        missing = [index for index in range(start, start + FRAMES_PER_RUN) if index not in files]
        kept, incomplete = 0, 0
        for offset in range(0, complete_end, FRAMES_PER_TRIAL):
            first = start + offset
            if any(index not in files for index in range(first, first + FRAMES_PER_TRIAL)):
                incomplete += 1
                continue
            plan.append({'run': run, 'split': split, 'first_index': first,
                         'last_index': first + FRAMES_PER_TRIAL - 1, 'offset_seconds': offset * SEGMENT_SECONDS})
            kept += 1
        coverage.append({'run': run, 'split': split, 'complete_trials': kept,
                         'incomplete_trials': incomplete, 'tail_segments': FRAMES_PER_RUN - complete_end,
                         'missing_indices': missing})
        log(f'Run {run:02d} ({split}): {kept} trials, {incomplete} incomplete groups, '
            f'{FRAMES_PER_RUN - complete_end} tail segments, {len(missing)} missing files.',
            'WARN' if incomplete or FRAMES_PER_RUN != complete_end else 'INFO')
    rows = {split: np.array([i for i, trial in enumerate(plan) if trial['split'] == split], dtype = int)
            for split in SPLITS}
    if any(len(indices) == 0 for indices in rows.values()):
        raise ValueError('Every selected split needs at least one complete trial; check run rules and source files.')

    # infer the raw trial layout; EEG concatenates time, fMRI stacks segment vectors
    first_array = np.load(files[plan[0]['first_index']], allow_pickle = False)
    if MODALITY == 'eeg':
        if first_array.ndim != 2 or first_array.shape[0] < EEG_CHANNEL_COUNT or first_array.shape[1] != eeg_samples_per_frame:
            raise ValueError(f'Expected EEG (at least {EEG_CHANNEL_COUNT} channels, {eeg_samples_per_frame} samples).')
        segment_shape = (EEG_CHANNEL_COUNT, eeg_samples_per_frame)
        trial_shape = (EEG_CHANNEL_COUNT, eeg_samples_per_frame * FRAMES_PER_TRIAL)
    else:
        if first_array.ndim != 1 or not first_array.size:
            raise ValueError('Expected a non-empty fMRI voxel vector per file.')
        segment_shape = first_array.shape
        trial_shape = (FRAMES_PER_TRIAL, first_array.size)
    dimensions = int(np.prod(trial_shape))
    del first_array
    capacity = int(PLOT_MEMORY_MB * 1024 ** 2) // (dimensions * 8 * len(SPLITS))
    per_split = min(MAX_PLOT_TRIALS_PER_SPLIT, capacity)
    if per_split < 1:
        raise ValueError('PLOT_MEMORY_MB cannot hold one trial per split; increase the budget.')
    # sample: uniform without replacement within each split (only selected ONCE)
    # later pair draws are with replacement
    rng = np.random.default_rng(SEED)
    sampled_rows = np.sort(np.concatenate([rng.choice(indices, min(len(indices), per_split), replace = False)
                                           for indices in rows.values()]))
    # lookup stores the sample row index for each trial row; later pair draws are with replacement
    sample_lookup = {int(row): i for i, row in enumerate(sampled_rows)}
    # samples stores the raw trial vectors for the visual sample; used for pairwise comparisons (filled at around line 260)
    samples = np.empty((len(sampled_rows), dimensions), dtype = np.float64)
    # sample_splits stores the split for each sampled trial; used for pairwise comparisons
    # NOTE: should be an array of strings (objects)
    sample_splits = np.array([plan[row]['split'] for row in sampled_rows])
    log(f'{len(plan)} {MODALITY.upper()} trials of {FRAMES_PER_TRIAL * SEGMENT_SECONDS:g} s; '
        f'shape {trial_shape}. Visual sample: {len(samples)} trials ({samples.nbytes / 1024 ** 2:.1f} MiB).')
    log('Raw values are not filtered or standardised. Cosine measures direction; raw distances retain amplitude.')

    # keep only group centroids and scalar dispersion; no full dataset or N-by-N matrix
    # NOTE: "states" stores running statistics
    # NOTE: "states" has the following keys:
    # states['all']      # all selected trials
    # states['train']    # training trials
    # states['test']     # testing trials
    # states['run_x']    # trials from run x (run -> one episode in real case)
    # NOTE: for each key (group), the sub dict has keys:
    # 'count'     # number of trials in the group
    # 'mean'      # running mean
    # 'm2'        # running sum of squared distances to the mean (used for variance) (Welford’s algorithm)
    # 'nonzero'   # number of trials whose values are not all zero
    # 'unit_sum'  # Sum of trial vectors after dividing each by its length; Used to calculate average cosine similarity efficiently
    # NOTE: Welford’s algorithm:
    #       It accounts for the mean moving as new trials arrive,
    #       so we do not need to revisit earlier trials
    states: dict[str, dict[str, Any]] = {}
    with threadpool_limits(limits = COMPUTE_THREADS):
        for row, trial in enumerate(plan):
            # load segments
            segments = []
            for index in range(trial['first_index'], trial['last_index'] + 1):
                array = np.load(files[index], allow_pickle = False)
                if MODALITY == 'eeg' and array.ndim == 2:
                    array = array[:EEG_CHANNEL_COUNT]
                if array.shape != segment_shape or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
                    raise ValueError(f"Invalid shape, dtype or non-finite values in '{files[index]}'.")
                segments.append(array)
            # construct our defined trial (with the right length)
            grouped = np.concatenate(segments, axis = -1) if MODALITY == 'eeg' else np.stack(segments)
            # NOTE: fix type and flatten
            vector = grouped.astype(np.float64, copy = False).reshape(-1)
            # calculate norm
            norm = float(np.linalg.norm(vector))
            # NOTE: unit vector for cosine similarity; zero-norm trials are excluded from the mean cosine
            unit = vector / norm if norm > 0 else np.zeros(dimensions)
            for key in ('all', trial['split'], f"run_{trial['run']}"):
                if key not in states:
                    states[key] = {'count': 0, 'mean': np.zeros(dimensions), 'm2': 0.0,
                                   'nonzero': 0, 'unit_sum': np.zeros(dimensions)}
                state = states[key]
                state['count'] += 1
                delta = vector - state['mean']
                state['mean'] += delta / state['count']
                # NOTE: Welford's M2:
                # m2(new) = m2(old) + delta * (x - mean(new))
                # delta is the difference between the new vector and the **old** mean
                state['m2'] += float(delta @ (vector - state['mean']))
                if norm > 0:
                    state['nonzero'] += 1
                    state['unit_sum'] += unit
            trial['rms'] = norm / np.sqrt(dimensions)
            # NOTE: write the trial vector to the visual sample if it was selected;
            #       later pair draws are with replacement
            if row in sample_lookup:
                samples[sample_lookup[row]] = vector
            if (row + 1) % PROGRESS_EVERY_TRIALS == 0 or row + 1 == len(plan):
                remaining = round((monotonic() - started) * (len(plan) - row - 1) / (row + 1))
                minutes, seconds = divmod(remaining, 60)
                log(f'Loaded {row + 1}/{len(plan)} ({100 * (row + 1) / len(plan):.1f}%), '
                    f'estimated loading time remaining {minutes}m {seconds:02d}s.')

        # exact **cross-split** moments use all trials, without materialising pairwise differences
        statistics = { key: summarise(state, dimensions) for key, state in states.items() }
        cross_split = {}
        for left, right in combinations_with_replacement(SPLITS, 2):
            if left == right:
                continue
            a, b = states[left], states[right]
            delta = a['mean'] - b['mean']
            distance2 = float(delta @ delta)
            denominator = a['nonzero'] * b['nonzero']
            cosine = float(a['unit_sum'] @ b['unit_sum']) / denominator if denominator else None
            cross_split[f'{left}--{right}'] = {
                'centroid_rmse': float(np.sqrt(distance2 / dimensions)),
                'rms_pairwise_difference': float(np.sqrt(max(0.0, distance2 + a['m2'] / a['count'] + b['m2'] / b['count']) / dimensions)),
                'mean_pairwise_cosine': float(np.clip(cosine, -1, 1)) if cosine is not None else None}
        for split in SPLITS:
            log(f'{split}: {statistics[split]}', 'OK')
        for comparison, values in cross_split.items():
            log(f'{comparison}: {values}', 'OK')

        # sample original-space distances for distributions; reduction never feeds these metrics
        # NOTE: pair_values is a dict (keys are the pair labels) of a dict (metrics) of a list of sampled numbers
        pair_values, pair_statistics = {}, {}
        for left, right in combinations_with_replacement(SPLITS, 2):
            # fetch splits (indices)
            a = np.flatnonzero(sample_splits == left)
            b = np.flatnonzero(sample_splits == right)
            # NOTE: skip if the same split has fewer than two sampled trials; no self-pairs
            if left == right and len(a) < 2:
                continue
            # get random pairs with replacement;
            # NOTE: self-pairs are excluded for within-split comparisons
            first = rng.integers(len(a), size = PAIRS_PER_COMPARISON)
            if left == right:
                second = (first + rng.integers(1, len(a), size = PAIRS_PER_COMPARISON)) % len(a)
            else:
                second = rng.integers(len(b), size = PAIRS_PER_COMPARISON)
            # calculate distances and cosines for the sampled pairs
            distances, cosines = [], []
            for i, j in zip(a[first], b[second]):
                x, y = samples[i], samples[j]
                delta = x - y
                distances.append(float(np.linalg.norm(delta) / np.sqrt(dimensions)))
                denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
                if denominator > 0:
                    cosines.append(float(np.clip((x @ y) / denominator, -1, 1)))
            label = f'{left}--{right}'
            pair_values[label] = {'rmse': distances, 'cosine': cosines}
            pair_statistics[label] = {}
            for metric, values in pair_values[label].items():
                pair_statistics[label][metric] = ({'draws': len(values), 'mean': float(np.mean(values)),
                    'std': float(np.std(values)), 'quantiles_05_50_95': np.quantile(values, [0.05, 0.5, 0.95]).tolist()}
                    if values else {'draws': 0, 'mean': None, 'std': None, 'quantiles_05_50_95': None})

        # fit just one reduction on the chosen split;
        # NOTE: test data should never determines the PCA axes
        embedding, explained = None, None
        if PCA_ENABLED:
            fit_rows = np.flatnonzero(sample_splits == PCA_FIT_SPLIT)
            if len(fit_rows) < 3 or dimensions < 2 or statistics[PCA_FIT_SPLIT]['mean_feature_variance'] == 0:
                log('PCA omitted: need at least three sampled fit trials, two features and non-zero variance.', 'WARN')
            else:
                fit_data = samples[fit_rows]
                # NOTE: np.ptp: Range of values (maximum - minimum) along an axis.
                #       The name of the function comes from the acronym for 'peak to peak'
                if not np.any(np.ptp(fit_data, axis = 0)):
                    log('PCA omitted: sampled fit trials are identical.', 'WARN')
                else:
                    log(f'Fitting two-component PCA on {len(fit_rows)} sampled {PCA_FIT_SPLIT} trials.')
                    pca = PCA(n_components = 2, svd_solver = 'randomized', iterated_power = 2,
                              whiten = False, random_state = SEED)
                    # NOTE: only fit on the chosen split
                    pca.fit(fit_data)
                    # transform all data
                    embedding = pca.transform(samples) # shape (len(samples), 2)
                    explained = pca.explained_variance_ratio_.tolist()
                    log(f'PCA explained variance: PC1 {explained[0]:.2%}, PC2 {explained[1]:.2%}.')
                del fit_data

    # create output path
    output = Path(OUTPUT_PATH)
    output.mkdir(parents = True, exist_ok = True)
    # create distinct output filenames
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    prefix = f'{MODALITY}_subj{SUBJECT_ID}_{FRAMES_PER_TRIAL}frames_{stamp}'
    # collector of output artifacts' paths
    figures = []

    # colour maps and markers for the plots
    colours = {'train': '#0072B2', 'val': '#009E73', 'test': '#D55E00'}
    markers = {'train': 'o', 'val': 's', 'test': '^'}
    # override matplotlib settings
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 10, 'axes.titlesize': 12, 'figure.facecolor': 'white'})

    # PLOT 1: PCA projection, trial RMS over time, between-trial dispersion, run centroid cosine
    fig, axes = plt.subplots(2, 2, figsize = (14, 10), layout = 'constrained')
    fig.suptitle(f'{MODALITY.upper()} · subject {SUBJECT_ID} · {FRAMES_PER_TRIAL * SEGMENT_SECONDS:g} s trials · raw signals', fontsize = 16)
    # PCA projection
    ax = axes[0, 0]
    if embedding is not None:
        for split in SPLITS:
            selected = sample_splits == split
            ax.scatter(*embedding[selected].T, s = 28, alpha = 0.65, c = colours[split],
                       marker = markers[split], label = f'{split} (n = {selected.sum()})', edgecolors = 'none')
        ax.set(xlabel = f'PC1 ({explained[0]:.1%} fit variance)', ylabel = f'PC2 ({explained[1]:.1%} fit variance)',
               title = f'PCA sample · fitted on {PCA_FIT_SPLIT} only')
        ax.legend(frameon = False)
    else:
        ax.text(0.5, 0.5, 'PCA disabled or insufficient variation', ha = 'center', va = 'center', transform = ax.transAxes)
        ax.set_title('PCA projection')
    # Trial RMS over time (RMS: root mean square)
    # NOTE: this only reflects the raw amplitude of the signal
    #       it has nothing to do with measuring difference between trials or self variance
    ax = axes[0, 1]
    for split in SPLITS:
        selected = [trial for trial in plan if trial['split'] == split]
        times = [(trial['run'] - 1) * FRAMES_PER_RUN * SEGMENT_SECONDS / 60 + trial['offset_seconds'] / 60 for trial in selected]
        ax.scatter(times, [trial['rms'] for trial in selected], s = 8, alpha = 0.55,
                   color = colours[split], marker = markers[split], label = split)
    ax.set(xlabel = 'Concatenated run time (minutes; breaks excluded)', ylabel = 'Trial RMS (source units)',
           title = 'Raw amplitude over time · all trials')
    ax.legend(frameon = False)
    # Between-trial dispersion (for each run)
    ax = axes[1, 0]
    for entry in coverage:  # for each run (episode)
        values = statistics.get(f"run_{entry['run']}")
        if values is not None and values['mean_feature_variance'] is not None:
            ax.bar(entry['run'], values['mean_feature_variance'], color = colours[entry['split']])
    ax.set(xlabel = 'Run', ylabel = 'Mean feature variance (source units²)', title = 'Between-trial dispersion · all trials')
    ax.set_xticks([entry['run'] for entry in coverage])
    # Run centroid cosine similarity (not trial pairs)
    # centroids are ONE per run (episode)
    ax = axes[1, 1]
    active_runs = [entry['run'] for entry in coverage if f"run_{entry['run']}" in states]
    centroids = np.stack([states[f'run_{run}']['mean'] for run in active_runs])
    norms = np.linalg.norm(centroids, axis = 1)
    valid = norms > 0
    centroids /= np.where(valid, norms, 1)[:, None]
    with threadpool_limits(limits = COMPUTE_THREADS):
        run_cosine = np.clip(centroids @ centroids.T, -1, 1)
    run_cosine[~valid, :] = np.nan
    run_cosine[:, ~valid] = np.nan
    palette = plt.get_cmap('coolwarm').copy()
    palette.set_bad('#dddddd')
    image = ax.imshow(np.ma.masked_invalid(run_cosine), vmin = -1, vmax = 1, cmap = palette)
    ax.set(xticks = range(len(active_runs)), yticks = range(len(active_runs)),
           xticklabels = active_runs, yticklabels = active_runs, xlabel = 'Run', ylabel = 'Run',
           title = 'Cosine between run centroids (not trial pairs)')
    for axis in (ax.xaxis, ax.yaxis):
        for tick, run in zip(axis.get_ticklabels(), active_runs):
            split = next(split for split, runs in split_runs.items() if run in runs)
            tick.set_color(colours[split])
    fig.colorbar(image, ax = ax, label = 'Cosine similarity', shrink = 0.8)

    # save PLOT 1
    for extension in ('png', 'pdf'):
        path = output / f'{prefix}_overview.{extension}'
        fig.savefig(path, dpi = FIGURE_DPI)
        figures.append(path.name)
    plt.close(fig)


    # PLOT 2: histograms of sampled pairwise differences
    # use shared bin edges and density so unequal split sizes do not dominate the histograms
    fig, axes = plt.subplots(1, 2, figsize = (13, 4.5), layout = 'constrained')
    fig.suptitle('Original-space trial differences · sampled pairs, with replacement')
    for ax, metric, title in zip(axes, ('rmse', 'cosine'), ('Pairwise RMSE (source units)', 'Pairwise cosine similarity')):
        available = [values[metric] for values in pair_values.values() if values[metric]]
        if available:
            bins = np.histogram_bin_edges(np.concatenate(available), bins = 35)
            for label, values in pair_values.items():
                if values[metric]:
                    left, right = label.split('--')
                    ax.hist(values[metric], bins = bins, density = True, histtype = 'step', linewidth = 1.8,
                            color = colours[left] if left == right else None,
                            linestyle = '-' if left == right else '--', label = label)
            ax.legend(frameon = False)
        else:
            ax.text(0.5, 0.5, 'No valid pairs', ha = 'center', transform = ax.transAxes)
        ax.set(xlabel = title, ylabel = 'Density')
        ax.grid(alpha = 0.15)

    # save PLOT 2
    for extension in ('png', 'pdf'):
        path = output / f'{prefix}_pair_distributions.{extension}'
        fig.savefig(path, dpi = FIGURE_DPI)
        figures.append(path.name)
    plt.close(fig)

    # retain exact definitions and trial identities alongside the visual sample for reproducibility
    report = {
        'configuration': {'dataset_path': str(Path(DATASET_PATH).resolve()), 'output_path': str(output.resolve()),
            'modality': MODALITY, 'subject_id': SUBJECT_ID, 'source_directory': str(source.resolve()),
            'frames_per_trial': FRAMES_PER_TRIAL, 'segment_seconds': SEGMENT_SECONDS,
            'trial_shape': trial_shape, 'flattened_features': dimensions, 'splits': SPLITS,
            'split_runs': split_runs, 'frames_per_run': FRAMES_PER_RUN, 'run_starts': run_starts,
            'eeg_channel_count': EEG_CHANNEL_COUNT if MODALITY == 'eeg' else None,
            'eeg_sampling_rate_hz': EEG_SAMPLING_RATE_HZ if MODALITY == 'eeg' else None,
            'eeg_samples_per_frame': eeg_samples_per_frame,
            'seed': SEED, 'max_plot_trials_per_split': MAX_PLOT_TRIALS_PER_SPLIT,
            'plot_memory_mb': PLOT_MEMORY_MB, 'pairs_per_comparison': PAIRS_PER_COMPARISON,
            'compute_threads': COMPUTE_THREADS, 'accumulation_dtype': 'float64'},
        'definitions': {
            'mean_feature_variance': 'Mean across flattened features of unbiased between-trial variance (ddof=1).',
            'rms_pairwise_difference': 'Square root of mean pairwise squared RMSE; not the arithmetic mean RMSE. Within groups exclude self-pairs.',
            'mean_pairwise_cosine': 'Exact mean over all distinct within-group or all cross-group trial pairs; zero-norm trials excluded.',
            'centroid_rmse': 'RMSE between split mean trial vectors, in source units.',
            'sampling': 'Uniform without replacement within each split for visual trials; pair draws with replacement from that sample.',
            'interpretation': 'Descriptive only: neighbouring trials are dependent; splits also differ in run/stimulus. Raw cosine is sensitive to shared offsets.',
            'preprocessing': 'None. EEG excludes trailing non-EEG channels. PCA centres features without scaling or whitening; cosine uses norms internally.'},
        'coverage': coverage, 'statistics_all_trials': statistics, 'cross_split_all_trials': cross_split,
        'sampled_pair_statistics': pair_statistics, 'trials': plan,
        'pca': {'enabled': PCA_ENABLED, 'fit_split': PCA_FIT_SPLIT, 'explained_variance_ratio': explained,
                'sampled_trial_rows': sampled_rows.tolist(), 'sampled_splits': sample_splits.tolist(),
                'coordinates': embedding.tolist() if embedding is not None else None},
        'run_centroid_cosine': {'runs': active_runs,
            'matrix': [[float(value) if np.isfinite(value) else None for value in row] for row in run_cosine]},
        'figures': figures, 'elapsed_seconds': monotonic() - started}
    destination = output / f'{prefix}.json'
    destination.write_text(json.dumps(report, indent = 2, allow_nan = False), encoding = 'utf-8')
    log(f"Saved statistics and four figures to '{output}'; report: '{destination.name}'.", 'OK')
