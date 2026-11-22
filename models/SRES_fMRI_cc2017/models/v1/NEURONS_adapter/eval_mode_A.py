# file: eval_mode_A.py
# Compare original and packed mode-A fMRI (de-noised)
# run with `python -m models.SRES_fMRI_cc2017.models.v1.NEURONS_adapter.eval_mode_A`

from __future__ import annotations

# [SETUP] ===============================================================
ORIGINAL_TRAIN = './data/cc2017/subj01_train_fmri.pt'
ORIGINAL_TEST = './data/cc2017/subj01_test_fmri.pt'
PROCESSED_TRAIN = './data/cc2017_SRES_mode_A/subj01_train_fmri.pt'
PROCESSED_TEST = './data/cc2017_SRES_mode_A/subj01_test_fmri.pt'
OUTPUT_DIR = './results/SRES_fMRI_cc2017/mode_A_comparison'

SAMPLING_HZ = 0.5                             # Original (scanner) sampling rate
TRAIN_RUN_LENGTHS = ','.join(['240'] * 18)    # Volume counts for each run, preserving boundaries (chronological run order)
TEST_RUN_LENGTHS = ','.join(['240'] * 5)

# NOTE: the excluded beginning and end happened in file `pack_mode_A.py`
#       at line 109-110 (indexed fill) + line 119 (fill beginning and end with original values)
EXCLUDE_START_TRS = 15                        # Match packing burn-in; whole-file MSE is also reported
EXCLUDE_END_TRS = 4                           # Match packing trailing exclusion

SIGNAL_BAND_HZ = (0.01, 0.1)                  # Assumed band for a spectral proxy, not identified neural signal (BOLD response band); SNR estimator
SPECTROGRAM_WINDOW_SEC = 64.0                 # Local frequency resolution is approximately 1 / window duration (Use 30 for an HRF-scale view with coarser frequency resolution)
SPECTROGRAM_HOP_SEC = 8.0                     # Step between local spectral windows
VOXEL_CHUNK = 256                             # Bound memory used by paired statistics and Fourier transforms (Voxels processed at once)
CPU_THREADS = 4                               # Torch CPU threads
RASTERISE_PDF = True                          # Rasterise heatmaps (PDF graphics)
HEATMAP_RANGE_DB = 60.0                       # Shared colour range below the largest PSD; limits numerical-floor artefacts (Shared colour range below the maximum PSD, in dB)
FIGURE_DPI = 160                              # PNG resolution and rasterised PDF resolution
LOG_EVERY = 20                                # Progress interval in voxel chunks
# =======================================================================

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

from ....modules.runtime import log, progress, save_json


def paired_sums(original: np.ndarray, processed: np.ndarray, sampling_hz: float) -> dict[str, float]:
    """
    Accumulate paired statistics over [time, repetition, voxel] blocks without crossing runs
    """
    difference = processed - original
    # Centre each voxel/repetition time series independently for temporal correlation
    centred_original = original - original.mean(axis = 0, keepdims = True)
    centred_processed = processed - processed.mean(axis = 0, keepdims = True)
    energy_original = np.square(centred_original).sum(axis = 0)
    energy_processed = np.square(centred_processed).sum(axis = 0)
    denominator = np.sqrt(energy_original * energy_processed)
    valid = denominator > 0
    correlation = (centred_original * centred_processed).sum(axis = 0)
    return {'elements': float(original.size), 'squared_change': float(np.square(difference).sum()),
            'absolute_change': float(np.abs(difference).sum()), 'signed_change': float(difference.sum()),
            'original_energy': float(np.square(original).sum()), 'processed_energy': float(np.square(processed).sum()),
            'original_mean_sum': float(original.sum()), 'processed_mean_sum': float(processed.sum()),
            'original_tv_sum': float(np.abs(np.diff(original, axis = 0)).sum() * sampling_hz),
            'processed_tv_sum': float(np.abs(np.diff(processed, axis = 0)).sum() * sampling_hz),
            'difference_elements': float((len(original) - 1) * original.shape[1] * original.shape[2]),
            'correlation_sum': float(np.clip(correlation[valid] / denominator[valid], -1, 1).sum()),
            'correlation_count': float(valid.sum()), 'series_count': float(valid.size)}


def main(args: argparse.Namespace) -> None:
    """
    Measure paired changes and average spectral power within runs, preserving temporal boundaries
    """
    # Check units and frequency support before loading the potentially large tensors
    if not np.isfinite(args.sampling_hz) or args.sampling_hz <= 0:
        raise ValueError('Sampling frequency must be positive and finite')
    if min(args.voxel_chunk, args.cpu_threads, args.figure_dpi, args.log_every) < 1:
        raise ValueError('Chunk size, CPU threads, figure DPI and progress interval must be positive')
    if min(args.exclude_start_trs, args.exclude_end_trs) < 0:
        raise ValueError('Run-edge exclusions cannot be negative')
    low, high = args.signal_band_hz
    if not 0 < low < high < args.sampling_hz / 2:
        raise ValueError('Signal band must lie strictly between zero and the Nyquist frequency')
    if not np.isfinite([args.spectrogram_window_sec, args.spectrogram_hop_sec]).all():
        raise ValueError('Spectrogram durations must be finite')
    window = round(args.spectrogram_window_sec * args.sampling_hz)
    hop = round(args.spectrogram_hop_sec * args.sampling_hz)
    if window < 4 or not 1 <= hop <= window:
        raise ValueError('Use a spectrogram window of at least four samples and a hop between one sample and the window length')
    if not np.isfinite(args.heatmap_range_db) or args.heatmap_range_db <= 0:
        raise ValueError('Heatmap colour range must be positive and finite')
    torch.set_num_threads(args.cpu_threads)
    folder = Path(args.output_dir)
    folder.mkdir(parents = True, exist_ok = True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report: dict[str, Any] = {'settings': vars(args), 'splits': {}, 'interpretation': {
        'mse': 'Change from original BOLD, not error against clean ground truth; packed rows are counted once',
        'interior': 'Run edges excluded using the supplied counts; they must match the packing configuration',
        'spectral_ratio': '10 log10(in-band power / other positive-frequency power); a SNR estimator',
        'spectra': 'Hann-windowed, temporally demeaned periodograms; DC excluded from band-power calculations',
        'heatmaps': 'Mean local PSD across voxels, repetitions and runs at matching elapsed run time, never mean signals',
        'repeat_handling': 'Repetitions analysed independently; original and processed shapes must match'}}
    log(f'Local spectral resolution approximately {args.sampling_hz / window:.5g} Hz; Nyquist {args.sampling_hz / 2:g} Hz')
    log('Whole-file metrics include original boundary fallbacks; interior metrics use the configured exclusions')
    heatmaps: dict[str, dict[str, Any]] = {}
    spectra: dict[str, dict[str, Any]] = {}

    for phase in ('train', 'test'):
        # mmap avoids loading complete modern Torch archives; older archives use ordinary CPU loading
        tensors = []
        for path in (getattr(args, f'original_{phase}'), getattr(args, f'processed_{phase}')):
            try:
                tensor = torch.load(path, map_location = 'cpu', weights_only = True, mmap = True)
            except RuntimeError as exc:
                if 'mmap' not in str(exc):
                    raise
                log(f'{path}: archive does not support mmap; loading into CPU memory', 'WARN')
                tensor = torch.load(path, map_location = 'cpu', weights_only = True)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim not in (2, 3):
                raise ValueError(f'{path}: expected a tensor [time, repetition, voxel] or [time, voxel]')
            tensors.append(tensor.unsqueeze(1) if tensor.ndim == 2 else tensor)
        original, processed = tensors
        if original.shape != processed.shape or min(original.shape) < 1:
            raise ValueError(f'{phase}: original and processed shapes must match, including repetitions')
        lengths = [int(value) for value in getattr(args, f'{phase}_run_lengths').split(',')]
        retained = [length - args.exclude_start_trs - args.exclude_end_trs for length in lengths]
        if sum(lengths) != len(original) or min(retained) < window:
            raise ValueError(f'{phase}: run lengths must cover the tensor and each retained run must fit the spectrogram window')
        # Use a common FFT grid for aggregation without resampling the input time series
        nfft = max(retained)
        frequencies = np.fft.rfftfreq(nfft, d = 1 / args.sampling_hz)
        band = (frequencies >= low) & (frequencies <= high)
        outside = (frequencies > 0) & ~band
        if not band.any() or not outside.any():
            raise ValueError('The selected band cannot be resolved on the run frequency grid')
        max_windows = 1 + (max(retained) - window) // hop
        heat_sum = np.zeros((2, window // 2 + 1, max_windows), dtype = np.float64)
        heat_count = np.zeros(max_windows, dtype = np.float64)
        psd_sum = np.zeros((2, len(frequencies)), dtype = np.float64)
        psd_count = 0
        totals: dict[str, dict[str, float]] = {'all': {}, 'interior': {}}
        run_reports = []
        offset, completed = 0, 0
        started = monotonic()
        work = len(lengths) * ((original.shape[2] + args.voxel_chunk - 1) // args.voxel_chunk)
        for run, length in enumerate(lengths, start = 1):
            run_sums: dict[str, float] = {}
            for first in range(0, original.shape[2], args.voxel_chunk):
                last = min(first + args.voxel_chunk, original.shape[2])
                # Float64 within a bounded chunk avoids cancellation in small paired differences
                x = original[offset:offset + length, :, first:last].double().numpy()
                y = processed[offset:offset + length, :, first:last].double().numpy()
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f'{phase}, run {run}: non-finite input values')
                stop = length - args.exclude_end_trs
                xi, yi = x[args.exclude_start_trs:stop], y[args.exclude_start_trs:stop]
                for label, a, b in (('all', x, y), ('interior', xi, yi)):
                    partial = paired_sums(a, b, args.sampling_hz)
                    for key, value in partial.items():
                        totals[label][key] = totals[label].get(key, 0.0) + value
                        if label == 'interior':
                            run_sums[key] = run_sums.get(key, 0.0) + value
                # Compute each series' power first; averaging raw signals would cancel opposing responses
                series_count = xi.shape[1] * xi.shape[2]
                for index, values in enumerate((xi, yi)):
                    values = values.transpose(1, 2, 0).copy()
                    _, psd = signal.periodogram(values, fs = args.sampling_hz, window = 'hann',
                                                detrend = 'constant', nfft = nfft, scaling = 'density', axis = -1)
                    psd_sum[index] += psd.sum(axis = (0, 1)) * len(xi)
                    local_frequencies, local_times, power = signal.spectrogram(
                        values, fs = args.sampling_hz, window = 'hann', nperseg = window,
                        noverlap = window - hop, detrend = 'constant', scaling = 'density', mode = 'psd', axis = -1)
                    heat_sum[index, :, :len(local_times)] += power.sum(axis = (0, 1))
                heat_count[:len(local_times)] += series_count
                psd_count += series_count * len(xi)
                completed += 1
                if completed % args.log_every == 0 or completed == work:
                    progress(f'Analysing {phase}', completed, work, started)
            run_reports.append({'run': run, 'volumes': length, 'retained_volumes': retained[run - 1],
                                'interior_change_mse': run_sums['squared_change'] / run_sums['elements']})
            offset += length
        # Convert additive statistics into interpretable metrics, recording undefined values as null
        summary: dict[str, Any] = {'shape': list(original.shape), 'runs': run_reports}
        for label, sums in totals.items():
            count = sums['elements']
            summary[label] = {
                'elements': int(count), 'denoising_change_mse': sums['squared_change'] / count,
                'change_rmse': (sums['squared_change'] / count) ** 0.5,
                'change_mae': sums['absolute_change'] / count, 'mean_shift': sums['signed_change'] / count,
                'relative_change_energy': sums['squared_change'] / sums['original_energy'] if sums['original_energy'] > 0 else None,
                'mean_temporal_correlation': sums['correlation_sum'] / sums['correlation_count'] if sums['correlation_count'] > 0 else None,
                'nonconstant_series': int(sums['correlation_count']), 'series_count': int(sums['series_count'])}
            for label_source in ('original', 'processed'):
                summary[label][label_source] = {
                    'mean': sums[f'{label_source}_mean_sum'] / count,
                    'rms': (sums[f'{label_source}_energy'] / count) ** 0.5,
                    'temporal_tv_per_sec': sums[f'{label_source}_tv_sum'] / sums['difference_elements']}
        mean_psd = psd_sum / psd_count
        frequency_step = args.sampling_hz / nfft
        summary['spectral'] = {'frequency_step_hz': frequency_step, 'scope': 'interior',
                               'aggregation': 'Power averaged over series, weighted by retained run duration'}
        for index, name in enumerate(('original', 'processed')):
            in_power = float(mean_psd[index, band].sum() * frequency_step)
            out_power = float(mean_psd[index, outside].sum() * frequency_step)
            summary['spectral'][name] = {
                'in_band_power': in_power, 'out_of_band_power': out_power,
                'in_band_fraction': in_power / (in_power + out_power) if in_power + out_power > 0 else None,
                'band_power_ratio_db': 10 * np.log10(in_power / out_power) if min(in_power, out_power) > 0 else None}
        report['splits'][phase] = summary
        heatmaps[phase] = {'power': heat_sum / heat_count[None, None, :], 'frequencies': local_frequencies,
                          'times': (args.exclude_start_trs + window / 2 + np.arange(max_windows) * hop) / args.sampling_hz,
                          'series_per_time': heat_count}
        spectra[phase] = {'frequencies': frequencies, 'power': mean_psd}
        log(f'{phase}: whole-file MSE = {summary["all"]["denoising_change_mse"]:.6g}, interior MSE = {summary["interior"]["denoising_change_mse"]:.6g}')
        for name in ('original', 'processed'):
            value = summary['spectral'][name]['band_power_ratio_db']
            log(f'{phase} {name}: band-power ratio = {value:.3f} dB (proxy)' if value is not None else f'{phase} {name}: band-power ratio undefined (zero power)')
        del tensors, original, processed, tensor

    # Share one absolute colour scale across all four panels so amplitude changes remain visible
    floor = np.finfo(np.float64).tiny
    db_maps = {phase: 10 * np.log10(np.maximum(item['power'], floor)) for phase, item in heatmaps.items()}
    upper = max(float(values.max()) for values in db_maps.values())
    lower = upper - args.heatmap_range_db
    fig, axes = plt.subplots(2, 2, figsize = (12, 7), constrained_layout = True, sharey = True)
    for column, phase in enumerate(('train', 'test')):
        item = heatmaps[phase]
        for row, label in enumerate(('Original', 'Mode A')):
            ax = axes[row, column]
            # Explicit cell edges also support a run containing just one spectral window
            time_edges = np.r_[item['times'] - hop / (2 * args.sampling_hz), item['times'][-1] + hop / (2 * args.sampling_hz)]
            frequency_edges = np.r_[item['frequencies'] - args.sampling_hz / (2 * window), item['frequencies'][-1] + args.sampling_hz / (2 * window)]
            mesh = ax.pcolormesh(time_edges, frequency_edges, db_maps[phase][row], shading = 'flat',
                                 cmap = 'magma', vmin = lower, vmax = upper, rasterized = args.rasterise_pdf)
            ax.axhline(low, color = 'cyan', linestyle = '--', linewidth = 0.7)
            ax.axhline(high, color = 'cyan', linestyle = '--', linewidth = 0.7)
            ax.set_ylim(0, args.sampling_hz / 2)
            ax.set_title(f'{label} — {phase}')
            ax.set_xlabel('Elapsed time within a run (s)')
            ax.set_ylabel('Frequency (Hz)')
    fig.suptitle('Mean within-run spectral power; dashed lines mark the assumed band')
    fig.colorbar(mesh, ax = axes.ravel().tolist(), extend = 'min', label = 'PSD (dB relative to 1 signal-unit²/Hz)')
    heat_path = folder / f'mode_A_{stamp}_time_frequency'
    for extension in ('png', 'pdf'):
        fig.savefig(heat_path.with_suffix('.' + extension), dpi = args.figure_dpi)
    plt.close(fig)

    # A paired spectrum makes frequency-selective attenuation easier to see than colour alone
    fig, axes = plt.subplots(1, 2, figsize = (11, 4), constrained_layout = True)
    for ax, phase in zip(axes, ('train', 'test')):
        item = spectra[phase]
        for index, label in enumerate(('Original', 'Mode A')):
            ax.plot(item['frequencies'][1:], 10 * np.log10(np.maximum(item['power'][index, 1:], floor)), label = label)
        ax.axvspan(low, high, color = 'grey', alpha = 0.15)
        ax.set(title = phase, xlabel = 'Frequency (Hz)', ylabel = 'PSD (dB relative to 1 signal-unit²/Hz)')
        ax.grid(alpha = 0.2)
        ax.legend()
    psd_path = folder / f'mode_A_{stamp}_spectra'
    for extension in ('png', 'pdf'):
        fig.savefig(psd_path.with_suffix('.' + extension), dpi = args.figure_dpi)
    plt.close(fig)
    # Save plotted numerical arrays for reproducibility without expanding JSON with large matrices
    arrays = {f'{phase}_{key}': value for phase, item in heatmaps.items() for key, value in item.items()}
    arrays.update({f'{phase}_psd_{key}': value for phase, item in spectra.items() for key, value in item.items()})
    array_path = folder / f'mode_A_{stamp}_spectra.npz'
    np.savez_compressed(array_path, **arrays)
    report['outputs'] = {'heatmap': str(heat_path), 'spectra': str(psd_path), 'arrays': str(array_path)}
    json_path = folder / f'mode_A_{stamp}_metrics.json'
    save_json(json_path, report)
    log(f'Saved metrics, figures and numerical spectra to {folder}', 'OK')


if __name__ == '__main__':
    # Keep all defaults in the setup zone; the launcher passes the chosen settings explicitly
    parser = argparse.ArgumentParser(description = 'Compare original and mode-A packed fMRI on CPU')
    parser.add_argument('--original-train', default = ORIGINAL_TRAIN)
    parser.add_argument('--original-test', default = ORIGINAL_TEST)
    parser.add_argument('--processed-train', default = PROCESSED_TRAIN)
    parser.add_argument('--processed-test', default = PROCESSED_TEST)
    parser.add_argument('--output-dir', default = OUTPUT_DIR)
    parser.add_argument('--sampling-hz', type = float, default = SAMPLING_HZ)
    parser.add_argument('--train-run-lengths', default = TRAIN_RUN_LENGTHS)
    parser.add_argument('--test-run-lengths', default = TEST_RUN_LENGTHS)
    parser.add_argument('--exclude-start-trs', type = int, default = EXCLUDE_START_TRS)
    parser.add_argument('--exclude-end-trs', type = int, default = EXCLUDE_END_TRS)
    parser.add_argument('--signal-band-hz', type = float, nargs = 2, default = SIGNAL_BAND_HZ)
    parser.add_argument('--spectrogram-window-sec', type = float, default = SPECTROGRAM_WINDOW_SEC)
    parser.add_argument('--spectrogram-hop-sec', type = float, default = SPECTROGRAM_HOP_SEC)
    parser.add_argument('--voxel-chunk', type = int, default = VOXEL_CHUNK)
    parser.add_argument('--cpu-threads', type = int, default = CPU_THREADS)
    parser.add_argument('--rasterise-pdf', action = argparse.BooleanOptionalAction, default = RASTERISE_PDF)
    parser.add_argument('--heatmap-range-db', type = float, default = HEATMAP_RANGE_DB)
    parser.add_argument('--figure-dpi', type = int, default = FIGURE_DPI)
    parser.add_argument('--log-every', type = int, default = LOG_EVERY)
    main(parser.parse_args())
