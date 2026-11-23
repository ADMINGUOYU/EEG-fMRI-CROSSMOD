#!/usr/bin/env bash
# file: eval_mode_A.sh

set -euo pipefail

# [SETUP] ===============================================================
PYTHON="python"                            # Python executable in the project environment

ORIGINAL_TRAIN="./data/cc2017/subj01_train_fmri.pt"
ORIGINAL_TEST="./data/cc2017/subj01_test_fmri.pt"
PROCESSED_TRAIN="./data/cc2017_SRES_mode_A/subj01_train_fmri.pt"
PROCESSED_TEST="./data/cc2017_SRES_mode_A/subj01_test_fmri.pt"
OUTPUT_DIR="./results/SRES_fMRI_cc2017/mode_A_comparison"

SAMPLING_HZ=0.5                            # Original (scanner) sampling rate
TRAIN_RUN_LENGTHS="240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240"
TEST_RUN_LENGTHS="240,240,240,240,240"     # Volume counts for each run, preserving boundaries

EXCLUDE_START_TRS=15                       # Match packing burn-in; whole-file MSE is also reported
EXCLUDE_END_TRS=4                          # Match packing trailing exclusion

SIGNAL_BAND_HZ=(0.01 0.1)                  # Assumed band for a spectral proxy, not identified neural signal (BOLD response band); SNR estimator
SPECTROGRAM_WINDOW_SEC=64                  # Use 30 for an HRF-scale view with coarser frequency resolution
SPECTROGRAM_HOP_SEC=8                      # Step between local spectral windows
VOXEL_CHUNK=256                            # Voxels processed at once
CPU_THREADS=4                              # Torch CPU threads
RASTERISE_PDF=true                         # Rasterise heatmaps (PDF graphics)
HEATMAP_RANGE_DB=60                        # Shared colour range below the maximum PSD, in dB
FIGURE_DPI=160                             # PNG resolution and rasterised PDF resolution
LOG_EVERY=20                               # Progress interval in voxel chunks
# =======================================================================

# Limit numerical-library threads and keep paths relative to the project root
if [[ ! -f "models/SRES_fMRI_cc2017/models/v1/config.py" ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Run from the project root\n' >&2
    exit 1
fi
export OMP_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export PYTHONUNBUFFERED=1
ARGS=(--original-train "$ORIGINAL_TRAIN" --original-test "$ORIGINAL_TEST"
      --processed-train "$PROCESSED_TRAIN" --processed-test "$PROCESSED_TEST" --output-dir "$OUTPUT_DIR"
      --sampling-hz "$SAMPLING_HZ" --train-run-lengths "$TRAIN_RUN_LENGTHS" --test-run-lengths "$TEST_RUN_LENGTHS"
      --exclude-start-trs "$EXCLUDE_START_TRS" --exclude-end-trs "$EXCLUDE_END_TRS"
      --signal-band-hz "${SIGNAL_BAND_HZ[@]}" --spectrogram-window-sec "$SPECTROGRAM_WINDOW_SEC"
      --spectrogram-hop-sec "$SPECTROGRAM_HOP_SEC" --voxel-chunk "$VOXEL_CHUNK"
      --cpu-threads "$CPU_THREADS" --heatmap-range-db "$HEATMAP_RANGE_DB" --figure-dpi "$FIGURE_DPI" --log-every "$LOG_EVERY")
if [[ "$RASTERISE_PDF" == true ]]; then
    ARGS+=(--rasterise-pdf)
else
    ARGS+=(--no-rasterise-pdf)
fi
exec "$PYTHON" -m models.SRES_fMRI_cc2017.models.v1.NEURONS_adapter.eval_mode_A "${ARGS[@]}"
