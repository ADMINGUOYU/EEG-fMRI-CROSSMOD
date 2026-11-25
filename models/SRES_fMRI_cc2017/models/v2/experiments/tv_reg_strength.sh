#!/usr/bin/env bash
# file: tv_reg_strength.sh
# Run the v2 tv reg strength experiment

set -euo pipefail

# [SETUP] ===============================================================
PYTHON="python"                             # Python executable in the project environment
GPU_DEVICES="0"                             # Physical GPU for this single-GPU experiment
DATA_ROOT="./data/cc2017"                   # Original fMRI tensors !!! never a packed model output !!!
SUBJECT=1                                   # Subject ID: 1, 2 or 3
TRAIN_RUN_LENGTHS="240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240"
                                            # Original TR counts per training run
TEST_RUN_LENGTHS="240,240,240,240,240"      # Original TR counts per testing run
VALIDATION_RUNS="17,18"                     # Held-out training runs; test data never selects weights
TR_SEC=2.0                                  # Original sampling interval: 2 seconds = 0.5 Hz

INPUT_DOWNSAMPLE_FACTOR=2                   # Retain run-local rows 0, D, 2D, ... without filtering
TSR_FACTOR=4                                # Activity samples per retained input; divisible by D

WINDOW_TRS=32                               # Original TRs per window; divisible by D
WINDOW_STRIDE=4                             # Original TRs between windows; divisible by D
BURNIN_TRS=15                               # Original leading TRs excluded; rounded upwards to a retained TR
LOOKAHEAD_TRS=4                             # Original trailing TRs excluded; rounded upwards to a retained TR
AVERAGE_REPEATS=false                       # Apply the same repetition policy to fitting and evaluation
HRF_LENGTH_SEC=30.0                         # Causal HRF support in physical seconds
HRF_MODE="canonical"                        # canonical or dilated_learnable
HRF_DILATION_MIN=0.8                        # Lower bound on optional learned dilation
HRF_DILATION_MAX=1.2                        # Upper bound on optional learned dilation
TV_STRENGTHS="0,0.001,0.01,0.1"             # Activity TV weights, including the zero-weight control
SIGNED_ACTIVATIONS=true                     # Allow negative activity
LEARN_BASELINE=true                         # Learn a time-constant BOLD baseline per voxel
ENABLE_SPATIAL_SPSF=false                   # Apply the optional spatial observation operator
SPATIAL_MATRIX_PATH=""                      # Verified voxel-order sPSF matrix
SPSF_FWHM_MM=2.5                            # Expected sPSF width
WIDTH=16                                    # Initial U-Net channel count
DEPTH=2                                     # Temporal pooling levels
KERNEL_SIZE=3                               # Odd temporal convolution width
VOXEL_BATCH_SIZE=256                        # Sampled training voxels and inference/metric chunk size
PRETRAIN_SYNTHETIC=true                     # Initialise from noise-free synthetic BOLD, not omitted data
SYNTHETIC_STEPS=500                         # Synthetic inverse updates before real-data fitting
SYNTHETIC_MIN_SHOT_SEC=2.0                  # Shortest synthetic activity block
SYNTHETIC_MAX_SHOT_SEC=8.0                  # Longest synthetic activity block
SYNTHETIC_AMPLITUDE=1.0                     # Synthetic activity amplitude standard deviation
EPOCHS=30                                   # Real-data training epochs
LR=0.0001                                   # AdamW peak learning rate
WEIGHT_DECAY=0.01                           # AdamW regularisation
WARMUP_FRACTION=0.05                        # Fraction of updates used for warm-up
MINIMUM_LR_RATIO=0.05                       # Final cosine learning-rate fraction
MAX_GRAD_NORM=1.0                           # Gradient clipping; zero disables it
BATCH_SIZE=2                                # Windows per batch
WORKERS=0                                   # DataLoader workers
DEVICE_TYPE="cuda"                          # Logical CUDA device or cpu
SEEDS="42"                                  # Comma-separated paired initialisation seeds
LOG_EVERY=20                                # Progress interval in batches
SAVE_EVERY=5                                # Numbered checkpoint interval
EVALUATION_CHECKPOINT="best"                # best validation prediction or last epoch
OUTPUT_ROOT="./results/SRES_fMRI_cc2017/v2/experiments"
SAVE_RECONSTRUCTIONS=true                   # Save per-run reconstructed BOLD and validity masks
PLOT_RUN=1                                  # One-based run to illustrate in each split
PLOT_VOXELS="0,1,2"                         # Zero-based voxel indices for example traces
PLOT_MAX_TRS=120                            # Original TRs shown per example
# =======================================================================

# Keep relative paths rooted at the project and select one physical GPU
if [[ ! -f "models/SRES_fMRI_cc2017/models/v2/config.py" ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Run from the project root\n' >&2
    exit 1
fi
if [[ "$DEVICE_TYPE" == cuda && ( -z "$GPU_DEVICES" || "$GPU_DEVICES" == *,* ) ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Select one physical GPU\n' >&2
    exit 1
fi

# export GPU selection
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
# export unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# fetch bash script path
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"

# Pass each setup value explicitly and preserve the launcher with the experiment
ARGS=(--launch-script "$SCRIPT_PATH")
ARGS+=(--data-root "$DATA_ROOT")
ARGS+=(--subject "$SUBJECT")
ARGS+=(--train-run-lengths "$TRAIN_RUN_LENGTHS")
ARGS+=(--test-run-lengths "$TEST_RUN_LENGTHS")
ARGS+=(--validation-runs "$VALIDATION_RUNS")
ARGS+=(--tr-sec "$TR_SEC")
ARGS+=(--input-downsample-factor "$INPUT_DOWNSAMPLE_FACTOR")
ARGS+=(--tsr-factor "$TSR_FACTOR")
ARGS+=(--window-trs "$WINDOW_TRS")
ARGS+=(--window-stride "$WINDOW_STRIDE")
ARGS+=(--burnin-trs "$BURNIN_TRS")
ARGS+=(--lookahead-trs "$LOOKAHEAD_TRS")
if [[ "$AVERAGE_REPEATS" == true ]]; then ARGS+=(--average-repeats); else ARGS+=(--no-average-repeats); fi
ARGS+=(--hrf-length-sec "$HRF_LENGTH_SEC")
ARGS+=(--hrf-mode "$HRF_MODE")
ARGS+=(--hrf-dilation-min "$HRF_DILATION_MIN")
ARGS+=(--hrf-dilation-max "$HRF_DILATION_MAX")
ARGS+=(--tv-strengths "$TV_STRENGTHS")
if [[ "$SIGNED_ACTIVATIONS" == true ]]; then ARGS+=(--signed-activations); else ARGS+=(--no-signed-activations); fi
if [[ "$LEARN_BASELINE" == true ]]; then ARGS+=(--learn-baseline); else ARGS+=(--no-learn-baseline); fi
if [[ "$ENABLE_SPATIAL_SPSF" == true ]]; then ARGS+=(--enable-spatial-spsf); else ARGS+=(--no-enable-spatial-spsf); fi
ARGS+=(--spatial-matrix-path "$SPATIAL_MATRIX_PATH")
ARGS+=(--spsf-fwhm-mm "$SPSF_FWHM_MM")
ARGS+=(--width "$WIDTH")
ARGS+=(--depth "$DEPTH")
ARGS+=(--kernel-size "$KERNEL_SIZE")
ARGS+=(--voxel-batch-size "$VOXEL_BATCH_SIZE")
if [[ "$PRETRAIN_SYNTHETIC" == true ]]; then ARGS+=(--pretrain-synthetic); else ARGS+=(--no-pretrain-synthetic); fi
ARGS+=(--synthetic-steps "$SYNTHETIC_STEPS")
ARGS+=(--synthetic-min-shot-sec "$SYNTHETIC_MIN_SHOT_SEC")
ARGS+=(--synthetic-max-shot-sec "$SYNTHETIC_MAX_SHOT_SEC")
ARGS+=(--synthetic-amplitude "$SYNTHETIC_AMPLITUDE")
ARGS+=(--epochs "$EPOCHS")
ARGS+=(--lr "$LR")
ARGS+=(--weight-decay "$WEIGHT_DECAY")
ARGS+=(--warmup-fraction "$WARMUP_FRACTION")
ARGS+=(--minimum-lr-ratio "$MINIMUM_LR_RATIO")
ARGS+=(--max-grad-norm "$MAX_GRAD_NORM")
ARGS+=(--batch-size "$BATCH_SIZE")
ARGS+=(--workers "$WORKERS")
ARGS+=(--device "$DEVICE_TYPE")
ARGS+=(--seeds "$SEEDS")
ARGS+=(--log-every "$LOG_EVERY")
ARGS+=(--save-every "$SAVE_EVERY")
ARGS+=(--evaluation-checkpoint "$EVALUATION_CHECKPOINT")
ARGS+=(--output-root "$OUTPUT_ROOT")
if [[ "$SAVE_RECONSTRUCTIONS" == true ]]; then ARGS+=(--save-reconstructions); else ARGS+=(--no-save-reconstructions); fi
ARGS+=(--plot-run "$PLOT_RUN")
ARGS+=(--plot-voxels "$PLOT_VOXELS")
ARGS+=(--plot-max-trs "$PLOT_MAX_TRS")
exec "$PYTHON" -m models.SRES_fMRI_cc2017.models.v2.experiments.tv_reg_strength "${ARGS[@]}"
