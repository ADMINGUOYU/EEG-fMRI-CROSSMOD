#!/usr/bin/env bash
# file: run.sh
# Run from the project root; configure v2 training or inference here

set -euo pipefail

# [SETUP] ===============================================================
PYTHON="python"                            # Python executable in the project environment
GPU_DEVICES="0"                            # Physical GPU ID exposed to this single-GPU run
DEVICE_TYPE="cuda"                         # Compute device: cuda or cpu

# Action and dataset; training fits the deconvolver directly
ACTION="train"                             # train or infer
DATA_ROOT="./data/cc2017"                  # Directory containing the cc2017 train/test fMRI tensors
SUBJECT=1                                  # Subject ID: 1, 2 or 3
# Comma-separated TR counts for the 18 training movie runs
TRAIN_RUN_LENGTHS="240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240,240"
TEST_RUN_LENGTHS="240,240,240,240,240"     # Comma-separated TR counts for the five testing movie runs
VALIDATION_RUNS="17,18"                    # One-based training run IDs reserved for validation
STORED_DELAY_SEC=4.0                       # Acquisition delay already included in the saved stimulus pairing, in seconds

# Temporal windows; defaults retain a full 30-second causal HRF history
TR_SEC=2.0                                 # Seconds between consecutive fMRI volumes
WINDOW_TRS=32                              # Number of consecutive volumes in each input window
WINDOW_STRIDE=4                            # Step between window starts, in TRs
BURNIN_TRS=15                              # Leading TRs excluded from losses and export to allow HRF history
LOOKAHEAD_TRS=4                            # Trailing TRs excluded from losses and export, not a receptive-field limit
TSR_FACTOR=4                               # Activity samples per TR; 1 retains the original temporal grid

# Forward physics and self-supervised losses
HRF_LENGTH_SEC=30.0                        # Duration of the causal HRF kernel, in seconds
HRF_MODE="canonical"                       # canonical or dilated_learnable for bounded HRF timing adjustment
HRF_DILATION_MIN=0.8                       # Lower bound on learnable HRF time dilation
HRF_DILATION_MAX=1.2                       # Upper bound on learnable HRF time dilation
LAMBDA_TV=0.01                             # Temporal total-variation penalty weight
SIGNED_ACTIVATIONS=true                    # Allow negative latent activity; false applies a non-negative constraint
LEARN_BASELINE=true                        # Learn a per-voxel additive BOLD baseline in the observation model
ENABLE_SPATIAL_SPSF=false                  # Apply the supplied spatial blur operator during HRF reblurring
SPATIAL_MATRIX_PATH=""                     # Optional Torch spatial matrix with verified voxel order and FWHM metadata
SPSF_FWHM_MM=2.5                           # Expected spatial blur width in millimetres; must match matrix metadata

# Temporal architecture; spatial blur requires all voxels during training
WIDTH=16                                   # Channels in the first temporal U-Net level
DEPTH=2                                    # Number of temporal downsampling levels in each U-Net
KERNEL_SIZE=3                              # Odd temporal convolution width, at least 3
VOXEL_BATCH_SIZE=256                       # Voxels per temporal processing chunk to control memory use

# Optional supervised initialisation on synthetic block activity
PRETRAIN_SYNTHETIC=true                    # Initialise the deconvolver using synthetic activity
SYNTHETIC_STEPS=500                        # Number of synthetic pre-training optimiser steps
SYNTHETIC_MIN_SHOT_SEC=2.0                 # Shortest synthetic constant-activity block, in seconds
SYNTHETIC_MAX_SHOT_SEC=8.0                 # Longest synthetic constant-activity block, in seconds
SYNTHETIC_AMPLITUDE=1.0                    # Standard deviation of sampled synthetic block amplitudes

# Checkpoints and inference: choose "last.pt" or "best.pt" explicitly
INFER_CHECKPOINT=""                        # Trained checkpoint to load for inference
RESUME=""                                  # Training checkpoint restoring weights, optimiser, scheduler and RNG states
INFER_PHASE="test"                         # Inference data split: train or test
INFER_AVERAGE_REPEATS=true                 # Average repeated presentations before inference

# Optimisation and run management
RESULTS_ROOT="./results"                   # Base directory for run outputs, logs and checkpoints
SEED=42                                    # Random seed for reproducible sampling and initialisation
WORKERS=0                                  # DataLoader worker processes; 0 loads in the main process
BATCH_SIZE=2                               # Temporal windows per DataLoader batch
EPOCHS=30                                  # Total training epochs, including any restored epochs
LR=0.0001                                  # Peak learning rate for AdamW
WEIGHT_DECAY=0.01                          # AdamW weight-decay coefficient
WARMUP_FRACTION=0.05                       # Fraction of optimiser steps used for learning-rate warm-up
MINIMUM_LR_RATIO=0.05                      # Final cosine-schedule learning rate as a fraction of LR
MAX_GRAD_NORM=1.0                          # Gradient-norm clipping threshold; 0 disables clipping
LOG_EVERY=20                               # Progress reporting interval in batches or synthetic steps
SAVE_EVERY=5                               # Epoch interval for numbered checkpoints; last and best are saved separately
# =======================================================================

# Preserve the project-root working directory used by all package paths.
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
if [[ ! -f "models/SRES_fMRI_cc2017/models/v2/config.py" ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Run from the project root.\n' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export PYTHONUNBUFFERED=1
if [[ "$DEVICE_TYPE" == cuda && ( -z "$GPU_DEVICES" || "$GPU_DEVICES" == *,* ) ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Select one GPU for this launcher.\n' >&2
    exit 1
fi

# Pass every setting explicitly; the copied launcher records the experiment.
ARGS=(--device "$DEVICE_TYPE" --launch-script "$SCRIPT_PATH")
ARGS+=(--action "$ACTION")
ARGS+=(--data-root "$DATA_ROOT")
ARGS+=(--subject "$SUBJECT")
ARGS+=(--train-run-lengths "$TRAIN_RUN_LENGTHS")
ARGS+=(--test-run-lengths "$TEST_RUN_LENGTHS")
ARGS+=(--validation-runs "$VALIDATION_RUNS")
ARGS+=(--stored-delay-sec "$STORED_DELAY_SEC")
ARGS+=(--tr-sec "$TR_SEC")
ARGS+=(--window-trs "$WINDOW_TRS")
ARGS+=(--window-stride "$WINDOW_STRIDE")
ARGS+=(--burnin-trs "$BURNIN_TRS")
ARGS+=(--lookahead-trs "$LOOKAHEAD_TRS")
ARGS+=(--tsr-factor "$TSR_FACTOR")
ARGS+=(--hrf-length-sec "$HRF_LENGTH_SEC")
ARGS+=(--hrf-mode "$HRF_MODE")
ARGS+=(--hrf-dilation-min "$HRF_DILATION_MIN")
ARGS+=(--hrf-dilation-max "$HRF_DILATION_MAX")
ARGS+=(--lambda-tv "$LAMBDA_TV")
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
ARGS+=(--checkpoint "$INFER_CHECKPOINT")
ARGS+=(--resume "$RESUME")
ARGS+=(--phase "$INFER_PHASE")
if [[ "$INFER_AVERAGE_REPEATS" == true ]]; then ARGS+=(--average-repeats); else ARGS+=(--no-average-repeats); fi
ARGS+=(--results-root "$RESULTS_ROOT")
ARGS+=(--seed "$SEED")
ARGS+=(--workers "$WORKERS")
ARGS+=(--batch-size "$BATCH_SIZE")
ARGS+=(--epochs "$EPOCHS")
ARGS+=(--lr "$LR")
ARGS+=(--weight-decay "$WEIGHT_DECAY")
ARGS+=(--warmup-fraction "$WARMUP_FRACTION")
ARGS+=(--minimum-lr-ratio "$MINIMUM_LR_RATIO")
ARGS+=(--max-grad-norm "$MAX_GRAD_NORM")
ARGS+=(--log-every "$LOG_EVERY")
ARGS+=(--save-every "$SAVE_EVERY")

case "$ACTION" in
    train) MODULE="train" ;;
    infer) MODULE="infer" ;;
    *) printf "Unknown ACTION: %s\n" "$ACTION" >&2; exit 1 ;;
esac
exec "$PYTHON" -m "models.SRES_fMRI_cc2017.models.v2.$MODULE" "${ARGS[@]}"
