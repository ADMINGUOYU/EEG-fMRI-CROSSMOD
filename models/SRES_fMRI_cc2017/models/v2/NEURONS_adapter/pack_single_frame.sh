#!/usr/bin/env bash
# file: pack_single_frame.sh
# Infer and pack v2 activity at the original TR (same temporal resolution as before)

# To remove the generated directory safely,
# use its actual path WITHOUT a trailing slash:
#   rm -ri -- "${OUTPUT_DIR}"
# This removes symlinks themselves, leaving their original targets untouched
# Do not delete paths inside linked directories, for example ${OUTPUT_DIR}/masks/*

set -euo pipefail

# [SETUP] ===============================================================
PYTHON="python"                            # Python executable in the project environment

GPU_DEVICES="0"                            # Physical GPU ID for this single-GPU run
DEVICE_TYPE="cuda"                         # cuda or cpu

CHECKPOINT=""                              # Trained v2 checkpoint; subject is read from it
DATA_ROOT="./data/cc2017"                  # Original cc2017 root supplying fMRI tensors and linked assets
OUTPUT_DIR="./data/cc2017_SRES_v2_single_frame"

COLLAPSE_MODE_T="average_pool"             # Average subframes or choose a zero-based index, e.g. "0"

BATCH_SIZE=2                               # Temporal windows per batch
WORKERS=0                                  # DataLoader processes; 0 uses the main process
SEED=42                                    # Seed for reproducible initialisation
LOG_EVERY=20                               # Console progress interval in batches
# =======================================================================

# Keep relative paths rooted at the project and expose the selected GPU as cuda:0
if [[ ! -f "models/SRES_fMRI_cc2017/models/v2/config.py" ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Run from the project root\n' >&2
    exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Set CHECKPOINT in pack_single_frame.sh\n' >&2
    exit 1
fi
if [[ "$DEVICE_TYPE" == cuda && ( -z "$GPU_DEVICES" || "$GPU_DEVICES" == *,* ) ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Select one physical GPU\n' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export PYTHONUNBUFFERED=1
ARGS=(--checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR"
      --collapse-mode-t "$COLLAPSE_MODE_T" --device "$DEVICE_TYPE" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
      --seed "$SEED" --log-every "$LOG_EVERY")
exec "$PYTHON" -m models.SRES_fMRI_cc2017.models.v2.NEURONS_adapter.pack_single_frame "${ARGS[@]}"
