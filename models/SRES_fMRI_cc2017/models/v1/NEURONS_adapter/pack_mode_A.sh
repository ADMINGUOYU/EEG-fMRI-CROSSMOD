#!/usr/bin/env bash
# file: pack_mode_A.sh
# Infer and pack mode-A BOLD; run this script from the project root

# To remove the generated directory safely,
# use its actual path WITHOUT a trailing slash:
#   rm -ri -- ${OUTPUT_DIR}
# This removes symlinks themselves, leaving their original targets untouched
# Do not delete paths inside linked directories, for example ${OUTPUT_DIR}/masks/*

set -euo pipefail

# [SETUP] ===============================================================
PYTHON="python"                            # Python executable in the project environment
GPU_DEVICES="0"                            # Physical GPU ID for this single-GPU run
DEVICE_TYPE="cuda"                         # cuda or cpu
CHECKPOINT=""                              # Trained stage-1 or stage-2 checkpoint; subject is read from it
DATA_ROOT="./data/cc2017"                  # Original cc2017 root supplying fMRI tensors and linked assets
OUTPUT_DIR="./data/cc2017_SRES_mode_A"     # Destination for both split tensors and supporting symlinks; existing outputs are refused
BATCH_SIZE=2                               # Temporal windows per batch
WORKERS=0                                  # DataLoader processes; 0 uses the main process
MC_TEST_SAMPLES=8                          # Noisy predictions averaged per window
SEED=42                                    # Random seed for inference noise
LOG_EVERY=20                               # Console progress interval in batches
# =======================================================================

# Keep relative paths rooted at the project and expose the selected GPU as cuda:0
if [[ ! -f "models/SRES_fMRI_cc2017/modules/config.py" ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Run from the project root\n' >&2
    exit 1
fi
if [[ -z "$CHECKPOINT" ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Set CHECKPOINT in pack_mode_A.sh\n' >&2
    exit 1
fi
if [[ "$DEVICE_TYPE" == cuda && ( -z "$GPU_DEVICES" || "$GPU_DEVICES" == *,* ) ]]; then
    printf '\033[1;31m[SRES_fMRI_cc2017] ERROR\033[0m: Select one physical GPU\n' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export PYTHONUNBUFFERED=1
ARGS=(--checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR"
      --device "$DEVICE_TYPE" --batch-size "$BATCH_SIZE" --workers "$WORKERS"
      --mc-test-samples "$MC_TEST_SAMPLES" --seed "$SEED" --log-every "$LOG_EVERY")
exec "$PYTHON" -m models.SRES_fMRI_cc2017.models.v1.NEURONS_adapter.pack_mode_A "${ARGS[@]}"
