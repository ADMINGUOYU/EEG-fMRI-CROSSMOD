#!/usr/bin/env bash
# Run from the project root. Edit this configuration block to configure the run.
set -euo pipefail

# [SET UP INFO] ==========================
PYTHON="python"
DB_PATH="./data/CineBrain_database/lmdb"
CAPTIONS_PATH="./data/CineBrain/captions-qwen-2.5-vl-7b.json"
CACHE_DIR="./data/cache/huggingface"
HF_ENDPOINT="https://huggingface.co"
LOCAL_FILES_ONLY=false
RESULTS_ROOT="./results"
SUBJECT_ID=1
FMRI_ROI="visual"
DEVICE="cuda"
WORKERS=4
SEED=42
MAX_SAMPLES="" # empty selects the whole split
PHASE="train"
BATCH_SIZE=16
EPOCHS=50
MAX_STEPS=""
LR=0.0001
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION=1
MAX_GRAD_NORM=1.0
SAVE_EVERY=500
PRECISION="bf16"
RESUME="" # optional checkpoint; reconstruction recovers architecture from it
SIGLIP_MODEL="google/siglip2-so400m-patch14-384"
TEACHER_FRAME_BATCH_SIZE=16
NORMALISE_CONTRASTIVE=false
ENCODER_WIDTH=2048
ENCODER_LAYERS=12
ENCODER_HEADS=32
# =======================================

# Require the project root as the working directory; do not change it silently.
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../../../../.." && pwd)"
if [[ "$(pwd -P)" != "$PROJECT_ROOT" ]]; then
    printf '\033[1;31m[CineSync] ERROR\033[0m: Run this script from the project root: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

# Pass every configured value explicitly; the copied launcher records the run settings.
ARGS=(
    --db-path "$DB_PATH" --captions-path "$CAPTIONS_PATH"
    --cache-dir "$CACHE_DIR" --hf-endpoint "$HF_ENDPOINT" --results-root "$RESULTS_ROOT"
    --subject-id "$SUBJECT_ID" --fmri-roi "$FMRI_ROI" --phase "$PHASE"
    --device "$DEVICE" --workers "$WORKERS" --seed "$SEED" --batch-size "$BATCH_SIZE"
    --launch-script "$SCRIPT_PATH"
)
if [[ "$LOCAL_FILES_ONLY" == true ]]; then ARGS+=(--local-files-only); fi
if [[ -n "$MAX_SAMPLES" ]]; then ARGS+=(--max-samples "$MAX_SAMPLES"); fi
ARGS+=(--epochs "$EPOCHS" --lr "$LR" --weight-decay "$WEIGHT_DECAY"
       --gradient-accumulation "$GRADIENT_ACCUMULATION" --max-grad-norm "$MAX_GRAD_NORM"
       --save-every "$SAVE_EVERY" --precision "$PRECISION")
if [[ -n "$MAX_STEPS" ]]; then ARGS+=(--max-steps "$MAX_STEPS"); fi
if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi
ARGS+=(--siglip-model "$SIGLIP_MODEL" --teacher-frame-batch-size "$TEACHER_FRAME_BATCH_SIZE"
       --encoder-width "$ENCODER_WIDTH" --encoder-layers "$ENCODER_LAYERS" --encoder-heads "$ENCODER_HEADS")
if [[ "$NORMALISE_CONTRASTIVE" == true ]]; then ARGS+=(--normalise-contrastive); fi
exec "$PYTHON" -m models.CineSync.models.CineSync_fMRI.python_scrips.train_align "${ARGS[@]}"
