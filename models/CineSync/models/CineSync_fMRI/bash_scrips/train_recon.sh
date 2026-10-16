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
DEVICE_TYPE="cuda"    # device type only; select GPU indices or UUIDs below
GPU_DEVICES="0,1,2,3" # one GPU per process, in local-rank order
WORKERS=4
SEED=42
MAX_SAMPLES="" # empty selects the whole split
PHASE="train"
BATCH_SIZE=2
EPOCHS=100000
MAX_STEPS="5000"
LR=0.0001
LR_DECAY_STYLE="linear" # constant, linear or cosine
LR_DECAY_STEPS="" # empty uses the planned number of optimiser updates
LR_WARMUP=0.01 # fraction of the decay horizon
LR_DECAY_RATIO=0.1 # cosine minimum as a fraction of LR
LR_AUTO_WARMUP_STEPS=100 # set to zero to disable the initial fine-tuning cap
LR_AUTO_WARMUP_RATE=0.05
WEIGHT_DECAY=0.01
GRADIENT_ACCUMULATION=1
MAX_GRAD_NORM=1.0
SAVE_EVERY=500
PRECISION="bf16"
RESUME="" # optional checkpoint; reconstruction recovers architecture from it
ALIGN_CHECKPOINT="./results/CineSync/CineSync_fMRI/align/<actual path>/checkpoints/last.pt"
COGVIDEO_MODEL="THUDM/CogVideoX-5b"
LORA_RANK=64
LORA_ALPHA=64
CONDITION_DROPOUT=0.1
HEIGHT=480
WIDTH=720
NPROC=4
# =======================================

# Require the project root as the working directory; do not change it silently.
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../../../../.." && pwd)"
if [[ "$(pwd -P)" != "$PROJECT_ROOT" ]]; then
    printf '\033[1;31m[CineSync] ERROR\033[0m: Run this script from the project root: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

# Select physical GPUs consistently for single- and multi-GPU runs.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
if [[ "$DEVICE_TYPE" == cuda ]]; then
    IFS=',' read -r -a SELECTED_GPUS <<< "$GPU_DEVICES"
    if [[ -z "$GPU_DEVICES" || "${#SELECTED_GPUS[@]}" -ne "$NPROC" ]]; then
        printf '\033[1;31m[CineSync] ERROR\033[0m: Set one GPU_DEVICES entry per process.\n' >&2
        exit 1
    fi
    printf '\033[1;34m[CineSync] INFO\033[0m: Using GPUs %s.\n' "$CUDA_VISIBLE_DEVICES"
fi

if [[ "$NPROC" -gt 1 && "$DEVICE_TYPE" != cuda ]]; then
    printf '\033[1;31m[CineSync] ERROR\033[0m: Multi-GPU training requires DEVICE_TYPE=cuda.\n' >&2
    exit 1
fi

# Pass every configured value explicitly; the copied launcher records the run settings.
ARGS=(
    --db-path "$DB_PATH" --captions-path "$CAPTIONS_PATH"
    --cache-dir "$CACHE_DIR" --hf-endpoint "$HF_ENDPOINT" --results-root "$RESULTS_ROOT"
    --subject-id "$SUBJECT_ID" --fmri-roi "$FMRI_ROI" --phase "$PHASE"
    --device "$DEVICE_TYPE" --workers "$WORKERS" --seed "$SEED" --batch-size "$BATCH_SIZE"
    --launch-script "$SCRIPT_PATH"
)
if [[ "$LOCAL_FILES_ONLY" == true ]]; then ARGS+=(--local-files-only); fi
if [[ -n "$MAX_SAMPLES" ]]; then ARGS+=(--max-samples "$MAX_SAMPLES"); fi
ARGS+=(--epochs "$EPOCHS" --lr "$LR" --weight-decay "$WEIGHT_DECAY"
       --lr-decay-style "$LR_DECAY_STYLE" --lr-warmup "$LR_WARMUP" --lr-decay-ratio "$LR_DECAY_RATIO"
       --lr-auto-warmup-steps "$LR_AUTO_WARMUP_STEPS" --lr-auto-warmup-rate "$LR_AUTO_WARMUP_RATE"
       --gradient-accumulation "$GRADIENT_ACCUMULATION" --max-grad-norm "$MAX_GRAD_NORM"
       --save-every "$SAVE_EVERY" --precision "$PRECISION")
if [[ -n "$LR_DECAY_STEPS" ]]; then ARGS+=(--lr-decay-steps "$LR_DECAY_STEPS"); fi
if [[ -n "$MAX_STEPS" ]]; then ARGS+=(--max-steps "$MAX_STEPS"); fi
if [[ -n "$RESUME" ]]; then ARGS+=(--resume "$RESUME"); fi
ARGS+=(--align-checkpoint "$ALIGN_CHECKPOINT" --cogvideo-model "$COGVIDEO_MODEL"
       --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --condition-dropout "$CONDITION_DROPOUT"
       --height "$HEIGHT" --width "$WIDTH")
if [[ "$NPROC" -gt 1 ]]; then
    exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" -m models.CineSync.models.CineSync_fMRI.python_scrips.train_recon "${ARGS[@]}"
fi
exec "$PYTHON" -m models.CineSync.models.CineSync_fMRI.python_scrips.train_recon "${ARGS[@]}"
