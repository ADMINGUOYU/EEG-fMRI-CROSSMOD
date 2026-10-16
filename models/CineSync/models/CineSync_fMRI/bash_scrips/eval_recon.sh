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
DEVICE_TYPE="cuda" # device type only; select a GPU index or UUID below
GPU_DEVICES="0"    # one GPU for evaluation
WORKERS=4
SEED=42
MAX_SAMPLES="" # empty selects the whole split
PHASE="test"
BATCH_SIZE=1
CHECKPOINT="./results/CineSync/CineSync_fMRI/recon/<actual path>/checkpoints/last.pt"
COGVIDEO_MODEL="" # empty uses the base model recorded in the checkpoint
STEPS=50
GUIDANCE_SCALE=6
SEMANTIC_METRICS=true
CLIP_MODEL="openai/clip-vit-base-patch32"
DINO_MODEL="facebook/dinov2-base"
VIDEO_METRIC_MODEL="MCG-NJU/videomae-base-finetuned-kinetics"
I3D_TORCHSCRIPT="" # provide a validated feature-only checkpoint to report FVD
SAVE_ARRAYS=false
# =======================================

# Require the project root as the working directory; do not change it silently.
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../../../../.." && pwd)"
if [[ "$(pwd -P)" != "$PROJECT_ROOT" ]]; then
    printf '\033[1;31m[CineSync] ERROR\033[0m: Run this script from the project root: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

# Select one physical GPU for evaluation.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
if [[ "$DEVICE_TYPE" == cuda ]]; then
    IFS=',' read -r -a SELECTED_GPUS <<< "$GPU_DEVICES"
    if [[ -z "$GPU_DEVICES" || "${#SELECTED_GPUS[@]}" -ne "1" ]]; then
        printf '\033[1;31m[CineSync] ERROR\033[0m: Set exactly one GPU_DEVICES entry for evaluation.\n' >&2
        exit 1
    fi
    printf '\033[1;34m[CineSync] INFO\033[0m: Using GPUs %s.\n' "$CUDA_VISIBLE_DEVICES"
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
ARGS+=(--checkpoint "$CHECKPOINT")
ARGS+=(--steps "$STEPS" --guidance-scale "$GUIDANCE_SCALE" --clip-model "$CLIP_MODEL"
       --dino-model "$DINO_MODEL" --video-metric-model "$VIDEO_METRIC_MODEL")
if [[ "$SEMANTIC_METRICS" == true ]]; then ARGS+=(--semantic-metrics); else ARGS+=(--no-semantic-metrics); fi
if [[ "$SAVE_ARRAYS" == true ]]; then ARGS+=(--save-arrays); fi
if [[ -n "$COGVIDEO_MODEL" ]]; then ARGS+=(--cogvideo-model "$COGVIDEO_MODEL"); fi
if [[ -n "$I3D_TORCHSCRIPT" ]]; then ARGS+=(--i3d-torchscript "$I3D_TORCHSCRIPT"); fi
exec "$PYTHON" -m models.CineSync.models.CineSync_fMRI.python_scrips.eval_recon "${ARGS[@]}"
