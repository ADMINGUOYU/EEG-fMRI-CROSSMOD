#!/usr/bin/env bash
# Run from the project root. Edit this configuration block to configure the run.
set -euo pipefail

# [SET UP INFO] ==========================
PYTHON="python"
RUN_DIR="<path to run artifact directory produced by eval_recon>"
CACHE_DIR="./data/cache/huggingface"
HF_ENDPOINT="https://huggingface.co"
LOCAL_FILES_ONLY=false
DEVICE_TYPE="cuda"
GPU_DEVICES="0" # one GPU for evaluation
WORKERS=4
BATCH_SIZE=1
SEED=42
SEMANTIC_METRICS=true
CLIP_MODEL="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
DINO_MODEL="facebook/dinov2-base"
IMAGE_METRIC_MODEL="google/vit-base-patch16-224"
VIDEO_METRIC_MODEL="MCG-NJU/videomae-base-finetuned-kinetics"
I3D_TORCHSCRIPT="" # provide a validated feature-only checkpoint to report FVD
# =======================================

# Require the project root so saved dataset paths resolve as in eval_recon.
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../../../../.." && pwd)"
if [[ "$(pwd -P)" != "$PROJECT_ROOT" ]]; then
    printf '\033[1;31m[CineSync] ERROR\033[0m: Run this script from the project root: %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi

# Select one physical GPU for metric models.
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
if [[ "$DEVICE_TYPE" == cuda ]]; then
    IFS=',' read -r -a SELECTED_GPUS <<< "$GPU_DEVICES"
    if [[ -z "$GPU_DEVICES" || "${#SELECTED_GPUS[@]}" -ne "1" ]]; then
        printf '\033[1;31m[CineSync] ERROR\033[0m: Set exactly one GPU_DEVICES entry for evaluation.\n' >&2
        exit 1
    fi
    printf '\033[1;34m[CineSync] INFO\033[0m: Using GPUs %s.\n' "$CUDA_VISIBLE_DEVICES"
fi

# Dataset paths and subject/split selection come from the saved run arguments.
ARGS=(
    --run-dir "$RUN_DIR" --cache-dir "$CACHE_DIR" --hf-endpoint "$HF_ENDPOINT"
    --device "$DEVICE_TYPE" --workers "$WORKERS" --batch-size "$BATCH_SIZE" --seed "$SEED"
    --clip-model "$CLIP_MODEL" --dino-model "$DINO_MODEL"
    --image-metric-model "$IMAGE_METRIC_MODEL" --video-metric-model "$VIDEO_METRIC_MODEL"
)
if [[ "$LOCAL_FILES_ONLY" == true ]]; then ARGS+=(--local-files-only); fi
if [[ "$SEMANTIC_METRICS" == true ]]; then ARGS+=(--semantic-metrics); else ARGS+=(--no-semantic-metrics); fi
if [[ -n "$I3D_TORCHSCRIPT" ]]; then ARGS+=(--i3d-torchscript "$I3D_TORCHSCRIPT"); fi
exec "$PYTHON" -m models.CineSync.models.CineSync_fMRI.python_scrips.eval_artifact "${ARGS[@]}"
