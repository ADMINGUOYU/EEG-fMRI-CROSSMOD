#!/usr/bin/env bash
# Run from the project root after activating the project environment.
set -euo pipefail

# [SETUP] ================================================================
PYTHON="python"
GPU_DEVICES="0"
MASTER_PORT="29521"
EXPERIMENT="neurons"
STAGES="12"                 # Any combination of 1,2,3,4,5,6,e (see README).
MODE="enhance"              # Upstream enhance/motion/self combinations.
SUBJECT=1
DATA_ROOT="./data/cc2017"
WEIGHTS_ROOT="./data/cc2017/pretrained_weights"
RESULTS_ROOT="./results/NEURONS/NEURONS"
CACHE_ROOT="./data/cache"   # To configure custom paths, look down and configure the exports
SD_MODEL="stable-diffusion-v1-5/stable-diffusion-v1-5"

MODEL_NAME="testing"
SEED=42
N_BLOCKS=4
N_FRAMES=6
HIDDEN_DIM=4096
PRIOR_SCALE=30
LR_SCHEDULER="cycle"
SAVE_CHECKPOINTS=true
TENSORBOARD=true
RESUME_FROM_CKPT=""
BACKBONE_BATCH_SIZE=120
BACKBONE_EPOCHS=30
BACKBONE_MIXUP=1.0
BACKBONE_LR=5e-5
DECOUPLER_BATCH_SIZE=10
DECOUPLER_EPOCHS=50
DECOUPLER_MIXUP=0.0
DECOUPLER_LR=5e-5
RECON_BATCH_SIZE=20
BLURRY_RECON=false

VIDEO_LENGTH=16
VIDEO_WIDTH=256
VIDEO_HEIGHT=256
WITHOUT_XFORMERS=true
VIDEO_CONFIG="models/NEURONS/models/NEURONS/configs/NeuroClips/control.yaml"
INFERENCE_CONFIG="models/NEURONS/models/NEURONS/configs/inference/inference-v3.yaml"
# Optional stage-specific arguments retained by the upstream parsers.
TRAIN_EXTRA_ARGS=()
RECON_EXTRA_ARGS=()
CAPTION_EXTRA_ARGS=()
VIDEO_EXTRA_ARGS=()
METRICS_EXTRA_ARGS=()
DECOUPLED_EXTRA_ARGS=()
# =======================================================================

if [[ ! -f "models/NEURONS/models/NEURONS/train_neurons.py" ]]; then
    printf '\033[1;31m[NEURONS] ERROR\033[0m: Run this script from the project root.\n' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
export MASTER_PORT

# Delete any existing things and configure custom cache paths
export HF_HOME="$PWD/$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PWD/$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$PWD/$CACHE_ROOT"

export PYTHONUNBUFFERED=1
IFS=',' read -r -a GPU_IDS <<< "$GPU_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"
PACKAGE="models.NEURONS.models.NEURONS"
EXP_DIR="$RESULTS_ROOT/exp_$EXPERIMENT/subj_$SUBJECT"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$EXP_DIR"

# Use explicit process counts rather than a machine's saved Accelerate settings.
LAUNCH=("$PYTHON" -m accelerate.commands.launch --num_machines 1 --machine_rank 0
        --num_processes "$NUM_GPUS" --mixed_precision fp16 --main_process_port "$MASTER_PORT")
if (( NUM_GPUS > 1 )); then LAUNCH+=(--multi_gpu); fi
COMMON=(--subj "$SUBJECT" --exp_dir "$EXP_DIR" --cache_dir "$HF_HUB_CACHE")
MODEL=(--model_name "$MODEL_NAME" --root_dir "$DATA_ROOT" --weights_dir "$WEIGHTS_ROOT"
       --pretrained-model-path "$SD_MODEL" --seed "$SEED"
       --n_blocks "$N_BLOCKS" --n_frames "$N_FRAMES" --hidden_dim "$HIDDEN_DIM")
TRAIN=("${COMMON[@]}" "${MODEL[@]}" --prior_scale "$PRIOR_SCALE" --lr_scheduler_type "$LR_SCHEDULER")
if [[ "$SAVE_CHECKPOINTS" == true ]]; then TRAIN+=(--ckpt_saving); else TRAIN+=(--no-ckpt_saving); fi
if [[ "$TENSORBOARD" == true ]]; then TRAIN+=(--tensorboard); else TRAIN+=(--no-tensorboard); fi
if [[ -n "$RESUME_FROM_CKPT" ]]; then TRAIN+=(--resume_from_ckpt "$RESUME_FROM_CKPT"); fi
RECON=("${COMMON[@]}" "${MODEL[@]}" --batch_size "$RECON_BATCH_SIZE")
if [[ "$BLURRY_RECON" == true ]]; then RECON+=(--blurry_recon); else RECON+=(--no-blurry_recon); fi
SUFFIX=""
if [[ "$MODE" == *enhance* ]]; then SUFFIX="_enhance"; fi

# Keep upstream stage ordering and write each stage's console log beside its outputs.
if [[ "$STAGES" == *1* ]]; then
    "$PYTHON" -u -m "$PACKAGE.train_neurons" "${TRAIN[@]}" \
        --batch_size "$BACKBONE_BATCH_SIZE" --num_epochs "$BACKBONE_EPOCHS" \
        --mixup_pct "$BACKBONE_MIXUP" --max_lr "$BACKBONE_LR" "${TRAIN_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/backbone_log_$TIMESTAMP.txt"
fi
if [[ "$STAGES" == *2* ]]; then
    "$PYTHON" -u -m "$PACKAGE.train_neurons" "${TRAIN[@]}" --neurons_decoupler \
        --batch_size "$DECOUPLER_BATCH_SIZE" --num_epochs "$DECOUPLER_EPOCHS" \
        --mixup_pct "$DECOUPLER_MIXUP" --max_lr "$DECOUPLER_LR" "${TRAIN_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/decoupler_log_$TIMESTAMP.txt"
fi
if [[ "$STAGES" == *3* ]]; then
    "$PYTHON" -m "$PACKAGE.recon_keyframe_neurons$SUFFIX" "${RECON[@]}" "${RECON_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/recon_log_$TIMESTAMP.txt"
fi
if [[ "$STAGES" == *4* ]]; then
    "$PYTHON" -m "$PACKAGE.caption_keyframe$SUFFIX" "${COMMON[@]}" "${CAPTION_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/caption_log_$TIMESTAMP.txt"
fi
if [[ "$STAGES" == *5* ]]; then
    VIDEO=(--exp "$EXP_DIR" --mode "$MODE" --subj "$SUBJECT" --root_dir "$DATA_ROOT"
           --cache_dir "$HF_HUB_CACHE" --pretrained-model-path "$SD_MODEL"
           --config "$VIDEO_CONFIG" --inference-config "$INFERENCE_CONFIG"
           --L "$VIDEO_LENGTH" --W "$VIDEO_WIDTH" --H "$VIDEO_HEIGHT")
    if [[ "$WITHOUT_XFORMERS" == true ]]; then VIDEO+=(--without-xformers); fi
    "${LAUNCH[@]}" --module "$PACKAGE.scripts.neuroclips_video$SUFFIX" "${VIDEO[@]}" "${VIDEO_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/video_log_$TIMESTAMP.txt"
fi
if [[ "$STAGES" == *6* ]]; then
    "$PYTHON" -m "$PACKAGE.run_metrics" "${COMMON[@]}" --mode "$MODE" --root_dir "$DATA_ROOT" "${METRICS_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/eval_${MODE}_$TIMESTAMP.txt"
fi
if [[ "$STAGES" == *e* ]]; then
    "$PYTHON" -m "$PACKAGE.gen_decoupled_outputs" "${RECON[@]}" "${DECOUPLED_EXTRA_ARGS[@]}" \
        2>&1 | tee "$EXP_DIR/decoupled_log_$TIMESTAMP.txt"
fi
