#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the MetaWorld expert-data directory}"
: "${ROBOENGINE_MASK_ROOT:=${MASK_ROOT:-}}"
: "${ROBOENGINE_MASK_ROOT:?Set ROBOENGINE_MASK_ROOT to the RoboEngine robot-mask directory}"
: "${VLM_MODEL:?Set VLM_MODEL to the local InternVL3-2B directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the output directory}"

RUN_ID="${RUN_ID:-motionweave_metaworld6_h4_20k}"
GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-37801}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  src/train_motionweave.py \
  --tasks pick-place-v2,disassemble-v2,stick-pull-v2,assembly-v2,shelf-place-v2,hand-insert-v2 \
  --data_root "${DATA_ROOT}" \
  --roboengine_mask_root "${ROBOENGINE_MASK_ROOT}" \
  --vlm_model "${VLM_MODEL}" \
  --strategy fsdp \
  --gradient_checkpointing \
  --vla_tune_mode full \
  --batch_size 8 \
  --total_steps 20000 \
  --save_every 20000 \
  --lr 1e-5 \
  --weight_decay 0.0 \
  --max_grad_norm 1.0 \
  --action_horizon 4 \
  --diffusion_train_steps 100 \
  --diffusion_inference_steps 10 \
  --action_head_objective flow \
  --repeated_diffusion_steps 8 \
  --action_norm_type bounds \
  --flow_sample_clip 1.0 \
  --dit_dim 768 \
  --dit_layers 12 \
  --dit_heads 12 \
  --aimg_dim 512 \
  --hrc_dim 768 \
  --hrc_heads 8 \
  --motion_grounding_loss_weight 0.05 \
  --log_dir "${RUN_DIR}" \
  --save_path "${RUN_DIR}/final.pth" \
  2>&1 | tee "${RUN_DIR}/train.log"

