#!/usr/bin/env bash
set -euo pipefail

: "${CKPT:?Set CKPT to a MotionWeave checkpoint}"
: "${VLM_MODEL:?Set VLM_MODEL to the local InternVL3-2B directory}"
: "${EVAL_ROOT:?Set EVAL_ROOT to the evaluation output directory}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
mkdir -p "${EVAL_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM=false

python src/eval_motionweave_firstsuccess.py \
  --ckpt "${CKPT}" \
  --vlm_model "${VLM_MODEL}" \
  --tasks pick-place-v2,disassemble-v2,stick-pull-v2,assembly-v2,shelf-place-v2,hand-insert-v2 \
  --device cuda:0 \
  --num_episodes 25 \
  --max_steps 175 \
  --execute_horizon 2 \
  --seed "${SEED}" \
  --save_dir "${EVAL_ROOT}"

