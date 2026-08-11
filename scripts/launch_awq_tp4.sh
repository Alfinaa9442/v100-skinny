#!/usr/bin/env bash
# Qwen3.5-27B-AWQ, TP4, stock fork (TurboMind SM70 AWQ path), no
# speculative decoding. Settings matched to the NVFP4 benchmark runs
# (32768 ctx, 2 seqs, 8192 batched tokens, CUDA graphs) for a direct
# AWQ-vs-NVFP4 comparison on the same base model.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-122
cd ~

# Reference arm. HF cache location is site-specific; override AWQ_SNAP to
# point at your own QuantTrio/Qwen3.5-27B-AWQ snapshot directory.
AWQ_CACHE="${AWQ_CACHE:-/srv/1cat/cache/hf}"
SNAP="${AWQ_SNAP:-$(ls -d "$AWQ_CACHE"/hub/models--QuantTrio--Qwen3.5-27B-AWQ/snapshots/*/ 2>/dev/null | head -1)}"
if [ -z "$SNAP" ] || [ ! -d "$SNAP" ]; then
  echo "ERROR: AWQ snapshot not found. Set AWQ_SNAP to the snapshot dir." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export VLLM_SKINNY_NVFP4=0

exec python -m vllm.entrypoints.openai.api_server \
  --model "$SNAP" \
  --served-model-name qwen3.5-27b-awq \
  --trust-remote-code \
  --dtype float16 \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.88 \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 0.0.0.0 \
  --port 8000
