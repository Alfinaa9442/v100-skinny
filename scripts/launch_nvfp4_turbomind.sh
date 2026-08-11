#!/usr/bin/env bash
# Qwen3.5-27B-NVFP4, TP4, fork's production TurboMind NVFP4 route
# (design-doc env contract), no speculative decoding. Settings matched
# to the other benchmark runs for direct comparison.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-122
cd ~

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export VLLM_SM70_QUANT_BACKEND=turbomind
export VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SKINNY_NVFP4=0
unset VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS 2>/dev/null || true

exec python -m vllm.entrypoints.openai.api_server \
  --model "$HOME/models/Qwen3.5-27B-NVFP4" \
  --served-model-name qwen3.5-27b-nvfp4-turbomind \
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
