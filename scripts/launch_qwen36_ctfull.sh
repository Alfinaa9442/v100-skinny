#!/usr/bin/env bash
# Qwen3.6-27B-NVFP4, TP4, serving through the custom skinny kernel.
#
# Route: VLLM_SM70_NVFP4_TURBOMIND=0 pins the scheme to the (patched)
# MarlinNvFp4LinearKernel; VLLM_SKINNY_NVFP4=1 dispatches M<=64 to the
# skinny kernel (decode + speculative-verify widths) with marlin serving
# M>64 (prefill). First calls self-check numerically vs marlin.
# Speculation OFF (no --speculative-config; the matrix notes "no valid
# MTP weights" for 27B-NVFP4 anyway).
#
# max-model-len 32768: both weight formats resident (~8GB/rank on 16GB
# cards) leaves less KV headroom than the 32GB reference target.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate 1cat-vllm-122
cd ~  # outside the source checkout so the wheel's extensions load

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export VLLM_SM70_NVFP4_TURBOMIND=0
export VLLM_SM70_QUANT_BACKEND=marlin  # forces_marlin() -> SM70 allowed
export VLLM_SKINNY_NVFP4=${VLLM_SKINNY_NVFP4:-1}
# Session default: 4-bit lm_head ON (speed-first); end-user/default
# policy is OFF — see marlin_patched.py policy comment.
export VLLM_SKINNY_LMHEAD=${VLLM_SKINNY_LMHEAD:-1}
export VLLM_SKINNY_NVFP4_SRC="$HOME/flatness-run/skinny_kernels.cu"
unset VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS 2>/dev/null || true

# ---- preflight: checkpoint layout ------------------------------------------
# NVIDIA's published checkpoint cannot be served directly on Volta: its FP8
# attention/GDN projections have no SM70 hardware path, and the skinny
# dispatch hooks the compressed-tensors W4A16 scheme. One conversion is
# required; after it, only one source shard is needed at runtime.
SKINNY_MODEL_DIR="${SKINNY_MODEL_DIR:-$HOME/models/Qwen3.6-27B-NVFP4-CTfull}"
SKINNY_SRC_DIR="${SKINNY_SRC_DIR:-$HOME/models/Qwen3.6-27B-NVFP4}"
SKINNY_LMHEAD_SHARD="$SKINNY_SRC_DIR/model-00003-of-00003.safetensors"

if [ ! -f "$SKINNY_MODEL_DIR/config.json" ]; then
  echo "ERROR: serving checkpoint not found at $SKINNY_MODEL_DIR" >&2
  echo "" >&2
  echo "NVIDIA's nvidia/Qwen3.6-27B-NVFP4 export must be repacked once before" >&2
  echo "it can be served on Volta (FP8 layers down-converted to NVFP4; output" >&2
  echo "written in compressed-tensors format):" >&2
  echo "" >&2
  echo "  CONV_OUT=$SKINNY_MODEL_DIR python scripts/convert_modelopt_to_ct.py" >&2
  echo "" >&2
  exit 1
fi

if [ "${VLLM_SKINNY_LMHEAD_NATIVE:-0}" = "1" ] && [ ! -f "$SKINNY_LMHEAD_SHARD" ]; then
  echo "ERROR: native lm_head enabled but source shard not found:" >&2
  echo "  $SKINNY_LMHEAD_SHARD" >&2
  echo "" >&2
  echo "This shard carries NVIDIA's original lm_head codes/scales, read at" >&2
  echo "every boot. Point VLLM_SKINNY_LMHEAD_NATIVE at its path if it lives" >&2
  echo "elsewhere, or set VLLM_SKINNY_LMHEAD_NATIVE=0 to render in BF16." >&2
  exit 1
fi

if [ -f "$SKINNY_SRC_DIR/model-00001-of-00003.safetensors" ]; then
  echo "NOTE: the repack is done, so source shards 1-2 (~18.6 GB) are no longer"
  echo "      needed to serve — runtime reads only model-00003-of-00003.safetensors."
  echo "      Reclaim with:  rm $SKINNY_SRC_DIR/model-0000[12]-of-00003.safetensors"
  echo "      Keep them to re-run the conversion or re-verify weight provenance."
fi
# ---------------------------------------------------------------------------

# NUMA pinning: all 4 GPUs are on node 0; unpinned boots re-roll thread/page
# placement (~3% round-time variance, hardware_audit #1/#2).
# SKINNY_NUMA=0 disables the pin for experiments.
if [ "${SKINNY_NUMA:-1}" = "1" ]; then
  NUMA_PREFIX="numactl --cpunodebind=0 --membind=0"
else
  NUMA_PREFIX=""
fi

exec $NUMA_PREFIX python -m vllm.entrypoints.openai.api_server \
  --model "$HOME/models/Qwen3.6-27B-NVFP4-CTfull" \
  --served-model-name qwen3.6-27b-nvfp4-skinny \
  --trust-remote-code \
  --dtype float16 \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization ${GMU:-0.88} \
  --max-model-len 32768 \
  --max-num-seqs ${MNS:-2} \
  --max-num-batched-tokens ${MBT:-8192} \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8000 \
  ${EXTRA_VLLM_ARGS:-}
