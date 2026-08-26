#!/bin/bash
set -e

if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  ALLOC_DEFAULT=expandable_segments:False
  export VLLM_WSL2_ENABLE_PIN_MEMORY=1
else
  ALLOC_DEFAULT=expandable_segments:True
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-$ALLOC_DEFAULT}
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$PWD/venv/bin:$PATH"
export VLLM_API_KEY="$(cat api_key.txt)"
export VLLM_SPEC_DECODE_ATTN=1
export VLLM_DFLASH2_LOOKUP=1

MODEL=models/Qwen3.8-27B-W4A16-AutoRound
DRAFT=models/Qwen3.8-27B-DFlash2-W4A16
GPU=1
PORT=18020
GPU_UTIL=0.93 # 0.95 (WSL: 0.93)
MAX_LEN=256000
ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int4_per_token_head"
VISION=0
ENABLE_THINKING=false
DRAFT_TOKENS=7
MAX_SEQS=1

CG=$(( MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1) ))
ASYNC_SCHED=1 # If DRAFT_TOKENS is above 7 set it to 0
PREFIX_CACHE=1


VISION_ARGS="--language-model-only"
[ "$VISION" = 1 ] && VISION_ARGS='--mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'

PREFIX_ARGS=""
[ "$PREFIX_CACHE" = 1 ] && PREFIX_ARGS="--enable-prefix-caching --mamba-cache-mode align"

ASYNC_ARGS=$([ "$ASYNC_SCHED" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)


CUDA_VISIBLE_DEVICES=$GPU exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count 1 \
  ${VISION_ARGS} \
  ${ATTN_ARGS} \
  --mamba-ssm-cache-dtype float16 \
  ${ASYNC_ARGS} \
  --max-num-batched-tokens 2048 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs "{\"enable_thinking\": $ENABLE_THINKING}" \
  ${PREFIX_ARGS}
