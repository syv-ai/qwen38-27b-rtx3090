#!/bin/bash
# 256k on one 24 GB card: int4 per-token-head KV cache (experimental).
#
# Trades decode speed for context capacity: TRITON_ATTN + int4 KV serve a
# 314,915-token pool at MAX_LEN=256000 (vs 57,669 for the shipped bf16
# config, 268k for CTX=huge/KVarN) on stock vLLM machinery — no kvarn
# install. Costs ~20% decode vs the FLASH_ATTN bf16 path and the int4
# cache's quality at depth is unmeasured here; see the README section.
# Needs patches/int4-kv-per-token-head.patch (applied by setup like the
# rest of patches/). Profile contributed in PR #42 (@lachhabw).
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
export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
# The multi-query 3D verify for the int4 cache (patches/spec-decode-int4-kv-mq3d.patch) was opt-in and nothing
# set it, so this profile ran the stock 2D verify: on a 4090 at 120k, DFlash2 k=7, fresh prefill, decode 43.9 / 26.5 /
# 16.6 tok/s at 24k / 49k / 88k tokens; with it on, 90.4 / 73.7 / 60.2 (and 48.1 at 145k on a 200k boot). Set
# INT4_MQ_3D=0 to get the old path back.
export VLLM_INT4_MQ_3D=${INT4_MQ_3D:-1}
# INT8_ACT=int8: the W4A8 Marlin path for prefill, the same knob start_qwen.sh has (docs/optimizations.md). On this
# profile it lifts fresh prefill +42% at 24k tokens to +13% at 145k (2,765 / 1,884 / 1,246 / 835 tok/s against
# 1,942 / 1,446 / 1,039 / 736 on a 4090, 200k max length) with decode unchanged; the quality trade is the documented
# int8 one, so it stays opt-in here too. INT8_LAYERS narrows it the same way.
INT8_ACT=${INT8_ACT-}
INT8_LAYERS=${INT8_LAYERS-mlp|linear_attn|self_attn}
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
[ -n "$INT8_ACT" ] && [ -n "$INT8_LAYERS" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS

# Ghost regions from a previous OOM-killed server hold host RAM hostage
# (gotcha: 70 restarts of accumulation); same cleanup as start_qwen.sh.
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[alternative] removing stale offload region $f"; rm -f "$f"; }
  done
fi

MODEL=${MODEL:-models/Qwen3.8-27B-W4A16-AutoRound}
DRAFT=${DRAFT:-models/Qwen3.8-27B-DFlash2-W4A16}
# SPEC=dflash2 (default) or SPEC=off. This script used to hardcode the drafter
# and silently ignore SPEC -- PR #46's campaign ran an "A/B" against SPEC=off
# that was really two spec-on arms (the tell: 2.29 emitted tokens per step on
# an arm that must read 1.00). Unrecognized values refuse for the same reason.
SPEC=${SPEC:-dflash2}
case "$SPEC" in
  # draft_sample_method: see start_qwen.sh -- on 0.28 the draft-logits buffer is only
  # allocated when the config asks, and without it acceptance drops ~16% (#73).
  dflash2) SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":${DFLASH_TOKENS:-7},\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}") ;;
  off|none) SPEC_ARGS=() ;;
  *) echo "SPEC=$SPEC is not a mode here: dflash2 (default) or off." >&2; exit 1 ;;
esac
PORT=${PORT:-18020}
GPU_UTIL=${GPU_UTIL:-0.95}   # 0.93 on WSL2 or with a desktop compositor on the card
MAX_LEN=${MAX_LEN:-256000}   # 256000 needs the full 0.95; drop MAX_LEN before GPU_UTIL
ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int4_per_token_head"
VISION=${VISION:-0}
ENABLE_THINKING=${ENABLE_THINKING:-false}
DRAFT_TOKENS=${DFLASH_TOKENS:-7}
MAX_SEQS=${MAX_SEQS:-1}

CG=$(( MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1) ))
# DFLASH_TOKENS>7 needs the synchronous scheduler, same as start_qwen.sh.
if [ "$DRAFT_TOKENS" -gt 7 ]; then ASYNC_SCHED=${ASYNC_SCHED:-0}; else ASYNC_SCHED=${ASYNC_SCHED:-1}; fi
PREFIX_CACHE=${PREFIX_CACHE:-1}

VISION_ARGS="--language-model-only"
[ "$VISION" = 1 ] && VISION_ARGS='--mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'

PREFIX_ARGS=""
[ "$PREFIX_CACHE" = 1 ] && PREFIX_ARGS="--enable-prefix-caching --mamba-cache-mode align"

# Array, not $( ... || echo ... ): the fallback makes it errexit-safe, but an
# unquoted expansion still word-splits; match SPEC_ARGS/METRICS_ARGS.
ASYNC_ARGS=(--no-async-scheduling)
[ "$ASYNC_SCHED" = 1 ] && ASYNC_ARGS=(--async-scheduling)

# REQ_METRICS=1: per-request timing fields + usage on every response (issue #51).
# Not with --disable-log-stats (the timing fields need the engine-stats path).
# Array, not $( [ ] && echo ): the command substitution exits 1 when the test
# is false, which under `set -e` killed this script silently (#59).
METRICS_ARGS=()
[ "${REQ_METRICS:-0}" = 1 ] && METRICS_ARGS=(--enable-per-request-metrics --enable-force-include-usage)

exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count 1 \
  ${VISION_ARGS} \
  ${ATTN_ARGS} \
  --mamba-ssm-cache-dtype float16 \
  "${ASYNC_ARGS[@]}" \
  --max-num-batched-tokens 2048 \
  "${SPEC_ARGS[@]}" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  "${METRICS_ARGS[@]}" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs "{\"enable_thinking\": $ENABLE_THINKING}" \
  ${PREFIX_ARGS} \
  ${EXTRA_ARGS}
