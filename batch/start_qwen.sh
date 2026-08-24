#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — BATCH / THROUGHPUT mode.
# Measured: ~1,000 tok/s aggregate @ 64 concurrent (128 in / 512 out), 150k context.
#
# Hard-won settings, don't change casually:
#  - --mamba-ssm-cache-dtype float16: the Gated DeltaNet recurrent state is
#    fp32 by default (Qwen's config says so) and costs ~150 MB per resident
#    request; fp16 halves that AND halves the state traffic per decode step.
#    That is what lets all 64 requests actually run at once (fp32: only 37).
#    Perplexity is unchanged (8.045 vs 8.046 on our en/da/code check).
#  - VLLM_MARLIN_INPUT_DTYPE=int8: int8 tensor-core (W4A8) Marlin path for the
#    MLP GEMMs, weights stay int4. Roughly +35% aggregate on top of the state
#    change for +2.2% perplexity. Needs both marlin patches from patches/.
#    INT8_LAYERS=gate_up is the gentler variant (+0.9% PPL, ~+15%);
#    INT8_ACT= (empty) turns it off entirely (pure W4A16, quality-neutral).
#  - --language-model-only skips the vision tower entirely (0.858 GiB on this
#    checkpoint); VISION=1 keeps it for a client that sends images
#  - expandable_segments is required: the DeltaNet prefill kernels allocate
#    transient workspace and fragment the allocator, OOMs at util >= 0.978 without it
#  - gpu-memory-utilization 0.972 is the sweet spot on a headless box
#    (X/display holds ~220 MB; 0.98 fails the startup free-memory check)
#  - max-num-batched-tokens 2048 beats 8192 here: bigger chunks inflate the
#    profiled activation peak, which shrinks the KV/state page pool
#  - kv-cache-dtype fp8 roughly doubles the usable context/pool

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_SEQS=${MAX_SEQS:-64}
API_SERVERS=${API_SERVERS:-1}
# KV=fp8 (default): FlashInfer fp8 KV cache, 150k context, fastest.
# KV=kvarn: the KVarN 4-bit-key / 2-bit-value cache (kvarn/ in this repo,
# needs `bash kvarn/install.sh` once): 262k context, ~2x the token capacity,
# +0.2% perplexity, ~20% slower decode at long context and lower short-request
# throughput (see docs/long-context.md).
# KV=int4pth: vLLM's built-in int4 per-token-head KV cache on the Triton
# attention backend: 262k context with no extra install, ~1.5x slower decode /
# 2.3x slower prefill at 100k than fp8 (docs/long-context.md).
KV=${KV:-fp8}
if [ "$KV" = "int4pth" ]; then
  MAX_LEN=${MAX_LEN:-262144}
  GPU_UTIL=${GPU_UTIL:-0.93}
  KV_ARGS="--kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN"
elif [ "$KV" = "kvarn" ]; then
  MAX_LEN=${MAX_LEN:-262144}
  GPU_UTIL=${GPU_UTIL:-0.93}
  KV_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  # fp16 staging pool for the tiles still being written: share of free memory
  # after weights; 0.25 keeps all 64 slots, smaller values cap max-num-seqs
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.25}
else
  MAX_LEN=${MAX_LEN:-150000}
  GPU_UTIL=${GPU_UTIL:-0.972}
  KV_ARGS="--kv-cache-dtype fp8"
fi
# int8 activations: "int8" (default) or empty for W4A16; layers: regex on the
# layer name, "mlp" (default: gate_up_proj + down_proj) or "gate_up"
INT8_ACT=${INT8_ACT-int8}
INT8_LAYERS=${INT8_LAYERS-mlp}

# PREFIX_CACHE=1: reuse the KV of a shared prompt prefix across requests, and resume the
# recurrent (GDN) state from the last cached block boundary. For an API backend where every
# request carries the same system prompt / document this is the difference between paying
# for that prefix once and paying for it every time: 64 requests sharing a 5.8k-token system
# prompt (conc 32) take 222 s without it and 17 s with it. Costs ~14% of the KV pool
# (223,821 -> 193,298 tokens) and nothing on workloads with no shared prefix (870 vs 876
# tok/s on the 128/512 row). Hybrid models keep this opt-in upstream.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
fi

# Tool / function calling. Without BOTH flags vLLM rejects any request carrying
# `tools` with tool_choice "auto": 400 '"auto" tool choice requires
# --enable-auto-tool-choice and --tool-call-parser to be set'. TOOLS=0 turns it off.
#
# qwen3_coder is a deliberate choice for this model, not a vLLM default and not a
# leftover -- do NOT "correct" it to hermes. The parser has to match the format the
# chat template asks the model for, and Qwen3.8's asks for XML --
# <tool_call><function=NAME><parameter=K>V</parameter> -- NOT the JSON body that
# hermes, the usual answer for a Qwen model, reads. Getting that wrong does not
# error: the call comes back as ordinary content and the client sees no tool_calls,
# which reads as the model being bad at tools rather than as a misconfigured server.
# The name is the call format, not the checkpoint -- nothing here is Qwen3-Coder.
# qwen3_coder, qwen3_xml and mimo are three names for one Qwen3EngineToolParser in
# 0.27.1, which is the tool-side adapter of the same parser engine that
# --reasoning-parser qwen3 already uses (vllm/parser/qwen3.py).
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
TOOL_ARGS=$([ "${TOOLS:-1}" = 1 ] && echo --enable-auto-tool-choice --tool-call-parser $TOOL_PARSER)

# Vision. --language-model-only drops the vision tower cleanly -- no weights loaded,
# 0.858 GiB on this checkpoint (gotcha 9) -- and stays the default. VISION=1 keeps
# the tower, for a client that sends images: screenshots into a coding assistant,
# captioning, document photos.
#
# Only --language-model-only needs a knob. It is hardcoded in the exec line below, so
# the alternative is countering it with --no-language-model-only from EXTRA_ARGS and
# depending on which flag argparse saw last -- which regresses silently: images are
# still accepted and still counted as prompt tokens, and the model answers from
# placeholder embeddings. The two flags VISION=1 adds have no such conflict and can
# be overridden from EXTRA_ARGS, which is expanded after them. The pixel cap is
# shipped rather than left to the processor default because vLLM profiles the encoder
# at the largest image it will accept, and that peak comes out of the KV pool:
# 2097152 px = 2048 image tokens.
if [ "${VISION:-0}" = 1 ]; then
  VISION_ARGS='--limit-mm-per-prompt {"image":{"count":1}} --mm-processor-kwargs {"size":{"shortest_edge":65536,"longest_edge":2097152}}'
  # VISION_OFFLOAD keeps the tower's weights in pinned host RAM and copies each module to
  # the GPU for the duration of its own forward (patches/vision-tower-cpu-offload.patch).
  # It defaults ON, because on 24 GB SPEC=dflash2 + VISION=1 does not boot without it:
  # the tower is 0.85 GiB of the ~1.1 GiB transient margin the KV_MEM comment sizes, and
  # graph capture then dies allocating the split-KV verify buffer --
  #   torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 787.50 MiB is free
  #     (spec_decode_attn.py:184, self.part_o)
  # measured here, VISION=1 VISION_OFFLOAD=0 SPEC=dflash2, RTX 3090 at 250 W. With the
  # offload the same config comes up with the full 69,758-token pool and reads images.
  #
  # It is close to free: isolated-tower measurement at PCIe 4.0 x16, one 8192-patch image,
  # median of 10 forwards, 891.3 -> 9.0 MiB of resident weights and 1160.5 -> 308.2 MiB of
  # peak allocation for 296 -> 333 ms of encode, output bit-exact either way. Set
  # VISION_OFFLOAD=0 only on a card with room to spare, where 36 ms per image buys nothing.
  [ "${VISION_OFFLOAD:-1}" = 1 ] && export VLLM_VISION_CPU_OFFLOAD_GB=${VLLM_VISION_CPU_OFFLOAD_GB:-1}
else
  VISION_ARGS="--language-model-only"
fi

export PATH="$REPO/venv/bin:$PATH"
# Off under WSL, where the VMM calls break Marlin repack — see the long note in
# single-user/start_qwen.sh. Overridable both ways.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  ALLOC_DEFAULT=expandable_segments:False
  [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && echo \
    "WSL detected: PYTORCH_CUDA_ALLOC_CONF=$ALLOC_DEFAULT (VMM breaks Marlin repack under the paravirt driver; set it explicitly to override)"
else
  ALLOC_DEFAULT=expandable_segments:True
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-$ALLOC_DEFAULT}
# flashinfer's sampling.cu does not build with older system nvcc (12.0);
# the attention kernels JIT fine. Remove this if you have a recent CUDA toolkit.
export VLLM_USE_FLASHINFER_SAMPLER=0
# "Off" for these is UNSET, not empty. vllm/envs.py registers VLLM_MARLIN_INPUT_DTYPE
# through env_with_choices(..., None, ["int8", "fp8"]), which rejects "" outright --
# `ValueError: Invalid value '' ... Valid options: ['int8', 'fp8']` -- so exporting the
# empty string killed the engine at startup instead of turning the feature off. That is
# the documented way to disable it (issue #20), so export only when non-empty.
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
[ -n "$INT8_LAYERS" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS

# API key: put it in api_key.txt in the repo root, or export VLLM_API_KEY.
if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  ${VISION_ARGS} \
  $KV_ARGS \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --compilation-config "{\"max_cudagraph_capture_size\":64,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]}" \
  --reasoning-parser qwen3 \
  ${TOOL_ARGS} \
  ${EXTRA_ARGS}
