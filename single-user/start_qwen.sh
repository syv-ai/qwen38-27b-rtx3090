#!/bin/bash
# Qwen3.8-27B on a single RTX 3090 — SINGLE USER / LOW LATENCY mode.
#
# Same base config as batch mode, plus MTP speculative decoding: the checkpoint
# keeps Qwen's multi-token-prediction head, so the model drafts 3-4 tokens ahead
# and verifies them in one pass. Measured on realistic chat prompts with the
# `-fast` model variant (see "Fast variant" below): ~114 tok/s at the model's
# default sampling, ~124 tok/s greedy (vs 46 tok/s without speculation).
# What makes 4 drafts pay off, in order of importance:
#  - the drafter scores a 40k-token draft head (prepare/build_draft_vocab.py) — and the
#    id list matters: a vocabulary counted over the model's OWN outputs covers
#    97.5% of what it generates (96% on code); the earlier web-text list only 92%
#    (83% on code), and every miss is a forced rejection (108 vs 98 tok/s greedy)
#  - the MTP module and lm_head requantized to int4 with GPTQ calibrated on the
#    model's hidden states (drafter/): 850 -> 215 MB per draft, 1.27 -> 0.65 GB
#    lm_head per verify, +0.6% perplexity, acceptance unchanged
#  - patches/spec-decode-attn.patch: split-KV attention for the 5-query verify
#    step (FA2 leaves 58 of 82 SMs idle there); patches/sampler-...: sort-free
#    top-k, multi-block softmax, drafts truncated to the target's top-k/top-p
#  - draft_sample_method=probabilistic: drafts are sampled, not argmax'ed, which
#    lifts acceptance at temperature > 0
# Speculative decoding is exact: none of this changes what gets sampled.
#
# CTX=fast (default here): FlashAttention + bf16 KV, 4 drafts, 64k context.
# CTX=long: fp8 KV via FlashInfer, 150k context, 3 drafts (k=4 crashes on
#   FlashInfer as soon as one request finishes while another is mid-generation,
#   vLLM 0.27.1); the split-KV attention patch is bf16-KV only, so ~90/98 tok/s.
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/), 200k context with MTP, at roughly
#   half the decode rate past 100k — see below and docs/long-context.md.
#
# Fast variant: MODEL defaults to models/Qwen3.8-27B-W4A16-AutoRound-fast when it
# exists (int4-GPTQ lm_head + MTP, own-output draft vocab; drafter/README.md), else
# the base dir (int8 lm_head/MTP: ~108/107 tok/s with the shipped draft vocab).
#
# max-num-seqs is 8 here: fewer state slots to reserve (each request holds
# k+1 recurrent-state slots), and past a handful of concurrent users you
# should be running batch mode anyway. Int8 activations are pointless at
# batch size 1 (memory-bound), so this mode stays W4A16.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$DIR")"
cd "$REPO"

if [ -z "$MODEL" ] && [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_SEQS=${MAX_SEQS:-}
# 0.93 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode
# path allocates beyond the startup memory profile (docs/gotchas.md, gotcha 4).
GPU_UTIL=${GPU_UTIL:-0.93}
API_SERVERS=${API_SERVERS:-1}
# CTX=long (default): fp8 KV via FlashInfer, 150k context, 3 drafts.
# CTX=fast: bf16 KV via FlashAttention, ~64k context, 4 drafts (~+7%).
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/ in this repo, run kvarn/install.sh
#           once), 200k context with MTP. The decode tax is a function of context,
#           not a constant: ~6% on short prompts, but 2.13x at 112k (32.0 vs fp8's
#           68.1 tok/s), of which ~1.98x is step time and the rest is MTP acceptance
#           falling from 2.56 to 2.38 tokens per step. Take it when the request
#           would not otherwise fit, not for speed (see docs/long-context.md).
CTX=${CTX:-fast}
# SPEC=mtp (default): Qwen's own MTP head, k drafts chained (the numbers above).
# SPEC=dflash2: the DFlash2 block drafter (incoai/Qwen3.8-27B-DFlash2, requantized
#   to W4A16 by this repo: prepare/fetch_dflash2.py), 7 drafts in ONE non-autoregressive
#   pass + a path selector; runs on vLLM's V2 model runner
#   (patches/dflash2-backport.patch). CTX=fast (bf16, 64k), CTX=long (int8,
#   128k) or, with kvarn/install.sh, CTX=huge (KVarN 4/2-bit, 240k + prefix
#   caching); see README "DFlash2".
SPEC=${SPEC:-mtp}
# SPEC_ATTN=1: split-KV Triton attention for the multi-query verify step
# (patches/spec-decode-attn.patch); bf16 KV only, so CTX=fast only.
if [ "$CTX" = "fast" ]; then
  MAX_LEN=${MAX_LEN:-65536}
  DRAFT_TOKENS=${DRAFT_TOKENS:-4}
  ATTN_ARGS="--attention-backend FLASH_ATTN --kv-cache-dtype bfloat16"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$CTX" = "huge" ]; then
  MAX_LEN=${MAX_LEN:-200000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype kvarn_k4v2_g128 --block-size 128"
  export KVARN_POOL_MEM_FRAC=${KVARN_POOL_MEM_FRAC:-0.15}
else
  MAX_LEN=${MAX_LEN:-150000}
  DRAFT_TOKENS=${DRAFT_TOKENS:-3}
  ATTN_ARGS="--kv-cache-dtype fp8"
fi
if [ "$SPEC" = "dflash2" ] && [ "$CTX" = "long" ]; then
  # int8 per-token-head KV on the Triton backend: the same 5.2 GiB pool holds 136,429
  # tokens instead of 69,758, because patches/hybrid-sw-block-promote.patch stops the
  # drafter's 5 sliding-window layers from taking 385 nearly-empty blocks, and
  # patches/spec-decode-attn-int8.patch lets the split-KV verify kernel read the quantized
  # cache (vLLM's own Triton attention will not split KV for a multi-query verify, which
  # is every step here, and costs 7.4 ms per layer at 128k against this kernel's 1.3).
  # Costs prefill: 251 s to load a 112k document against FLASH_ATTN's ~112 s. With
  # PREFIX_CACHE=1 only the first turn pays it (5.9 s afterwards), which is why this is a
  # mode for a RAG or coding front-end that loads a document once, not for general chat.
  ATTN_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head"
  export VLLM_SPEC_DECODE_ATTN=${SPEC_ATTN:-1}
elif [ "$SPEC" = "dflash2" ] && [ "$CTX" = "huge" ]; then
  # KVarN 4/2-bit KV on the V2 runner (kvarn/, with kvarn-v2-runner.patch as its
  # second stage): the pinned pool holds 268k tokens at 245760 max-model-len.
  # The split-KV verify attention is bf16-KV only -- the KVarN backend brings
  # its own dequant path, so the env stays off here.
  export VLLM_SPEC_DECODE_ATTN=0
elif [ "$SPEC" = "dflash2" ] && [ "$CTX" != "fast" ]; then
  echo "SPEC=dflash2 supports CTX=fast (bf16, 64k), CTX=long (int8, 128k) and CTX=huge (KVarN, 240k; kvarn/install.sh); CTX=$CTX keeps SPEC=mtp" >&2
  SPEC=mtp
fi
if [ "$SPEC" = "dflash2" ]; then
  if [ -z "$DRAFT" ]; then
    for d in Qwen3.8-27B-DFlash2-W4A16 Qwen3.8-27B-DFlash2; do
      [ -f "$REPO/models/$d/model.safetensors" ] && DRAFT=$REPO/models/$d && break
    done
  fi
  [ -n "$DRAFT" ] || { echo "SPEC=dflash2 needs the drafter: venv/bin/python prepare/fetch_dflash2.py" >&2; exit 1; }
  # Lookup-augmented drafting: when the model is reproducing something from its context,
  # draft from the context instead of from the drafter
  # (patches/dflash2-lookup-drafting.patch).
  export VLLM_DFLASH2_LOOKUP=${LOOKUP:-1}
  # DFLASH_TOKENS is the *verify* block, which no longer has to equal the drafter's: the
  # DFlash2 checkpoint always proposes the 7 tokens it was trained for, and any position
  # past that is filled from the request's own context, costing the drafter nothing. The
  # block length is adaptive -- the long block is only scheduled while the lookup is
  # actually firing -- so ordinary steps still verify 8 tokens.
  #
  # DFLASH_TOKENS=15 is "reproduction mode": +50% where the model reproduces its context
  # (388 vs 259 tok/s reproducing a document verbatim) and +9% on the short-prompt C1 set,
  # against 3-20% on long-context work that mixes prose with quoting, 4 request slots
  # instead of 8 and 56k of context instead of 64k. Worth setting for a coding assistant
  # applying edits or a RAG front-end quoting sources; the default stays 7.
  DRAFT_TOKENS=${DFLASH_TOKENS:-7}
  SPEC_CFG="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$DRAFT_TOKENS}"
  # The split-KV verify attention (patches/spec-decode-attn.patch) sizes its partial
  # buffers once for the longest query block it will see -- a captured CUDA graph holds
  # their addresses, so they must not be grown later.
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  # The lookup produces WRONG OUTPUT on a prefix-cache hit when the verify block is long
  # and the cache is KVarN. Turn 1 is correct, turn 2 over the same document is not: it
  # tracks the source for 38-45 characters and then diverges, deterministically, at every
  # prompt length tested. Fresh server per data point, three requests each, 11 configs:
  #
  #   k=7  CTX=huge  lookup on   cold 794/794   self-hit 794/794   clean
  #   k=15 CTX=fast  lookup on   cold 794/794   self-hit 794/794   clean
  #   k=7  CTX=fast  lookup on   cold 794/794   self-hit 794/794   clean
  #   SPEC=mtp CTX=huge          cold 794/794   self-hit 794/794   clean
  #   k=15 CTX=huge  lookup on   cold 794/794   self-hit 38-45/79x  WRONG (5 residues)
  #   k=15 CTX=huge  PIECEWISE   cold 794/794   self-hit 38/793     WRONG
  #   k=15 CTX=huge  LOOKUP=0    cold 794/794   self-hit 794/794   clean
  #
  # So it needs the long block AND KVarN AND the lookup AND a cache hit; it is NOT the
  # cudagraph capture (PIECEWISE breaks identically, so CTX=huge's forced PIECEWISE is no
  # protection) and it is NOT Bug B, which is fixed. Turning the lookup off is the only
  # known-correct setting, and it is what the long block was FOR, so this combination has
  # no fast correct configuration yet. Fall back rather than serve wrong turn-2 answers.
  # -z "${LOOKUP:-}" so an EXPLICIT LOOKUP=1 still gets through: someone reproducing this
  # needs to be able to ask for the broken combination on purpose.
  if [ -z "${LOOKUP:-}" ] && [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ] \
     && [ "$CTX" = "huge" ] && [ "${PREFIX_CACHE:-0}" = "1" ]; then
    echo "DFLASH_TOKENS>7 with CTX=huge and PREFIX_CACHE=1 corrupts the second turn over a" >&2
    echo "shared prefix (lookup drafting on the KVarN cache). Forcing the lookup off, which" >&2
    echo "is correct but slow (4.5 vs 14.3 tok/step): use DFLASH_TOKENS=7 for this mode, or" >&2
    echo "CTX=fast/long if you want the long verify block. LOOKUP=1 asks for it anyway." >&2
    export VLLM_DFLASH2_LOOKUP=0
  fi
  if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
    # Adaptive block length means the worker tells the scheduler how many draft tokens to
    # put up for verification next step, and vLLM only feeds that back on the synchronous
    # scheduling path (async scheduling pads every decode step to num_speculative_tokens).
    # Measured cost of losing async scheduling at batch 1: under 1%.
    ASYNC_SCHED=${ASYNC_SCHED:-0}
  fi
  # Memory: patches/hybrid-kv-groups-v2-cudagraph.patch stops the drafter's 5
  # sliding-window layers from padding the target's attention/GDN layers (78 instead of
  # 105 KB of pool per token), which is what makes 64k reachable here. The V2 runner's
  # profiled activation peak swings ~1 GiB between starts, so the pool is pinned by bytes
  # rather than by gpu-memory-utilization: 5.2 GiB -> 69,758 tokens = 1.06x at 64k,
  # leaving ~1.1 GiB for transients (the same margin MTP mode runs with). Soak-tested
  # with a 60k prompt, 4x16k concurrent and 8x4k generations. Lower it if you also run
  # something else on the card; KV_MEM= (empty) falls back to GPU_UTIL.
  #
  # A longer verify block costs pool twice: bigger CUDA graphs, and one aligned recurrent
  # state page per speculative block. That second term is what scales -- NOT the slot
  # count: 1 slot and 8 slots differ by about 8 MiB in total, so cutting MAX_SEQS buys no
  # context. Single-user mode keeps 4 slots when the block is long for the graphs.
  if [ "$CTX" = "huge" ]; then
    # KVarN pool: ~20 KB/token effective. 5.26 GiB pinned -> 268,169 tokens of
    # KV at 245760 max-model-len with 2 slots (single-user long-context; the
    # graphs stay at the k=7 size).
    MAX_SEQS=${MAX_SEQS:-2}
    MAX_LEN=${DFLASH_MAX_LEN:-245760}
    KV_MEM=${KV_MEM-5261334938}
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
  elif [ "$CTX" = "long" ]; then
    # int8 KV: measured 136,429 tokens of pool at DFLASH_TOKENS=7 with prefix caching on
    # (138,696 without), against bf16's 69,758 in the same pinned 5.2 GiB. DFLASH_TOKENS>7
    # at this context is untested -- the graphs and the state pages both grow.
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-131072}
    KV_MEM=${KV_MEM-5583457484}
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
    else
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
    fi
  elif [ "$DRAFT_TOKENS" -gt 7 ]; then
    # 4 slots and 56k instead of 8 and 64k: the aligned state pages and the bigger decode
    # graphs are what the long block costs, and this is where they still fit next to the
    # 5.2 GiB pool (57,669 tokens). DFLASH_TOKENS=7 gets 8 slots and 64k back.
    MAX_SEQS=${MAX_SEQS:-4}
    MAX_LEN=${DFLASH_MAX_LEN:-57344}
    KV_MEM=${KV_MEM-5583457484}
    # Decode graphs are captured for both block lengths (the drafter's and the full verify
    # block), or the short step -- the common one -- runs piecewise and costs 8%. That is
    # 1.8 GiB of graphs instead of 1.45.
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
  else
    MAX_LEN=${DFLASH_MAX_LEN:-65536}
    KV_MEM=${KV_MEM-5583457484}
    # If you tune GPU_UTIL instead, make the V2 runner count its CUDA graphs (~1.2-1.3 GiB
    # at these capture sizes) as well:
    export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
  fi
  MAX_SEQS=${MAX_SEQS:-8}
  # The V2 model runner captures decode graphs in multiples of k+1 tokens: cover MAX_SEQS requests.
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1)))}
  [ -n "$KV_MEM" ] && EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"
else
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-32}
fi

# PREFIX_CACHE=1: reuse the KV of a shared prompt prefix across requests, and resume the
# recurrent (GDN) state from the last cached block boundary instead of re-running the prompt.
# Turn-2+ of a chat with a 24k document goes from ~23 s to ~1 s; costs one extra state page
# per request (~16% of the KV pool). Hybrid models keep this opt-in upstream.
if [ "${PREFIX_CACHE:-0}" = "1" ]; then
  EXTRA_ARGS="--enable-prefix-caching --mamba-cache-mode align ${EXTRA_ARGS}"
  # KVarN runs --block-size 128; match the prefix hash unit to its tile so cache
  # hits land on tile boundaries (a non-multiple of 128 corrupts the pool).
  [ "$CTX" = "huge" ] && EXTRA_ARGS="--prefix-match-unit 128 ${EXTRA_ARGS}"
  # DFlash2 only: prefix caching and a CAPTURED (FULL) verify step do not mix on
  # that path. It is the capture, not the drafter: eager is clean, and so is
  # PIECEWISE, which keeps the compiled graphs and leaves only the multi-query
  # verify uncaptured.
  #   long context, labd copy@20k   FULL 1.97 tok/step 38 tok/s
  #                                 PIECEWISE 7.83 tok/step 132 tok/s   (3.5x)
  #   short prompts, de/en/code     FULL 78/125/202 tok/s
  #                                 PIECEWISE 74/102/176 tok/s          (-13..18%)
  # Read that -13..18% as SHORT PROMPTS ONLY. Past 8k the two modes are within
  # noise on bare metal (8k/16k/32k/50k: 112/78/69/58 FULL vs 109/86/73/56
  # PIECEWISE), so PIECEWISE is close to free at the lengths this mode is for.
  # Under GPU passthrough on a VM it costs 2-3x instead (PR #13): the uncaptured
  # verify is launch-bound, and launches are not free there.
  # The capture mode is fixed at boot, so CTX=huge takes the trade. Treat
  # CUDAGRAPH_MODE=FULL_AND_PIECEWISE as unsafe: what it does is corrupt one
  # prompt length in every 128 (gotcha 37, bench/bugb_sweep.py), which a coarse
  # sweep reads as a length threshold and a lucky sweep misses entirely.
  # This is NOT dflash2-only, which is what we used to claim here. SPEC=mtp with
  # CTX=huge captured FULL by default and has the same bug: at the residue that
  # matches its 4-token verify block it stops immediately, returning "" or "#"
  # with finish_reason=stop while every other residue is 794/794 verbatim. The
  # broken residue is R = 117 + k: 124 at DFLASH_TOKENS=7, 122 at 5, 120 at 3,
  # and the same 120 under mtp k=3. Equivalently the last 128-token tile has
  # 11-k free slots, i.e. verify block + free = 12 in every config measured.
  # k=5 is what rules out the attention block size as the driver -- same 2176
  # block as k=7, different residue. mtp and dflash2 at the same draft count
  # break identically, so the drafter is not implicated and every KVarN
  # speculator reaches it. It also needs a prefix-cache HIT to
  # fire at all, which is why PREFIX_CACHE=0 always looked clean (PR #13).
  # Correctness first, and it is close to free for MTP as well -- same depth
  # ladder, only the capture toggled, 8k/16k/32k/50k:
  #   FULL      87.8 / 86.1 / 70.4 / 63.5 tok/s
  #   PIECEWISE 93.5 / 83.8 / 70.3 / 59.6 tok/s
  # so PIECEWISE now covers the whole of CTX=huge, not just dflash2. The old
  # claim here that forcing it "would cost decode for nothing" was wrong twice.
  [ "$CTX" = "huge" ] &&
    CG_MODE=",\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-PIECEWISE}\""
fi

# ASYNC_SCHED=0 (set above for a long DFlash2 verify block) runs the scheduler
# synchronously, which is the only path on which vLLM lets the worker choose how many draft
# tokens to put up for verification. Note --async-scheduling is already the default in
# 0.27.1: --no-async-scheduling is what turns it off.
ASYNC_ARGS=$([ "${ASYNC_SCHED:-1}" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)

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

export PATH="$REPO/venv/bin:$PATH"
# Overridable: expandable_segments needs CUDA VMM, which WSL2's paravirt
# driver rejects ("CUDA driver error: device not ready" during Marlin repack)
# — set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False in .env on WSL2.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export VLLM_USE_FLASHINFER_SAMPLER=0

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
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  $ATTN_ARGS \
  --mamba-ssm-cache-dtype float16 \
  ${ASYNC_ARGS} \
  --max-num-batched-tokens 2048 \
  --speculative-config "$SPEC_CFG" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]${CG_MODE}}" \
  --reasoning-parser qwen3 \
  ${TOOL_ARGS} \
  ${EXTRA_ARGS}
