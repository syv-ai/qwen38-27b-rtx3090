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
#  - patches/0007-spec-decode-attn.patch: split-KV attention for the 5-query verify
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
#   KNOW THE EXPOSURE: fp8 KV has exactly ONE backend on sm86 -- FLASH_ATTN
#   refuses it (needs FA3/SM90+) and TRITON_ATTN refuses it (needs SM89+),
#   both measured -- so this tier runs FlashInfer with no A/B possible, and
#   issue #34 tracks a deterministic Xid-31 MMU write-fault seen twice on one
#   3090 under fp8+MTP+prefix caching at ~28-34k context. The flashinfer-free
#   fallback is the int8 tier (gotcha 44): what SPEC=dflash2 CTX=long already
#   ships, or for mtp: VLLM_SPEC_DECODE_ATTN=1 EXTRA_ARGS="--attention-backend
#   =TRITON_ATTN --kv-cache-dtype=int8_per_token_head" at ~25% wall cost at
#   depth (23.7 vs 18.9 s for 17.9k in + 256 out, measured).
# CTX=huge: KVarN 4/2-bit KV cache (kvarn/), 200k context with MTP, at roughly
#   half the decode rate past 100k — see below and docs/long-context.md.
#
# Fast variant: MODEL defaults to models/Qwen3.8-27B-W4A16-AutoRound-fast when it
# exists (int4-GPTQ lm_head + MTP, own-output draft vocab; drafter/README.md), else
# the base dir (int8 lm_head/MTP: ~108/107 tok/s with the shipped draft vocab).
#
# max-num-seqs is 8 here: fewer state slots to reserve (each request holds
# k+1 recurrent-state slots), and past a handful of concurrent users you
# should be running batch mode anyway. Int8 activations buy nothing at
# batch size 1 *decode* (memory-bound) — but prefill is compute-bound at any
# concurrency, so INT8_ACT below borrows batch mode's W4A8 path for the
# prompt-side win; see "Prefill" in ../batch/README.md.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# flashinfer-cubin (the no-nvcc route, README Setup) publishes 0.6.13 against
# flashinfer-python 0.6.16.post3; without this the import refuses the pair (#35).
export FLASHINFER_DISABLE_VERSION_CHECK=1

# A dead engine leaves its OffloadingConnector region behind as
# /dev/shm/vllm_offload_*.mmap; the next boot then dies with OSError: Bad
# address in shared_offload_region.py, and under restart policies that loops
# (#33 found 70 ghosts after one host OOM). Unlink stale regions no live
# process maps. VLLM_OFFLOAD_KEEP_SHM=1 skips this (several engines sharing
# /dev/shm across namespaces, where the liveness scan cannot see the owner).
if [ "${VLLM_OFFLOAD_KEEP_SHM:-0}" != 1 ]; then
  for f in /dev/shm/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    grep -lqs "$f" /proc/[0-9]*/maps 2>/dev/null || { echo "[start_qwen] removing stale offload region $f"; rm -f "$f"; }
  done
fi
REPO="$(dirname "$DIR")"
cd "$REPO"

if [ -z "$MODEL" ] && [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
PORT=${PORT:-18020}
MAX_SEQS=${MAX_SEQS:-}
# INT8_ACT=int8 turns on the W4A8 Marlin path (weights stay int4, activations
# quantized per token to int8, int8 tensor cores) for the layers INT8_LAYERS
# selects — the same knob batch mode ships on by default. At batch size 1 it
# does nothing for decode (memory-bound; C1 122±5 vs 121 tok/s over repeats,
# tok/step 3.2 either way) but prefill is compute-bound at every concurrency
# — a 4k prefill on this stack is 79% Marlin GEMM time with 15 ms of GPU idle
# (torch profile), so the GEMM dtype is the whole game. Measured on the
# seeded protocol (bench/run_benchmarks.sh --prefill), dflash2 k=15, PC=1:
#
#   prefill tok/s        1k     4k     16k    51k
#   W4A16 (default)    1,437  1,494  1,410  1,200
#   INT8_LAYERS=mlp    1,638  1,696  1,587  1,320   (+14/+13/+13/+10%)
#   INT8_LAYERS=all*   1,845  1,937  1,791  1,423   (+29/+30/+27/+19%)
#
#   *all = "mlp|linear_attn|self_attn" — every linear except the int8-weight
#    lm_head/embed and the MTP module. The 51k row gains least because 16
#    full-attention layers grow quadratically (~40% of time at 51k) and FA2 at
#    head_dim 256 has no faster sm86 alternative (FlashInfer measured within
#    1.5%).
#
# Quality (bench/quality_battery.py, this stack): all-linears = GSM8K 95.0%
# (fast-variant baseline 96.5), PPL 8.423 vs 8.095 (+4.1%, mostly prose;
# code ~flat); mlp = the batch-mode trade (+2.2% PPL, GSM8K 95.0). IFBench
# was flat on the batch int8 default. "mlp|linear_attn" (GDN-only middle
# point) crashes at first forward on this torch/vllm combo — an inductor
# codegen bug with the mixed set; use mlp or all.
INT8_ACT=${INT8_ACT-}
INT8_LAYERS=${INT8_LAYERS-mlp|linear_attn|self_attn}
[ -n "$INT8_ACT" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
[ -n "$INT8_ACT" ] && [ -n "$INT8_LAYERS" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS
# PREFILL_ATTN=int8: int8-QK Triton attention for the hd256 full-attention
# layers during prefill (patches/0024-prefill-attn-int8.patch): 1.27-1.35x FA2 on
# the attention itself, worth up to ~+5% end-to-end at 51k on top of INT8_ACT
# (1,839/1,498 tok/s at 16k/51k with both on). Prefill-only; decode and the
# split-KV verify keep their existing paths. fp16 selects the same kernel
# without quantization (a debugging mode); empty keeps FA2.
PREFILL_ATTN=${PREFILL_ATTN-}
[ -n "$PREFILL_ATTN" ] && export VLLM_PREFILL_ATTN=$PREFILL_ATTN
# 0.93 here, NOT batch mode's 0.972: the DeltaNet workspace in the MTP decode
# path allocates beyond the startup memory profile (docs/gotchas.md, gotcha 4).
GPU_UTIL=${GPU_UTIL:-0.93}
API_SERVERS=${API_SERVERS:-1}
# CTX=fast (default): bf16 KV via FlashAttention, ~64k context, 4 drafts (~+7%).
# CTX=long: fp8 KV via FlashInfer, 150k context, 3 drafts.
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
#   (patches/0009-dflash2-backport.patch). CTX=fast (bf16, 64k), CTX=long (int8,
#   128k) or, with kvarn/install.sh, CTX=huge (KVarN 4/2-bit, 240k + prefix
#   caching); see README "DFlash2".
# SPEC=off (or none): no speculative decoding at all. This used to fall through
#   to the mtp branch silently -- an A/B with "SPEC=off" was really running two
#   spec-on arms, which is exactly the trap PR #46's campaign walked into with
#   alternative.sh. Any other value now refuses instead of proceeding.
SPEC=${SPEC:-mtp}
# SPEC_ATTN=1: split-KV Triton attention for the multi-query verify step
# (patches/0007-spec-decode-attn.patch); bf16 KV only, so CTX=fast only.
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
  # tokens instead of 69,758, because patches/0012-hybrid-sw-block-promote.patch stops the
  # drafter's 5 sliding-window layers from taking 385 nearly-empty blocks, and
  # patches/0013-spec-decode-int8-kv.patch lets the split-KV verify kernel read the quantized
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
  # (patches/0011-dflash2-lookup-drafting.patch).
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
  # The split-KV verify attention (patches/0007-spec-decode-attn.patch) sizes its partial
  # buffers once for the longest query block it will see -- a captured CUDA graph holds
  # their addresses, so they must not be grown later.
  export VLLM_SPEC_DECODE_ATTN_QMAX=${VLLM_SPEC_DECODE_ATTN_QMAX:-$((DRAFT_TOKENS + 1))}
  # The ADAPTIVE verify length corrupts a prefix-cache hit under KVarN. When the block
  # alternates 8<->16 step to step and the request resumed from a cache hit, turn 2 over
  # the same document tracks the source for ~38 characters and then diverges -- turn 1 is
  # correct. Deterministic, at five of six prompt residues tested.
  #
  # It is the length CHANGING, not the lookup content, and not the KVarN kernels. Fresh
  # server per row, three requests each (one to arm the cache, two measured), sha-compared:
  #   baseline, adaptive 8<->16          self-hit 38/793 WRONG   12.71 tok/step
  #   VLLM_DFLASH2_LOOKUP_ADAPTIVE=0     self-hit 794/794 clean  14.71 tok/step
  #   PREFIX_CACHE=0, adaptive on        self-hit 794/794 clean  14.33
  #   KVARN_FUSED_VERIFY=0, adaptive on  self-hit 38/793 WRONG   12.71
  # and the constant-length setting is clean at every residue that broke (16/64/96/124),
  # cold and warm, byte-identical.
  #
  # Note what the earlier controls actually varied: DFLASH_TOKENS=7, LOOKUP=0 and SPEC=mtp
  # all make the block a CONSTANT length, so "clean" there was never evidence about the
  # block being short or the lookup being off. And a wrong draft cannot corrupt greedy
  # output at all -- rejection_sampler.py emits target_argmax whether it accepts or not --
  # so the draft content was never a candidate. The damage is on the target's forward.
  #
  # Pinning the length is not a sacrifice here: 14.71 tok/step is the FASTEST number in the
  # series, above the adaptive path's own 14.29 cold. Root cause still open; this is a
  # correct and fast setting, not a workaround with a cost.
  if [ -z "${LOOKUP_ADAPTIVE:-}" ] && [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ] \
     && [ "$CTX" = "huge" ] && [ "${PREFIX_CACHE:-0}" = "1" ]; then
    echo "DFLASH_TOKENS>7 + CTX=huge + PREFIX_CACHE=1: pinning the verify block to" >&2
    echo "$((DRAFT_TOKENS + 1)) tokens (VLLM_DFLASH2_LOOKUP_ADAPTIVE=0). The adaptive length" >&2
    echo "corrupts the second turn over a shared prefix on the KVarN cache; pinned is both" >&2
    echo "correct and faster. LOOKUP_ADAPTIVE=1 asks for the adaptive path anyway." >&2
    export VLLM_DFLASH2_LOOKUP_ADAPTIVE=0
  fi
  if [ "$VLLM_DFLASH2_LOOKUP" = "1" ] && [ "$DRAFT_TOKENS" -gt 7 ]; then
    # Adaptive block length means the worker tells the scheduler how many draft tokens to
    # put up for verification next step, and vLLM only feeds that back on the synchronous
    # scheduling path (async scheduling pads every decode step to num_speculative_tokens).
    # Measured cost of losing async scheduling at batch 1: under 1%.
    ASYNC_SCHED=${ASYNC_SCHED:-0}
  fi
  # Tensor parallelism changes two calibrations below, both measured in #40
  # (controlled 1-vs-2x3090 A/B on this harness):
  #
  # 1. The pinned KV_MEM is a 24-GiB-single-card constant and vLLM applies
  #    --kv-cache-memory PER WORKER, so at TP=2 the pin strands ~8 GiB per card:
  #    137,210 tokens of pool where GPU_UTIL sizing gets 302,223, with every
  #    decode delta inside run-to-run spread. Under TP>1, unless the user pinned
  #    one, size from GPU_UTIL instead. (The pin exists because the single-card
  #    transient margin is sharp -- gotcha 39; TP halves the per-card footprint
  #    and the same reporter's boxes boot clean unpinned.)
  # 2. DFLASH_TOKENS>7 relies on the lookup lane filling the verify tail; at
  #    TP=2 the one datapoint so far (#40) shows the tail accepting nothing and
  #    the wider block costing -27% at C1. Until that is diagnosed, warn.
  TP_SIZE=1
  case " ${EXTRA_ARGS:-} " in *"-tensor-parallel-size"*|*" -tp "*)
    TP_SIZE=$(printf %s " $EXTRA_ARGS" | sed -En "s/.* (--tensor-parallel-size[= ]|-tp )([0-9]+).*/\2/p")
    TP_SIZE=${TP_SIZE:-1}
  ;; esac
  if [ "$TP_SIZE" -gt 1 ]; then
    if [ -z "${KV_MEM+x}" ]; then
      echo "[start_qwen] tensor-parallel-size $TP_SIZE: skipping the single-card KV_MEM" \
           "pin, sizing the KV pool from GPU_UTIL=$GPU_UTIL (issue #40; export KV_MEM" \
           "to pin it, KV_MEM= for this behavior explicitly)."
      KV_MEM=
    fi
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      echo "[start_qwen] WARNING: DFLASH_TOKENS=$DRAFT_TOKENS at TP=$TP_SIZE lost 27% at" \
           "C1 in the one A/B so far (#40): the lookup-filled verify tail accepted" \
           "nothing there. Until diagnosed, DFLASH_TOKENS=7 is the measured setting" \
           "for TP>1." >&2
    fi
  fi
  # Memory: patches/0010-hybrid-kv-groups-v2-cudagraph.patch stops the drafter's 5
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
    # KVarN pool: ~20 KB/token effective. 4.90 GiB pinned -> 268,169 tokens of
    # KV at 245760 max-model-len with 2 slots (single-user long-context; the
    # graphs stay at the k=7 size).
    MAX_SEQS=${MAX_SEQS:-2}
    # 221184 rather than 245760 above 7 drafts, the same trade CTX=fast makes at
    # DFLASH_TOKENS>7. The pinned pool is 4.90 GiB; one request at max_model_len needs
    # 5.16 GiB at k=15, so the server refuses to start at 245760 -- "5.16 GiB KV cache is
    # needed, which is larger than the available KV cache", which is the boot failure an
    # independent 3090 Ti reported (PR #13). Raising the cudagraph reservation does NOT
    # fix this and I checked: KV_MEM is pinned, so the pool does not grow when graph
    # memory is accounted differently. 221184 = 1728 x 128 leaves ~2.6% margin (229376 was still 1% short).
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      MAX_LEN=${DFLASH_MAX_LEN:-221184}
    else
      MAX_LEN=${DFLASH_MAX_LEN:-245760}
    fi
    KV_MEM=${KV_MEM-5261334938}
    # Above 7 drafts the decode graphs are captured for BOTH block lengths, which is
    # ~1.8 GiB rather than ~1.45 -- the same arithmetic CTX=long and CTX=fast already
    # branch on. This branch did not, and said so in a comment ("the graphs stay at the
    # k=7 size") that stops being true the moment anyone sets DFLASH_TOKENS. The pool is
    # then sized as if that memory were free and the server does not come up at 240k on
    # 24 GB, which is what an independent 3090 Ti report hit (PR #13).
    if [ "$DRAFT_TOKENS" -gt 7 ]; then
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1900}
    else
      export VLLM_V2_CUDAGRAPH_MEM_MIB=${VLLM_V2_CUDAGRAPH_MEM_MIB:-1400}
    fi
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
  # The V2 model runner captures decode graphs in multiples of k+1 tokens: cover MAX_SEQS
  # requests, but never ask for more than 64 query tokens' worth. Every default in this
  # script lands on 64 or below (8x8 at k=7, 4x16 at k=15), and VLLM_V2_CUDAGRAPH_MEM_MIB
  # above is sized for that. `DFLASH_TOKENS=15 MAX_SEQS=8` asks for 128, which boots and
  # then dies on the first concurrent batch -- torch.OutOfMemoryError inside the engine,
  # EngineDeadError, every request 500 while /health still answers. Past the cap the
  # bigger batches run piecewise instead of captured: slower, alive. Set CG explicitly to
  # override, and raise VLLM_V2_CUDAGRAPH_MEM_MIB with it.
  CG=${CG:-$((MAX_SEQS * (DRAFT_TOKENS + 1) > 64 ? 64 : MAX_SEQS * (DRAFT_TOKENS + 1)))}
  # Seats are admissions, not residency. Every RESIDENT request needs 1+k recurrent-state
  # slots out of the same pool before it stores one token of context: 0.88 GiB at
  # DFLASH_TOKENS=7 and 1.66 GiB at 15 (gotcha 33), i.e. ~0.098 GiB per (k+2), which the
  # ramp in bench/conc_ladder.py confirms live (15.8% of the CTX=fast pool per request,
  # six resident, the seventh preempted). Print what the pool can actually hold so nobody
  # reads MAX_SEQS as a concurrency setting -- raising it past this queues, and once the
  # pool is full it preempts and recomputes.
  if [ -n "$KV_MEM" ]; then
    RESIDENT=$(( KV_MEM / ((DRAFT_TOKENS + 2) * 104988089) ))
    [ "$RESIDENT" -lt 1 ] && RESIDENT=1
    if [ "$MAX_SEQS" -gt "$RESIDENT" ]; then
      echo "[start_qwen] note: the pinned pool holds about $RESIDENT resident requests at" \
           "DFLASH_TOKENS=$DRAFT_TOKENS (state pages), fewer with long prompts;" \
           "MAX_SEQS=$MAX_SEQS admits more than that and the rest queue."
    fi
  fi
  # Above ~12 seats at CTX=huge the seats stop being merely useless and become fatal, and
  # not through the graph budget that gotcha 38 caps -- CG is already pinned at 64 in
  # every one of these. The per-seat runner allocations eat the transient headroom the
  # first prefill needs, so the engine boots, captures, answers /health 200, and then
  # dies on the first real prompt with torch.OutOfMemoryError inside the caching
  # allocator. Measured on one 24 GiB 3090, SPEC=dflash2 k=7, free VRAM after boot
  # against a single ~3.7k-token prompt (gotcha 39):
  #   MAX_SEQS=8   596 MiB  ok      MAX_SEQS=12  416 MiB  ok
  #   MAX_SEQS=10  456 MiB  ok      MAX_SEQS=16  356 MiB  DEAD, twice, same byte counts
  # The transient set is elastic -- it squeezes to full speed with 8 MiB free -- but the
  # first real batch's per-seat allocations are not, so the floor is sharp: through the
  # KV_MEM door at 8 seats it sits between 396 (dead) and 436 MiB (fine) of free VRAM,
  # same allocator fingerprint as the seat door (gotcha 39, both tables). Warning rather
  # than a clamp: the number is a VRAM budget, and a card larger than 24 GiB will have
  # room where this one does not. Lower KV_MEM if you genuinely need the seats.
  if [ "$CTX" = "huge" ] && [ "$MAX_SEQS" -gt 12 ]; then
    echo "[start_qwen] WARNING: MAX_SEQS=$MAX_SEQS at CTX=huge killed the engine on the" \
         "first prompt on a 24 GiB card (boots, serves /health, then OutOfMemoryError)." \
         "12 is the highest verified here. Lower MAX_SEQS or lower KV_MEM." >&2
  fi
  # The same room through the other door: a KV_MEM pinned above the profile default
  # spends the same headroom. On Linux the shortfall is a loud OutOfMemoryError on the
  # first real batch; on WSL2 it is SILENT -- WDDM backs the failed mapping with host
  # memory and prefill quietly runs 5-10x slower with nothing in the logs (issue #25).
  KV_STOCK=5583457484; [ "$CTX" = "huge" ] && KV_STOCK=5261334938
  if [ -n "$KV_MEM" ] && [ "$KV_MEM" -gt "$KV_STOCK" ]; then
    echo "[start_qwen] WARNING: KV_MEM=$KV_MEM is above the profile default $KV_STOCK." \
         "The extra pool is taken from the headroom prefill and the first concurrent" \
         "batch allocate from, and the floor is close: +150 MiB ran, +200 MiB killed" \
         "the engine at CTX=huge MAX_SEQS=8 (gotcha 39 has the ladder). On Linux the" \
         "miss is a loud OutOfMemoryError; on WSL2 it is SILENT: no error, prefill" \
         "5-10x slower. Ladder 4k/16k TTFT against known-good rates before trusting it." >&2
  fi
  [ -n "$KV_MEM" ] && EXTRA_ARGS="--kv-cache-memory=$KV_MEM ${EXTRA_ARGS}"
elif [ "$SPEC" = "off" ] || [ "$SPEC" = "none" ]; then
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG=""
  CG=${CG:-32}
elif [ "$SPEC" = "mtp" ]; then
  MAX_SEQS=${MAX_SEQS:-8}
  SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$DRAFT_TOKENS,\"draft_sample_method\":\"${DRAFT_SAMPLE:-probabilistic}\"}"
  CG=${CG:-32}
else
  echo "SPEC=$SPEC is not a mode: mtp (default), dflash2, off. Refusing rather than" >&2
  echo "silently running mtp -- an unrecognized SPEC in an A/B measures the wrong thing." >&2
  exit 1
fi
SPEC_ARGS=()
[ -n "$SPEC_CFG" ] && SPEC_ARGS=(--speculative-config "$SPEC_CFG")

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
  #   short prompts, de/en/code     FULL 78/125/202 tok/s
  #                                 PIECEWISE 74/102/176 tok/s          (-13..18%)
  # The long-context row that used to sit above this one -- FULL 1.97 tok/step / 38 tok/s
  # against PIECEWISE's 7.83 / 132, "3.5x" -- was measuring the residue bug, not the
  # capture mode, and a75ee4b fixed it. Re-measured at HEAD with only the capture
  # toggled (labd_bench --ctx 20000, dflash2 CTX=huge PREFIX_CACHE=1, decode tok/s):
  #   copy/code/edit/quote/summary/qa  FULL      167/111/85/55/48/43   all six 65.7
  #                                    PIECEWISE 166/111/83/62/49/43   all six 67.6
  # i.e. the same, with `quote` diverging the way greedy does. At this context length
  # the capture mode is not a speed decision in either direction.
  # Treat that -13..18% as an UPPER bound. @mjungnickel18 measures 0.2-2.3% for the
  # same comparison on bare metal when only runs with identical STEP COUNTS are
  # compared, and he is right that greedy runs which take a different number of steps
  # are not comparable -- mine did not control for that. Unresolved: I have not
  # re-measured with his method. The number that is not in dispute is the long-context
  # one above, where the gap is 3.5x and no amount of step-count matching closes it.
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
  # SPEC=mtp keeps PIECEWISE for CORRECTNESS, not preference. Do not "optimise" this
  # away: under FULL, mtp at CTX=huge still breaks at one prompt length in 128 --
  # templated length == L (mod 128), L = k+1 = 4 -- with a prefix cache hit. Measured on
  # current main, fresh server, residues 3/4/5:
  #   residue 3   794/794    residue 4   0 chars, finish_reason=stop    residue 5   794/794
  # The LOCATION is deterministic; the DAMAGE is not. The same residue has returned an
  # empty answer here, a one-character answer, and -- on a third geometry (MAX_LEN=240000,
  # tool parser attached) -- 400 tokens of fluent Danish inventing a translation task,
  # 2 of 1146 characters matching the document. So do not test for a symptom; test
  # whether the copy came back (bench/verbatim.py).
  # @mjungnickel18 found it by sweeping all 128 residues (127 clean, one broken, and it
  # is 4); an earlier version of this comment claimed mtp was "correct under FULL" on
  # five sampled residues that did not include 4 -- five distinct samples miss a single
  # broken residue C(127,5)/C(128,5) = 123/128 = 96% of the time, not the 82% this
  # comment used to say. a75ee4b fixed the same shape for dflash2 but does not
  # cover this path.
  #
  # dflash2 does get FULL, and to the same evidence standard: ALL 128 residues swept
  # under FULL, self-hit measured, 0 broken -- so this is not the five-sample luck the
  # mtp claim was. On top of that FULL is 96.5% GSM8K against PIECEWISE's 95.0%, and
  # worth 2-3x under GPU passthrough (PR #13), where the uncaptured verify is
  # launch-bound rather than bandwidth-bound. THAT one is a preference and may be
  # revisited; the mtp line above may not, until residue 4 comes back verbatim.
  if [ "$CTX" = "huge" ] && [ "$SPEC" != "dflash2" ]; then
    CG_MODE=",\"cudagraph_mode\":\"${CUDAGRAPH_MODE:-PIECEWISE}\""
  fi
fi
# An explicit CUDAGRAPH_MODE is honoured on every path, including dflash2, which
# otherwise takes vLLM's default. Without this branch there was no way to ask a
# dflash2 server for PIECEWISE, so the FULL-against-PIECEWISE comparison this repo
# quotes could not be re-measured after a75ee4b changed what FULL does.
if [ -n "${CUDAGRAPH_MODE:-}" ] && [ -z "${CG_MODE:-}" ]; then
  CG_MODE=",\"cudagraph_mode\":\"$CUDAGRAPH_MODE\""
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
  # the GPU for the duration of its own forward (patches/0015-vision-tower-cpu-offload.patch).
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

# fp16 activations do not work with the speculative path, and the way you find that out
# is late and cryptic (#27): the split-KV verify kernel hardcodes tl.bfloat16 for the
# query cast and the dot accumulate (patches/0007-spec-decode-attn.patch:217,236), so it
# fails to *compile* at first attention with "Both operands must be same dtype. Got bf16
# and fp16" -- a triton CompilationError that never names the dtype you set.
#
# Scope, stated honestly because an earlier version of this comment overreached: the
# BLOCKER I can point at is that kernel. @ahnguyen17 also hit a device-side assert in
# rejection_sample() with SPEC_ATTN=0, and I first wrote that up as a second bf16
# assumption -- it is not. rejection_sampler_utils.py contains zero bf16/fp16 literals;
# it promotes to tl.float32 on load and allocates its buffers float32/int64, so it is
# dtype-agnostic. That assert has some other cause (it was seen on a w8a16 GPTQ target
# whose GEMM path was already suspect, and a device-side assert reads like an
# out-of-bounds index, not a dtype mismatch), and it is not established.
#
# So this refuses the combination rather than claiming to enumerate why it breaks:
# fp16 + a speculator is untested here and its default path is bf16-only. SPEC=none is
# the escape hatch -- a limitation of these kernels, not of any checkpoint.
case " ${EXTRA_ARGS:-} " in
  *" --dtype float16 "*|*" --dtype=float16 "*|*" --dtype fp16 "*|*" --dtype=fp16 "*|*" --dtype half "*|*" --dtype=half "*)
    if [ "$SPEC" != "none" ] && [ "$SPEC" != "off" ]; then
      echo "--dtype float16 needs SPEC=off: this repo's speculative path is bf16-only." >&2
      echo "  the split-KV verify attention casts to tl.bfloat16 (patches/0007-spec-decode-attn.patch)," >&2
      echo "  so it fails to compile at the first attention. See issue #27." >&2
      exit 1
    fi ;;
esac

export PATH="$REPO/venv/bin:$PATH"
# expandable_segments needs CUDA VMM, which WSL2's paravirt driver rejects during
# Marlin repack. It is the single most reported failure on Windows (#2, #26) and it
# does not announce itself as an allocator problem -- the same VMM rejection surfaces
# as "CUDA driver error: device not ready", "CUDA driver error: out of memory" (on a
# card with 23 GiB free for a 16 GiB model) or a torch stable-ABI error out of
# aten::empty, with dmesg carrying "dxgkio_make_resident: Ioctl failed: -12". So
# detect WSL and default it off there rather than documenting a workaround: three
# separate reporters found the note in docs/docker.md only after losing a day.
# Still overridable both ways, and untouched on native Linux, where gotcha 3 applies
# and turning it off costs you the top of the GPU_UTIL range.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null || [ -n "${WSL_DISTRO_NAME:-}" ]; then
  ALLOC_DEFAULT=expandable_segments:False
  [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && echo \
    "WSL detected: PYTORCH_CUDA_ALLOC_CONF=$ALLOC_DEFAULT (VMM breaks Marlin repack under the paravirt driver; set it explicitly to override)"
else
  ALLOC_DEFAULT=expandable_segments:True
fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-$ALLOC_DEFAULT}
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ -z "$VLLM_API_KEY" ] && [ -f "$REPO/api_key.txt" ]; then
  export VLLM_API_KEY="$(cat "$REPO/api_key.txt")"
fi

exec venv/bin/vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host ${HOST:-0.0.0.0} --port $PORT \
  --gpu-memory-utilization $GPU_UTIL \
  --max-model-len $MAX_LEN \
  --max-num-seqs $MAX_SEQS \
  --api-server-count $API_SERVERS \
  ${VISION_ARGS} \
  $ATTN_ARGS \
  --mamba-ssm-cache-dtype float16 \
  ${ASYNC_ARGS} \
  --max-num-batched-tokens 2048 \
  "${SPEC_ARGS[@]}" \
  --compilation-config "{\"max_cudagraph_capture_size\":$CG,\"custom_ops\":[\"+rms_norm\",\"+silu_and_mul\"]${CG_MODE}}" \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  ${TOOL_ARGS} \
  ${EXTRA_ARGS}
