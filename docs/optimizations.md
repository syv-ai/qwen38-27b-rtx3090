# What this repo does that stock vLLM doesn't

Nine things stock vLLM doesn't give you on this model — summarised, then explained — plus the two speculative-decoding modes. For what each one is worth in tokens per second, see [what each step buys](../README.md#what-each-step-buys).

[← back to the main README](../README.md)

## The short version

One line each; the rest of this page is the long version.

1. **Both embedding matrices requantized** (`prepare/quant_lm_head.py`, `prepare/quant_embed.py`)
   — the public W4A16 quants leave two 2.5 GB bf16 matrices alone. 2.6 GB back.
2. **A two-line vLLM patch** so the model code actually uses vLLM's quantized
   embedding kernel (`patches/0001-qwen3_5-embed-quant.patch`).
3. **16-bit recurrent state** — the GDN state, not the KV cache, is what bounds
   concurrency here: 37 of 64 requests were running before this.
4. **int8 tensor cores for the batched GEMMs, with a bug fix** — vLLM's W4A8
   Marlin path produces garbage on this checkpoint (negative group scales read
   as unsigned); two patches fix it and make it per-layer selectable.
5. **Cheap speculative drafts, and a draft vocabulary counted over the model's
   own outputs** — 97.5% coverage vs 92% for a web-text list, and every miss is
   a forced rejection. Worth 10% of single-stream throughput on its own.
6. **Two decode-path patches for the verify step** — split-KV attention for
   multi-query decode (FA2 leaves 58 of 82 SMs idle there) and a sort-free
   top-k/top-p sampler.
7. **Tuned flags that are easy to get wrong**, plus vLLM PR #50021 vendored for
   an illegal memory access in the DeltaNet spec-decode kernels.
8. **Speculation that reads the context** — when the model is reproducing
   something from its prompt, draft it from the prompt, and verify a longer
   block than the drafter can fill (`patches/0011-dflash2-lookup-drafting.patch`):
   **381 tok/s** reproducing a 25k-token document verbatim, against 260 for the
   first version of this and 159 without it, still lossless.
9. **Prefix caching for a hybrid model** — opt-in upstream; `PREFIX_CACHE=1`
   makes a follow-up chat turn on a 24k document cost ~1 s instead of ~23 s, and
   64 requests sharing a system prompt 17 s instead of 222 s.
10. **int8 prefill for single-user mode** (`INT8_ACT=int8`) — prefill is
   compute-bound at every batch size, so the W4A8 tensor-core path is worth
   +27-30% prefill (+19% at 51k) on the dflash2 stack with decode unchanged;
   plus a benchmark-harness fix (seeded prompts) that makes the prefill
   numbers honest at all.

## In full

1. **Both embedding matrices requantized.** Qwen3.8-27B has untied embeddings,
   so the public W4A16 quants carry two separate 2.5 GB bf16 matrices (lm_head
   and embed_tokens) that nobody bothered to quantize. `prepare/quant_lm_head.py` and
   `prepare/quant_embed.py` requantize both to int8 group-128 in place (~0.6%
   round-trip error, no quality regression we could find). That's 2.6 GB of
   VRAM back.
2. **A small vLLM patch for those embeddings.** vLLM ships a dequant-on-gather
   kernel for int-quantized embedding tables but the qwen3_5 model code never
   wires it up — neither in the main model nor in the MTP draft module.
   `patches/0001-qwen3_5-embed-quant.patch` fixes both (two lines each).
3. **16-bit recurrent state.** 48 of the 64 layers are Gated DeltaNet with a
   fixed recurrent state per sequence, and Qwen's config asks for it in fp32:
   ~150 MB per request, allocated up front, read and written on every decode
   step. On this architecture that state — not the KV cache — is what bounds
   concurrency: with `--max-num-seqs 64` only 37 requests were ever actually
   running (log line `Running: 37 reqs, Waiting: 27`). `--mamba-ssm-cache-dtype
   float16` halves the footprint and the traffic; all 64 run, and perplexity is
   unchanged to three decimals (fp16 keeps 10 mantissa bits; we did not use
   bf16's 7).
4. **int8 tensor cores for the batched GEMMs, with a bug fix.** At 40-64
   concurrent sequences the decode step is bound by fp16 tensor-core math
   (~63 TFLOPS sustained at 250 W). vLLM already has a W4A8-INT8 Marlin path
   (`VLLM_MARLIN_INPUT_DTYPE=int8`: weights stay int4, activations are
   quantized to int8 per token, the MMA runs on int8 tensor cores at 4× the
   rate) — but on this checkpoint it produced garbage while benchmarking
   beautifully. The kernel reads its int16-requantized group scales as
   *unsigned*, and AutoRound symmetric exports have ~50% negative scales.
   `patches/0003-marlin-int8-negative-scales.patch` folds the sign into the int4
   codes at load time; `patches/0002-marlin-int8-layer-select.patch` lets you pick
   which layers get int8 activations (and keeps it off the int8-weight lm_head,
   which would otherwise refuse to load).
5. **Cheap speculative drafts, and a draft vocabulary that covers what the
   model actually says.** The shipped MTP draft module is bf16 (850 MB) and
   every draft token also runs the full 248k-row lm_head (1.3 GB), so each
   extra draft cost ~3 ms and MTP-3 was already slower than MTP-2.
   `prepare/quant_mtp.py` requantizes the draft module (int8; the fast variant uses
   GPTQ int4, `drafter/`), `prepare/build_draft_vocab.py` builds a 40k-token draft head
   and `patches/0004-qwen3_5-mtp-draft-vocab.patch` makes the drafter use it. A
   draft now costs ~0.5-1 ms and four of them pay off. The id list matters
   more than anything else in this repo's single-user numbers: a token outside
   the draft vocabulary can never be proposed, so it is a guaranteed rejection
   that also cuts the chain. The list we now ship (`prepare/draft_vocab_ids.json`) is
   counted over 5.4M tokens of the model's own outputs and covers 97.5% of what
   it generates (96% on code); the earlier web-text list covered 92% (83% on
   code) and cost 10% of single-stream throughput on its own.
6. **Two decode-path patches for the multi-query verify step.**
   `patches/0007-spec-decode-attn.patch`: FlashAttention-2 only splits the KV
   sequence across thread blocks when a request has one query token; the MTP
   verify step has five, so a 24-head model runs attention on 24 of the 3090's
   82 SMs — 57 µs per layer at 1.5k context, 1.3 ms at 16k. A small Triton
   split-KV kernel replaces it (23 µs / 120 µs).
   `patches/0006-sampler-small-topk-fast-softmax.patch`: vLLM's top-k/top-p
   masking sorts the whole 248k vocab for every row and its softmax runs one
   thread block per row (140 µs for a single 248k-wide row, called several
   times per step); with top-k ≤ 64 known on the host the mask is one
   `torch.topk`, the softmax is multi-block, and drafts are sampled from the
   same truncated support as the target. Together +4% at default sampling.
7. **Tuned flags that are easy to get wrong**, each documented in the launch
   scripts and the gotchas below, plus vLLM PR
   [#50021](https://github.com/vllm-project/vllm/pull/50021) vendored as
   `patches/0005-vllm-pr50021-gdn-spec-bounds.patch` (bounds checks in the DeltaNet
   speculative-decode kernels; we hit the illegal-memory-access it fixes with
   several concurrent MTP requests).
8. **Speculation that reads the context.** A block drafter sees a 2,048-token window; a
   long-context assistant spends much of its output reproducing what it was given.
   `patches/0011-dflash2-lookup-drafting.patch` proposes the continuation of the most recent
   earlier occurrence of what was just generated — from anywhere in the request's own
   history — with a point-mass draft distribution so the verify stays exact. Because those
   tokens cost the drafter nothing, the verify block is no longer capped at the drafter's
   own (7 tokens), and the long block is only scheduled while the lookup is firing:
   reproducing a document verbatim goes 7.83 → 15.0 tokens per step, 260 → 381 tok/s.
9. **Prefix caching for a hybrid model, on purpose.** vLLM keeps it opt-in for
   mamba/GDN hybrids; `PREFIX_CACHE=1` turns it on in both modes with the recurrent state
   resumed from the last cached block boundary. Follow-up chat turns on a 24k document:
   23 s → 1 s. 64 API requests sharing a 5.8k system prompt: 222 s → 17 s.

### int8 prefill for single-user mode (`INT8_ACT=int8`)

Single-user mode stayed W4A16 because int8 activations buy nothing at batch
size 1 *decode* — true, and beside the point for prefill, which is
compute-bound at every concurrency. A torch profile of a 4k prefill on the
dflash2 stack is 79% Marlin GEMM time with 15 ms of GPU idle out of 2.05 s, so
the GEMM dtype is the whole game. `INT8_ACT=int8` (default layer set
`mlp|linear_attn|self_attn`) borrows batch mode's W4A8 path for every linear
except the int8-weight lm_head/embed and the MTP module:

| prefill tok/s (dflash2 k=15, PC=1, seeded protocol) | 1k | 4k | 16k | 51k |
|---|---|---|---|---|
| W4A16 (mode default) | 1,437 | 1,494 | 1,410 | 1,200 |
| `INT8_ACT=int8 INT8_LAYERS=mlp` | 1,638 | 1,696 | 1,587 | 1,320 |
| `INT8_ACT=int8` (all linears) | **1,845** | **1,937** | **1,791** | **1,423** |

Decode is unchanged (122±5 vs 121 tok/s C1 over repeats, 3.2 tok/step both
ways) and quality is the documented int8 trade: GSM8K 95.0% against the fast
variant's 96.5, perplexity +4.1% (mostly prose, code flat), IFBench flat on
the batch-mode precedent. The 51k row gains least because the 16
full-attention layers grow quadratically to ~40% of prefill there, and FA2 at
head_dim 256 has no faster sm86 alternative (FlashInfer measured within 1.5%,
and it costs the split-KV verify path at decode).

**int8-QK prefill attention** (`PREFILL_ATTN=int8`,
`patches/0024-prefill-attn-int8.patch`) attacks what is left after the GEMMs: the
16 full-attention layers, whose head_dim of 256 pins FA2 at 54-57 TFLOPS on
sm86 (85% of the card's practical fp16 mma rate — no fp16 rewrite can win).
A Triton kernel runs QK^T on int8 tensor cores at 2x the fp16 rate,
SageAttention-style: K is smoothed by its per-head channel mean (softmax-
invariant, so exact up to int8 rounding — cos > 0.99999 vs fp32 at 4-51k),
gathered and quantized once per layer-chunk into contiguous int8 scratch;
Q rows carry per-row scales; P.V stays bf16. On the attention itself it is
1.27x FA2 at 4k rising to 1.35x at 51k; end-to-end with `INT8_ACT` it adds
+2.7% at 16k and +5.3% at 51k (1,839 / 1,498 tok/s), decode unchanged.
Prefill-only by construction: the branch fires for single-request prefill
chunks on the exact serving geometry and falls through to FA2 otherwise.

Things this campaign measured that did NOT pay, so nobody re-walks them:

- **The benchmark harness was mismeasuring prefill.** `vllm bench serve`
  defaults to `--seed 0`, so every call replays the same prompts; with
  `--enable-prefix-caching` that hands later calls silent partial prefix hits
  whose size depends on the server's pool geometry. The old protocol read one
  config 15-20% low and another 3× high (a 16k prefill "measured" at 4.0 s
  that cold costs 11.2 s). `bench/run_benchmarks.sh` now seeds every call;
  the tables above are the seeded numbers, and older published prefill rows
  are not comparable.
- **`SPEC=off` prefills ~20% slower than `SPEC=dflash2`** — removing the
  drafter demotes the server from the V2 model runner to V1. The drafter's
  own prefill cost on the V2 runner is nil (6 kernel launches in a 4k
  profile); the "~15% TTFT" figure that used to circulate here predates the
  fused context-KV precompute.
- **`PREFIX_CACHE=0` prefills ~20% slower than `PREFIX_CACHE=1`** on this
  stack, align mode ruled out as the cause (PC=0 + `--mamba-cache-mode align`
  measures the same as plain PC=0). Do not turn the cache off "for speed".
- **Bigger prefill chunks do not boot** under the pinned `KV_MEM`:
  `--max-num-batched-tokens` 4096 and 8192 both inflate the profiled
  activation peak past the transient floor (engine init fails). 2048 stays.
- **Marlin tile tuning for the W4A8 GEMMs washes out end-to-end.** The
  standalone sweep (min-of-rounds, burst clocks) shows +2-20% per GEMM over
  stock tiles at M=2048 — and exactly +0.4% end-to-end, because sustained
  250 W throttling flattens the differences the bursts show.
  `patches/0023-marlin-tune-table.patch` ships the wiring anyway (off by default,
  `VLLM_MARLIN_TUNE=1`) for cards running without a power cap.
- `INT8_LAYERS="mlp|linear_attn"` (the GDN-only middle point) crashes at
  first forward — an inductor codegen bug with the mixed set on this
  torch/vllm pin. Use `mlp` or the full default.

### DFlash2 (`SPEC=dflash2`)

The one lever left after all of the above is acceptance, and Qwen's MTP head
is a single-layer chain drafter at its ceiling. [DFlash2](https://inco.ai/blog/dflash2/)
(Inco, Aug 2026) is a different drafter for this exact target:
5 Qwen3-style layers that predict the whole 7-token block in one
non-autoregressive pass from the target's layer 5/19/33/47/61 hidden states,
plus a selector that walks a coherent path through 16 candidates per slot. On
the bf16 model it reports 4.80 tokens per step vs 4.28 for MTP at the same block
size. What it took to make it pay on a 24 GB card, in order:

1. **Backport.** vLLM's support is [PR #52816](https://github.com/vllm-project/vllm/pull/52816)
   on main, on the V2 model runner. `patches/0009-dflash2-backport.patch` carries it to
   0.27.1 plus the pieces of main it silently relies on (sentinel `-1` sample
   rows, sliding-window null-block guards, K draft slots, NaN guards) and one
   semantic fix: 0.27.1 caches temperature-*applied* draft logits, main caches
   raw ones, and the PR's selector cached raw scores — on 0.27.1 that would have
   verified against the wrong q for 0 < T ≠ 1. The draft also shares the target's
   *quantized* lm_head (upstream refuses), and the V2 sampler now takes our
   sort-free small-k top-k/top-p path. MTP mode is untouched (re-measured:
   110.7 / 113.4 tok/s, 73,777-token pool).
2. **The drafter itself is 1.92B parameters, 3.85 GB in bf16** — read once per
   step, that is +5 ms on a 3090 and no gain (106 / 112 tok/s, measured), and it
   leaves a 21k-token KV pool. `drafter/capture_dflash2.py` hooks the drafter's
   own linear layers inside vLLM on 400 real prompts (~290k rows per layer, plus
   the context-KV precompute's input distribution for the k/v rows) and
   `drafter/quant_dflash2.py` GPTQ-quantizes the 36 matrices to W4A16
   compressed-tensors (Marlin): **1.19 GB**, shipped as
   [syvai/Qwen3.8-27B-DFlash2-W4A16](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16)
   (`prepare/fetch_dflash2.py`). int4 costs ~5% acceptance at default sampling (3.2 vs
   3.4 tokens per step) and nothing at greedy; keeping `fc` in bf16 did not
   recover it.
3. **Result** (`bench/run_benchmarks.sh single`, fast variant target): 26.5 ms
   per step vs MTP's 24.8, 3.14-3.34 tokens per step vs 2.8-2.9 → **117.8 tok/s
   at default sampling and 125.7 greedy at C1** (MTP: 111-115 / 115-124), with
   the best runs of this drafter reading 133.8 / 138.5, and a higher decode rate
   at C2-C8. Same output distribution by construction (perplexity 8.094,
   GSM8K 96.0-96.5%).

4. **Getting the context back to 64k** took a second patch
   (`patches/0010-hybrid-kv-groups-v2-cudagraph.patch`), because the first version of
   this mode capped out at 40k. vLLM sizes a hybrid model's KV groups by the
   *smallest* bucket of same-type layers — with the drafter that is its 5
   sliding-window layers, so the target's 16 attention layers were padded to 20
   and its 48 GDN layers to 50: 25% more pool for every token of context, to pad
   the layers that were not the problem. Sliding-window groups only ever hold
   window-many blocks, so padding *them* costs ~7 MB per request instead. That
   takes the pool from 105 to 78 KB per token (MTP: 75), i.e. 45,383 tokens at
   40k → 69,758 at 64k. The same patch makes the V2 runner's CUDA-graph memory
   explicit (`VLLM_V2_CUDAGRAPH_MEM_MIB`): upstream it returns 0, so ~1.2 GiB of
   graphs lands on top of `--gpu-memory-utilization` — ask for 0.93, run at 0.98.
   Since the runner's profiled activation peak also swings ~1 GiB between starts,
   this mode pins the pool by bytes (`KV_MEM`, 5.2 GiB) rather than by
   utilization, and start-up is then deterministic (69,758 tokens twice over).

### Drafting from the context (`LOOKUP=1`)

The drafter reads a 2,048-token window. A long-context assistant spends much of its output
*reproducing* things — quoting a document, listing commands it was shown, rewriting a
paragraph while keeping the code — and those tokens are sitting verbatim in the prompt, tens
of thousands of tokens beyond what the drafter can see. `patches/0011-dflash2-lookup-drafting.patch`
scans the request's own token history (the buffer vLLM already keeps) for the most recent
occurrence of the longest suffix of what has been generated so far, and proposes the tokens
that followed it — one Triton program per request, batch-size independent, with an
`NMIN`-token reject test before any candidate is extended. It stays lossless: greedy
verification never reads the draft distribution, and every position the lookup filled gets a
point mass on the proposed token, which is a legal proposal for vLLM's rejection sampler
(acceptance becomes p(x), residual computed from the same buffer).

Four things decide what that is worth.

**The verify block no longer has to be the drafter's block.** `dflash_config.block_size` is a
property of the checkpoint — 8 = one anchor plus the 7 mask tokens DFlash2 was trained for —
and vLLM made it the target's verify length as well, so a verbatim copy could never exceed 8
tokens per step. It sat on that ceiling: 7.83 of 8 accepted while reproducing a document's
first 60 lines. The drafter now keeps its own block while the target verifies a longer one
(`DFLASH_TOKENS`), and the positions past the drafter's block are filled from the context. They cost the drafter nothing — no extra mask tokens, no extra pass of its
candidate head — which is the point: the context is a free source of drafts, the drafter is
not.

**The long block is only scheduled while a copy is running.** Each extra verify position
costs about 1 ms of attention at 25k context, so the speculator reports per step how many of
its proposals the scheduler should actually put up for verification: the drafter's 7
normally, the whole block when (a) the lookup has a match with enough tokens left to fill
the tail, and (b) the step that just finished emitted at least a full short block's worth of
tokens — twice in a row. A single saturated step happens inside ordinary prose and the block
it buys is wasted; two in a row is a copy. The flags are read from a pinned copy that landed
asynchronously, one step stale: reading them synchronously is a device synchronise on every
decode step and measured 5%, more than the long block is worth on most work. vLLM only feeds
the draft count back to the scheduler on the synchronous scheduling path, so this mode runs
`--no-async-scheduling`; at batch 1 that costs under 1%.

Against the same server with the long block disabled, the trigger is a gain on every task in
the suite: +55% reproducing a document, +10% rewriting one, +2-3% on prose.

**The proposal is fused with the drafter's, not substituted for it.** A match of at least
`VLLM_DFLASH2_LOOKUP_NSTRONG` (8) tokens is taken on its own; a shorter one only if the
drafter independently proposed the same first `_AGREE` (2) tokens. Two independent sources
agreeing is the cheap confidence signal — the drafter looked at the hidden state, the lookup
looked at the text — and it is what stops a coincidental 6-token match from costing
acceptance on prose, which the all-or-nothing first version did.

**A match may overlap the suffix it matched**, so a repeating pattern (a list marker, an
indent, a fence) is proposed from its own period instead of missed.

Measured at 25k context, greedy, against the same server with the previous version
(tokens per step / decode tok/s):

| | no lookup | default (`DFLASH_TOKENS=7`) | `DFLASH_TOKENS=15` |
|---|---|---|---|
| reproduce the first 60 lines verbatim | 4.72 / 159 | 7.83 / 260 | **14.97 / 381** |
| shorten this, keep the commands | 2.70 / 90 | 3.19 / 107 | **3.50 / 113** |
| quote and explain | 3.01 / 101 | 3.21 / 107 | **3.35 / 110** |
| reproduce every command | 4.62 / 153 | 5.23 / 173 | 5.32 / 166 |
| free-form summary / Q&A | 2.15 / 72 | 2.08 / 69 | **2.13 / 71** / 2.01 / 67 |
| C1, 8 short chat prompts | 3.22 / 126 | 3.33 / 131 | **3.42 / 133** |

So the long block is worth **+47%** where the model reproduces its context, and a few percent
on most other work once it is only scheduled while a copy is actually running. It ships as a
mode — `DFLASH_TOKENS=15`, for a coding assistant applying edits or a RAG front-end quoting
sources — because it also costs 4 request slots instead of 8 and 56k of context instead of
64k. Quality is unchanged: GSM8K 96.5% (200 questions, greedy) with the lookup on, the
same as without it, and 7 of 9 long greedy prompts come back token-identical against the
same server with `LOOKUP=0` (the two that differ are near-tie flips, gotcha 14).

The mode also costs 4 request slots instead of 8 and 56k context instead of 64k, because
`--mamba-cache-mode align` reserves recurrent-state pages per slot per speculative block.

Pair it with `PREFIX_CACHE=1` (vLLM's hybrid prefix caching, opt-in upstream): a follow-up
turn on a 24k-token document costs **~1 s instead of ~23 s** because the attention KV is
reused and the recurrent state resumes from the last cached block boundary, for one extra
state page per request (~16% of the KV pool). Prefill stops dominating a chat, and then
drafting from the context is what makes the decode fast.

`PREFIX_CACHE=1` works in **batch mode** too, and matters just as much there: 64 requests
sharing one 5,820-token system prompt take 222 s without it and **16.9 s** with it (median
latency 94.9 s → 8.0 s), for ~14% of the KV pool and no change on workloads without a shared
prefix. If your API backend sends the same instructions with every request, this is the
single biggest thing in this repo for you. (The other three changes on this page —
lookup drafting, the KV-group fix and the V2 graph accounting — only fire on the DFlash2
path and are inert in batch mode.)

Its limits are memory- and window-shaped: each *resident* request holds 1+7
recurrent-state slots — 0.88 GiB, 15.8% of the `CTX=fast` pool, before it stores any
context — so six requests are resident and the seventh is preempted, against eight for
MTP's 0.44 GiB; from 8 concurrent long generations MTP's e2e is higher again (309 vs
235 tok/s). The drafter's 2,048-token attention window separately loses acceptance on
long-context tasks (2.3-2.6 vs 2.6-3.0 tokens per step at 12-36k, where MTP is 5-10%
ahead e2e). `CTX=long`/`huge` stay MTP.

It is not only memory-shaped, though, and that sentence used to say it was. Per-stream
decode falls hard with concurrency — 137 / 97 / 46 tok/s at 1 / 2 / 4 distinct
4k-token streams — because every resident request adds ~7 ms to the forward pass
(25.9 → 49.1 ms). Aggregate throughput still rises (137 → 309 tok/s) and nothing is
preempted, so the verify step is batching; it is latency per user that goes. MTP does
the same thing (126 / 103 / 46 per stream, ~5 ms per resident), so this is the hybrid
model plus speculation on one 3090, not something DFlash2 does wrong — what is
DFlash2's own is running out of state pages at five residents where MTP holds eight
and reaches 383 tok/s aggregate at C8. For **one person** with normal context — what single-user mode is
for — it is the fastest config in this repo. Full table in
[single-user/README.md](../single-user/README.md).
