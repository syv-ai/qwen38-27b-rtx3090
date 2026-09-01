# Long context: 262k with KVarN, and the built-in KV modes

How to get past the ~200k the fp8 KV cache allows, and what vLLM's own per-token-head quantization modes are worth on this card.

[← back to the main README](../README.md)

## 262k context with KVarN

With fp8 KV the pool holds ~200k tokens, so 150k is the shipped max and ~195k
the ceiling; the model's full 262,144 is out of reach because 16 attention
layers × 4 KV heads × 256 dims × 2 bytes (K+V, fp8) is 2 KB per token. The
way past that is a smaller cache, not a different engine, and
[KVarN](https://github.com/huawei-csl/KVarN) (Huawei CSL) has the best one we
know of: Hadamard rotation + iterative variance normalization + 4-bit keys /
2-bit values per 128-token tile, at ~840 B/token/layer here. It ships as a
fork of vLLM 0.23; [kvarn/](../kvarn/) is our port of its dense backend onto the
0.27.1 this repo runs (`bash kvarn/install.sh`, then `KV=kvarn` in batch mode
or `CTX=huge` in single-user mode).

Measured on the 3090 (`--kv-cache-dtype kvarn_k4v2_g128 --block-size 128`,
fp16 recurrent state, batch defaults otherwise). **Every row in this table is the
batch config, which runs no speculative decoding** — `ms/token` is `vllm bench
serve`'s mean TPOT, so one model step per output token. Single-user mode
speculates, and its numbers are in the next table; do not compare across the two.

| batch mode (no speculation) | fp8 KV (default) | KVarN k4v2 |
|---|---|---|
| KV pool, batch mode | ~205-225k tokens (150k max, ~195k ceiling) | 302-344k tokens with 64 slots, **420k with 4 slots — 262k fits with room for 1.6 such requests** |
| KV pool, single-user mode (MTP-3, `CTX=long`/`huge`) | 150k max | 200k max |
| needle-in-a-haystack, greedy | — | correct at 4k / 16k / 30k / 100k / 240k, both depths |
| perplexity (en/da/code, 33k tokens) | 8.223 | 8.236 (+0.16%) |
| prefill, 1k / 16k / 100k inputs | 1,812 / 1,595 / 997 tok/s | 1,741 / 1,569 / 1,050 tok/s (same within ±5%) |
| single stream at 100k context | TTFT 99 s, 27 ms/token | TTFT 94 s, 33 ms/token (1.22×) |
| 4 × 60k-token requests, 1,024 out | only 3 fit → 256 s total, ITL 33 ms | all 4 resident → 242 s total, ITL 49 ms |
| 64 concurrent short requests (128/512) | 876 tok/s | 692 tok/s (38 resident: 2048-token blocks cost as much per short request as fp8's 800-token block) |

### What it costs in single-user mode, which is where it hurts

The 1.22× above is batch mode at 100k. Single-user mode speculates, and the tax is
much larger there — reported in [#11](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/11)
and reproduced here. MTP-3, one request, 112,648-token prompts, `PREFIX_CACHE=1`,
two general tasks (summarize, answer-a-question), streamed so prefill is excluded
(`bench/labd_bench.py <tag> --ctx 100000 --corpus ~/bench/labd_corpus_long.txt
--tasks qa,summary`) — the only variable is `--kv-cache-dtype`:

| single-user, MTP-3, 112k context | fp8 (`CTX=long`) | KVarN (`CTX=huge`) |
|---|---|---|
| KV pool | 174,489 tokens | 292,035 |
| decode, summarize | 71.4 tok/s | 33.1 |
| decode, answer a question | 64.9 | 30.8 |
| **decode, both** | **68.1 tok/s — 14.7 ms/token** | **32.0 tok/s — 31.3 ms/token (2.13×)** |
| accepted tokens per step | 2.56 | 2.38 |
| TTFT, cold | 152.2 s | 146.7 s |

Two things worth separating out of that 2.13×. About 1.98× is raw step time. The rest
is acceptance: KVarN costs ~7% of it (2.38 against 2.56 tokens per step), because the
quantized cache shifts the target's logits enough that the draft head agrees less often.
Speculation stays exact — the sampled distribution is unchanged, and perplexity moves
+0.16% — but acceptance feeds straight back into throughput, so "quality-neutral" does
not imply "speed-neutral" for a speculating server.

For a short-prompt chat load the gap nearly closes: MTP-3 on real prompts reads
84 / 89 tok/s at fp8 against 79 / 88 with KVarN (base variant, earlier draft vocab),
about 1.06×. The tax is a function of context length, not a constant.

So: same VRAM, 1.6-2× the tokens, full 262k context, output quality intact, prefill
unchanged — and a decode tax that runs from ~6% on short single-user prompts through
1.22× in batch mode at 100k to **2.13× in single-user mode at 112k**. Past ~100k of
context you are buying 1.7× the context for less than half the decode rate, which is
worth it when the alternative is not fitting the request at all and a bad trade
otherwise. Which is why it's a mode and not the default — nothing changes unless you
set `KV=kvarn` (batch) or `CTX=huge` (single-user); the KV-cache format is an
engine-level choice in vLLM, so it can't be switched per request. Port notes and what
to watch when bumping vLLM are in [kvarn/README.md](../kvarn/README.md).

(vLLM 0.27.1 also has TurboQuant built in — `--kv-cache-dtype turboquant_4bit_nc`
gives a similar 413k-token pool here and about 15% slower decode, but its
chunked-prefill path allocates O(context) scratch outside the memory profile
and OOMs at 32k+ prompts on this card at 0.972 utilization, and at 128k even
at 0.90. KVarN's prefill path is bounded and did 240k.)

### The built-in per-token-head modes

vLLM 0.27.1 also ships `int8_per_token_head`, `fp8_per_token_head` and
`int4_per_token_head` (dynamic per-token, per-head scales; the int4 one with a
rotation and asymmetric zero-points), all only in the Triton attention
backend. Measured on the 3090 in the batch config at 0.93 utilization, same
script for every column (`fp8_per_token_head` does not start on sm86: Triton's
fp8 KV needs SM89+). **Batch config again, so no speculation** — see the table
above for what these caches cost a speculating single-user server:

| batch mode (no speculation) | fp8 (FlashInfer) | int8_per_token_head (Triton) | int4_per_token_head (Triton) | KVarN k4v2 |
|---|---|---|---|---|
| KV pool at 0.93 util | 164k tokens | 178k | **355k — 262k fits (1.35×)** | 302-420k |
| perplexity (same battery) | 8.235 | 8.231 | 8.257 (+0.3%) | +0.16% |
| needle, greedy | 100k ok | 100k ok | 100k ok, **240k ok** | 4k…240k ok |
| prefill 1k / 16k | 1,773 / 1,601 tok/s | 1,739 / 1,187 | 1,710 / 1,194 | 1,741 / 1,569 |
| 100k context, single stream | TTFT 100 s, 26.8 ms/token | 231 s, 40.8 ms | 220 s, 41.4 ms | 94 s, 33 ms |
| 64 concurrent short (128/512) | 839 tok/s | 850 | 835 | 692 |

Reading: `int8_per_token_head` buys nothing over fp8 here (same byte per
element, quality already neutral) and costs the Triton backend's long-context
speed. `int4_per_token_head` is a genuine zero-install alternative to KVarN for
the 262k use case — it fits, passes the 240k needle, and keeps short-request
throughput that KVarN's 2048-token blocks lose — at 2.3× the prefill time and
1.5× the decode time at 100k, because vLLM's Triton attention is that much
slower than FlashInfer/FlashAttention on this card at long context (the same
backend tax the single-user mode avoids by staying on FlashAttention). If the
Triton backend catches up, it becomes the simpler choice; today KVarN is
faster at long context and `int4_per_token_head` is faster on many short
requests. To try it: `--kv-cache-dtype int4_per_token_head --attention-backend
TRITON_ATTN --max-model-len 262144` (batch/start_qwen.sh: `KV=int4pth`).

## DFlash2 past 64k (`SPEC=dflash2 CTX=long`)

The block drafter was pinned to `CTX=fast` — bf16 KV on FlashAttention, 64k at
`DFLASH_TOKENS=7` and 56k at 15 — because bf16 KV is 64 KB per token and the pinned
5.2 GiB pool is exactly that much. `CTX=long` moves it to an `int8_per_token_head` cache
on the Triton backend and roughly doubles the context:

| | bf16 (`CTX=fast`) | int8 (`CTX=long`) |
|---|---|---|
| context, `DFLASH_TOKENS=7` | 69,758 tokens | **138,696** (136,429 with prefix caching) |
| context, `DFLASH_TOKENS=15` | 57,669 | **114,224** |

Two patches make that work, and neither changes anything at bf16:

- **[0012-hybrid-sw-block-promote.patch](../patches/0012-hybrid-sw-block-promote.patch)** — without it,
  int8 costs *more* memory than bf16, not less: 6.82 GiB to serve 32,768 tokens. vLLM equalizes
  KV page sizes by scaling a layer's block size up by an integer ratio, and the drafter's five
  sliding-window layers are born at the smallest kernel block, 16. That ratio is an integer at
  bf16 only by coincidence — the target's 4 KV heads × 256 and the drafter's 8 × 128 both come
  to 4096 B per token per layer — and a per-token-head cache breaks it by adding one fp32 scale
  per head. The drafter's layers then keep a 16-token block while their page is padded to the
  full 1.71 MiB primary page: 385 blocks at 1.88% utilisation, a constant 5.2 GiB. The patch
  rounds those layers' block up (16 → 864) so their page covers the maximum instead.
- **[0013-spec-decode-int8-kv.patch](../patches/0013-spec-decode-int8-kv.patch)** — the split-KV verify
  kernel reads the quantized cache and is wired into the Triton backend, which otherwise cannot
  split KV for a multi-query verify at all (`use_3d` is off whenever `max_seqlen_q > 1`, and
  every DFlash2 step is a verify). Per attention layer at 128k, 8 query tokens: 1.3 ms for this
  kernel against 7.4 ms for vLLM's unified attention and 10.1 ms for FA2.

### What it is actually good for

Measured against the option that already covers this context, on 112,655-token prompts
(`bench/labd_bench.py --ctx 100000 --corpus ~/bench/labd_corpus_long.txt`), both with
`PREFIX_CACHE=1`:

| task | `SPEC=dflash2 CTX=long` (k=15) | `SPEC=mtp CTX=long` |
|---|---|---|
| reproduce the document verbatim | 14.19 tok/step, **154.8 tok/s** | 3.81, 101.4 |
| list every command | 5.32, 69.8 | 3.52, **93.6** |
| rewrite but keep | 5.09, 66.8 | 3.79, **100.6** |
| quote and explain | 2.17, 34.6 | 2.75, **72.9** |
| summarize | 2.17, 34.8 | 2.68, **71.4** |
| answer a question | 2.01, 32.1 | 2.34, **61.9** |
| all six | 3.10, 47.0 | 2.95, **78.6** |
| TTFT, first turn / cached turn | 316.8 s / 6.1 s | **151.9 s / 2.4 s** |

**So: +53% where the model reproduces its context, and about 2:1 behind everywhere else, with
twice the TTFT.** `SPEC=mtp CTX=long` remains the better general long-context server, and it
reaches 150k rather than 114k. Reach for `SPEC=dflash2 CTX=long` when the workload is a RAG
front-end quoting sources or a coding assistant applying edits to a large file it has already
loaded — where the answer is mostly text that is already in the prompt — and not otherwise.

The lookup drafter itself does not decay with context: it accepts 14.19 of a possible 16
tokens per step at 112k, against 15.0 at 25k and 50k. What costs the mode is the step, at
91.7 ms against bf16's 48.2 ms at 50k — the Triton backend, the drafter's own five layers, and
the int8 kernel's padded-stride penalty (its head dim is 260 B, so odd KV heads start off a
16-byte boundary; ~13% end to end, and reading the cache as int32 instead would recover most
of it). Prefill is the larger cost and this kernel cannot help there — a prefill chunk is 2048
query tokens, far above the block sizes it is for.
