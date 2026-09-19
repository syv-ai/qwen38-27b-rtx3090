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
0.28.0 this repo runs (`bash kvarn/install.sh`, then `KV=kvarn` in batch mode
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
much larger there — reported in [#11](https://github.com/syv-ai/HyperQwen/issues/11)
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

(vLLM 0.28.0 also has TurboQuant built in — `--kv-cache-dtype turboquant_4bit_nc`
gives a similar 413k-token pool here and about 15% slower decode, but its
chunked-prefill path allocates O(context) scratch outside the memory profile
and OOMs at 32k+ prompts on this card at 0.972 utilization, and at 128k even
at 0.90. KVarN's prefill path is bounded and did 240k.)

### The built-in per-token-head modes

vLLM 0.28.0 also ships `int8_per_token_head`, `fp8_per_token_head` and
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

- **[hybrid-sw-block-promote.patch](../patches/hybrid-sw-block-promote.patch)** — without it,
  int8 costs *more* memory than bf16, not less: 6.82 GiB to serve 32,768 tokens. vLLM equalizes
  KV page sizes by scaling a layer's block size up by an integer ratio, and the drafter's five
  sliding-window layers are born at the smallest kernel block, 16. That ratio is an integer at
  bf16 only by coincidence — the target's 4 KV heads × 256 and the drafter's 8 × 128 both come
  to 4096 B per token per layer — and a per-token-head cache breaks it by adding one fp32 scale
  per head. The drafter's layers then keep a 16-token block while their page is padded to the
  full 1.71 MiB primary page: 385 blocks at 1.88% utilisation, a constant 5.2 GiB. The patch
  rounds those layers' block up (16 → 864) so their page covers the maximum instead.
- **[spec-decode-int8-kv.patch](../patches/spec-decode-int8-kv.patch)** — the split-KV verify
  kernel reads the quantized cache and is wired into the Triton backend, which otherwise cannot
  split KV for a multi-query verify at all (`use_3d` is off whenever `max_seqlen_q > 1`, and
  every DFlash2 step is a verify). Per attention layer at 128k, 8 query tokens: 1.3 ms for this
  kernel against 7.4 ms for vLLM's unified attention and 10.1 ms for FA2.
- **[triton-spec-attn-fp8-kv.patch](../patches/triton-spec-attn-fp8-kv.patch)** (sm89 and up) — the
  same split-KV verify kernel reads vLLM's per-tensor fp8 cache (`--kv-cache-dtype fp8`), so the
  fp8 pool can run with `TRITON_ATTN` on both sides and keep FULL CUDA graphs: on a 4090 at 120k,
  DFlash2 fp8 decodes 108.7 / 88.6 / 81.0 tok/s at 24k / 49k / 88k tokens against 71.4 / 50.9 /
  35.4 with vLLM's unified attention on the same route, and 84.4 / 81.7 / 83.5 for FlashInfer with
  PIECEWISE (issue #87). The route is `SPEC=dflash2 CTX=fast` with
  the exact launch line, verbatim (the `"attention_backend":"TRITON_ATTN"` inside the speculative
  config is the part that is easy to drop, and dropping it silently costs the FULL graphs):

  ```
  SPEC=dflash2 CTX=fast EXTRA_ARGS="--attention-backend TRITON_ATTN --kv-cache-dtype fp8 --speculative-config '{\"method\":\"dflash\",\"model\":\"/app/models/Qwen3.8-27B-DFlash2-W4A16\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"probabilistic\",\"attention_backend\":\"TRITON_ATTN\"}'" bash single-user/start_qwen.sh
  ```

  Measured on both a WSL2 4090 (Docker Desktop) and a native one (Ubuntu 24.04, driver 580.159, this
  image): on native, prose prompts of ~24k / ~50k / ~89k tokens decode 100.6 / 96.0 / 87.4 tok/s with
  this patch against 68.3 / 49.8 / 34.6 with vLLM's unified attention (median of 3, greedy, 256 out,
  fresh prefill, `MAX_LEN=120000 PREFIX_CACHE=1 MAX_SEQS=4`), prefill unchanged by the patch on either
  box (2.4k / 2.0k / 1.6k tok/s native). Not for the 3090: Triton has no fp8 conversion on sm86.
  `INT8_ACT=int8` on the fp8 route is slow with or without this patch (a Marlin variant choice,
  tracked separately): do not stack the two until the memory-pressure question behind it is
  understood.

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

## DFlash2 at 240k: `CTX=huge` (KVarN) also combines with `SPEC=dflash2`

```bash
bash kvarn/install.sh                # applies the v0.28.0 KVarN + V2-runner ports
SPEC=dflash2 CTX=huge PREFIX_CACHE=1 bash single-user/start_qwen.sh
```

Where `CTX=long` doubles the DFlash2 pool with int8 KV (138k), the KVarN cache
takes the same idea further: 268k tokens of pool at 245760 max-model-len, on the
same pinned budget. No kernel work — the KVarN Triton kernels run unmodified on
the V2 runner; the seven fixes in `kvarn/kvarn-v2-runner-0.28.0.patch` are allocator and
geometry logic (the patch header walks through them, including an upstream vLLM
bug in the mamba align resume path, and a NaN path in the DFlash2 candidate
selector that KVarN noise exposes on verbatim-reproduction content). Two
machines, both RTX 3090 at 250 W, `bench/labd_bench.py --ctx 20000` — the
contributor's WSL2 box and this repo's bare-metal one, which do not agree on
decode rate and do agree on everything else:

| `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` | WSL2 | bare metal |
|---|---|---|
| copy (reproduction) | 130 tok/s, 7.8 tok/step | 164 tok/s, 7.83 tok/step |
| code / edit / quote / summary / qa | 89 / 65 / 44 / 38 / 36 | 109 / 83 / 58 / 51 / 43 |
| all six tasks together | 53 tok/s, 3.0 tok/step | 67 tok/s, 3.15 tok/step |
| verbatim reproduction, 25k document | correct | 1,150 / 1,150 chars |
| KV capacity at 245760 max-model-len | 268,169 tokens | 268,169 tokens |
| GSM8K exact-match (thinking off) | 97.0% (n=200) | 95.2% (n=600), 95.0% (n=200) |
| 100k-deep needle, both turns | correct | — |
| turn 2 over a 100k cached prefix | 4.7 s (vs 169 s cold) | — |

<sub>Context for the GSM8K column: every configuration this repo already ships
reads 95.0-96.5% on the same 200-question harness ([quality.md](quality.md)),
and 95.0% is the batch-mode default. 95.2% at n=600 (±0.9 points) therefore sits
inside the band rather than below it — which is the useful comparison, since this
mode inherits KVarN's lossy 4/2-bit cache and should be judged against the other
lossy configurations rather than against bf16. Repeat runs of the reproduction
check on bare metal are bit-identical (same step count, same 1,150 characters),
which is the property that was missing before `PIECEWISE` — see below.</sub>

One caveat to the "all of it is lossless" paragraph above: the speculation here
is still exact, but this mode inherits KVarN's 4/2-bit KV cache, which is lossy —
the same trade `CTX=huge` already makes (deep-needle retrieval passes at 200k).

The WSL2 column's ~20% deficit is a WSL2 tax, not a Windows tax, and leaving
WSL for native Windows does not recover it: the same contributor ran
`aivrar/vllm-windows-build` (0.27.1, 18 of 19 patches apply after a CRLF→LF
pass) on the same box and measured native Windows *slower* than WSL2 — 66.0
vs 76.2 tok/s across the task mix, a 4.4× longer warm boot, and the same
WDDM paging behavior underneath ([#25](https://github.com/syv-ai/HyperQwen/issues/25)).
The bare-metal column is reachable from a Windows box only by putting Linux
on the metal.

**On WSL2, every `SPEC=dflash2` profile needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`** —
not just `CTX=huge`. The drafter's architecture forces vLLM's V2 model runner
(`_is_dflash2_draft()` in `config/vllm.py`), the V2 runner allocates UVA buffers
before the weights load, and vLLM leaves pinned memory off by default under WSL2,
so a clean venv aborts with `RuntimeError: UVA is not available` before it prints
anything model-shaped. Those buffers work fine on the paravirt driver. Note the
name: `VLLM_WSL_PIN_MEMORY` is **not** a vLLM variable and setting it does
nothing — this README named it for 22 minutes on 2026-08-21 (`589daae`, fixed in
`27f51fa`), so a tree cloned in that window will have it.

**On WSL2 the usable dedicated memory is about half a gigabyte less than the
same card on bare metal, and the shipped `SPEC=dflash2` boot sits about 50 MiB
under it.** Anything larger (a wider verify block, a bigger drafter, a raised
pin, a boot that recompiles a graph) runs two to six times slower instead of
failing, and the log does not say so; `nvidia-smi` looks the same either way.
Gotcha 58 has the counters to read, the four costs that turned out to be this,
and the profile that gives it room (`KV_MEM=3000000000 DFLASH_MAX_LEN=8192`,
free for the shipped head at width 7).

**Running a DSpark drafter.** vLLM 0.28.0 can serve RadixArk/Qwen3.8-27B-DSpark
(bf16, seven drafts per step like the shipped head) once two things are in
place: `patches/dspark-draft-quant-config.patch` (the loader refuses a bf16
drafter beside the quantized target without it), and a copy of the checkpoint
whose `config.json` names the architecture `Qwen3DSparkModel` instead of
`DSparkDraftModel` (the registry maps the published name to the DeepSeek V4
class; the weights are unchanged, so hard-link the safetensors). Then
`DRAFT=/path/to/that/copy DRAFT_METHOD=dspark KV_MEM=3000000000
DFLASH_MAX_LEN=8192 SPEC=dflash2 CTX=fast bash single-user/start_qwen.sh`. It
serves, and loses to the shipped head on the same requests: 3.19 against 3.77
tokens per step (accepted drafts plus the bonus token) and 115 against 160 tok/s on a 4090, 3.37 against 3.83
and 114 against 150 on a 3090 (issue #25, items 15 and 16). Documented so nobody
re-derives the two errors, not as a recommendation.

One knob this mode used to set for you, and now sets only for MTP:
`cudagraph_mode=PIECEWISE`. Prefix caching and a *captured* (FULL) verify step
did not mix on this path. On WSL2 that showed up as acceptance collapsing to
about one token per step; on bare metal it also **corrupted the output** —
special-token ids leaking into the stream, 1 of 1,176 characters matching the
source instead of all of them. It is the capture rather than the drafter: eager
is clean, `LOOKUP=0` is not, forcing a fixed verify-block length is not, and
PIECEWISE — which keeps the compiled graphs and leaves only the multi-query
verify uncaptured — restored both the speed and the correctness on both machines.

`dflash2` has its FULL graphs back (`a75ee4b` fixed the residue, `b356e31` then
swept **all 128** residues under FULL with 0 broken), so at HEAD
`SPEC=dflash2 CTX=huge PREFIX_CACHE=1` runs captured. `SPEC=mtp CTX=huge`
still forces PIECEWISE, and that one is a correctness constraint rather than a
preference: under FULL it breaks at one prompt length in 128 and nobody has
fixed it. Either way prefix caching stays on, which is what the mode is for —
turn 2 over a cached 100k document costs 4.7 s against 169 s cold.

`CUDAGRAPH_MODE=FULL_AND_PIECEWISE` overrides the MTP line for anyone hunting
the root cause. Treat that as unsafe rather than merely slower.

What that trade costs, re-measured at HEAD. The numbers this README used to carry here
had `FULL_AND_PIECEWISE` at 38 tok/s (1.97 per step) on the 25k copy task against
PIECEWISE's 132, and called it 3.5x. That was not the capture mode — it was the residue
bug, which `a75ee4b` fixed. With the same server and only the capture toggled
(`bench/labd_bench.py --ctx 20000`, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1`, decode tok/s):

| | copy | code | edit | quote | summary | qa | all six |
|---|---|---|---|---|---|---|---|
| FULL (the default now) | 167.1 | 111.1 | 84.7 | 55.0 | 47.8 | 43.4 | 65.7 (3.03/step) |
| PIECEWISE | 166.3 | 111.3 | 83.0 | 62.4 | 48.6 | 43.1 | 67.6 (3.18/step) |

They are the same. Five of the six are within 2%; `quote` differs by 13% in PIECEWISE's
favour, which is greedy divergence on the task that diverges most, and it is what puts
PIECEWISE 3% ahead overall. So at this context length the capture mode is not a
performance decision at all, in either direction.

Short prompts are where a difference was measured, and that measurement is older:
78/125/202 tok/s captured against 74/102/176 piecewise on de/en/code, i.e. **13-18%**.
Treat that as an upper bound — @mjungnickel18 measures 0.2-2.3% for the same comparison
when only runs with identical step counts are compared, and he is right that greedy runs
which take a different number of steps are not comparable. Past 8k the two are within
noise on bare metal (111.8 vs 109.3 tok/s at 8k, 78.2 vs 86.1 at 16k, 68.9 vs 73.3 at
32k, 58.4 vs 56.0 at 50k, unique prompts, one server per mode). Under GPU passthrough on
a VM the same comparison costs 2-3x, reported in
[#13](https://github.com/syv-ai/HyperQwen/pull/13) and consistent with the
uncaptured verify being launch-bound: launches that are nearly free here are not free
there.

Two limits worth knowing before you point this at anything: what it does with
more than one user, and what the long verify block costs.

**It is a one-stream mode, and the limit is the pool rather than `MAX_SEQS`.** An
earlier version of this paragraph said the knob was the seat count and that
`MAX_SEQS=8` lifts it. It does not, and @mjungnickel18 was right to push back in
[#25](https://github.com/syv-ai/HyperQwen/issues/25). A *resident* request
reserves 1+k = 8 recurrent-state slots — **15.8% of the 69,758-token `CTX=fast` pool**,
~0.82 GiB of its pinned 5.20, which is the 0.88 GiB [gotcha 33](gotchas.md) fitted
from the memory model — before it holds one token of context. Seven fit with 128-token
prompts, five with 4k-token ones and two with 16k ones. MTP's k=4 costs 0.44 GiB, so
eight fit, four of them at 16k. Past that the extras queue, and once the pool is full
something has to be preempted and recomputed to make room — a ramp of tiny requests
hits that at the seventh. `MAX_SEQS` decides how many requests are *admitted*, not how
many can run. The pool itself is **unchanged** by the setting (268,169 tokens at
`CTX=huge` either way, ~8 MiB total between 1 slot and 8 —
[gotcha 33](gotchas.md)), which is the part of the old paragraph that was right.

What concurrency actually costs, measured with `bench/conc_ladder.py` on distinct
4k-token prompts — each salted so nothing is served out of the prefix cache — 256-token
answers, `MAX_SEQS=8`, `CTX=fast`, 250 W:

| streams | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| **dflash2** per-stream decode tok/s | 137 | 97 | 46 | 33 |
| **dflash2** aggregate decode tok/s | 137 | 225 | 309 | *5 resident, no steady state* |
| **dflash2** ms per forward pass | 25.9 | 32.8 | 49.1 | — |
| **mtp** per-stream decode tok/s | 126 | 103 | 46 | 23 |
| **mtp** aggregate decode tok/s | 124 | 212 | 280 | **383** |
| **mtp** ms per forward pass | 24.8 | 29.8 | 43.1 | 62.3 |

Two things to read off it. The verify step *does* batch — aggregate throughput keeps
climbing for both speculators, and no preemption happens at any of these points — so
"the block verify does not batch" is not what is going on. What it costs is latency:
each additional resident request adds about 7 ms to every forward pass under DFlash2
(5 ms under MTP), so the second user roughly halves your tokens per second and the
fourth roughly quarters them. That 7 ms is not attention over their context — with
128-token prompts the slope is the same (25.2 → 45.8 ms from one resident to four) —
it is the per-request recurrent state, the same thing that limits residency. And MTP keeps scaling to 8 streams where DFlash2 runs
out of pool at 5, which is the whole of its C8 advantage. Point one person at DFlash2;
point a team at `SPEC=mtp` or batch mode.

**On `CTX=huge`, raising `MAX_SEQS` is worse than not raising it.** That profile
defaults to 2 seats, and the reason is the same state page against a smaller pinned pool
(4.90 GiB): five residents with a short prompt, four with a 16k one. Force it
to 8 and feed it eight independent 16k-token streams and the scheduler starts evicting
— **10 preemptions** in one run, peak occupancy 99.4%, per-stream 3 / 7 / 72 tok/s,
end-to-end aggregate down to 10.3 from 13.0 at a single stream. The same eight streams
against the shipped 2 seats: **0 preemptions and 14.4 tok/s**, i.e. 40% more work done
by admitting fewer requests. An earlier version of this README recommended exactly that
override. Leave the seats where they are.

Make the streams long and independent and it stops being about decode at all. Eight
16k-token prompts with nothing shared between them: end-to-end aggregate **15.8 tok/s**
against 131 for a single stream, mean TTFT 71.7 s, two requests resident — but the
decode-only aggregate over the same run is 183 tok/s, tokens per step is unchanged at
3.71 and nothing is preempted. The run is 131k tokens of prompt at ~1,600 tok/s and
2,048 tokens of answer, so it is a prefill measurement wearing a decode measurement's
units, and `SPEC=mtp` reads the same 15.0-15.9 there. If your clients each bring their
own long document, that is the number you get, and no speculator changes it.

What *does* change it is sharing the document. The same eight 16k streams with one
shared prefix and `PREFIX_CACHE=1` (`bench/conc_ladder.py --shared`):

| streams | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| end-to-end aggregate tok/s | 15.5 | 128.4 | **147.8** | 68.3 |
| mean TTFT | 14.6 s | 1.3 s | 2.6 s | 11.5 s |

One stream pays the prefill; everyone after it hits the cache, and four concurrent
readers of the same document get 148 tok/s end-to-end against 15.9 when the documents
differ. That is the shape of workload this mode is for — a chat client or a coding
front-end against one codebase — and it is nearly a 10x difference from the same server
on the same prompt length.

`DFLASH_TOKENS=15` doubles the state page to 1.66 GiB: three residents with an empty
context, **two** with 4k-token prompts, against five. That mode is single-user in the
literal sense, and its launcher default of `MAX_SEQS=4` is already the tighter number —
do not raise it. `DFLASH_TOKENS=15 MAX_SEQS=8` used to boot, answer `/health` and then
die on the first concurrent batch (`torch.OutOfMemoryError` in the engine, every request
500); the launcher now caps the captured-graph size so that configuration degrades to
piecewise instead ([gotcha 38](gotchas.md)). It does boot at 240k since `82bd62d`,
which caps `max_model_len` to 221,184 above 7 drafts rather than letting the server fail
to come up.


The capture mode is fixed at boot, and for `SPEC=mtp CTX=huge` the trade is not
optional. What FULL does there is corrupt one prompt length in every 128, and only
for a request that hits the prefix cache. The broken residue is a function of the
draft count (`R = 117 + k`, the same for both speculators), which is why scoping
the workaround to `dflash2` was wrong the first time — `SPEC=mtp CTX=huge` shipped
with the same bug at residue 4. Piecewise costs MTP nothing measurable
(87.8/86.1/70.4/63.5 tok/s captured against 93.5/83.8/70.3/59.6 piecewise over
8k-50k), so it keeps PIECEWISE until residue 4 comes back verbatim under a full
sweep.

**Do not test that residue by its symptom.** The location is deterministic and the
damage is not: the same `mtp` residue has returned an empty answer, a one-character
answer, and 400 tokens of fluent Danish inventing a task the prompt never asked for
(2 of 1,146 characters matching the document). A detector keyed on "it repeats" or
"it came back empty" passes at least one of those. Gotcha 37 in
[gotchas.md](gotchas.md) has the residue table; `bench/residue_sweep.py`
sweeps all 128 residues and judges every answer on how much of the document came
back, and `bench/verbatim.py` self-tests that rule against all three shapes.

## 256k the stock way: int4 KV (`single-user/alternative.sh`, experimental)

```bash
bash single-user/alternative.sh      # TRITON_ATTN + --kv-cache-dtype int4_per_token_head
```

Where KVarN reaches 268k with its own kernels, vLLM's stock
`int4_per_token_head` cache now combines with the DFlash2 drafter too:
**314,915 tokens of pool at 256000 max-model-len** (1.23× concurrency) on one
24 GB card, no `kvarn/install.sh`. Three boot blockers stood in the way — a
padded-page view error under the hybrid block-promotion geometry, and a
causal-only assert plus missing per-seq-causal plumbing in the int4 Triton
kernel, which the drafter's 8-row draft block needs
(`patches/int4-kv-per-token-head.patch`, contributed in
[#42](https://github.com/syv-ai/HyperQwen/pull/42) by @lachhabw).

The trade: the Triton attention backend plus the per-step int4 unpack cost
about 20% of decode against the shipped config on short prompts (~86 vs ~104
tok/s e2e on the same probe), and — unlike KVarN, which has GSM8K and
100k-needle numbers above — int4-KV quality at depth now reads:

| metric | result |
|---|---|
| GSM8K exact-match (200 questions, greedy, thinking off) | **96.0%** |
| 100k-token needle at 90% depth (`bench/needle_test.py`) | **retrieved** |

Measured on an RTX 4090 (24 GB) with `bench/quality_battery.py int4kv --gsm-only
--gsm-n 200` and `bench/needle_test.py 100000 0.9`; the 96.0% sits inside the band
the other configurations read (95.0-96.5%, docs/quality.md). Tool calling
round-trips correctly and the lookup lane works; the rest of this configuration
is still experimental.

## Vision serving on one 3090 (`VISION=1 SPEC=dflash2 CTX=long`)

The 0.85 GiB tower fits resident next to the full pool. Measured 2026-09-16 on the W4A16
uncensored quant, DFlash2 k=7, int8 KV, TRITON_ATTN, `GPU_UTIL=0.88`:

| `MAX_LEN` | `KV_MEM` pin | result |
|---|---|---|
| 131072 (profile default) | 5583457484 (5.2 GiB) | boots, vision live (`SEEN` on image input) |
| 98304 | 4529848320 (4.2 GiB) | boots, tower fits |
| 65536 | 3758096384 (3.5 GiB) | boots, tower fits |
| 98304 | 5583457484 (5.2 GiB, oversized) | boots, short prompts avg ~84 tok/s |

Rules found the hard way:

- `VISION_OFFLOAD=0`. The offload path pins each tower module with `pin_memory()` in
  `wrap_modules` (`vllm/model_executor/offloader/uva.py`), and that call OOMs on WSL2 even
  with gigabytes free. Resident (0.85 GiB) just works.
- Do not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on WSL2. It breaks the Marlin
  custom ops during weight processing (`CUDA driver error: device not ready` right after the
  shards load). The default (False on WSL2) is correct here.
- A pool smaller than the context need refuses to boot (`KV cache needed larger than
  available`), it does not slow down. Note: a 16 tok/s episode during tuning resolved after
  restoring the full 5.2 GiB pin (short prompts back to ~84 tok/s avg); the mechanism is not
  fully understood, so keep the full pin unless the tower needs the room.
- `.env` for this setup: `VISION=1`, `VISION_OFFLOAD=0`, `MAX_LEN=98304`,
  `KV_MEM=5583457484`, `GPU_UTIL=0.88`, `SPEC=dflash2`, `CTX=long`.

## Long re-baseline, 2026-09-16 night (`CTX=long`, profile defaults, vision on)

Short story probe, idle box, thinking off: **76.8 tok/s** (221 tokens in 2.9 s), clearing
the 65 tok/s bar. Fast mode on the same model and drafter managed only 17 to 19 tok/s
idle, so the per-forward cost there is the open question, not speculation (acceptance
sits near one third in both modes) and not the pool (full pin in both).

Boot lesson: the 131k plus resident tower fit is marginal. One boot OOMed allocating
12 MiB with 23.12 GiB held by PyTorch; the identical config booted clean on retry.
That matches the launcher's note that the profiled activation peak swings about
1 GiB between starts. If a boot OOMs at the margin, retry before resizing.
