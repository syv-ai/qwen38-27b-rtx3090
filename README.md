# Qwen3.8-27B on one RTX 3090

![Stock vLLM against this repo, same card, same prompts](docs/media/demo.gif)

Serving setup for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) on a
single 24 GB consumer GPU with vLLM — 150k token context and an OpenAI-compatible
API with key auth, in two ready-made modes.

## Quick start

The image is prebuilt and pushed to
[ghcr.io](https://github.com/syv-ai/qwen38-27b-rtx3090/pkgs/container/qwen38-27b-rtx3090)
on every commit — the build applies all `patches/` and runs `verify.sh` as its
gate, so `latest` is always the current stack. The first start pulls it (9.5 GB),
downloads and requantizes the model (~20 GB, once, into `./models`), and serves
on port 18020. Pick a mode — one GPU serves one at a time:

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 && cd qwen38-27b-rtx3090

docker compose --profile single up -d    # one or a few people chatting
docker compose --profile batch  up -d    # API backend, many concurrent requests
```

| | `--profile batch` → [batch/](batch/) | `--profile single` → [single-user/](single-user/) |
|---|---|---|
| for | API backends, pipelines, many concurrent requests | one or a few people chatting |
| aggregate, 64 concurrent (128 in / 512 out) | **~1,035 tok/s** steady-state decode, 948 end-to-end (~1,222 / 1,042 with all layers int8) | n/a (8 slots) |
| single-stream (C1) decode rate, realistic prompts | 46 tok/s | MTP: **121** tok/s at default sampling, **120** greedy (`CTX=fast`, 64k; 96 / 102 with `CTX=long`, 150k). DFlash2 (`SPEC=dflash2`): **127** default, **130** greedy |
| reproducing its own context (quoting a document, applying an edit) | 46 tok/s | **381 tok/s** at 25k context — 15.0 tokens per verify step, drafted straight from the prompt (`SPEC=dflash2` + `DFLASH_TOKENS=15`) |
| trick | 16-bit recurrent state + int8 tensor-core GEMMs | MTP speculation with 4 cheap drafts, a draft vocabulary that covers what the model says, calibrated int4 lm_head/drafter, split-KV verify attention; optionally native vLLM 0.28.0 DFlash2 (7 drafts in one pass, int4-requantized) with a verify block the context fills |
<sub>Single-stream numbers re-measured 2026-08-22 on current main with
`bash bench/run_benchmarks.sh single` — `vllm bench serve`, the 8 prompts in
`bench/prompts_real.jsonl`, 1024 output tokens, C1, decode rate taken as
`C / mean TPOT`. Quote them against that harness: a client with a different output
length is not measuring the same thing, and mixing the two is how
[#3](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/3) got confusing.</sub>

> Version note: this branch pins vLLM 0.28.0; the throughput and quality tables are
> retained as reference baselines while the v0.28.0 GPU matrix is being re-measured.

Both modes share one install — the mode is just which launch script you run.
Speculation wins below ~8 concurrent users on short prompts, plain batching above;
on long independent sessions the crossover is much earlier, because a speculating
request reserves recurrent-state pages the pool has few of — the concurrency
paragraph under "DFlash2 at 240k" has the measurement. Numbers are `vllm bench serve` on an
RTX 3090 at a 250 W power limit. If the card is yours alone, the fastest
configuration is three environment variables away:
[If you are the only user](#if-you-are-the-only-user-do-this).

Prefill is a separate budget from either: ~1,810 tok/s at 1k inputs in batch
mode, and in single-user mode ~1,440 tok/s stock or **~1,850-1,940 with
`INT8_ACT=int8`** (1,423 at 51k in), measured on the seeded benchmark protocol
([full matrix](batch/README.md#prefill); older published prefill rows came
from an unseeded harness that let the prefix cache contaminate the numbers,
and are not comparable). How each number was won:
[docs/optimizations.md](docs/optimizations.md).

The server listens on `0.0.0.0` and is unauthenticated unless you give it a key.
For anything past your own machine, add one first — everything reads it from
`.env` or `api_key.txt`, and nothing needs it otherwise:

```bash
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env
```

No compose, no clone — plain Docker runs the same image with one command and
prepares the model itself on the first boot (into a named volume, so it
survives container replacement):

```bash
docker run -d --name qwen --gpus all --ipc=host -p 18020:18020 \
  -v qwen-models:/app/models -v qwen-cache:/cache \
  --restart unless-stopped ghcr.io/syv-ai/qwen38-27b-rtx3090:latest
```

`batch` after the image name is the other mode, and the knobs compose reads
from `.env` become `-e` flags (`-e VLLM_API_KEY=...`, `-e SPEC=dflash2`, ...) —
[docs/docker.md](docs/docker.md#plain-docker-no-compose) has the mapping.

Or by hand in a venv (same steps: model download, requantization, vLLM
patches, `verify.sh`) — see [Setup](#setup).

### If you are the only user, do this

The command above starts the conservative default — MTP speculation, 8 request
slots, 64k context, 120 tok/s greedy at C1. Two settings are worth more than
every other knob in this repo put together, and a third is worth a great deal on
one particular workload:

```bash
printf 'SPEC=dflash2\nPREFIX_CACHE=1\n' >> .env
# add DFLASH_TOKENS=15 if your answers quote your prompts — see below
docker compose --profile single up -d
```

or, in the venv install:

```bash
venv/bin/python prepare/fetch_dflash2.py   # once, 1.2 GB (Docker's prepare step does it for you)
SPEC=dflash2 PREFIX_CACHE=1 bash single-user/start_qwen.sh
```

`SPEC=dflash2` swaps Qwen's MTP head for the DFlash2 block drafter: 7 tokens
proposed in one pass instead of 4 chained ones. `DFLASH_TOKENS=15` then lets the
target verify 16 tokens per step — the drafter still proposes the 7 it was
trained for, and the remaining positions are filled from the request's own
context, which costs nothing to draft and is exactly right whenever the answer
quotes the prompt. `PREFIX_CACHE=1` keeps the document you already sent, both
its attention KV and its recurrent state. One request at a time, greedy, RTX
3090 at 250 W:

| decode | MTP (default) | `SPEC=dflash2` | `+ DFLASH_TOKENS=15` |
|---|---|---|---|
| 8 real chat prompts | 118 tok/s | 132 | **133** |
| reproducing a 25k-token document | n/a* | 260 | **382** |
| request slots / context | 8 / 64k | 8 / 64k | 4 / 56k |

`VLLM_DFLASH2_CHAIN=1` adds drafter-free n-gram chains on top
([#38](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/38), ported from
@Dmtrii-tesla's fork with permission): while a request keeps reproducing its
context, whole verify blocks come from history alone and the drafter's forward
and graph replay are skipped until the first rejected token — +7% on the copy
cell here (256.9 → 276 tok/s at `DFLASH_TOKENS=7`), flat on prose, greedy
requests only by default (`patches/dflash2-ngram-chains.patch` explains why
sampling keeps the drafter). Off by default.

<sub>\* drafting from the context only exists in `SPEC=dflash2`. The two right
columns are one server session, where run-to-run greedy divergence is ±3-5%;
reproduce them with `venv/bin/python bench/labd_bench.py <tag> --ctx 20000`.</sub>

`PREFIX_CACHE=1` is orthogonal to the other two and worth as much again in a
chat client: a second turn against that same 25k-token document takes 0.56 s to
first token instead of 22.4 s, with the answers unchanged token for token.

**Read that table by column, not by its last cell.** `SPEC=dflash2` is the upgrade
for everyone; `DFLASH_TOKENS=15` is for one workload. On chat it is worth 1%,
because the eight positions past the drafter's own block are filled from the
prompt and a chat answer does not quote the prompt — measured over
`bench/prompts_real.jsonl`, positions 7-14 take **72 of 11,069 accepted tokens
(0.65%)**, and @changtimwu measured exactly zero for them on a TP=2 box in
[#22](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/22). What you pay for
that 1% is half the request slots and 8k of context, because a 16-token verify
block doubles the recurrent-state page every resident request holds (1.66 GiB
against 0.88 by the gotcha-33 fit). So: set it if you are quoting documents or
applying edits, where it is worth 47%, and leave it at the default 7 for a chat
or agentic client. `DRAFT_TOKENS`/`DFLASH_TOKENS` is one variable you can flip
per service.

All of it is lossless: speculative decoding samples the same distribution as no
speculation at all, the prefix cache resumes recurrent state rather than
approximating it, and GSM8K reads 96.0-96.5% across the three columns.
`SPEC=dflash2` is a one-user mode either way
(see [concurrency](#dflash2-at-240k-ctxhuge-kvarn-also-combines-with-specdflash2)).
Every other knob: [single-user/](single-user/).

### DFlash2 at 240k: `CTX=huge` (KVarN) also combines with `SPEC=dflash2`

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
reads 95.0-96.5% on the same 200-question harness ([docs/quality.md](docs/quality.md)),
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
WDDM paging behavior underneath ([#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25)).
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
Gotcha 53 has the counters to read, the four costs that turned out to be this,
and the profile that gives it room (`KV_MEM=3000000000 DFLASH_MAX_LEN=8192`,
free for the shipped head at width 7).

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
[#13](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/13) and consistent with the
uncaptured verify being launch-bound: launches that are nearly free here are not free
there.

Two limits worth knowing before you point this at anything: what it does with
more than one user, and what the long verify block costs.

**It is a one-stream mode, and the limit is the pool rather than `MAX_SEQS`.** An
earlier version of this paragraph said the knob was the seat count and that
`MAX_SEQS=8` lifts it. It does not, and @mjungnickel18 was right to push back in
[#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25). A *resident* request
reserves 1+k = 8 recurrent-state slots — **15.8% of the 69,758-token `CTX=fast` pool**,
~0.82 GiB of its pinned 5.20, which is the 0.88 GiB [gotcha 33](docs/gotchas.md) fitted
from the memory model — before it holds one token of context. Seven fit with 128-token
prompts, five with 4k-token ones and two with 16k ones. MTP's k=4 costs 0.44 GiB, so
eight fit, four of them at 16k. Past that the extras queue, and once the pool is full
something has to be preempted and recomputed to make room — a ramp of tiny requests
hits that at the seventh. `MAX_SEQS` decides how many requests are *admitted*, not how
many can run. The pool itself is **unchanged** by the setting (268,169 tokens at
`CTX=huge` either way, ~8 MiB total between 1 slot and 8 —
[gotcha 33](docs/gotchas.md)), which is the part of the old paragraph that was right.

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
piecewise instead ([gotcha 38](docs/gotchas.md)). It does boot at 240k since `82bd62d`,
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
[docs/gotchas.md](docs/gotchas.md) has the residue table; `bench/residue_sweep.py`
sweeps all 128 residues and judges every answer on how much of the document came
back, and `bench/verbatim.py` self-tests that rule against all three shapes.

### Third-party checkpoints (uncensored builds and others)

`MODEL=` points the launchers at any Qwen3.8-27B checkpoint in the same
`compressed-tensors` shape. Two routes, easiest first.

**Ready-made:**
[leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound](https://huggingface.co/leminkozey/Qwen3.8-27B-Uncensored-W4A16-AutoRound)
([#45](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/45)) is an
abliterated Qwen3.8-27B already quantized with this repo's own recipe —
AutoRound W4A16 body plus the `prepare/` head requant — so it serves without
any preparation. Its author measured ~100 tok/s warm at `SPEC=dflash2
CTX=huge` on a 3090 with coherent output and a 45k-context needle retrieved,
and a second tester confirmed `SPEC=mtp` works. Community-built and
community-verified; not benchmarked on this repo's reference box.

**Any other export**, including single-shard and asymmetric-AWQ ones the base
model's three `quant_*.py` scripts cannot open, goes through the streaming
requant (contributed in
[#37](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/37)). The worked
example is
[philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-Uncensored-Aggressive-W4A16-AWQ)
— an abliterated (de-refused) Qwen3.8-27B, W4A16 AWQ, with the vision tower and
the grafted MTP head both preserved. Prepare it once, then serve it:

```bash
venv/bin/python prepare/fetch_thirdparty.py          # ~18.6 GB; or: fetch_thirdparty.py <hf-repo>
venv/bin/python prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16
venv/bin/python prepare/build_draft_vocab.py  models/Qwen3.8-27B-Uncensored-W4A16 \
  --ids prepare/draft_vocab_ids.json

MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 SPEC=mtp CTX=long PREFIX_CACHE=1 \
  MAX_LEN=100000 bash single-user/start_qwen.sh
```

It needs `prepare/quant_heads_stream.py` rather than the three `quant_*.py` steps
the base model uses, for two reasons that are properties of the checkpoint and not
of the model: it ships as **one 18.6 GB shard**, which the three scripts read into
RAM whole before rewriting, and its body is **asymmetric AWQ**, which those scripts
would copy onto the symmetric tensors they write — vLLM then looks for a
`weight_zero_point` that was never written. The streaming script handles both and
produces the same tensors otherwise; `bash verify.sh --no-server` with `MODEL=` set
checks the result exactly as it checks the base model.

**`SPEC=dflash2` needs its pool resized for this checkpoint.** After requantization
it is 15.68 GiB of weights against the fast variant's 14.71, and the DFlash2 branch
pins the KV pool *in bytes* (`KV_MEM`) rather than sizing it from
`--gpu-memory-utilization`, so the pool does not give that gigabyte back. The server
loads, captures graphs, and then dies on the split-KV verify buffer:

```
Model loading took 15.71 GiB
reserved 5.2 GiB memory for KV Cache as specified by kv_cache_memory_bytes config
torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 926.44 MiB is free
```

Hand that gigabyte back and it comes up. `CTX=long` (int8 KV) is the one to spend it
on, because it buys roughly twice the context per byte of pool that `CTX=fast` does:

```bash
MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 SPEC=dflash2 CTX=long PREFIX_CACHE=1 \
  KV_MEM=4456028569 DFLASH_MAX_LEN=98304 bash single-user/start_qwen.sh
```

Measured here, RTX 3090 at 250 W: a 4.15 GiB pool holding **103,033 tokens** at
98,304 `max-model-len` (4.6% margin) and **85.7 tok/s** greedy on a 400-token
answer. The checkpoint keeps its vision tower, and that run had `VISION=1` — images
came back described correctly — so the numbers are an upper bound on what the
default `VISION=0` needs, which drops the tower's weights entirely.

`SPEC=mtp` needs no `KV_MEM` of its own: its pool is profiled from `GPU_UTIL` rather
than pinned, so it absorbs the extra gigabyte by shrinking the pool for you. It is the
mode to reach for first on this checkpoint. The pool it lands on will not hold
`CTX=long`'s stock 150k, though, which is what the `MAX_LEN=100000` above is — the
figure this checkpoint has been run at.

### 256k the stock way: int4 KV (`single-user/alternative.sh`, experimental)

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
[#42](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/42) by @lachhabw).

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

### More than one GPU

Everything here is written for one 24 GB card; multi-GPU goes through untouched
via `EXTRA_ARGS`:

```bash
SPEC=dflash2 PREFIX_CACHE=1 EXTRA_ARGS="--tensor-parallel-size 2" bash single-user/start_qwen.sh
```

What the second card is worth is now measured, not assumed —
[#40](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/40) ran a controlled
1-vs-2×3090 A/B on this harness (same box, same install, PCIe 4.0 x8, **no
NVLink**, 275 W):

- **+16–35% decode across C1–C8** (DFlash2 greedy: 127.4 → 161.6 at C1). Worth
  it at batch 1: decode is bandwidth-bound here, and TP=2 is a second memory
  system, not idle compute.
- **The DFlash2 residency ceiling is a 24 GB property, not a drafter
  property.** On one card DFlash2 collapses at C8 (3.9 s TTFT — the
  recurrent-state pool exhaustion documented above) and MTP wins; on two cards
  DFlash2 wins at every concurrency measured. On 2×24 GB, point everyone at
  DFlash2.
- **The launcher no longer pins `KV_MEM` under TP>1** (their finding, this
  fix): the pin is a single-card constant applied per worker, and it stranded
  ~16 GiB across two cards — 137,210 tokens of pool where `GPU_UTIL` sizing
  gets 302,223, at no measured decode cost. Export `KV_MEM` to pin anyway.
- **Keep `DFLASH_TOKENS=7` at TP>1** for now: the one T=15 datapoint at TP=2
  (same issue) lost 27% at C1 with the lookup-filled tail accepting nothing,
  which is under diagnosis. The launcher warns.
- NVLink appears to buy little:
  [#7](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/7)'s NVLink box at
  330 W and #40's PCIe-x8 box at 275 W land within a few percent of each other
  at C1.

Also reported working: **2× RTX 5060 Ti 16 GB**
([#22](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/22)) — the "would
not fit on one card" case. The graph budget and `MAX_SEQS` defaults are still
single-card calibrations; more A/Bs like #40's are the most useful numbers you
can send.

## Benchmarks

Full tables per mode in [batch/README.md](batch/README.md) and
[single-user/README.md](single-user/README.md); quality in
[docs/quality.md](docs/quality.md). Reproduce any of it with
`bash bench/run_benchmarks.sh batch|single` against your own server.

### vs. ninfer-3090

[ninfer-3090](https://github.com/Don-Chad/ninfer-3090) is a standalone C++/CUDA engine
that publishes cohort benchmarks for this model on this card. Theirs are 1,024-token
answers from 29-34-token prompts, greedy, MTP3, int8 KV, prefix reuse off, an
8,192-token context window, and **thinking on** at `reasoning_effort=medium`, so their
1,024 tokens include reasoning. Ours are 8 realistic chat prompts (English, Danish,
code), 1,024-token answers, model-default sampling, thinking off:

| Cohort | ninfer-3090 (MTP3) | this repo, batch | single-user, MTP | single-user, DFlash2 |
|---|---|---|---|---|
| C1 | 71.00 tok/s | 45.5 | 111.1 | **121.8** |
| C2 | 90.66 tok/s | 86.3 | 191.8 | **195.5** |
| C4 | 100.28 tok/s | 168.3 | 268.5 | **278.9** |
| C8 | 165.33 tok/s | 324.9 | **407.3** | 389.9† |
| C64 (128 in / 512 out) | not supported | **~1,035** | — | — |

† measured before `PREFIX_CACHE=1` became the single-user default. With
prefix caching on, cached prefixes (and, under `--mamba-cache-mode align`,
their recurrent-state pages) stay resident through the cohort ladder, so the
DFlash2 residency ceiling bites earlier and this cell reads ~324 tok/s with a
2-3 s TTFT on the current stack — independently measured at 321.8 in
[#40](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/40), which is what
prompted the re-measurement. C1–C4 read the same or slightly better than the
table. One card, many concurrent users: `SPEC=mtp` remains the right mode.

Decode rate, C × 1000 / mean TPOT. All four of our columns were re-measured together
on the current stack with `bench/run_benchmarks.sh`, keeping the second run after each
restart as the script advises; greedy instead of default sampling reads
131.2 / 214.6 / 285.7 / 405.5 for DFlash2. Run-to-run spread on the same server is
5-8%, so treat one-decimal differences between the three right-hand columns as noise —
C1 and C8 are where the modes genuinely separate.

Theirs is the **decode** column of their table; their end-to-end column reads
70.19 / 89.43 / 97.89 / 161.28, and an earlier version of this table quoted *those*
against our decode rate, which was not like-for-like. What still is not like-for-like,
in their favour and ours: their C1 is a single prompt in a single run with no error
bars, thinking is on for them and off for us, and they publish no power limit or driver
version — ours is an RTX 3090 pinned at 250 W. Peak VRAM is comparable (23.0 vs
22.1 GiB at C8). The gap is mostly vLLM's continuous batching plus the memory this
repo's requantization frees up.

### Quality

The whole stack is quantized, so the honest question is what it costs. Short
version: **IFBench 78.3** prompt-level strict vs 79.5 for the unquantized model
(one point), **perplexity 8.09** on 33k held-out tokens, **GSM8K 96.5%** (200
questions, greedy). Speculative decoding — MTP, DFlash2 and the lookup drafter —
is exact by construction and changes none of it; the int8-activation steps in
batch mode are the only knobs that trade accuracy for speed, and they cost
0.9-3.7% perplexity depending on how far you push them. Per-configuration
tables: [docs/quality.md](docs/quality.md).

### Results from other hardware

Community reproductions of the single-user headline number, harness runs first.
`bench/run_benchmarks.sh single`, greedy, second run (the first reads low):

**Set the power limit before you compare anything.** Every number in this repo
is an RTX 3090 at 250 W, and on this card that is not a soft preference. A
sustained-load ladder from [#62](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/62)
(14 minutes per cell, same service): 200 W gives 57.5 tok/s at 781 MHz, 250 W
gives 85.6 at 978 MHz, and 280 W gives 86.7 — it hits 90 °C within two minutes,
pins the fan at 100% and throttles back to the same throughput. Prefill loses
about the same third at 200 W. So a quiet home box capped at 200 W is measuring
its power cap rather than this stack, and nothing above 250 W is worth the
noise.

| card | power | C1 decode | notes | source |
|---|---|---|---|---|
| RTX 3090 (reference) | 250 W | 133 tok/s | pool 57,669 tok, ppl 8.09 | this README |
| RTX 4090 | 450 W | **135.5 tok/s** | pool 57,669 and ppl 8.0921 reproduce exactly; no-spec control 60.3 (DFlash2 worth 2.31x); +1.9% from ~8% more bandwidth — batch-1 decode is bandwidth-bound, the extra compute has nothing to bite on | [#32](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/32) |

Measured with their own clients rather than the harness — comparable to each
other only loosely, and not rows for the table above:

- **CMP 170HX 40 GB (GA100, sm80)**: 133.7 tok/s median (3x900 tok, greedy) on
  the shipped fast target — the first sm80 datapoint, level with the 3090 —
  and 97.8 tok/s on their own w8a16 int8 target after the sm80 repack
  workaround in [#27](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/27)
  (gotcha 41).
- **RTX 5090 32 GB (sm120)**: ~410-449 tok/s on code and ~198 on prose at
  `CTX=fast`, 500 W cap, roughly flat out to `CTX=huge` at 240k — different
  prompts, output length and rate definition, so deliberately not in the table
  (their own insistence, and correct). Setup gotchas and the full ladder:
  [#35](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/35).
- **RTX 4090, Windows 11 / WSL2 (Docker path)**: reproduces with zero repo
  changes; CTX ladder incl. huge's pool byte-identical to the 3090 reference
  (268,169), concurrency ladder to N=8, and a measured both-ways case for
  leaving the `KV_MEM` pin alone — [docs/wsl2-4090.md](docs/wsl2-4090.md).
- **RTX 4090, Windows 11 / WSL2, second box**: all three single-user profiles
  plus the experimental int4 one (230,830-token pool at 160k), and 135k
  real-task numbers on the MTP + FP8 daily-driver profile — 62 tok/s decode on
  QA over the document, TTFT 5.4 s → 0.33 s on a repeat turn. Also the
  `nvidia-smi dmon` detector for WSL2 host-backed memory now in gotcha 43 —
  [#61](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/61).
- **RTX 3090, Windows 11 / WSL2**: independent confirmation of the int8 prefill
  stack on Ampere — `INT8_ACT=int8` +59%/+57%/+37% at 5k/21k/66k, the int8-QK
  attention adding +1.8% at 21k and +6.3% at 66k on top, against this repo's
  +2.7% at 16k and +5.3% at 51k. Plus the power-limit ladder quoted above —
  [#62](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/62).
- **Dual-GPU reports**: the controlled 1-vs-2×3090 A/B in
  [#40](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/40) (+16–35%,
  161.6 C1 greedy at 275 W, PCIe x8 without NVLink; independently reproduced
  in-thread at 153.6/250 W by a second dual-3090 box), the NVLink dual 3090 in
  [#7](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/7), dual 5060 Ti in
  [#22](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/22). See "More
  than one GPU" above for what transfers.

### Why this isn't just `vllm serve`

Nine things, from requantizing both embedding matrices to drafting straight out
of the prompt — one line each, then the reasoning and measurements, in
[docs/optimizations.md](docs/optimizations.md).

### What each step buys

Measured cumulatively on the 3090, 64 concurrent, 128 in / 512 out, `vllm bench
serve` random dataset:

| step | what it does | e2e output tok/s | steady-state decode |
|---|---|---|---|
| W4A16 AutoRound body (as published) + fp8 KV | int4 Marlin kernels, 66.7k-token pool | 370 (48 conc, 256/256) | — |
| + lm_head / embed_tokens int8 | 2.6 GB of cache pages back | 516 | ~585 (37 requests resident) |
| + fp16 recurrent state | 64 requests resident, half the state traffic | 707 | ~830 |
| + int8 activations, MLP (default) | int8 tensor cores on 74% of the FLOPs | 942 | ~1,094 |
| + int8 activations, everything (`INT8_LAYERS=.`, needs `GPU_UTIL=0.95`) | | 1,042 | ~1,222 |

And single-stream on realistic prompts (single-user mode, T = model default /
greedy):

| step | tok/s | tokens per step | draft acceptance, position 0 |
|---|---|---|---|
| no speculation | 46 / 46 | 1.0 | — |
| MTP-2 as shipped (bf16 drafter, full head, fp32 state) | 66 / 79 | 2.1 / 2.4 | 65% / 80% |
| MTP-4, int8 drafter, 40k draft head, fp16 state | 78 / 99 | 2.2 / 2.7 | 58% / 70% |
| + probabilistic draft sampling (`CTX=fast`, k=4) | 90 / 98 | 2.6 / 2.7 | 69% / 70% |
| same with 3 drafts on FlashInfer/fp8 KV (`CTX=long`, 150k) | 84 / 89 | 2.5 / 2.4 | 69% / 71% |
| + sampler patch, split-KV verify attention | 93 / 99 | 2.6 / 2.6 | 69% / 70% |
| + draft vocab counted over the model's own outputs | 107 / 109 | 2.9 / 2.9 | 74% / 74% |
| + GPTQ-int4 lm_head (calibrated) | 109 / 112 | 2.8 / 2.8 | 73% / 73% |
| + GPTQ-int4 MTP module (**fast variant, shipped**) | **~114 / 118-124** | 2.8 / 2.9-3.0 | 74% / 77% |
| DFlash2 block drafter instead of MTP (`SPEC=dflash2`, int4-requantized) | **118 / 126** | 3.14 / 3.34 | ~75% / ~78% |
| + drafting from the context (`LOOKUP=1`, on by default) | **130** at C1, up to **259** where the model reproduces its context | 3.3-7.8 | |
| + a 16-token verify block the context fills (`DFLASH_TOKENS=15`) | **133** at C1, up to **381** reproducing context | 3.4-15.0 | |

(Steps 4-6 are the same 8-prompt protocol; greedy is deterministic for a
given server and request order but differs between configs and even with
prefix-cache hits, so single runs carry ±3-5% on tokens/step —
`bench/run_benchmarks.sh single` reproduces 111.1 / 120.0 tok/s decode at C1,
the best repeats read 119 / 124.)
Going deeper (k=5) loses again: 106 / 105. k=4 is the knee, but on vLLM
0.28.0's FlashInfer backend (needed for fp8 KV, i.e. for 150k context) four
drafts crash the engine with an illegal memory access as soon as one request
finishes while another is mid-generation — club-3090 reports the same "n=4
eventually dies, n=3 stable" pattern — so `CTX=long` drafts 3 and gives up
~7%; `CTX=fast` (FlashAttention, bf16 KV, ~64k context, the default) keeps k=4
and is also the only backend the split-KV attention patch applies to.

Two things that did *not* help, measured rather than assumed: fine-tuning the
MTP head on the model's own outputs (KL halves, greedy top-1 on response
tokens unchanged; `drafter/README.md`), and retuning Marlin's tile
configuration for M ≤ 16 on sm86 (3-7% per GEMM in isolation,
nothing measurable end to end — the remaining gap to peak bandwidth is the
memory system's ramp on 16-92 MB reads, not the kernel).

## Setup

The default install is the container ([Quick start](#quick-start) — the
prebuilt image already contains everything this section builds), so this
manual venv path is for hacking on the stack, or running it bare-metal.

> **Python 3.14 works natively** — nothing in this repo needs changing, but
> `python3.14-dev` does need installing. See [docs/python-314.md](docs/python-314.md),
> with a full RTX 3090 reproduction of the tables below in
> [docs/reproductions/native-3090.md](docs/reproductions/native-3090.md).

You need: a 24 GB Ampere or newer NVIDIA card, a recent driver, Python 3.12,
~40 GB disk — and if the host has less than ~16 GB of free RAM, load the
weights with the streamer instead of the stock loader (gotcha 45: the stock
loader peaks at whatever RAM exists; the streamer is bounded and faster). Everything below is CPU-safe to run while the GPU does other
things; the container details live in [docs/docker.md](docs/docker.md).

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 ~/qwen-serving
cd ~/qwen-serving

python3 -m venv venv
venv/bin/pip install vllm==0.28.0 huggingface_hub hf_transfer ninja \
  flashinfer-python flashinfer-cubin==0.6.13 pandas
# pandas is what `vllm[bench]` pulls in for the custom-dataset path: without it
# bench/prefill_ab.sh's decode guard dies with "Please install vllm[bench] for
# bench support" after the prefill rows have already run.
# flashinfer makes the DFlash2 selector ~2x faster than its torch.topk fallback,
# and vLLM only *uses* it if nvcc is on PATH or flashinfer-cubin is installed --
# a bare `pip install flashinfer-python` silently falls back with one INFO line
# (#35). cubin publishes up to 0.6.13, so the version pair needs
# FLASHINFER_DISABLE_VERSION_CHECK=1, which the launchers export. Do not fix the
# mismatch by downgrading flashinfer-python: that drags torch back and breaks
# vLLM's C extension.

# model, ~19.5 GB
HF_HUB_ENABLE_HF_TRANSFER=1 venv/bin/hf download \
  dbirks/Qwen3.8-27B-W4A16-AutoRound \
  --local-dir models/Qwen3.8-27B-W4A16-AutoRound

# requantize lm_head + embeddings + the MTP draft module (CPU only, a few minutes)
venv/bin/python prepare/quant_lm_head.py models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_embed.py   models/Qwen3.8-27B-W4A16-AutoRound
venv/bin/python prepare/quant_mtp.py     models/Qwen3.8-27B-W4A16-AutoRound
# 40k-token draft head for single-user mode (uses the shipped id list)
venv/bin/python prepare/build_draft_vocab.py models/Qwen3.8-27B-W4A16-AutoRound \
  --ids prepare/draft_vocab_ids.json
# single-user "fast" variant (~1 GB from the Hub, hardlinks the rest): int4-GPTQ
# lm_head + drafter; single-user/start_qwen.sh picks it up automatically
venv/bin/python prepare/fetch_fast_variant.py
# optional: the W4A16 DFlash2 block drafter (1.2 GB) for SPEC=dflash2 single-user mode
venv/bin/python prepare/fetch_dflash2.py
# optional: a third-party checkpoint instead of the base model (e.g. the uncensored
# build, ~18.6 GB, its own requant step; MODEL= serves it -- see "Third-party
# checkpoints" above)
venv/bin/python prepare/fetch_thirdparty.py
venv/bin/python prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16

# patch vllm (all compatible patches are written against 0.28.0; reapply after upgrades)
for p in patches/*.patch; do
  case "$p" in
    patches/dflash2-backport.patch) echo "skip $p (DFlash2 is native in vLLM 0.28.0)"; continue ;;
  esac
  patch -p1 -d venv/lib/python3.12/site-packages/vllm < "$p"
done
# optional: the KVarN 4/2-bit KV cache for 262k context (docs/long-context.md)
bash kvarn/install.sh

# api key — optional, but the server binds 0.0.0.0 and is open without one
openssl rand -hex 24 > api_key.txt
```

Then `bash verify.sh --no-server` — it checks the venv and vLLM version, that
every compatible patch in `patches/` is actually applied, and that the model has been
requantized (lm_head, embeddings, MTP module, draft head). Then pick a mode
and follow its README:

- **[batch/](batch/)** — throughput. `bash batch/start_qwen.sh`
- **[single-user/](single-user/)** — latency. `bash single-user/start_qwen.sh`

First start takes a few minutes (torch.compile, CUDA graph capture, flashinfer
JIT). Test it:

```bash
curl http://localhost:18020/v1/chat/completions \
  -H "Authorization: Bearer $(cat api_key.txt 2>/dev/null)" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b",
       "messages": [{"role": "user", "content": "hej"}],
       "chat_template_kwargs": {"enable_thinking": false}}'
```

Qwen recommends temperature 0.7 / top_p 0.8 for instruct mode, and 1.0 / 0.95
with thinking enabled (the default).

Tool calling works over the same endpoint — send `tools` with `tool_choice:
"auto"` and the reply carries `tool_calls`. Both launchers set
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`; the parser has to
read Qwen's XML call format, which is what this model's chat template emits —
not the JSON that `hermes` reads. `TOOLS=0` turns it off.

To check the numbers on your own card: `bash verify.sh` (also probes the live
server and prints which attention backend and KV pool it came up with), then
`bash bench/run_benchmarks.sh batch` or `... single` reproduces the tables
above against the running server (`--prefill` and `--long` add the prefill
matrix and the long-context rows), `bash bench/real_rep.sh <tag> 3 0` repeats
the single-stream row, and `python bench/quality_battery.py <tag>` the
perplexity / GSM8K rows. For the concurrency rows,
`python bench/conc_ladder.py --n 1,2,4,8 --ctx-tokens 4096`; for the prompt-length
bug, `python bench/residue_sweep.py <tag>` (all 128 residues) with
`python bench/verbatim.py` as its offline self-test.

## The rest

| | |
|---|---|
| [docs/optimizations.md](docs/optimizations.md) | Every optimization in full: why it was needed, what it measured, which patch implements it. Includes the two speculative-decoding modes (MTP and DFlash2) and the lookup drafter. |
| [docs/gotchas.md](docs/gotchas.md) | 18 things that each cost us hours — read before debugging something that looks like a vLLM bug. |
| [docs/quality.md](docs/quality.md) | IFBench, perplexity and GSM8K per configuration. |
| [docs/docker.md](docs/docker.md) | The container image, and an independent WSL2 reproduction. |
| [docs/long-context.md](docs/long-context.md) | 262k context with the KVarN 4/2-bit KV cache, what vLLM's own per-token-head KV modes are worth here, and how to run the DFlash2 drafter past 64k (`CTX=long`, 114-139k — worth it only for context reproduction). |
| [batch/](batch/) · [single-user/](single-user/) | The two serving modes: full benchmark tables, every env knob, systemd units. |
| [prepare/](prepare/) | The one-time model-preparation scripts run by [Setup](#setup) (and by `docker compose run --rm prepare`). |
| [drafter/](drafter/) | How the draft vocabulary, the int4 drafters and the DFlash2 requantization were built — including what did not work. |
| [kvarn/](kvarn/) | The KVarN 4/2-bit KV cache port. |

## License

Apache-2.0, same as the model.
