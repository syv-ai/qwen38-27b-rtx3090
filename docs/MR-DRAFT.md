# MR draft: speculative decoding at depth, the CPU tier, and two fixes

> This file is the source for the PR #46 description; the live body on GitHub
> is generated from it. Edit here first.

## Summary

This PR is the writeup of a two box effort: an RTX 4090 under Windows 11 / WSL2
(Docker) and an RTX 3090 on native Ubuntu (Python 3.14, venv, no container).
Everything below was measured on at least one of the boxes, and the load
bearing claims were measured on both. The detailed notes ship with the PR:
[docs/wsl2-4090.md](docs/wsl2-4090.md), [docs/ubuntu-3090.md](docs/ubuntu-3090.md),
and [docs/reproductions/native-3090.md](docs/reproductions/native-3090.md).

What it carries:

1. **A fix for the CPU offload tier on WSL2** (`patches/0020-offload-wsl2-devptr.patch`).
   The OffloadingConnector crashes with an illegal memory access on WSL2. We
   root caused it, fixed it, and validated the fix on both platforms. This is
   probably an upstream vLLM bug rather than a repo bug: any WSL2 user of the
   connector should hit it, with any model.
2. **A one flag fix for prefix caching going dead** on int4 KV + DFlash2
   (`--prefix-match-unit 848`), with the analysis of why it dies.
3. **An int4 attention kernel for speculative decoding**
   (`patches/0021-spec-decode-int4-kv-mq3d.patch`), opt in via `VLLM_INT4_MQ_3D=1`.
   About 3.3x end to end on deep int4 spec decode. It defaults off, and section
   3 lists the tests it still owes before it earns default on.
4. **A measurement correction** that matters to anyone benchmarking this stack
   over SSE. It reversed one of our own early conclusions.
5. **Operating notes**: what actually stays cached at depth, which settings
   matter under mixed load, and what the engine's startup numbers do and do not
   tell you.

## 1. If you benchmark speculation over SSE, read this first

With speculation on, each SSE chunk is one speculative step, not one token; at
n7 a chunk averages about 3 tokens. A benchmark that counts chunks per second
therefore undercounts spec-on throughput by roughly 3x while counting spec-off
correctly, so every comparison is biased against speculation. We made exactly
this mistake, concluded "speculation loses at depth" from it, and retracted.
The fix: request `stream_options.include_usage` and read
`usage.completion_tokens`; report emitted tokens per step alongside as the
sanity check. Both boxes reproduced the effect independently. Every spec-on
number in this PR is counted in tokens.

## 2. Deep context numbers (72.6k token prompt, 4090/WSL2)

| config | decode tok/s (deep) |
|---|---|
| int4 KV + MQ-3D kernel, DFlash2 n7 | **70.5 to 75.2** |
| long (int8 KV), DFlash2 n7 | 64 to 68 |
| huge (KVarN 4/2-bit), DFlash2 n7 | ~60 to 63 |
| int4 KV, spec off | 35.55 |
| int4 KV, stock #42 dispatch, DFlash2 n7 | 22.3 |

With the right kernel and flags, speculation wins by about 2x at depth. Our
earlier conclusion that it loses was two errors compounding: the chunk counting
above, plus the stock int4 dispatch, which genuinely does lose (0.63x against
spec off).

## 3. The int4 MQ-3D kernel

Stock #42 launches its spec-verify attention once per query position (a 2D
grid), but DFlash2's verify is an 8 row multi-query batch, so each step walks
the whole KV serially: measured 8.14 steps/s at 72.6k depth. Dispatching all
query rows in one 3D launch with a reducer guard (33 lines) restores split-KV
for the verify batch: 8.14 to 25.2 steps/s, which is 22.3 to ~73 tok/s end to
end (~3.3x).

It ships opt in (`VLLM_INT4_MQ_3D=1`, default off) because six checks are still
owed before anyone should trust it as a default: a 72k operator/logit oracle,
full length exactness, per position logit comparison, eager+captured+q=9
parity, a shallow crossover floor (shallow currently reads -16%), and capture
aware dual variant switching. It is its own commit, so it drops cleanly if you
want the rest without it.

## 4. Prefix cache zero reuse on int4 + DFlash2: the one flag fix

Symptom: with int4 KV and DFlash2 together, the prefix cache never hits.
Every request re-prefills; a warm 72.6k prompt costs 62 seconds every time.
The cross box isolation: int8 + DFlash2 hits, int4 without DFlash2 hits,
therefore the bug is the interaction of the two.

Cause: two sites in the kvarn v2 port carry assumptions that contradict each
other. The GCD computation excludes sliding window groups (its comment says
their block divides the primary by construction, which is true) and lands on
hash unit 1696. But the SW guard then requires the SW block (848) to be a
multiple of that hash unit, and 848 % 1696 is never 0: the exclusion
guarantees the guard fails. Under int8 both values are 864, so the
contradiction is invisible there.

Fix: the port's own escape hatch, `--prefix-match-unit 848` (the GCD). Every
clause then passes by arithmetic (1696 % 848 = 0, 848 % 848 = 0, both read
clauses). Verified: warm whale TTFT 62s to 4.3s. The rule that generalizes:
the prefix match unit must equal the SW block, and it is per model, KV dtype,
and draft length... 848 is this model's number, not a constant.

## 5. Retention: what actually stays cached

Some vocabulary we used throughout, so the numbers below read plainly: a
**whale** is one large prompt (say 72k tokens); **minnows** are the small 4k
requests competing with it for the same engine.

The engine's startup line ("Maximum concurrency for N tokens per request:
X.XXx") divides the pool by max_model_len. We measured what that number means
in practice, and the answer depends on the question you ask:

- **Capacity** (recheck only the newest context): the newest one is warm.
  On the 3090, 2.24s against a 50.35s cold prime, a 22x speedup.
- **Round robin** (recheck A, B, C in order, the pattern a multi user service
  actually runs): **0 of 3 hit**, because each recheck's own prefill evicts
  the next context before it is asked.

Then we measured the per context cost directly, by finding the largest context
length where two contexts both stay warm (per length salts, offload tier off,
and each boundary re-run alone in a fresh boot before we called it a
measurement): a cached context costs about **2.5 to 2.9x its token count** at
51k to 61k on the 4090 (int4 geometry, 300,583 token pool), and about
**2.3 to 2.7x** at 25k to 29k on the 3090 (int8, 136,429 pool). The extra cost
is the per sequence state: mamba state pages are constant size per sequence,
therefore they do not shrink with shorter prompts, and the drafter adds its
own groups.

We then tried to pin down the cost model and killed both simple candidates,
one per geometry, with pre-registered tests: pure multiplicative
(`cost = m * L`) fails on int4 (the K=2 boundary caps m at 2.93 but a K=3 test
demands more than 3.18; no m satisfies both), and pure additive
(`cost = L + F`) fails on int8 (the K=2 and K=3 boundaries demand F in
disjoint intervals). An affine shape fits every boundary on both boxes, but
with two free parameters against six one sided constraints that is a shape
with room to spare, not a finding; this PR ships the measured brackets and
leaves the cost model open.

The practical takeaways:

- The startup concurrency number is **correct for a single full length
  request** (section 7 has the measurement) but overstates warm multi context
  capacity by 2x or more, and nothing in the log says which case you are in.
- **The offload fix in section 6 changes the round robin story**: with the
  CPU tier, the same 3-whale round robin goes from 0/3 to 3/3. Not because
  the GPU holds them (it holds about one), but because eviction becomes a
  RAM restore instead of a full re-prefill.

## 6. The OffloadingConnector fix

Full writeup: docs/wsl2-4090.md, the offload section. The short version:

The Triton `swap_blocks` kernel dereferences **host** virtual addresses of the
`cudaHostRegistered` /dev/shm region. On native Linux, UVA makes host VA equal
device VA, so this works; but that is a coincidence of the platform, not a
contract of the API. WSL2 registers successfully and maps the region at a
**different** device address, therefore the kernel faults (illegal memory
access) on the first real eviction.

The fix: after registration, ask `cudaHostGetDevicePointer` for the device
address, carry the delta on the CPU views, and add it where
`compute_sub_block_ptrs` builds pointers. The delta is 0 on native Linux, so
no behavior change there. Every failure branch raises: a platform where the
device pointer is unavailable refuses to start instead of running with
addresses that fault later.

Validation, both platforms:

| arm | WSL2 / 4090 | native / 3090 |
|---|---|---|
| with fix | 3x 72.6k whales primed, all recheck warm in 4.2 to 4.3s, 0 IMAs | 52.4 tok/s, tokens byte identical to baseline, **805 MB** moved through the tier live |
| fix removed | the IMA returns (count 3), first prime dies | returns to baseline exactly |
| loader | resolves in-process, delta prints and translates | `CDLL(None)`, rc=0, delta=0 |

One honesty note that section 9 leans on: the first version of this fix
"passed" its native arms while inert, because a bare `CDLL("libcudart.so")`
throws on standard pip installs (the wheels ship only versioned sonames), so
its happy path never executed. v3 resolves from the process image first and
raises on every failure, therefore a live patched engine is itself proof the
device pointer query ran and succeeded.

Follow-on hardening, tracked but not in this PR: typed two pointer separation
instead of a delta attribute, transfer records, restored state checksums, and
lifecycle/teardown coverage.

## 7. Concurrency: two serving modes, and MAX_SEQS

- **Speculation wins at low concurrency, batch mode wins past ~C10.** Long
  profile C4 on the 4090: 517 tok/s aggregate. Batch mode on the 3090 at
  `GPU_UTIL=0.972` (boots on a headless native box; WSL2 refuses that
  setting): ms per forward pass goes 23.3 to 28.2 across C1 to C16, a 21%
  spread over a 16x batch, with **zero preemptions at every level**. The spec
  profile preempted twice at N=8 on the same box, so the two modes do not
  just differ in speed: they differ in what goes wrong under pressure.
- **MAX_SEQS is part of the profile's design, not a free knob.** On the 3090,
  running the long profile at its designed MAX_SEQS=4 instead of 8 was worth
  +6.1% aggregate at N=4, purely from not over provisioning slots. And on
  huge, MAX_SEQS=2 shows 44.7% KV used at N=2, which looks like headroom;
  but 2 x 245,760 is more than the 268,169 pool, so two full length requests
  can never both fit. MAX_SEQS there is a worst case admission bound,
  deliberately overcommitted for realistic prompt sizes. The kv% gauge
  understates that risk, the concurrency banner overstates capacity
  (section 5), therefore neither gauge is safe to tune MAX_SEQS against:
  size it to the prompt lengths your workload actually sends.
- **The banner is honest at N=1, measured.** A single request on huge climbed
  to 234,158 tokens (95.3% of max_model_len) and completed with exact needle
  recall at every rung. The last 4.7% is untested because the corpus ran out,
  not the engine.
- **Prefill blocks small requests, and one flag fixes it.** A 72.6k prefill
  takes ~65 seconds, and while it runs, small requests starve (about 1 tok/s).
  `--max-num-batched-tokens 512` caps how much prefill each scheduler pass
  takes, therefore decode steps interleave and a 4k request answers in ~5s
  during the big prefill. The cost: the big prefill runs about 15% slower.
- Pin one whale prefix per box and it holds warm across turns.

## 8. Operator notes

- The 8 GB `/dev/shm` offload region **outlives the process** (confirmed on
  both boxes; one teardown kill left it resident). shm is RAM backed, so the
  stranded file holds those gigabytes until removed. Clean stale regions at
  shutdown as well as startup; on a small RAM box the stranded region is the
  next boot's OOM.
- Docker needs `--ipc host` for the offload tier: the 64 MB default shm makes
  the region's `madvise(MADV_POPULATE_WRITE)` fail with EFAULT at boot.
- `single-user/alternative.sh` hardcodes `--speculative-config` and ignores
  `SPEC=off`, so an A/B against it silently runs two spec-on arms. The tell:
  the emitted per step receipt read 2.29 on an arm that must read 1.00 with
  speculation off. An unrecognized `SPEC` should refuse, not proceed.
- `GPU_UTIL=0.972` never fits under WSL2 (compositor and WDDM overhead); it
  is fine on headless native. fp8 MAX_LEN geometry does not transfer to int4.
- `verify.sh` verifies patch application, not runtime behavior; we were bitten
  by the difference twice.

## 9. How we worked

Every arm ran with predictions registered before the result came back, and the
standard for "fixed" was the full bracket: reproduce the failure, apply the
fix, watch it pass, remove the fix, watch it fail again. Six mechanism stories
died on instruments built to kill them, and several of the deaths were our own
published claims, retracted within hours.

Three rules we paid for and now keep:

- **A pass from a branch that never executed is indistinguishable from a
  pass.** Only the execution count tells you which you have (the v1 loader
  story in section 6).
- **A retention measurement is only valid when nothing else can produce a
  fast recheck.** A tier restore, a shared prefix between test rungs, and
  pool residue from a prior rung all read as "retained" through a latency
  threshold. Corollaries: retention numbers are only valid with the offload
  tier off; test rungs must share no prefix; and a boundary rung must
  reproduce alone in a fresh boot before the number is real.
- **Over-determination is the only thing that can kill a model, and
  pre-registration is what makes the kill honest.** Two boundaries fitting
  two one parameter models is an exactly determined fit that cannot lose;
  a second boundary at a different K rejects one model outright, but only
  because both predictions were written down first.

The same discipline killed one of our own contributions before it shipped: a
startup banner patch that printed model derived warm context capacity passed
its internal consistency check, then failed its pre-registered tier-off test
(predicted 2/2, measured 0/2). It is parked on an experiment branch as
mechanism data, unmerged, because a wrong honesty line is worse than none.
Corrections throughout the docs are struck in place rather than cleaned, so
the record keeps its own history readable.
