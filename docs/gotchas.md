# Gotchas

Things that each cost us hours, in rough order of pain. Worth skimming before you debug something that looks like a vLLM bug.

[← back to the main README](../README.md)

1. **A benchmark cannot tell you the output is garbage.** The int8-activation
   path served nonsense for an hour of beautiful throughput numbers before a
   perplexity check caught it. Whatever you change, run
   `bench/quality_battery.py` (perplexity + GSM8K against the live server)
   before you believe a tok/s number.
2. **Restart onto a dirty GPU and you silently lose 25%.** vLLM profiles free
   memory once at startup. If the previous process is still releasing VRAM at
   that moment, the cache pool comes out ~40% smaller and stays that way. No
   warning, the server runs fine, throughput is just quietly bad. The systemd
   units in both mode dirs carry an `ExecStartPre` gate that waits for the GPU
   to be actually free.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is not optional.** The
   DeltaNet prefill kernels allocate transient workspace; without it the
   allocator fragments and the engine OOMs at runtime once
   `gpu-memory-utilization` goes past ~0.975.
4. **With MTP enabled, even that isn't enough — single-user mode runs
   `gpu-memory-utilization 0.93`.** The speculative decode path's DeltaNet
   workspace grows beyond what vLLM's startup memory profiling measures, and
   the engine dies mid-request on long generations at 0.95+. It survives short
   benchmarks, which is exactly how it fools you. We soak-tested 0.93 with a
   100k-token prompt plus 6k-token generations at 4 concurrent.
5. **The torch.compile cache does not know about your env vars.** Switching
   `INT8_LAYERS` between runs replays a compiled graph that expects the other
   layer set and dies with `KeyError: 'input_global_scale'`. Our patch
   registers the selection env vars with vLLM so they become part of the cache
   key; if you invent your own, `VLLM_DISABLE_COMPILE_CACHE=1`.
6. **Random-token benchmarks are meaningless for speculative decoding.** The same
   server does 35, 83 or 151 tok/s on `--dataset-name random` depending on what
   the noise turns into, because acceptance depends entirely on whether the
   drafter can guess it. Use real prompts (`--dataset-name custom`). (This is our
   own measurement, not a description of anyone else's harness — ninfer-3090's
   published cohorts use short real prompts, not random tokens.)
7. **Bigger prefill chunks make things worse.** `--max-num-batched-tokens
   8192` inflates the profiled activation peak, which shrinks the cache pool,
   which caps concurrency. 2048 wins on this card.
8. **Benchmark twice.** The first run after any restart includes JIT warmup
   and reads 30-50% low.
9. **`--language-model-only` drops the vision tower cleanly** (no weights
   loaded), and it is the default in both start scripts. `VISION=1` keeps the
   tower for a client that sends images. The tower is **0.858 GiB**, not the
   2.7 GB this entry used to give: `model.visual.*` sums to 0.858 GiB of BF16 in both
   `Qwen3.8-27B-W4A16-AutoRound` and the `-fast` variant, and a runtime A/B
   agrees — model loading reads 15.13 GiB against 14.26 with
   `--language-model-only`, same server, same config, with non-weight overhead
   0.42 against 0.41 GiB either way. Quantization is not the explanation: the
   tower is BF16 in both dirs. Where 2.7 GB came from I could not work out.
   The KV pool came out within 0.4% across that pair (183,673 against 184,438
   tokens at 150k/fp8) — but the profiled activation peak differed by 0.87 GiB
   between those two starts, which is the same run-to-run swing the V2 runner
   shows, so read the pool difference as noise rather than as a measurement of
   what the tower costs.
10. **`prompt_logprobs` on long prompts OOMs the engine at 0.972 utilization**
    (a 300-token prompt needs ~300 MB of fp32 logits and there is no headroom).
    Run quality checks at 0.93.
11. **Don't chase the DeltaNet kernels.** `bench/tune_gdn.py` microbenchmarks
    the decode kernel across block/warp configs: it already runs at ~85% of the
    3090's memory bandwidth and every variant lands within 3%. The state dtype
    (point 3 above) is the lever, not the kernel.

12. **The draft vocabulary is the single-user ceiling.** A draft head can only
    propose tokens in its id list; a miss is a certain rejection that also ends
    the chain. Count the list over the model's *own* outputs (`drafter/gen_data.py`,
    then the frequency step in `prepare/build_draft_vocab.py`), not over web text —
    92% vs 97.5% coverage was the difference between 98 and 109 tok/s greedy.
    Coverage saturates around 40k rows; the model only ever emits ~54k distinct
    tokens.
13. **FlashAttention-2 does not split KV for multi-query decode.** With k
    speculative tokens the verify step has k+1 queries per request and FA2's
    varlen path then runs one thread block per (request, head): 24 blocks on 82
    SMs, 57 µs per layer at 1.5k context and 1.3 ms at 16k. vLLM's Triton
    unified attention has the same restriction (`max_seqlen_q > 1` → 2-D
    kernel). `patches/spec-decode-attn.patch` (`VLLM_SPEC_DECODE_ATTN=1`, bf16
    KV only) is a 180-line Triton fix. Watch its query cap: the kernel used to
    handle at most `BLOCK_M / (heads per kv head)` = 10 query tokens and fall
    back silently past that, which doubled the step at 25k context the moment
    the verify block grew to 16. It now tiles the query rows instead.
14. **Greedy is not deterministic across drafter configs.** The target rounds
    differently when it verifies 5 tokens vs 1, so a different drafter changes
    the generated text at near-ties and the 8-prompt acceptance numbers move
    ±3%. Repeat before trusting a small difference; `drafter/README.md` has an
    offline chain simulator that removes the noise.
15. **A stale torch.compile cache bites anything that changes tensor shapes
    behind vLLM's back.** The compiled graph bakes in e.g. the Marlin workspace
    size; a new env knob that changes it must be registered in `envs.py`
    (`patches/speed-knobs-envs.patch`) or you get `assert_size_stride ...
    expected size 328==82` from a cached artifact.
16. **The very first start gets a smaller KV pool.** vLLM sizes the pool from
    the peak memory of a profiling forward pass, and on a cold torch.compile
    cache that pass also runs inductor's autotuning: batch mode profiles a
    1.96 GiB activation peak instead of 1.09 GiB and comes up with 196k KV
    tokens instead of 224k (`Maximum concurrency ... 1.31x` in the log instead
    of 1.49x). Restart once after the cache is warm (venv: `~/.cache/vllm`,
    Docker: the `qwen-cache` volume) and the pool is back to the README numbers.
    (The WSL2 notes above pin it the other way round — record the cold-start
    `--kv-cache-memory` recommendation and pass it via `EXTRA_ARGS` — if you
    prefer the extra transient headroom to the extra KV pages.)
17. **vLLM picks the speculative method from the model *path*.** `"dflash" in
    model_path` switches `method` to dflash — for the *target* too, since MTP
    uses the target path as its draft model. A checkout under a directory with
    "dflash" in its name turns `SPEC=mtp` into a crash in `EAGLEConfig`
    (`'Qwen3_5Config' object has no attribute 'vocab_size'`). Name your
    directories accordingly.
18. **The V2 model runner (`SPEC=dflash2`) does not count its CUDA graphs when
    sizing the KV pool** (~1.2 GiB on top of whatever `--gpu-memory-utilization`
    you asked for), the hybrid allocator sizes KV groups by the smallest layer
    bucket, and the profiled activation peak varies by ~1 GiB between starts of
    the *same* config — three ways to get a server that either wastes a quarter
    of its pool or dies mid-request. `patches/hybrid-kv-groups-v2-cudagraph.patch`
    fixes the first two; for the third, pin the pool in bytes
    (`--kv-cache-memory`, what `KV_MEM` does) instead of tuning utilization. That
    runner also answers `thinking_token_budget` with 400, and the first request
    after a cold start JIT-compiles four Triton kernels (~5 s once; cached in
    `~/.triton`).
19. **`INT8_LAYERS=.` needs `GPU_UTIL=0.95`.** Quantizing the activations of every linear
    layer (rather than just the MLP) is worth ~11% throughput — 1,042 vs 942 tok/s at 64
    concurrent — but the extra per-layer scratch no longer fits batch mode's 0.972: the
    engine dies with `torch.OutOfMemoryError` inside `chunk_fwd_o` once ~17 requests are
    resident, which reads as every request returning 500 while `/health` still answers.
20. **A Triton kernel's scratch buffers may not grow after CUDA graph capture.** The
    split-KV verify attention sizes its partial buffers from the longest query block it has
    been asked for. Once the block got longer than the drafter's — and once a small prefill
    chunk could land on the same kernel — that "longest so far" changed mid-run, the buffers
    were reallocated, and the captured decode graph went on reading the freed ones:
    `CUDA error: an illegal memory access was encountered`, a few hundred tokens into the
    first request. `VLLM_SPEC_DECODE_ATTN_QMAX` (set by `single-user/start_qwen.sh` from
    `DFLASH_TOKENS`) fixes the size at startup instead.
21. **Async scheduling pins the number of speculative tokens.** vLLM only feeds draft token
    ids — and therefore the *count* the worker wants verified — back to the scheduler on the
    synchronous path (`EngineCore.post_step`). With async scheduling on, every decode step is
    padded to `num_speculative_tokens` and a worker asking for fewer is ignored, silently.
    Adaptive block length (`LOOKUP=1` with `DFLASH_TOKENS > 7`) needs `ASYNC_SCHED=0`; at
    batch 1 that costs under 1%.
22. **`--async-scheduling` is already the default in 0.27.1.** The flag exists and passing it
    changes nothing; `--no-async-scheduling` is what turns it off. Two hours of "the adaptive
    block isn't working" was this.
23. **A longer verify block costs KV pool per request slot, not per token.**
    `--mamba-cache-mode align` reserves `2 + num_speculative_blocks` recurrent-state pages
    per slot, so `DFLASH_TOKENS=31` with 8 slots wants 5.3 GiB before a single token of
    context and refuses to start. Single-user mode drops to 4 slots when the block is long,
    which is what makes the long block affordable at all.
24. **The DFlash draft pass is a captured CUDA graph, so its Python runs once.**
    `DFlashSpeculator._generate_draft` — everything the speculator does per step, including
    the lookup — is replayed from a graph. The Triton kernels inside it do run every step and
    do read live buffers, so the lookup itself works; but host-side Python in there executes
    at *capture* time only. A counter, a pinned copy of a flag, a decision computed there is
    frozen at whatever the warm-up produced, silently. Anything the host must see per step
    belongs in a method the model runner calls per step (`next_num_draft_tokens`), reading
    device tensors the replayed kernels wrote. Three separate "the trigger doesn't fire"
    debugging rounds were this.
25. **`torch.cuda.is_current_stream_capturing()` is not a usable guard on this path.** It
    reads True inside the captured draft pass — which is correct, and exactly why a guard
    written as `if not is_current_stream_capturing():` silently disables the code it guards
    for the entire run, not just during warm-up.
26. **rsync preserves mtimes, and Python trusts mtimes.** Copying a source file into
    `site-packages` with `rsync -a` can leave the `.pyc` newer than the `.py`, in which case
    the interpreter keeps running the old bytecode and every measurement lands on the
    previous revision. Delete `__pycache__` after installing patched files.
27. **A shorter draft block than `num_speculative_tokens` loses the decode CUDA graphs.**
    The V2 runner captures uniform-decode graphs at `decode_query_len = num_speculative_tokens
    + 1` and dispatch requires an exact match, so scheduling the drafter's 8-token block on a
    16-token server matches nothing and the step runs piecewise: 27.9 ms against 25.9 ms for
    the same work, on every short step. `cudagraph_utils.py` already knows how to capture
    several decode lengths (it does it for dynamic speculative decoding); the lookup patch
    adds the drafter's block to that list. Costs 1.8 GiB of graphs instead of 1.45.
28. **A verify block costs step time in steps, not smoothly, and the two stairs are at 16
    and 21 query tokens.** Measured on a copy at 25k context: 39.5 ms per step at 16 query
    tokens (`DFLASH_TOKENS=15`), 47.8 at 19, 47.2 at 21 — a jump between 16 and 19 and then
    flat. The first stair is the target's W4A16 GEMMs: GPTQ-Marlin tiles the M dimension in
    16 rows (`m_block_size = 16 * thread_m_blocks`, `thread_m_blocks = div_ceil(prob_m,
    16)`), so a 17th query token buys a second M block in all 64 layers and the tokens up to
    32 are then free. The second is the verify attention: `SpecDecodeAttention._plan`
    (patches/spec-decode-attn.patch) puts `q_len * G` rows in a 128-row tile, so with this
    model's `G = 24/4 = 6` one tile holds `128 // 6 = 21` query tokens and a 22nd re-reads
    the request's whole KV segment (250/583/1132 us per layer at 8/16/32).
    So there are exactly two sensible block lengths — 16 query tokens, the last one on the
    bottom stair, and 21, the most tokens obtainable for the price of the second. 31 pays
    both stairs and was never worth measuring; two attempts to start it died on memory
    first.
29. **A verify block that outgrows its CUDA-graph reservation OOMs at run time, not at
    startup.** `--kv-cache-memory` pins the pool, so `VLLM_V2_CUDAGRAPH_MEM_MIB` no longer
    sizes it — it only reserves headroom, and if it under-reserves, the server starts, logs a
    healthy pool, and then dies on the first prefill with 50 MiB left. Graph memory grows
    with the block: measured 1.82 GiB at `DFLASH_TOKENS=15`, 2.12 at 18, 2.27 at 20 (the
    capture list length barely matters — 2.21 GiB at 20 with `CG` cut from 63 to 42). Budget
    a request as `64 KiB * context + 102 MiB * (DFLASH_TOKENS + 2)`, the second term being
    the aligned recurrent-state pages, and take the extra graph memory out of the pool.
30. **The draft model is not redundant during a copy, even when the lookup overwrites every
    token it proposed.** It looks like free money: on a step the lookup controller selected,
    a qualifying match is long enough to take the head of the block too, so all seven of the
    drafter's tokens are replaced before anything is verified — skip its forward and save
    ~3 ms of a 39 ms step. Measured, that trade loses: 15.21 tokens per step becomes 13.79
    for a 5% cheaper step, a net 6% down. The drafter is covering the positions *past the
    end of the match*, which is exactly where a copy lands when the text it is reproducing
    diverges. Restricting the skip to steps where the match reaches the end of the block
    recovers the acceptance but only two runs in three — the flag it keys on is one step
    stale, and a stricter condition is more sensitive to that. Both variants are gone; this
    entry is here so the idea does not look untried.
31. **Any controller state that outlives one step has to be per-request, or batch > 1 stops
    being reproducible.** The lookup's block-length decision is batch-wide by design — a long
    block costs step time on every request in the batch — and taking it from the current
    step's flags is fine, because those are a function of the requests present. Holding it
    across steps is not: `VLLM_DFLASH2_LOOKUP_STICKY` keeps the long block on through steps
    where the flags say no, so with several requests in flight the block length a copying
    request gets depends on when the *others* arrived, and the block is one chunk through the
    recurrent layers, so that changes its greedy text. `bench/labd_soak.py` caught a verbatim
    copy coming out differently in two rounds of an identical four-way batch, and OK in three
    of three with the hold off. It is now applied only with one request in flight. The proper
    fix is per-request draft counts, which `get_uniform_token_count` in
    `gpu/cudagraph_utils.py` will not dispatch a graph for — a ragged batch runs piecewise
    and costs 8%, more than the hold is worth.
32. **Halving the KV element size can *cost* memory on a hybrid model with a draft model.**
    `unify_kv_cache_spec_page_size` equalizes page sizes by scaling a layer's block size up by
    the integer ratio `max_page / own_page`, and pads the *page* instead when that ratio is not
    an integer. Sliding-window layers are born at the backend's smallest kernel block — 16 —
    precisely because the code picking it assumes unify will scale it up
    (`_largest_kernel_block_within` in `model_executor/layers/attention/attention.py`: "the
    smallest block is fine — `unify` scales it up by an integer ratio"). When the ratio is not
    an integer that assumption fails silently and every block of that layer pays a whole
    primary page. Divisibility here holds at bf16 only by coincidence — the target's 4 KV heads
    × 256 and the DFlash2 drafter's 8 × 128 both come to 4096 B per token per layer — and
    `int8_per_token_head` breaks it by adding one fp32 scale *per head* (2080 vs 2112 B/token;
    2112 = 2⁶·3·11 shares no factor with the primary page). The drafter's 5 layers then took
    `cdiv(2047 + 4096, 16) + 1 = 385` blocks of 1.71 MiB at 1.88% utilisation — a constant
    5.155 GiB, 75.6% of the per-request budget. Measured: int8 needed **6.82 GiB to serve
    32,768 tokens** where bf16 serves 69,758 in 5.2 GiB, i.e. 2.4× worse from halving the
    dtype. `patches/hybrid-sw-block-promote.patch` rounds such a layer's block *up* instead
    (16 → 864), which turns that into 138,696 tokens. The tell in a log is an "estimated
    maximum model length" that is a small multiple of 16.
33. **The aligned recurrent-state pages scale with the verify block, not with the slot count.**
    Gotcha 23 says "per request slot"; that is wrong. Measured by asking for an impossible
    `max_model_len` and fitting the two numbers vLLM prints: the fixed term is 0.88 GiB at
    `DFLASH_TOKENS=7` and 1.66 GiB at 15 — the ratio 0.53 is exactly 9/17, i.e. `(k+2)` — while
    `MAX_SEQS` 1 against 8 moves it by about **8 MiB in total**. So dropping to one slot for a
    genuinely single-user server buys no context at all, and `MAX_SEQS=4` at a long block is
    about CUDA graph memory, not state pages.
34. **Asking for an impossible `max_model_len` is the cheapest way to read the memory model.**
    vLLM prints "X GiB KV cache is needed ... available Y GiB ... estimated maximum model
    length is Z" and dies in ~90 s, before torch.compile finishes and long before graph
    capture. Two such points give slope and intercept for `needed(context)`, and the slope
    comes out at exactly `16 × 4 × 256 × 2 × 2 = 65,536` B/token for bf16 — so the fit can be
    checked against arithmetic rather than trusted. Beware that `estimate_max_model_len` is a
    binary search over `max_memory_usage_bytes`, which rounds up to whole blocks, so the
    estimate is quantised by the block size: at an 864-token block the granularity is coarse
    and a two-point inversion at small lengths is unreliable.
35. **`KV_MEM` assumes the card is headless, and the failure lands long after
    startup looks fine.** The single-user pool is pinned in bytes rather than sized
    from `GPU_UTIL` (gotcha 33 and the comment in `single-user/start_qwen.sh` say
    why), and 5.2 GiB is what fits when nothing else is on the GPU. With a desktop
    session on the same card — Xorg plus a compositor plus a browser is easily
    ~1.3 GiB — the server still starts, still captures its graphs, still reports a
    pool, and then dies later on a real request when the spec-decode `part_o`
    buffer cannot get its ~1.5 GiB (`spec_decode_attn.py`). Nothing at startup
    warns you. On a card you also render on, drop `KV_MEM` by at least what the
    desktop is holding (`nvidia-smi` before you start the server): `KV_MEM=4000000000`
    was enough for the reporter of
    [#12](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/12). Setting `KV_MEM=`
    empty falls back to `GPU_UTIL`, which profiles the actual free memory instead.
36. **A model dir with no `tokenizer.json` is not an error to transformers — it is an
    empty vocabulary, and vLLM reports it as a reasoning-parser problem.**
    `AutoTokenizer.from_pretrained` on a dir that has `config.json` but no tokenizer
    files returns a `Qwen2Tokenizer` with `vocab_size == 1` that encodes *everything*
    to `[]` — `tok.encode("hello world")` is `[]`, not an exception. Nothing complains
    until `VllmConfig.__post_init__` asks the qwen3 reasoning parser for `<think>`,
    gets `[]` back, and raises

    ```
    ReasoningConfig: failed to tokenize reasoning strings:
    reasoning_start_str='', reasoning_end_str=''.
    ```

    which names neither the tokenizer nor the directory, and prints the strings as
    empty because they are the *unset* config fields, not the ones the parser supplied.
    Reported as [#15](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/15), where it
    looked like a `SPEC=dflash2` bug: it reproduces with no speculative config at all,
    and the reason only the single-user modes failed is that they serve
    `models/Qwen3.8-27B-W4A16-AutoRound-fast` while batch mode serves the base dir.
    `verify.sh` now encodes `<think>` against every dir we pass to `--model` instead of
    only checking that the dir exists, and `docker/prepare.sh` counts `tokenizer.json`
    as part of a complete download.
37. **Bug B needs a prefix-cache HIT, and then fires at one prompt length in every
    128. It is not dflash2-only.**
    Under `CTX=huge` with a CAPTURED (FULL) verify step, a request that hits the
    prefix cache and whose prompt length lands on one particular residue mod 128
    collapses: `SPEC=dflash2 DFLASH_TOKENS=7` gives 1.97 tok/step and degenerate
    repetition (`4/3595` characters verbatim, one 40-char block ×79), `SPEC=mtp`
    stops dead and returns `""` or `"#"` with `finish_reason=stop`. Every other
    residue is 794/794 verbatim. Deterministic — repeats are bit-identical.

    Two conditions, and it took two people to see both. The **hit** is necessary:
    a fresh server, one request, no warm-up, never collapses at any length
    ([#13](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/13), mjungnickel18) —
    which is also why `PREFIX_CACHE=0` always looked clean. The **residue** decides
    whether a hit corrupts, and it is a clean function of the draft count:

    | config | k | verify block L=k+1 | attention block | broken R | free slots 128-R |
    |---|---|---|---|---|---|
    | `dflash2`, `DFLASH_TOKENS=7` | 7 | 8 | 2176 (=17x128) | 124 | 4 |
    | `dflash2`, `DFLASH_TOKENS=5` | 5 | 6 | **2176** | **122** | 6 |
    | `dflash2`, `DFLASH_TOKENS=3` | 3 | 4 | 2048 (=16x128) | 120 | 8 |
    | `mtp`, `DRAFT_TOKENS=3` | 3 | 4 | 2048 | 120 | 8 |

    **R = 117 + k**, equivalently the final 128-token tile has exactly `11 - k`
    free slots, equivalently `L + free = 12` in every configuration measured.
    Fitted on three values of k across two speculators, so treat the constant as a
    fit rather than a derivation — but note what it implies: the step reserves or
    touches a **fixed 12 slots regardless of the verify block length**, which is
    the most specific lead this bug has produced.

    The `DFLASH_TOKENS=5` row is the one that matters for method. Same attention
    block as `DFLASH_TOKENS=7` (2176), different broken residue (122 against 124),
    which rules out the attention block size. An earlier version of this entry
    claimed R tracked the verify block on three points where the two co-varied;
    that was retracted as unevidenced, and then confirmed by running the
    configuration that separates them.

    Confirmed periodic in every case: 24,956 / 25,084 / 25,212 / 25,340 at k=7,
    25,082 / 25,210 / 25,338 at k=5, 25,080 / 25,208 / 25,336 at k=3. Hold the document
    byte-identical and pad the *instruction* by one token and a broken length goes
    clean, so it is the token count rather than the corpus. What is established: `mtp` and
    `dflash2` at the same draft count break at the same residue, so the drafter is
    not implicated and the shared multi-query verify against a partially-hit
    prefix is.

    Mitigation: `CTX=huge` forces `cudagraph_mode=PIECEWISE` for **every**
    speculator, which is clean at every residue. It costs nothing measurable —
    `SPEC=mtp` over 8k/16k/32k/50k is 87.8/86.1/70.4/63.5 tok/s captured against
    93.5/83.8/70.3/59.6 piecewise. This repo previously scoped that workaround to
    `dflash2` on the theory that MTP's short verify step captures correctly; it
    does not, and `SPEC=mtp CTX=huge` shipped with the bug.

    A third trap, learned the hard way on `DFLASH_TOKENS=15`: `bench/bugb_sweep.py`
    used to report the RAW prompt length, not the chat-templated one the engine
    actually sees (+12 tokens for the Qwen3 wrapper). That offset is why the rule
    first read as `R = 117 + k` and then as a mysterious constant 12; both were the
    same relation seen through a harness bug. It also made a k=15 sweep look
    structureless until the offset was applied, at which point the lowest-acceptance
    row sat exactly on `== L`. The script now templates before counting. And read
    `repeats` next to `verbatim`: the verbatim column is a longest-prefix match, so a
    single wrong character at offset 38 reports `38/791` no matter how good the rest
    is -- only `repeats` tells you whether it actually collapsed.

    Two traps for anyone measuring this. Sweep prompt length in steps of **1
    token** — at a coarse grid one broken sample below and one above reads as a
    cliff, which is how it was first diagnosed. And send each length to a **fresh
    server**, or request N inherits request N-1's blocks and you measure history
    instead of length; `bench/labd_bench.py` sends two warm-ups on `doc[:4000]`,
    which arms the trigger for everything after it. `bench/bugb_sweep.py` prints
    the `mod 128` column for this.
