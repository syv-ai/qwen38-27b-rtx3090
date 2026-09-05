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

   On `SPEC=dflash2` that 0.858 GiB is not optional headroom, it is the difference
   between booting and not. Measured here on a 3090 at 250 W, `SPEC=dflash2 VISION=1
   VISION_OFFLOAD=0`, `CTX=fast`: the engine dies in graph capture with
   `torch.OutOfMemoryError: Tried to allocate 960.00 MiB ... 787.50 MiB is free`
   (`spec_decode_attn.py:184`, the split-KV verify `part_o` buffer). The pool there is
   pinned by bytes (`KV_MEM`), so the tower cannot come out of the KV cache — it comes
   out of the ~1.1 GiB transient margin, and it is 0.85 of it. `VISION_OFFLOAD=1` (the
   default) makes the same config come up at the full 69,758-token pool and read images.
   The `SPEC=mtp` path has no such problem: it boots either way, and there the pool is
   profiling-sized, so what the tower costs is buried in the ±0.87 GiB profiling swing
   above (79,271 against 80,055 tokens across the pair — 1%, i.e. noise).

   What `VISION_OFFLOAD=1` does: the tower's weights live in pinned host RAM and each
   module is copied to the GPU for the duration of its own forward
   (`patches/vision-tower-cpu-offload.patch`). Isolated-tower measurement, RTX 3090 at
   PCIe 4.0 x16, one 8192-patch image, median of 10 forwards: resident weights 891.3 ->
   9.0 MiB, peak allocation 1160.5 -> 308.2 MiB, encode 296 -> 333 ms, output bit-exact
   against the resident tower. Note which offload path that is: vLLM's UVA *zero-copy*
   mode saves the same memory and costs 3327 ms, because the GEMMs then re-read operand
   tiles over PCIe inside the inner loop. The patch forces the bulk-copy path and does
   not touch what `--cpu-offload-gb` does elsewhere -- which could not reach the tower
   anyway, since the offloader is only installed in `make_layers()`.

   One precision from re-verifying the premise on a headless 3090, same 250 W:
   with nothing else on the card, `VISION_OFFLOAD=0` *does* boot -- and lands at
   440 MiB free after boot, inside gotcha 39's kill zone (396 MiB free died on a
   concurrent burst where 436 survived). The hard no-boot above needs something
   else holding a share of the card -- the measuring box also ran a desktop
   compositor and a browser, which is the normal state of a 3090 in a
   workstation. Same conclusion from both geometries: a resident tower puts the
   engine at the headroom cliff, the offload puts it at the full margin, and
   that is why the default is on. On the current tree the pool prints 68,605
   tokens (`KV_MEM` has moved since the 69,758 above was measured), identical
   between `VISION=0` and `VISION=1`, and the image round-trip reads a marker
   that exists only in the pixels either way.
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
22. **`--async-scheduling` is already the default in 0.28.0.** The flag exists and passing it
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

    That is about the SIZE of the pool. It says nothing about how much of the pool a
    *running* request takes, and there the per-request model is right — it is the same
    page, and the two arrive at it independently (0.88 GiB fitted here, ~0.82 GiB from
    the live occupancy below). Measured live at `CTX=fast` (`bench/conc_ladder.py`, and the ramp in the
    issue-25 notes): one resident `dflash2` request with an empty context occupies
    **15.8%** of the 69,758-token pool, so six or seven fit and the next one is
    **preempted**; with 4k-token prompts it is 19.8% and five fit, with 16k-token
    prompts two. One MTP request takes 8.2% of its
    86,727, so eight fit. Both numbers are the k+1 recurrent-state slots, which is why
    they are in the ratio 8:5. The two facts together are the whole of the concurrency
    story for this mode: extra seats do not cost you pool, and they do not buy you
    residents either.
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
    residue is 794/794 verbatim.

    **The location is deterministic; the damage is not.** Repeats are bit-identical
    on one server, but the same `mtp` residue has now produced three different
    outputs on three geometries: an empty answer, a one-character answer, and — on a
    box running `MAX_LEN=240000` with a tool parser attached — 400 tokens of fluent
    Danish that open with a malformed `<think>` under `enable_thinking=false` and
    invent a translation task, `2/1146` verbatim at 3.38 tok/step
    ([#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25), mjungnickel18).
    So do not test for a symptom. The only property all three share is that the copy
    did not come back, which is what `bench/verbatim.py` scores and what both sweeps
    now judge on.

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

    Mitigation, as it stands at HEAD: `CTX=huge` forces `cudagraph_mode=PIECEWISE`
    for `SPEC=mtp`, which is clean at every residue and costs nothing measurable —
    `SPEC=mtp` over 8k/16k/32k/50k is 87.8/86.1/70.4/63.5 tok/s captured against
    93.5/83.8/70.3/59.6 piecewise. This repo previously scoped that workaround to
    `dflash2` on the theory that MTP's short verify step captures correctly; it
    does not, and `SPEC=mtp CTX=huge` shipped with the bug.

    `dflash2` has since got FULL capture back (`a75ee4b` fixed its residue, and
    `b356e31` swept **all 128** residues under FULL with 0 broken), so the two
    speculators are no longer on the same default. `mtp` keeps PIECEWISE as a
    correctness constraint until residue 4 comes back verbatim under a full sweep,
    not until a particular symptom stops appearing.

    A third trap, learned the hard way on `DFLASH_TOKENS=15`: `bench/bugb_sweep.py`
    used to report the RAW prompt length, not the chat-templated one the engine
    actually sees (+12 tokens for the Qwen3 wrapper). That offset is why the rule
    first read as `R = 117 + k` and then as a mysterious constant 12; both were the
    same relation seen through a harness bug. It also made a k=15 sweep look
    structureless until the offset was applied, at which point the lowest-acceptance
    row sat exactly on `== L`. The script now templates before counting.

    Do not judge a row by its failure signature, and that includes `repeats`. An
    earlier version of this entry said "only `repeats` tells you whether it actually
    collapsed"; two of the three shapes above repeat nothing, and a rule that
    demanded repetition is what filed the `mtp` break as "diverged, probably fine"
    through several full sweeps. Both sweeps now score **coverage** — the fraction of
    the answer's 40-character windows that occur in the source — against the median
    of the other lengths in the same run (`bench/verbatim.py`, which self-tests
    against all three shapes: `venv/bin/python bench/verbatim.py`). Coverage rather
    than the old longest-prefix column because a prefix match reports `38/791` for a
    single wrong character at offset 38 no matter how good the rest is; the prefix
    and repeat counts are still printed, but nothing is decided on them.

    Two traps for anyone measuring this. Sweep prompt length in steps of **1
    token** — at a coarse grid one broken sample below and one above reads as a
    cliff, which is how it was first diagnosed. And send each length to a **fresh
    server**, or request N inherits request N-1's blocks and you measure history
    instead of length; `bench/labd_bench.py` sends two warm-ups on `doc[:4000]`,
    which arms the trigger for everything after it. `bench/bugb_sweep.py` prints
    the `mod 128` column for this.

    And do not sample residues. With one broken length in 128, five distinct samples
    miss it `C(127,5)/C(128,5)` = 123/128 = **96%** of the time — this repo once
    wrote 82% there, which is the figure for six broken residues, and hung a
    "5 of 5 clean" claim on it. `bench/residue_sweep.py` walks all 128 by stepping
    the pad one token at a time, which covers each residue exactly once.
38. **The decode-graph budget is sized for 64 query tokens, and `MAX_SEQS` multiplies
    into it.** `CG = MAX_SEQS x (k+1)` is what the V2 runner captures, and
    `VLLM_V2_CUDAGRAPH_MEM_MIB` is reserved for what the shipped defaults produce —
    8x8 at `DFLASH_TOKENS=7`, 4x16 at 15, i.e. 64 either way. Ask for
    `DFLASH_TOKENS=15 MAX_SEQS=8` and it becomes 128: the server boots, captures its
    graphs, answers `/health`, and then dies on the first concurrent batch with
    `torch.OutOfMemoryError` inside the engine — `EngineDeadError`, every request 500,
    `/health` still 200. Same shape as gotcha 18 and as
    [#18](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/18): a memory bill that
    the startup profile does not see. `single-user/start_qwen.sh` now caps the derived
    `CG` at 64, which leaves every shipped default untouched and makes the oversized
    batches run piecewise instead of not at all. Set `CG` explicitly to override, and
    raise `VLLM_V2_CUDAGRAPH_MEM_MIB` with it.

39. **The engine needs non-KV headroom for its first real batch, `MAX_SEQS` and
    `KV_MEM` are two doors into the same shortfall — and on WSL2 the failure is
    silent.** First seen as a seat-count death: `MAX_SEQS > 12` at `CTX=huge` kills
    the engine and the graphs are innocent — same visible failure as gotcha 38
    (boots, captures, `/health` 200, dies on the first prompt with
    `torch.OutOfMemoryError`), different bill. `CG` is pinned at its 64 cap in every
    one of the runs below, so the memory is going to allocations that scale with
    `max_num_seqs` itself, not with the captured batch. Free VRAM after boot on one
    24 GiB 3090, `SPEC=dflash2 CTX=huge PREFIX_CACHE=1` k=7, then a single ~3.7k-token
    prompt:

    | `MAX_SEQS` | free after boot | one ~3.7k prompt |
    |---|---|---|
    | 8 | 596 MiB | ok |
    | 10 | 456 MiB | ok |
    | 12 | 416 MiB | ok |
    | **16** | **356 MiB** | **dead** |

    The allocator says it plainly: `expandable_segments: memory mapping failed with OOM
    on device 0 while trying to map 20971520 bytes (free: 20578304, total:
    25272516608)` — 20 MB wanted against 20 MB left, on a card with 24 GB. It needs no
    concurrency at all: `num_running_reqs=1`, `step_counter=0`, `kv_cache_usage=0.18`.
    Reproduced twice with byte-identical counters.

    The seat count is only one door into that shortfall.
    [@mjungnickel18 named the real subject](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25#issuecomment-5392694387)
    — *how much non-KV headroom does the engine need*, with `MAX_SEQS` and `KV_MEM` as
    two doors into the same room — after a `KV_MEM` pin on his box produced a failure
    this table does not contain (below). The same room walked through the `KV_MEM`
    door, seats pinned at 8, salted prompts, best of 3, prefill tok/s from TTFT:

    | `KV_MEM` | free after boot | 4k / 16k prefill | 8×16k concurrent | outcome |
    |---|---|---|---|---|
    | 5,261,334,938 (stock) | 576 MiB | 1,156 / 1,107 | ok — free bottoms at 110 MiB | ok |
    | 5,414,427,034 | 436 MiB | 1,159 / 1,100 | ok — free bottoms at **8 MiB** | ok |
    | 5,466,855,834 | **396 MiB** | 1,155 / 1,099 — full speed | **dead in 34 s, every request 500** | dead, twice |

    Same fingerprint at the bottom: the identical four failed 20,971,520-byte mappings
    with byte-identical free counters across both repeats, ending in
    `torch.OutOfMemoryError: Tried to allocate 24.00 MiB ... 37.62 MiB is free`. Two
    refinements the second ladder forces. The transient working set is *elastic*: it
    takes ~380 MiB when the room exists (watch `memory.free` during a prefill: 576 →
    194 MiB at stock) but squeezes without measurable cost — at 436 MiB free the 8-way
    cell ran at full throughput with 8 MiB left. What is rigid is the first real
    batch's allocation bill, sized by `max_num_seqs` and by how many requests actually
    run: 16 seats died on a single prompt, while 8 seats at 396 MiB prefilled 16k
    single-stream at full speed and died the moment eight ran at once. No
    configuration anywhere on either ladder was ever merely *slow*.

    That last sentence is the platform note, and it is the part that cost a week of
    cross-box debugging in [#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25):
    on bare-metal Linux this failure has exactly two states, full speed or a loud named
    `torch.OutOfMemoryError`. Under WSL2 the WDDM driver backs the failed mapping with
    host memory instead, so the same exhaustion produces **no error at all** — just
    prefill at a fifth of the rate (232 tok/s at 4k against ~1,000 healthy, measured by
    @mjungnickel18 under a `KV_MEM` pin that left ~630 MiB free). A WSL user who raises
    `KV_MEM` gets the context they asked for, no warning, and 5–10× the TTFT, with
    nothing in the logs and no `nvidia-smi` number that flags it. The two boxes in
    [#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25) make it concrete: the
    same pin (`KV_MEM=6871947673` at `CTX=fast`, `MAX_SEQS=2`; the boxes produce
    byte-identical pool geometry, 81,368 tokens at the fixed sibling pin) read
    ~630 MiB free after boot on WSL and served — slowly — for four days, while on bare
    metal it boots with 98 MiB free and the first prompt kills the engine. WSL's free-after-boot overstates the Linux number by roughly whatever WDDM
    is host-backing, so a headroom rule of thumb tuned on one platform does not
    transfer to the other in either direction. The detector is the one that found it: a
    prompt-length ladder against a known-good rate. On WSL, ladder any `KV_MEM` above
    stock before trusting it; the launcher now prints a warning when the pin exceeds
    the profile default. A cheaper live check, from a second WSL2 box in
    [#61](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/61): `nvidia-smi dmon`
    while it generates. Healthy decode on a 24 GB card is high SM occupancy *and*
    high power draw; host-backed memory shows as **SM near 100% at only 100-200 W**,
    because the SMs are stalled on PCIe rather than doing work. That reporter's rule
    of thumb — keep ~2 GB of VRAM free by lowering `GPU_UTIL`, and do not chase the
    context back with `MAX_LEN`, since the pool is sized by `GPU_UTIL` — matches the
    two boxes above.

    The launcher warns above 12 seats rather than clamping, because unlike `CG` this is
    a VRAM budget rather than a shape: a card bigger than 24 GiB has room where this
    one does not. Seats and pool trade against each other — if you want seats, buy them
    with a lower `KV_MEM`; if you want context, the ladder above is the price list, and
    on this card the floor at 8 seats sits between 396 and 436 MiB of free headroom.
    The shipped `CTX=huge` default is `MAX_SEQS=2`, so nothing here is reachable
    without an override.

    Worth reading next to the concurrency section of the README: seats above the
    residency were already useless (they queue, then preempt). Past 12 at `CTX=huge`
    they stop being useless and become fatal.

40. **Tool calling / structured output under a speculator killed requests at the
    grammar's end** ([#31](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/31),
    fixed by `patches/xgrammar-spec-terminated.patch`). A speculative verify window
    can legally accept tokens past the point where the xgrammar matcher terminates —
    the newline after a closing `</tool_call>` tag, the stop token itself, anything
    after it under `ignore_eos`. The v0.28.0 base treats both arrivals as failure, the
    scheduler logs `Unexpected: grammar rejected tokens ... Terminating request`,
    and the client gets an HTTP error for a request whose output was completely
    valid. The longer the verify block, the more reliably the window covers the
    tokens around the stop, which is why `DFLASH_TOKENS=15` + `--tool-call-parser`
    surfaced it first. Reproduced on the shipped config with a `json_schema` +
    `ignore_eos` request — `grammar rejected tokens [16, 22, 198, 92, 248046, 198]`,
    where 92 is the brace that completes the JSON, 248046 the stop token that
    terminates the matcher, and the trailing newline killed the request. The patch
    backports upstream's current semantics: tokens after termination are ignored,
    real mid-grammar rejections still fail loudly.

    Two log signatures to keep apart, because they look alike. The fatal one is the
    `grammar rejected tokens` line above — gone with the patch. The non-fatal one is
    a burst of `Failed to advance FSM for request ... Please file an issue.` with
    **no** `Terminating request` after it: that is the bitmask builder advancing
    draft tokens past a reasoning end that landed mid-window, a rejection the code
    explicitly tolerates. It is noise, the request completes normally, and it
    predates (and survives) this fix.

41. **sm80 (GA100) Marlin repack can Xid-31 the whole card under memory
    pressure — and the kernel in the traceback is innocent.** Community
    finding, [@ahnguyen17 in #27](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/27#issuecomment-5397500895),
    on a CMP 170HX 40 GB: with ~27 GB resident, `gptq_marlin_repack`'s GB-scale
    int64 intermediates (k×n int64 ≈ 1.4 GB per 27B layer, several live at
    once) churn sm80 VMM mappings until an unrelated, trivially correct
    elementwise kernel takes an async write fault — the faulting frame drifts
    between runs, the Xid 31 wedges the card until reboot, and
    `compute-sanitizer` is clean on sm86 with identical inputs. Their
    workaround, serving in production since: compute the repack on CPU
    (bit-exact, ~3 min extra boot) —
    [`sm80-int8-repack-cpu-fallback.patch`](https://github.com/ahnguyen17/cmp-170hx-vllm)
    — with `expandable_segments` kept **off**, which on that card is an
    independent Xid-31 trigger. Not shipped here (no sm80 to regression-test
    against); recorded so the next GA100/A100 report starts from the answer
    instead of from five reboots.


42. **The OffloadingConnector's CPU tier can be silently useless: uniform
    blocks meet asymmetric chunk sizes, and one request evicts everything
    (issue #33).** The tier allocates equal-size blocks sized for the LARGEST
    group's offload chunk. Under KVarN the drafter's sliding-window group
    carries 128-token chunks against the 2,176-token maximum, so every SW
    crumb occupies a full ~14.6 MiB block — a single 23k-token request eats
    ~264 of a 4 GiB tier's 293 blocks and LRU-evicts every previous
    document. Stores succeed, `complete_store` succeeds, and every
    cross-request lookup is a MISS: 41 GB written, 0 bytes ever read back,
    with nothing in the logs. On bf16 KV the SW group happens to share the
    large per-token size (gotcha 25's 4096-B coincidence), the geometry
    stays uniform, and the same connector uplifts at PCIe speed — the KV
    dtype was never the mechanism, the chunk geometry it induces was. Since
    `offload-dflash-eagle-groups.patch` the config builder warns at boot
    with the waste factor and the `cpu_bytes_to_use` multiplier that would
    compensate (~17x under KVarN). Same patch fixes an adjacent quiet bug:
    upstream only ever sets `is_eagle_group` for DeepSeek V4, so the
    connector's fallback marked EVERY group as draft attention under
    `method=dflash`/`mtp` and silently excluded each group's trailing chunk
    from store while decoding; with dflash the flag now lands on the
    drafter's sliding-window group alone. Also: on bare-metal Linux the
    connector refuses this stack's default allocator
    (`expandable_segments:True`) at config validation — run it with
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (or the cumem
    allocator), which the WSL2 branch of the launchers already defaults to.
    And when eviction probing, keep the resend prompt BYTE-identical: a
    two-token label difference shifts every block hash and manufactures a
    convincing, fake "per-request hash instability" (ask how we know).

43. **"Every request re-prefills" is measurable, and the cause is usually the
    client's bytes, not the cache.** Reported against an agent client in
    [#47](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/47) (44 s mean
    TTFT at ~44k context, i.e. a full recompute per turn, while plain chat
    clients on the same server sat at the documented decode rates). Work the
    list in order:

    1. **Measure, per request.** The launchers pass
       `--enable-prompt-tokens-details`, so every response's
       `usage.prompt_tokens_details.cached_tokens` says how much of that
       prompt hit. (vLLM never emits DeepSeek's `prompt_cache_hit_tokens`
       field; this is the equivalent.) Server-side,
       `vllm:prefix_cache_queries` / `vllm:prefix_cache_hits` on `/metrics`
       give the same as counters.
    2. **Know the floor.** Hits are counted in whole hash units and the
       recurrent state resumes only at aligned boundaries
       (`--mamba-cache-mode align`), so the hit length truncates DOWN to a
       multiple of the unit — a prompt shorter than one unit can never hit,
       and a shared prefix pays up to one unit of recompute past the match.
       This is a fixed tax, not the 100%-miss failure mode.
    3. **Byte-identity is over the RENDERED prompt.** What the cache hashes
       is the chat-templated token stream: system prompt + tool definitions +
       every message, in order. One changed byte at position P invalidates
       everything after P. The classic offenders are dynamic content early in
       the payload: a timestamp or "current status" block in the system
       prompt, a heartbeat line spliced into the history, compaction that
       rewrites old turns, tool lists whose order is not stable. Diff two
       consecutive requests' FULL bodies (not just system + tools — the
       messages array too) and find the first differing byte; that byte is
       where your cache hit ends.
    4. **Interleaving evicts.** A cached prefix on this hybrid model holds
       KV blocks plus a recurrent-state page (~16% of the pool per request at
       k=7), and the pool is small. Two conversations round-robining — an
       agent's heartbeat pinging between chat turns is exactly that — can
       each evict the other before its recheck: measured as 0-of-3 warm in a
       3-context round-robin on 24 GB
       ([docs/wsl2-4090.md](wsl2-4090.md), retention section). The CPU
       offload tier turns that back into 3-of-3 (a RAM restore instead of a
       re-prefill).
    5. **The isolating experiment.** Bypass every proxy and fire the same
       long prompt twice at bare vLLM: if TTFT collapses on the second call,
       the server cache is healthy and the variable is the client payload
       (or a proxy that mutates it); if it does not, look at the server —
       and at 2 and 4 above.

44. **`CTX=long`'s fp8 KV cache has exactly one attention backend on sm86, and
    it is the one cell of the matrix this repo cannot A/B.** `FLASH_ATTN`
    refuses fp8 KV at startup ("requires FA3 on SM90 or FA4 on SM100" —
    [#34](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/34)) and
    `TRITON_ATTN` refuses it too ("native FP8 (fp8e4nv) requires SM89+",
    measured on the reference 3090), so the tier always auto-selects
    FlashInfer. #34 tracks a deterministic Xid-31 MMU write-fault (same
    virtual address twice, ~40 h uptime each) on that combination with MTP +
    prefix caching + chunked prefill at 28-34k context; cause unattributed
    between flashinfer's workspace and the async-scheduling window as of this
    entry. If you hit it, the flashinfer-free fallback is the int8 tier:
    `SPEC=dflash2 CTX=long` ships it by default, and for `SPEC=mtp` it is
    `VLLM_SPEC_DECODE_ATTN=1 EXTRA_ARGS="--attention-backend=TRITON_ATTN
    --kv-cache-dtype=int8_per_token_head"`. Measured cost on the reference
    box: 17.9k in + 256 out takes 23.7 s against fp8/FlashInfer's 18.9
    (~25% wall at that depth, mostly Triton prefill); in exchange the same
    pinned pool holds more tokens at int8's geometry. The default stays
    fp8/FlashInfer: two faults on one box do not justify a 25% tax on every
    other box, but you should know which combination you are running.

45. **On a low-RAM host, don't let the stock loader race page-cache eviction —
    stream the weights.** A 16 GB host (~10 GiB actually free) died loading the
    15.9 GiB checkpoint at shard 5/8
    ([#39](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/39)). Measured
    here under a 10 GiB cgroup cap standing in for that box: the **stock
    loader's memory peak was the cap to the byte** (10,737,418,240) — it loads
    by consuming everything and betting reclaim keeps up, which a fast NVMe
    wins and a busy desktop loses; pushed past the edge it reclaim-thrashes so
    hard the process stops responding even to SIGKILL (uninterruptible I/O),
    which is also what a "hung load" looks like from the outside. The **Run:ai
    streamer bounds the load instead, and is faster here**: 6.3 s vs 11.7 s to
    load, 8.46 GB process-wide peak under the same cap (its `memory_limit`
    caps the staging window; the rest is the engine's ordinary host
    footprint):

    ```bash
    venv/bin/pip install runai-model-streamer humanize
    # NOT runai-model-streamer-s3: it force-imports boto3 at package import
    # and breaks the loader on a box without it; local files don't need it.
    EXTRA_ARGS='--load-format=runai_streamer --model-loader-extra-config={"memory_limit":2147483648}' \
      bash single-user/start_qwen.sh
    ```

    Two adjacent facts from the same experiment: `swapon` cannot save the
    stock loader (mmap'd read-only file pages evict, they never swap — only
    the engine's anonymous memory benefits), and the whole test ran under
    `SPEC=off`, which as of this entry is a real mode rather than a silent
    fall-through to mtp.

    Two more from the #39 reporter's own box, once the loader was solved:
    `memory_limit` sized at or above the checkpoint's largest single tensor is
    the conservative setting (the bf16 `embed_tokens` is 2,542,796,800 bytes on
    both variants; the 2 GiB above loaded fine on the reference box, 2542796800
    is what they settled on) — and the *next* failure on a 24 GB card was not
    RAM at all: the plain `-W4A16-AutoRound` checkpoint leaves only 1.56 GiB for
    KV at `GPU_UTIL=0.93`, which cannot hold the 64k default (`max seq len
    65536 ... 4.76 GiB KV cache is needed`). The `-fast` variant (int4 lm_head
    and MTP head, `prepare/fetch_fast_variant.py`) is the launcher default for
    exactly that reason; with it the same box came up at a 72k-token pool.

46. **`vllm bench serve` defaults to `--seed 0`, and with prefix caching that
    poisons every A/B.** Same seed = same prompts call to call; later calls
    get partial prefix-cache hits whose size depends on the arm's pool
    geometry, so the contamination differs *between the configs you are
    comparing*. Measured: a 16k prefill "at" 4.0 s that cold costs 11.2 s
    (spec-off arm, big pool) next to a baseline reading 15-20% low. Every
    random-dataset call needs its own `--seed`; `bench/run_benchmarks.sh`
    does this now (`SEED_BASE` pins the sequence).

47. **Changing `VLLM_MARLIN_INT8_INCLUDE_RE` can replay a stale AOT-compiled
    graph.** The layer-select envs are in vLLM's compile hash, but
    `torch_aot_compile` keeps its own cache; switching the include set can
    crash at the first forward with a stable-ABI `aten::empty` RuntimeError
    from a cached inductor artifact. Wipe `~/.cache/vllm/torch_compile_cache`
    when switching layer sets. Separately, `INT8_LAYERS="mlp|linear_attn"`
    (int8 GDN + fp16 attention) crashes even from a clean cache — an inductor
    codegen bug with that mixed set on this torch pin; `mlp` and the full
    default both compile fine.

48. **`--max-num-batched-tokens` above 2048 does not boot in single-user
    dflash2 mode.** 4096 and 8192 both inflate the profiled activation peak
    past the transient floor next to the pinned `KV_MEM` pool: the engine
    fails initialization (batch mode documented the softer version of this —
    bigger chunks shrink the pool; with the pool pinned, the same memory
    comes out of the floor instead).

49. **Align-mode prefix caching periodically drops whole conversations to a
    0% hit — a geometry lottery plus an inverted eviction order on the one
    mamba state page that unlocks them.** A hybrid cache hit is the
    *intersection* of per-group hits, and the mamba group can only resume
    from a retained state snapshot; without one at or below the attention
    match (minus one 448-token EAGLE margin with spec decode), a fully
    cached multi-thousand-token attention prefix reads as a 0% miss and
    re-prefills from scratch (issue #52, upstream vllm#45238 — the same
    veto is why `--prefix-match-unit` can make things *worse* with spec
    decode). Three stock behaviors compose: align mode materializes ~one
    usable snapshot per turn at the last prefill chunk boundary, and on
    ~22% of turns (448/2048) it lands inside the EAGLE margin — that is
    the reported "every 4-5 turns" period; the fallback (the previous
    turn's snapshot, i.e. the block the hit resumed from) is CoW-released
    early in the turn; and mid-decode frees put every reusable snapshot at
    the *front* of the free queue while the attention blocks they unlock
    sit at the back. Any interleaved traffic evicts the snapshots first,
    and an unlucky turn with the fallback gone reconciles to 0. Measured
    (12-turn conversation, two unrelated requests between turns, shrunk
    pool): 81-91% hits for five turns, then 0% on every turn, TTFT 2.1 s →
    17-31 s, the coordinator logging a discarded 16,576-token attention
    match. `patches/mamba-align-checkpoint-order.patch` keeps up to three
    state blocks per running request per mamba group — the CoW-carried key
    plus the last written prompt-region snapshots — until request end,
    freed last. It ships **default off** (measured no-harm at two pool
    sizes, but the win regime — context comparable to the pool over many
    turns — is not cheaply reproducible in a short cell); enable with
    `VLLM_MAMBA_ALIGN_KEEP_CHECKPOINTS=1` if you see the periodic spikes. Two stronger variants — pinning the snapshots against
    eviction, bounded or not — measured *worse*: each skipped eviction
    lands on the conversation's own attention tail instead, which breaks
    the same hit from the other side. If between-turn traffic exceeds the
    whole free pool, nothing survives by policy; that regime needs a
    bigger `KV_MEM`, not a smarter queue.

50. **First-request Triton compiles on a fresh boot came from four separate
    warmup gaps, and the last one is invisible without logging what Triton
    specialises on.** Issue #48's fingerprint — a stall in the first large
    chunked prefill after boot, preceded by `jit_monitor` warnings — had, on
    the reference 3090 (`SPEC=dflash2 CTX=huge`, one 30k-token first
    request), four kernels compiling inside request 1, each with its own
    cause: (1) `_prepare_dflash_inputs_kernel`'s `BLOCK_SIZE` ladder only
    reaches 256 on a large prefill continuation, never in decode-shaped
    dummies (`patches/dflash2-prewarm.patch` compiles every rung at boot);
    (2) the rejection sampler's three kernels are upstream vLLM's and never
    run in the profile-time dummy sampler pass, which has no draft tokens
    (`patches/spec-sampler-prewarm.patch` runs one spec-shaped verify in
    `kernel_warmup()`); (3) KVarN's block→slot lookup was sized to a 1024
    floor at profile time and resized by the first serving build, and its
    size is a kernel constexpr (`NUM_BLOCKS_LOOKUP`) — now derived once at
    impl construction from the KV budget as an upper bound (48,934 slots for
    6,103 real blocks; the kernels bound-check, so over-sizing is free); and
    (4) the one that survived all of the above: Triton specialises *integer*
    arguments on divisibility by 16, and the block table's row stride is its
    width — the warmup's `cdiv(max_model_len, group)` = 1920 carries the
    attribute, the runner's real table is 1921 wide and does not, so the two
    launches were different compiled variants however faithfully the shapes
    were mirrored. `stride_bt_b` joined `MAX_BLOCKS_PER_REQ` in the kernel's
    `do_not_specialize`. Measured after all four: **zero** `JIT compilation
    during inference` lines on the same boot and request. The tool that
    found (4): `KVARN_SPEC_DEBUG=1` logs pointer alignment and integer
    divisibility / equal-to-1 for the warmup launch and the first real
    launches of the packed-kv kernel — diff the two lines.
    `--jit-monitor-verbose` prints the signature of each in-request compile
    but truncates the specialisation attributes at 120 characters, which is
    why it could name the kernel and not the cause. `KVARN_LOOKUP_BLOCKS`
    pins the lookup size if a deployment ever needs to.

51. **`prompt_logprobs` is wrong on `CTX=huge` + `SPEC=mtp` + prefix caching, and
    the NaN 400s are only its visible half.** Reported as
    [#64](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/64) from a WSL2
    3090 — `bench/quality_battery.py --ppl-only` failing with
    `{"message":"Out of range float values are not JSON compliant: nan"}` on
    some documents, and perplexity drifting 3.7% between identical runs.
    Reproduced here on bare metal, on the reporter's own document indices, so
    it is not a WSL2 effect. What it actually takes is all three of KVarN, MTP
    and `PREFIX_CACHE=1`; the same 106-document battery, same checkpoint, two
    concurrent workers:

    | profile | NaN batteries | en perplexity |
    |---|---|---|
    | `CTX=huge` `SPEC=mtp` `PREFIX_CACHE=1` | 4 of 5 | 12.6-13.7, drifting |
    | `CTX=huge` `SPEC=mtp` `PREFIX_CACHE=0` | 0 of 5 | **10.7628**, identical across runs |
    | `CTX=huge` `SPEC=off` `PREFIX_CACHE=1` | 0 of 5 | 10.7646 |
    | `CTX=huge` `SPEC=dflash2` `PREFIX_CACHE=1` | 0 of 5 | 10.7643 |
    | `CTX=fast`, `off` / `mtp` / `dflash2` | 0 of 6 | 10.7614 / 10.7659 / 10.7633 |

    So the inflated perplexity and the NaN are one bug, not two: in the broken
    combination the logprobs that come back are ~23% worse on English than the
    same server produces with prefix caching off, and the requests whose
    corruption reaches a non-finite float are the ones that 400. Every clean
    configuration agrees to four decimals, which is also what makes the broken
    one unmistakable.

    Two things that look like the cause and are not. **The documents**: sent one
    at a time on a single thread, all of them return clean logprobs — it needs
    co-scheduled requests, and the failures land on adjacent index pairs, which
    under two workers are exactly the pairs that share a prefill batch. **The
    pool size**: within MTP it is geometry-dependent (failed at 299k and 312k
    tokens of pool, clean at 265k, 352k, 359k and 390k), which is what makes it
    look intermittent across boots — but `SPEC=dflash2` pinned to 312,242
    tokens, matching the failing MTP geometry to 0.02%, is clean, so the
    speculator is the variable and the geometry only decides whether it fires.
    `MAX_LEN` is not involved: 240000 and the 245760 default both fail and both
    pass depending on the rest.

    Practical rule until the read path is fixed: measure perplexity or anything
    else using `prompt_logprobs` on KVarN with `PREFIX_CACHE=0`, or on a
    non-MTP speculator. Ordinary generation is not implicated — needle
    retrieval and decode rates are normal on the same server.

52. **`DFLASH_TOKENS=15` asserted at engine start on the int4 path, because the
    drafter's promoted block only has to *cover* the primary page, not divide
    it.** Filed as
    [#63](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/63), fixed in
    `patches/hybrid-sw-block-promote.patch`. `alternative.sh`
    (`int4_per_token_head`) died in a bare `assert` in
    `kv_cache_coordinator.py` at `DFLASH_TOKENS=15` — at any `MAX_LEN`, with
    prefix caching on or off — while `DFLASH_TOKENS=7` on the identical config
    booted.

    The chain, all of it visible in the boot log:

    | | `DFLASH_TOKENS=7` | `DFLASH_TOKENS=15` |
    |---|---|---|
    | mamba page (grows with the spec-decode state) → primary block | 1696 | **1840** |
    | drafter's covering block (16 → smallest multiple whose page covers) | 848 | **928** |
    | primary / drafter | exactly 2 | 1.983 |
    | result | boots | `AssertionError` |

    The scheduler's granularity is the LCM of the primary groups' blocks and
    every group's block has to divide it. `_promote_indivisible_block_sizes`
    only guaranteed the drafter's page *covers* the maximum, and 848 divided
    1696 by luck. The fix rounds the promotion up to the smallest divisor of
    the primary block instead — 928 → 1840, after which the primary layers
    scale 1840 → 3680 through the branch they already take.

    Two things worth knowing. **It is an int4-only shape**: on
    `int8_per_token_head` (`CTX=long`) the primary block and the drafter's
    covering block come out *equal* (864 at 7, 944 at 15), so divisibility is
    free and both arms are byte-identical before and after the fix —
    `CTX=long DFLASH_TOKENS=7` still pools 138,696 tokens. **The wider verify
    block is not free on int4**: at 15 the pool is 53,908 tokens against
    142,843 at 7, and the 256k default no longer fits (`estimated maximum
    model length is 180320`, a clear `ValueError` rather than an assert).

53. **A benchmark row without its compile-cache state is not reproducible, because the
    autotuner's timing race picks the kernels and the kernels pick the trajectory.**
    Filed as [#75](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/75). Six Triton
    kernels in the chunked Gated DeltaNet path are autotuned at first use; vLLM caches the
    winners, but a fresh container or `VLLM_DISABLE_COMPILE_CACHE=1` re-runs the race, and a
    different winner is a different reduction order, a different last bit, and at greedy a
    different token at the first near tie. Measured on one image with the cache wiped before
    each boot: 8 boots, 8 distinct winner sets, 5 distinct trajectories, the same twelve-turn
    tool conversation ranging from 58 to 189 tool calls. With the cache persisted, two boots
    were identical to the call. So: mount a persistent cache into bench containers rather than
    disabling it for hygiene, and copy the `*.autotune.json` records beside a published row so
    a reader can tell whether two rows are even comparable. A quiet bare box re-times to the
    same winners even with the cache deleted; container timing noise is what makes the draw
    vary. The same cache state also moves the profiled peak activation and therefore the KV
    pool, by 0.92 GiB on the reference 3090, which is a separate channel with a separate fix
    (state it, or pin `--kv-cache-memory`).

54. **A high acceptance rate can mean the drafter is good or the generation has collapsed,
    and the counter cannot tell you which.** Degenerate text is trivially predictable. One
    repetition loop during the [#73](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/73)
    work scored 79.7% acceptance per drafted token against a normal 35 to 45%, with a
    distinct-word ratio of 0.051 against 0.77 to 0.83, and it was echoing the instruction
    appended to its own prompt. So acceptance is heavy-tailed, a t-test over a dozen short
    generations is the wrong instrument, and an arm that happens to sample more collapsed
    trajectories looks like an acceptance *improvement*. Guard on drafted tokens per round
    above about 1.2 times the drafter width (calibrated on 96 rows: above the width flags 40%
    of ordinary rows, above 1.2 times it flags only the real events), or on distinct-word
    ratio when the text is available. The guard reads the engine's own repetition signal,
    because adaptive verify length extends the block only while a request is reproducing its
    context. Run the guard **before** any stratification on block size: a degenerate row sorts
    into the long-block stratum by construction and one such row moved a stratum estimate by
    two tokens per step.

55. **Per-request acceptance figures depend on what ran before the request, and on some boxes the
    request's whole trajectory does.** Two observations, two boxes. On a quiet native 3090, same
    build, same seeds, one boot, only the request order changed: drafts and accepted tokens came back
    identical on every seed and drafted tokens did not, so `1 + accepted / drafts` repeated exactly
    while `accepted / drafted` had an order-dependent denominator. On an RTX 4090 under WSL2 with a
    prompt that keeps the long block engaged, the same design changed drafts and accepted tokens too
    on four of six seeds, one seed by a factor of two, so there tokens per step itself moved with
    order. Two mechanisms, both real: the long block is sticky and coasts on prior state without
    consulting the emitted count, so a request that follows a long-block request inherits some of its
    block length; and a different block length is a different verify batch shape, which on a
    numerically knife-edged model can flip a near-tie token even with the seed fixed. Prefix caching
    was the obvious third candidate, since the requests share a prompt; with the engine started
    `--no-enable-prefix-caching` and the log confirming it, the order effect was unchanged, so it
    is not the driver. What follows is the same either way: hold the request
    order fixed within a comparison, report tokens per step rather than acceptance per drafted token,
    and treat per-request figures from a sequence as dependent samples, never as independent ones.

56. **At `DFLASH_TOKENS=15` on ordinary text the engine drafts 7 and queries 8, so 15 and 7 are
    the same experiment unless the text repeats.** With the lookup on, the drafter's block is
    clamped to the checkpoint's trained block (7), `num_query_per_req` follows it (8), and
    adaptive verify length asks for the long block only while a request is reproducing its
    context (`dflash2/speculator.py`, on by default). Measured over sixty single-prompt rows
    on the reference 3090: 55% at exactly 7.000 drafted tokens per round, and the lookup
    supplying 4.3% of the eight positions it could fill. On an eight-prompt cohort at 1024
    output tokens the long block engaged on 44% of rows and those rows ran markedly faster.
    Consequences: matching results at both widths are weak evidence that an effect is not
    lookup-specific; anything that acts only in the long block, including
    `dflash2-z-adaptive-emitted.patch`, is invisible on a short single-prompt cell and only
    shows on a cohort; and `VLLM_DFLASH2_LOOKUP_CHEAP_CTX` (default 0, so the branch is dead)
    takes the long block unconditionally below a context threshold and would invalidate any
    bisect run across it.

57. **Two passes with a fixed seed are two replays, and a short single-prompt cell cannot see a
    cohort-scale effect at any number of seeds.** The engine seeds from zero and the noise draw
    is a function of seed and position, so repeat boots on a quiet box come back bit-identical
    on every counter, and "reproducible to three significant figures from two passes" measures
    a deterministic harness rather than bounding an effect. The 16% DFlash2 acceptance
    regression in [#73](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/73) was invisible
    on one README prompt at 256 output tokens with six seeds *and* with thirty (seed spread
    ten points, sd about four), and reproduced immediately through `bench/prefill_ab.sh` on the
    cohort at 1024. Prompt choice alone moved acceptance from 22.7% on the cohort to 36.8% on a
    236-character prompt, larger than the regression. Three different questions were asked of
    that short cell during the bisect and it was underpowered for all of them. Match the
    workload to the claim, vary the seed per request, and read a bare number from a fixed-seed
    pass as one draw.
