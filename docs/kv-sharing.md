# Cross-layer KV cache sharing

**Status: experiment, off by default, unmeasured.** Nothing here has run on a
GPU yet. The patch is written and the harness is written; what is missing is the
only thing that decides whether the idea survives.

## What this is

Qwen3.8-27B is 64 layers with `full_attention_interval: 4`, so 16 layers own a
KV cache and 48 are gated-delta-net layers that own a recurrent state instead.
Each owner stores 2 x 4 heads x 256 head_dim per token:

| storage | global KV cache | a 57,669-token pool |
|---|---:|---:|
| fp16 | 65,536 B/token | 3.52 GiB |
| int8 | 32,768 B/token | 1.76 GiB |
| int4 | 16,384 B/token | 0.88 GiB |

DeepSeek-V4.1-Flash reports 890 bytes per token. A large part of how it gets
there is that most of its attention layers do not own a cache at all: across 40
layers only four compress their own, and the rest read those. The same trick is
available to us, because vLLM already implements the mechanism.

An `Attention` layer constructed with `kv_sharing_target_layer_name` is skipped
when the runner builds KV cache specs, so **no memory is allocated for it**, and
its cache tensor is aliased to the target's. vLLM validates that the target
comes before it in the model and has the same attention type. The Qwen3.5 model
code simply never passes the argument.
`patches/qwen3_5-kv-cache-sharing.patch` passes it.

## What this is not

It is not the Late Layer KV Approximation doing the rounds after
DeepSeek-V4.1-Flash shipped (`kishida/Q3-8B-KVA-Projector`). That technique
trains a small projector to *guess* the back half's KV entries from a
mid-network hidden state, so the back half's forward pass can be skipped during
prefill. It saves prefill compute and leaves the cache exactly the same size.

It also does not port to this model. Its 50% comes from Qwen3-8B being entirely
full-attention, so the whole back half is skippable. Three quarters of our
layers are gated-delta-net, whose recurrent state is a sequential function of
every prompt token and cannot be written per-position, so the back half's
forward pass has to run anyway and only the eight full-attention mixers in it
come out:

| prompt tokens | ceiling on prefill saving |
|---|---:|
| 4,096 | 4.1% |
| 16,384 | 6.4% |
| 51,200 | 11.7% |

That is a ceiling with a perfect projector and no kernel overhead, against ~50%
for the model it was built on. It was not worth a training run.

## The layouts

`KV_SHARE` (launcher) sets `VLLM_QWEN_KV_SHARE` (vLLM).

- `group:N` groups consecutive full-attention layers N at a time; the first of
  each group owns the cache. `group:2` leaves owners at layers 3, 11, 19, 27,
  35, 43, 51, 59. The target is always within three attention layers, which
  should be the gentler layout.
- `suffix:K` keeps the first K owners and points every later layer at the K-th.
  `suffix:4` leaves owners at 3, 7, 11, 15 and has layers 19 through 63 all read
  layer 15. This is the You Only Cache Once layout, and it is the only one
  `--kv-sharing-fast-prefill` can use, because that flag requires the sharing
  layers to form a suffix. It should be the more damaging layout, and it is the
  one that also buys prefill time.

`group:4` and `suffix:4` both leave four owners, which is DeepSeek's count.

## Running the sweep

```bash
bash bench/kv_share_sweep.sh                      # off, group:2, group:4
ARMS="off group:2 group:4 suffix:4" bash bench/kv_share_sweep.sh
GSM_N=100 NEEDLE=0 bash bench/kv_share_sweep.sh   # quicker first look
```

It boots on port 18021, so it does not fight a production server on 18020.
Budget about 25 minutes per arm. Results land in `bench/results-kv-share/`.

Three things the harness refuses to get wrong, each because the mistake has
already produced a confident wrong answer in this repo:

1. **It verifies the arm took effect**, by reading the owner count out of the
   server log and comparing it against the arithmetic. An arm that silently ran
   stock would report no quality loss and be believed. That is how the
   patch-integrity CI job stayed green while checking five of thirty patches.
2. **It boots each arm twice and measures the second boot.**
   `VLLM_QWEN_KV_SHARE` is in the torch.compile cache key, so every arm's first
   boot compiles cold, and a cold compile profiles about 0.9 GiB more peak
   activation than a warm one. Comparing a cold arm against a warm arm compares
   compile-cache states, not layouts.
3. **It measures perplexity with `PREFIX_CACHE=0`**, because prompt_logprobs is
   corrupted by prefix caching on this hybrid model and reads about 23% high.

## What would make this shippable

The baseline is the `off` arm from the same sweep on the same box. Numbers from
another session are not a baseline.

Untrained, the honest expectation is that `group:2` degrades and `group:4` falls
apart. The useful output is the *shape* of the curve, because that is what says
whether a light adapter fine-tune could close the gap or whether these layers
are too dissimilar to share at all.

Roughly:

- `group:2` within noise of baseline on GSM8K and needle retrieval would be a
  genuine surprise and would make a fine-tune obviously worth costing.
- `group:2` down a few points, `group:4` broken, says the mechanism works and
  the weights need adapting. That is the interesting middle, and it is what a
  cheap QLoRA on the 16 attention layers would target.
- `group:2` already broken says these layers carry genuinely different
  information and sharing is not a retrofit. Write it up as a gotcha and close
  the book.

Needle retrieval at depth is the sharpest instrument of the three. Perplexity
moves least and hides the failure mode people actually notice.

## Known risks

- **Cache group geometry.** Shared layers are appended to their target's KV
  cache group. On this hybrid model, group composition has been the source of
  several hard bugs already: block promotion needing to divide rather than cover,
  and scratch allocation landing outside the memory profile. Expect the first
  failure here rather than in model quality.
- **Speculative decoding.** The sweep runs `SPEC=dflash2` because that is
  production. If an arm fails to boot, re-run it with `SPEC=off` before
  concluding anything about the sharing layout.
- **The `off` arm must log nothing.** The harness aborts the baseline if it
  sees a sharing line, because a baseline that is quietly sharing would flatten
  every difference in the table.
