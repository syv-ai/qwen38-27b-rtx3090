# prepare/ — one-time model preparation

The published W4A16 quant of Qwen3.8-27B is not servable on 24 GB as it ships: two
2.5 GB bf16 embedding matrices and an unquantized MTP draft module. These scripts
fix that in place, on the CPU, once. They are the [Setup](../README.md#setup) steps,
and `docker compose run --rm prepare` (see [docker/prepare.sh](../docker/prepare.sh))
runs exactly them, each skipped when its result is already in the model dir.

Run from the repo root, in order — `quant_lm_head.py` first, because
`build_draft_vocab.py` slices its rows:

```bash
V=venv/bin/python; M=models/Qwen3.8-27B-W4A16-AutoRound
$V prepare/quant_lm_head.py $M      # lm_head -> int8 group-128, in place: ~1.3 GB freed
$V prepare/quant_embed.py   $M      # embed_tokens likewise (untied): another ~1.3 GB
$V prepare/quant_mtp.py     $M      # the mtp.* draft module (~850 MB bf16) -> int8
$V prepare/build_draft_vocab.py $M --ids prepare/draft_vocab_ids.json
$V prepare/fetch_fast_variant.py    # optional, ~1 GB: the single-user "fast" variant
$V prepare/fetch_dflash2.py         # optional, 1.2 GB: the DFlash2 drafter (SPEC=dflash2)
```

`build_draft_vocab.py` writes a 40,960-row slice of `lm_head` for the MTP drafter to
score instead of the full 248k vocabulary; `draft_vocab_ids.json` is the shipped id
list, and `--corpus` counts your own instead. It needs
[patches/0004-qwen3_5-mtp-draft-vocab.patch](../patches/0004-qwen3_5-mtp-draft-vocab.patch).

## A different checkpoint

`quant_heads_stream.py` does the work of `quant_lm_head.py` + `quant_embed.py` +
`quant_mtp.py` in one pass, for checkpoints those three cannot open: **single-shard**
ones (they read a shard into RAM whole; the uncensored build ships one 18.6 GB
`model.safetensors`) and **asymmetric AWQ** bodies (they clone `config_groups.group_0`
onto the symmetric tensors they write, so vLLM then looks for a `weight_zero_point`
that does not exist). Same math, same output tensors, peak RSS well under the shard
size (9.7 GB measured on the 18.6 GB example here -- still not a low-RAM tool).

```bash
$V prepare/fetch_thirdparty.py                          # ~18.6 GB (or: fetch_thirdparty.py <hf-repo>)
$V prepare/quant_heads_stream.py models/Qwen3.8-27B-Uncensored-W4A16
$V prepare/build_draft_vocab.py  models/Qwen3.8-27B-Uncensored-W4A16 \
  --ids prepare/draft_vocab_ids.json
```

Then `MODEL=$PWD/models/Qwen3.8-27B-Uncensored-W4A16 bash single-user/start_qwen.sh`.
`SPEC=dflash2` additionally needs its pinned pool resized, because this checkpoint is
~1 GB heavier than the one those constants were measured on — the command and the
numbers are in the main README, "A different checkpoint: the uncensored build".
`--mtp-bits 4` and `--keep-fc` exist for experimenting with the draft module; the
defaults (int8, `mtp.fc` quantized) are what was verified.

The two `fetch_*` scripts only download: the fast variant is the int4-GPTQ lm_head and
drafter plus a draft vocabulary counted over the model's own outputs (worth ~15% in
single-user mode), and `fetch_dflash2.py` is the W4A16 DFlash2 block drafter. Both are
rebuildable from scratch — that is what [drafter/](../drafter/) is.

`bash verify.sh --no-server` checks every step above against the model dir and names
the script to run for whatever is missing. Each in-place script backs up what it
rewrites next to the original (`.bak*`), so a step can be undone without re-downloading
19.5 GB. Why each one is worth doing, with measurements:
[docs/optimizations.md](../docs/optimizations.md).
