"""Build a vocab-truncated draft head for MTP speculative decoding.

The MTP drafter has to run the 248k-row lm_head once per draft token, and at
4-6 drafts per step that head read (1.3 GB int8) dominates the draft cost.
Speculative decoding stays exact no matter what the drafter proposes, so the
drafter can use a head restricted to the N most frequent tokens: tokens
outside the shortlist are simply never drafted (that position gets rejected
and the target's own sample is used, as always).

This script counts token frequencies over a text corpus (Danish + English +
code + the model's own outputs), picks the top N ids (plus special tokens),
slices those rows out of the already-int8-quantized lm_head, and stores them
as `mtp.draft_lm_head.*` in model_extra_tensors.safetensors, plus the id map in
`mtp_draft_vocab_ids.pt`. Needs the matching vLLM patch
(patches/0004-qwen3_5-mtp-draft-vocab.patch) to be used.

Usage (from the repo root):
  python prepare/build_draft_vocab.py /path/to/model --ids prepare/draft_vocab_ids.json  # shipped id list
  python prepare/build_draft_vocab.py /path/to/model --n 40960 --corpus f1 f2 ...        # or count your own
Corpus files: .txt/.jsonl (uses "prompt"/"response"/"messages"/"text" fields)/.parquet(text)/.py
The shipped draft_vocab_ids.json was counted over Danish web text (fineweb-2),
English Wikipedia, Python source and the model's own chat outputs (8.8M tokens);
held-out coverage 95%.
"""
import glob, json, os, sys, shutil, collections
import torch
from safetensors import safe_open
from safetensors.torch import save_file

d = sys.argv[1].rstrip("/") + "/"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40960
corpus = sys.argv[sys.argv.index("--corpus") + 1:] if "--corpus" in sys.argv else []
ids_file = sys.argv[sys.argv.index("--ids") + 1] if "--ids" in sys.argv else None

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(d)

def texts_from(path, limit_bytes=20_000_000):
    n = 0
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for t in pq.read_table(path, columns=["text"]).column("text").to_pylist():
            yield t; n += len(t)
            if n > limit_bytes: return
    elif path.endswith(".jsonl"):
        for line in open(path):
            try: r = json.loads(line)
            except Exception: continue
            parts = []
            for k in ("prompt", "response", "text"):
                if isinstance(r.get(k), str): parts.append(r[k])
            if isinstance(r.get("messages"), list):
                parts += [m.get("content", "") for m in r["messages"] if isinstance(m.get("content"), str)]
            t = "\n".join(parts); yield t; n += len(t)
            if n > limit_bytes: return
    else:
        t = open(path, errors="ignore").read(); yield t

counts = collections.Counter()
held = collections.Counter()
total = 0
if ids_file:
    ids = sorted(set(json.load(open(ids_file))))
    print(f"using {len(ids)} ids from {ids_file}")
    corpus = []
for i, path in enumerate(corpus):
    for j, t in enumerate(texts_from(path)):
        ids = tok(t, add_special_tokens=False).input_ids
        (held if j % 10 == 0 else counts).update(ids)
        total += len(ids)
print(f"corpus tokens: {total}")

special = set(tok.all_special_ids)
if ids_file:
    special = set()
for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
    tid = tok.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid >= 0: special.add(tid)
if not ids_file:
    top = [t for t, _ in counts.most_common() if t not in special][: N - len(special)]
    ids = sorted(set(top) | special)
    cover = sum(c for t, c in held.items() if t in set(ids)) / max(1, sum(held.values()))
    print(f"draft vocab: {len(ids)} ids, held-out token coverage {cover*100:.2f}%")
    for n_try in (16384, 32768, 49152, 65536):
        s = set(t for t, _ in counts.most_common(n_try)) | special
        c = sum(c for t, c in held.items() if t in s) / max(1, sum(held.values()))
        print(f"  coverage at N={n_try}: {c*100:.2f}%")
    json.dump(ids, open(d + "draft_vocab_ids.json", "w"))
    print(f"id list written to {d}draft_vocab_ids.json (copy it next to this script to reuse)")

# slice lm_head rows
idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
head_shard = wm["lm_head.weight_packed"]
with safe_open(d + head_shard, framework="pt") as f:
    wp = f.get_tensor("lm_head.weight_packed")   # [vocab, K/8] int32
    ws = f.get_tensor("lm_head.weight_scale")    # [vocab, K/group]
    shape = f.get_tensor("lm_head.weight_shape")
ids_t = torch.tensor(ids, dtype=torch.int64)
sub_p = wp.index_select(0, ids_t).contiguous()
sub_s = ws.index_select(0, ids_t).contiguous()
sub_shape = torch.tensor([len(ids), int(shape[1])], dtype=torch.int64)
print(f"draft head: packed {tuple(sub_p.shape)} {sub_p.dtype}, scales {tuple(sub_s.shape)} {sub_s.dtype}, "
      f"{(sub_p.numel()*4 + sub_s.numel()*2)/1e6:.0f} MB")

extra = "model_extra_tensors.safetensors"
# The three quant_*.py scripts leave the base model's extras in this file; a
# single-shard export processed by prepare/quant_heads_stream.py has no extras
# file at all (#37) -- the draft head becomes its first content.
tensors = {}
meta = None
if os.path.exists(d + extra):
    with safe_open(d + extra, framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    if not os.path.exists(d + extra + ".bak-draft"):
        shutil.copy(d + extra, d + extra + ".bak-draft")
tensors["mtp.draft_lm_head.weight_packed"] = sub_p
tensors["mtp.draft_lm_head.weight_scale"] = sub_s
tensors["mtp.draft_lm_head.weight_shape"] = sub_shape
save_file(tensors, d + extra, metadata=meta or {"format": "pt"})
for s in ("weight_packed", "weight_scale", "weight_shape"):
    wm[f"mtp.draft_lm_head.{s}"] = extra
json.dump(idx, open(d + "model.safetensors.index.json", "w"), indent=2)
torch.save(ids_t, d + "mtp_draft_vocab_ids.pt")
print("done")
