"""Requantize lm_head, embed_tokens and the MTP module to int8/int4
(group-128, symmetric) in compressed-tensors pack-quantized format, in place.

Same math and output as quant_lm_head.py / quant_embed.py / quant_mtp.py, but
for checkpoints those three cannot handle:

  - single-shard checkpoints. They read a whole shard into a dict before
    rewriting it; philbert440/Qwen3.8-27B-Uncensored-* ships one 18.6 GB
    model.safetensors (2384 tensors), which does not fit in RAM here. This
    streams the shard tensor-by-tensor instead, copying untouched tensors as
    raw bytes, so peak RSS is a few GB regardless of shard size.

  - asymmetric bodies. The three scripts deepcopy config_groups.group_0 and
    override only num_bits/targets. AWQ exports have symmetric=false and
    zp_dtype=torch.int8, so the cloned group would declare asymmetric quant
    for the symmetric tensors written here and vLLM would look for a
    weight_zero_point that does not exist. The groups written below always say
    symmetric=true / zp_dtype=null.

Usage: venv/bin/python prepare/quant_heads_stream.py /path/to/model [--mtp-bits 8|4] [--keep-fc]

This is what the uncensored checkpoint needs (prepare/fetch_uncensored.py); the base
model is 7 shards and symmetric, so quant_lm_head/quant_embed/quant_mtp serve it fine.

The rewritten shards replace the originals; the pre-quant files are kept as
<shard>.bak-orig (renamed, not copied). config.json and the safetensors index
are backed up as .bak-quant.

Publication safety (F04): the FIRST .bak-orig is the pristine rollback point
and is never overwritten — on single-shard exports the MTP and head tensors
share one physical shard, and re-running would otherwise back up an
already-modified shard over the original.
"""

import copy
import json
import os
import struct
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

GROUP = 128
HEAD_BITS = 8
MTP_BITS = int(sys.argv[sys.argv.index("--mtp-bits") + 1]) if "--mtp-bits" in sys.argv else 8
KEEP_FC = "--keep-fc" in sys.argv
ROWS = 16384  # quantize this many rows at a time, to bound peak RSS

MTP_LINEARS = ([] if KEEP_FC else ["mtp.fc"]) + [
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
]

d = sys.argv[1].rstrip("/") + "/"

DTYPE_STR = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32",
    torch.int8: "I8", torch.int32: "I32", torch.int64: "I64", torch.uint8: "U8",
}


def quantize(w, bits):
    """int-N group-wise symmetric quant, row-chunked. Returns packed/scale/err."""
    qmax = 2 ** (bits - 1) - 1
    out_f, in_f = w.shape
    assert in_f % GROUP == 0, w.shape
    packed_parts, scale_parts = [], []
    num, den = 0.0, 0.0
    for lo in range(0, out_f, ROWS):
        chunk = w[lo:lo + ROWS].to(torch.float32)
        g = chunk.reshape(chunk.shape[0], in_f // GROUP, GROUP)
        s = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-10)
        q = torch.clamp(torch.round(g / s), -qmax - 1, qmax).to(torch.int8)
        deq = (q.to(torch.float32) * s).reshape(chunk.shape[0], in_f)
        num += (deq - chunk).pow(2).sum().item()
        den += chunk.pow(2).sum().item()
        packed_parts.append(pack_to_int32(q.reshape(chunk.shape[0], in_f), bits, packed_dim=1).contiguous())
        scale_parts.append(s.squeeze(-1).contiguous())
        del chunk, g, q, deq
    return torch.cat(packed_parts), torch.cat(scale_parts), (num / den) ** 0.5


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def stream_rewrite(src, dst, drop, add):
    """Copy src to dst, omitting tensor names in `drop` and appending `add`
    (name -> tensor). Untouched tensors are copied as raw bytes."""
    hdr, data_start = read_header(src)
    meta = hdr.pop("__metadata__", None)
    keep = [(k, v) for k, v in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0])
            if k not in drop]

    new_hdr, off = {}, 0
    if meta is not None:
        new_hdr["__metadata__"] = meta
    plan = []
    for k, v in keep:
        b0, b1 = v["data_offsets"]
        size = b1 - b0
        new_hdr[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + size]}
        plan.append(("copy", data_start + b0, size))
        off += size
    for k, t in add.items():
        t = t.contiguous()
        size = t.numel() * t.element_size()
        new_hdr[k] = {"dtype": DTYPE_STR[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + size]}
        plan.append(("write", t, size))
        off += size

    blob = json.dumps(new_hdr, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for kind, a, size in plan:
            if kind == "copy":
                fi.seek(a)
                left = size
                while left:
                    chunk = fi.read(min(left, 64 << 20))
                    if not chunk:
                        raise IOError(f"short read in {src}")
                    fo.write(chunk)
                    left -= len(chunk)
            else:
                # .numpy() has no bfloat16; reinterpret as bytes instead
                fo.write(memoryview(a.contiguous().view(torch.uint8).numpy()))
    return off


idx_path = d + "model.safetensors.index.json"
_orig_index = open(idx_path, "rb").read()
# F04: the rollback asset is written atomically too — a truncated .bak is
# as useless as no .bak.
_tmp_bak = idx_path + ".bak-quant.tmp"
open(_tmp_bak, "wb").write(_orig_index)
os.replace(_tmp_bak, idx_path + ".bak-quant")
idx = json.loads(_orig_index)
wm = idx["weight_map"]

lm_key = "lm_head.weight"
emb_key = next(k for k in wm if k.endswith("embed_tokens.weight"))

# ---- lm_head + embed_tokens: one streaming pass per shard that holds them.
# Single-shard exports (the original case) land in one group; multi-shard
# exports with the two heads in different shards get one pass per shard ----
groups = {}
for key, scale_dtype in ((lm_key, torch.float16), (emb_key, torch.bfloat16)):
    groups.setdefault(wm[key], []).append((key, scale_dtype))

for big, keys in groups.items():
    add = {}
    with safe_open(d + big, framework="pt") as f:
        for key, scale_dtype in keys:
            w = f.get_tensor(key)
            out_f, in_f = w.shape
            packed, scale, err = quantize(w, HEAD_BITS)
            print(f"  {key}: {(out_f, in_f)} int{HEAD_BITS} g{GROUP}, round-trip rel error {err:.4f}")
            assert err < 0.01, f"quantization error too high for {key}, aborting"
            base = key[:-len(".weight")]
            add[base + ".weight_packed"] = packed
            # linears take fp16 scales; the embedding path creates them in params_dtype
            add[base + ".weight_scale"] = scale.to(scale_dtype)
            add[base + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
            del w, packed, scale

    print(f"rewriting {big} (streaming)")
    tmp = d + big + ".tmp"
    stream_rewrite(d + big, tmp, drop={k for k, _ in keys}, add=add)
    # F04: keep the FIRST .bak-orig as the pristine copy. Overwriting it with an
    # already-modified shard (MTP and head tensors share one physical shard on
    # single-shard exports) destroys the rollback point.
    if not os.path.exists(d + big + ".bak-orig"):
        os.replace(d + big, d + big + ".bak-orig")
    else:
        os.remove(d + big)
    os.replace(tmp, d + big)
    del add

    for key, _ in keys:
        base = key[:-len(".weight")]
        del wm[key]
        for s in ("weight_packed", "weight_scale", "weight_shape"):
            wm[f"{base}.{s}"] = big

# ---- MTP module (small shard, fits in RAM) ----
mtp_shards = {wm[m + ".weight"] for m in MTP_LINEARS}
assert len(mtp_shards) == 1, f"mtp weights span several shards: {mtp_shards}"
mtp_shard = mtp_shards.pop()
print(f"mtp linears live in {mtp_shard}, quantizing to int{MTP_BITS} g{GROUP}")

tensors = {}
with safe_open(d + mtp_shard, framework="pt") as f:
    mtp_meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
for m in MTP_LINEARS:
    w = tensors.pop(m + ".weight")
    out_f, in_f = w.shape
    packed, scale, err = quantize(w, MTP_BITS)
    print(f"  {m}: {(out_f, in_f)} round-trip rel error {err:.4f}")
    tensors[m + ".weight_packed"] = packed
    tensors[m + ".weight_scale"] = scale.to(torch.float16)
    tensors[m + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    del wm[m + ".weight"]
    for s in ("weight_packed", "weight_scale", "weight_shape"):
        wm[f"{m}.{s}"] = mtp_shard
# F04: same pristine-backup guard for the MTP shard.
if not os.path.exists(d + mtp_shard + ".bak-orig"):
    os.replace(d + mtp_shard, d + mtp_shard + ".bak-orig")
else:
    os.remove(d + mtp_shard)
save_file(tensors, d + mtp_shard + ".tmp", metadata=mtp_meta or {"format": "pt"})
os.replace(d + mtp_shard + ".tmp", d + mtp_shard)
del tensors
# F04: atomic index publication (the commit point — replaced last).
_tmp_idx = d + "model.safetensors.index.json.tmp"
json.dump(idx, open(_tmp_idx, "w"), indent=2)
os.replace(_tmp_idx, d + "model.safetensors.index.json")

# ---- config.json ----
cfg_path = d + "config.json"
c = json.load(open(cfg_path))
json.dump(c, open(cfg_path + ".bak-quant.tmp", "w"), indent=2)
os.replace(cfg_path + ".bak-quant.tmp", cfg_path + ".bak-quant")
qc = c["quantization_config"]


def group(bits, targets):
    g = copy.deepcopy(qc["config_groups"]["group_0"])
    g["targets"] = targets
    w = g["weights"]
    w["num_bits"] = bits
    # tensors written here are symmetric with no zero point, regardless of
    # what the body group uses (AWQ bodies are asymmetric).
    w["symmetric"] = True
    w["zp_dtype"] = None
    w["group_size"] = GROUP
    w["strategy"] = "group"
    w["type"] = "int"
    return g


qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head" and i not in MTP_LINEARS]
qc["config_groups"]["group_1"] = group(HEAD_BITS, ["re:.*lm_head$"])
qc["config_groups"]["group_2"] = group(HEAD_BITS, ["re:.*embed_tokens$"])
qc["config_groups"]["group_3"] = group(
    MTP_BITS, ["re:^mtp\\.layers\\..*"] if KEEP_FC else ["re:^mtp\\..*"]
)
# F04: atomic config publication.
_tmp_cfg = d + "config.json.tmp"
json.dump(c, open(_tmp_cfg, "w"), indent=2)
os.replace(_tmp_cfg, d + "config.json")
print("done")
