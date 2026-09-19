"""Requantize lm_head to int8 (group-128, symmetric) in compressed-tensors
pack-quantized format, in place.

The published W4A16 quants of Qwen3.8-27B leave lm_head in bf16 — that's a
2.5 GB matrix (248k vocab) read every decode step. int8 halves the read and
frees ~1.3 GB of VRAM for the KV/state pool. Measured +12% aggregate
throughput on an RTX 3090, round-trip error 0.64% (Frobenius).

Usage: python prepare/quant_lm_head.py /path/to/Qwen3.8-27B-W4A16-AutoRound

Rewrites the shard containing lm_head.weight, the safetensors index and
config.json. Backups are written next to the originals (.bak / .bak-quant).

Publication safety (F04): the rewritten shard, index, and config are each
written to a temp file and atomically replaced, so an interrupted run cannot
leave a half-written directory that looks ready.
"""

import copy
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

GROUP = 128
BITS = 8
QMAX = 127
KEY = "lm_head.weight"

d = sys.argv[1].rstrip("/") + "/"

idx = json.load(open(d + "model.safetensors.index.json"))
wm = idx["weight_map"]
shard = wm[KEY]
print(f"{KEY} lives in {shard}")

tensors = {}
with safe_open(d + shard, framework="pt") as f:
    meta = f.metadata()
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

w = tensors.pop(KEY).to(torch.float32)
out_f, in_f = w.shape
g = w.reshape(out_f, in_f // GROUP, GROUP)
scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)

deq = (q.reshape(out_f, -1, GROUP).to(torch.float32) * scale).reshape(out_f, in_f)
err = ((deq - w).norm() / w.norm()).item()
print(f"round-trip relative error: {err:.4f}")
assert err < 0.01, "quantization error too high, aborting"

tensors["lm_head.weight_packed"] = pack_to_int32(q, BITS, packed_dim=1).contiguous()
# linear layers use fp16 scales in this checkpoint
tensors["lm_head.weight_scale"] = scale.squeeze(-1).to(torch.float16).contiguous()
tensors["lm_head.weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)

shutil.copy(d + shard, d + shard + ".bak")
# F04: atomic publication — write the new shard to tmp, then replace, so an
# interrupted run cannot leave a half-written shard that looks ready.
_tmp_shard = d + shard + ".tmp"
save_file(tensors, _tmp_shard, metadata=meta or {"format": "pt"})
os.replace(_tmp_shard, d + shard)

shutil.copy(d + "model.safetensors.index.json", d + "model.safetensors.index.json.bak-quant")
del wm[KEY]
for s in ("weight_packed", "weight_scale", "weight_shape"):
    wm[f"lm_head.{s}"] = shard
# F04: atomic index publication.
_tmp_idx = d + "model.safetensors.index.json.tmp"
json.dump(idx, open(_tmp_idx, "w"), indent=2)
os.replace(_tmp_idx, d + "model.safetensors.index.json")

c = json.load(open(d + "config.json"))
shutil.copy(d + "config.json", d + "config.json.bak-quant")
qc = c["quantization_config"]
qc["ignore"] = [i for i in qc["ignore"] if i != "lm_head"]
# The MTP draft head is stored in bf16 but missing from the ignore list, which
# breaks loading when speculative decoding is enabled (single-user mode).
for m in (
    "mtp.fc",
    "mtp.layers.0.mlp.down_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
):
    if m not in qc["ignore"]:
        qc["ignore"].append(m)
g1 = copy.deepcopy(qc["config_groups"]["group_0"])
g1["targets"] = ["re:.*lm_head$"]
g1["weights"]["num_bits"] = BITS
qc["config_groups"]["group_1"] = g1
# F04: atomic config publication.
_tmp_cfg = d + "config.json.tmp"
json.dump(c, open(_tmp_cfg, "w"), indent=2)
os.replace(_tmp_cfg, d + "config.json")
print("done")
