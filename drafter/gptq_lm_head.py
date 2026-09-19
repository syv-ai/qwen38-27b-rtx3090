"""GPTQ-quantize lm_head to int4 (g128, sym) using captured final hidden states as
calibration, and write a model-dir variant (copied shards).

  python gptq_lm_head.py <src_model_dir> <dst_dir> [--bits 4] [--calib-rows 400000] [--mse-clip]
Also reports the KL(bf16 head || quantized head) on held-out hidden states for RTN vs GPTQ.

Write safety (F04): variant artifacts are COPIED, never hardlinked.
build_draft_vocab.py rewrites the extras shard in place; a hardlink would
mutate the SOURCE model dir through the shared inode.
"""
import json, os, sys, shutil, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_utils import accumulate_hessian, gptq_quantize, rtn_quantize, dequant

S = sys.argv[1].rstrip("/") + "/"; D = sys.argv[2].rstrip("/") + "/"
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 4
NCAL = int(sys.argv[sys.argv.index("--calib-rows") + 1]) if "--calib-rows" in sys.argv else 400000
MSE = "--mse-clip" in sys.argv
DATA = os.path.join(HERE, "data")
GROUP = 128
dev = "cuda"

idx = json.load(open(S + "model.safetensors.index.json")); wm = idx["weight_map"]
shard = wm["lm_head.weight_packed"]
with safe_open(S + shard + ".bak", "pt") as f:
    W = f.get_tensor("lm_head.weight").to(dev)      # bf16 [V,K]
V, K = W.shape

hid = np.load(f"{DATA}/hidden.npy", mmap_mode="r")
T = hid.shape[0]
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(T, size=min(NCAL + 20000, T), replace=False))
cal, held = rows[:NCAL], rows[NCAL:]
H = torch.zeros(K, K, device=dev); n_seen = 0
t0 = time.time()
for a in range(0, len(cal), 32768):
    x = torch.from_numpy(np.array(hid[cal[a:a + 32768]])).view(torch.bfloat16).to(dev)
    H, n_seen = accumulate_hessian(H, x, n_seen)
print(f"hessian from {n_seen} rows in {time.time()-t0:.0f}s")
xh = torch.from_numpy(np.array(hid[held])).view(torch.bfloat16).to(dev)


def kl_eval(Wq):
    tot = 0.0; n = 0
    for a in range(0, xh.shape[0], 1024):
        x = xh[a:a + 1024]
        lp = torch.log_softmax((x @ W.t()).float(), -1)
        lq = torch.log_softmax((x @ Wq.to(torch.bfloat16).t()).float(), -1)
        tot += (lp.exp() * (lp - lq)).sum().item(); n += x.shape[0]
    return tot / n


q_rtn, s_rtn = rtn_quantize(W, BITS, GROUP)
print(f"RTN int{BITS}: KL {kl_eval(dequant(q_rtn, s_rtn)):.5f}")
del q_rtn, s_rtn; torch.cuda.empty_cache()
t0 = time.time()
qs, ss = [], []
for r0 in range(0, V, 16384):          # rows are independent under GPTQ; chunk to bound memory
    q_, s_ = gptq_quantize(W[r0:r0 + 16384], H, bits=BITS, group=GROUP, blocksize=GROUP, mse_clip=MSE)
    qs.append(q_.cpu()); ss.append(s_.cpu()); del q_, s_
    torch.cuda.empty_cache()
q, s = torch.cat(qs).to(dev), torch.cat(ss).to(dev)
dq = dequant(q, s)
print(f"GPTQ int{BITS}{' +mse' if MSE else ''}: KL {kl_eval(dq):.5f}  ({time.time()-t0:.0f}s)")
rel = ((dq - W.float()).norm() / W.float().norm()).item(); del dq; torch.cuda.empty_cache()
print(f"GPTQ round-trip rel error {rel:.4f}")

# ---- write variant dir
# NOTE (F04): never hardlink variant artifacts. build_draft_vocab.py rewrites
# model_extra_tensors.safetensors and the lm_head shard in place; a hardlink
# would mutate the SOURCE model dir through the shared inode. Copy instead.
os.makedirs(D, exist_ok=True)
for f in os.listdir(S):
    if f.startswith("model-0000") and f.endswith(".safetensors") and f != shard and not os.path.exists(D + f):
        shutil.copy(S + f, D + f)
for f in ["tokenizer.json", "model_extra_tensors.safetensors", "mtp_draft_vocab_ids.pt"]:
    if not os.path.exists(D + f):
        shutil.copy(S + f, D + f)
for f in ["chat_template.jinja", "generation_config.json", "processor_config.json", "quantization_config.json", "tokenizer_config.json"]:
    shutil.copy(S + f, D + f)
tensors = {}
with safe_open(S + shard, "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        if not k.startswith("lm_head."):
            tensors[k] = f.get_tensor(k)
tensors["lm_head.weight_packed"] = pack_to_int32(q.cpu(), BITS, packed_dim=1).contiguous()
tensors["lm_head.weight_scale"] = s.to(torch.float16).cpu().contiguous()
tensors["lm_head.weight_shape"] = torch.tensor([V, K], dtype=torch.int64)
if os.path.exists(D + shard):
    os.remove(D + shard)
save_file(tensors, D + shard, metadata=meta or {"format": "pt"})
json.dump(idx, open(D + "model.safetensors.index.json", "w"), indent=2)
c = json.load(open(S + "config.json"))
c["quantization_config"]["config_groups"]["group_1"]["weights"]["num_bits"] = BITS
json.dump(c, open(D + "config.json", "w"), indent=2)
print("wrote", D, "(draft head still needs prepare/build_draft_vocab.py --ids if lm_head bits changed)")
