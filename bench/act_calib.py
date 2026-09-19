#!/usr/bin/env python3
"""Measure, per linear layer, how much per-token int8 activation quantization
perturbs that layer's output (relative L2 error of y=Wx vs y=W·q(x)) on a
small calibration corpus. Runs vLLM in-process (no multiprocessing) with the
plain W4A16 kernels and forward pre-hooks. Output: JSON {layer_name: err} and
a sorted listing.
Use it to pick INT8_LAYERS for batch/start_qwen.sh: layers with small error are
safe to run with int8 activations, the rest cost perplexity. On Qwen3.8-27B the
GDN in_proj (early layers) and down_proj (last layers) are the worst.
 Usage: PATH=venv/bin:$PATH VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
        python bench/act_calib.py models/Qwen3.8-27B-W4A16-AutoRound act_calib.json

Split isolation (F08): calibrates on TRAIN splits only — the wikitext-2,
fineweb-2, and GSM8K TEST splits are the held-out evaluation sets
quality_battery.py scores, so any calibration path containing 'test' is
refused rather than silently contaminating the weights.
"""
import json, os, sys, glob, re, torch
import pyarrow.parquet as pq

model_path, out_path = sys.argv[1], sys.argv[2]
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.pop("VLLM_MARLIN_INPUT_DTYPE", None)
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams

llm = LLM(model=model_path, served_model_name="q", gpu_memory_utilization=0.85, max_model_len=2048,
          max_num_seqs=4, enforce_eager=True, language_model_only=True, kv_cache_dtype="fp8",
          mamba_ssm_cache_dtype="float16", max_num_batched_tokens=2048)

# find the torch model
def find_model(obj, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if depth > 14 or id(obj) in seen: return None
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        if any(n.endswith("gate_up_proj") for n, _ in obj.named_modules()): return obj
        return None
    for attr in ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model", "engine", "_model", "runner"):
        if hasattr(obj, attr):
            try: child = getattr(obj, attr)
            except Exception: continue
            r = find_model(child, depth+1, seen)
            if r is not None: return r
    if hasattr(obj, "get_model"):
        try:
            r = obj.get_model()
            if isinstance(r, torch.nn.Module): return r
        except Exception: pass
    return None
model = find_model(llm)
if model is None:
    # last resort: collective_rpc into the worker
    try:
        model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.get_model()
    except Exception as e:
        print("fallback failed:", e)
assert model is not None, "could not find model"
print("model:", type(model).__name__, flush=True)

pat = re.compile(r"(gate_up_proj|down_proj|in_proj_qkvz|out_proj|qkv_proj|o_proj)$")
stats = {}
def make_hook(name, mod):
    def hook(m, inputs):
        x = inputs[0]
        x2 = x.reshape(-1, x.shape[-1])
        n = min(x2.shape[0], 64)
        idx = torch.linspace(0, x2.shape[0]-1, n).long().to(x2.device) if x2.shape[0] > n else torch.arange(x2.shape[0], device=x2.device)
        xs = x2[idx]
        scale = xs.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        q = (torch.round(xs / scale).clamp(-128, 127) * scale).to(xs.dtype)
        with torch.no_grad():
            m._calib_off = True
            y = m(xs); yq = m(q)
            m._calib_off = False
            if isinstance(y, tuple): y = y[0]; yq = yq[0]
            err = ((yq.float() - y.float()).norm() / y.float().norm().clamp(min=1e-8)).item()
            xerr = ((q.float() - xs.float()).norm() / xs.float().norm().clamp(min=1e-8)).item()
        s = stats.setdefault(name, [0.0, 0.0, 0]); s[0] += err; s[1] += xerr; s[2] += 1
    def guarded(m, inputs):
        if getattr(m, "_calib_off", False): return
        return hook(m, inputs)
    return guarded

count = 0
for name, mod in model.named_modules():
    if pat.search(name) and "mtp" not in name and "lm_head" not in name and hasattr(mod, "forward") and "Linear" in type(mod).__name__:
        mod.register_forward_pre_hook(make_hook(name, mod)); count += 1
print("hooked layers:", count, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.environ.get("QUALITY_DATA", os.path.join(HERE, "quality-data"))
# F08: calibration must not consume evaluation test data. quality_battery.py
# scores wikitext-2 TEST, fineweb-2 dan_Latn TEST, and GSM8K TEST, so this
# harness calibrates on TRAIN splits only. Override with ACT_CALIB_* env vars;
# any path containing 'test' is refused.
def _refuse_test(path):
    if "test" in path.lower():
        print(f"REFUSING calibration from evaluation path: {path}", file=sys.stderr)
        sys.exit(1)
    return path
WIKI = os.environ.get("ACT_CALIB_WIKI", f"{Q}/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet")
FW2 = os.environ.get("ACT_CALIB_FINEWEB", f"{Q}/fineweb2/data/dan_Latn/train/000_00000.parquet")
_refuse_test(WIKI); _refuse_test(FW2)
texts = []
t = "".join(pq.read_table(_refuse_test(WIKI)).column("text").to_pylist())
texts += [t[i:i+3000] for i in range(0, 8*3000, 3000)]
tb = pq.read_table(_refuse_test(FW2), columns=["text"]).column("text").to_pylist()
texts += [d[:3000] for d in tb if len(d) > 3000][:8]
code = sorted(glob.glob(os.path.join(HERE, "..", "venv/lib/python3.12/site-packages/vllm/v1/core/*.py")))
texts += [open(f).read()[:3000] for f in code[:6]]
# chat-formatted prompts too
for p in [json.loads(l)["prompt"] for l in open(os.path.join(HERE, "prompts_real.jsonl"))]:
    texts.append(p)
texts = [t for t in texts if t and t.strip()]
print("calibration texts:", len(texts), flush=True)
sp = SamplingParams(max_tokens=32, temperature=0)
llm.generate(texts, sp)
res = {k: {"out_err": v[0]/v[2], "act_err": v[1]/v[2], "n": v[2]} for k, v in stats.items()}
json.dump(res, open(out_path, "w"), indent=1)
rows = sorted(res.items(), key=lambda kv: -kv[1]["out_err"])
print("worst 25 layers by relative output error from per-token int8 activations:")
for k, v in rows[:25]: print(f"  {v['out_err']:.4f}  (act {v['act_err']:.4f})  {k}")
import statistics
for kind in ["gate_up_proj", "down_proj", "in_proj_qkvz", "out_proj", "qkv_proj", "o_proj"]:
    vals = [v["out_err"] for k, v in res.items() if k.endswith(kind)]
    if vals: print(f"{kind}: n={len(vals)} median={statistics.median(vals):.4f} max={max(vals):.4f}")
