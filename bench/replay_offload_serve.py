"""replay_offload_serve.py: does the CPU offload tier ever SERVE a stored request back? (syv-ai #95 oracle)

Set VLLM_API_KEY and DEPTH_CORPUS_GLOB. Run against a server booted with --kv-offloading-size and a GPU pool
SMALLER than the tier, so a request is evicted from GPU but fits the tier; then it can only come back by a load.

Sequence against a running server:
  1. request X (a fixed, unsalted prompt of DEPTH chars): cold prefill; record TTFT and the CPU->GPU transfer counter.
  2. requests Y1..Yn (fresh prompts, same depth): push X out of the GPU prefix cache into the CPU tier (or out entirely).
  3. request X again: if the tier serves, TTFT is a fraction of (1) and the CPU->GPU counter grows; if the tier stores
     but never serves (the #52735 defect), TTFT is a full re-prefill and the counter stays put.

    TIER_GIB=<--kv-offloading-size> python replay.py TAG PORT DEPTH_CHARS N_EVICTORS
Prints one JSON line per request and a final verdict line.

Sizing rule (the verdict is meaningless otherwise): the GPU pool must be smaller than the tier IN TOKENS, and the
evictor traffic must exceed the GPU pool while staying under the tier. A token costs several times more in the tier
than on the GPU here (37 KB vs 124 KB on a 4090 with DFlash2 k=7 fp8; 6.7 KB vs 80 KB on a 3090 with MTP; these are
per-configuration ratios, and the GPU pool does not scale linearly with KV_MEM on this model, so read both from the
boot log and the metrics rather than computing them), so a tier that is 3x the pool in bytes can be smaller than it in tokens. If the stores overflow the tier, X is evicted from the tier
before X-again and the run reads exactly like the #52735 veto (ratio ~1, zero loads, zero external hits). With
TIER_GIB set the script refuses that verdict and prints INVALID-TIER-OVERFLOW instead.
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
TAG, PORT, DEPTH, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
KEY = os.environ.get("VLLM_API_KEY", "")
CORPUS = os.environ.get("DEPTH_CORPUS_GLOB", "*.txt")  # any glob of plain-text files to build fresh prompts from
TEXT = "\n\n".join(open(f, encoding="utf-8").read() for f in sorted(glob.glob(CORPUS)))
while len(TEXT) < (N + 2) * DEPTH + 100000:
    TEXT += "\n\n" + TEXT
H = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}


def metrics():
    r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/metrics", headers=H), timeout=30).read().decode()
    out = {}
    for line in r.splitlines():
        if (line.startswith("vllm:kv_offload") or line.startswith("vllm:prefix_cache") or line.startswith("vllm:external_prefix_cache")) and "_bucket" not in line and "_created" not in line:
            m = re.match(r'(\S+?)(\{[^}]*\})? ([0-9.e+-]+)$', line)
            if m:
                out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out


def ask(label, prompt, max_tokens=32):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
    t0 = time.time()
    resp = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body.encode(), headers=H), timeout=None)
    first = None
    usage = None
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        try:
            j = json.loads(line[5:].strip())
        except Exception:
            continue
        if first is None and any((c.get("delta") or {}).get("content") for c in j.get("choices", [])):
            first = time.time()
        if j.get("usage"):
            usage = j["usage"]
    ttft = round((first or time.time()) - t0, 2)
    m = metrics()
    served = {k.replace("{engine=\"0\",model_name=\"qwen3.8-27b\",", "{"): v for k, v in m.items() if "total_bytes" in k or "load" in k or "store" in k or "usage_perc" in k or "skipped" in k or "lookup" in k or "prefix_cache" in k}
    rec = {"tag": TAG, "req": label, "ttft_s": ttft, "total_s": round(time.time() - t0, 1),
           "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached": ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
           "offload": served}
    print(json.dumps(rec), flush=True)
    return rec


def fixed(i):
    start = i * DEPTH
    return (f"[replay {TAG} #{i}] Read the following document, then answer.\n\n" + TEXT[start:start + DEPTH]
            + "\n\nIn one sentence, what is the main theme of the document above?")


def store_gb(m):
    return sum(v for k, v in m.items() if "bytes" in k.lower() and ("store" in k.lower() or "GPU_to_CPU" in k)) / 1e9


m0 = metrics()  # counters are cumulative for the life of the server; the guard needs the stores of THIS run
x1 = ask("X-cold", fixed(0))
for i in range(1, N + 1):
    ask(f"Y{i}", fixed(i))
x2 = ask("X-again", fixed(0))
ratio = x2["ttft_s"] / max(x1["ttft_s"], 0.01)
loads = {k: v for k, v in x2["offload"].items() if "CPU_to_GPU" in k or "load" in k.lower()}
stores_gb = store_gb(metrics()) - store_gb(m0)
TIER_GIB = float(os.environ.get("TIER_GIB", "0") or 0)


def verdict(ratio, stores_gb, tier_gib):
    """SERVED / NOT-SERVED, or INVALID-TIER-OVERFLOW when the stores could not have fit the tier."""
    if tier_gib and stores_gb > tier_gib * 1.073741824:
        return "INVALID-TIER-OVERFLOW"
    return "SERVED" if ratio < 0.5 else "NOT-SERVED"


V = verdict(ratio, stores_gb, TIER_GIB)
if V == "INVALID-TIER-OVERFLOW":
    print(f"INVALID: {stores_gb:.2f} GB stored into a {TIER_GIB:g} GiB tier; X left the tier before X-again, so this run cannot "
          "tell a veto from an eviction. Shrink the GPU pool or the evictor count, or grow the tier.", file=sys.stderr)
print(json.dumps({"tag": TAG, "verdict": V, "stores_gb": round(stores_gb, 2), "tier_gib": TIER_GIB, "cold_ttft_s": x1["ttft_s"], "again_ttft_s": x2["ttft_s"], "ratio": round(ratio, 2), "cached_again": x2["cached"], "loads": loads}), flush=True)
if V == "INVALID-TIER-OVERFLOW":
    sys.exit(2)
