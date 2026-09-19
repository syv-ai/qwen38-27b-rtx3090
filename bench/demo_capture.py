"""Record one lane of the side-by-side demo: every token with the time it arrived.

There is one GPU, so the two configurations cannot run at the same time. Each lane is
captured separately against a live server and the video replays them together at their
real recorded speed -- the timings are measured, the side-by-side is a replay.

  venv/bin/python bench/demo_capture.py <lane-name> [--out ~/bench/demo]

Run it once per lane, against a server started in that configuration:

  bash single-user/start_qwen.sh --no-spec-baseline       # or a plain `vllm serve`
  venv/bin/python bench/demo_capture.py baseline

  SPEC=dflash2 DFLASH_TOKENS=15 PREFIX_CACHE=1 bash single-user/start_qwen.sh
  venv/bin/python bench/demo_capture.py dflash2

Writes <out>/<lane>.json: per-prompt token arrival times relative to the first token,
so the renderer never has to guess. Decode rate excludes prefill, the same convention
the READMEs use.
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
BASE = os.environ.get("DEMO_BASE", "http://127.0.0.1:18020")
LANE = sys.argv[1] if len(sys.argv) > 1 else "lane"
OUT = os.path.expanduser(
    sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "~/bench/demo")

DOC_TOKENS = 25000
_corpus = os.path.expanduser("~/bench/labd_corpus.txt")
doc = open(_corpus).read()[: int(DOC_TOKENS * 2.9)] if os.path.exists(_corpus) else ""

# English throughout: the demo is the repo's front page, and the frame should be
# readable by anyone landing on it. No max_tokens -- each answer runs to its own
# stop token, so the video shows real completions rather than a truncation point
# chosen to flatter one lane.
PROMPTS = [
    ("chat", "Chat",
     "Explain briefly why 24 GB of VRAM is a hard constraint when serving a 27B "
     "model, and what can be done about it."),
    ("code", "Code",
     "Write a Python function that merges overlapping intervals. Include a short "
     "docstring and three test cases."),
    ("copy", "Reproduce a 25k-token document",
     "Document:\n\n" + doc + "\n\nReproduce the first 60 lines of the document "
     "verbatim. No commentary, just the text."),
]


def run(key, label, content, max_tokens=None):
    payload = {"model": "qwen3.8-27b",
               "messages": [{"role": "user", "content": content}],
               "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.time()
    first = None
    toks = []          # (ms since first token, text)
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            ev = json.loads(body)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                piece = ch.get("delta", {}).get("content")
                if piece:
                    now = time.time()
                    if first is None:
                        first = now
                    toks.append([round((now - first) * 1000, 1), piece])
    end = time.time()
    ttft = (first or end) - t0
    decode_s = end - (first or end)
    n = usage.get("completion_tokens", len(toks))
    rate = (n - 1) / decode_s if decode_s > 1e-3 else 0.0
    print(f"{LANE:10s} {key:6s} prompt={usage.get('prompt_tokens', 0):6d} out={n:4d} "
          f"ttft={ttft:6.2f}s decode={rate:6.1f} tok/s", flush=True)
    return {"key": key, "label": label, "tokens": toks, "n_out": n,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "ttft_s": round(ttft, 3), "decode_s": round(decode_s, 3),
            "decode_tok_s": round(rate, 1)}


# One short warm-up so Triton/graph JIT does not land inside the first measured prompt.
run("warm", "warmup", "Write one sentence about the weather.", max_tokens=32)

results = [run(*p) for p in PROMPTS]
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, f"{LANE}.json")
json.dump({"lane": LANE, "prompts": results}, open(path, "w"), indent=1)
print("wrote", path)
