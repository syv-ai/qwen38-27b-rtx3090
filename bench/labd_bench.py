"""What lookup-augmented drafting is for: long-context tasks that reproduce their input.

Six greedy tasks against a 25k-token document (reproduce verbatim, list the commands,
rewrite-but-keep, quote-and-explain, summarize, answer), measured the way this repo reports
decode — streamed, so the prefill is not averaged in — plus tokens per step from the
server's own spec-decode metrics.

The corpus is frozen on first run (~/bench/labd_corpus.txt) so numbers stay comparable as
the docs it is built from change. Run against a server started with SPEC=dflash2:

  venv/bin/python bench/labd_bench.py <tag> [--ctx 20000] [--max-tokens 512]
                                      [--tasks copy,code,edit,quote,summary,qa]
                                      [--corpus ~/bench/labd_corpus_long.txt]

--corpus points at a different document. The frozen one is ~84k tokens, so anything past
about --ctx 65000 silently measures a shorter prompt than you asked for; labd_corpus_long.txt
is the same text followed by vLLM source (varied, so a suffix lookup gets no free matches
from a repeated section) and its first 244,038 characters are byte-identical, which keeps
every number taken at --ctx 20000 comparable.

Corpus contract (F11): stream usage with completion_tokens is required — chunk
counts are never reported as tokens (missing usage exits 1). Metrics are summed
over engines/label series (F02).
"""
import glob
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
BASE = "http://127.0.0.1:18020"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


CORPUS = os.path.expanduser(arg("--corpus", "~/bench/labd_corpus.txt"))
CTX = int(arg("--ctx", 20000))
MAXTOK = int(arg("--max-tokens", 512))


def metrics():
    # Sum over engines/label series: overwriting instead of summing silently
    # drops all but the last series on multi-engine servers.
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
    d = {}
    for line in urllib.request.urlopen(req).read().decode().splitlines():
        for k in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(k + " ") or line.startswith(k + "{"):
                d[k] = d.get(k, 0.0) + float(line.split()[-1])
    return (d.get("vllm:spec_decode_num_drafts_total", 0.0),
            d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0))


if not os.path.exists(CORPUS):
    docs = [open(f).read() for f in sorted(glob.glob(os.path.expanduser("~/qwen-serving/*.md")) +
                                           glob.glob(os.path.expanduser("~/qwen-serving/*/README.md")))]
    text = "\n\n".join(docs)
    while len(text) < 200000:
        text += "\n\n" + "\n\n".join(docs)
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    open(CORPUS, "w").write(text)
_full = open(CORPUS).read()
_want = int(CTX * 3.6)
if len(_full) < _want:
    print(f"WARNING: {CORPUS} is {len(_full)} chars, --ctx {CTX} wants {_want}; "
          f"the prompt will be shorter than you asked for. Use --corpus "
          f"~/bench/labd_corpus_long.txt.", file=sys.stderr)
doc = _full[:_want]

ALL_TASKS = [
    ("copy", "Gengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."),
    ("code", "Gengiv alle bash-kommandoer fra teksten i én samlet liste, ordret."),
    ("edit", "Omskriv afsnittet om Docker så det er kortere, men behold alle kommandoer ordret."),
    ("quote", "Citér ordret de afsnit der handler om hukommelse og KV-cache, og forklar dem."),
    ("summary", "Skriv en grundig, struktureret opsummering på dansk."),
    ("qa", "Svar udførligt: hvilke optimeringer gav mest, og hvorfor?"),
]
want = arg("--tasks", "")
TASKS = [t for t in ALL_TASKS if not want or t[0] in want.split(",")]

# Warm-up: the first long-block step JIT-compiles Triton kernels, and those seconds would
# otherwise land inside the first task's decode window (worth 30% on it).
for warm in (64, 64):
    payload = {"model": "qwen3.8-27b",
               "messages": [{"role": "user", "content": "Dokument:\n\n" + doc[:4000] +
                             "\n\nGengiv ordret de første 10 linjer af dokumentet."}],
               "max_tokens": warm, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    urllib.request.urlopen(req, timeout=900).read()

tot = {"steps": 0.0, "acc": 0.0, "out": 0.0, "dec": 0.0}
rows = []
for name, q in TASKS:
    payload = {"model": "qwen3.8-27b",
               "messages": [{"role": "user", "content": "Dokument:\n\n" + doc + "\n\n" + q}],
               "max_tokens": MAXTOK, "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    d0, a0 = metrics()
    t0 = time.time()
    t_first = None
    n_chunks = 0
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
                if ch.get("delta", {}).get("content"):
                    if t_first is None:
                        t_first = time.time()
                    n_chunks += 1
    t_end = time.time()
    d1, a1 = metrics()
    steps = d1 - d0
    tps = 1 + (a1 - a0) / max(steps, 1)
    # F11: chunk count is not a token count. Usage is required (stream_options
    # include_usage is set above); fall back only to an explicit INVALID exit.
    if "completion_tokens" not in usage:
        print(f"LABD {TAG} {name:8s} RESULT FAIL: stream usage missing "
              f"completion_tokens — refusing to report chunk counts as tokens",
              flush=True)
        sys.exit(1)
    out = usage["completion_tokens"]
    ttft = (t_first or t_end) - t0
    decode = (out - 1) / max(t_end - (t_first or t_end), 1e-3)
    e2e = out / max(t_end - t0, 1e-3)
    rows.append((name, out, steps, tps, decode, e2e, ttft))
    tot["steps"] += steps
    tot["acc"] += a1 - a0
    tot["out"] += out
    tot["dec"] += t_end - (t_first or t_end)
    print(f"LABD {TAG} {name:8s} prompt={usage.get('prompt_tokens', 0):6d} out={out:4d} "
          f"steps={steps:5.0f} tok/step={tps:5.2f} decode={decode:6.1f} e2e={e2e:6.1f} "
          f"ttft={ttft:5.2f}s", flush=True)
print(f"LABD {TAG} TOTAL out={tot['out']:.0f} steps={tot['steps']:.0f} "
      f"tok/step={1 + tot['acc'] / max(tot['steps'], 1):.2f} "
      f"decode={tot['out'] / max(tot['dec'], 1e-3):.1f} tok/s")
os.makedirs(os.path.expanduser("~/bench/results"), exist_ok=True)
json.dump({"tag": TAG, "ctx": CTX, "rows": rows,
           "total_tok_per_step": 1 + tot["acc"] / max(tot["steps"], 1),
           "total_decode": tot["out"] / max(tot["dec"], 1e-3)},
          open(os.path.expanduser(f"~/bench/results/labd_{TAG}.json"), "w"), indent=1)
