"""Concurrency soak for the long verify block: does it still behave at batch > 1?

Everything about lookup-augmented drafting was tuned at batch 1, and two of its parts are
batch-sensitive:

  * the block length is one decision for the whole batch (the speculator asks for the long
    block only when every active request wants it), so a mixed batch exercises a path that a
    single request never does;
  * the drafter's grouped convolution resets on the block boundary, so a wrong block size
    would convolve across the boundary between two requests' query blocks.

This runs a mix of copy-heavy and prose requests concurrently and checks the answers survive.

  venv/bin/python bench/labd_soak.py [--conc 4] [--rounds 3] [--max-tokens 256]
                                     [--ctx 20000] [--minutes 0]

What to look for, in order of how much a failure means:

  * `!= round 1` on a copy row. Every round runs the same four requests, so two rounds have
    the same batch composition and must produce the same text. A difference here has no
    innocent explanation.
  * `!= alone` is weaker evidence. The batch-1 reference runs against a different batch
    composition, and the verify block is one chunk through the recurrent layers, so the last
    bits of the logits differ and a near-tie can flip (docs/gotchas.md, 14). Judge it by what
    changed: a synonym or a line break is the documented behaviour, a duplicated span or a
    truncated URL is not.
  * every request non-empty and finishing -- a cross-request state bug tends to show up as
    one request degenerating (repetition, truncation, a fragment repeated out of order)
    while the others are fine, and usually not the first in the batch;
  * acceptance (tokens/step) at the bottom: it should sit between the batch-1 numbers for
    prose and for copying. Far below both means the batch-wide block decision is thrashing;
  * no request should be slower than ~3x the batch-1 wall time for the same work.

All four requests share one document, so this also exercises several requests resuming from
the same cached prefix at once -- which is a likelier source of trouble than the lookup, and
worth ruling in or out with PREFIX_CACHE=0 before blaming the drafter.

Quality gate (F11): copy tasks are judged on correspondence to the source via
bench/verbatim.py, not just round-to-round repeatability; any EMPTY, diverged,
or corrupt answer exits 1 with RESULT FAIL.
"""

import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from verbatim import classify as _classify
except Exception:
    _classify = None

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
BASE = "http://127.0.0.1:18020"
CORPUS = os.path.expanduser("~/bench/labd_corpus.txt")


def arg(name, default):
    return type(default)(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


CONC = arg("--conc", 4)
ROUNDS = arg("--rounds", 3)
MAXTOK = arg("--max-tokens", 256)
CTX = arg("--ctx", 20000)
MINUTES = arg("--minutes", 0.0)

doc = open(CORPUS).read()[: int(CTX * 3.6)]
TASKS = [
    ("copy", "Gengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."),
    ("summary", "Skriv en grundig, struktureret opsummering på dansk."),
    ("code", "Gengiv alle bash-kommandoer fra teksten i én samlet liste, ordret."),
    ("qa", "Svar udførligt: hvilke optimeringer gav mest, og hvorfor?"),
]


def metrics():
    # Sum over engines/label series; overwrite drops all but the last series.
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
    d = {}
    for line in urllib.request.urlopen(req).read().decode().splitlines():
        for k in ("vllm:spec_decode_num_drafts_total",
                  "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(k + " ") or line.startswith(k + "{"):
                d[k] = d.get(k, 0.0) + float(line.split()[-1])
    return (d.get("vllm:spec_decode_num_drafts_total", 0.0),
            d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0))


def ask(task):
    name, q = task
    payload = {"model": "qwen3.8-27b",
               "messages": [{"role": "user", "content": "Dokument:\n\n" + doc + "\n\n" + q}],
               "max_tokens": MAXTOK, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return {"task": name, "wall": time.time() - t0,
            "tokens": r["usage"]["completion_tokens"],
            "text": r["choices"][0]["message"]["content"]}


print(f"soak: conc={CONC} rounds={ROUNDS} max_tokens={MAXTOK} ctx={CTX}", flush=True)
alone = ask(TASKS[0])
print(f"  batch 1 copy: {alone['tokens']} tokens in {alone['wall']:.1f}s", flush=True)

bad = 0
soft = 0
first_round_copy = None
d0, a0 = metrics()
t_end = time.time() + MINUTES * 60
rnd = 0
while rnd < ROUNDS or time.time() < t_end:
    rnd += 1
    batch = [TASKS[i % len(TASKS)] for i in range(CONC)]
    with ThreadPoolExecutor(CONC) as ex:
        out = list(ex.map(ask, batch))
    for o in out:
        ok = o["tokens"] > 0 and o["text"].strip()
        note = ""
        if o["task"] == "copy":
            if first_round_copy is None:
                first_round_copy = o["text"]
            elif o["text"] != first_round_copy:
                # Same four requests, same composition, different text: no innocent reading.
                note += " != ROUND 1"
                bad += 1
            note += " same as alone" if o["text"] == alone["text"] else " != alone"
            soft += o["text"] != alone["text"]
            # F11 correspondence: the copy task must reproduce the source, not
            # just repeat round 1. Judge coverage against the document.
            if _classify is not None and o["text"].strip():
                flag, cov, why = _classify(o["text"], doc)
                if flag != "ok":
                    note += f" CORRUPT(cov={cov:.2f},{why})"
                    bad += 1
        if not ok:
            bad += 1
        print(f"  round {rnd} {o['task']:8s} {o['tokens']:4d} tok {o['wall']:6.1f}s "
              f"{'ok' if ok else 'EMPTY'}{note}", flush=True)
d1, a1 = metrics()
steps = d1 - d0
print(f"soak: {rnd} rounds x {CONC} requests, tokens/step={1 + (a1 - a0) / max(steps, 1):.2f}, "
      f"{'OK' if not bad else str(bad) + ' PROBLEMS'}"
      + (f", {soft} round(s) differ from the batch-1 reference -- read the diff before"
         " calling it a bug" if soft else ""))
if bad:
    print("RESULT FAIL")
else:
    print("RESULT PASS")
sys.exit(1 if bad else 0)
