"""Bug B: sweep prompt length and report exact prompt tokens, acceptance, and whether
the verbatim answer is correct.

  venv/bin/python bench/bugb_sweep.py 19500 19800 20000 20200 ...

What it found, before a75ee4b: against SPEC=dflash2 CTX=huge PREFIX_CACHE=1 with a
CAPTURED (FULL) verify step, one broken length in every 128 -- every prompt length
==124 (mod 128) collapsed to 1.97 tok/step and degenerate repetition while every other
residue was clean. See docs/gotchas.md gotcha 37. The `mod128` column is the point of
the script. That residue is fixed and dflash2 runs FULL again; the one still open is
SPEC=mtp CTX=huge under FULL at residue 4, which is why that combination is pinned to
PIECEWISE.

Sweep in steps of 1 token near a broken length, not in steps of 100: at a coarse grid
this reads as a threshold, which is how it was first (mis)diagnosed.

To sweep every residue rather than lengths you picked, use bench/residue_sweep.py --
five hand-picked lengths miss a 1-in-128 break 96% of the time.

Fail-closed (F02): any broken length exits 1 with RESULT FAIL.
"""
import json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from verbatim import classify, median, prefix_match, repeats  # noqa: E402


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
BASE = "http://127.0.0.1:" + os.environ.get("PORT", "18020")
DOC = open(os.path.expanduser(os.environ.get("CORPUS", "~/bench/labd_corpus.txt"))).read()

from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained(os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound-fast"))


def metrics():
    # Sum over engines/label series (see labd_bench.py): overwrite drops series.
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
    d = {}
    for line in urllib.request.urlopen(req).read().decode().splitlines():
        for k in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"):
            if line.startswith(k + " ") or line.startswith(k + "{"):
                d[k] = d.get(k, 0.0) + float(line.split()[-1])
    return d.get("vllm:spec_decode_num_drafts_total", 0.0), d.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)


print(f"{'ctx':>7} {'prompt tok':>11} {'mod128':>7} {'tok/step':>9} {'verbatim':>12} "
      f"{'cov':>5} {'repeats':>8}")
rows = []
for ctx in [int(a) for a in sys.argv[1:]]:
    doc = DOC[: int(ctx * 3.6)]
    content = ("Dokument:\n\n" + doc +
               "\n\nGengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten.")
    # The ENGINE sees the chat-templated prompt, not this string: the Qwen3 wrapper
    # with enable_thinking=false is +12 tokens. Measuring the raw content was how the
    # residue got misread as "R = 117 + k" and then as a mysterious constant 12 -- the
    # real rule is just (templated length) == L (mod 128). Never report the raw count.
    ptok = len(TOK.encode(TOK.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
    body = json.dumps({"model": "qwen3.8-27b",
                       "messages": [{"role": "user", "content": content}],
                       "max_tokens": 400, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    d0 = metrics()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}),
        timeout=1200).read().decode())
    d1 = metrics()
    # `or ""`: a collapse-to-stop returns content=null, and indexing that raised a
    # TypeError that ended the sweep silently -- which is how the mtp failure stayed
    # invisible through several full runs (issue #25, mjungnickel18).
    ans = r["choices"][0]["message"].get("content") or ""
    dn = d1[0] - d0[0]
    tps = (d1[1] - d0[1]) / dn + 1 if dn else float("nan")
    # Judge on the OUTPUT, not on an absolute tok/step: the ceiling is the verify
    # block, so a healthy k=3 run sits at 3.96 and an absolute threshold calls it
    # broken. And judge by COVERAGE against the other lengths in the sweep rather than
    # by any particular failure signature -- the same residue has come back empty, one
    # character long, and 400 fluent tokens that invent a task, and a rule written for
    # one of those shapes files the others as healthy (bench/verbatim.py).
    n, rep = prefix_match(ans, doc), repeats(ans)
    ref = median([r["cov"] for r in rows]) if len(rows) >= 5 else None
    flag, cov, why = classify(ans, doc, ref=ref)
    rows.append(dict(ctx=ctx, ptok=ptok, tps=tps, ans=ans, doc=doc, cov=cov))
    print(f"{ctx:>7} {ptok:>11} {ptok % 128:>7} {tps:>9.2f} "
          f"{str(n) + '/' + str(len(ans)):>12} {cov:>5.2f} {rep:>8}  {flag}"
          + (f"  ({why})" if why else ""))

if len(rows) >= 5:
    # Second pass now that the whole neighbourhood exists: a length judged against the
    # first four rows was judged against too little.
    ref = median([r["cov"] for r in rows])
    bad = [(r["ptok"] % 128, classify(r["ans"], r["doc"], ref=ref))
           for r in rows]
    bad = [(res, why) for res, (flag, _, why) in bad if flag != "ok"]
    print(f"\nneighbourhood coverage (median of {len(rows)}): {ref:.2f}")
    print(f"{len(bad)} broken of {len(rows)} lengths"
          + ("" if not bad else "  -> " + ", ".join(f"residue {r} ({w})" for r, w in bad)))
# Fail-closed: broken lengths must fail the process for CI.
if len(rows) >= 5 and bad:
    print("RESULT FAIL")
    sys.exit(1)
print("RESULT PASS")
