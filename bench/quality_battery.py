#!/usr/bin/env python3
"""Quick quality battery against the running server. Catches "benchmarks great,
outputs garbage" in a minute — run it after every kernel/quant change.
  1. perplexity over ~300-token windows: wikitext-2 test (en), fineweb-2 dan test (da),
     vLLM's own python source (code)
  2. GSM8K exact-match, 200 test questions, thinking off, greedy
Data (once):
  hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test-*" --local-dir bench/quality-data/wikitext
  hf download openai/gsm8k --repo-type dataset --include "main/test-*" --local-dir bench/quality-data/gsm8k
  hf download HuggingFaceFW/fineweb-2 --repo-type dataset --include "data/dan_Latn/test/000_00000.parquet" --local-dir bench/quality-data/fineweb2
Perplexity needs prompt_logprobs, which needs memory headroom: run the server with
GPU_UTIL=0.93 for this (gotcha 10 in the README).

On CTX=huge (KVarN) with SPEC=mtp, measure perplexity with PREFIX_CACHE=0. That
combination corrupts prompt_logprobs: some requests 400 with "Out of range float
values are not JSON compliant: nan", and the batteries that do complete read
~23% high on English (12.6-13.7 against 10.76) and drift run to run. Setting
PREFIX_CACHE=0 makes the same server exact and stable to four decimals. SPEC=off
and SPEC=dflash2 on KVarN are unaffected, at the same pool size, and so is
CTX=fast on all three settings, so the published quality tables are not
implicated. Measured in #64 (gotcha 46).
Usage: python bench/quality_battery.py <tag> [--ppl-only] [--gsm-only] [--gsm-n 200]
       [--min-acc 0.9] [--max-ppl 9.0]

Quality gate (F11): every PPL window must contribute finite logprobs, all three
domains must be present, GSM8K retains per-item predictions (pred/gold/ok) for
paired error analysis, and any incompleteness exits 1 with RESULT FAIL.
--min-acc/--max-ppl enforce predeclared promotion thresholds when passed.
"""
import json, os, sys, glob, re, math, time, random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""
KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(HERE, "..", "api_key.txt"))
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")
# data dir: wikitext-2 test parquet, fineweb-2 dan_Latn test parquet, gsm8k test parquet (see README)
Q = os.environ.get("QUALITY_DATA", os.path.join(HERE, "quality-data"))
tag = sys.argv[1]
ppl_only = "--ppl-only" in sys.argv; gsm_only = "--gsm-only" in sys.argv
gsm_n = int(sys.argv[sys.argv.index("--gsm-n")+1]) if "--gsm-n" in sys.argv else 200

def post(path, payload, timeout=1200):
    req = urllib.request.Request(API+path, data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def docs():
    out = []
    t = pq.read_table(f"{Q}/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet").column("text").to_pylist()
    txt = "".join(t); 
    for i in range(0, min(len(txt), 40*1200), 1200): out.append(("en", txt[i:i+1200]))
    tb = pq.read_table(f"{Q}/fineweb2/data/dan_Latn/test/000_00000.parquet", columns=["text"]).column("text").to_pylist()
    random.Random(0).shuffle(tb)
    n=0
    for d in tb:
        if len(d) > 1500:
            out.append(("da", d[:1200])); n+=1
        if n>=40: break
    files = sorted(glob.glob(os.path.join(HERE, "..", "venv/lib/python3.12/site-packages/vllm/v1/core/*.py")))
    for f in files:
        s = open(f).read()
        for i in range(0, min(len(s), 4*1200), 1200):
            if len(s[i:i+1200]) > 800: out.append(("code", s[i:i+1200]))
        if sum(1 for l,_ in out if l=="code") >= 40: break
    return out

def ppl_one(item):
    lang, text = item
    r = post("/completions", {"model":"qwen3.8-27b","prompt":text,"max_tokens":1,"temperature":0,
                              "prompt_logprobs":0,"echo":False})
    # F11: missing prompt_logprobs used to be silently skipped, shrinking the
    # measured corpus without a trace. Every window must contribute finite
    # logprobs or the run is INVALID.
    try:
        pl = r["choices"][0]["prompt_logprobs"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"ppl window missing prompt_logprobs ({lang}): {str(e)[:100]}")
    if not isinstance(pl, list) or len(pl) < 2:
        raise RuntimeError(f"ppl window has no scored tokens ({lang})")
    lps = []
    for e in pl[1:]:
        if e is None:
            continue
        v = list(e.values())[0]
        lp = v["logprob"] if isinstance(v, dict) else v
        if not isinstance(lp, (int, float)) or not math.isfinite(lp):
            raise RuntimeError(f"ppl window has nonfinite logprob ({lang}): {lp!r}")
        lps.append(lp)
    if not lps:
        raise RuntimeError(f"ppl window contributed zero tokens ({lang})")
    return lang, sum(lps), len(lps)

def run_ppl():
    items = docs()
    # F11: frozen-domain coverage — every lane must be present or the
    # aggregate is not comparable to published tables.
    have = {lang for lang, _ in items}
    missing = {"en", "da", "code"} - have
    if missing:
        raise RuntimeError(f"ppl corpus missing domains: {sorted(missing)}")
    rows = []
    def _one(item):
        lang, s, n = ppl_one(item)
        rows.append({"lang": lang, "logprob_sum": s, "tokens": n})
        return lang, s, n
    with ThreadPoolExecutor(2) as ex: res = list(ex.map(_one, items))
    agg = {}
    for lang, s, n in res:
        a = agg.setdefault(lang, [0.0, 0]); a[0]+=s; a[1]+=n
    for lang, (s, n) in agg.items():
        if n <= 0:
            raise RuntimeError(f"ppl domain {lang} contributed zero tokens")
    out = {lang: (math.exp(-s/n), n) for lang,(s,n) in agg.items()}
    tot_s = sum(s for s,n in agg.values()); tot_n = sum(n for s,n in agg.values())
    out["all"] = (math.exp(-tot_s/tot_n), tot_n)
    return out, rows

def extract_num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s.replace("$",""))
    return m[-1].replace(",","") if m else None

def gsm_one(row):
    q, a = row
    gold = a.split("####")[-1].strip().replace(",","")
    try:
        r = post("/chat/completions", {"model":"qwen3.8-27b","messages":[{"role":"user","content":q+"\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."}],
            "max_tokens":768,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}})
        txt = r["choices"][0]["message"]["content"] or ""
        m = re.search(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", txt)
        pred = (m.group(1).replace(",","") if m else extract_num(txt))
        try: ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
        except Exception: ok = False
        # F11: per-item evidence for paired error analysis (was aggregate-only).
        return {"ok": ok, "pred": pred, "gold": gold,
                "completion_tokens": r["usage"]["completion_tokens"]}
    except Exception as e:
        return {"ok": False, "pred": None, "gold": gold,
                "completion_tokens": 0, "error": f"{type(e).__name__}: {str(e)[:120]}"}

def run_gsm():
    t = pq.read_table(f"{Q}/gsm8k/main/test-00000-of-00001.parquet")
    rows = list(zip(t.column("question").to_pylist(), t.column("answer").to_pylist()))[:gsm_n]
    if len(rows) < gsm_n:
        raise RuntimeError(f"GSM8K: want {gsm_n} rows, test split has {len(rows)}")
    with ThreadPoolExecutor(32) as ex: res = list(ex.map(gsm_one, rows))
    if len(res) != len(rows):
        raise RuntimeError("GSM8K: incomplete coverage")
    acc = sum(1 for r in res if r["ok"])/len(res)
    toks = sum(r["completion_tokens"] for r in res)/len(res)
    return acc, toks, res

def _opt(name, default=None):
    return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default

t0=time.time(); result={"tag":tag}
failures = []
try:
    if not gsm_only:
        result["ppl"], result["ppl_rows"] = run_ppl()
        print(tag, "PPL", {k:(round(v[0],4),v[1]) for k,v in result["ppl"].items()}, f"{time.time()-t0:.0f}s", flush=True)
    if not ppl_only:
        t1=time.time(); acc, toks, gsm_rows = run_gsm(); result["gsm8k"]={"n":gsm_n,"acc":acc,"mean_tokens":toks,"rows":gsm_rows}
        print(tag, f"GSM8K n={gsm_n} acc={acc:.3f} mean_tokens={toks:.0f} {time.time()-t1:.0f}s", flush=True)
    # F11: predeclared promotion thresholds, enforced only when passed.
    min_acc = _opt("--min-acc"); max_ppl = _opt("--max-ppl")
    if min_acc is not None and not ppl_only and result["gsm8k"]["acc"] < float(min_acc):
        failures.append(f"gsm acc {result['gsm8k']['acc']:.3f} < --min-acc {min_acc}")
    if max_ppl is not None and not gsm_only and result["ppl"]["all"][0] > float(max_ppl):
        failures.append(f"ppl {result['ppl']['all'][0]:.4f} > --max-ppl {max_ppl}")
except Exception as e:
    failures.append(f"INVALID: {type(e).__name__}: {e}")
with open(os.path.join(Q, f"result_{tag}.json"),"w") as f: json.dump(result,f)
if failures:
    print(tag, "RESULT FAIL:", "; ".join(failures), flush=True)
    sys.exit(1)
print(tag, "RESULT PASS", flush=True)
