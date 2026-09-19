"""API feature smoke test against the running server: the request-level features a different
model runner could break (logprobs, n, stop, seeds, structured outputs, penalties, streaming,
thinking, prompt_logprobs, a 20k-token prompt). Prints PASS/FAIL per feature.

  venv/bin/python bench/api_smoke.py          # key from api_key.txt or VLLM_API_KEY, PORT=18020

Fail-closed (F02): any hard FAIL exits 1 with RESULT FAIL. The
thinking_token_budget probe is soft (WARN): this build silently ignores the
param (HTTP 200, measured 2026-09-08) instead of 400, which is recorded but
does not red the suite; re-harden if the server starts validating.
"""
import json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""
KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
PORT = os.environ.get("PORT", "18020")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
URLC = f"http://127.0.0.1:{PORT}/v1/completions"


def post(url, payload, stream=False):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    r = urllib.request.urlopen(req, timeout=600)
    if stream:
        return r.read().decode()
    return json.loads(r.read())


def chat(msg, **kw):
    p = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": msg}], "max_tokens": 64,
         "chat_template_kwargs": {"enable_thinking": False}}
    p.update(kw)
    return post(URL, p)


results = []
def check(name, fn, soft=False):
    try:
        ok, info = fn()
    except Exception as e:  # noqa
        ok, info = False, f"{type(e).__name__}: {str(e)[:200]}"
    results.append((name, ok, info, soft))
    tag = "PASS " if ok else ("WARN " if soft else "FAIL ")
    print(tag + name + " — " + str(info)[:200], flush=True)


def t_greedy_det():
    a = chat("Nævn tre danske byer, kun navnene.", temperature=0)["choices"][0]["message"]["content"]
    b = chat("Nævn tre danske byer, kun navnene.", temperature=0)["choices"][0]["message"]["content"]
    return a == b, a[:80]
def t_seed():
    a = chat("Skriv et digt på fire linjer om regn.", temperature=1.0, seed=1234, max_tokens=48)["choices"][0]["message"]["content"]
    b = chat("Skriv et digt på fire linjer om regn.", temperature=1.0, seed=1234, max_tokens=48)["choices"][0]["message"]["content"]
    return a == b, a[:80]
def t_logprobs():
    r = chat("Hvad er 2+2?", temperature=0, logprobs=True, top_logprobs=5, max_tokens=8)
    lp = r["choices"][0]["logprobs"]["content"]
    return len(lp) > 0 and len(lp[0]["top_logprobs"]) == 5 and all(x["logprob"] <= 0 for x in lp), f"{len(lp)} tokens, first={lp[0]['token']!r}"
def t_n2():
    r = chat("Giv mig et tilfældigt tal mellem 1 og 1000.", temperature=1.0, n=2, max_tokens=16)
    return len(r["choices"]) == 2, [c["message"]["content"][:30] for c in r["choices"]]
def t_stop():
    r = chat("Tæl fra 1 til 20, ét tal per linje.", temperature=0, stop=["5"], max_tokens=64)
    c = r["choices"][0]
    return "5" not in c["message"]["content"] and c["finish_reason"] == "stop", c["message"]["content"][:40].replace("\n", "/")
def t_json_schema():
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}}, "required": ["city", "population"]}
    r = chat("Giv Danmarks hovedstad og dens indbyggertal.", temperature=0,
             response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}, max_tokens=64)
    d = json.loads(r["choices"][0]["message"]["content"])
    return isinstance(d.get("population"), int) and isinstance(d.get("city"), str), d
def t_min_tokens_penalty():
    r = chat("Sig hej.", temperature=0.7, min_tokens=30, presence_penalty=1.2, frequency_penalty=0.3, max_tokens=48)
    return r["usage"]["completion_tokens"] >= 30, r["usage"]
def t_stream():
    body = post(URL, {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Skriv to sætninger om vejret."}],
                      "max_tokens": 48, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
    chunks = [l for l in body.splitlines() if l.startswith("data: ") and "[DONE]" not in l]
    return len(chunks) > 5, f"{len(chunks)} chunks"
def t_thinking():
    r = chat("Hvad er 17*23? Svar kort.", temperature=0.6, max_tokens=512, chat_template_kwargs={"enable_thinking": True})
    m = r["choices"][0]["message"]
    return ("391" in (m.get("content") or "")) and bool(m.get("reasoning_content") or m.get("reasoning")), (m.get("content") or "")[:60]
def t_completions_echo_logprobs():
    r = post(URLC, {"model": "qwen3.8-27b", "prompt": "København er hovedstaden i", "max_tokens": 4, "temperature": 0,
                    "echo": True, "logprobs": 1})
    lp = r["choices"][0]["logprobs"]
    return len(lp["tokens"]) > 4 and lp["token_logprobs"][0] is None, lp["tokens"][:6]
def t_long_ctx_greedy():
    doc = ("Danmark er et land i Nordeuropa. " * 40 + "\n") * 60   # ~20k tokens
    t0 = time.time()
    r = chat(doc + "\nHvor mange gange står ordet Nordeuropa i teksten ovenfor, cirka? Svar kort.", temperature=0, max_tokens=32)
    return r["usage"]["prompt_tokens"] > 12000, f"prompt={r['usage']['prompt_tokens']} tok, {time.time()-t0:.1f}s, {r['choices'][0]['message']['content'][:50]!r}"
def t_thinking_budget_rejected():
    # Unknown/unsupported params are silently IGNORED by this build (HTTP 200,
    # measured 2026-09-08 on CTX=long/SPEC=dflash2) — not 400 as an earlier
    # revision of this check assumed for the V2 runner. Soft check: records
    # the behavior without red-ing the suite; do not promote silent-ignore
    # into a PASS, and re-harden if the server starts validating.
    try:
        chat("Hej", max_tokens=8, thinking_token_budget=10)
        return False, "accepted (silently ignored; server does not validate)"
    except urllib.error.HTTPError as e:
        return e.code == 400, f"HTTP {e.code} (expected 400 on the V2 runner)"

for name, fn, soft in [("greedy determinism", t_greedy_det, False), ("seeded sampling determinism", t_seed, False), ("logprobs/top_logprobs", t_logprobs, False),
                 ("n=2", t_n2, False), ("stop strings", t_stop, False), ("json_schema structured output", t_json_schema, False),
                 ("min_tokens + penalties", t_min_tokens_penalty, False), ("streaming", t_stream, False), ("thinking mode", t_thinking, False),
                 ("completions echo+logprobs (prompt_logprobs)", t_completions_echo_logprobs, False), ("20k-token prompt", t_long_ctx_greedy, False),
                 ("thinking_token_budget -> 400", t_thinking_budget_rejected, True)]:
    check(name, fn, soft)
n_pass = sum(1 for _, ok, _, _ in results if ok)
n_warn = sum(1 for _, ok, _, soft in results if not ok and soft)
n_fail = sum(1 for _, ok, _, soft in results if not ok and not soft)
print("SUMMARY", n_pass, "/", len(results), "passed", f"({n_warn} warn, {n_fail} fail)")
# Fail-closed: a printed hard failure must also fail the process for CI.
if n_fail:
    print(f"RESULT FAIL ({n_fail} of {len(results)} failed)")
    sys.exit(1)
print("RESULT PASS")
