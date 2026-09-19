"""Every residue mod 128, one prompt each, on a warm server.

Sampling a handful of residues is not a test. With one broken length in 128, five
distinct samples miss it C(127,5)/C(128,5) = 123/128 = **96.1%** of the time (96.2% if
you sample with replacement) -- so "5 of 5 clean" is barely evidence of anything, and
this repo shipped a wrong claim on exactly that arithmetic. An earlier version of this
docstring said 82%, which is the figure for six broken residues rather than one;
corrected by @mjungnickel18 in issue #25. Sweep all of them or say you sampled.

Stepping the pad by 1 covers all 128 residues exactly once, because one pad token is
one prompt token here -- no coverage gaps. Each prompt is sent TWICE and the second
(a full self-hit) is the measurement, which is the condition the bug needs.

Broken is judged by how much of the answer is a verbatim copy of the source, against
what the OTHER residues on this same server managed -- never by a failure signature.
The same residue has produced an empty answer, a one-character answer and 400 tokens of
confident derailment on three different configurations; see bench/verbatim.py.

  venv/bin/python bench/residue_sweep.py <label> [start] [count]

Fail-closed (F02): any broken residue exits 1 with RESULT FAIL.
"""
import hashlib, json, os, re, sys, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verbatim import classify, median, prefix_match, repeats  # noqa: E402

LABEL = sys.argv[1]
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 128
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)


def _key(path):  # a key is optional; keyless servers ignore the header
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(REPO, "api_key.txt"))
BASE = "http://127.0.0.1:" + os.environ.get("PORT", "18020")
DOC = open(os.path.expanduser(os.environ.get("CORPUS", "~/bench/labd_corpus.txt"))).read()[:72000]
from transformers import AutoTokenizer  # noqa: E402
TOK = AutoTokenizer.from_pretrained(os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound-fast"))


def metrics():
    txt = urllib.request.urlopen(urllib.request.Request(
        BASE + "/metrics", headers={"Authorization": "Bearer " + KEY}), timeout=30).read().decode()
    g = lambda n: sum(float(x) for x in re.findall(
        rf"^{re.escape(n)}\{{[^}}]*}} ([0-9.e+]+)$", txt, re.M)) or 0.0
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")


def once(content):
    body = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": content}],
                       "max_tokens": 300, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    d0 = metrics()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}),
        timeout=1800).read().decode())
    d1 = metrics()
    # `or ""`: a collapse-to-stop returns content=null, and indexing that ended an
    # earlier sweep with a TypeError instead of a finding.
    ans = r["choices"][0]["message"].get("content") or ""
    dn = d1[0] - d0[0]
    tps = (d1[1] - d0[1]) / dn + 1 if dn else float("nan")
    return tps, ans


seen = []
for pad in range(START, START + COUNT):
    content = ("Dokument:\n\n" + DOC +
               "\n\nGengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."
               + " x" * pad)
    real = len(TOK.encode(TOK.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False), add_special_tokens=False))
    once(content)                                  # arm the prefix cache
    tps, ans = once(content)                       # measure the self-hit
    # Provisional, against the residues swept so far; the authoritative call is the
    # second pass below, once the whole neighbourhood exists.
    ref = median([c for _, _, c, _ in seen]) if len(seen) >= 5 else None
    flag, cov, why = classify(ans, DOC, ref=ref)
    seen.append((real % 128, real, cov, dict(tps=tps, ans=ans)))
    if flag != "ok":
        print(f"[{LABEL}] BROKEN residue={real % 128:3d} len={real} tok/step={tps:.2f} "
              f"cov={cov:.2f} chars={len(ans)} prefix={prefix_match(ans, DOC)} "
              f"rep={repeats(ans)} sha={hashlib.sha1(ans.encode()).hexdigest()[:12]} ({why})",
              flush=True)
    else:
        # Every row, not every sixteenth: the discriminator is this residue against its
        # neighbours, and you cannot apply it to a log that only prints the outliers.
        print(f"[{LABEL}] .. residue={real % 128:3d} ok cov={cov:.2f} chars={len(ans)} "
              f"sha={hashlib.sha1(ans.encode()).hexdigest()[:12]} "
              f"({pad - START + 1}/{COUNT})", flush=True)

ref = median([c for _, _, c, _ in seen])
bad = []
for res, real, cov, d in seen:
    flag, cov, why = classify(d["ans"], DOC, ref=ref)
    if flag != "ok":
        bad.append((res, why))
print(f"[{LABEL}] neighbourhood coverage (median of {len(seen)}): {ref:.2f}")
print(f"[{LABEL}] DONE: {len(bad)} broken of {COUNT} residues"
      + ("" if not bad else "  -> " + ", ".join(f"{r} ({w})" for r, w in bad)), flush=True)
# Fail-closed: broken residues must fail the process for CI.
if bad:
    print(f"[{LABEL}] RESULT FAIL")
    sys.exit(1)
print(f"[{LABEL}] RESULT PASS")
