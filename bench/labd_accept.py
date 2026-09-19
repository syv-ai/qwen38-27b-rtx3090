"""Teacher-forced acceptance: tokens per step while the server produces ONE fixed sequence.

labd_bench.py asks two servers the same question and compares tokens/step. That only works
if both write the same text. On "quote and explain" they do not -- greedy argmax ties break
differently once the batch shape changes -- and the 10% swings it reported were trajectory
divergence, not speed. This harness removes the trajectory as a variable: it pins a token
sequence once, then measures how well each configuration's speculator predicts *that*
sequence.

APPROACH: prefix-resync teacher forcing.

  A target continuation T (a list of token ids) is captured once and frozen on disk. The
  replay then walks it in chunks of C tokens: request j sends `prompt = P + T[:j*C]` as raw
  token ids to /v1/completions with max_tokens=C, and the four vllm:spec_decode_* counters
  are diffed around it. Every chunk therefore starts from a byte-identical prompt on every
  server being compared, and the returned ids are compared against T[j*C : j*C+C] so a chunk
  that went off-trajectory is detected exactly rather than silently averaged in.

  Two things make this valid for LABD specifically:

  * Forced tokens go into the *prompt*, and states.py writes the whole prompt into
    `req_states.all_token_ids` at offset 0 with `total_len = prefill_len`. That is exactly
    the array `suffix_lookup` scans (`lo = max(nmin-1, total_len - search_max)` up to
    `total_len - 1`), so the lookup sees the same history it would have seen had the tokens
    been generated. A forced prefix is indistinguishable from a produced one, to the drafter
    and to the lookup alike.
  * Exact ids, not text. The target is captured with `logprobs:0` +
    `return_tokens_as_token_ids:true`, so it is the model's own token ids, and it is replayed
    as ids. Nothing goes through a detokenize/retokenize round trip that could resegment it.

WHY NOT THE OBVIOUS ALTERNATIVES

  * Structured outputs / a literal guided regex would force the text, but the scheduler
    filters draft tokens against the grammar before scheduling them
    (`grammar.validate_tokens(spec_token_ids)` in `update_draft_token_ids{,_in_output}`,
    padding the rejects with -1). That mutates num_draft_tokens and num_accepted_tokens --
    the exact counters being measured -- and mutates them differently for a 7-slot and a
    15-slot server. Non-starter here.
  * prompt_logprobs / echo score the sequence during prefill. No decode steps, no drafts,
    nothing to count.
  * logit_bias, allowed_token_ids and bad_words are per-request constants. They cannot force
    a different token at each position.

WHAT IT CANNOT CONTROL FOR

  1. Chunk boundaries reset the adaptive long-block state machine. `next_num_draft_tokens`
     returns `draft_block` while `last_num_emitted is None`, the long block needs two
     qualifying steps in a row (`want and self._prev_want`), and the flag copy is one step
     stale. So each chunk pays roughly three short steps before a long block can be
     scheduled, and this harness under-reports long-block scheduling by about
     3/steps-per-chunk. steps/chunk is printed so the bias is visible; sweep --chunk to see
     which way it moves a number before believing it.
  2. It does not force *within* a chunk. Divergence at token 5 of a 128-token chunk leaves
     123 tokens off-trajectory. Those chunks are detected and reported as dirty, and the
     headline excludes them -- but that exclusion is itself a selection: divergence happens
     where the model is unsure, which is also where acceptance is low, so the clean-only
     number is biased slightly high. Watch the dirty count, not just the tok/step.
  3. The target was captured under *some* configuration. Forcing server B onto server A's
     text measures how well B predicts A's continuation, not B's own. Freeze the target once
     and reuse the same file for every configuration in a comparison; never recapture in the
     middle of one. The file records which tag captured it, and the run warns when it is the
     one doing the capturing.
  4. Wall clock is not comparable across --chunk values. Each resync is a fresh prefill
     (a prefix-cache hit, but still a scheduler admission and an uncached tail block) and a
     fresh first decode step. `decode` below is measured first-token-to-last inside a chunk,
     which excludes the resync prefill but not that first step. tok/step is the measurement;
     tok/s is a sanity check.
  5. /metrics counters are process-global. Anything else talking to the server during a run
     lands in the same numbers. Run this alone.
  6. Ties still break differently. This harness does not make two servers numerically
     identical, it only stops a divergence from compounding for 400 tokens. Divergences show
     up as dirty chunks instead of as a fake speed difference.

Usage (on syv, against a server started with SPEC=dflash2):

  venv/bin/python bench/labd_accept.py <tag> [--ctx 20000] [--max-tokens 512]
                                       [--chunk 128] [--block 7]
                                       [--tasks copy,code,edit,quote,summary,qa]
                                       [--base http://127.0.0.1:18020] [--recapture]
                                       [--keep-dirty]

Quality gate (F11): an empty target or zero clean chunks exits 1 with RESULT
FAIL instead of printing a nominal ~1 tok/step headline.
"""
import glob
import hashlib
import json
import os
import re
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
CORPUS = os.path.expanduser("~/bench/labd_corpus.txt")
TARGETS = os.path.expanduser("~/bench/targets")
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def flag(name):
    return name in sys.argv


BASE = arg("--base", "http://127.0.0.1:18020")
MODEL = arg("--model", "qwen3.8-27b")
CTX = int(arg("--ctx", 20000))
MAXTOK = int(arg("--max-tokens", 512))
CHUNK = int(arg("--chunk", 128))
BLOCK = int(arg("--block", 7))

SCALARS = ("vllm:spec_decode_num_drafts_total",
           "vllm:spec_decode_num_draft_tokens_total",
           "vllm:spec_decode_num_accepted_tokens_total")
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
POS_RE = re.compile(r'position="(\d+)"')


def post(path, payload, stream=False, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + KEY})
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else json.loads(r.read().decode())


def metrics():
    """(drafts, draft_slots, accepted, {position: accepted}) summed over engines."""
    req = urllib.request.Request(BASE + "/metrics", headers={"Authorization": "Bearer " + KEY})
    scal = {k: 0.0 for k in SCALARS}
    pos = {}
    for line in urllib.request.urlopen(req).read().decode().splitlines():
        if not line or line[0] == "#":
            continue
        # "name{labels} value" or "name value"; prometheus_client also emits _created lines,
        # which fall through both branches below.
        name = line.split("{", 1)[0].split(" ", 1)[0]
        val = float(line.rsplit(" ", 1)[-1])
        if name in scal:
            scal[name] += val
        elif name == PER_POS:
            m = POS_RE.search(line)
            if m:
                pos[int(m.group(1))] = pos.get(int(m.group(1)), 0.0) + val
    return (scal[SCALARS[0]], scal[SCALARS[1]], scal[SCALARS[2]], pos)


def tokenize_chat(content):
    """The prompt token ids /v1/chat/completions would build for this message."""
    r = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": content}],
                           "add_generation_prompt": True,
                           "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
    return r["tokens"], r["max_model_len"]


def ids_from_logprobs(tokens):
    """`return_tokens_as_token_ids` renders each token as "token_id:1234"."""
    out = []
    for t in tokens:
        if not t.startswith("token_id:"):
            return None
        out.append(int(t[9:]))
    return out


def generate(prompt_ids, max_tokens):
    """Stream a completion from raw prompt ids. -> (ids, ttft, decode_seconds, usage)."""
    payload = {"model": MODEL, "prompt": prompt_ids, "max_tokens": max_tokens,
               "temperature": 0, "logprobs": 0, "return_tokens_as_token_ids": True,
               "stream": True, "stream_options": {"include_usage": True}}
    toks, usage, t_first = [], {}, None
    t0 = time.time()
    with post("/v1/completions", payload, stream=True) as r:
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
                lp = ch.get("logprobs") or {}
                if lp.get("tokens"):
                    if t_first is None:
                        t_first = time.time()
                    toks.extend(lp["tokens"])
    t_end = time.time()
    ids = ids_from_logprobs(toks)
    if ids is None:
        raise SystemExit("server did not honour return_tokens_as_token_ids -- "
                         "cannot pin an exact token sequence, refusing to guess")
    return ids, (t_first or t_end) - t0, t_end - (t_first or t_end), usage


if not os.path.exists(CORPUS):
    docs = [open(f).read() for f in sorted(glob.glob(os.path.expanduser("~/qwen-serving/*.md")) +
                                           glob.glob(os.path.expanduser("~/qwen-serving/*/README.md")))]
    text = "\n\n".join(docs)
    while len(text) < 200000:
        text += "\n\n" + "\n\n".join(docs)
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    open(CORPUS, "w").write(text)
doc = open(CORPUS).read()[: int(CTX * 3.6)]

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
os.makedirs(TARGETS, exist_ok=True)

# Warm-up, same reason as labd_bench: the first long-block step JIT-compiles Triton kernels
# and those seconds would otherwise land inside the first chunk's decode window. Done before
# any counter is read, so it never enters a measurement.
warm_ids, _ = tokenize_chat("Dokument:\n\n" + doc[:4000] +
                            "\n\nGengiv ordret de første 10 linjer af dokumentet.")
for _ in range(2):
    generate(warm_ids, 64)
# num_speculative_tokens, straight from the server: the per-position counter has one series
# per draft slot, so its cardinality is what the server was started with.
NSPEC = max(len(metrics()[3]), BLOCK + 1)

tot = {"steps": 0.0, "slots": 0.0, "acc": 0.0, "head": 0.0, "tail": 0.0,
       "forced": 0.0, "dec": 0.0, "chunks": 0, "clean": 0}
rows = []
for name, q in TASKS:
    prompt_ids, max_model_len = tokenize_chat("Dokument:\n\n" + doc + "\n\n" + q)
    sig = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()[:16]
    path = os.path.join(TARGETS, f"labd_target_{name}_{sig}_{MAXTOK}.json")

    if flag("--recapture") or not os.path.exists(path):
        ids, _, _, _ = generate(prompt_ids, MAXTOK)
        json.dump({"task": name, "prompt_sig": sig, "captured_by": TAG,
                   "max_tokens": MAXTOK, "ctx": CTX, "ids": ids}, open(path, "w"))
        print(f"LABDACC {TAG} {name:8s} CAPTURED target n={len(ids)} -> {path} "
              f"(this run is scoring itself; freeze and reuse for the comparison)",
              flush=True)
    tgt = json.load(open(path))
    target = tgt["ids"]
    if len(prompt_ids) + len(target) > max_model_len:
        target = target[: max_model_len - len(prompt_ids) - 1]

    c = {"steps": 0.0, "slots": 0.0, "acc": 0.0, "head": 0.0, "tail": 0.0,
         "forced": 0.0, "dec": 0.0, "chunks": 0, "clean": 0, "first_bad": None}
    for off in range(0, len(target), CHUNK):
        want_ids = target[off: off + CHUNK]
        d0, s0, a0, p0 = metrics()
        got, _, dec, _ = generate(prompt_ids + target[:off], len(want_ids))
        d1, s1, a1, p1 = metrics()
        clean = got == want_ids
        c["chunks"] += 1
        c["clean"] += 1 if clean else 0
        if not clean and c["first_bad"] is None:
            c["first_bad"] = off + next((i for i, (g, w) in enumerate(zip(got, want_ids))
                                         if g != w), len(got))
        if clean or flag("--keep-dirty"):
            c["steps"] += d1 - d0
            c["slots"] += s1 - s0
            c["acc"] += a1 - a0
            for pos in set(p0) | set(p1):
                delta = p1.get(pos, 0.0) - p0.get(pos, 0.0)
                c["head" if pos < BLOCK else "tail"] += delta
            c["forced"] += len(got)
            c["dec"] += dec

    steps = max(c["steps"], 1.0)
    tps = 1 + c["acc"] / steps
    slots = c["slots"] / steps
    # slots/step sits at draft_block on short steps and at num_speculative_tokens on long
    # ones, so where it lands between the two is the fraction of steps that took the long
    # block. This is the number the "never scheduled" claim should be checked against.
    long_frac = max(0.0, (slots - BLOCK) / max(NSPEC - BLOCK, 1))
    rows.append((name, len(prompt_ids), int(c["forced"]), c["chunks"], c["clean"],
                 c["steps"], tps, slots, long_frac, c["head"] / steps, c["tail"] / steps,
                 c["forced"] / max(c["dec"], 1e-3), c["first_bad"]))
    for k in ("steps", "slots", "acc", "head", "tail", "forced", "dec", "chunks", "clean"):
        tot[k] += c[k]
    bad = "-" if c["first_bad"] is None else str(c["first_bad"])
    print(f"LABDACC {TAG} {name:8s} prompt={len(prompt_ids):6d} forced={int(c['forced']):4d} "
          f"chunks={c['clean']:3d}/{c['chunks']:<3d} steps={c['steps']:5.0f} "
          f"tok/step={tps:5.2f} slots/step={slots:5.2f} long={long_frac * 100:5.1f}% "
          f"head={c['head'] / steps:5.2f} tail={c['tail'] / steps:5.2f} "
          f"decode={c['forced'] / max(c['dec'], 1e-3):6.1f} firstbad={bad}", flush=True)

steps = max(tot["steps"], 1.0)
print(f"LABDACC {TAG} TOTAL forced={tot['forced']:.0f} chunks={tot['clean']:.0f}/{tot['chunks']:.0f} "
      f"steps={tot['steps']:.0f} tok/step={1 + tot['acc'] / steps:.3f} "
      f"slots/step={tot['slots'] / steps:.2f} head={tot['head'] / steps:.2f} "
      f"tail={tot['tail'] / steps:.2f} decode={tot['forced'] / max(tot['dec'], 1e-3):.1f} tok/s")
# F11: an empty target or zero clean chunks used to print a nominal ~1 tok/step.
# Fail closed: no clean evidence means no headline number.
failures = []
if tot["chunks"] == 0:
    failures.append("no chunks measured (empty target?)")
if tot["clean"] == 0:
    failures.append(f"zero clean chunks of {tot['chunks']} (all diverged)")
if failures:
    print(f"LABDACC {TAG} RESULT FAIL: " + "; ".join(failures), flush=True)
    sys.exit(1)
print(f"LABDACC {TAG} RESULT PASS", flush=True)
os.makedirs(os.path.expanduser("~/bench/results"), exist_ok=True)
json.dump({"tag": TAG, "ctx": CTX, "chunk": CHUNK, "block": BLOCK, "rows": rows,
           "total_tok_per_step": 1 + tot["acc"] / steps,
           "total_slots_per_step": tot["slots"] / steps,
           "total_head_per_step": tot["head"] / steps,
           "total_tail_per_step": tot["tail"] / steps,
           "clean_chunks": tot["clean"], "chunks": tot["chunks"]},
          open(os.path.expanduser(f"~/bench/results/labdacc_{TAG}.json"), "w"), indent=1)
