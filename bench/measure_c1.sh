#!/bin/bash
# bench/measure_c1.sh — reference measure script for bench/paired_run.sh.
#
# Runs the C1 real-prompt cohort (the real_rep.sh row: 8 realistic prompts,
# 1024 tokens each, one at a time) against the server the paired driver
# booted, and writes the result-JSON contract to $1:
#   {"status": "PASS|FAIL|INVALID", "reason": ..., "measurements": {...},
#    "artifacts": [...]}
# Fail-closed: any missing field or failed request INVALIDates the repeat
# (exit 1). The paired driver decides PASS/FAIL across repeats; a single
# measure never passes a comparison on its own.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
RESULT=${1:?usage: measure_c1.sh <result.json>}
# Same seam as paired_run.sh: a native python cannot resolve cygwin/MSYS
# /e/-style paths, so normalize argv once (identity on Linux).
command -v cygpath >/dev/null 2>&1 && RESULT=$(cygpath -m "$RESULT")
export PATH="$REPO/venv/bin:$PATH"
export OPENAI_API_KEY=${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}
PORT=${PORT:-18020}
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
B="venv/bin/vllm bench serve --host 127.0.0.1 --port $PORT --model $MODEL --served-model-name qwen3.8-27b"
T=${T:-}
[ -n "$T" ] && TA="--temperature $T" || TA=""

_invalid() { # $1=reason
  python3 -c "import json; json.dump({'status':'INVALID','reason':'$1','measurements':{},'artifacts':[]},open('$RESULT','w'))"
  echo "measure_c1: INVALID: $1" >&2; exit 1
}
snap() { curl -s "http://127.0.0.1:$PORT/metrics" -H "Authorization: Bearer $OPENAI_API_KEY" \
  | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | grep -v created | awk '{print $NF}' | tr "\n" " "; }

S0=$(snap)
$B --dataset-name custom --dataset-path "$HERE/prompts_real.jsonl" --custom-output-len 1024 \
  --num-prompts 8 --max-concurrency 1 $TA > "$RESULT.bench.log" 2>&1 || _invalid "bench-client-failed"
S1=$(snap)
F="$RESULT.bench.log"
OUT=$(awk '/Total generated tokens/ {print $4}' "$F")
DUR=$(awk '/Benchmark duration/ {print $4}' "$F")
E2E=$(awk '/Output token throughput/ {print $5}' "$F")
TPOT=$(awk '/Mean TPOT/ {print $4}' "$F")
TTFT=$(awk '/Mean TTFT/ {print $4}' "$F")
[ -n "$OUT" ] && [ -n "$DUR" ] && [ -n "$E2E" ] && [ -n "$TPOT" ] && [ -n "$TTFT" ] \
  || _invalid "missing-fields-in-bench-log"
python3 - "$S0" "$S1" "$OUT" "$DUR" "$E2E" "$TPOT" "$TTFT" <<PY || _invalid "nonfinite-metric"
import json, sys, math
a = [float(x) for x in sys.argv[1].split()]
b = [float(x) for x in sys.argv[2].split()]
out, dur, e2e, tpot, ttft = (float(sys.argv[i]) for i in range(3, 8))
if not all(math.isfinite(v) for v in (out, dur, e2e, tpot, ttft)) or dur <= 0:
    sys.exit(1)
d = [y - x for x, y in zip(a, b)]
steps, acc = d[0], d[1]
tok_step = 1 + acc / steps if steps > 0 else 0.0
if not math.isfinite(tok_step):
    sys.exit(1)
json.dump({
    "status": "PASS", "reason": "c1-cohort-complete",
    "measurements": {"e2e": e2e, "tpot": tpot, "ttft": ttft,
                     "tok_per_step": tok_step, "out_tokens": out},
    "artifacts": ["bench.log"],
}, open("$RESULT", "w"), indent=2)
print(f"C1 out={out:.0f} e2e={e2e:.1f} tok/s tpot={tpot:.2f}ms ttft={ttft:.0f}ms tok/step={tok_step:.2f}")
PY
mv "$RESULT.bench.log" "$(dirname "$RESULT")/bench.log"
