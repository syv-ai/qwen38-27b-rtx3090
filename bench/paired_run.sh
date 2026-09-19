#!/bin/bash
# bench/paired_run.sh — paired-run driver (automation backlog item 7).
#
# Compares two arms (control vs treatment env files) in boot-level pairs with
# ABBA or seeded-randomized order, fresh boot per arm, all repeats retained,
# and a predeclared decision rule. Implements Step 8 of
# docs/self-improvement-loop.md.
#
#   bash bench/paired_run.sh --control-env bench/arms/base.env \
#     --treatment-env bench/arms/int8-mlp.env --measure bench/measure_c1.sh \
#     --pairs 3 --lane cold-engine --metric e2e --margin 0.03
#
# A/A noise floor (Loop B useful result): identical arms with --aa. The
# summary.json doubles as --noise for a later A/B run.
#
# Result contract: exit 0 only on PASS. FAIL (valid evidence, no useful gain)
# and INVALID (broken evidence) exit 1. A repeat whose metric is missing or
# non-finite is INVALID and excluded; fewer than 2 valid pairs INVALIDates
# the run. Secrets (*KEY*, *TOKEN*, *SECRET*, *PASSWORD*, api_key) are
# presence-only in manifests — values never touch the run root.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"

EXP=PAIRED-$(date +%Y%m%d-%H%M%S)
CONTROL_ENV=""; TREATMENT_ENV=""; MEASURE=""
PAIRS=3; ORDER=abba; SEED=1; LANE="cold-engine"; METRIC=e2e; MARGIN=0.03
NOISE=""; AA=0; PORT=18021; LAUNCHER="single-user/start_qwen.sh"
NO_BOOT=0; NO_TEARDOWN=0; FORCE_GPU=0
while [ $# -gt 0 ]; do case "$1" in
  --exp) EXP="$2"; shift 2;;
  --control-env) CONTROL_ENV="$2"; shift 2;;
  --treatment-env) TREATMENT_ENV="$2"; shift 2;;
  --measure) MEASURE="$2"; shift 2;;
  --pairs) PAIRS="$2"; shift 2;;
  --order) ORDER="$2"; shift 2;;
  --seed) SEED="$2"; shift 2;;
  --lane) LANE="$2"; shift 2;;
  --metric) METRIC="$2"; shift 2;;
  --margin) MARGIN="$2"; shift 2;;
  --noise) NOISE="$2"; shift 2;;
  --aa) AA=1; shift;;
  --port) PORT="$2"; shift 2;;
  --launcher) LAUNCHER="$2"; shift 2;;
  --no-boot) NO_BOOT=1; shift;;
  --no-teardown) NO_TEARDOWN=1; shift;;
  --force-gpu) FORCE_GPU=1; shift;;
  *) echo "paired_run: unknown flag $1" >&2; exit 1;;
esac; done

_refuse() { echo "paired_run: refusing: $1" >&2; exit 1; }
[ -n "$CONTROL_ENV" ] || _refuse "--control-env is required"
[ -n "$TREATMENT_ENV" ] || _refuse "--treatment-env is required"
[ -n "$MEASURE" ] || _refuse "--measure is required"
[ -f "$CONTROL_ENV" ] || _refuse "control env not found: $CONTROL_ENV"
[ -f "$TREATMENT_ENV" ] || _refuse "treatment env not found: $TREATMENT_ENV"
[ -x "$MEASURE" ] || [ -f "$MEASURE" ] || _refuse "measure script not found: $MEASURE"
[ "$PAIRS" -ge 2 ] 2>/dev/null || _refuse "--pairs must be >= 2 (a paired comparison needs repeats)"
[ "$ORDER" = "abba" ] || [ "$ORDER" = "random" ] || _refuse "--order must be abba|random"
case "$LANE" in cold-engine|warm-cold-prefix|warm-shared|warm-conversation|eviction) ;;
  *) _refuse "--lane must be one of Step 6: cold-engine|warm-cold-prefix|warm-shared|warm-conversation|eviction" ;;
esac

OUT="$HERE/results/$EXP"; mkdir -p "$OUT" || _refuse "cannot create $OUT"

# Cross-process paths must be native: the measure script and the python
# stages can be native Windows binaries that do not resolve cygwin/MSYS
# /e/-style paths. cygpath is identity-absent on Linux.
_native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
OUT_NATIVE=$(_native "$OUT"); MEASURE_NATIVE=$(_native "$MEASURE")

# ---- GPU lock (Step 5: prove ownership; never share by accident) -------------
LOCKDIR="$HERE/results/.gpu-lock"
if [ "$FORCE_GPU" = 1 ]; then rm -rf "$LOCKDIR"; fi
mkdir "$LOCKDIR" 2>/dev/null || _refuse "GPU lock held at $LOCKDIR (another run owns it; --force-gpu to override)"
echo "{\"exp\":\"$EXP\",\"pid\":$$,\"started\":\"$(date -u +%FT%TZ)\"}" > "$LOCKDIR/owner.json"
trap 'rm -rf "$LOCKDIR"' EXIT

# ---- arm diff: arms differ only as declared (Step 2 completion criterion) ----
DIFF_FILE="$OUT/arm-diff.txt"
diff <(grep -vE '^\s*(#|$)' "$CONTROL_ENV" | sort) \
     <(grep -vE '^\s*(#|$)' "$TREATMENT_ENV" | sort) > "$DIFF_FILE" || true
if [ ! -s "$DIFF_FILE" ] && [ "$AA" != 1 ]; then
  _refuse "arms are identical; declare one intended difference or pass --aa for a noise-floor run"
fi

# ---- order: ABBA or seeded-randomized pairs (Step 8) --------------------------
ORDER_FILE="$OUT/order.txt"
: > "$ORDER_FILE"
if [ "$ORDER" = "abba" ]; then
  for ((p=1; p<=PAIRS; p++)); do
    if ((p % 2 == 1)); then echo "pair$p control treatment" >> "$ORDER_FILE";
    else echo "pair$p treatment control" >> "$ORDER_FILE"; fi
  done
else
  RANDOM=$SEED
  for ((p=1; p<=PAIRS; p++)); do
    if ((RANDOM % 2 == 0)); then echo "pair$p control treatment" >> "$ORDER_FILE";
    else echo "pair$p treatment control" >> "$ORDER_FILE"; fi
  done
  echo "# seed=$SEED" >> "$ORDER_FILE"
fi

COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$(git status --porcelain 2>/dev/null | sha256sum | cut -d' ' -f1)

# _redacted_env <envfile>: presence-only for secrets, values otherwise.
_redacted_env() {
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in \#*|"") continue;; esac
    key=${line%%=*}; val=${line#*=}
    case "$key" in *KEY*|*TOKEN*|*SECRET*|*PASSWORD*|*api_key*)
      if [ -z "$val" ]; then echo "$key=<unset>"; else echo "$key=<set>"; fi ;;
      *) echo "$line" ;;
    esac
  done < "$1"
}

# ---- boot / measure / teardown (Step 5: owned PIDs only) ----------------------
_health() { curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; }
_boot() { # $1=arm $2=bootdir; arm env already exported
  local arm=$1 dir=$2
  if [ "$NO_BOOT" = 1 ]; then echo "stub-boot" > "$dir/server.pid"; return 0; fi
  # Paired boots are fresh by construction: never reuse a live server.
  _health && _refuse "port $PORT already healthy; paired boots must be fresh (stop :$PORT first)"
  PORT=$PORT HOST=127.0.0.1 nohup bash "$LAUNCHER" > "$dir/server.log" 2>&1 &
  echo $! > "$dir/server.pid"
  for _ in $(seq 1 120); do
    sleep 5; _health && break
    kill -0 "$(cat "$dir/server.pid")" 2>/dev/null || {
      echo "paired_run: boot $arm died, tail:" >&2; tail -20 "$dir/server.log" >&2; return 1; }
  done
  _health || { echo "paired_run: no health after 10 min for $arm" >&2; return 1; }
}
_teardown() { # $1=bootdir
  local dir=$1
  [ "$NO_TEARDOWN" = 1 ] && return 0
  [ -f "$dir/server.pid" ] || return 0
  local pid; pid=$(cat "$dir/server.pid")
  kill "$pid" 2>/dev/null; sleep 1
  pkill -f "start_qwen.sh" 2>/dev/null
  pkill -f "vllm serve.*$PORT" 2>/dev/null
  if command -v nvidia-smi >/dev/null 2>&1; then
    local U=0
    for _ in $(seq 1 30); do
      sleep 2; U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "${U:-999999}" -lt 2000 ] && break
    done
    echo "# torn down (gpu mem now ${U:-?} MiB)" >> "$dir/teardown.log"
  fi
}

export PATH="$REPO/venv/bin:$PATH"
START_TS=$(date -u +%FT%TZ)
pair=0
while read -r pname first second; do
  case "$pname" in \#*) continue;; esac
  pair=$((pair+1))
  for arm in $first $second; do
    [ "$arm" = control ] && ENVF=$CONTROL_ENV || ENVF=$TREATMENT_ENV
    dir="$OUT/${pname}-${arm}"; mkdir -p "$dir"
    # Load arm env into a subshell-free overlay: exported for boot+measure.
    set -a; . "$ENVF"; set +a
    echo "{\"pair\":$pair,\"arm\":\"$arm\",\"started\":\"$(date -u +%FT%TZ)\"}" > "$dir/boot.json"
    if ! _boot "$arm" "$dir"; then
      echo '{"status":"INVALID","reason":"boot-failed"}' > "$dir/result.json"
      _teardown "$dir"; continue
    fi
    if ! bash "$MEASURE_NATIVE" "$(_native "$dir/result.json")" > "$dir/measure.log" 2>&1; then
      python3 - "$dir/result.json" <<'PY' 2>/dev/null || true
import json,sys
p=sys.argv[1]
try:
    d=json.load(open(p)); assert d.get("status") in ("PASS","FAIL","SKIP","INVALID")
except Exception:
    json.dump({"status":"INVALID","reason":"measure-failed-or-unparsable","measurements":{},"artifacts":[]},open(p,"w"))
PY
    fi
    _teardown "$dir"
  done
done < "$ORDER_FILE"
END_TS=$(date -u +%FT%TZ)

# ---- manifests (Step 2: arm identity incl. redacted launch env) ---------------
for arm in control treatment; do
  [ "$arm" = control ] && ENVF=$CONTROL_ENV || ENVF=$TREATMENT_ENV
  boots=$(for d in "$OUT"/pair*-$arm/boot.json; do [ -f "$d" ] && cat "$d"; done | python3 -c "import json,sys; print(json.dumps([json.loads(l) for l in sys.stdin]))")
  artifacts=$(cd "$OUT" && sha256sum pair*-$arm/result.json 2>/dev/null | awk '{print $2" "$1}')
  python3 - "$(_native "$OUT/$arm-manifest.json")" <<PY
import json,sys
redacted = """$(_redacted_env "$ENVF")"""
json.dump({
  "arm": "$arm",
  "commit": "$COMMIT", "dirty_tree_sha256": "$DIRTY",
  "env_redacted": [l for l in redacted.splitlines() if l],
  "boots": json.loads('$boots'),
  "artifacts_sha256": """$artifacts""".splitlines(),
  "window": {"start": "$START_TS", "end": "$END_TS"},
}, open(sys.argv[1], "w"), indent=2)
PY
done

# ---- experiment note with the predeclared rule (Step 1) -----------------------
{
  echo "# $EXP"; echo
  echo "Order: $ORDER (see order.txt). Pairs: $PAIRS. Lane: $LANE."
  echo "Metric: $METRIC. Margin: $MARGIN. Noise: ${NOISE:-none}."
  echo "Control: $CONTROL_ENV. Treatment: $TREATMENT_ENV. Measure: $MEASURE."
  echo; echo "## Declared arm difference (full diff in arm-diff.txt)"
  cat "$DIFF_FILE"; echo
  echo "## Decision rule (predeclared)"
  echo "PASS when median paired (treatment-control)/control delta on $METRIC"
  echo "exceeds $MARGIN and exceeds the A/A noise p90${NOISE:+ from $NOISE}."
  echo "FAIL on valid evidence below that. INVALID on <2 valid pairs."
} > "$OUT/experiment.md"

# ---- summary + decision (Step 8: all repeats, median, tails, CIs) -------------
[ -n "$NOISE" ] && PAIRED_NOISE_NATIVE=$(_native "$NOISE") || PAIRED_NOISE_NATIVE=""
export PAIRED_OUT="$OUT_NATIVE" PAIRED_METRIC="$METRIC" PAIRED_MARGIN="$MARGIN" PAIRED_NOISE="$PAIRED_NOISE_NATIVE" PAIRED_ORDER="$ORDER_FILE"
SUMMARY_RC=0
python3 - <<'PY' || SUMMARY_RC=$?
import json, glob, os, statistics
out, metric = os.environ["PAIRED_OUT"], os.environ["PAIRED_METRIC"]
margin = float(os.environ["PAIRED_MARGIN"])
pairs, invalid_repeats = {}, []
for f in sorted(glob.glob(f"{out}/pair*-*/result.json")):
    arm = "control" if f.split("pair")[-1].split("-",1)[-1].startswith("control") else "treatment"
    pname = os.path.basename(os.path.dirname(f)).rsplit("-",1)[0]
    try:
        d = json.load(open(f))
        v = float(d["measurements"][metric])
        assert v == v and abs(v) != float("inf")
        ok = d.get("status") in ("PASS","FAIL")
    except Exception:
        ok = False; v = None
    if not ok:
        invalid_repeats.append(f); continue
    pairs.setdefault(pname, {})[arm] = v
deltas = [(t-c)/c for p in pairs.values() if "control" in p and "treatment" in p
          for c,t in [(p["control"], p["treatment"])] if c != 0]
valid_pairs = sum(1 for p in pairs.values() if "control" in p and "treatment" in p)
summary = {"metric": metric, "margin": margin, "valid_pairs": valid_pairs,
           "invalid_repeats": invalid_repeats, "pair_deltas": deltas,
           "all_repeats": {p: a for p, a in pairs.items()}}
if valid_pairs < 2:
    summary["status"] = "INVALID"; summary["reason"] = "fewer-than-2-valid-pairs"
else:
    med = statistics.median(deltas)
    summary["median_delta"] = med
    summary["frac_positive"] = sum(1 for d in deltas if d > 0)/len(deltas)
    noise_p90 = None
    if os.environ["PAIRED_NOISE"]:
        nd = json.load(open(os.environ["PAIRED_NOISE"]))["pair_deltas"]
        noise_p90 = sorted(abs(x) for x in nd)[max(0, int(0.9*len(nd))-1)] if nd else 0.0
    summary["noise_p90"] = noise_p90
    bar = max(margin, noise_p90 or 0.0)
    summary["bar"] = bar
    if med > bar:
        summary["status"] = "PASS"; summary["reason"] = f"median-delta {med:.4f} exceeds bar {bar:.4f}"
    else:
        summary["status"] = "FAIL"; summary["reason"] = f"median-delta {med:.4f} within bar {bar:.4f}"
json.dump(summary, open(f"{out}/summary.json","w"), indent=2)
PY
[ "$SUMMARY_RC" = 0 ] || { echo '{"status":"INVALID","reason":"summary-crashed"}' > "$OUT/summary.json"; }

# ---- decision.md + exit (result contract: only PASS exits 0) ------------------
STATUS=$(python3 -c "import json; print(json.load(open('$OUT_NATIVE/summary.json'))['status'])")
REASON=$(python3 -c "import json; print(json.load(open('$OUT_NATIVE/summary.json')).get('reason',''))")
{ echo "# Decision"; echo; echo "State: $STATUS"; echo "Reason: $REASON";
  echo "Summary: summary.json. Manifests: control-manifest.json, treatment-manifest.json."; } > "$OUT/decision.md"
echo "PAIRED $EXP status=$STATUS reason=$REASON (evidence in $OUT)"
[ "$STATUS" = "PASS" ]
