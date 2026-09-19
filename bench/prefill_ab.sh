#!/bin/bash
# Prefill A/B driver: boot single-user server with an env set, measure the
# prefill rows twice (keep the second pass), guard decode at C1, tear down.
#
#   bash bench/prefill_ab.sh baseline
#   INT8_ACT=int8 INT8_LAYERS=mlp bash bench/prefill_ab.sh int8-mlp
#   INT8_ACT=int8 INT8_LAYERS=mlp CHUNK=4096 bash bench/prefill_ab.sh int8-mlp-c4096
#
# Rows default to what fits production (DFLASH_TOKENS=15 -> 57344 max):
#   ROWS="1024 4096 16384 51200"
# Results under bench/results-prefill-ab/<arm>/ ; one ROW line per measurement.
# F10 ownership: this script only ever tears down a server it started itself
# (PID + start-time identity checked before any kill; no broad pkill). A
# server already on the port is reused ONLY with REUSE=1 — otherwise the
# script refuses, so one GPU never silently mixes two configurations.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
ARM=${1:?arm name}
ROWS=${ROWS:-"1024 4096 16384 51200"}
PORT=${PORT:-18021}          # do not fight the production port (18020)
OUT="$HERE/results-prefill-ab/$ARM"; mkdir -p "$OUT"
export PATH="$REPO/venv/bin:$PATH"
export OPENAI_API_KEY=${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast}
B="venv/bin/vllm bench serve --host 127.0.0.1 --port $PORT --model $MODEL --served-model-name qwen3.8-27b"
OWNED_PID=""
teardown_owned() {
  # Kill ONLY the server this run started, after verifying the PID still maps
  # to our command line (PID reuse would otherwise kill a stranger).
  if [ -n "$OWNED_PID" ] && [ "${KEEP:-0}" != 1 ]; then
    if tr '\0' ' ' < "/proc/$OWNED_PID/cmdline" 2>/dev/null | grep -q "start_qwen.sh"; then
      kill "$OWNED_PID" 2>/dev/null || true
    else
      echo "# owned pid $OWNED_PID no longer maps to start_qwen.sh — not killing"
    fi
  fi
}
trap teardown_owned EXIT
# F03: protocol v2 thinking flag for the custom cohort below.
THINK_KWARGS='{"enable_thinking": false}'

# ---- boot -------------------------------------------------------------------
if curl -sf -o /dev/null http://127.0.0.1:$PORT/health; then
  if [ "${REUSE:-0}" != 1 ]; then
    echo "server already on :$PORT and REUSE!=1 — refusing to measure against an"
    echo "unowned server of unknown configuration (set REUSE=1 to opt in)"; exit 1
  fi
  echo "server already on :$PORT — reusing by explicit REUSE=1 opt-in"
else
  # INT8_ACT/INT8_LAYERS: same translation batch/start_qwen.sh does (single-user
  # script has no wiring yet); CHUNK rides EXTRA_ARGS — the launcher's hardcoded
  # --max-num-batched-tokens 2048 is overridden because EXTRA_ARGS expands last.
  [ -n "${INT8_ACT:-}" ] && export VLLM_MARLIN_INPUT_DTYPE=$INT8_ACT
  [ -n "${INT8_LAYERS:-}" ] && export VLLM_MARLIN_INT8_INCLUDE_RE=$INT8_LAYERS
  [ -n "${CHUNK:-}" ] && EXTRA_ARGS="${EXTRA_ARGS:-} --max-num-batched-tokens $CHUNK"
  echo "# booting arm=$ARM  envs: SPEC=${SPEC:-dflash2} CTX=${CTX:-fast} DFLASH_TOKENS=${DFLASH_TOKENS:-15} PREFIX_CACHE=${PREFIX_CACHE:-1} INT8_ACT=${INT8_ACT:-} INT8_LAYERS=${INT8_LAYERS:-} CHUNK=${CHUNK:-} EXTRA_ARGS=${EXTRA_ARGS:-}"
  SPEC=${SPEC:-dflash2} CTX=${CTX:-fast} DFLASH_TOKENS=${DFLASH_TOKENS:-15} \
  PREFIX_CACHE=${PREFIX_CACHE:-1} PORT=$PORT HOST=127.0.0.1 EXTRA_ARGS="${EXTRA_ARGS:-}" \
  nohup bash single-user/start_qwen.sh > "$OUT/server.log" 2>&1 &
  OWNED_PID=$!
  echo $OWNED_PID > "$OUT/server.pid"
  for i in $(seq 1 120); do
    sleep 5; curl -sf -o /dev/null http://127.0.0.1:$PORT/health && break
    kill -0 "$OWNED_PID" 2>/dev/null || { echo "server died, tail:"; tail -20 "$OUT/server.log"; exit 1; }
  done
  curl -sf -o /dev/null http://127.0.0.1:$PORT/health || { echo "no health after 10 min"; exit 1; }
fi
{ echo "# arm=$ARM date=$(date -u +%FT%TZ)"; echo "# boot env: SPEC=${SPEC:-dflash2} CTX=${CTX:-fast} DFLASH_TOKENS=${DFLASH_TOKENS:-15} PREFIX_CACHE=${PREFIX_CACHE:-1} INT8_ACT=${INT8_ACT:-} INT8_LAYERS=${INT8_LAYERS:-} CHUNK=${CHUNK:-} EXTRA_ARGS=${EXTRA_ARGS:-} MODEL=$MODEL protocol=v2-thinking-off"; } > "$OUT/manifest.txt"
nvidia-smi --query-gpu=memory.used,memory.total,power.limit --format=csv,noheader | tee "$OUT/gpu-after-boot.txt"

num() { awk "/$1/ {print \$$2}" "$3"; }
metrics() { curl -s http://127.0.0.1:$PORT/metrics -H "Authorization: Bearer $OPENAI_API_KEY"; }
spec() { metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | awk '{print $NF}' | tr "\n" " "; }
# F02: fail closed — every benchmark call must exit 0 and emit its summary.
# All-failed runs still exit 0 with zeroed metrics: require successes too.
run_bench() { # outfile, then benchmark args...
  local F=$1; shift
  "$@" > "$F" 2>&1 || { echo "FAIL: benchmark exited nonzero (see $F)"; return 1; }
  grep -q "Output token throughput" "$F" || { echo "INVALID: $F missing throughput line"; return 1; }
  grep -qE "^Successful requests:\s+[1-9]" "$F" || { echo "INVALID: $F has zero successful requests"; return 1; }
  grep -qE "^Failed requests:\s+0\b" "$F" || { echo "INVALID: $F has failed requests"; return 1; }
}

# ---- warmup (JIT shapes: small + one large continuation) --------------------
# Every call gets its own --seed: the bench default (0) reuses the same prompts
# call to call, and with PREFIX_CACHE=1 that hands later calls silent
# prefix-cache hits whose size depends on the arm's pool geometry — pass 1 of
# spec-off measured 4.0 s for a 16k prefill that cold costs 11.2 s.
$B --dataset-name random --seed 901 --random-input-len 256 --random-output-len 64 --num-prompts 8 --max-concurrency 4 > /dev/null 2>&1
$B --dataset-name random --seed 902 --random-input-len 16384 --random-output-len 1 --num-prompts 2 --max-concurrency 1 > /dev/null 2>&1

# ---- prefill rows, two passes, keep the second ------------------------------
# Pass 2's seeds are the same in every arm, so all arms measure identical prompts.
IDX=0
for PASS in 1 2; do
  for L in $ROWS; do
    IDX=$((IDX+1))
    N=4; [ "$L" -ge 16384 ] && N=3; [ "$L" -ge 40000 ] && N=2
    run_bench "$OUT/pf_${L}_p$PASS.log" env $B --dataset-name random --seed $((PASS*1000+IDX)) --random-output-len 1 --random-input-len $L --num-prompts $N --max-concurrency 1
    IN=$(num "Total input tokens" 4 "$OUT/pf_${L}_p$PASS.log"); DUR=$(num "Benchmark duration" 4 "$OUT/pf_${L}_p$PASS.log")
    TTFT=$(num "Mean TTFT" 4 "$OUT/pf_${L}_p$PASS.log")
    [ "$PASS" = 2 ] && echo "ROW $ARM prefill len=$L | $(python3 -c "print(f'{$IN/$DUR:.0f}')") tok/s | meanTTFT=$TTFT ms"
  done
done

# ---- decode guard: C1 cohort, default sampling + tok/step -------------------
S0=$(spec)
run_bench "$OUT/cohort_c1.log" env $B --dataset-name custom --dataset-path "$HERE/prompts_real.jsonl" --custom-output-len 1024 --num-prompts 8 --max-concurrency 1 --chat-template-kwargs "$THINK_KWARGS"
S1=$(spec)
TS=$(python3 -c "
a='$S0'.split(); b='$S1'.split()
try:
    d=float(b[0])-float(a[0]); acc=float(b[1])-float(a[1]); print(f'{1+acc/d:.2f}' if d>0 else '-')
except Exception: print('-')")
echo "ROW $ARM decode C1 | decode=$(python3 -c "print(f'{1000/$(num "Mean TPOT" 4 "$OUT/cohort_c1.log"):.1f}')") tok/s | tok/step=$TS | meanTTFT=$(num "Mean TTFT" 4 "$OUT/cohort_c1.log") ms"

# ---- teardown (owned server only; the EXIT trap does the kill) --------------
# F10: no pkill, no PID-file kill of strangers. teardown_owned verifies the
# PID still maps to our start_qwen.sh before signalling.
teardown_owned; OWNED_PID=""
if [ "${KEEP:-0}" != 1 ]; then
  for i in $(seq 1 30); do
    sleep 2; U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 2000 ] && break
  done
  echo "# torn down (gpu mem now ${U:-?} MiB)"
fi
echo "# raw logs in $OUT"
echo "RESULT PASS"
