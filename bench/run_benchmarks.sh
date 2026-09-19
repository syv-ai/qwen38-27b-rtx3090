#!/bin/bash
# Reproduce the README numbers against the server you have running on this
# box. Prints one ROW line per measurement; ~10 min for a mode, longer with
# --prefill / --long.
#
#   bash bench/run_benchmarks.sh batch            # 64-concurrent + real-prompt cohorts (batch mode)
#   bash bench/run_benchmarks.sh single           # real-prompt cohorts, T=default and greedy, MTP acceptance
#   bash bench/run_benchmarks.sh batch --prefill  # + the prefill matrix (1k..100k)
#   bash bench/run_benchmarks.sh batch --long     # + 1x100k and 4x60k long-context requests (needs the context)
#
# Run it twice after a restart and keep the second numbers: the first run
# after start includes JIT warmup and reads 30-50% low. Then run
# bench/quality_battery.py — a fast server that emits garbage is worth nothing.
#
# Workload protocol v2: the real-prompt cohort explicitly renders prompts with
# enable_thinking=false (see THINK_KWARGS). Rows printed by this script are
# protocol v2. Older retained rows without that flag are protocol v1 and must
# not be compared against v2 numbers.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
MODE=${1:-batch}; shift || true
DO_PREFILL=0; DO_LONG=0
for a in "$@"; do case $a in --prefill) DO_PREFILL=1;; --long) DO_LONG=1;; esac; done
export PATH="$REPO/venv/bin:$PATH"
export OPENAI_API_KEY=${VLLM_API_KEY:-$(cat "$REPO/api_key.txt" 2>/dev/null)}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-18020}
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
B="venv/bin/vllm bench serve --host $HOST --port $PORT --model $MODEL --served-model-name qwen3.8-27b"
# F05: immutable per-run directory + manifest. OUT may be overridden, but the
# default is a fresh timestamped dir so runs never overwrite each other.
OUT=${OUT:-$HERE/results/run-$(date +%Y%m%d-%H%M%S)}; mkdir -p "$OUT"

curl -sf -o /dev/null http://$HOST:$PORT/health || { echo "no server on $HOST:$PORT"; exit 1; }
metrics() { curl -s http://$HOST:$PORT/metrics -H "Authorization: Bearer $OPENAI_API_KEY"; }
# tok/step needs the spec-decode counters, which vLLM registers only when
# speculation is configured. Batch mode has no speculative decoding
# (speculative_config=None), so the counters are absent, grep exits 1, and under
# `set -e` + `pipefail` the assignment `S0=$(spec)` aborted the whole run at the
# first cohort - silently, with no FAIL/INVALID line and no cohort log. tokstep()
# already renders missing values as '-', so treat absence as an empty sample.
spec() { metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens)_total" | awk '{print $NF}' | tr "\n" " "; return 0; }
num() { awk "/$1/ {print \$$2}" "$3"; }
# F03: cohort workload identity. The custom dataset is rendered client-side by
# the benchmark client, so thinking mode must be pinned here, not on the server.
THINK_KWARGS='{"enable_thinking": false}'
WORKLOAD_PROTOCOL="v2-thinking-off"
# F02: fail closed. Every benchmark invocation must succeed AND emit the
# required summary lines before any ROW is published. All-failed runs still
# exit 0 and print zeroed metrics, so explicitly require successful requests.
require_metric() { # logfile pattern...
  local F=$1; shift
  for pat in "$@"; do
    grep -q "$pat" "$F" || { echo "INVALID: $F missing '$pat'"; return 1; }
  done
  grep -qE "^Successful requests:\s+[1-9]" "$F" || { echo "INVALID: $F has zero successful requests"; return 1; }
  grep -qE "^Failed requests:\s+0\b" "$F" || { echo "INVALID: $F has failed requests"; return 1; }
}
run_bench() { # outfile, then benchmark args...
  local F=$1; shift
  "$@" > "$F" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL: benchmark exited $rc (see $F)"; return 1; fi
  require_metric "$F" "Output token throughput" "Mean TPOT" "Benchmark duration" || return 1
}
write_manifest() {
  # Best-effort provenance; never leak the API key.
  {
    echo "{"
    echo "  \"workload_protocol\": \"$WORKLOAD_PROTOCOL\","
    echo "  \"think_kwargs\": $THINK_KWARGS,"
    echo "  \"mode\": \"$MODE\","
    echo "  \"date\": \"$(date -u +%FT%TZ)\","
    echo "  \"commit\": \"$(git rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"dirty_shortstat\": \"$(git diff --shortstat 2>/dev/null | tr '\"' '_' | head -c 500)\","
    echo "  \"client_version\": \"$(venv/bin/vllm --version 2>/dev/null | head -n 1 | tr '\"' '_')\","
    echo "  \"server_url\": \"$HOST:$PORT\","
    echo "  \"model_arg\": \"$MODEL\","
    echo "  \"seed_base\": \"$SEED\","
    echo "  \"args\": \"$*\""
    echo "}"
  } > "$OUT/manifest.json"
  cp "$HERE/prompts_real.jsonl" "$OUT/prompts_real.jsonl" 2>/dev/null || true
  echo "# manifest: $OUT/manifest.json protocol=$WORKLOAD_PROTOCOL"
}
row() { # label logfile conc
  local L=$1 F=$2 C=$3
  require_metric "$F" "Output token throughput" "Mean TPOT" "Median TPOT" "Mean TTFT" "Benchmark duration" || return 1
  local E2E=$(num "Output token throughput" 5 $F) TPOT=$(num "Mean TPOT" 4 $F) MTPOT=$(num "Median TPOT" 4 $F) TTFT=$(num "Mean TTFT" 4 $F) DUR=$(num "Benchmark duration" 4 $F)
  # F09: e2e output throughput (output tokens / wall time) is the headline
  # throughput. C/medTPOT is a cohort latency score (a proxy), NOT decode
  # throughput: it does not prove C concurrent resident decoders.
  local PROXY=$(python3 -c "print(f'{$C*1000/$MTPOT:.0f}')" 2>/dev/null)
  echo "ROW $L [proto=$WORKLOAD_PROTOCOL] | e2e=$E2E tok/s (throughput) | cohort-latency-score(C/medTPOT,proxy)=$PROXY | medTPOT=$MTPOT ms | meanTTFT=$TTFT ms | dur=${DUR}s"
}
tokstep() { python3 -c "
a='$1'.split(); b='$2'.split()
try:
    d=float(b[0])-float(a[0]); acc=float(b[1])-float(a[1]); print(f'{1+acc/d:.2f}' if d>0 else '-')
except Exception: print('-')"; }

echo "# $(date) mode=$MODE server=$HOST:$PORT protocol=$WORKLOAD_PROTOCOL"
# Every random-dataset call gets its own --seed. The bench default (0) hands
# every call the SAME prompts, and with --enable-prefix-caching that turns
# later calls into silent partial prefix-cache hits whose size depends on the
# server's pool geometry: measured on the single-user stack, the old protocol
# read prefill 15-20% LOW on one config and 3x HIGH on another (a 16k prefill
# "measured" at 4.0 s that cold costs 11.2 s). Distinct seeds per call make
# every request a cold miss; SEED_BASE pins them for reproducibility.
SEED=${SEED_BASE:-1000}
write_manifest "$MODE"
run_bench "$OUT/warmup.log" env $B --dataset-name random --seed $((SEED+=1)) --random-input-len 256 --random-output-len 256 --num-prompts 16 --max-concurrency 8 > /dev/null 2>&1 || { echo "warmup failed"; exit 1; }

if [ "$MODE" = "batch" ]; then
  for W in "128 512" "256 256"; do set -- $W
    run_bench $OUT/batch_${1}_${2}.log env $B --dataset-name random --seed $((SEED+=1)) --ignore-eos --random-input-len $1 --random-output-len $2 --num-prompts 256 --max-concurrency 64
    row "64conc $1in/$2out" $OUT/batch_${1}_${2}.log 64
  done
fi
# real-prompt cohorts (both modes)
for T in default 0; do
  [ "$MODE" = "batch" ] && [ "$T" = "0" ] && break
  TARG=""; [ "$T" = "0" ] && TARG="--temperature 0"
  for C in 1 2 4 8; do
    S0=$(spec)
    run_bench $OUT/cohort_T${T}_c$C.log env $B --dataset-name custom --dataset-path $HERE/prompts_real.jsonl --custom-output-len 1024 --num-prompts 8 --max-concurrency $C $TARG --chat-template-kwargs "$THINK_KWARGS"
    S1=$(spec)
    L="cohort C$C real prompts T=$T"; F=$OUT/cohort_T${T}_c$C.log
    E2E=$(num "Output token throughput" 5 $F); MTPOT=$(num "Median TPOT" 4 $F); MTP=$(num "Mean TPOT" 4 $F)
    PROXY=$(python3 -c "print(f'{$C*1000/$MTP:.1f}')")
    echo "ROW $L [proto=$WORKLOAD_PROTOCOL] | e2e=$E2E tok/s (throughput) | cohort-latency-score(C/meanTPOT,proxy)=$PROXY | tok/step=$(tokstep "$S0" "$S1") | meanTTFT=$(num "Mean TTFT" 4 $F) ms"
  done
done
if [ $DO_PREFILL = 1 ]; then
  pf() { LEN=$1; C=$2; N=$3
    run_bench $OUT/prefill_${LEN}_c$C.log env $B --dataset-name random --seed $((SEED+=1)) --random-output-len 1 --random-input-len $LEN --num-prompts $N --max-concurrency $C
    IN=$(num "Total input tokens" 4 $OUT/prefill_${LEN}_c$C.log); DUR=$(num "Benchmark duration" 4 $OUT/prefill_${LEN}_c$C.log)
    echo "ROW prefill len=$LEN conc=$C | $(python3 -c "print(f'{$IN/$DUR:.0f}')") tok/s | meanTTFT=$(num "Mean TTFT" 4 $OUT/prefill_${LEN}_c$C.log) ms"; }
  pf 1024 1 16; pf 1024 4 32; pf 1024 16 64
  pf 4096 1 8;  pf 4096 4 16;  pf 4096 16 32
  pf 16384 1 4; pf 16384 4 8;  pf 16384 8 16
  pf 65536 1 2; pf 65536 2 4
  pf 102400 1 2
fi
if [ $DO_LONG = 1 ]; then
  run_bench $OUT/long_100k.log env $B --dataset-name random --seed $((SEED+=1)) --ignore-eos --random-input-len 100000 --random-output-len 256 --num-prompts 1 --max-concurrency 1
  echo "ROW 1x100k/256 | meanTTFT=$(num "Mean TTFT" 4 $OUT/long_100k.log) ms | TPOT=$(num "Mean TPOT" 4 $OUT/long_100k.log) ms"
  run_bench $OUT/long_4x60k.log env $B --dataset-name random --seed $((SEED+=1)) --ignore-eos --random-input-len 60000 --random-output-len 1024 --num-prompts 4 --max-concurrency 4
  echo "ROW 4x60k/1024 conc4 | e2e=$(num "Output token throughput" 5 $OUT/long_4x60k.log) tok/s | medITL=$(num "Median ITL" 4 $OUT/long_4x60k.log) ms | dur=$(num "Benchmark duration" 4 $OUT/long_4x60k.log)s"
fi
echo "# raw logs in $OUT"
echo "RESULT PASS"
