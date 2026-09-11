#!/bin/bash
# Cross-layer KV cache sharing sweep (patches/qwen3_5-kv-cache-sharing.patch).
#
# Asks the only question that matters about the feature: these 16 full_attention
# layers were never trained to share a KV cache, so how much quality does each
# sharing layout cost? The memory saving is arithmetic and needs no experiment --
# group:2 halves the global KV cache, group:4 quarters it. The damage does.
#
#   bash bench/kv_share_sweep.sh                      # off group:2 group:4
#   ARMS="off group:2 group:4 suffix:8" bash bench/kv_share_sweep.sh
#   GSM_N=100 NEEDLE=0 bash bench/kv_share_sweep.sh   # quicker first look
#
# Results under bench/results-kv-share/<arm>/, one ROW line per measurement.
# Budget roughly 25 minutes per arm with the defaults.
#
# THREE THINGS THIS SCRIPT REFUSES TO GET WRONG, because each of them has
# produced a confident wrong answer in this repo before:
#
#  1. It verifies the arm actually took effect, by reading the owner count out
#     of the server log and comparing it to the arithmetic. An arm that silently
#     ran stock would report "no quality loss" and be believed. That is exactly
#     how the patch-integrity CI job stayed green for months while checking
#     nothing (docs/gotchas.md, the `git apply` prefix entry).
#  2. It boots each arm TWICE and measures the second boot. VLLM_QWEN_KV_SHARE
#     is part of the torch.compile cache key -- it has to be, it changes which
#     layers own a cache -- so every arm's first boot compiles cold, and a cold
#     compile profiles ~0.9 GiB more peak activation than a warm one. Comparing
#     a cold arm against a warm arm is comparing compile-cache states, not
#     layouts. Never A/B a KV pool across boots without this.
#  3. It runs perplexity with PREFIX_CACHE=0. prompt_logprobs is corrupted by
#     prefix caching on this hybrid model and reads ~23% high (gotcha 51).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
ARMS=${ARMS:-"off group:2 group:4"}
PORT=${PORT:-18021}          # do not fight the production port
GSM_N=${GSM_N:-200}
NEEDLE=${NEEDLE:-1}
NEEDLE_TOKENS=${NEEDLE_TOKENS:-32000}
ROOT="$HERE/results-kv-share"; mkdir -p "$ROOT"
export PATH="$REPO/venv/bin:$PATH"
export VLLM_API="http://127.0.0.1:$PORT/v1"
SUMMARY="$ROOT/summary.txt"; : > "$SUMMARY"

# 16 full_attention layers on Qwen3.8-27B (64 layers, full_attention_interval 4).
FULL=16
owners_for() {   # arm -> how many layers should still own a KV cache
  case "$1" in
    off)      echo $FULL ;;
    group:*)  python3 -c "import math;print(math.ceil($FULL/${1#group:}))" ;;
    suffix:*) echo "${1#suffix:}" ;;
    *)        echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

stop_server() {
  pkill -f "[v]llm serve" 2>/dev/null
  for i in $(seq 1 60); do pgrep -f "[v]llm serve" >/dev/null || break; sleep 2; done
  pgrep -f "[v]llm serve" >/dev/null && pkill -9 -f "[v]llm serve"
  sleep 5
}

boot() {   # boot(arm, logfile) -> waits for health, returns 1 if it never came up
  local arm=$1 log=$2
  local share=""; [ "$arm" != off ] && share=$arm
  KV_SHARE="$share" SPEC=${SPEC:-dflash2} CTX=${CTX:-fast} \
  DFLASH_TOKENS=${DFLASH_TOKENS:-15} PREFIX_CACHE=0 \
  PORT=$PORT HOST=127.0.0.1 \
  nohup bash single-user/start_qwen.sh > "$log" 2>&1 &
  echo $! > "$log.pid"
  for i in $(seq 1 150); do
    sleep 5; curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && return 0
    kill -0 "$(cat "$log.pid")" 2>/dev/null || { echo "server died; tail:"; tail -30 "$log"; return 1; }
  done
  echo "no health after 12 min"; tail -30 "$log"; return 1
}

for ARM in $ARMS; do
  OUT="$ROOT/${ARM//:/-}"; mkdir -p "$OUT"
  WANT=$(owners_for "$ARM")
  echo "=== arm $ARM  (expect $WANT of $FULL full_attention layers to own a cache)"

  # --- boot 1: warm the compile cache for THIS arm's cache key, then discard ---
  stop_server
  boot "$ARM" "$OUT/boot-warm.log" || { echo "ROW $ARM FAILED-TO-BOOT" | tee -a "$SUMMARY"; continue; }
  stop_server

  # --- boot 2: the one we measure ------------------------------------------
  boot "$ARM" "$OUT/server.log" || { echo "ROW $ARM FAILED-TO-BOOT-WARM" | tee -a "$SUMMARY"; continue; }

  # --- did the arm actually take effect? -----------------------------------
  GOT=$(grep -o "KV sharing ([^)]*): [0-9]* of [0-9]*" "$OUT/server.log" | tail -1 | awk '{print $4}')
  if [ "$ARM" = off ]; then
    if [ -n "$GOT" ]; then
      echo "ROW $ARM ABORT: baseline arm logged KV sharing ($GOT owners)" | tee -a "$SUMMARY"; stop_server; continue
    fi
  elif [ "$GOT" != "$WANT" ]; then
    echo "ROW $ARM ABORT: expected $WANT owners, server logged '${GOT:-nothing}'." | tee -a "$SUMMARY"
    echo "     The patch is not applied, or the launcher did not export VLLM_QWEN_KV_SHARE." | tee -a "$SUMMARY"
    stop_server; continue
  fi

  POOL=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$OUT/server.log" | tail -1 | tr -d ',' | awk '{print $5}')
  nvidia-smi --query-gpu=memory.used,power.limit --format=csv,noheader > "$OUT/gpu.txt"
  echo "ROW $ARM owners=${GOT:-$FULL}/$FULL pool=${POOL:-?} tokens vram=$(cat "$OUT/gpu.txt")" | tee -a "$SUMMARY"

  # --- quality: perplexity + GSM8K, then needle at depth --------------------
  python3 bench/quality_battery.py "kvshare-${ARM//:/-}" --gsm-n "$GSM_N" 2>&1 | tee "$OUT/quality.log"
  grep -E "PPL|GSM8K" "$OUT/quality.log" | sed "s/^/ROW $ARM /" | tee -a "$SUMMARY"

  if [ "$NEEDLE" = 1 ]; then
    for D in 0.1 0.5 0.9; do
      python3 bench/needle_test.py "$NEEDLE_TOKENS" "$D" > "$OUT/needle-$D.log" 2>&1
      echo "ROW $ARM needle tokens=$NEEDLE_TOKENS depth=$D $(tail -1 "$OUT/needle-$D.log")" | tee -a "$SUMMARY"
    done
  fi

  stop_server
done

echo
echo "=== summary ==="
cat "$SUMMARY"
echo
echo "Baseline is the 'off' arm measured on the same box in the same sweep."
echo "Numbers from a different session are not a baseline; boot-to-boot drift on"
echo "this stack is larger than several of the effects people have tried to claim."
