#!/bin/bash
# Post-boot serving warmup: exercise the serving path (decode, speculative
# verify, small and concurrent batches) before production traffic.
#
# The launcher's boot-time profile runs only profiling dummies, so a cold
# Triton cache may still compile serving-path variants on the first real
# request (gotchas 8, 39, 50). This script spends that cost in a controlled
# pass: one small request first (decode + speculative-verify capture), then a
# larger concurrent batch (the multi-sequence verify path).
#
# Exits 1 if the server is not healthy or any pass fails, so the serving
# wrapper (single-user/qwen-server.sh) treats it as a failed boot.
#
#   bash bench/warmup.sh             # 127.0.0.1:18020, the launcher's model
#   PORT=18021 bash bench/warmup.sh  # a different port
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
cd "$REPO"
export PATH="$REPO/venv/bin:$PATH"

# Reuse the API key the launcher started the server with. start_qwen.sh
# exports VLLM_API_KEY (from $VLLM_API_KEY or api_key.txt), and vLLM reads
# that variable to bind --api-key; vllm bench serve presents OPENAI_API_KEY,
# so the two must agree or every request 401s. An explicit OPENAI_API_KEY
# wins, then VLLM_API_KEY, then api_key.txt, then a harmless placeholder (the
# server ignores the key when none was bound).
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -n "${VLLM_API_KEY:-}" ]; then
        export OPENAI_API_KEY="$VLLM_API_KEY"
    elif [ -f "$REPO/api_key.txt" ]; then
        export OPENAI_API_KEY="$(cat "$REPO/api_key.txt")"
    else
        export OPENAI_API_KEY="EMPTY"
    fi
fi

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18020}
# Resolve the model the way single-user/start_qwen.sh does: the fast variant
# when present, else the base dir. Only the tokenizer matters here (its vocab
# size drives random-token generation; no weights are loaded), and the request
# is matched on --served-model-name, so any variant of the same model works.
MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
    MODEL="$REPO/models/Qwen3.8-27B-W4A16-AutoRound"
    if [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
        MODEL="$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast"
    fi
fi

# Build the command as an array, not a string: an unquoted "$B" re-splits and
# re-globs, so a path with a space or glob character would break, and a
# quoted "$B" would not word-split at all. "${BENCH[@]}" expands each element
# verbatim. (venv/bin/vllm is relative because we cd "$REPO" above.)
BENCH=(venv/bin/vllm bench serve --host "$HOST" --port "$PORT" --model "$MODEL" --served-model-name qwen3.8-27b)

curl -sf -o /dev/null "http://$HOST:$PORT/health" || { echo "[warmup] no server on $HOST:$PORT" >&2; exit 1; }
echo "[warmup] server ready, warming the serving path"
"${BENCH[@]}" --dataset-name random --random-input-len 128 --random-output-len 128 --num-prompts 2 --max-concurrency 1 --ignore-eos --temperature 0 > /dev/null 2>&1 \
  || { echo "[warmup] small-batch pass failed" >&2; exit 1; }
"${BENCH[@]}" --dataset-name random --random-input-len 256 --random-output-len 256 --num-prompts 8 --max-concurrency 4 --ignore-eos --temperature 0 > /dev/null 2>&1 \
  || { echo "[warmup] concurrent-batch pass failed" >&2; exit 1; }
echo "[warmup] done"
