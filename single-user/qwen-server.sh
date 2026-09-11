#!/usr/bin/env bash
# Single-user serving wrapper: boot the launcher, wait for /health, run the
# post-boot serving warmup, then wait on the server.
#
# Why: start_qwen.sh ends in `exec vllm serve`, so a systemd unit has no
# after-boot hook. A cold boot may still JIT-compile Triton /
# speculative-decoding variants on the first serving requests (the repo's #48
# prewarm covers most of them; the ones it misses still compile in request 1).
# On tightly packed 24 GB configurations, those transient allocations can OOM
# the engine even after /health returns 200 (gotcha 39's "boots, /health 200,
# dies on the first prompt" shape). This wrapper spends that bill in a
# controlled pass before traffic.
#
#   WARMUP=1 (default)   wait for /health, run bench/warmup.sh, then serve
#   WARMUP=0             legacy behavior: just wait on the server
#
# The wrapper's exit code is the server's, so Restart=on-failure behaves as
# before. Point the unit's ExecStart here:
#   ExecStart=/bin/bash %h/qwen-serving/single-user/qwen-server.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$DIR")"
PORT=${PORT:-18020}
WARMUP=${WARMUP:-1}
WARMUP_ATTEMPTS=${WARMUP_ATTEMPTS:-30}
WARMUP_INTERVAL=${WARMUP_INTERVAL:-5}

# start_qwen.sh ends in `exec vllm serve`, so its pid becomes the server's
# and `wait` below tracks it.
bash "$DIR/start_qwen.sh" &
SERVER_PID=$!

cleanup() {
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup INT TERM

if [ "$WARMUP" = "1" ]; then
    echo "[server] waiting for server health..."
    HEALTHY=0
    for i in $(seq 1 "$WARMUP_ATTEMPTS"); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[server] server exited before becoming healthy" >&2
            # Reap the child and exit with its status (nonzero restarts the unit).
            wait "$SERVER_PID"
            exit $?
        fi
        if curl -sf -o /dev/null "http://127.0.0.1:$PORT/health"; then
            HEALTHY=1
            echo "[server] HTTP 200 OK"
            break
        fi
        sleep "$WARMUP_INTERVAL"
    done
    if [ "$HEALTHY" != "1" ]; then
        echo "[server] health timeout after $((WARMUP_ATTEMPTS * WARMUP_INTERVAL))s" >&2
        cleanup
        exit 1
    fi
    # warmup.sh inherits this wrapper's environment (MODEL, HOST, API key) and
    # re-derives the same defaults as start_qwen.sh, so it targets the server
    # just started; only PORT needs pinning to the port checked above.
    if PORT="$PORT" bash "$REPO/bench/warmup.sh"; then
        echo "[server] Warmup OK"
        echo "[server] server ready for traffic"
    else
        echo "[server] warmup failed" >&2
        cleanup
        exit 1
    fi
fi

wait "$SERVER_PID"
