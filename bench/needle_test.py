#!/usr/bin/env python3
"""Needle-in-a-haystack probe against the running server.

Builds a ~TARGET_TOKENS filler context, hides a secret passcode at a given
fractional depth, asks the model for it, and reports whether the answer
contains the passcode. Complements quality_battery.py's GSM8K lane for the
"quality at depth" question on long-context KV configs.

Usage:
    python bench/needle_test.py [target_tokens] [depth]
    # default: 100000 tokens, needle at 90% depth
"""
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def _key(path):  # same convention as quality_battery.py
    try:
        return open(path).read().strip()
    except OSError:
        return ""


KEY = os.environ.get("VLLM_API_KEY") or _key(os.path.join(HERE, "..", "api_key.txt"))
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")

TARGET_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
DEPTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9

NEEDLE = "ZXCVBNM12345"
needle_line = f"The secret passcode is {NEEDLE}. Remember it exactly."

# "All work and no play makes Jack a dull boy. " is ~46 chars, ~11 tokens.
unit = "All work and no play makes Jack a dull boy. "
filler = unit * int(TARGET_TOKENS / 11)

depth = int(len(filler) * DEPTH)
context = filler[:depth] + "\n\n" + needle_line + "\n\n" + filler[depth:]

prompt = context + "\n\nQuestion: what is the secret passcode? Reply with the passcode only."


def post(payload, timeout=1800):
    req = urllib.request.Request(
        API + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


t0 = time.perf_counter()
resp = post({
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 32,
})
elapsed = time.perf_counter() - t0

answer = (resp["choices"][0]["message"].get("content") or "")
print(f"context ~{TARGET_TOKENS} tokens, needle at {DEPTH:.0%} depth")
print(f"elapsed {elapsed:.1f}s, answer: {answer[:120]!r}")
print("RETRIEVED" if NEEDLE in answer else "MISSED")
