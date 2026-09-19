#!/bin/bash
# Sourced after REPO is set. Preserve explicit MODEL, otherwise prefer the fast
# variant when present. Shared by the single-user launcher and its Docker gate.
if [ -z "${MODEL:-}" ] && [ -d "$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast" ]; then
  MODEL=$REPO/models/Qwen3.8-27B-W4A16-AutoRound-fast
fi
MODEL=${MODEL:-$REPO/models/Qwen3.8-27B-W4A16-AutoRound}
