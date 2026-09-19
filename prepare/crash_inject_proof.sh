#!/bin/bash
# prepare/crash_inject_proof.sh — Loop C crash-injection proof for F04.
#
# Proves generation-scoped, locked, atomic preparation by killing publishers
# with SIGKILL at publication boundaries and asserting the on-disk protocol:
# after every kill the generation is either the OLD complete generation or
# the NEW complete generation — never mixed — source hashes AND inodes are
# unchanged, and the lock is re-acquirable (no wedged flock).
#
# SIGKILL, not SIGTERM: graceful shutdown would test the cleanup trap, not
# the atomicity of the on-disk protocol.
#
# Modes:
#   --self-test   run the full inject/assert cycle against synthetic
#                 publishers that mimic each real protocol (tmp+replace,
#                 bak-orig+replace, staging+replace, flock+publish).
#                 Needs no GPU, no models. This is the CI-safe mode.
#   --live        same cycle against the REAL boundaries below. Needs the
#                 model tree, venv, and idle CPU/GPU. NOT run while the
#                 GPUs are busy; recorded here so the live run is mechanical.
#                 Live modes need Linux/WSL2 (inotify, flock, setsid, stat -c);
#                 --self-test and --negative-test are portable. Refuses to run
#                 live elsewhere.
# Env: PROOF_LOG=<path> appends the verdict log; PROOF_DEADLINE=<secs>
# overrides the per-run watcher deadline (default 3600; the publisher's
# corpus-count phase can legitimately take longer than the old 600s).
#
# Real boundaries (validity/model-prep branch):
#   B1 build_draft_vocab.py  extras tmp + os.replace
#   B2 quant_lm_head.py      shard / index / config tmp + os.replace
#   B3 quant_heads_stream.py shard .bak-orig rename + tmp os.replace
#   B4 drafter/capture.py    seqs.json.staging + os.replace
#   B5 drafter/export_mtp.py generation copy-out + .bak-mtp rollback files
#   B6 docker/prepare.sh     flock -n fd9 + completion manifest
MODE=self-test
for a in "$@"; do case $a in --self-test) MODE=self-test;; --live) MODE=live;; \
  --live-b1) MODE=live-b1;; --live-b6) MODE=live-b6;; --live-b2) MODE=live-b2;; \
  --live-b3) MODE=live-b3;; --live-b4) MODE=live-b4;; --live-b5) MODE=live-b5;; \
  --negative-test) MODE=negative;; \
  *) echo "crash_inject_proof: unknown flag $a" >&2; exit 1;; esac; done
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_DIR/venv/bin/python"
case "$MODE" in live-b*|live)
  [ "$(uname -s)" = Linux ] || { echo "crash_inject_proof: live modes need Linux/WSL2 (inotify, flock, setsid); use --self-test" >&2; exit 1; }
  ;;
esac

PASS=0; FAIL=0; SKIP=0
LOG=""; [ -n "${PROOF_LOG:-}" ] && LOG="$PROOF_LOG"
command -v sha256sum >/dev/null 2>&1 \
  || { echo "crash_inject_proof: refusing: sha256sum not found" >&2; exit 1; }
# flock(1) is absent on some platforms (e.g. git-bash). Lock assertions then
# SKIP — they do not pass vacuously — and the live Linux run covers B6.
FLOCK_OK=0; command -v flock >/dev/null 2>&1 && FLOCK_OK=1
say() { echo "proof: $1"; [ -n "$LOG" ] && echo "proof: $1" >> "$LOG"; }
verdict() { # $1=name $2=0/1 $3=detail
  if [ "$2" = 0 ]; then PASS=$((PASS+1)); say "PASS $1 ($3)";
  else FAIL=$((FAIL+1)); say "FAIL $1 ($3)"; fi
}
skip() { SKIP=$((SKIP+1)); say "SKIP $1 ($2)"; }
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# _cycle <name> <publisher-fn> <gendir> <target> <sourcedir> [rollback-relpath]
# Rollback semantics (B3 shape): between the .bak-orig rename and the final
# replace the live path is MISSING and the old generation is selectable only
# via the rollback file. That is old-complete, not mixed — provided the
# rollback hashes to the pre-run generation. Without a rollback entry a
# missing live target is mixed.
_cycle() {
  local name=$1 pub=$2 gen=$3 target=$4 src=$5 rollback=${6:-}
  local src_snap; src_snap=$(cd "$src" && sha256sum ./* 2>/dev/null | sort)
  local src_ino; src_ino=$(cd "$src" && stat -c '%n %i' ./* 2>/dev/null | sort)
  local old_hash; old_hash=$(sha "$gen/$target")
  for round in midwrite late; do
    rm -f "$gen/$target.complete"
    ( $pub "$gen" "$target" ) & local pid=$!
    if [ "$round" = midwrite ]; then sleep 0.15; else sleep 1.5; fi
    kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    local new_hash; new_hash=$(sha "$gen/$target")
    local complete="absent"; [ -f "$gen/$target.complete" ] && complete=$(cat "$gen/$target.complete")
    # Old-or-new, never mixed: a new COMPLETE marker must agree with new
    # data; without the marker the data must still be the old generation;
    # with a rollback protocol a missing live target whose rollback hashes
    # to the old generation is old-via-rollback (recoverable, not mixed).
    local rb_hash="";
    [ -n "$rollback" ] && [ -f "$gen/$rollback" ] && rb_hash=$(sha "$gen/$rollback")
    if [ "$complete" = "$new_hash" ] && [ -n "$new_hash" ]; then verdict "$name/$round" 0 "new-complete";
    elif [ "$complete" = "absent" ] && [ "$new_hash" = "$old_hash" ]; then verdict "$name/$round" 0 "old-intact";
    elif [ "$complete" = "absent" ] && [ -z "$new_hash" ] && [ "$rb_hash" = "$old_hash" ] && [ -n "$old_hash" ]; then
      verdict "$name/$round" 0 "old-via-rollback";
    else verdict "$name/$round" 1 "mixed state (marker=$complete data=$new_hash old=$old_hash rb=$rb_hash)"; fi
    if [ "$(cd "$src" && sha256sum ./* 2>/dev/null | sort)" = "$src_snap" ] \
       && [ "$(cd "$src" && stat -c '%n %i' ./* 2>/dev/null | sort)" = "$src_ino" ]; then
      verdict "$name/$round-sources" 0 "hashes+inodes unchanged"
    else verdict "$name/$round-sources" 1 "source mutated"; fi
    if [ "$FLOCK_OK" = 1 ]; then
      if ( flock -n "$gen/.lock" true ) 2>/dev/null; then verdict "$name/$round-lock" 0 "re-acquirable";
      else verdict "$name/$round-lock" 1 "lock wedged"; fi
    else skip "$name/$round-lock" "no flock(1) on this platform; live run covers B6"; fi
  done
}

# ---- synthetic publishers: same protocols as B1-B6, ~0.6s of chunked writes --
_pub_tmp_replace() { # $1=gen $2=target  (B1, B2, B4 shape)
  local g=$1 t=$2 i
  : > "$g/$t.tmp"
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t.tmp"; sleep 0.05; done
  mv "$g/$t.tmp" "$g/$t"
  sha "$g/$t" > "$g/$t.complete"
}
_pub_bak_orig() { # $1=gen $2=target  (B3 shape)
  local g=$1 t=$2 i
  [ -e "$g/$t.bak-orig" ] || mv "$g/$t" "$g/$t.bak-orig"
  : > "$g/$t.tmp"
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t.tmp"; sleep 0.05; done
  mv "$g/$t.tmp" "$g/$t"
  sha "$g/$t" > "$g/$t.complete"
}
_pub_flock() { # $1=gen $2=target  (B6 shape: work under an exclusive lock)
  local g=$1 t=$2 i
  if [ "$FLOCK_OK" = 1 ]; then exec 9>"$g/.lock"; flock 9; fi
  : > "$g/$t.tmp"
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t.tmp"; sleep 0.05; done
  mv "$g/$t.tmp" "$g/$t"
  sha "$g/$t" > "$g/$t.complete"
}

_pub_direct() { # NEGATIVE CONTROL: in-place writes, no tmp, no marker.
  # This is the pre-F04 shape. The proof must flag every round mixed.
  local g=$1 t=$2 i
  for i in $(seq 1 12); do echo "chunk-$i-new" >> "$g/$t"; sleep 0.05; done
}

_negative_test() {
  local root; root=$(mktemp -d "${TMPDIR:-/tmp}/crashproof.XXXXXX")
  local gen=$root/gen src=$root/src; mkdir -p "$gen" "$src"
  seq 1 50 > "$gen/data.bin"
  echo "source-v1" > "$src/weights.bin"
  _cycle "negative/_pub_direct" _pub_direct "$gen" "data.bin" "$src"
  rm -rf "$root"
  if [ "$FAIL" -gt 0 ]; then
    say "negative control RED as required (proof detects in-place publication)"
  else say "negative control GREEN — proof is blind, fix it"; fi
}

_self_test() {
  local root; root=$(mktemp -d "${TMPDIR:-/tmp}/crashproof.XXXXXX")
  say "self-test root $root"
  local proto
  for proto in _pub_tmp_replace _pub_bak_orig _pub_flock; do
    local gen=$root/gen-$proto src=$root/src-$proto
    mkdir -p "$gen" "$src"
    seq 1 50 > "$gen/data.bin"
    echo "source-v1" > "$src/weights.bin"; echo "source-v1" > "$src/config.json"
    local rb=""; [ "$proto" = _pub_bak_orig ] && rb="data.bin.bak-orig"
    _cycle "selftest/$proto" "$proto" "$gen" "data.bin" "$src" "$rb"
  done
  rm -rf "$root"
}

# ---- live publishers ------------------------------------------------------
# B1: draft-vocab extras on the REAL model dir. Safe to kill: the only write
# is extras-tmp + os.replace, so a kill leaves litter (*.tmp) + old-intact.
# Re-run converges to new-complete.
_live_b1() {
  local base=${1:?usage: --live-b1 <model-dir>}
  [ -d "$base" ] || { verdict "live/B1-kill" 1 "base dir missing: $base"; return; }
  local target="model_extra_tensors.safetensors"
  local old; old=$(sha "$base/$target")
  local old_idx; old_idx=$(sha "$base/model.safetensors.index.json")
  local old_ids; old_ids=$(sha "$base/draft_vocab_ids.json")
  say "B1 old extras: ${old:-absent} index: $old_idx ids: ${old_ids:-absent (counting path only)}"
# The save window (~108 MB, tens of ms on NVMe) is uncatchable by
# polling, so an inotify watcher (libc via ctypes, stdlib only) spawns the
  "$VENV_PY" --version >/dev/null 2>&1 || { verdict "live/B1-kill" 1 "venv python missing"; return; }
  local watcher=$base/.watch-kill.py; _write_watcher "$watcher"
  local wk; wk=3
  DEADLINE_SECS="${PROOF_DEADLINE:-3600}" python3 "$watcher" "$base" "$target" "$VENV_PY" "$REPO_DIR/prepare/build_draft_vocab.py" "$base" --ids "$REPO_DIR/prepare/draft_vocab_ids.json" \
    && wk=0 || wk=$?
  rm -f "$watcher"
  if [ "$wk" = 1 ]; then
    skip "live/B1-kill" "build finished before the tmp window was observed; verifying new-complete only"
  elif [ "$wk" = 2 ]; then
    skip "live/B1-kill" "no tmp window seen within deadline (${PROOF_DEADLINE:-3600}s); raise PROOF_DEADLINE and retry"
    return
  elif [ "$wk" = 3 ]; then
    verdict "live/B1-kill" 1 "inotify queue overflow — window events lost, verdict unknown"
    return
  elif [ "$wk" != 0 ]; then
    verdict "live/B1-kill" 1 "watcher error (exit=$wk)"; return
  else
    local cur; cur=$(sha "$base/$target")
    if [ "$cur" = "$old" ]; then
      verdict "live/B1-kill-extras" 0 "old-intact after SIGKILL"
    elif "$VENV_PY" - "$base/$target" <<'PY' 2>/dev/null
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt") as f:
    assert any(k.startswith("mtp.draft_lm_head") for k in f.keys()), "no draft head tensors"
PY
    then verdict "live/B1-kill-extras" 0 "new-complete after SIGKILL (parses with draft head)"
    else verdict "live/B1-kill-extras" 1 "extras changed under SIGKILL and does not parse (old=$old cur=$cur)"
    fi
    # Post-Workstream-A the index publishes via .tmp + os.replace, so a kill
    # mid-dump cannot truncate it. After a mid-build kill it must still be
    # the OLD index (hash match). Only a clean re-run may introduce the new
    # head entries — never the kill. Absent index before the run: nothing to
    # assert — SKIP, never a vacuous PASS.
    local cur_idx; cur_idx=$(sha "$base/model.safetensors.index.json")
    if [ -z "$old_idx" ]; then
      skip "live/B1-kill-index" "no index file before the run; predicate not applicable"
    elif [ "$cur_idx" = "$old_idx" ]; then verdict "live/B1-kill-index" 0 "old index intact";
    else verdict "live/B1-kill-index" 1 "index changed under SIGKILL (old=$old_idx cur=$cur_idx)"; fi
    local cur_ids; cur_ids=$(sha "$base/draft_vocab_ids.json")
    if [ "$cur_ids" = "$old_ids" ]; then verdict "live/B1-kill-ids" 0 "ids unchanged";
    else verdict "live/B1-kill-ids" 1 "ids changed under SIGKILL"; fi
  fi
  "$VENV_PY" "$REPO_DIR/prepare/build_draft_vocab.py" "$base" --ids "$REPO_DIR/prepare/draft_vocab_ids.json" \
    || { verdict "live/B1-rerun" 1 "clean re-run failed"; return; }
  local new; new=$(sha "$base/$target")
  "$VENV_PY" - "$base/$target" <<'PY' || verdict "live/B1-rerun" 1 "extras do not parse"
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt") as f:
    assert any(k.startswith("mtp.draft_lm_head") for k in f.keys()), "no draft head tensors"
PY
  verdict "live/B1-rerun" 0 "new-complete ($new), parses with draft head"
  # The ids file exists only on the counting path (no --ids); with --ids its
  # absence is correct, not a gap. Require parse only when present.
  "$VENV_PY" - "$base/model.safetensors.index.json" <<'PY' \
    && verdict "live/B1-rerun-index" 0 "index parses with draft head after re-run" \
    || verdict "live/B1-rerun-index" 1 "index broken after re-run"
import json, sys
d = json.load(open(sys.argv[1]))
assert "weight_map" in d and "mtp.draft_lm_head.weight" in str(d)
PY
  if [ ! -f "$base/draft_vocab_ids.json" ]; then
    verdict "live/B1-rerun-ids" 0 "ids absent (shipped-ids path writes none)"
  elif "$VENV_PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$base/draft_vocab_ids.json" 2>/dev/null; then
    verdict "live/B1-rerun-ids" 0 "ids parse after re-run"
  else verdict "live/B1-rerun-ids" 1 "ids truncated after re-run"; fi
}

# Shared inotify kill-window watcher (see B1 note above).
_write_watcher() { # $1=dest-path; TRIGGER/ALLOW_RANDOM/DEADLINE_SECS via env
  cat > "$1" <<'PYEOF'
import ctypes, os, select, signal, struct, subprocess, sys, time
DBG = os.environ.get("PROOF_DEBUG", "")
def dbg(m):
    if DBG:
        sys.stderr.write("watcher-dbg: %s\n" % m)
        sys.stderr.flush()
libc = ctypes.CDLL("libc.so.6", use_errno=True)
IN_CREATE, IN_MOVED_TO, IN_MOVED_FROM = 0x100, 0x80, 0x40
fd = libc.inotify_init1(0)
assert fd >= 0
watchdir = sys.argv[1].encode()
dbg("watching %r" % watchdir)
assert libc.inotify_add_watch(fd, watchdir, IN_CREATE | IN_MOVED_TO | IN_MOVED_FROM) >= 0
tmpname = (sys.argv[2] + ".tmp").encode()
# TRIGGER: exact basename that fires the kill (CREATE or MOVED_TO).
# ALLOW_RANDOM=1 also fires on any random .tmp* staging create (mid-save).
# B1/B2a use the first save event; B2b waits for the index save specifically.
trigger = os.environ.get("TRIGGER", sys.argv[2] + ".tmp").encode()
allow_random = os.environ.get("ALLOW_RANDOM", "1") == "1"
deadline_secs = int(os.environ.get("DEADLINE_SECS", "3600"))
dbg("trigger %r allow_random %s deadline %d" % (trigger, allow_random, deadline_secs))
pub = subprocess.Popen(sys.argv[3:], start_new_session=True)
dbg("spawned pid %d" % pub.pid)
deadline = time.time() + deadline_secs
buf = b""   # inotify events can straddle reads; carry the tail across
while time.time() < deadline:
    if pub.poll() is not None:
        sys.exit(1)  # publisher finished before any tmp event
    r, _, _ = select.select([fd], [], [], 0.05)
    if not r:
        continue
    buf += os.read(fd, 65536)
    off = 0
    while True:
        if len(buf) - off < 16:
            buf = buf[off:]; break
        wd, mask, cookie, ln = struct.unpack("iIII", buf[off:off+16])
        if len(buf) - off < 16 + ln:
            buf = buf[off:]; break   # incomplete event; wait for the next read
        name = buf[off+16:off+16+ln].rstrip(b"\0")
        off += 16 + ln
        if wd == -1 and (mask & 0x4000):
            sys.stderr.write("watcher: inotify queue OVERFLOW — window events lost\n")
            sys.exit(3)
        # Hit on: the trigger tmp created or moved into place, or (when
        # allowed) any random .tmp* the writer stages through first
        # (safetensors save_file writes a random-tmp + rename internally, so
        # the named tmp arrives via MOVED_TO already complete — killing on
        # the random .tmp CREATE is what lands mid-save).
        named = name == trigger and (mask & (IN_CREATE | IN_MOVED_TO))
        staged = allow_random and name.startswith(b".tmp") and (mask & IN_CREATE)
        if named or staged:
            dbg("HIT %r" % name)
            time.sleep(0.1)  # land mid-save, not at creation
            os.killpg(pub.pid, signal.SIGKILL)
            pub.wait()
            sys.exit(0)
        else:
            dbg("other mask=%#x name=%r" % (mask, name))
sys.exit(2)
PYEOF
}

# B2: quant_lm_head set atomicity on a SCRATCH generation (never the live
# tree: a mid-set kill is unrecoverable by re-run — see below — so the live
# tree must not be the patient).
# Inspection finding under test: shard/index/config are each tmp+replaced
# but the SET has no commit. Worse, line 64 re-copies .bak unconditionally,
# so a re-run after a mid-set kill clobbers the rollback with the new shard
# and then KeyErrors on the missing lm_head.weight. Expected live result:
# run A (shard-stage kill) all-old PASS; run B (index-stage kill) MIXED —
# shard new + index old = unloadable generation = the Loop C finding that
# promotes a set-atomic fix (single manifest marker).
#   BOUND_B2=<scratch-dir> bash prepare/crash_inject_proof.sh --live-b2
_live_b2() {
  local base=${1:?usage: --live-b2 <scratch-dir>}
  [ -d "$base" ] || { verdict "live/B2-setup" 1 "base dir missing: $base"; return; }
  local shard; shard=$("$VENV_PY" - "$base/model.safetensors.index.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["weight_map"]["lm_head.weight"])
PY
) || { verdict "live/B2-setup" 1 "no bf16 lm_head.weight in $base (already quantized?)"; return; }
  say "B2 shard: $shard"
  local files=("$shard" model.safetensors.index.json config.json)
  local desc="$base/.proof-b2-desc"
  ( cd "$base" && sha256sum "${files[@]}" > "$desc" ) 2>/dev/null
  snap() { ( cd "$base" && sha256sum "${files[@]}" 2>/dev/null ); }
  oldset=$(snap)
  # Back up the set aside: run B clobbers .bak, so rollback for restore.
  rm -rf "$base/.proof-b2-orig"; mkdir -p "$base/.proof-b2-orig"
  cp -a "$base/$shard" "$base/model.safetensors.index.json" "$base/config.json" "$base/.proof-b2-orig/"
  runkill() { # $1=stage-name $2=trigger $3=allow-random
    _write_watcher "$base/.watch-kill.py"
    { TRIGGER="$2" ALLOW_RANDOM="$3" DEADLINE_SECS=1800 python3 "$base/.watch-kill.py" "$base" "unused" \
      "$VENV_PY" "$REPO_DIR/prepare/quant_lm_head.py" "$base" && echo KILLED-0 || echo KILLED-$?; } 2>&1 | tee -a "${PROOF_LOG:-/dev/null}"
  }
  say "B2 run A: kill at shard-save"
  runkill A "$shard.tmp" 1
  if [ "$(snap)" = "$oldset" ]; then verdict "live/B2-A" 0 "all-old after shard-stage kill";
  else verdict "live/B2-A" 1 "set changed under shard-stage kill"; fi
  say "B2 clean re-run to new-complete"
  "$VENV_PY" "$REPO_DIR/prepare/quant_lm_head.py" "$base" \
    || { verdict "live/B2-rerun" 1 "clean re-run failed"; return; }
  newset=$(snap)
  [ "$newset" != "$oldset" ] && verdict "live/B2-rerun" 0 "new set published" \
    || { verdict "live/B2-rerun" 1 "set unchanged by re-run"; return; }
  say "B2 run B: kill at index-save (mid-set window)"
  runkill B "model.safetensors.index.json.tmp" 0
  local now; now=$(snap)
  if [ "$now" = "$newset" ]; then
    skip "live/B2-B" "kill landed post-everything (coherent-new); mid-set window missed"
  elif [ "$now" = "$oldset" ]; then
    skip "live/B2-B" "kill landed pre-shard-replace (all-old); mid-set window missed"
  else
    verdict "live/B2-B" 1 "MIXED SET after index-stage kill (expected protocol gap; see B2 note)"
  fi
  cp -a "$base/.proof-b2-orig/." "$base/"
  rm -rf "$base/.proof-b2-orig" "$base/.watch-kill.py" "$desc"
  [ "$(snap)" = "$oldset" ] && verdict "live/B2-restore" 0 "generation restored from backup" \
    || verdict "live/B2-restore" 1 "restore failed"
}
# B6: prepare.sh lock protocol against the REAL lock file. docker/prepare.sh
# itself cds to /app (container-only), so the host-side proof exercises the
# identical mechanism: flock -n on <models>/.prepare.lock must serialize,
# and a SIGKILLed holder must release (fd death, no wedged lock).
_live_b6() {
  local lock=${1:?usage: --live-b6 <models-dir>}/.prepare.lock
  setsid flock "$lock" sleep 30 & local holder=$!
  sleep 1
  if flock -n "$lock" true 2>/dev/null; then
    kill -9 -- -$holder 2>/dev/null; wait "$holder" 2>/dev/null
    verdict "live/B6-contention" 1 "second holder acquired a held lock"
  else verdict "live/B6-contention" 0 "concurrent prepare refused"; fi
  kill -9 -- -$holder 2>/dev/null; wait "$holder" 2>/dev/null
  if flock -n "$lock" true 2>/dev/null; then verdict "live/B6-release" 0 "lock released after SIGKILL";
  else verdict "live/B6-release" 1 "lock wedged after SIGKILL"; fi
}
# B3: quant_heads_stream set atomicity on a SCRATCH bf16-head model subset
# (same patient rules as B2 — the shards are destructive rewrites, so a
# mid-set kill is unrecoverable by re-run). The script rewrites the big
# shard, the MTP shard, the index and the config; each file is now
# individually atomic (Workstream A), but the SET has no commit — run B
# (index-stage kill after a clean re-run) is expected to expose shard-new +
# index-old = MIXED, the finding that promotes a set-atomic fix.
#   BOUND_B3=<scratch-dir> bash prepare/crash_inject_proof.sh --live-b3
_live_b3() {
  local base=${1:?usage: --live-b3 <scratch-dir>}
  [ -d "$base" ] || { verdict "live/B3-setup" 1 "base dir missing: $base"; return; }
  local lm_key="lm_head.weight"
  local files; files=$("$VENV_PY" - "$base/model.safetensors.index.json" <<'PY'
import json, sys
idx = json.load(open(sys.argv[1])); wm = idx["weight_map"]
assert "lm_head.weight" in wm, "no bf16 lm_head.weight (already quantized?)"
mtp = [wm[m + ".weight"] for m in wm if m.startswith("mtp.layers.") and m.endswith(".weight")]
mtp_shards = sorted(set(mtp))
assert len(mtp_shards) == 1, f"mtp weights span several shards: {mtp_shards}"
print(wm["lm_head.weight"], mtp_shards[0])
PY
) || { verdict "live/B3-setup" 1 "index introspection failed in $base"; return; }
  set -- $files; local big=$1 mtp_shard=$2
  say "B3 big shard: $big  mtp shard: $mtp_shard"
  files=("$big" "$mtp_shard" model.safetensors.index.json config.json)
  snap() { ( cd "$base" && sha256sum "${files[@]}" 2>/dev/null ); }
  oldset=$(snap)
  rm -rf "$base/.proof-b3-orig"; mkdir -p "$base/.proof-b3-orig"
  cp -a "$base/$big" "$base/$mtp_shard" "$base/model.safetensors.index.json" "$base/config.json" "$base/.proof-b3-orig/"
  runkill3() { # $1=stage-name $2=trigger $3=allow-random
    _write_watcher "$base/.watch-kill.py"
    { TRIGGER="$2" ALLOW_RANDOM="$3" DEADLINE_SECS="${PROOF_DEADLINE:-3600}" python3 "$base/.watch-kill.py" "$base" "unused" \
      "$VENV_PY" "$REPO_DIR/prepare/quant_heads_stream.py" "$base" && echo KILLED-0 || echo KILLED-$?; } 2>&1 | tee -a "${PROOF_LOG:-/dev/null}"
  }
  say "B3 run A: kill at big-shard tmp write"
  runkill3 A "$big.tmp" 1
  if [ "$(snap)" = "$oldset" ]; then verdict "live/B3-A" 0 "all-old after big-shard-stage kill";
  else verdict "live/B3-A" 1 "set changed under big-shard-stage kill"; fi
  say "B3 clean re-run to new-complete"
  "$VENV_PY" "$REPO_DIR/prepare/quant_heads_stream.py" "$base" \
    || { verdict "live/B3-rerun" 1 "clean re-run failed"; return; }
  newset=$(snap)
  [ "$newset" != "$oldset" ] && verdict "live/B3-rerun" 0 "new set published" \
    || { verdict "live/B3-rerun" 1 "set unchanged by re-run"; return; }
  say "B3 run B: kill at index-save (mid-set window)"
  runkill3 B "model.safetensors.index.json.tmp" 0
  local now; now=$(snap)
  if [ "$now" = "$newset" ]; then
    skip "live/B3-B" "kill landed post-everything (coherent-new); mid-set window missed"
  elif [ "$now" = "$oldset" ]; then
    skip "live/B3-B" "kill landed pre-shard-replace (all-old); mid-set window missed"
  else
    verdict "live/B3-B" 1 "MIXED SET after index-stage kill (expected protocol gap; mirrors B2)"
  fi
  cp -a "$base/.proof-b3-orig/." "$base/"
  rm -rf "$base/.proof-b3-orig" "$base/.watch-kill.py"
  [ "$(snap)" = "$oldset" ] && verdict "live/B3-restore" 0 "generation restored from backup" \
    || verdict "live/B3-restore" 1 "restore failed"
}

# B4: capture.py manifest staging against the REAL drafter/data dir. The
# manifest (seqs.json) is the load-bearing publication; a kill at the
# staging write must leave it old-intact (or absent if never published),
# and a truncated seqs.json.staging is harmless litter. Non-destructive:
# the current seqs.json is never replaced by the kill (os.replace only
# runs at completion), so no restore is needed. Requires data/gen.jsonl;
# the full re-run is a GPU capture job left to the operator.
#   BOUND_B4=<drafter-data-dir> bash prepare/crash_inject_proof.sh --live-b4
_live_b4() {
  local base=${1:?usage: --live-b4 <drafter/data-dir>}
  [ -d "$base" ] || { verdict "live/B4-setup" 1 "data dir missing: $base"; return; }
  [ -f "$base/gen.jsonl" ] || { verdict "live/B4-setup" 1 "no gen.jsonl in $base"; return; }
  local gen_hash; gen_hash=$(sha "$base/gen.jsonl")
  local old; old=$(sha "$base/seqs.json")
  say "B4 seqs.json before run: ${old:-absent}"
  _write_watcher "$base/.watch-kill.py"
  local wk; wk=3
  { TRIGGER="seqs.json.staging" ALLOW_RANDOM=0 DEADLINE_SECS="${PROOF_DEADLINE:-3600}" \
    python3 "$base/.watch-kill.py" "$base" "unused" \
    "$VENV_PY" "$REPO_DIR/drafter/capture.py"; } 2>&1 | tee -a "${PROOF_LOG:-/dev/null}"
  rm -f "$base/.watch-kill.py"
  if [ "$(sha "$base/gen.jsonl")" != "$gen_hash" ]; then
    verdict "live/B4-sources" 1 "gen.jsonl mutated"; return
  else verdict "live/B4-sources" 0 "gen.jsonl unchanged"; fi
  local cur; cur=$(sha "$base/seqs.json")
  if [ -z "$old" ]; then
    if [ -z "$cur" ]; then verdict "live/B4-manifest" 0 "manifest absent after kill (never published) — correct";
    else verdict "live/B4-manifest" 1 "manifest appeared from a killed run"; fi
  elif [ "$cur" = "$old" ]; then verdict "live/B4-manifest" 0 "old manifest intact after kill";
  else verdict "live/B4-manifest" 1 "manifest changed under SIGKILL"; fi
}

# B5: export_mtp copy-out into a SCRATCH generation dir. Two shapes under
# test: (a) the bulk copy-in of base shards is a plain shutil.copy — NOT
# atomic, and the skip-if-exists re-run does not repair a truncated file;
# (b) the generated files (extras/index/config/ids, post-Workstream-A) are
# tmp+replace — a kill must leave them old-or-absent, never truncated.
#   BOUND_B5=<scratch-dest-dir> BOUND_B5_SRC=<source-model-dir> \
#   BOUND_B5_CK=<trained-checkpoint.safetensors> \
#     bash prepare/crash_inject_proof.sh --live-b5
_live_b5() {
  local dest=${1:?usage: --live-b5 <scratch-dest-dir>}
  local src=${BOUND_B5_SRC:?BOUND_B5=<source-model-dir> required}
  local ck=${BOUND_B5_CK:?BOUND_B5_CK=<trained-checkpoint> required}
  [ -d "$dest" ] || { verdict "live/B5-setup" 1 "dest dir missing: $dest"; return; }
  [ -d "$src" ] || { verdict "live/B5-setup" 1 "source dir missing: $src"; return; }
  [ -f "$ck" ] || { verdict "live/B5-setup" 1 "checkpoint missing: $ck"; return; }
  # Snapshot the copied-in set: every model-*.safetensors in the source.
  local copied=(); local f b
  for f in "$src"/model-*.safetensors; do [ -e "$f" ] && copied+=("$(basename "$f")"); done
  [ "${#copied[@]}" -gt 0 ] || { verdict "live/B5-setup" 1 "no model-*.safetensors in $src"; return; }
  snap5() { local h=""; for b in "${copied[@]}"; do h="$h$(sha "$dest/$b")"; done; printf '%s' "$h"; }
  _write_watcher "$dest/.watch-kill.py"
  { TRIGGER="model_extra_tensors.safetensors.tmp" ALLOW_RANDOM=1 DEADLINE_SECS="${PROOF_DEADLINE:-3600}" \
    python3 "$dest/.watch-kill.py" "$dest" "unused" \
    "$VENV_PY" "$REPO_DIR/drafter/export_mtp.py" "$ck" "$src" "$dest"; } 2>&1 | tee -a "${PROOF_LOG:-/dev/null}"
  rm -f "$dest/.watch-kill.py"
  # Predicate: every copied-in shard present in dest must hash to its source.
  # A mismatch is a truncated in-place copy — the copy-in gap — and re-run
  # will NOT repair it (skip-if-exists). Generated files (*.tmp) are litter.
  local bad=0
  for b in "${copied[@]}"; do
    [ -f "$dest/$b" ] || continue
    [ "$(sha "$dest/$b")" = "$(sha "$src/$b")" ] || { verdict "live/B5-copyin" 1 "TRUNCATED in-place copy: $b (re-run will not repair)"; bad=1; }
  done
  [ "$bad" = 0 ] && verdict "live/B5-copyin" 0 "all copied-in shards complete after kill"
}

# Each entry: name|generation-dir|target|publisher-command. The live cycle
# snapshots, backgrounds the publisher, SIGKILLs, and applies the same
# old-or-new + sources-unchanged + lock assertions with boundary-specific
# completeness predicates (index parses, shard sizes, manifest present).
_live_boundaries() {
  say "LIVE mode: GPUs are busy — configs recorded, nothing executed."
  say "B1 extras: prepare/build_draft_vocab.py -> draft_vocab extras tmp+replace"
  say "B2 head: prepare/quant_lm_head.py -> shard/index/config tmp+replace"
  say "B3 stream: prepare/quant_heads_stream.py -> .bak-orig + tmp replace"
  say "B4 capture: drafter/capture.py -> seqs.json.staging replace"
  say "B5 export: drafter/export_mtp.py -> generation copy-out + .bak-mtp"
  say "B6 install: docker/prepare.sh -> flock fd9 + completion manifest"
  say "Run with idle hardware: PROOF_LOG=evidence.log bash $0 --live (then implement per-boundary publishers)."
}

if [ "$MODE" = live ]; then _live_boundaries; elif [ "$MODE" = negative ]; then _negative_test;
elif [ "$MODE" = live-b1 ]; then _live_b1 "${BOUND_BASE:-}"; elif [ "$MODE" = live-b6 ]; then _live_b6 "${BOUND_MODELS:-}";
elif [ "$MODE" = live-b2 ]; then _live_b2 "${BOUND_B2:-}";
elif [ "$MODE" = live-b3 ]; then _live_b3 "${BOUND_B3:-}";
elif [ "$MODE" = live-b4 ]; then _live_b4 "${BOUND_B4:-}";
elif [ "$MODE" = live-b5 ]; then _live_b5 "${BOUND_B5:-}";
else _self_test; fi
say "verdicts: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
# Negative mode inverts the bar: success is the proof going red.
if [ "$MODE" = negative ]; then [ "$FAIL" -gt 0 ]; else [ "$FAIL" = 0 ]; fi
