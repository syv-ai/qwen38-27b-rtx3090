#!/usr/bin/env python3
"""CPU fixture for the F07 KVarN materialize-scratch capacity guard.

Loads KVarNConfig from the overlay source (stdlib-only, no torch/GPU) and
proves the sizing invariants the GPU guard in kvarn_decode_attention relies
on, plus the boundary semantics of materialize_fits:

  pytest bench/test_fa_scratch_capacity.py   # or: python bench/test_fa_scratch_capacity.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
_CFG = os.path.join(REPO, "kvarn", "files", "vllm", "model_executor",
                    "layers", "quantization", "kvarn", "config.py")

_spec = importlib.util.spec_from_file_location("kvarn_overlay_config", _CFG)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
KVarNConfig = _mod.KVarNConfig


def test_single_request_always_fits_within_cap():
    # A single request up to max_model_len must fit whenever max_model_len
    # itself is within the cap (the CTX=huge deployment: 240k < 262144).
    rows = KVarNConfig.fa_scratch_rows(2, 240000)
    assert KVarNConfig.materialize_fits(240000, rows)


def test_boundary_equal_fits():
    assert KVarNConfig.materialize_fits(100, 100) is True
    assert KVarNConfig.materialize_fits(101, 100) is False
    assert KVarNConfig.materialize_fits(0, 100) is True
    assert KVarNConfig.materialize_fits(-1, 100) is False


def test_over_capacity_batch_trips_guard():
    # Two max-context requests exceed the default cap: the guard must trip.
    cap = KVarNConfig.fa_scratch_cap()
    rows = KVarNConfig.fa_scratch_rows(2, 240000)
    assert rows <= cap
    assert KVarNConfig.materialize_fits(2 * 240000, rows) is False


def test_allocation_never_below_floor():
    assert KVarNConfig.fa_scratch_rows(1, 8) >= 4096


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print("RESULT " + ("FAIL" if failed else "PASS"))
    sys.exit(1 if failed else 0)
