"""How the multi-query verify attention scales with context, and what it would cost at 128k.

The split-KV kernel in patches/0007-spec-decode-attn.patch was tuned at a 25k context with
NUM_SEGMENTS hardcoded to 16. Long-context DFlash2 needs to know three things before any
KV-format work is worth starting:

  1. what a verify step costs per attention layer at 64k / 128k / 150k, not 25k
  2. whether raising the segment count (a one-constant change) is the whole win
  3. what vLLM's own Triton attention would do instead -- it is the backend a quantized KV
     cache forces us onto, and its split-KV ("3D") path is gated off whenever
     max_seqlen_q > 1 (triton_unified_attention.py:1047), which every verify step is.
     Arm C measures that gate closed; arm D measures it forced open.

Offline: allocates its own paged cache, needs no server. ~1.5 GB at the longest shape.

  venv/bin/python spec_attn_ctx_scan.py [--quick]
"""
import os
import sys

import torch

from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention
from vllm.v1.attention.ops import triton_unified_attention as tua
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"
Hq, Hkv, D, BS = 24, 4, 256, 480   # BS=480: the block size this model actually runs with
G = Hq // Hkv
scale = D ** -0.5
QUICK = "--quick" in sys.argv

KV_LENS = [25_000, 64_000] if QUICK else [25_000, 50_000, 64_000, 100_000, 128_000, 150_000]
Q_LENS = [8, 16]
SEGMENTS = [16, 32, 64]


def make(kv_len, q_len):
    nb = (kv_len + BS - 1) // BS
    num_blocks = nb + 4
    kc = torch.randn(num_blocks, BS, Hkv, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(num_blocks, BS, Hkv, D, device=dev, dtype=torch.bfloat16)
    bt = torch.randperm(num_blocks, device=dev)[:nb].to(torch.int32).view(1, nb)
    q = torch.randn(q_len, Hq, D, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, q_len], dtype=torch.int32, device=dev)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=dev)
    return q, kc, vc, bt, cu, seqused


def bench(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000  # us


def tua_call(q, kc, vc, out, cu, seqused, bt, q_len, kv_len, segm=None):
    """Stock Triton unified attention. segm=None -> 2D (what the gate allows today)."""
    kw = {}
    if segm is not None:
        nq_blocks = q.shape[0] * G  # generous upper bound on query blocks
        kw = dict(
            seq_threshold_3D=1024,
            num_par_softmax_segments=segm,
            softmax_segm_output=torch.empty(nq_blocks, Hq, segm, D, device=dev, dtype=torch.float32),
            softmax_segm_max=torch.empty(nq_blocks, Hq, segm, device=dev, dtype=torch.float32),
            softmax_segm_expsum=torch.empty(nq_blocks, Hq, segm, device=dev, dtype=torch.float32),
        )
    tua.unified_attention(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=q_len,
        seqused_k=seqused, max_seqlen_k=kv_len, softmax_scale=scale, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0,
        q_descale=None, k_descale=None, v_descale=None, **kw)


print(f"# BS={BS} Hq={Hq} Hkv={Hkv} D={D}, batch=1, us per attention layer")
print(f"{'kv':>8s} {'q':>3s} {'ours@16':>8s} {'ours@32':>8s} {'ours@64':>8s} "
      f"{'FA2':>8s} {'triton2D':>9s} {'triton3D':>9s}")

results = {}
for kv_len in KV_LENS:
    for q_len in Q_LENS:
        q, kc, vc, bt, cu, seqused = make(kv_len, q_len)
        out = torch.empty_like(q)
        row = {}

        for nseg in SEGMENTS:
            att = SpecDecodeAttention(max_num_reqs=4, num_heads=Hq, head_dim=D,
                                      device=dev, qmax=32, num_segments=nseg)
            row[f"ours@{nseg}"] = bench(
                lambda a=att: a.run(q, kc, vc, out, cu, seqused, bt, scale, 1, q_len))
            del att

        row["FA2"] = bench(lambda: flash_attn_varlen_func(
            q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=q_len,
            seqused_k=seqused, max_seqlen_k=kv_len, softmax_scale=scale,
            causal=True, block_table=bt, fa_version=2))

        try:
            row["triton2D"] = bench(lambda: tua_call(q, kc, vc, out, cu, seqused, bt, q_len, kv_len))
        except Exception as e:
            row["triton2D"] = float("nan")
            print(f"  ! triton2D kv={kv_len} q={q_len}: {type(e).__name__}: {str(e)[:120]}")
        try:
            row["triton3D"] = bench(lambda: tua_call(q, kc, vc, out, cu, seqused, bt, q_len, kv_len, segm=32))
        except Exception as e:
            row["triton3D"] = float("nan")
            print(f"  ! triton3D kv={kv_len} q={q_len}: {type(e).__name__}: {str(e)[:120]}")

        results[(kv_len, q_len)] = row
        print(f"{kv_len:8d} {q_len:3d} "
              f"{row['ours@16']:8.1f} {row['ours@32']:8.1f} {row['ours@64']:8.1f} "
              f"{row['FA2']:8.1f} {row['triton2D']:9.1f} {row['triton3D']:9.1f}")
        del q, kc, vc, bt, cu, seqused, out
        torch.cuda.empty_cache()

# What a whole step costs: 16 attention layers, plus the GDN layers and the GEMMs, which
# do not grow with context. The verify attention is the only context-dependent term.
print()
print("# 16 attention layers, ms per verify step (attention only)")
print(f"{'kv':>8s} {'q':>3s} {'ours@best':>10s} {'FA2':>8s}")
for (kv_len, q_len), row in results.items():
    best = min(row[f"ours@{n}"] for n in SEGMENTS)
    print(f"{kv_len:8d} {q_len:3d} {best * 16 / 1000:10.2f} {row['FA2'] * 16 / 1000:8.2f}")
