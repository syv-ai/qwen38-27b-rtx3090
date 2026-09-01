"""Correctness + microbenchmark of the split-KV spec-decode attention kernel
(patches/0007-spec-decode-attn.patch) against vLLM's FlashAttention-2 call, including the
long query blocks lookup-augmented drafting asks for (16 and 32 tokens per request).
Run inside the vLLM venv on the GPU after applying the patch:
  venv/bin/python bench/test_spec_decode_attn.py"""
import sys, os, time, math
import torch
from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = "cuda"
Hq, Hkv, D, BS = 24, 4, 256, 432
scale = D ** -0.5


def make(kv_lens, q_len, num_blocks=None):
    B = len(kv_lens)
    max_blocks = max((l + BS - 1) // BS for l in kv_lens)
    num_blocks = num_blocks or (B * max_blocks + 8)
    kc = torch.randn(num_blocks, BS, Hkv, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(num_blocks, BS, Hkv, D, device=dev, dtype=torch.bfloat16)
    perm = torch.randperm(num_blocks, device=dev)
    bt = torch.zeros(B, max_blocks, dtype=torch.int32, device=dev)
    for b in range(B):
        nb = (kv_lens[b] + BS - 1) // BS
        bt[b, :nb] = perm[b * max_blocks: b * max_blocks + nb].to(torch.int32)
    q = torch.randn(B * q_len, Hq, D, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(0, B * q_len + 1, q_len, dtype=torch.int32, device=dev)
    seqused = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    return q, kc, vc, bt, cu, seqused


def ref(q, kc, vc, bt, kv_lens, q_len):
    outs = []
    for b, L in enumerate(kv_lens):
        nb = (L + BS - 1) // BS
        k = kc[bt[b, :nb].long()].reshape(-1, Hkv, D)[:L].float()   # [L,Hkv,D]
        v = vc[bt[b, :nb].long()].reshape(-1, Hkv, D)[:L].float()
        qq = q[b * q_len:(b + 1) * q_len].float()                     # [q,Hq,D]
        G = Hq // Hkv
        k = k.repeat_interleave(G, dim=1); v = v.repeat_interleave(G, dim=1)
        s = torch.einsum("qhd,khd->hqk", qq, k) * scale
        qpos = torch.arange(L - q_len, L, device=dev)[:, None]
        kpos = torch.arange(L, device=dev)[None, :]
        s = s.masked_fill((kpos > qpos)[None], float("-inf"))
        p = torch.softmax(s, -1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
    return torch.cat(outs, 0)


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000  # us


att = SpecDecodeAttention(max_num_reqs=64, num_heads=Hq, head_dim=D, device=dev, qmax=64)
print("correctness")
for kv_lens, q_len in [([1500], 5), ([37, 1000, 4321], 5), ([16000], 1), ([700, 8], 8), ([432], 5),
                       ([433, 431], 3), ([1500], 16), ([2000, 300], 16), ([9000], 21), ([1500], 22),
                       ([25000], 32), ([600, 4000], 32), ([1000], 64)]:
    q, kc, vc, bt, cu, seqused = make(kv_lens, q_len)
    out = torch.empty_like(q)
    att.run(q, kc, vc, out, cu, seqused, bt, scale, len(kv_lens), q_len)
    r = ref(q, kc, vc, bt, kv_lens, q_len)
    fa = torch.empty_like(q)
    flash_attn_varlen_func(q=q, k=kc, v=vc, out=fa, cu_seqlens_q=cu, max_seqlen_q=q_len, seqused_k=seqused,
                           max_seqlen_k=max(kv_lens), softmax_scale=scale, causal=True, block_table=bt, fa_version=2)
    err = (out.float() - r).abs().max().item(); err_fa = (fa.float() - r).abs().max().item()
    print(f"  kv={kv_lens} q={q_len}: max|ours-ref|={err:.4f}  max|FA-ref|={err_fa:.4f}  {'OK' if err < 0.05 else 'FAIL'}")

print("timing (us) per attention layer, batch=1")
print(f"  {'kv':>7s} {'q_len':>5s} {'ours':>8s} {'FA2':>8s}")
for L in [1500, 4000, 25000, 60000]:
    for Q in [8, 16, 32]:
        kv_lens = [L]
        q, kc, vc, bt, cu, seqused = make(kv_lens, Q)
        out = torch.empty_like(q); fa = torch.empty_like(q)
        t_ours = bench(lambda: att.run(q, kc, vc, out, cu, seqused, bt, scale, 1, Q), iters=50)
        t_fa = bench(lambda: flash_attn_varlen_func(q=q, k=kc, v=vc, out=fa, cu_seqlens_q=cu, max_seqlen_q=Q, seqused_k=seqused,
                                                    max_seqlen_k=L, softmax_scale=scale, causal=True, block_table=bt, fa_version=2), iters=50)
        print(f"  {L:7d} {Q:5d} {t_ours:8.1f} {t_fa:8.1f}")
