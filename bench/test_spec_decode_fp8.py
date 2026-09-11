"""fp8 path of the split-KV spec-decode attention: correctness against a dequantized reference and
timing against the bf16 kernel and FA2 (bf16). Run inside the vLLM venv after applying patches/triton-spec-attn-fp8-kv.patch (sm89+):
  venv/bin/python bench/test_spec_decode_fp8.py"""
import time, torch
from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention
from vllm.vllm_flash_attn import flash_attn_varlen_func
torch.manual_seed(0); dev = "cuda"
Hq, Hkv, D, BS = 24, 4, 256, 432
scale = D ** -0.5
FP8 = torch.float8_e4m3fn; FMAX = torch.finfo(FP8).max

def make(kv_lens, q_len):
    B = len(kv_lens); mb = max((l + BS - 1) // BS for l in kv_lens); nb = B * mb + 8
    kc = torch.randn(nb, BS, Hkv, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(nb, BS, Hkv, D, device=dev, dtype=torch.bfloat16)
    perm = torch.randperm(nb, device=dev); bt = torch.zeros(B, mb, dtype=torch.int32, device=dev)
    for b in range(B):
        n = (kv_lens[b] + BS - 1) // BS; bt[b, :n] = perm[b * mb: b * mb + n].to(torch.int32)
    q = torch.randn(B * q_len, Hq, D, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(0, B * q_len + 1, q_len, dtype=torch.int32, device=dev)
    return q, kc, vc, bt, cu, torch.tensor(kv_lens, dtype=torch.int32, device=dev)

def quant(x):  # per-tensor fp8 e4m3 the way vLLM's cache holds it: x = fp8 * scale
    s = (x.float().abs().amax() / FMAX).clamp(min=1e-6)
    return (x.float() / s).clamp(-FMAX, FMAX).to(FP8), s.reshape(1).to(torch.float32)

def ref(q, k4, v4, bt, kv_lens, q_len):  # k4/v4: dequantized float caches
    outs = []
    for b, L in enumerate(kv_lens):
        n = (L + BS - 1) // BS
        k = k4[bt[b, :n].long()].reshape(-1, Hkv, D)[:L]; v = v4[bt[b, :n].long()].reshape(-1, Hkv, D)[:L]
        qq = q[b * q_len:(b + 1) * q_len].float(); G = Hq // Hkv
        k = k.repeat_interleave(G, 1); v = v.repeat_interleave(G, 1)
        s = torch.einsum("qhd,khd->hqk", qq, k) * scale
        qpos = torch.arange(L - q_len, L, device=dev)[:, None]; kpos = torch.arange(L, device=dev)[None, :]
        s = s.masked_fill((kpos > qpos)[None], float("-inf")); p = torch.softmax(s, -1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
    return torch.cat(outs, 0)

def bench(fn, iters=50):
    for _ in range(10): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / iters * 1000

att = SpecDecodeAttention(max_num_reqs=64, num_heads=Hq, head_dim=D, device=dev, qmax=64)
print("correctness (fp8 kernel vs fp32 reference on the dequantized cache; bf16 kernel on the same values as control)")
worst = 0.0
for kv_lens, q_len in [([1500], 5), ([37, 1000, 4321], 5), ([16000], 1), ([700, 8], 8), ([432], 5), ([433, 431], 3),
                       ([1500], 16), ([2000, 300], 16), ([9000], 21), ([25000], 32), ([600, 4000], 32), ([1000], 64), ([90000], 8)]:
    q, kc, vc, bt, cu, su = make(kv_lens, q_len)
    k8, ks = quant(kc); v8, vs = quant(vc)
    k4 = k8.float() * ks; v4 = v8.float() * vs
    out = torch.empty_like(q); att.run(q, k8, v8, out, cu, su, bt, scale, len(kv_lens), q_len, k_descale=ks, v_descale=vs)
    r = ref(q, k4, v4, bt, kv_lens, q_len)
    ctrl = torch.empty_like(q); att.run(q, k4.to(torch.bfloat16), v4.to(torch.bfloat16), ctrl, cu, su, bt, scale, len(kv_lens), q_len)
    e = (out.float() - r).abs().max().item(); ec = (ctrl.float() - r).abs().max().item(); worst = max(worst, e)
    print(f"  kv={kv_lens} q={q_len}: max|fp8-ref|={e:.4f}  max|bf16-ref|={ec:.4f}  {'OK' if e < 0.05 else 'FAIL'}")
print("WORST", f"{worst:.4f}")
print("fp8 QUERY too (vLLM quantizes q on the fp8 cache path)")
worst_q = 0.0
for kv_lens, q_len in [([1500], 5), ([37, 1000, 4321], 5), ([700, 8], 8), ([25000], 32), ([90000], 8), ([1000], 64)]:
    q, kc, vc, bt, cu, su = make(kv_lens, q_len)
    k8, ks = quant(kc); v8, vs = quant(vc); q8, qs = quant(q)
    k4 = k8.float() * ks; v4 = v8.float() * vs; q4 = (q8.float() * qs).to(torch.bfloat16)
    out = torch.empty_like(q); att.run(q8, k8, v8, out, cu, su, bt, scale, len(kv_lens), q_len, k_descale=ks, v_descale=vs, q_descale=qs)
    r = ref(q4, k4, v4, bt, kv_lens, q_len)
    e = (out.float() - r).abs().max().item(); worst_q = max(worst_q, e)
    print(f"  kv={kv_lens} q={q_len}: max|fp8q-ref|={e:.4f}  {'OK' if e < 0.05 else 'FAIL'}")
print("WORST_Q", f"{worst_q:.4f}")
print("timing (us) per attention layer, batch=1: fp8 kernel / bf16 kernel / FA2 bf16")
for L in [1500, 4000, 25000, 60000, 90000]:
    for Q in [8, 16]:
        q, kc, vc, bt, cu, su = make([L], Q); k8, ks = quant(kc); v8, vs = quant(vc)
        out = torch.empty_like(q); fa = torch.empty_like(q)
        q8, qs = quant(q)
        t8 = bench(lambda: att.run(q8, k8, v8, out, cu, su, bt, scale, 1, Q, k_descale=ks, v_descale=vs, q_descale=qs))
        t16 = bench(lambda: att.run(q, kc, vc, out, cu, su, bt, scale, 1, Q))
        tfa = bench(lambda: flash_attn_varlen_func(q=q, k=kc, v=vc, out=fa, cu_seqlens_q=cu, max_seqlen_q=Q, seqused_k=su,
                                                   max_seqlen_k=L, softmax_scale=scale, causal=True, block_table=bt, fa_version=2))
        print(f"  kv={L:6d} q={Q:2d}  fp8 {t8:8.1f}  bf16 {t16:8.1f}  FA2 {tfa:8.1f}")
