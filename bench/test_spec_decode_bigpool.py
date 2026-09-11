"""Issue #86: int32 overflow of blk * stride in _spec_attn_partial on a large KV pool.
A request whose blocks sit high in a pool larger than 2^31 elements / stride_kb reads the wrong memory
(bf16: stride_kb = 880*4*256 = 901,120 elements, so block ids above 2,383 overflow). Places one request's
blocks at the TOP of a 2,600-block pool and checks the kernel against the reference; bf16 and fp8 paths.
Run inside the vLLM venv after the patches are applied (any card): venv/bin/python bench/test_spec_decode_bigpool.py
The fp8 half has two designed skips (a kernel without the fp8 path; a card without fp8 conversion, sm86) and
reports which one fired; anything else there fails the run. Exit code 0 = all OK; a CUDA illegal access kills
the process (that is the 'before' result)."""
import sys, torch
from vllm.v1.attention.ops.spec_decode_attn import SpecDecodeAttention
torch.manual_seed(0); dev = "cuda"
Hq, Hkv, D, BS = 24, 4, 256, 880
scale = D ** -0.5
FP8 = torch.float8_e4m3fn; FMAX = torch.finfo(FP8).max
NB = 2600  # blocks in the pool: NB*BS*Hkv*D = 2.34e9 elements > 2^31
att = SpecDecodeAttention(max_num_reqs=8, num_heads=Hq, head_dim=D, device=dev, qmax=64)
def run(dtype_name, L=20000, q_len=8):
    nb_req = (L + BS - 1) // BS
    if dtype_name == "bf16":
        kc = torch.empty(NB, BS, Hkv, D, device=dev, dtype=torch.bfloat16).normal_()
        vc = torch.empty(NB, BS, Hkv, D, device=dev, dtype=torch.bfloat16).normal_()
        k8 = v8 = None; kd = vd = None
    else:
        kc = torch.empty(NB, BS, Hkv, D, device=dev, dtype=torch.bfloat16).normal_()
        s = (kc.abs().amax() / FMAX).clamp(min=1e-6); k8 = (kc / s).clamp(-FMAX, FMAX).to(FP8); kd = s.reshape(1).float(); del kc
        vc = torch.empty(NB, BS, Hkv, D, device=dev, dtype=torch.bfloat16).normal_()
        t = (vc.abs().amax() / FMAX).clamp(min=1e-6); v8 = (vc / t).clamp(-FMAX, FMAX).to(FP8); vd = t.reshape(1).float(); del vc
        torch.cuda.empty_cache()
        kc = k8; vc = v8   # the reference dequantizes only the request's own blocks below
    ids = torch.arange(NB - nb_req, NB, device=dev, dtype=torch.int32)   # the highest block ids
    bt = ids.reshape(1, nb_req)
    q = torch.randn(q_len, Hq, D, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, q_len], dtype=torch.int32, device=dev); su = torch.tensor([L], dtype=torch.int32, device=dev)
    out = torch.empty_like(q)
    if dtype_name == "bf16":
        att.run(q, kc, vc, out, cu, su, bt, scale, 1, q_len)
    else:
        att.run(q, k8, v8, out, cu, su, bt, scale, 1, q_len, k_descale=kd, v_descale=vd)
    torch.cuda.synchronize()
    k = kc[ids.long()].reshape(-1, Hkv, D)[:L].float(); v = vc[ids.long()].reshape(-1, Hkv, D)[:L].float()
    if kd is not None:
        k = k * kd; v = v * vd   # dequantized reference values for the request's blocks
    G = Hq // Hkv; k = k.repeat_interleave(G, 1); v = v.repeat_interleave(G, 1)
    s_ = torch.einsum("qhd,khd->hqk", q.float(), k) * scale
    qpos = torch.arange(L - q_len, L, device=dev)[:, None]; kpos = torch.arange(L, device=dev)[None, :]
    s_ = s_.masked_fill((kpos > qpos)[None], float("-inf")); p = torch.softmax(s_, -1)
    r = torch.einsum("hqk,khd->qhd", p, v)
    e = (out.float() - r).abs().max().item()
    print(f"  {dtype_name}: pool {NB} blocks, request on blocks {NB-nb_req}..{NB-1}, L={L}: max|ours-ref|={e:.4f} {'OK' if e < 0.05 else 'FAIL'}", flush=True)
    return e < 0.05
print("  (pre-fix expectation: without the int64 cast this next line dies with a CUDA illegal", flush=True)
print("   memory access and no assertion prints -- that killed process IS the bug, not a broken test)", flush=True)
results = [run("bf16")]
torch.cuda.empty_cache()   # the bf16 pool and its reference are gone with run()'s frame; give the fp8 half a clean card
# The fp8 half has exactly two designed skips; anything else is a failure of the run, not a skip.
try:
    results.append(run("fp8"))
except TypeError as e:
    if "k_descale" in str(e):
        print("  fp8: SKIP(by design): this kernel has no fp8 path (no k_descale argument)", flush=True)
    else:
        raise
except Exception as e:
    msg = str(e)
    if "fp8e4nv" in msg and "not supported in this architecture" in msg:
        print(f"  fp8: SKIP(by design): no fp8 conversion on this card ({type(e).__name__})", flush=True)
    else:
        print(f"  fp8: FAIL(run incomplete): {type(e).__name__}: {msg.splitlines()[0][:110]}", flush=True)
        results.append(False)
ok = all(results)
print("BIGPOOL", "OK" if ok else "FAIL"); sys.exit(0 if ok else 1)
