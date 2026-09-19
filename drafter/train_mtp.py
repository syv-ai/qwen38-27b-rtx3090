"""Fine-tune the Qwen3.8-27B MTP draft module on the target model's own hidden states /
output distribution (self-distillation), so speculative decoding accepts more tokens.

Inputs (from capture.py): data/hidden.npy [T,5120] bf16-as-uint16, data/tokens.npy [T] int32,
data/seqs.json. The MTP module (bf16) is read from the model dir's
model_extra_tensors.safetensors.bak-mtp; embed_tokens / lm_head bf16 from the pre-quant
backup shards; the draft-vocab id list from mtp_draft_vocab_ids.pt.

Objective (row t: h_t = target hidden after token t, x_{t+1} the next token):
  depth 1:  z1 = MTP(h_t, x_{t+1}; pos t+1)         q1 = softmax(head(z1))   p1 = target dist for x_{t+2}
  depth d:  zd = MTP(z_{d-1}, x_{t+d}; pos t+d)     (the drafter's own hidden fed back, as vLLM
            chains drafts; attends to depth-1 KV of positions <= t+1 plus its own chain)
  loss = sum_d CE(p_d, q_d) with p_d = softmax(lm_head(h_{t+d})) restricted to the draft vocab.
Matches exactly what vLLM's mtp drafter computes (with the mtp-draft-vocab patch).

  python train_mtp.py --out runs/r1 [--epochs 2] [--lr 2e-5] [--depths 2] [--train-head]
                      [--tokens-per-step 32768] [--eval-only 1] [--init ckpt.safetensors]
Writes <out>/mtp_bf16.safetensors (vLLM tensor names, bf16) + log.txt.

Row identity + completion (F14): the effective draft-vocab IDs (custom
--draft-ids or the model dir's) are bundled as <out>/draft_vocab_ids.json with
every checkpoint so export_mtp.py can verify row identity instead of attaching
whatever IDs the source dir carries; the train split is preflighted non-empty
(val_frac can no longer reserve the whole corpus). Logged KL/top-1 are
token-normalized (sums of per-row stats divided by contributing tokens).
"""
import os, sys, json, math, time, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound"))
ap.add_argument("--data", default=os.path.join(HERE, "data"))
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--head-lr", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--tokens-per-step", type=int, default=32768)
ap.add_argument("--micro-tokens", type=int, default=8192, help="padded tokens per micro-batch")
ap.add_argument("--depths", type=int, default=1, help="unrolled draft depths to train (1..4)")
ap.add_argument("--depth-weights", default="", help="comma list, default all 1")
ap.add_argument("--train-head", action="store_true", help="also train the 40k-row draft head")
ap.add_argument("--full-vocab", action="store_true", help="use the full lm_head as draft head")
ap.add_argument("--val-frac", type=float, default=0.02)
ap.add_argument("--max-seqs", type=int, default=0)
ap.add_argument("--eval-only", default="")
ap.add_argument("--init", default="", help="start from a previously trained mtp_bf16.safetensors")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--eval-every", type=int, default=200)
ap.add_argument("--head-chunk", type=int, default=1024)
ap.add_argument("--draft-ids", default="", help="json list of draft-vocab token ids (default: the model dir's mtp_draft_vocab_ids.pt)")
ap.add_argument("--include-prompt", action="store_true", help="also train/eval on prompt positions (default: response tokens only)")
ap.add_argument("--no-cov-weight", action="store_true", help="do not weight positions by the target's in-draft-vocab mass")
ap.add_argument("--dump-hessians", default="", help="(with --eval-only) accumulate GPTQ Hessians of every MTP linear over the val set and save to this file")
args = ap.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed)
dev = torch.device("cuda")
os.makedirs(args.out, exist_ok=True)
logf = open(os.path.join(args.out, "log.txt"), "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); logf.write(s + "\n"); logf.flush()


log("args", vars(args))
# ------------------------------------------------------------------ model
cfg = AutoConfig.from_pretrained(args.model)
tc = cfg.text_config if hasattr(cfg, "text_config") else cfg
tc._attn_implementation = "sdpa"
H = tc.hidden_size
mtp_layer_idx = [i for i, t in enumerate(tc.layer_types) if t == "full_attention"][-1]
DEPTHS = args.depths
DW = [float(x) for x in args.depth_weights.split(",")] if args.depth_weights else [1.0] * DEPTHS
assert len(DW) == DEPTHS


def _causal_attn_lse(q, k, v, scale):
    """Causal SDPA that also returns the per-row log-sum-exp (memory-efficient kernel; has autograd)."""
    out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, True, scale=scale)
    return out, lse[..., : q.shape[2]].to(torch.float32)


class MTP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(H, eps=tc.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(H, eps=tc.rms_norm_eps)
        self.layer = Qwen3_5DecoderLayer(tc, mtp_layer_idx)
        self.norm = Qwen3_5RMSNorm(H, eps=tc.rms_norm_eps)
        self.rope = Qwen3_5TextRotaryEmbedding(tc)
        self._mask_cache = {}

    def _attn(self, x, cos, sin, kv_hist, mask):
        """One decoder-layer pass with explicit attention so previous depths' K/V can be
        used as history. x [B,L,H]; kv_hist: list of (k,v) [B,nkv,L,hd] from earlier depths."""
        sa = self.layer.self_attn
        B, L, _ = x.shape
        res = x
        h = self.layer.input_layernorm(x)
        qg = sa.q_proj(h).view(B, L, -1, sa.head_dim * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(B, L, -1)
        q = sa.q_norm(q).transpose(1, 2)
        k = sa.k_norm(sa.k_proj(h).view(B, L, -1, sa.head_dim)).transpose(1, 2)
        v = sa.v_proj(h).view(B, L, -1, sa.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if not kv_hist:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sa.scaling, enable_gqa=True)
        else:
            # depth >= 2: causal attention over the depth-1 history (rows <= t) merged with the
            # single-key chain terms (depths 2..d-1 and this depth, all at row t) via log-sum-exp.
            G = q.shape[1] // k.shape[1]
            k1, v1 = kv_hist[0]
            k1e = k1.repeat_interleave(G, dim=1); v1e = v1.repeat_interleave(G, dim=1)
            o1, lse1 = _causal_attn_lse(q, k1e, v1e, sa.scaling)          # [B,nh,L,hd], [B,nh,L]
            m = lse1
            terms = []
            for kk, vv in list(kv_hist[1:]) + [(k, v)]:
                s_ = (q * kk.repeat_interleave(G, dim=1)).sum(-1) * sa.scaling   # [B,nh,L]
                terms.append((s_, vv.repeat_interleave(G, dim=1)))
                m = torch.maximum(m, s_)
            w1 = torch.exp(lse1 - m)
            num = w1.unsqueeze(-1) * o1
            den = w1
            for s_, vv in terms:
                w = torch.exp(s_ - m)
                num = num + w.unsqueeze(-1) * vv
                den = den + w
            o = (num / den.unsqueeze(-1)).to(q.dtype)
        o = o.transpose(1, 2).reshape(B, L, -1) * torch.sigmoid(gate)
        x = res + sa.o_proj(o)
        res = x
        x = res + self.layer.mlp(self.layer.post_attention_layernorm(x))
        return x, (k, v)

    def depth_mask(self, L, depth):
        """[L, depth*L] bool: query row t sees depth-1 keys <= t and its own chain (col t) of
        depths 2..depth."""
        key = (L, depth)
        m = self._mask_cache.get(key)
        if m is None:
            t = torch.arange(L, device=dev)
            parts = [t[None, :] <= t[:, None]]
            eye = t[None, :] == t[:, None]
            parts += [eye] * (depth - 1)
            m = torch.cat(parts, dim=1)
            self._mask_cache[key] = m
        return m

    def forward_depths(self, h_all, x_all, n_max, depths):
        """h_all [B,Lf,H] target hidden, x_all [B,Lf] tokens. Rows t = 0..L-1 with L = Lf-1.
        Returns list over depth d of z_d [B,L,H] (pre-head, normed), row t predicting x_{t+d+1}."""
        L = n_max - 1
        outs = []
        kv_hist = []
        z_prev = None
        for d in range(1, depths + 1):
            if d == 1:
                hid = h_all[:, :L]
            else:
                hid = z_prev
            tok = x_all[:, d:d + L]                          # x_{t+d}
            e = F.embedding(tok, emb_w)
            x = self.fc(torch.cat([self.pre_fc_norm_embedding(e), self.pre_fc_norm_hidden(hid)], -1))
            pos = torch.arange(d, d + L, device=dev).unsqueeze(0).expand(x.shape[0], L)
            cos, sin = self.rope(x, pos)
            x, kv = self._attn(x, cos, sin, kv_hist, None)
            kv_hist.append(kv)
            z = self.norm(x)
            outs.append(z)
            z_prev = z
        return outs


d = args.model.rstrip("/") + "/"
mtp = MTP()
if args.init:
    with safe_open(args.init, "pt") as f:
        w = {k: f.get_tensor(k) for k in f.keys()}
else:
    with safe_open(d + "model_extra_tensors.safetensors.bak-mtp", "pt") as f:
        w = {k: f.get_tensor(k) for k in f.keys() if k.startswith("mtp.")}
name_map = {"mtp.fc.weight": "fc.weight", "mtp.norm.weight": "norm.weight",
            "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight"}
sd = {}
for k, v in w.items():
    if k.startswith("mtp.layers.0."):
        sd["layer." + k[len("mtp.layers.0."):]] = v
    elif k in name_map:
        sd[name_map[k]] = v
missing, unexpected = mtp.load_state_dict(sd, strict=False)
missing = [m for m in missing if "rope" not in m and "inv_freq" not in m]
assert not missing and not unexpected, (missing, unexpected)
mtp = mtp.to(dev).float()

idx = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
emb_shard = idx.get("model.language_model.embed_tokens.weight_packed") or idx.get("model.language_model.embed_tokens.weight")
lm_shard = idx.get("lm_head.weight_packed") or idx.get("lm_head.weight")
# quant_embed.py briefly wrote ".bak"; accept that name too, but only when the two
# shards differ -- if they are the same file, ".bak" is quant_lm_head.py's backup and
# its embed_tokens is already packed.
emb_bak = d + emb_shard + ".bak_embed"
if not os.path.exists(emb_bak) and emb_shard != lm_shard:
    emb_bak = d + emb_shard + ".bak"
with safe_open(emb_bak, "pt") as f:
    emb_w = f.get_tensor("model.language_model.embed_tokens.weight").to(dev)     # bf16 [V,H]
with safe_open(d + lm_shard + ".bak", "pt") as f:
    lm_w = f.get_tensor("lm_head.weight").to(dev)                                # bf16 [V,H]
V = lm_w.shape[0]
if args.full_vocab:
    draft_ids = None
    head_w = lm_w
else:
    if args.draft_ids:
        draft_ids = torch.tensor(sorted(set(json.load(open(args.draft_ids)))), device=dev, dtype=torch.long)
    else:
        draft_ids = torch.load(d + "mtp_draft_vocab_ids.pt").to(dev).long()
    if args.init and "mtp.draft_lm_head.weight" in w:
        head_w = w["mtp.draft_lm_head.weight"].to(dev)
    else:
        head_w = lm_w[draft_ids].clone()
head_param = nn.Parameter(head_w.float()) if args.train_head else None
log(f"MTP layer idx {mtp_layer_idx}; params {sum(p.numel() for p in mtp.parameters())/1e6:.0f}M; "
    f"draft vocab {'full' if draft_ids is None else draft_ids.numel()}; train_head={args.train_head}; depths={DEPTHS}")

# ------------------------------------------------------------------ data
seqs = json.load(open(f"{args.data}/seqs.json"))
if args.max_seqs:
    seqs = seqs[:args.max_seqs]
hid = np.load(f"{args.data}/hidden.npy", mmap_mode="r")
toks = np.load(f"{args.data}/tokens.npy", mmap_mode="r")
seqs = [s for s in seqs if s["n"] >= 8 + DEPTHS]
rng = random.Random(0)
rng.shuffle(seqs)
n_val = max(8, int(len(seqs) * args.val_frac))
# F14: never reserve the whole corpus for validation — an empty train set
# used to proceed to a training loop with no microbatch.
if len(seqs) - n_val < 1:
    n_val = max(0, len(seqs) - 1)
val_seqs, train_seqs = seqs[:n_val], seqs[n_val:]
if not train_seqs:
    raise SystemExit(f"F14: refusing to train on an empty train split "
                     f"({len(seqs)} eligible seqs, val_frac={args.val_frac})")
log(f"{len(train_seqs)} train seqs / {len(val_seqs)} val seqs; train tokens {sum(s['n'] for s in train_seqs)}")


def load_seq(s):
    o, n = s["off"], s["n"]
    h = torch.from_numpy(np.array(hid[o:o + n])).view(torch.bfloat16)
    t = torch.from_numpy(np.array(toks[o:o + n])).long()
    return h, t


def make_batch(seq_list):
    hs, ts = zip(*[load_seq(s) for s in seq_list])
    Lf = max(len(t) for t in ts)
    B = len(ts)
    h_all = torch.zeros(B, Lf, H, dtype=torch.bfloat16)
    x_all = torch.zeros(B, Lf + DEPTHS, dtype=torch.long)   # padded so x_{t+d} slices exist
    n = torch.zeros(B, dtype=torch.long)
    n_prompt = torch.zeros(B, dtype=torch.long)
    for i, (h, t) in enumerate(zip(hs, ts)):
        h_all[i, :len(t)] = h; x_all[i, :len(t)] = t; n[i] = len(t); n_prompt[i] = seq_list[i]["n_prompt"]
    return h_all, x_all, n, n_prompt


def pack(seq_list, budget):
    seq_list = sorted(seq_list, key=lambda s: -s["n"])
    batches, cur, cur_max = [], [], 0
    for s in seq_list:
        m = max(cur_max, s["n"])
        if cur and m * (len(cur) + 1) > budget:
            batches.append(cur); cur, cur_max = [], 0; m = s["n"]
        cur.append(s); cur_max = m
    if cur:
        batches.append(cur)
    return batches


# ------------------------------------------------------------------ loss
DBG = {"n": 0, "agree_r": 0, "agree_t": 0, "inv": 0, "agree_r_and_inv": 0, "agree_t_and_inv": 0, "parg_eq_true": 0}
def head_loss(z, h_tgt, train, weight, x_true=None):
    """z [N,H] fp32 draft states, h_tgt [N,H] bf16 target hidden giving the target dist.
    Returns (kl_sum, top1_sum, accept_sum) as tensors; if train, backprops (weight/N) * CE."""
    N = z.shape[0]
    z_d = z.detach().requires_grad_(train)
    hw = head_param if head_param is not None else head_w
    kl_t = torch.zeros((), device=dev); ag_t = torch.zeros((), device=dev); ac_t = torch.zeros((), device=dev)
    ok_rows = []
    for a in range(0, N, args.head_chunk):
        b = min(N, a + args.head_chunk)
        with torch.no_grad():
            tl = (h_tgt[a:b].to(dev) @ lm_w.t()).float()
            if draft_ids is not None:
                pf = torch.softmax(tl, -1)
                cov = pf[:, draft_ids].sum(-1)                  # target mass inside the draft vocab
                tl = tl[:, draft_ids]
            else:
                cov = torch.ones(b - a, device=dev)
            p = torch.softmax(tl, -1)
            p_arg = p.argmax(-1)
            ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)
            wrow = torch.ones_like(cov) if args.no_cov_weight else cov
        dl = (z_d[a:b].to(torch.bfloat16) @ hw.to(torch.bfloat16).t()).float()
        logq = torch.log_softmax(dl, -1)
        ce = -(p * logq).sum(-1)
        if train:
            ((ce * wrow).sum() * (weight / N)).backward()
        kl_t += (ce.detach() - ent).sum()
        okc = (dl.argmax(-1) == p_arg)
        if x_true is not None:   # vLLM's greedy criterion: draft == the actual next token
            am = dl.argmax(-1)
            ok_true = (draft_ids[am] if draft_ids is not None else am) == x_true[a:b]
            if os.environ.get("MTP_DEBUG"):
                inv = torch.isin(x_true[a:b], draft_ids) if draft_ids is not None else torch.ones_like(ok_true)
                DBG["n"] += ok_true.numel(); DBG["agree_r"] += okc.sum().item(); DBG["agree_t"] += ok_true.sum().item()
                DBG["inv"] += inv.sum().item(); DBG["agree_r_and_inv"] += (okc & inv).sum().item(); DBG["agree_t_and_inv"] += (ok_true & inv).sum().item()
                DBG["parg_eq_true"] += ((draft_ids[p_arg] if draft_ids is not None else p_arg) == x_true[a:b]).sum().item()
            okc = ok_true
        ok_rows.append(okc)
        ag_t += okc.float().sum()
        ac_t += torch.minimum(p, torch.softmax(dl.detach(), -1)).sum(-1).sum()
    return kl_t, ag_t, ac_t, (z_d.grad if train else None), torch.cat(ok_rows) if ok_rows else torch.zeros(0, dtype=torch.bool, device=dev)


def run_batch(seq_list, train):
    h_all, x_all, n, n_prompt = make_batch(seq_list)
    n_max = int(n.max())
    h_all_d = h_all.to(dev); x_all_d = x_all.to(dev); n_d = n.to(dev); np_d = n_prompt.to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        zs = mtp.forward_depths(h_all_d, x_all_d, n_max, DEPTHS)
    L = n_max - 1
    t = torch.arange(L, device=dev)[None, :]
    stats = []
    zs_v, grads, ok_maps = [], [], []
    B = len(seq_list)
    for dd, z in enumerate(zs, start=1):
        valid = (t + dd) <= (n_d[:, None] - 1)              # h_{t+d} exists
        if not args.include_prompt:
            valid = valid & ((t + dd + 1) >= np_d[:, None])  # predicted token x_{t+d+1} is a response token
        zz = z.float()[valid]
        idxs = (t + dd).expand(B, L)[valid]
        bidx = torch.arange(B, device=dev)[:, None].expand(B, L)[valid]
        h_tgt = h_all_d[bidx, idxs]                          # target hidden h_{t+d}
        x_true = x_all_d[:, dd + 1: dd + 1 + L][valid] if not train else None
        kl, ag, ac, g, okr = head_loss(zz, h_tgt, train, DW[dd - 1], x_true)
        stats.append((kl, ag, ac, zz.shape[0]))
        if train:
            zs_v.append(zz); grads.append(g)
        else:
            full = torch.zeros(B, L, dtype=torch.bool, device=dev); full[valid] = okr
            ok_maps.append(full)
    if train:
        torch.autograd.backward(zs_v, grads)
    if not train:
        # greedy chain simulation per sequence: like vLLM's k=DEPTHS spec decode with argmax drafts
        sim = {"steps": 0, "tokens": 0, "pos0_ok": 0}
        for b_ in range(B):
            n_ = int(n[b_]); p_ = 0 if args.include_prompt else max(0, int(n_prompt[b_]) - 2)
            oks = [ok_maps[d_][b_].cpu() for d_ in range(len(ok_maps))]
            while p_ < n_ - 1 - DEPTHS:
                acc = 0
                for d_ in range(DEPTHS):
                    if bool(oks[d_][p_]): acc += 1
                    else: break
                sim["steps"] += 1; sim["tokens"] += acc + 1; sim["pos0_ok"] += int(bool(oks[0][p_]))
                p_ += acc + 1
        stats.append(sim)
    return stats


@torch.no_grad()
def evaluate():
    mtp.eval()
    acc = [[0.0, 0.0, 0.0, 0] for _ in range(DEPTHS)]
    sim = {"steps": 0, "tokens": 0, "pos0_ok": 0}
    for b in pack(val_seqs, args.micro_tokens):
        st = run_batch(b, train=False)
        for i, (kl, ag, ac, N) in enumerate(st[:DEPTHS]):
            acc[i][0] += kl.item(); acc[i][1] += ag.item(); acc[i][2] += ac.item(); acc[i][3] += N
        for k_ in sim: sim[k_] += st[DEPTHS][k_]
    mtp.train()
    r = {}
    for i, (kl, ag, ac, N) in enumerate(acc, start=1):
        r[f"d{i}"] = {"kl": round(kl / N, 4), "top1": round(ag / N, 4), "accept": round(ac / N, 4), "n": N}
    if sim["steps"]:
        r["chain"] = {"tok_per_step": round(sim["tokens"] / sim["steps"], 3), "pos0": round(sim["pos0_ok"] / sim["steps"], 4), "steps": sim["steps"]}
    return r


if args.eval_only:
    if os.environ.get("MTP_DEBUG"):
        r0 = evaluate(); log("EVAL", json.dumps(r0)); log("DBG", {k: (v / max(1, DBG["n"]) if k != "n" else v) for k, v in DBG.items()}); sys.exit(0)
    hooks = []
    if args.dump_hessians:
        from gptq_utils import accumulate_hessian
        HS = {}
        lin = {"mtp.fc": mtp.fc,
               "mtp.layers.0.self_attn.q_proj": mtp.layer.self_attn.q_proj,
               "mtp.layers.0.self_attn.k_proj": mtp.layer.self_attn.k_proj,
               "mtp.layers.0.self_attn.v_proj": mtp.layer.self_attn.v_proj,
               "mtp.layers.0.self_attn.o_proj": mtp.layer.self_attn.o_proj,
               "mtp.layers.0.mlp.gate_proj": mtp.layer.mlp.gate_proj,
               "mtp.layers.0.mlp.up_proj": mtp.layer.mlp.up_proj,
               "mtp.layers.0.mlp.down_proj": mtp.layer.mlp.down_proj}

        def mk(name):
            def hook(mod, inp, out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                if name not in HS:
                    HS[name] = [torch.zeros(x.shape[-1], x.shape[-1], device=dev), 0]
                HS[name][0], HS[name][1] = accumulate_hessian(HS[name][0], x, HS[name][1])
            return hook
        for n_, m_ in lin.items():
            hooks.append(m_.register_forward_hook(mk(n_)))
    log("EVAL", json.dumps(evaluate()))
    if args.dump_hessians:
        torch.save({k: v[0].cpu() for k, v in HS.items()}, args.dump_hessians)
        log("hessians saved:", {k: v[1] for k, v in HS.items()})
    sys.exit(0)

# ------------------------------------------------------------------ train
groups = [{"params": list(mtp.parameters()), "lr": args.lr}]
if head_param is not None:
    groups.append({"params": [head_param], "lr": args.head_lr})
opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
base_lrs = [args.lr, args.head_lr][:len(groups)]
micro = pack(train_seqs, args.micro_tokens)
tokens_per_epoch = sum(s["n"] for s in train_seqs)
steps_per_epoch = max(1, tokens_per_epoch // args.tokens_per_step)
total_steps = int(steps_per_epoch * args.epochs)
log(f"micro-batches/epoch {len(micro)}, ~{steps_per_epoch} optimizer steps/epoch, {total_steps} total")


def lr_at(step):
    if step < args.warmup:
        return (step + 1) / args.warmup
    pr = (step - args.warmup) / max(1, total_steps - args.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))


def save():
    out = {}
    for k, v in mtp.state_dict().items():
        if "rope" in k or "inv_freq" in k or "_mask_cache" in k:
            continue
        if k.startswith("layer."):
            out["mtp.layers.0." + k[len("layer."):]] = v.to(torch.bfloat16).cpu()
        else:
            out["mtp." + k] = v.to(torch.bfloat16).cpu()
    if head_param is not None:
        out["mtp.draft_lm_head.weight"] = head_param.detach().to(torch.bfloat16).cpu()
    save_file(out, os.path.join(args.out, "mtp_bf16.safetensors"), metadata={"format": "pt"})
    # F14 row identity: a custom --draft-ids list used to live only in the
    # caller's shell history while export attached whatever IDs the source
    # model dir happened to carry. Bundle the effective IDs (and their
    # provenance) with every checkpoint so export can verify row identity.
    _ids_path = os.path.join(args.out, "draft_vocab_ids.json")
    _ids_src = args.draft_ids or (d + "mtp_draft_vocab_ids.pt")
    if draft_ids is None:
        json.dump({"vocab": "full", "ids": None, "source": None,
                   "model_dir": d, "full_vocab_size": V},
                  open(_ids_path, "w"), indent=1)
    else:
        json.dump({"vocab": "draft", "ids": sorted(set(draft_ids.tolist())),
                   "source": _ids_src, "model_dir": d},
                  open(_ids_path, "w"), indent=1)


log("EVAL step 0", json.dumps(evaluate()))
step = 0; t0 = time.time(); acc_tokens = 0
run_kl = [0.0] * DEPTHS; run_ag = [0.0] * DEPTHS; run_n = [0] * DEPTHS
mtp.train()
done = False
while not done:
    rng.shuffle(micro)
    for b in micro:
        st = run_batch(b, train=True)
        for i, (kl, ag, ac, N) in enumerate(st):
            run_kl[i] += kl.item(); run_ag[i] += ag.item(); run_n[i] += N
        acc_tokens += st[0][3]
        if acc_tokens >= args.tokens_per_step:
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * lr_at(step)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1; acc_tokens = 0
            if step % args.log_every == 0:
                el = time.time() - t0
                s = " ".join(f"d{i+1}: kl {run_kl[i]/max(1,run_n[i]):.4f} top1 {run_ag[i]/max(1,run_n[i]):.4f}" for i in range(DEPTHS))
                log(f"step {step}/{total_steps} {s} lr {opt.param_groups[0]['lr']:.2e} "
                    f"{sum(run_n[0] for _ in [0])/el:.0f} tok/s(window) {el/60:.1f} min")
                run_kl = [0.0] * DEPTHS; run_ag = [0.0] * DEPTHS; run_n = [0] * DEPTHS; t0 = time.time()
            if step % args.eval_every == 0 or step == total_steps:
                log(f"EVAL step {step}", json.dumps(evaluate()))
                save()
            if step >= total_steps:
                done = True; break
log("done")
