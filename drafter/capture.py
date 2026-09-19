"""Capture the target model's final hidden states (the tensor vLLM feeds to the MTP
drafter as `hidden_states`, i.e. output of the final RMSNorm) for every token of every
generated sequence in data/gen.jsonl. Runs vLLM in-process (no engine multiprocessing)
and hooks GPUModelRunner._model_forward.

Output (in data/):
  hidden.bf16   memmap [T, 5120] bf16 (stored as uint16)   row t of a sequence = h_t
  tokens.i32    memmap [T] int32 token ids
  seqs.json     [{"id","src","think","off","n","n_prompt"}, ...]
Row t of a sequence is the hidden state after consuming token t (it predicts token t+1).

  python capture.py [--limit N]
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

MODEL = os.environ.get("MODEL", os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound"))
D = os.path.join(HERE, "data")
LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
HID = 5120
MAX_LEN = 8192


def main():
    from vllm import LLM, SamplingParams
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    recs = [json.loads(l) for l in open(f"{D}/gen.jsonl")]
    if LIMIT:
        recs = recs[:LIMIT]
    seqs = []
    off = 0
    for r in recs:
        ids = r["prompt_ids"] + r["output_ids"]
        if len(ids) < 8 or len(ids) > MAX_LEN:
            continue
        seqs.append({"id": r["id"], "src": r["src"], "think": r["think"], "off": off, "n": len(ids),
                     "n_prompt": len(r["prompt_ids"]), "_ids": ids})
        off += len(ids)
    T = off
    print(f"{len(seqs)} sequences, {T} tokens -> {T * HID * 2 / 2**30:.1f} GiB", flush=True)
    hid_mm = np.lib.format.open_memmap(f"{D}/hidden.npy", mode="w+", dtype=np.uint16, shape=(T, HID))
    tok_mm = np.lib.format.open_memmap(f"{D}/tokens.npy", mode="w+", dtype=np.int32, shape=(T,))
    for s in seqs:
        tok_mm[s["off"]:s["off"] + s["n"]] = np.asarray(s["_ids"], dtype=np.int32)
    tok_mm.flush()
    # F14: the manifest is the publication event. Stage it until capture is
    # proven complete — publishing seqs.json up front lets train_mtp.py run on
    # a truncated corpus with no trace.
    json.dump([{k: v for k, v in s.items() if k != "_ids"} for s in seqs], open(f"{D}/seqs.json.staging", "w"))

    # ---- hook -------------------------------------------------------------------
    state = {"written": {}, "sched": None, "rows": 0}
    orig_exec = GPUModelRunner.execute_model
    orig_fwd = GPUModelRunner._model_forward

    def execute_model(self, scheduler_output, *a, **kw):
        state["sched"] = scheduler_output
        return orig_exec(self, scheduler_output, *a, **kw)

    def _model_forward(self, input_ids=None, positions=None, intermediate_tensors=None,
                       inputs_embeds=None, **model_kwargs):
        out = orig_fwd(self, input_ids=input_ids, positions=positions,
                       intermediate_tensors=intermediate_tensors, inputs_embeds=inputs_embeds, **model_kwargs)
        so = state["sched"]
        if so is not None and so.total_num_scheduled_tokens > 0:
            hs = out[0] if isinstance(out, tuple) else out
            nst = so.num_scheduled_tokens
            req_ids = self.input_batch.req_ids[: self.input_batch.num_reqs]
            n_tot = sum(nst[r] for r in req_ids)
            h = hs[:n_tot].to(torch.bfloat16).cpu().view(torch.uint16).numpy()
            pos = 0
            for rid in req_ids:
                n = nst[rid]
                if n == 0:
                    continue
                k = int(str(rid).split("-")[0])   # LLM request ids are '<counter>-<uuid>'
                if k < len(seqs):
                    s = seqs[k]
                    w = state["written"].get(k, 0)
                    take = min(n, s["n"] - w)
                    if take > 0:
                        hid_mm[s["off"] + w: s["off"] + w + take] = h[pos: pos + take]
                        state["written"][k] = w + take
                        state["rows"] += take
                pos += n
        return out

    GPUModelRunner.execute_model = execute_model
    GPUModelRunner._model_forward = _model_forward

    llm = LLM(
        model=MODEL, served_model_name="qwen3.8-27b",
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.90)),
        max_model_len=MAX_LEN, max_num_seqs=int(os.environ.get("MAX_SEQS", 8)),
        max_num_batched_tokens=int(os.environ.get("MNBT", 8192)),
        mamba_ssm_cache_dtype="float16", language_model_only=True,
        enable_prefix_caching=False,
        compilation_config={"max_cudagraph_capture_size": 16, "custom_ops": ["+rms_norm", "+silu_and_mul"]},
    )
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    t0 = time.time()
    CH = 256
    for i in range(0, len(seqs), CH):
        batch = seqs[i:i + CH]
        llm.generate([{"prompt_token_ids": s["_ids"]} for s in batch], sp, use_tqdm=False)
        bad = [s["id"] for s in batch if state["written"].get(seqs.index(s) if False else i + batch.index(s), 0) != s["n"]]
        el = time.time() - t0
        print(f"[{i + len(batch)}/{len(seqs)}] rows={state['rows']} ({state['rows'] / el:.0f} tok/s) "
              f"{el / 60:.1f} min  incomplete={len(bad)}", flush=True)
    hid_mm.flush()
    missing = [k for k, s in enumerate(seqs) if state["written"].get(k, 0) != s["n"]]
    print("done; incomplete sequences:", len(missing), missing[:20])
    # F14: publish the staged manifest only on proven completeness (and finite
    # row counts); otherwise fail so no downstream training consumes a partial
    # corpus. Partial memmaps stay on disk for inspection but are unpublished.
    if missing:
        print(f"F14: refusing to publish: {len(missing)} of {len(seqs)} sequences incomplete")
        raise SystemExit(1)
    os.replace(f"{D}/seqs.json.staging", f"{D}/seqs.json")
    print(f"published {D}/seqs.json ({len(seqs)} sequences, {state['rows']} rows)")


if __name__ == "__main__":
    main()
