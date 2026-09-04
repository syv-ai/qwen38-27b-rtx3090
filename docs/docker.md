# Docker, and WSL2

The container image (same stack, frozen) and an independent WSL2 reproduction with its memory caveats.

[← back to the main README](../README.md)

The container image is the same stack, frozen: Python 3.12 venv, vLLM 0.28.0 pinned
(torch 2.13 / cu130 / Triton 3.7.1), every compatible patch in `patches/` applied
(`dflash2-backport.patch` is retired because DFlash2 is native in v0.28.0), and
`verify.sh --install` run at build time, KVarN preinstalled. Host prerequisites:
an NVIDIA driver that speaks CUDA 13 (≥ 580), Docker with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured as a runtime. The 250 W power limit is a host setting
(`sudo nvidia-smi -pl 250`), the container cannot set it.

```bash
git clone https://github.com/syv-ai/qwen38-27b-rtx3090 && cd qwen38-27b-rtx3090
echo "VLLM_API_KEY=$(openssl rand -hex 24)" > .env   # all knobs live in .env (gitignored)
docker compose --profile single up -d               # or --profile batch
docker compose logs -f single
```

**The image is prebuilt**: every push to `main` builds and pushes
`ghcr.io/syv-ai/qwen38-27b-rtx3090:latest` (plus an immutable `sha-<7>` tag
per commit) from CI, with the Dockerfile's own patch application and
`verify.sh --install` as the gate — a patch that stops applying fails the
build and nothing is pushed. The first `up` pulls it (~9.5 GB,
`pull_policy: missing`); to pin a known
build, set `image: ghcr.io/syv-ai/qwen38-27b-rtx3090:sha-<7>` in a compose
override. Building locally instead still works — `docker compose build` (or
`up --build`) produces the identical image (~20 minutes) — and the `prepare` service downloads
the model into `./models` and runs the same requantization scripts as above
(CPU only, idempotent, ~20 GB + a few minutes; `FAST_VARIANT=0` in `.env`
skips the ~1 GB fast-variant download), then the server starts. The first
start also does the torch.compile / CUDA-graph / FlashInfer-JIT work (2–3
minutes); that lands in the `qwen-cache` volume, so later starts take ~1
minute. `docker compose ps` shows the healthcheck (`/health`, 15-minute start
period). Measured in the container on the 3090: single-user 112.6 / 115.7 tok/s
(e2e / decode, default sampling), batch 950 tok/s on the 128/512 × 64 row, the
same KV pools as the venv install — no container tax; the only first-start
difference is gotcha 16 below.

- Modes are compose profiles: `single` runs `single-user/start_qwen.sh`, `batch`
  runs `batch/start_qwen.sh`. One GPU, so one at a time
  (`docker compose --profile single down` before `--profile batch up -d`).
- Every start-script knob works from `.env`, which is passed straight into the
  container: `CTX=long`, `KV=kvarn`, `SPEC=dflash2`, `PREFIX_CACHE=1`, `MAX_LEN=`,
  `MAX_SEQS=`, `SPEC_ATTN=0`, `EXTRA_ARGS=...` (`prepare` also fetches the DFlash2 drafter;
  `DFLASH2=0` skips it). `PORT` (default 18020) and `MODELS_DIR` (default `./models`,
  so a venv install and the container can share one download) are read by
  compose itself.
- `docker compose run --rm single verify` runs `verify.sh` inside the container
  (GPU, patches, model). The entrypoint runs the idempotent `prepare` and then
  `verify.sh --no-server` before every start — so a missing or half-prepared
  model heals itself, and a real FAIL refuses to serve (`PREPARE=0` / `VERIFY=0`
  skip the two steps).
- Files that `prepare` writes to `./models` are root-owned: the container runs
  as root, like vLLM's own image.
- The image carries an nvcc (CUDA "base" + `cuda-nvcc`, not the 8 GB "devel"
  image) because FlashInfer JIT-compiles its fp8-KV attention kernel on first
  use (batch mode, `CTX=long`) and Triton needs a C compiler for its launchers.
- On WSL2 the batch default may fail vLLM's free-memory gate; put
  `GPU_UTIL=0.93` in `.env` (see the WSL2 notes below, an independent
  containerized reproduction that predates this compose file).

### Plain docker (no compose)

Compose is convenience, not a requirement — profiles, `.env` passing and the
separate `prepare` service, nothing the image needs. The same server, one
command, no checkout:

```bash
docker run -d --name qwen --gpus all --ipc=host -p 18020:18020 \
  -v qwen-models:/app/models -v qwen-cache:/cache \
  --restart unless-stopped ghcr.io/syv-ai/qwen38-27b-rtx3090:latest
```

- The entrypoint runs the same idempotent `prepare` before serving, so the
  empty `qwen-models` volume fills itself on the first boot (~20 GB download +
  requantization) and later boots pay a state check measured in seconds.
- `batch` after the image name is the throughput mode (still one GPU, one mode
  at a time — `docker rm -f qwen` first).
- Knobs that compose forwards from `.env` become `-e` flags:
  `-e VLLM_API_KEY=...`, `-e SPEC=dflash2 -e PREFIX_CACHE=1`,
  `-e GPU_UTIL=0.93` on WSL2, `-e EXTRA_ARGS=...`.
- To share one model download with a venv install or a compose checkout,
  bind-mount that directory instead of the named volume:
  `-v /path/to/models:/app/models`.
- `docker logs -f qwen` follows the boot; port and health endpoint are the
  same as compose (`curl localhost:18020/health`).

### WSL2 notes

An independent WSL2 reproduction at `e81fa39` used kernel
`6.6.87.2-microsoft-standard-WSL2`, Ubuntu 24.04, NVIDIA driver 591.86,
Docker Engine 29.2.0 / Compose 5.0.2, and one RTX 3090 exposed to the
container. All six launch configurations passed authenticated API/chat and
GPU-isolation checks with zero failed benchmark requests. The full failure
signatures and earlier five-profile matrix are in [issue #1](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/1).

| profile | measured cache | representative output throughput |
|---|---:|---:|
| `single-long` | 159,326 tokens, fixed as described below | 95.39 tok/s greedy, C1 |
| `single-fast` | 93,791 tokens | 114.17 tok/s greedy, C1 |
| `single-huge` | 320,000 cold / 327,272 warm | 79.84 / 81.66 tok/s, C1 sampled |
| `batch` | 201,832 tokens | 1,041.99 / 1,038.25 tok/s, C64 |
| `batch`, `KV=int4pth` | 437,414 tokens | 1,043.84 / 1,044.06 tok/s, C64 |
| `batch`, `KV=kvarn` | 334,183 cold / 350,192 warm | 843.72 / 852.42 tok/s, C64 |

Five WSL-specific behaviors are worth accounting for, and the first is a
hard abort rather than a tuning question:

1. **`SPEC=dflash2` needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1` in `.env`, on every
   `CTX` profile.** The DFlash2 drafter forces vLLM's V2 model runner, which
   allocates UVA buffers before the weights load; vLLM leaves pinned memory off by
   default under WSL2, so the container dies at `RuntimeError: UVA is not
   available` before anything model-shaped appears in the log. The buffers work
   fine on the paravirt driver. Check the spelling — `VLLM_WSL_PIN_MEMORY` is not
   a vLLM variable and reads as a silent no-op; a venv that survived an upgrade on
   hand-applied patches can hide this until it is rebuilt from a stock wheel
   ([#25](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/25)).
2. **The ordinary batch default may fail vLLM's startup free-memory gate.**
   On an otherwise clean card, WSL reported 22.75/24.0 GiB free, less than
   the 23.33 GiB requested by `GPU_UTIL=0.972`. Launching with
   `GPU_UTIL=0.93 bash batch/start_qwen.sh` retained a 201,832-token FP8
   pool, preserving the 150k context contract and expected C64 throughput.
   Keep 0.972 as the tuned native-Linux default; 0.93 is a WSL fallback.
3. **Cold and cached starts can profile different activation peaks.** A warm
   start may turn the difference into extra KV pages and leave less transient
   headroom than the cold start. For a deterministic service, compile once
   from a cold cache, record vLLM's conservative
   `Replace gpu_memory_utilization config with --kv-cache-memory=...`
   recommendation, verify that the resulting token pool exceeds
   `MAX_LEN`, and pass that machine/profile-specific byte value through
   `EXTRA_ARGS` on later starts. Stress concurrent prefill or
   `prompt_logprobs` before promoting it. Do not copy a byte value from a
   different card or profile.
4. **`expandable_segments` crashes Marlin repack under the paravirt driver, and
   the start scripts now turn it off for you on WSL.** They detect WSL from
   `/proc/sys/kernel/osrelease` (or `WSL_DISTRO_NAME`) and default to
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, printing a line saying so;
   native Linux keeps `:True`, where gotcha 3 applies. Set the variable explicitly
   in `.env` (Docker) or the environment (venv) to override either way.

   This is the most reported failure on Windows
   ([#2](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/2),
   [#26](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/26)), and it is worth
   knowing all of its faces, because none of them says "allocator". Same CUDA VMM
   rejection inside `process_weights_after_loading` / `gptq_marlin_repack`, four
   different messages:

   | signature | reported on |
   |---|---|
   | `RuntimeError: CUDA driver error: device not ready` | driver 610.74, 610.62, 610.57 (WSL 2.1.5, 2.7.11, 2.7.12) |
   | `RuntimeError: CUDA driver error: out of memory` | driver 591.86 — **not** a real OOM: same failure at `GPU_UTIL` 0.75 and 0.90 with 23 GiB free for a 16 GiB model, and `:False` fixes it at 0.93 |
   | `torch_call_dispatcher("aten::empty", ...) API call failed`, `ops.h:631` | driver 610.62 |
   | `dxgkio_make_resident: Ioctl failed: -12` in `dmesg` | alongside all of the above |

   Driver 591.86 does reproduce it, contrary to what this note said before —
   thanks to @willy92wins for the counter-example. It is not driver-version-gated,
   so the default is now WSL-gated instead.

5. **Two more things the venv path needs on WSL2 that the container does not.**
   Both from @willy92wins in
   [#2](https://github.com/syv-ai/qwen38-27b-rtx3090/issues/2):

   - **`nvcc` is not on `PATH`, and the error blames permissions.** Inductor
     shells out to a bare `nvcc` and dies with
     `PermissionError: [Errno 13] Permission denied: 'nvcc'`. It is not a
     permissions problem: `nvcc` is not on `PATH` at all, but WSL inherits the
     Windows `PATH`, which contains ACL-restricted directories, and `execvp`
     reports EACCES rather than ENOENT when the search hits one. The venv already
     ships one — prepend
     `venv/lib/python3.12/site-packages/nvidia/cu13/bin` to `PATH`. The image
     installs `cuda-nvcc` system-wide, which is why Docker never sees this.
   - **The default fp8 KV cache makes FlashInfer JIT-compile its e4m3 prefill
     kernel**, and that ninja build can fail here. `EXTRA_ARGS="--kv-cache-dtype
     auto"` sidesteps it — `EXTRA_ARGS` is last on the command line, so it wins
     over `KV_ARGS`.

6. **Windows host memory can kill the CPU-only prepare step.** An exit 137 while
   `prepare/quant_lm_head.py` is running, with little VRAM in use, is WSL/Docker
   host-memory pressure rather than a GPU OOM. Give WSL enough memory and swap in
   `%USERPROFILE%\\.wslconfig`, for example:

   ```ini
   [wsl2]
   memory=20GB
   swap=8GB
   processors=8
   ```

   Run `wsl --shutdown` after changing the file, then restart Docker Desktop.
   If the first preparation still exceeds the available host memory, set
   `FAST_VARIANT=0` in `.env` to skip the optional ~1 GB fast-variant download;
   this reduces the first-boot footprint but does not change the base model.

   Keep Linux venv files on WSL's native filesystem when the checkout is under
   `/mnt/c`. Some Ubuntu/DrvFs combinations fail during `python3 -m venv venv`
   with an `ensurepip` or `Operation not permitted` error. Create the venv under
   `$HOME` and link it into the checkout instead:

   ```bash
   python3 -m venv "$HOME/qwen38-venv"
   ln -s "$HOME/qwen38-venv" venv
   ```

   The model directory may remain on the Windows-mounted checkout; only the
   Linux venv needs native WSL storage. This workaround is for the manual WSL
   venv path. The prebuilt Docker image already contains its own venv.
