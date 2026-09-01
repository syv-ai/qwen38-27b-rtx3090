# Same stack as the README's venv install, frozen: Python 3.12 venv at /app/venv,
# vLLM 0.27.1 (torch 2.13 / cu130 / Triton 3.7.1), every patch in patches/ applied,
# the KVarN KV cache installed, verify.sh --install run at build time.
#
# The base image is CUDA "base" + nvcc, not "devel": vLLM's wheels bring their own
# CUDA libraries, but FlashInfer JIT-compiles its fp8-KV attention kernel with nvcc
# on first use (batch mode, CTX=long) and Triton needs a C compiler for its launchers.
# The compiled kernels and the torch.compile cache live in the /cache volume, so
# that only happens once.
#
#   docker compose --profile single up -d      (see docs/docker.md)
FROM nvidia/cuda:13.0.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      cuda-nvcc-13-0 cuda-cudart-dev-13-0 libcurand-dev-13-0 \
      build-essential patch curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.12 -m venv venv && venv/bin/pip install --upgrade pip
COPY docker/requirements.txt docker/requirements.txt
RUN venv/bin/pip install -r docker/requirements.txt

COPY . .
# The number prefix on each patch is the apply order: some patches build on
# hunks an earlier patch adds, so the glob below only works because the names
# sort in that order. A new patch takes the next free number.
RUN set -e; SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1); \
    for p in patches/*.patch; do echo "== $p"; patch -p1 -d "$SP" < "$p"; done; \
    bash kvarn/install.sh; \
    bash verify.sh --install

# HOME is a volume: torch.compile cache (~/.cache/vllm), Triton (~/.triton),
# FlashInfer JIT (~/.cache/flashinfer), HF hub cache.
RUN mkdir -p /cache /app/models && chmod 1777 /cache
ENV HOME=/cache VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_ENABLE_HF_TRANSFER=1
VOLUME ["/cache", "/app/models"]
EXPOSE 18020
ENTRYPOINT ["bash", "docker/entrypoint.sh"]
CMD ["single"]
