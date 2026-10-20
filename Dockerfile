ARG BASE_IMAGE=nvidia/cuda:13.2.0-devel-ubuntu24.04@sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d
ARG DEEPINFER_PLATFORM=linux/amd64
FROM --platform=${DEEPINFER_PLATFORM} ${BASE_IMAGE} AS toolchain

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN test "$(uname -m)" = x86_64 \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl git gcc-13 g++-13 make cmake pkg-config \
       python3.12 python3.12-venv python3.12-dev \
       libnuma-dev libibverbs-dev librdmacm-dev libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CC=gcc-13 \
    CXX=g++-13 \
    CUDA_HOME=/usr/local/cuda \
    DEEPINFER_ACCELERATOR=nvidia \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM toolchain AS build
ARG DEEPINFER_COMMIT=8b16149741a4935e1dec8cea27853f273b374c5b
ARG MAX_JOBS=4
ARG NVCC_THREADS=2
ENV TORCH_CUDA_ARCH_LIST=10.0a \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS}
WORKDIR /opt/deepinfer
COPY source.lock.json /opt/container-source.lock.json
COPY scripts/check-upstream.py scripts/package-runtime.py scripts/check-dependencies.py /opt/container-tools/
RUN [[ "${DEEPINFER_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    && git init . \
    && git remote add origin https://github.com/deepseek-ai/DeepInfer.git \
    && git fetch --depth 1 origin "${DEEPINFER_COMMIT}" \
    && git checkout --detach FETCH_HEAD \
    && test "$(git rev-parse HEAD)" = "${DEEPINFER_COMMIT}" \
    && python /opt/container-tools/check-upstream.py . /opt/container-source.lock.json \
    && python -m pip install --no-cache-dir pip==25.2 setuptools==80.9.0 wheel==0.45.1 \
    && ./build.sh \
    && python /opt/container-tools/check-dependencies.py --output /opt/deepinfer/dependency-check.json \
    && python /opt/container-tools/package-runtime.py /opt/deepinfer /opt/runtime /opt/container-source.lock.json

FROM toolchain AS runtime
ARG DEEPINFER_COMMIT=8b16149741a4935e1dec8cea27853f273b374c5b
LABEL org.opencontainers.image.title="DeepInfer NVIDIA offline runtime" \
      org.opencontainers.image.description="Community container recipe; GPU validation is a separate step" \
      org.opencontainers.image.source="https://github.com/runzhliu/deepinfer-dockerfile" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.deepinfer.upstream.revision="${DEEPINFER_COMMIT}" \
      io.deepinfer.accelerator="nvidia" \
      io.deepinfer.hardware-scope="SM100; no H20/L20/SM120 support claim"

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/runtime /opt/deepinfer
COPY scripts/entrypoint.sh /usr/local/bin/deepinfer-container
COPY scripts/doctor.py scripts/check-dependencies.py /opt/container-tools/
RUN groupadd --gid 10001 deepinfer \
    && useradd --uid 10001 --gid 10001 --home-dir /workspace --no-create-home deepinfer \
    && mkdir -p /workspace /models /var/cache/deepinfer \
    && chown 10001:10001 /workspace /var/cache/deepinfer \
    && chmod 0755 /usr/local/bin/deepinfer-container \
    && python /opt/container-tools/doctor.py --mode cpu

ENV PYTHONPATH=/opt/deepinfer \
    HOME=/workspace \
    XDG_CACHE_HOME=/var/cache/deepinfer \
    HF_HOME=/var/cache/deepinfer/huggingface \
    TORCH_EXTENSIONS_DIR=/var/cache/deepinfer/torch-extensions \
    EP_JIT_CACHE_DIR=/var/cache/deepinfer/deepep
USER 10001:10001
WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/deepinfer-container"]
CMD ["doctor", "--mode", "cpu"]
