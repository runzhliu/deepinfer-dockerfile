# DeepInfer Dockerfile

[简体中文](README.md) | English

A community container build project for [DeepSeek-AI / DeepInfer](https://github.com/deepseek-ai/DeepInfer). It pins the source commit, vendor submodules, and the base image's **linux/amd64 platform digest**, retaining the CUDA compiler and GCC for runtime JIT compilation.

**The current deliverable is a build recipe, not a published prebuilt image.** See [BUILD_STATUS.json](BUILD_STATUS.json) for verification results. Dockerfile static checks, source-input validation, and project tests are separate from a complete image build, GPU imports, model correctness, and benchmarks. Unverified steps are not marked as passed.

## Pinned versions and scope

| Component | Configuration |
| --- | --- |
| DeepInfer | `8b16149741a4935e1dec8cea27853f273b374c5b` |
| Base system | CUDA 13.2.0 devel / Ubuntu 24.04, pinned amd64 digest |
| Python / GCC | Python 3.12 / GCC 13 |
| PyTorch | Upstream pin `2.13.0+cu132` |
| Native / JIT compiler | NVCC `13.1.115`, recorded separately from PyTorch's cu132 runtime |
| NCCL | Upstream build-script pin `2.32.3` |
| Backend | NVIDIA; this project does not provide an Ascend image |
| Hardware scope | Upstream's declared SM100 scope; no H20, L20, RTX PRO 5000 / SM120 compatibility claim |
| Entry points | Official offline inference and engine benchmarks, not an OpenAI HTTP server |

The lock is [source.lock.json](source.lock.json). The base image uses `sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d`, not the multi-platform index digest. Updating the source requires updating file hashes and vendor gitlinks together, rather than changing only `DEEPINFER_COMMIT`.

APT packages and some transitive Python dependencies are not fully hash-pinned. This project therefore provides version constraints and build provenance rather than a promise of byte-identical rebuilds. A successful image contains `/opt/deepinfer/build-info.json` with actual installed versions, submodule commits, and validation scope.

## Build

A Linux amd64 build host is recommended; building does not require allocating GPUs. It needs access to Docker Hub, GitHub, Ubuntu repositories, PyPI, and the PyTorch CUDA wheel repository. The complete build includes several native extensions, so budget RAM, disk, and time accordingly instead of treating it as a small Python application.

```bash
git clone https://github.com/runzhliu/deepinfer-dockerfile.git
cd deepinfer-dockerfile
make check

IMAGE_NAME=deepinfer:cu132-sm100-8b16149741a4 \
MAX_JOBS=4 NVCC_THREADS=2 \
bash scripts/build.sh
```

Reduce `MAX_JOBS` and `NVCC_THREADS` if memory is constrained. Upstream FlashMLA / DeepSelect build recipes include SM100a and SM103a compilation targets. Changing an environment variable to SM90 or SM120 does not establish support for those GPUs.

The Dockerfile follows the pinned upstream `build.sh` sequence: dependency installation, the NCCL override, `scripts/build_vendor.sh`, then `scripts/build.sh build`. Separate Docker layers preserve completed PyTorch downloads after a native-build failure. The build stage installs `pciutils` because DeepSelect's platform detection invokes `lspci` even on CPU build hosts. This supplies a missing build tool without impersonating a GPU or bypassing register-spill checks.

A full build with NVCC `13.2.78` failed FlashMLA's strict register-spill check: affected kernels reported a 16-byte stack frame, 20 bytes of spill stores, and 28 bytes of spill loads. The current recipe retries with NVCC `13.1.115` while retaining cu132 PyTorch and all upstream checks. This toolchain adjustment still requires complete build and GPU validation; it is not evidence that the issue is resolved. Successful artifacts record NVCC, GCC, `CUDA_HOME`, and the PyTorch CUDA version separately in `build-info.json`.

The first 13.1 attempt installed the compiler and CUDA runtime development components but failed in DeepSelect because `cusparse.h` was missing; it did not establish whether FlashMLA passes with that compiler. The current recipe adds `cuda-libraries-dev-13-1`, uses the same development environment for build and runtime, and checks the cuSPARSE, cuBLAS, cuSOLVER, and cuRAND headers before downloading PyTorch. It also compiles a minimal CUDA translation unit including those headers. This validates development-toolkit completeness without running GPU tests.

### Installation references

The [deepinfer.run installation page](https://deepinfer.run/install/) identifies itself as an unaffiliated community guide and largely follows the [official README](https://github.com/deepseek-ai/DeepInfer#readme). It is useful for checking Python, GCC, accelerator selection, and offline entry points. The page does not provide a prebuilt image, a FlashMLA spill fix, or an H20 compatibility path, so it cannot replace actual build verification.

For Dockerfile static checks only:

```bash
docker build --check --platform linux/amd64 --target runtime .
```

This does not perform full compilation or GPU validation. `.github/workflows/check.yml` automatically runs project tests and this check, without automatically publishing an image.

## Three validation levels

### 1. CPU container checks

```bash
docker run --rm --network none \
  deepinfer:cu132-sm100-8b16149741a4 doctor --mode cpu
```

Checks Python, package metadata, `pip check`, lightweight DeepInfer import, and the existence of real native-extension files. **It does not load CUDA native modules, execute GPU operators, or establish model readiness.**

Upstream has one explicit dependency override: the PyTorch `2.13.0+cu132` wheel metadata requires NCCL `2.29.7`, while the pinned [DeepInfer build script](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/build.sh) force-installs `2.32.3`. The checker preserves the original `pip check` output and failing exit code, permits only this exact combination, and reports `passed_with_upstream_override`. Every other conflict still fails the build. The report is saved at `/opt/deepinfer/dependency-check.json`; package metadata is neither removed nor rewritten, and this exception does not establish GPU compatibility.

### 2. GPU imports and basic computation

Run on an SM100 host with a compatible driver and NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all --network none \
  deepinfer:cu132-sm100-8b16149741a4 doctor --mode gpu
```

Checks visible GPU compute capabilities, native-module imports, and a basic PyTorch CUDA matrix operation on each device. This is still not end-to-end DeepInfer model validation and does not cover every custom operator or multi-GPU communication path.

### 3. Official offline example with local weights

```bash
docker volume create deepinfer-cache

docker run --rm --gpus all --shm-size 32g \
  --mount type=bind,src=/absolute/path/to/model,dst=/models/model,readonly \
  --mount type=volume,src=deepinfer-cache,dst=/var/cache/deepinfer \
  -e DEEPINFER_HF_MODEL_DIR=/models/model \
  -e HF_HUB_OFFLINE=1 \
  deepinfer:cu132-sm100-8b16149741a4 \
  offline --ep 8 --check_structured_output
```

Use the official `DeepSeek-V4.1-Flash-1017` checkpoint, including its configuration, tokenizer, encoding files, and weights. Weights are not embedded, and secrets are not accepted as build arguments. `--ep 8` comes from the upstream example; it is neither a universal minimum GPU count nor a validated capacity claim. The container runs as UID/GID `10001:10001`; mounted weights must be readable and cache storage writable.

The default offline configuration uses a 32768 context. The `--long_context` sample uses 16389 input tokens and 4096 output tokens; it does not validate 1M context.

Multi-node execution requires upstream rendezvous, rank, and communication-network configuration. This project does not automatically modify host networking or a Kubernetes cluster.

## Image artifacts and publication

```bash
docker image inspect deepinfer:cu132-sm100-8b16149741a4
docker run --rm deepinfer:cu132-sm100-8b16149741a4 \
  cat /opt/deepinfer/build-info.json

docker tag deepinfer:cu132-sm100-8b16149741a4 \
  registry.example.com/inference/deepinfer:cu132-sm100-8b16149741a4
docker push registry.example.com/inference/deepinfer:cu132-sm100-8b16149741a4
```

Authenticate to private registries outside the build. Keep registry passwords, model API keys, company networks, host addresses, and storage endpoints out of Dockerfiles, build contexts, image layers, and commits. Record the destination registry digest after pushing. Build completion, push completion, and successful pulling on the target cluster are separate states.

`gpu_inference=not_executed` in `build-info.json` describes the image-build boundary and does not update automatically after later GPU tests. Save subsequent validation separately so image provenance is not mistaken for experimental model results.

## Why this is more than pip install

- Upstream requires a C++11-ABI PyTorch build. Native extensions must match the PyTorch / CUDA / NCCL combination.
- `./build.sh` compiles pinned vendor operators and links the DeepInfer extension into a platform directory. The runtime needs actual `.so` files rather than dangling links into a discarded build directory.
- Some operators use runtime JIT. The image retains CUDA devel, GCC, headers, and writable caches rather than switching to a slim image without compilers.
- Dependency checks permit only the explicit upstream NCCL override described above. Every other `pip check` error stops the build; package metadata is not removed to hide conflicts.
- Missing `libcuda.so.1` on a CPU build host is a GPU-import boundary. CUDA linker stubs must not be placed on the runtime library search path to impersonate a driver.

## Licensing

Container recipes and helper scripts use [MIT](LICENSE). DeepInfer is Apache-2.0 licensed, with its upstream license preserved at `/opt/deepinfer/LICENSE` in the image. Dependencies retain their own licenses. The NVIDIA base image remains subject to its upstream terms. This is not an official DeepSeek image.
