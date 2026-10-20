# DeepInfer Dockerfile

简体中文 | [English](README.en.md)

为 [DeepSeek-AI / DeepInfer](https://github.com/deepseek-ai/DeepInfer) 提供的社区容器构建项目。固定源码提交、vendor 子模块与基础镜像的 **linux/amd64 平台 digest**，保留 CUDA 编译器和 GCC，支持运行时 JIT。

**当前交付是构建配方，不是已经发布的预编译镜像。** 检查结果见 [BUILD_STATUS.json](BUILD_STATUS.json)。Dockerfile 静态检查、源码输入校验和项目测试，与镜像完整构建、GPU 导入、模型正确性、压测分别验收。没有验证的步骤不会标为通过。

## 固定版本与范围

| 项目 | 配置 |
| --- | --- |
| DeepInfer | `8b16149741a4935e1dec8cea27853f273b374c5b` |
| 基础系统 | CUDA 13.2.0 devel / Ubuntu 24.04，固定 amd64 digest |
| Python / GCC | Python 3.12 / GCC 13 |
| PyTorch | 上游固定 `2.13.0+cu132` |
| NCCL | 上游构建脚本固定 `2.32.3` |
| 后端 | NVIDIA；本项目不提供 Ascend 镜像 |
| 硬件范围 | 上游声明的 SM100；不承诺 H20、L20、RTX PRO 5000 / SM120 兼容 |
| 使用入口 | 官方离线推理和引擎基准；不是 OpenAI HTTP 服务 |

锁定文件为 [source.lock.json](source.lock.json)。基础镜像使用 `sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d`，不是多架构索引 digest。更换源码版本必须一起更新源码哈希和 vendor gitlink，不能只改 `DEEPINFER_COMMIT`。

APT 包和部分 Python 传递依赖并非全部按哈希锁定，因此这里提供版本约束和构建来源记录，不承诺逐字节一致的重建。成功构建后，镜像中的 `/opt/deepinfer/build-info.json` 会记录实际安装版本、子模块提交及验收范围。

## 构建

推荐在 Linux amd64 构建机上执行，无需在构建阶段分配 GPU。需要可访问 Docker Hub、GitHub、Ubuntu 软件源、PyPI 和 PyTorch CUDA wheel 源。完整构建包含多个原生扩展，不能按普通 Python 应用估计空间和时间；先确认构建机有足够 RAM、磁盘和工作额度。

```bash
git clone https://github.com/runzhliu/deepinfer-dockerfile.git
cd deepinfer-dockerfile
make check

IMAGE_NAME=deepinfer:cu132-sm100-8b16149741a4 \
MAX_JOBS=4 NVCC_THREADS=2 \
bash scripts/build.sh
```

内存紧张时降低 `MAX_JOBS` 和 `NVCC_THREADS`。FlashMLA / DeepSelect 的上游配方包含 SM100a、SM103a 的编译目标，不能通过把环境变量改成 SM90 或 SM120 就获得该硬件支持。

如果只做 Dockerfile 静态检查：

```bash
docker build --check --platform linux/amd64 --target runtime .
```

这条命令不执行完整编译，也不验证 GPU。`.github/workflows/check.yml` 自动运行项目测试和这项检查，不会自动发布镜像。

## 三级验证

### 1. CPU 容器检查

```bash
docker run --rm --network none \
  deepinfer:cu132-sm100-8b16149741a4 doctor --mode cpu
```

检查 Python 版本、包元数据、`pip check`、DeepInfer 轻量导入和原生扩展文件是否真实存在。**不加载 CUDA 原生模块、不执行 GPU 算子，也不宣称模型可用。**

### 2. GPU 导入与基本计算

在安装了适配驱动和 NVIDIA Container Toolkit 的 SM100 主机执行：

```bash
docker run --rm --gpus all --network none \
  deepinfer:cu132-sm100-8b16149741a4 doctor --mode gpu
```

检查可见 GPU 的计算能力、原生模块导入和每卡的基础 PyTorch CUDA 矩阵计算。它仍然不是 DeepInfer 模型端到端验证，也不覆盖所有定制算子和多卡通信。

### 3. 使用本地权重跑官方离线样例

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

模型使用官方 `DeepSeek-V4.1-Flash-1017` checkpoint；目录必须包含对应配置、分词器、编码文件与权重。镜像不内置模型、不接受密钥作为构建参数。`--ep 8` 来自官方样例，不代表所有硬件的最低卡数或已验证的容量结论。容器用户为 UID/GID `10001:10001`，应确保挂载的权重可读、缓存卷可写。

默认离线配置是 32768 上下文。`--long_context` 的样例是 16389 输入 Token、4096 输出 Token，不能当作 1M 上下文验证。

多机运行需要按上游示例配置 rendezvous、rank 和通信网络；本项目不自动修改宿主机网络或 Kubernetes 集群。

## 构建产物与发布

```bash
docker image inspect deepinfer:cu132-sm100-8b16149741a4
docker run --rm deepinfer:cu132-sm100-8b16149741a4 \
  cat /opt/deepinfer/build-info.json

docker tag deepinfer:cu132-sm100-8b16149741a4 \
  registry.example.com/inference/deepinfer:cu132-sm100-8b16149741a4
docker push registry.example.com/inference/deepinfer:cu132-sm100-8b16149741a4
```

私有仓库登录在构建外完成。不要把仓库密码、模型 API key、公司网络、机器地址或存储地址写进 Dockerfile、构建上下文、镜像层或提交记录。推送后应保存目标 registry 返回的 digest；构建完成、推送完成和目标集群实拉成功是不同状态。

`build-info.json` 中的 `gpu_inference=not_executed` 描述镜像构建阶段的边界，不会因为后来运行过 GPU 测试而自动更新。后续验证结果应单独保存，避免把镜像构建记录误当成模型实验成绩。

## 为什么不是简单的 pip install

- 上游要求 C++11 ABI 的 PyTorch，原生扩展与 PyTorch / CUDA / NCCL 的组合必须一致。
- `./build.sh` 会构建固定的 vendor 算子并把 DeepInfer 扩展链接到平台目录。运行阶段必须带上真实 `.so`，不能复制一个指向构建目录的失效符号链接。
- 部分算子使用运行时 JIT；最终镜像保留 CUDA devel、GCC、头文件和可写缓存，不能直接换成缺少编译工具的 slim 镜像。
- `pip check` 不通过会终止构建，不通过删包元数据掩盖冲突。
- 没有 NVIDIA 驱动的 CPU 构建机缺少 `libcuda.so.1` 是 GPU 导入阶段的问题。不要把链接用的 CUDA stub 放进运行时动态库搜索路径冒充驱动。

## 许可证

本项目的容器配方与辅助脚本使用 [MIT](LICENSE)。DeepInfer 使用 Apache-2.0，其许可证随上游源码保留在镜像 `/opt/deepinfer/LICENSE`；其他依赖遵循各自许可证。NVIDIA 基础镜像的使用遵循其上游条款。本项目不是 DeepSeek 官方镜像。
