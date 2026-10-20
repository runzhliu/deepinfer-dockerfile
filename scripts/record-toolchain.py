#!/usr/bin/env python3
"""Record the actual native compiler separately from PyTorch's CUDA runtime."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    import torch

    path = Path(sys.argv[1])
    info = json.loads(path.read_text())
    nvcc = shutil.which("nvcc")
    if not nvcc or not nvcc.startswith("/usr/local/cuda-13.1/"):
        raise RuntimeError("native compiler is not from the selected CUDA 13.1 toolkit")
    version = subprocess.check_output([nvcc, "--version"], text=True).strip()
    if "V13.1.115" not in version:
        raise RuntimeError("native compiler version differs from the pinned version")
    info["toolchain"] = {
        "cuda_home": os.environ["CUDA_HOME"],
        "nvcc_path": nvcc,
        "nvcc_version": version,
        "cxx_version": subprocess.check_output([os.environ["CXX"], "--version"], text=True).splitlines()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "register_spill_checks": "upstream_strict_checks_enabled",
        "gpu_validation": "not_executed",
    }
    path.write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info["toolchain"], indent=2))


if __name__ == "__main__":
    main()
