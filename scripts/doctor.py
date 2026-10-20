#!/usr/bin/env python3
"""Separate CPU packaging checks from real GPU import/operation checks."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path("/opt/deepinfer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    report = {"mode": args.mode, "checks": {}, "gpu_inference": "not_executed"}
    try:
        info = json.loads((ROOT / "build-info.json").read_text())
        report["upstream_commit"] = info["source"]["upstream_commit"]
        report["toolchain"] = info.get("toolchain", {"status": "not_recorded"})
        if platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("this recipe requires linux/amd64 and Python 3.12")
        sys.path.insert(0, str(ROOT))
        deepinfer = importlib.import_module("deepinfer")
        report["checks"]["python_package"] = deepinfer.__version__
        native = list((ROOT / "deepinfer/_so/nvidia-x86_64").glob("*.so"))
        if not native or any(p.is_symlink() or not p.is_file() for p in native):
            raise RuntimeError("native extension missing or dangling")
        report["checks"]["native_files"] = [p.name for p in native]
        report["checks"]["dependencies"] = json.loads(subprocess.check_output(
            [sys.executable, "/opt/container-tools/check-dependencies.py"], text=True, stderr=subprocess.STDOUT
        ))
        for name in ("torch", "nvidia-nccl-cu13", "deep_ep", "deep_gemm", "flash_mla", "deep_select"):
            report["checks"][name] = importlib.metadata.version(name)
        report["checks"]["native_import"] = "not_executed_cpu_mode"
        if args.mode == "gpu":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("no usable CUDA GPU/driver exposed to the container")
            devices = []
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                if capability != (10, 0):
                    raise RuntimeError(f"device {i} capability {capability} is outside this recipe's upstream target SM100")
                devices.append({"index": i, "name": torch.cuda.get_device_name(i), "capability": capability})
            for name in ("deepinfer._C", "deep_ep", "deep_gemm", "flash_mla", "deep_select"):
                importlib.import_module(name)
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    x = torch.ones((16, 16), device=f"cuda:{i}")
                    if not torch.equal(x @ x, torch.full_like(x, 16)):
                        raise RuntimeError(f"basic CUDA operation failed on device {i}")
                    torch.cuda.synchronize()
            from deepinfer.entrypoints.llm import LLM
            assert LLM is not None
            report["checks"]["native_import"] = "passed"
            report["checks"]["basic_cuda_operation"] = "passed"
            report["checks"]["devices"] = devices
        report["status"] = report["checks"]["dependencies"]["status"]
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return 1
    finally:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
