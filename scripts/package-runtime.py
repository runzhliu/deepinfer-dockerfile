#!/usr/bin/env python3
"""Package actual native libraries rather than dangling upstream build symlinks."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


def package(source: Path, target: Path, lock_path: Path) -> dict:
    target.mkdir(parents=True, exist_ok=False)
    for name in ("deepinfer", "deep_cluster", "examples", "eval"):
        shutil.copytree(source / name, target / name, symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("README.md", "LICENSE", "pyproject.toml"):
        shutil.copy2(source / name, target / name)
    for path in source.glob("requirements*.txt"):
        shutil.copy2(path, target / path.name)
    native = list((target / "deepinfer" / "_so" / "nvidia-x86_64").glob("*.so"))
    if not native or any(p.is_symlink() or not p.is_file() for p in native):
        raise ValueError("native extension missing or not materialized")
    submodules = subprocess.check_output(
        ["git", "-C", str(source), "submodule", "status", "--recursive"], text=True
    ).splitlines()
    if any(s.startswith(("-", "+", "U")) for s in submodules):
        raise ValueError("submodules are missing, changed, or conflicted")
    dependency_report = json.loads((source / "dependency-check.json").read_text())
    if dependency_report["status"] not in ("passed", "passed_with_upstream_override"):
        raise ValueError("dependency validation failed")
    shutil.copy2(source / "dependency-check.json", target / "dependency-check.json")
    result = {
        "source": json.loads(lock_path.read_text()),
        "submodules": submodules,
        "packages": sorted(
            [{"name": d.metadata["Name"], "version": d.version} for d in importlib.metadata.distributions()],
            key=lambda d: d["name"].lower(),
        ),
        "validation": {"dependencies": dependency_report, "gpu_inference": "not_executed"},
    }
    (target / "build-info.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    package(*(Path(x) for x in sys.argv[1:4]))
