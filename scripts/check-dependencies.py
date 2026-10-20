#!/usr/bin/env python3
"""Record pip's actual report and allow only upstream's explicit NCCL override."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

EXPECTED = {"torch": "2.13.0+cu132", "nvidia-nccl-cu13": "2.32.3"}
KNOWN = re.compile(
    r'^torch 2\.13\.0\+cu132 has requirement nvidia-nccl-cu13==2\.29\.7'
    r'(?:; platform_system == [\"\x27]Linux[\"\x27])?, '
    r'but you have nvidia-nccl-cu13 2\.32\.3\.$'
)


def classify(code, output, versions):
    report = {"raw_pip_check_exit_code": code, "raw_pip_check": output.strip(),
              "installed_versions": versions, "upstream_overrides": []}
    if versions != EXPECTED:
        report["status"] = "failed"
        report["reason"] = "installed versions differ from the locked upstream combination"
    elif code == 0:
        report["status"] = "passed"
    elif code == 1 and len(output.strip().splitlines()) == 1 and KNOWN.fullmatch(output.strip()):
        report["status"] = "passed_with_upstream_override"
        report["upstream_overrides"] = [{
            "dependency": "nvidia-nccl-cu13", "torch_declares": "2.29.7", "upstream_installs": "2.32.3",
            "reason": "DeepInfer's pinned build.sh explicitly replaces NCCL with --no-deps",
            "source": "https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/build.sh",
        }]
    else:
        report["status"] = "failed"
        report["reason"] = "unexpected pip check conflict or error"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    versions = {k: importlib.metadata.version(k) for k in EXPECTED}
    result = subprocess.run([sys.executable, "-m", "pip", "check"], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    report = classify(result.returncode, result.stdout, versions)
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(content)
    print(content, end="")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
