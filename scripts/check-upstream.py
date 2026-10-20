#!/usr/bin/env python3
"""Reject source drift before installing or compiling upstream code."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def verify(root: Path, lock: dict) -> dict:
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != lock["upstream_commit"]:
        raise ValueError(f"source commit mismatch: {actual}")
    for name, expected in lock["files_sha256"].items():
        path = root / name
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected:
            raise ValueError(f"upstream file changed: {name}")
    for name, expected in lock["vendor_commits"].items():
        entry = subprocess.check_output(["git", "-C", str(root), "ls-tree", "HEAD", name], text=True).split()
        if len(entry) < 3 or entry[0] != "160000" or entry[2] != expected:
            raise ValueError(f"vendor gitlink changed: {name}")
    return {"status": "passed", "commit": lock["upstream_commit"], "files": len(lock["files_sha256"])}


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())), indent=2))
