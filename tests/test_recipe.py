import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load("upstream_check", "check-upstream.py")
packager = load("runtime_packager", "package-runtime.py")


class SourceIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.file = self.root / "build.sh"
        self.file.write_text("#!/bin/sh\nexit 0\n")
        self.commit = "1" * 40
        self.vendor = "2" * 40
        self.lock = {"upstream_commit": self.commit,
                     "files_sha256": {"build.sh": hashlib.sha256(self.file.read_bytes()).hexdigest()},
                     "vendor_commits": {"vendor/Kernel": self.vendor}}

    def output(self, cmd, **kwargs):
        if "rev-parse" in cmd:
            return self.commit + "\n"
        return f"160000 commit {self.vendor}\tvendor/Kernel\n"

    def test_pinned_source_is_accepted(self):
        with patch.object(checker.subprocess, "check_output", side_effect=self.output):
            self.assertEqual(checker.verify(self.root, self.lock)["status"], "passed")

    def test_different_commit_is_rejected(self):
        lock = copy.deepcopy(self.lock)
        lock["upstream_commit"] = "3" * 40
        with patch.object(checker.subprocess, "check_output", side_effect=self.output):
            with self.assertRaisesRegex(ValueError, "commit mismatch"):
                checker.verify(self.root, lock)

    def test_changed_build_script_is_rejected(self):
        self.file.write_text("#!/bin/sh\nexit 1\n")
        with patch.object(checker.subprocess, "check_output", side_effect=self.output):
            with self.assertRaisesRegex(ValueError, "file changed"):
                checker.verify(self.root, self.lock)

    def test_changed_vendor_commit_is_rejected(self):
        lock = copy.deepcopy(self.lock)
        lock["vendor_commits"]["vendor/Kernel"] = "4" * 40
        with patch.object(checker.subprocess, "check_output", side_effect=self.output):
            with self.assertRaisesRegex(ValueError, "gitlink changed"):
                checker.verify(self.root, lock)


class NativePackagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        for name in ("deepinfer/_so/nvidia-x86_64", "deep_cluster", "examples", "eval", "build"):
            (self.source / name).mkdir(parents=True, exist_ok=True)
        for name in ("README.md", "LICENSE", "pyproject.toml"):
            (self.source / name).write_text("fixture\n")
        compiled = self.source / "build" / "_C.so"
        compiled.write_bytes(b"native-fixture")
        self.link = self.source / "deepinfer/_so/nvidia-x86_64/_C.so"
        self.link.symlink_to("../../../build/_C.so")
        self.lock_path = self.root / "lock.json"
        self.lock_path.write_text("{}")

    def test_native_symlink_is_materialized_and_no_gpu_pass_is_invented(self):
        with patch.object(packager.subprocess, "check_output", return_value=" 123 vendor/kernel\n"), \
             patch.object(packager.importlib.metadata, "distributions", return_value=[]):
            result = packager.package(self.source, self.root / "runtime", self.lock_path)
        native = self.root / "runtime/deepinfer/_so/nvidia-x86_64/_C.so"
        self.assertFalse(native.is_symlink())
        self.assertEqual(native.read_bytes(), b"native-fixture")
        self.assertEqual(result["validation"]["gpu_inference"], "not_executed")

    def test_changed_or_uninitialized_submodule_is_rejected(self):
        for status in ("-123 vendor/kernel\n", "+123 vendor/kernel\n", "U123 vendor/kernel\n"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as target:
                with patch.object(packager.subprocess, "check_output", return_value=status):
                    with self.assertRaisesRegex(ValueError, "submodules"):
                        packager.package(self.source, Path(target) / "runtime", self.lock_path)

    def test_build_resource_arguments_are_validated_before_docker(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/build.sh")],
                                env={"PATH": "/usr/bin:/bin", "MAX_JOBS": "0"},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("MAX_JOBS", result.stderr)


if __name__ == "__main__":
    unittest.main()
