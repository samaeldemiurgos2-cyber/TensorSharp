import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("pinned_manifest",
    Path(__file__).parents[1] / "verify-pinned-model-manifest.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class PinnedManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "model.gguf").write_bytes(b"fixture-model")
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "verification.json"
        self.data = dict(repo="publisher/model", revision="a" * 40, quant="F32",
            files=[dict(path="model.gguf", size=13, sha256=hashlib.sha256(b"fixture-model").hexdigest())])

    def run_verification(self):
        self.manifest.write_text(json.dumps(self.data))
        return verifier.verify(self.manifest, self.root, self.output, block_size=3)

    def test_streams_all_bytes_and_writes_runner_compatible_report(self):
        report = self.run_verification()
        self.assertEqual("verified", report["status"])
        self.assertEqual(13, report["shards"][0]["bytes_hashed"])
        self.assertEqual(str(self.root / "model.gguf"), report["shards"][0]["path"])
        self.assertEqual(report["shards"][0]["sha256"], report["shards"][0]["actual_sha256"])

    def test_does_not_trust_prior_verified_fields(self):
        self.data["files"][0].update(verified=True, actual_sha256="f" * 64, sha256="f" * 64)
        report = self.run_verification()
        self.assertEqual("failed", report["status"])
        self.assertFalse(report["shards"][0]["verified"])
        self.assertNotEqual("f" * 64, report["shards"][0]["actual_sha256"])

    def test_size_and_missing_file_failures_are_preserved(self):
        self.data["files"][0]["size"] = 99
        self.data["files"].append(dict(path="absent.gguf", size=1, sha256="b" * 64))
        report = self.run_verification()
        self.assertEqual("failed", report["status"])
        self.assertEqual(2, len(report["failures"]))
        self.assertTrue(all(shard["status"] == "failed" for shard in report["shards"]))

    def test_parent_escape_duplicate_empty_list_and_mutable_revision_refused(self):
        for mode in ("escape", "duplicate", "empty", "revision"):
            with self.subTest(mode=mode):
                original = json.loads(json.dumps(self.data))
                if mode == "escape": self.data["files"][0]["path"] = "../outside.gguf"
                elif mode == "duplicate": self.data["files"].append(dict(self.data["files"][0]))
                elif mode == "empty": self.data["files"] = []
                else: self.data["revision"] = "main"
                with self.assertRaises(ValueError): self.run_verification()
                self.data = original
                self.assertFalse(self.output.exists())

    def test_does_not_overwrite_previous_evidence(self):
        self.output.write_text("preserve")
        with self.assertRaises(ValueError): self.run_verification()
        self.assertEqual("preserve", self.output.read_text())


if __name__ == "__main__": unittest.main()
