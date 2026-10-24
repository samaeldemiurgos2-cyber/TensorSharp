"""Exercise provenance preparation with tiny sealed inputs, without GPUs/network."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

VALIDATION = Path(__file__).resolve().parents[1] / "validation"
SCRIPT = VALIDATION / "prepare-muse-spark-provenance.py"


class MuseSparkProvenanceProfileTests(unittest.TestCase):
    def setup_inputs(self, root):
        proof = root / "evidence"
        proof.mkdir()
        fixture = root / "TensorSharp/InferenceWeb.Tests/Fixtures/MuseSpark/unsloth_chat_template.jinja"
        fixture.parent.mkdir(parents=True)
        fixture.write_text("released template\n", encoding="utf-8")
        digest = hashlib.sha256(fixture.read_text(encoding="utf-8").encode()).hexdigest()
        quant = "UD-Q3_K_XL"
        entries = [{"rfilename": f"{quant}/Muse-Spark-Open-816B-A42B-{quant}-{i:05d}-of-00002.gguf",
                    "size": i, "lfs": {"size": i, "sha256": str(i) * 64}} for i in (1, 2)]
        projector = root / "models/muse-spark/mmproj-Q8_0.gguf"
        projector.parent.mkdir(parents=True)
        projector.write_bytes(b"projector")
        entries.append({"rfilename": projector.name, "size": projector.stat().st_size,
                        "lfs": {"size": projector.stat().st_size,
                                "sha256": hashlib.sha256(projector.read_bytes()).hexdigest()}})
        manifest = {"id": "unsloth/Muse-Spark-Open-816B-A42B-GGUF", "sha": "a" * 40, "siblings": entries}
        verification = {"status": "pass", "repo": manifest["id"], "revision": manifest["sha"],
                        "files": [{"file": e["rfilename"], "bytes": e["size"], "expected_bytes": e["size"],
                                   "sha256": e["lfs"]["sha256"], "expected_sha256": e["lfs"]["sha256"],
                                   "status": "pass"} for e in entries]}
        files = {"model-manifest.json": manifest, "download-full-hashes.json": verification,
                 "tokenizer-metadata.json": {"first_shard": str(root / "models/muse-spark" / entries[0]["rfilename"]),
                                             "template_utf8_sha256": digest}}
        for name in ("llama", "tensorsharp"):
            files[name + "-source-verification.json"] = {"status": "pass", "original_clean": True,
                                                         "revision": ("b" if name == "llama" else "c") * 40}
        for name, value in files.items():
            (proof / name).write_text(json.dumps(value), encoding="utf-8")
        (proof / "baseline-binaries.sha256").write_text(
            "d" * 64 + "  " + (root / "TensorSharp/host.dll").as_posix() + "\n" +
            "e" * 64 + "  " + (root / "llama.cpp/llama-server").as_posix() + "\n")
        template = root / "matched.jinja"
        template.write_text("matched template\n")
        for name in ("start-tensorsharp-matched.sh", "start-llama-spark.sh"):
            (root / name).write_text("#!/bin/bash\nexec echo prepared\n")
        options = {"environment": {"TS_MUSE_SPARK_STREAM_EXPERTS": "0",
                                  "TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES": "1",
                                  "TS_MUSE_SPARK_FUSE_RESIDUALS": "1"}}
        (proof / "ts-options.json").write_text(json.dumps(options))
        return proof, template

    def run_generator(self, root, proof, template, mode="layer", extra=()):
        argv = [str(SCRIPT), "--workspace", str(root), "--evidence", str(proof),
                "--manifest", str(proof / "model-manifest.json"), "--profiles", "f16",
                "--quantization", "UD-Q3_K_XL", "--cpu-moe", "16", "--devices", "8",
                "--visible-devices", "0,1,2,3,4,5,6,8", "--cpu-threads", "48",
                "--cpu-affinity", "0,1,2,3", "--layer-counts-f16", "20,6,6,6,6,6,6,6",
                "--distribution-in-engine-options", "--tensorsharp-parallelism", mode,
                "--no-prefix-cache", "--reference-template", str(template),
                "--tensorsharp-options", str(proof / "ts-options.json"), *extra]
        def run(command, **kwargs):
            if command[0] == "nvidia-smi": return "GPU fixture\n"
            if command[0] == "lscpu": return "Model name: fixture CPU\n"
            if command[0] == "uname": return "fixture kernel\n"
            if command[0] == "git":
                return "" if "status" in command else "f" * 40
            raise AssertionError(command)
        with patch.object(sys, "argv", argv), patch.object(sys, "path", [str(VALIDATION), *sys.path]), \
             patch("subprocess.check_output", side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        return {name: json.loads((proof / f"provenance-{name}-f16-planned.json").read_text())
                for name in ("tensorsharp", "llamacpp")}

    def test_tensor_and_layer_candidates_share_work_but_record_actual_strategy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof, template = self.setup_inputs(root)
            layer = self.run_generator(root, proof, template)
            tensor = self.run_generator(root, proof, template, mode="tensor")
            self.assertEqual(layer["tensorsharp"]["settings"], tensor["tensorsharp"]["settings"])
            self.assertEqual(tensor["tensorsharp"]["settings"], tensor["llamacpp"]["settings"])
            self.assertEqual({"mode": "tensor", "degree": 8},
                             tensor["tensorsharp"]["engine_options"]["distribution"])
            self.assertEqual([56, 61], tensor["llamacpp"]["engine_options"]["distribution"]["layer_ranges"][-1])
            for engine, report in tensor.items():
                self.assertFalse(report["launch_observed"])
                self.assertIsNone(report["server_log"])
                self.assertEqual(48, report["settings"]["cpu_threads"])
                self.assertEqual([0,1,2,3,4,5,6,8], report["settings"]["visible_gpu_devices"])
                self.assertNotIn("layer_ranges", report["settings"])
                self.assertEqual(3, len(report["weights_sha256"]))
                self.assertIn("taskset -c 0,1,2,3", report["server_command"])
                self.assertIn("SPARK_CPU_THREADS=48", report["server_command"])
            self.assertIn("--no-prefix-cache", tensor["tensorsharp"]["server_command"])
            self.assertIn("--no-cache-prompt", tensor["llamacpp"]["server_command"])
            self.assertIn("TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES=1", tensor["tensorsharp"]["server_command"])
            self.assertEqual(0, tensor["llamacpp"]["engine_options"]["prompt_cache_ram_mib"])
            self.assertEqual(["<|eot|>"], tensor["llamacpp"]["reference_prompt_adapter"]["request_stop"])
            projector = json.loads((proof / "projector-verification.json").read_text())
            self.assertIn("Prior complete", projector["verification_method"])

    def test_missing_failed_or_wrong_checkpoint_hash_is_rejected(self):
        for mutation in ("remove", "failed", "wrong"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                proof, template = self.setup_inputs(root)
                path = proof / "download-full-hashes.json"
                verified = json.loads(path.read_text())
                if mutation == "remove": verified["files"].pop(1)
                elif mutation == "failed": verified["files"][1]["status"] = "fail"
                else:
                    verified["files"][1]["sha256"] = "0" * 64
                    verified["files"][1]["expected_sha256"] = "0" * 64
                path.write_text(json.dumps(verified))
                with self.assertRaises(AssertionError): self.run_generator(root, proof, template)

    def test_duplicate_devices_or_wrong_layer_count_is_rejected(self):
        for tail in (("--visible-devices", "0,1,2,3,4,5,6,6"),
                     ("--layer-counts-f16", "35,7,7,7,6")):
            with self.subTest(tail=tail), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                proof, template = self.setup_inputs(root)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.run_generator(root, proof, template, extra=tail)

    def test_changed_embedded_template_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof, template = self.setup_inputs(root)
            path = proof / "tokenizer-metadata.json"
            value = json.loads(path.read_text())
            value["template_utf8_sha256"] = "0" * 64
            path.write_text(json.dumps(value))
            with self.assertRaises(AssertionError): self.run_generator(root, proof, template)

    def test_engine_environment_cannot_override_common_threads_or_devices(self):
        for key, value in (("OMP_NUM_THREADS", "32"), ("SPARK_CPU_THREADS", "32"),
                           ("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                proof, template = self.setup_inputs(root)
                path = proof / "ts-options.json"
                options = json.loads(path.read_text())
                options["environment"][key] = value
                path.write_text(json.dumps(options))
                with self.assertRaises(AssertionError):
                    self.run_generator(root, proof, template)


if __name__ == "__main__":
    unittest.main()
