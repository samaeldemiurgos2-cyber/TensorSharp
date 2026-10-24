import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("muse_spark_expert_prefetch", Path(__file__).parents[1] / "validation" / "prefetch-muse-spark-experts.py")
PREFETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFETCH)


class ExpertPrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.paths = [self.root / f"spark-{i:05d}-of-00002.gguf" for i in (1, 2)]
        self.paths[0].write_bytes(b"metadata-only fixture" * 8)
        self.paths[1].write_bytes(bytes(range(256)) * 16)
        self.layers = 62
        self.tensors = [SimpleNamespace(name=f"blk.{layer}.ffn_{part}_exps.weight",
                        data_offset=64 + (3 * layer + part_index) * 16, n_bytes=8,
                        tensor_type=SimpleNamespace(name="Q3_K"))
                        for layer in range(62) for part_index, part in enumerate(("gate", "up", "down"))]
        self.args = SimpleNamespace(model=self.paths[0], first_layer=32, last_layer=61,
                                    output=self.root / "result.json", workers=4, chunk_mib=8,
                                    read_mode="pread", check_only=False, verification=None)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def field(value):
        return SimpleNamespace(contents=lambda: value)

    def reader(self, path):
        return SimpleNamespace(data_offset=32, tensors=[] if path == self.paths[0] else self.tensors,
                               fields={"general.architecture": self.field("muse-spark"),
                                       "muse-spark.block_count": self.field(self.layers),
                                       "split.count": self.field(2),
                                       "split.no": self.field(self.paths.index(path))})

    def plan(self, first=32, last=61):
        return PREFETCH.select_ranges(self.paths, first, last, self.reader)

    def run_helper(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return PREFETCH.run(self.args, self.reader)

    def result(self):
        return json.loads(self.args.output.read_text())

    def test_metadata_only_first_shard_and_all_62_layers_validated(self):
        all_ranges, selected = self.plan()
        self.assertEqual(186, len(all_ranges))
        self.assertEqual(90, len(selected))
        self.assertEqual(set(range(32, 62)), {row["layer"] for row in selected})
        self.assertEqual({str(self.paths[1])}, {row["path"] for row in selected})
        self.assertEqual(720, sum(row["bytes"] for row in selected))

    def test_inclusive_arbitrary_bounds(self):
        self.assertEqual(3, len(self.plan(0, 0)[1]))
        self.assertEqual(3, len(self.plan(61, 61)[1]))
        self.assertEqual(9, len(self.plan(17, 19)[1]))
        self.assertEqual(186, len(self.plan(0, 61)[1]))

    def test_invalid_bounds_rejected(self):
        for first, last in ((-1, 2), (2, 1), (0, 62), (62, 62)):
            with self.subTest(first=first, last=last), self.assertRaisesRegex(ValueError, "Layer bounds"):
                self.plan(first, last)

    def test_actual_layer_count_must_be_62(self):
        for count in (61, 63):
            self.layers = count
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "62-layer"):
                self.plan()

    def test_missing_unselected_cpu_tensor_is_rejected(self):
        self.tensors.pop(0)
        with self.assertRaisesRegex(ValueError, "Missing selected"):
            self.plan()

    def test_duplicate_unselected_cpu_tensor_is_rejected(self):
        self.tensors.append(self.tensors[0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.plan()

    def test_invalid_layer_tensor_is_not_silently_ignored(self):
        self.tensors.append(SimpleNamespace(name="blk.62.ffn_up_exps.weight"))
        with self.assertRaisesRegex(ValueError, "invalid layer"):
            self.plan()

    def test_unselected_out_of_bounds_and_overlap_rejected(self):
        self.tensors[0].data_offset = 4095
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.plan()
        self.tensors[0].data_offset = self.tensors[1].data_offset
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.plan()

    def test_wrong_shard_ordinal_rejected(self):
        def bad_reader(path):
            result = self.reader(path)
            result.fields["split.no"] = self.field(1)
            return result
        with self.assertRaisesRegex(ValueError, "split.no"):
            PREFETCH.select_ranges(self.paths, 32, 61, bad_reader)

    def test_reads_only_selected_ranges_and_preserves_model_bytes(self):
        before = [path.read_bytes() for path in self.paths]
        reads = []
        def pread(fd, count, offset):
            reads.append((offset, count))
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, count)
        with patch.object(PREFETCH.WARM.os, "pread", pread, create=True):
            self.assertEqual(0, self.run_helper())
        report = self.result()
        self.assertEqual("completed", report["status"])
        self.assertEqual(186, report["validated_tensor_count"])
        self.assertEqual(90, report["completed_ranges"])
        self.assertEqual(720, report["prefetch"]["bytes_read"])
        self.assertEqual(4, report["prefetch"]["workers"])
        self.assertEqual(8 * 1024**2, report["prefetch"]["chunk_bytes"])
        self.assertEqual(sorted((row["offset"], row["bytes"]) for row in report["ranges"]), sorted(reads))
        self.assertEqual(before, [path.read_bytes() for path in self.paths])
        self.assertEqual(PREFETCH.range_hash(report["ranges"]), report["ranges_sha256"])
        self.assertEqual(2, len(report["tool_sha256"]))
        self.assertGreaterEqual(report["duration_seconds"], 0)
        self.assertIn("process_io_delta", report)

    def test_check_only_never_calls_range_reader(self):
        self.args.check_only = True
        with patch.object(PREFETCH.WARM, "warm_ranges", side_effect=AssertionError("read tensor bytes")):
            self.assertEqual(0, self.run_helper())
        report = self.result()
        self.assertEqual("checked", report["status"])
        self.assertNotIn("prefetch", report)

    def test_existing_output_is_preserved_before_inspecting_model(self):
        self.args.output.write_text("prior evidence")
        with patch.object(PREFETCH.WARM, "discover_shards", side_effect=AssertionError("model inspected")):
            with self.assertRaises(FileExistsError):
                self.run_helper()
        self.assertEqual("prior evidence", self.args.output.read_text())

    def test_model_path_cannot_be_used_as_output(self):
        self.args.output = self.paths[0]
        before = self.paths[0].read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_helper()
        self.assertEqual(before, self.paths[0].read_bytes())

    def test_header_failure_is_recorded_without_warming(self):
        self.tensors.pop(0)
        with patch.object(PREFETCH.WARM, "warm_ranges", side_effect=AssertionError("should not read")):
            self.assertEqual(1, self.run_helper())
        report = self.result()
        self.assertEqual("failed", report["status"])
        self.assertIn("Missing selected", report["error"])
        self.assertIn("finished_utc", report)

    def test_interrupt_records_completed_progress_and_io(self):
        def interrupted(rows, chunk_bytes, progress, workers, read_mode):
            progress(1, len(rows), rows[0], rows[0]["bytes"], 0.5)
            raise KeyboardInterrupt()
        with patch.object(PREFETCH.WARM, "warm_ranges", interrupted), \
             patch.object(PREFETCH.WARM, "process_io", side_effect=[{"read_bytes": 10}, {"read_bytes": 30}]):
            self.assertEqual(1, self.run_helper())
        report = self.result()
        self.assertEqual("failed", report["status"])
        self.assertIn("KeyboardInterrupt", report["error"])
        self.assertEqual(1, report["completed_ranges"])
        self.assertEqual(8, report["completed_range_bytes"])
        self.assertEqual({"read_bytes": 20}, report["process_io_delta"])

    def verification(self):
        path = self.root / "verification.json"
        data = dict(status="pass", repo="unsloth/Muse-Spark-Open-816B-A42B-GGUF", revision="test",
                    shards=[dict(file=shard.name, actual_bytes=shard.stat().st_size, size_status="pass",
                                 actual_sha256="a" * 64, sha256_status="pass") for shard in self.paths])
        path.write_text(json.dumps(data))
        return path, data

    def test_prior_hashes_preserved_with_explicit_non_rehash_scope(self):
        path, data = self.verification()
        result = PREFETCH.recorded_verification(path, self.paths)
        self.assertEqual(data, result["report"])
        self.assertEqual(PREFETCH.sha256(path), result["report_sha256"])
        self.assertIn("not recomputed", result["scope"])

    def test_verification_wrong_shards_size_or_failed_hash_rejected(self):
        for change in ("shards", "size", "hash"):
            path, data = self.verification()
            if change == "shards":
                data["shards"].pop()
            elif change == "size":
                data["shards"][0]["actual_bytes"] += 1
            else:
                data["shards"][0]["sha256_status"] = "fail"
            path.write_text(json.dumps(data))
            with self.subTest(change=change), self.assertRaises(ValueError):
                PREFETCH.recorded_verification(path, self.paths)

    def test_cli_guards_bounds_workers_chunk_and_output(self):
        base = ["--model", str(self.paths[0]), "--output", str(self.args.output), "--first-layer", "32", "--last-layer", "61"]
        for extra in (["--first-layer", "-1"], ["--last-layer", "62"], ["--workers", "0"], ["--workers", "9"], ["--chunk-mib", "0"], ["--chunk-mib", "65"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(PREFETCH, "run", side_effect=AssertionError("invalid args executed")):
                with self.assertRaises(SystemExit) as caught:
                    PREFETCH.main(base + extra)
                self.assertEqual(2, caught.exception.code)
        self.args.output.write_text("prior evidence")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            PREFETCH.main(base)
        self.assertEqual(2, caught.exception.code)
        self.assertEqual("prior evidence", self.args.output.read_text())


if __name__ == "__main__":
    unittest.main()
