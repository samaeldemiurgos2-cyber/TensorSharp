import importlib.util
import mmap
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("muse_spark_cpu_warmup", Path(__file__).parents[1] / "validation" / "warm-muse-spark-cpu-experts.py")
WARM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WARM)


class WarmCpuExpertsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "spark-00001-of-00001.gguf"
        self.path.write_bytes(bytes(range(256)) * 2)
        self.tensors = [self.tensor(f"blk.0.ffn_{part}_exps.weight", 64 + index * 32, 16)
                        for index, part in enumerate(("gate", "up", "down"))]

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def tensor(name, offset, size):
        return SimpleNamespace(name=name, data_offset=offset, n_bytes=size,
                               tensor_type=SimpleNamespace(name="Q3_K"))

    def reader(self, path):
        return SimpleNamespace(data_offset=32, tensors=self.tensors, fields={
            "general.architecture": SimpleNamespace(contents=lambda: "muse-spark"),
            "muse-spark.block_count": SimpleNamespace(contents=lambda: 2),
            "split.count": SimpleNamespace(contents=lambda: 1)})

    def plan(self):
        return WARM.plan_ranges([self.path], 1, self.reader)

    def test_complete_numbered_shards_required(self):
        self.assertEqual([self.path], WARM.discover_shards(self.path))
        with self.assertRaisesRegex(ValueError, "Missing shard"):
            WARM.discover_shards(self.path.with_name("spark-00001-of-00002.gguf"))

    def test_only_selected_expert_ranges_are_planned(self):
        self.tensors.extend([self.tensor("blk.1.ffn_gate_exps.weight", 256, 16),
                             self.tensor("blk.0.attn_q.weight", 288, 16)])
        rows = self.plan()
        self.assertEqual([64, 96, 128], [row["offset"] for row in rows])
        self.assertEqual(48, sum(row["bytes"] for row in rows))

    def test_tensor_outside_shard_is_rejected(self):
        self.tensors[0].data_offset = 510
        with self.assertRaisesRegex(ValueError, "exceeds its shard"):
            self.plan()

    def test_tensor_overlapping_header_is_rejected(self):
        self.tensors[0].data_offset = 16
        with self.assertRaisesRegex(ValueError, "exceeds its shard"):
            self.plan()

    def test_missing_and_duplicate_experts_are_rejected(self):
        last = self.tensors.pop()
        with self.assertRaisesRegex(ValueError, "Missing selected"):
            self.plan()
        self.tensors.extend([last, last])
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            self.plan()

    def test_overlapping_ranges_are_rejected(self):
        self.tensors[1].data_offset = 70
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.plan()

    def test_warming_reads_exact_ranges_in_bounded_chunks(self):
        rows = self.plan()
        original = self.path.read_bytes()
        reads = []
        def pread(fd, count, offset):
            reads.append((offset, count))
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, count)
        with patch.object(WARM.os, "pread", pread, create=True):
            result = WARM.warm_ranges(rows, 7)
        self.assertEqual(48, result["bytes_read"])
        self.assertEqual(3, result["completed_ranges"])
        self.assertEqual([(start + shift, count) for start in (64, 96, 128)
                          for shift, count in ((0, 7), (7, 7), (14, 2))], reads)
        self.assertEqual(original, self.path.read_bytes())

    def test_changed_shard_is_rejected_before_read(self):
        rows = self.plan()
        self.path.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "changed after"):
            WARM.warm_ranges(rows, 7)

    def test_parallel_readers_keep_exact_accounting(self):
        rows = self.plan()
        progress = []
        def pread(fd, count, offset):
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, count)
        with patch.object(WARM.os, "pread", pread, create=True):
            result = WARM.warm_ranges(rows, 7, lambda *row: progress.append(row), workers=4)
        self.assertEqual(48, result["bytes_read"])
        self.assertEqual(4, result["workers"])
        self.assertEqual([16, 32, 48], [row[3] for row in progress])

    def test_residency_query_does_not_change_file(self):
        rows = self.plan()
        before = self.path.read_bytes()
        report = WARM.residency(rows)
        if WARM.sys.platform == "linux":
            self.assertEqual("measured", report["status"])
            self.assertEqual(1.0, report["resident_fraction"])
            self.assertEqual(3, len(report["ranges"]))
        else:
            self.assertEqual("unavailable", report["status"])
        self.assertEqual(before, self.path.read_bytes())

    def test_mmap_touch_covers_unaligned_partial_last_page_and_closes_views(self):
        import numpy as np
        page = mmap.PAGESIZE
        data = bytearray(page * 4)
        for offset, value in ((13, 3), (page, 5), (page * 2, 7), (page * 3, 11)):
            data[offset] = value
        self.path.write_bytes(data)
        fd = os.open(self.path, os.O_RDONLY)
        try:
            # Includes a one-byte partial final page, missed by offset::page.
            self.assertEqual((3, 15), WARM.touch_chunk(fd, 13, 2 * page - 12, np))
            # An exact end boundary must not sample the next page.
            self.assertEqual((1, 3), WARM.touch_chunk(fd, 13, page - 13, np))
            self.assertEqual((1, 11), WARM.touch_chunk(fd, 3 * page, 1, np))
        finally:
            os.close(fd)
        self.assertEqual(bytes(data), self.path.read_bytes())

    def test_mmap_warming_is_bounded_parallel_and_records_sampled_bytes(self):
        page = mmap.PAGESIZE
        self.path.write_bytes(bytes(range(256)) * (page * 4 // 256))
        stat = self.path.stat()
        row = dict(path=str(self.path), tensor="test", offset=13, bytes=2 * page + 7,
                   file_size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        mappings = []
        original = mmap.mmap
        def record_mapping(fd, length, **kwargs):
            mapped = original(fd, length, **kwargs)
            mappings.append((mapped, length))
            return mapped
        with patch.object(WARM.mmap, "mmap", record_mapping):
            result = WARM.warm_ranges([row, row], page + 23, workers=2, read_mode="mmap-touch")
        self.assertEqual(2 * row["bytes"], result["bytes_covered"])
        self.assertEqual(0, result["bytes_read"])
        self.assertEqual(8, result["bytes_sampled"])
        self.assertEqual(2 * (13 + 36), result["sample_checksum"])
        self.assertTrue(all(mapped.closed for mapped, _ in mappings))
        self.assertTrue(all(length <= page + 23 + mmap.ALLOCATIONGRANULARITY - 1 for _, length in mappings))
        self.assertEqual(1.0, WARM.residency([row])["resident_fraction"] if WARM.sys.platform == "linux" else 1.0)

    def test_mmap_mapping_closes_when_numpy_raises(self):
        maps = []
        original = mmap.mmap
        def record_mapping(*args, **kwargs):
            mapped = original(*args, **kwargs)
            maps.append(mapped)
            return mapped
        def fail_view(*args, **kwargs):
            raise RuntimeError("injected view failure")
        fd = os.open(self.path, os.O_RDONLY)
        try:
            with patch.object(WARM.mmap, "mmap", record_mapping):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    WARM.touch_chunk(fd, 64, 16, SimpleNamespace(frombuffer=fail_view, uint8=None))
        finally:
            os.close(fd)
        self.assertTrue(all(mapped.closed for mapped in maps))

    def test_hybrid_bulk_reads_then_touches_each_bounded_chunk(self):
        rows = self.plan()
        events = []
        original_touch = WARM.touch_chunk
        def pread(fd, count, offset):
            events.append(("pread", offset, count))
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, count)
        def touch(fd, offset, count, np):
            events.append(("touch", offset, count))
            return original_touch(fd, offset, count, np)
        with patch.object(WARM.os, "pread", pread, create=True), patch.object(WARM, "touch_chunk", touch):
            result = WARM.warm_ranges(rows, 7, read_mode="pread-mmap-touch")
        self.assertEqual(48, result["bytes_covered"])
        self.assertEqual(48, result["bytes_read"])
        self.assertEqual(9, result["bytes_sampled"])
        self.assertEqual(18, len(events))
        for first, second in zip(events[::2], events[1::2]):
            self.assertEqual("pread", first[0])
            self.assertEqual("touch", second[0])
            self.assertEqual(first[1:], second[1:])

    def test_interrupt_does_not_drain_queued_ranges(self):
        reads = []
        rows = self.plan() * 100
        def pread(fd, count, offset):
            reads.append((offset, count))
            time.sleep(0.01)
            os.lseek(fd, offset, os.SEEK_SET)
            return os.read(fd, count)
        def interrupt(*args):
            raise KeyboardInterrupt()
        with patch.object(WARM.os, "pread", pread, create=True):
            with self.assertRaises(KeyboardInterrupt):
                WARM.warm_ranges(rows, 32, interrupt, workers=1)
        self.assertLess(len(reads), 10)


if __name__ == "__main__":
    unittest.main()
