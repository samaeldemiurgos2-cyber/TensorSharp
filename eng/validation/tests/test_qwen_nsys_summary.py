import importlib.util
from pathlib import Path
import sqlite3
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen_nsys_summary", ROOT / "qwen-nsys-summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
spec = importlib.util.spec_from_file_location("qwen_nsys_profile", ROOT / "qwen-nsys-profile.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class ProfileTests(unittest.TestCase):
    def test_union_handles_overlap_nested_and_invalid_intervals(self):
        self.assertEqual(15, summary.union_ns([(0, 10), (3, 5), (8, 12), (20, 23), (50, 49)]))
        self.assertEqual(0, summary.union_ns([]))

    def test_gpu_union_copy_bytes_and_cpu_api_are_separate(self):
        db = sqlite3.connect(":memory:")
        db.executescript('''
          CREATE TABLE StringIds (id INTEGER, value TEXT);
          INSERT INTO StringIds VALUES (1, 'nvfp4_gemm'), (2, 'gated_delta_net'), (3, 'cudaStreamSynchronize');
          CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, demangledName INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000000,5000000,1), (3000000,6000000,2);
          CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (7000000,9000000,2,4096);
          CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
          INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0,10000000,3);
        ''')
        r = summary.summarize(db, {"result": {"e2e_ms": 15}, "server_process_cpu": {"user_ms": 3, "system_ms": 1}})
        self.assertEqual(7, r["gpu_activity_union_ms"])
        self.assertEqual(8, r["gpu_first_to_last_span_ms"])
        self.assertEqual(1, r["gpu_gap_between_first_and_last_ms"])
        self.assertEqual(4096, r["gpu_categories"]["memcpy_DtoH"]["bytes"])
        self.assertEqual(4, r["gpu_categories"]["matrix_multiply"]["sum_ms"])
        self.assertEqual(10, r["host_api_tables"]["CUPTI_ACTIVITY_KIND_RUNTIME"]["inclusive_ms"])
        self.assertEqual(8, r["request_wall_minus_gpu_union_ms"])
        db.close()

    def test_missing_gpu_is_unavailable_not_zero(self):
        with sqlite3.connect(":memory:") as db:
            r = summary.summarize(db)
        self.assertIsNone(r["gpu_activity_union_ms"])
        self.assertTrue(r["missing"])
        self.assertEqual("quantization", summary.category("dequantize_row_nvfp4_cuda"))

    def test_proc_counters_keep_cpu_time_and_count_minor_major_faults(self):
        fields = ["0"] * 22
        fields[0] = "R"
        fields[7], fields[9] = "420", "3"
        fields[11], fields[12] = "12", "7"
        with patch.object(profile.Path, "read_text", return_value="123 (worker name) " + " ".join(fields)), \
                patch.object(profile.os, "sysconf", return_value=100):
            counters = profile.process_counters(123)
        self.assertEqual({"user_ms": 120.0, "system_ms": 70.0, "minor_faults": 420, "major_faults": 3}, counters)

    def test_fixture_selection_checks_digest_and_records_cap_changes(self):
        body = {"max_tokens": 128, "messages": []}
        row = {"scenario": "repeat19k", "lane": 0, "step": 1, "body": body,
               "request_sha256": profile.replay.digest(body)}
        selected = profile.select_row([row], "repeat19k", 0, 1, 1)
        self.assertEqual(128, row["body"]["max_tokens"])
        self.assertEqual(1, selected["body"]["max_tokens"])
        self.assertNotEqual(row["request_sha256"], selected["request_sha256"])
        row["request_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            profile.select_row([row], "repeat19k", 0, 1)
        with self.assertRaises(ValueError):
            profile.select_row([], "repeat19k", 0, 1)


if __name__ == "__main__":
    unittest.main()
