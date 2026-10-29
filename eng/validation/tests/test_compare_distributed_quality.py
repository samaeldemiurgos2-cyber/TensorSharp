"""Teacher-forced quality scoring maps every probe vector to its true next token."""
from array import array
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("compare_distributed_quality",
    Path(__file__).parents[1] / "compare-distributed-quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)

VOCAB = 64
TOKENS = [5, 9, 17, 33, 2, 40, 7]  # prefix P = 4 with decode_count 3
BACKENDS = (False, True) if quality.np is not None else (False,)


def program_schedule(tokens, decode_count, chunk):
    """Replicate DeepSeek41DistributedProbe/Program.cs lines 176-184 with end positions."""
    prefix = len(tokens) - decode_count
    plan = [("reset", None, 0), ("prefill", tokens[:prefix], prefix)]
    plan += [("decode", [tokens[i]], i + 1) for i in range(prefix, len(tokens))]
    plan.append(("reset", None, 0))
    plan += [("chunk", tokens[i:min(prefix, i + chunk)], min(prefix, i + chunk)) for i in range(0, prefix, chunk)]
    plan += [("refill", [tokens[i]], i + 1) for i in range(prefix, len(tokens))]
    plan += [("reset", None, 0), ("reset-prefill", tokens[:prefix], prefix)]
    return plan


def context_logits(position, seed=0):
    """A deterministic 'model': the same context yields the same reference vector in every phase."""
    rng = random.Random(1000 * seed + position)
    return [rng.gauss(0., 2.) for _ in range(VOCAB)]


def f32(values):
    data = array("f", values)
    return list(data), data


def write_probe(directory, vector, tokens=TOKENS, decode_count=3, chunk=2, text=True, **overrides):
    """Write a probe directory with Program.cs's probe.json schema and step-NNN-op.f32 files."""
    directory.mkdir(parents=True)
    token_bytes = json.dumps(tokens, separators=(",", ":")).encode()
    report = dict(schema_version=1, status="running", steps=[], weights_id="weights-sha256",
                  arguments=["--model", "m.gguf", "--text", "t.txt", "--weights-id", "weights-sha256",
                             "--output", str(directory), "--decode-count", str(decode_count), "--chunk", str(chunk)],
                  limitations=["Fixed-token numerical/control probe; not a model-quality or throughput benchmark."])
    if text:
        (directory / "tokens.json").write_bytes(token_bytes)
        report.update(text_sha256=hashlib.sha256(b"fixture text").hexdigest(), text="fixture text")
    report.update(tokens=tokens, tokens_sha256=hashlib.sha256(token_bytes).hexdigest(),
                  cpu_offload=dict(requested_layers=None, requested_all=None, requested_threads=None,
                                   configured_layers=0, configured_all=False, explicitly_set=False, configured_threads=0),
                  requested_kv_cache_dtype="f16", load_seconds=1.5,
                  topology=dict(mode="tensor_parallel", local_degree=1, global_degree=2, rank_offset=0, node_count=2),
                  vocab_size=VOCAB)
    for operation, inputs, position in program_schedule(tokens, decode_count, chunk):
        steps = report["steps"]
        if inputs is None:
            steps.append(dict(operation=operation, tokens=[], logits=None))
            continue
        values, data = f32(vector(len(steps), operation, position))
        if sys.byteorder != "little":
            data.byteswap()
        name = f"step-{len(steps):03d}-{operation}.f32"
        (directory / name).write_bytes(data.tobytes())
        steps.append(dict(operation=operation, tokens=inputs, logits=name, count=len(values),
                          argmax=values.index(max(values)), seconds=0.01))
    report.update(cache_truncation="not_requested; the distributed executor serves a single mirrored sequence",
                  status="completed", disposed=True)
    report.update(overrides)
    (directory / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return directory


def reference_vector(index, operation, position):
    return context_logits(position)


def noisy(scale, seed=7):
    def vector(index, operation, position):
        rng = random.Random(seed * 100003 + index)
        return [x + rng.gauss(0., scale) for x in context_logits(position)]
    return vector


class QualityComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference = write_probe(self.root / "reference", reference_vector)

    def candidate(self, vector=reference_vector, name="candidate", **options):
        return write_probe(self.root / name, vector, **options)

    def run_cli(self, *arguments):
        with patch.object(sys, "argv", ["compare-distributed-quality.py", *map(str, arguments)]), \
                patch("builtins.print") as printed:
            code = quality.main()
        return code, "\n".join(str(call.args[0]) for call in printed.call_args_list)

    def test_targets_follow_the_probe_schedule(self):
        steps = json.loads((self.reference / "probe.json").read_text())["steps"]
        rows, failures = quality.teacher_targets(TOKENS, steps)
        self.assertEqual([], failures)
        self.assertEqual([(1, "prefill", 4, 2), (2, "decode", 5, 40), (3, "decode", 6, 7), (4, "decode", 7, None),
                          (6, "chunk", 2, 17), (7, "chunk", 4, 2), (8, "refill", 5, 40), (9, "refill", 6, 7),
                          (10, "refill", 7, None), (12, "reset-prefill", 4, 2)],
                         [(row["index"], row["operation"], row["target_position"], row["target_token"]) for row in rows])
        uneven = write_probe(self.root / "uneven", reference_vector, chunk=3)
        rows, failures = quality.teacher_targets(TOKENS, json.loads((uneven / "probe.json").read_text())["steps"])
        self.assertEqual([], failures)
        self.assertEqual([(3, 33), (4, 2)], [(row["target_position"], row["target_token"])
                                             for row in rows if row["operation"] == "chunk"])

    def test_targets_refuse_inputs_that_leave_the_teacher_stream(self):
        reset = dict(operation="reset", tokens=[], logits=None)
        forward = lambda tokens: dict(operation="decode", tokens=tokens, logits="x.f32", count=VOCAB)
        cases = {"no reset": [forward([5])], "wrong token": [reset, forward([5]), forward([17]), forward([17])],
                 "unknown control": [reset, forward([5]), dict(operation="truncate", tokens=[1], logits=None), forward([9])]}
        for name, steps in cases.items():
            with self.subTest(name=name):
                rows, failures = quality.teacher_targets(TOKENS, steps)
                self.assertTrue(failures)
                self.assertTrue(all(row["target_token"] in (None, TOKENS[row["target_position"]]) for row in rows))
        rows, failures = quality.teacher_targets(TOKENS, cases["wrong token"])
        self.assertEqual([1], [row["index"] for row in rows])  # nothing after the divergence is scored

    def test_self_check_is_exact_on_both_backends(self):
        for use_numpy in BACKENDS:
            with self.subTest(use_numpy=use_numpy):
                report = quality.compare(self.reference, self.reference, self_check=True, use_numpy=use_numpy)
                self.assertEqual("passed", report["status"], report["failures"])
                self.assertTrue(report["self_check"]["passed"])
                overall = report["aggregates"]["all"]
                self.assertEqual(1.0, overall["ppl_ratio"])
                self.assertEqual(0.0, overall["mean_kl_reference_to_candidate"])
                self.assertEqual(0.0, overall["distributions"]["kl_candidate_to_reference"]["max"])
                self.assertEqual(1.0, overall["top1_agreement"])
                self.assertEqual(overall["vectors"], overall["bitwise_equal_vectors"])
        output = self.root / "self.json"
        code, text = self.run_cli("--self-check", self.reference, "--output", output)
        self.assertEqual(0, code)
        self.assertIn("status: passed (self-check", text)
        self.assertEqual("self-check", json.loads(output.read_text())["mode"])

    def test_metrics_match_hand_computation(self):
        reference, _ = f32([2.0, 1.0, 0.5, -1.0, 3.0, 0.0])
        candidate, _ = f32([2.1, 0.9, 0.5, -1.0, 2.9, 0.25])
        def log_softmax(values):
            total = sum(math.exp(x) for x in values)
            return [x - math.log(total) for x in values]
        la, lb = log_softmax(reference), log_softmax(candidate)
        kl_rc = sum(math.exp(p) * (p - q) for p, q in zip(la, lb))
        kl_cr = sum(math.exp(q) * (q - p) for p, q in zip(la, lb))
        for use_numpy in BACKENDS:
            with self.subTest(use_numpy=use_numpy):
                convert = (lambda v: quality.np.array(v, dtype=quality.np.float64)) if use_numpy else list
                row = quality.vector_metrics(convert(reference), convert(candidate), 1, use_numpy)
                self.assertAlmostEqual(-la[1], row["reference_nll"], places=12)
                self.assertAlmostEqual(-lb[1], row["candidate_nll"], places=12)
                self.assertAlmostEqual(kl_rc, row["kl_reference_to_candidate"], places=12)
                self.assertAlmostEqual(kl_cr, row["kl_candidate_to_reference"], places=12)
                self.assertGreater(row["kl_reference_to_candidate"], 0)
                self.assertEqual((4, 4, True), (row["reference_argmax"], row["candidate_argmax"], row["argmax_match"]))
                self.assertEqual([4, 0, 1, 2, 5], row["reference_top5"])
                self.assertEqual([4, 0, 1, 2, 5], row["candidate_top5"])
                self.assertEqual((3, 3), (row["reference_target_rank"], row["candidate_target_rank"]))
                self.assertAlmostEqual(max(abs(x - y) for x, y in zip(reference, candidate)), row["max_abs_logit_diff"], places=12)
                l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(reference, candidate))) / math.sqrt(sum(x * x for x in reference))
                self.assertAlmostEqual(l2, row["relative_l2"], places=12)

    def test_top_k_ties_break_by_lowest_index_on_both_backends(self):
        values = [1.0, 3.0, 3.0, 0.0, 3.0, 2.0, 3.0, 3.0]
        for use_numpy in BACKENDS:
            with self.subTest(use_numpy=use_numpy):
                vector = quality.np.array(values) if use_numpy else values
                row = quality.vector_metrics(vector, vector, None, use_numpy)
                self.assertEqual([1, 2, 4, 6, 7], row["reference_top5"])
                self.assertEqual(1, row["reference_argmax"])  # Program.cs: Array.IndexOf(logits, logits.Max())
                self.assertNotIn("reference_nll", row)

    @unittest.skipIf(quality.np is None, "numpy is not installed")
    def test_numpy_and_pure_python_reports_agree(self):
        candidate = self.candidate(noisy(.05))
        fast = quality.compare(self.reference, candidate, use_numpy=True)
        slow = quality.compare(self.reference, candidate, use_numpy=False)
        self.assertEqual(("numpy", "pure-python"), (fast["backend"]["name"], slow["backend"]["name"]))
        self.assertEqual(len(fast["vectors"]), len(slow["vectors"]))
        for left, right in zip(fast["vectors"], slow["vectors"]):
            for key, value in left.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(value, right[key], delta=1e-12 + 1e-9 * abs(value), msg=key)
                else:
                    self.assertEqual(value, right[key], key)
        self.assertEqual(fast["status"], slow["status"])

    def test_small_noise_passes_and_reports_every_phase(self):
        report = quality.compare(self.reference, self.candidate(noisy(.002)), per_phase_gates=True)
        self.assertEqual("passed", report["status"], report["failures"])
        self.assertEqual(["prefill", "decode", "chunk", "refill", "reset-prefill"], list(report["phases"]))
        self.assertEqual((10, 8), (report["aggregates"]["all"]["vectors"], report["aggregates"]["all"]["scored_vectors"]))
        unique = report["aggregates"]["unique_targets"]
        # tokens[4..6] from prefill + two scored decodes, plus the interior chunk boundary tokens[2].
        self.assertEqual(4, unique["scored_vectors"])
        self.assertEqual([4, 5, 6, 2], [row["target_position"] for row in report["vectors"]
                                        if row["index"] in (1, 2, 3, 6)])
        self.assertEqual({"decode": 2, "refill": 2, "chunk": 2, "prefill": 1, "reset-prefill": 1},
                         {phase: summary["scored_vectors"] for phase, summary in report["phases"].items()})
        self.assertTrue(all(gate["passed"] for gates in report["phase_gates"].values() for gate in gates.values()))
        self.assertTrue(any("unique target tokens" in warning for warning in report["warnings"]))
        self.assertLess(0, report["aggregates"]["all"]["mean_kl_reference_to_candidate"])
        self.assertEqual(0, report["aggregates"]["all"]["bitwise_equal_vectors"])

    def test_gates_fail_on_divergence_and_cli_exit_code_follows(self):
        broken = self.candidate(noisy(3.0), name="broken")
        report = quality.compare(self.reference, broken)
        self.assertEqual("failed", report["status"])
        self.assertFalse(report["gates"]["max_mean_kl"]["passed"])
        self.assertFalse(report["gates"]["min_top1"]["passed"])
        output = self.root / "broken.json"
        code, text = self.run_cli(self.reference, broken, "--output", output, "--max-mean-kl", "1000",
                                  "--min-top1", "0", "--max-ppl-ratio", "1000")
        self.assertEqual(0, code, text)
        code, text = self.run_cli(self.reference, broken, "--output", output)
        self.assertEqual(1, code)
        self.assertIn("gate max_mean_kl (all):", text)
        self.assertIn("FAIL", text)
        self.assertEqual("failed", json.loads(output.read_text())["status"])

    def test_perplexity_gate_is_one_sided(self):
        def boosted(amount):
            def vector(index, operation, position):
                values = context_logits(position)
                if position < len(TOKENS):
                    values[TOKENS[position]] += amount
                return values
            return vector
        better = quality.compare(self.reference, self.candidate(boosted(.5), name="better"),
                                 max_mean_kl=10, min_top1=0)
        self.assertLess(better["gates"]["max_ppl_ratio"]["value"], 1)
        self.assertTrue(better["gates"]["max_ppl_ratio"]["passed"])
        worse = quality.compare(self.reference, self.candidate(boosted(-.5), name="worse"),
                                max_mean_kl=10, min_top1=0)
        self.assertGreater(worse["gates"]["max_ppl_ratio"]["value"], 1.01)
        self.assertEqual(["Gate max_ppl_ratio (all) failed"], worse["failures"])

    def test_per_phase_gates_catch_one_broken_phase(self):
        tokens = list(range(3, 53))  # 50 tokens, decode_count 40 -> prefix 10, one 10-token chunk
        reference = write_probe(self.root / "long-reference", reference_vector, tokens=tokens, decode_count=40, chunk=10)
        def flipped_chunk(index, operation, position):
            values = context_logits(position)
            if operation == "chunk":
                order = sorted(range(VOCAB), key=lambda i: -values[i])
                values[order[1]] = values[order[0]] + 1e-3
            return values
        candidate = write_probe(self.root / "long-candidate", flipped_chunk, tokens=tokens, decode_count=40, chunk=10)
        overall = quality.compare(reference, candidate)
        self.assertEqual("passed", overall["status"], overall["failures"])
        self.assertLess(overall["aggregates"]["all"]["top1_agreement"], 1)
        phased = quality.compare(reference, candidate, per_phase_gates=True)
        self.assertEqual("failed", phased["status"])
        self.assertIn("Gate min_top1 (chunk) failed", phased["failures"])
        self.assertTrue(all("(chunk)" in message for message in phased["failures"]), phased["failures"])
        self.assertEqual(40, phased["aggregates"]["unique_targets"]["scored_vectors"])
        scoped = quality.compare(reference, candidate, gate_scope="unique_targets")
        self.assertEqual("unique_targets", scoped["gates"]["min_top1"]["scope"])
        self.assertEqual(1.0, scoped["gates"]["min_top1"]["value"])

    def test_hashes_and_run_identity_are_recorded(self):
        report = quality.compare(self.reference, self.candidate(noisy(.001)))
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest(self.reference / "probe.json"), report["reference"]["probe_json_sha256"])
        self.assertEqual(digest(self.reference / "tokens.json"), report["reference"]["tokens_json_sha256"])
        self.assertEqual("text", report["reference"]["token_source"])
        for row in report["vectors"]:
            self.assertEqual(digest(self.reference / row["reference_file"]), row["reference_sha256"])
            self.assertEqual(digest(self.root / "candidate" / row["candidate_file"]), row["candidate_sha256"])
        self.assertEqual(64, len(report["comparator_sha256"]))
        json.dumps(report, allow_nan=False)

    def test_identity_or_artifact_problems_fail(self):
        base = json.loads((self.reference / "probe.json").read_text())
        def mutate_report(field, value):
            def apply(directory):
                report = json.loads((directory / "probe.json").read_text())
                if callable(value):
                    value(report)
                else:
                    report[field] = value
                (directory / "probe.json").write_text(json.dumps(report))
            return apply
        def rewrite_vector(raw):
            def apply(directory):
                (directory / base["steps"][2]["logits"]).write_bytes(raw)
            return apply
        nan_vector = array("f", context_logits(5))
        nan_vector[3] = float("nan")
        cases = {
            "failed run": mutate_report("status", "failed"),
            "undisposed": mutate_report("disposed", False),
            "worker report": mutate_report("status", "worker_loop_returned"),
            "schema": mutate_report("schema_version", 2),
            "weights": mutate_report("weights_id", "other"),
            "vocab": mutate_report("vocab_size", VOCAB + 1),
            "text": mutate_report("text_sha256", "0" * 64),
            "tokens": mutate_report("tokens", TOKENS[:-1] + [8]),
            "step operation": mutate_report(None, lambda r: r["steps"][2].update(operation="refill")),
            "recorded argmax": mutate_report(None, lambda r: r["steps"][2].update(argmax=(r["steps"][2]["argmax"] + 1) % VOCAB)),
            "outside vector": mutate_report(None, lambda r: r["steps"][2].update(logits="../reference/" + r["steps"][2]["logits"])),
            "missing steps": mutate_report("steps", base["steps"][:-1]),
            "tokens.json": lambda directory: (directory / "tokens.json").write_text("[1,2]"),
            "truncated vector": rewrite_vector(b"\0" * (VOCAB * 4 - 4)),
            "nonfinite vector": rewrite_vector(nan_vector.tobytes()),
        }
        for name, apply in cases.items():
            with self.subTest(name=name):
                candidate = self.candidate(name="case-" + name.replace(" ", "-").replace(".", ""))
                apply(candidate)
                report = quality.compare(self.reference, candidate)
                self.assertEqual("failed", report["status"])
                self.assertTrue(report["failures"])
                quality.table(report)  # a failed report still renders

    def test_different_weights_can_be_downgraded_to_a_warning(self):
        candidate = self.candidate(noisy(.001), weights_id="quantized")
        self.assertEqual("failed", quality.compare(self.reference, candidate)["status"])
        report = quality.compare(self.reference, candidate, allow_different_weights=True)
        self.assertEqual("passed", report["status"], report["failures"])
        self.assertIn("Missing or different weights_id", report["warnings"])

    def test_token_file_runs_are_scored_with_a_warning(self):
        reference = write_probe(self.root / "tokens-reference", reference_vector, text=False)
        candidate = write_probe(self.root / "tokens-candidate", noisy(.001), text=False)
        report = quality.compare(reference, candidate)
        self.assertEqual("passed", report["status"], report["failures"])
        self.assertEqual("tokens", report["reference"]["token_source"])
        self.assertTrue(any("--tokens" in warning for warning in report["warnings"]))

    def test_cli_rejects_invalid_arguments(self):
        output = self.root / "invalid.json"
        cases = [["--self-check", self.reference, self.reference, "--output", output],
                 [self.reference, "--output", output],
                 [self.reference, self.reference, "--output", output, "--max-ppl-ratio", "0"],
                 [self.reference, self.reference, "--output", output, "--max-ppl-ratio", "nan"],
                 [self.reference, self.reference, "--output", output, "--max-mean-kl", "-1"],
                 [self.reference, self.reference, "--output", output, "--min-top1", "1.5"],
                 [self.reference, self.reference, "--output", output, "--gate-scope", "decode"]]
        for arguments in cases:
            with self.subTest(arguments=arguments[2:]), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as failure:
                    self.run_cli(*arguments)
                self.assertEqual(2, failure.exception.code)
        self.assertFalse(output.exists())

    def test_pure_python_cli_and_probe_json_path(self):
        output = self.root / "pure.json"
        code, text = self.run_cli(self.reference / "probe.json", self.candidate(noisy(.001)), "--output", output,
                                  "--no-numpy")
        self.assertEqual(0, code, text)
        self.assertEqual("pure-python", json.loads(output.read_text())["backend"]["name"])
        self.assertIn("unique_targets", text)
        self.assertIn("status: passed (compare; pure-python)", text)

    def test_unreadable_runs_are_usage_errors(self):
        output = self.root / "unreadable.json"
        (self.root / "garbled").mkdir()
        (self.root / "garbled" / "probe.json").write_text("{\"status\": ")
        for run in (self.root / "missing", self.root / "garbled", self.reference / "step-001-prefill.f32"):
            with self.subTest(run=run.name), patch("sys.stderr"):
                with self.assertRaises(SystemExit) as failure:
                    self.run_cli(self.reference, run, "--output", output)
                self.assertEqual(2, failure.exception.code)
        self.assertFalse(output.exists())

    def test_runs_that_disagree_with_themselves_are_flagged(self):
        consistent = quality.compare(self.reference, self.reference, self_check=True)
        pairs = consistent["self_consistency"]["reference"]["pairs"]
        # prefill target P is revisited by the last chunk and reset-prefill; decode targets by refill.
        self.assertEqual([(7, 1, 4), (8, 2, 5), (9, 3, 6), (10, 4, 7), (12, 1, 4)],
                         [(pair["index"], pair["first_index"], pair["target_position"]) for pair in pairs])
        self.assertTrue(all(pair["kl"] == 0 for pair in pairs))
        self.assertEqual(["reference"], list(consistent["self_consistency"]))
        self.assertFalse(any("disagrees with itself" in warning for warning in consistent["warnings"]))
        def lossy_chunks(index, operation, position):
            # A chunked prefill that drops state: identical on both sides, wrong against its own prefill.
            return context_logits(position, seed=3) if operation in ("chunk", "refill") else context_logits(position)
        reference = write_probe(self.root / "lossy-reference", lossy_chunks)
        candidate = write_probe(self.root / "lossy-candidate", lossy_chunks)
        report = quality.compare(reference, candidate)
        self.assertEqual("passed", report["status"], report["failures"])  # the shared defect passes every gate...
        flagged = [warning for warning in report["warnings"] if "disagrees with itself" in warning]
        self.assertEqual({"reference", "candidate"}, {warning.split()[0] for warning in flagged})  # ...but is reported
        self.assertTrue(any("chunk step 7 vs step 1" in warning for warning in flagged), flagged)
        phases = report["self_consistency"]["reference"]["phases"]
        self.assertEqual({"chunk": 1, "refill": 3, "reset-prefill": 1},
                         {phase: summary["count"] for phase, summary in phases.items()})
        self.assertEqual(0, phases["reset-prefill"]["max_kl"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__": unittest.main()
