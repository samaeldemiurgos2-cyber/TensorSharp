#!/usr/bin/env python3
"""Run the final native Q2_K smoke18 against the retained CPU oracle.

Requires an exclusive GPU window after stopping the full-model HTTP server.
This is a GGUF/F32-input-reference comparison, not exact quantized-kernel,
original-FP8-checkpoint, or llama.cpp parity. No CPU oracle is recomputed.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

PROMPT = [0, 128803, 3085, 344, 223, 20, 940, 223, 20, 33, 19414, 418, 1438, 270, 1167, 16, 128804, 128822]
NATIVE_SHA = "6b3b5ab3c333c59423bc10efe9f18b5f0483fd0a478823647471c5de7e736014"
REFERENCE_LOG_SHA = "10ebb42dc8940e75ccdccc8e344faa8dae52a9291e4caceec5a8571c060dc8e6"
CHECKPOINT_MANIFEST_SHA = "b1f3d500acfedb8de6b3217b573b8c46094911c44d059f8f078b488f7e646f35"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(value, reference):
    if value.shape != reference.shape or not np.isfinite(value).all() or not np.isfinite(reference).all():
        raise ValueError("Logits have invalid shape or nonfinite values")
    value_norm, reference_norm = np.linalg.norm(value), np.linalg.norm(reference)
    if value_norm <= 0 or reference_norm <= 0:
        raise ValueError("Cannot compare zero-norm logits")
    difference = value - reference
    return dict(max_absolute_error=float(np.max(np.abs(difference))),
                relative_l2=float(np.linalg.norm(difference) / reference_norm),
                cosine=float(value @ reference / (value_norm * reference_norm)),
                strict_allclose_atol_rtol_2e_5=bool(np.allclose(value, reference, atol=2e-5, rtol=2e-5)),
                argmax=int(np.argmax(value)), reference_argmax=int(np.argmax(reference)),
                top5=np.argsort(value)[-5:][::-1].tolist(), reference_top5=np.argsort(reference)[-5:][::-1].tolist())


def validate_reference_logits(reference, summary):
    if np.argsort(reference)[-10:][::-1].tolist() != summary["top_tokens"]:
        raise ValueError("Retained logits top10 do not match the SHA-pinned reference log")
    norm_tolerance = 8 * float(np.spacing(np.float32(summary["last_logits_l2"])))
    if abs(float(np.linalg.norm(reference)) - summary["last_logits_l2"]) > norm_tolerance:
        raise ValueError("Retained logits norm does not match the log within eight F32 ULPs")
    return norm_tolerance


def normalized_environment(source):
    environment = source.copy()
    removed = [key for key in environment if key.startswith(("TS_DSV4_", "TS_DSV41_", "GGML_"))]
    removed += ["TS_DSV41_TRACE_DIR", "NVIDIA_TF32_OVERRIDE"]
    for key in removed:
        environment.pop(key, None)
    return environment, removed


def trace_coverage(directory, prompt_count, vocab_size, n_layers=40, embedding_size=5120):
    """Check stable stage names/sizes; native and Python stage sets differ."""
    expected = {f"p000000_v41.{layer:02d}.{stage}.f32": prompt_count * embedding_size * 4
                for layer in range(n_layers) for stage in ("attn_output", "ffn_output")}
    expected[f"p000000_v41.{n_layers:02d}.logits.f32"] = vocab_size * 4
    missing, wrong_sizes = [], []
    for name, size in expected.items():
        path = directory / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != size:
            wrong_sizes.append(dict(file=name, expected_bytes=size, actual_bytes=path.stat().st_size))
    return dict(complete=not missing and not wrong_sizes, required_files=len(expected),
                scope="Per-layer attention/FFN outputs and terminal last-row logits, with exact F32 byte sizes; not every intermediate.",
                missing=missing, wrong_sizes=wrong_sizes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/workspace/deepseek41-work"))
    parser.add_argument("--repo", type=Path, default=Path("/workspace/TensorSharp"))
    parser.add_argument("--output", type=Path, help="Must not already exist")
    parser.add_argument("--checkpoint-manifest", type=Path, help="Previously verified seven-shard SHA manifest")
    tracing = parser.add_mutually_exclusive_group()
    tracing.add_argument("--trace", action="store_true", help="Retain first-forward layer traces; diagnostic I/O only")
    tracing.add_argument("--trace-after", action="store_true",
                         help="Keep primary generation untraced, then replay the prompt in a fresh slot for layer diagnostics")
    args = parser.parse_args()
    work, repo = args.work.resolve(), args.repo.resolve()
    output = (args.output or work / "final-smoke18-6b3").resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite an existing output directory: {output}")
    # Avoid a second full-checkpoint allocation while any GPU client is resident.
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"],
                               check=True, text=True, capture_output=True)
    if processes.stdout.strip():
        parser.error("GPU compute processes are still resident; stop the full-model host before this exclusive run")
    library = work / "native-v41-shared-pins-6b3b5ab3.so"
    reference_dir = work / "reference/q2-smoke18"
    reference_log = work / "reference/q2-smoke18.log"
    reference_path = reference_dir / "logits.npy"
    tokens = work / "smoke-tokens.json"
    smoke_source = repo / "eng/dsv41-smoke.py"
    model = Path("/workspace/models/DeepSeek-V4.1-Flash-Q2_K/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf")
    checkpoint_manifest = args.checkpoint_manifest or work / "final-smoke18-checkpoint-sha256.json"
    if sha(checkpoint_manifest) != CHECKPOINT_MANIFEST_SHA:
        raise ValueError("Checkpoint verification manifest differs from the reviewed seven-shard control")
    checkpoint = json.loads(checkpoint_manifest.read_text())
    if not checkpoint.get("passed") or len(checkpoint["shards"]) != 7:
        raise ValueError("Checkpoint manifest is not a successful seven-shard verification")
    for shard in checkpoint["shards"]:
        if (not shard.get("passed") or shard["actual_sha256"] != shard["expected_sha256"] or
                (model.parent / shard["file"]).stat().st_size != shard["expected_bytes"]):
            raise ValueError("Current shard size or retained verification does not match the reviewed checkpoint")
    if sha(library) != NATIVE_SHA or sha(reference_log) != REFERENCE_LOG_SHA:
        raise ValueError("Archived native or retained reference log hash does not match the reviewed control")
    log_text = reference_log.read_text()
    reference_summary = json.loads(log_text[log_text.index("{"):])
    if reference_summary["tokens"] != PROMPT or json.loads(tokens.read_text()) != PROMPT:
        raise ValueError("Exact reference/native token IDs differ from the reviewed prompt")
    if reference_summary["cache_type"] != "model":
        raise ValueError("Retained reference used a different cache policy")
    layers = [int(m.group(1)) for m in re.finditer(r"^layer (\d+):", log_text, re.M)]
    prefixes = Counter(path.name.split("_", 1)[0] for path in reference_dir.iterdir() if path.name.startswith("p"))
    reference_all = np.load(reference_path, mmap_mode="r")
    embedding = np.load(reference_dir / "p000000_embedding.npy", mmap_mode="r")
    if layers != list(range(40)) or dict(prefixes) != {"p000000": 660} or embedding.shape != (18, 4, 5120) or reference_all.shape != (18, 129280):
        raise ValueError("Retained traces do not establish the reviewed single 18-token position0 prefill")
    reference = reference_all[-1].astype(np.float64)
    if not np.isfinite(reference).all():
        raise ValueError("Reference logits are not finite")
    norm_tolerance = validate_reference_logits(reference, reference_summary)
    environment, removed = normalized_environment(os.environ)
    settings = dict(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", TS_DSV4_FA="1",
                    TS_DSV4_FUSED="1", TS_DSV4_GATHER="1", TS_DSV41_SPARSE_FA="0",
                    TS_DSV41_COMPACT_RAW_GATHER="0", TS_CPU_MOE_THREADS="24",
                    TS_DSV41_ENGRAM_THREADS="16", TS_DSV41_ENGRAM_WARM="0",
                    TS_DSV41_YOCO_KEEP="0")
    environment.update(settings)
    output.mkdir(parents=True)
    report_path = output / "smoke.json"
    command = [sys.executable, str(smoke_source), str(model), "--library", str(library),
               "--tokens", str(tokens), "--output", str(report_path), "--backend", "CUDA", "--gpus", "8",
               "--context", "4096", "--ubatch", "128", "--threads", "24", "--cpu-moe", "0", "--steps", "2"]
    if args.trace:
        command += ["--trace-dir", str(output / "trace")]
    elif args.trace_after:
        command += ["--trace-after-dir", str(output / "trace-after")]
    paths = [library, reference_log, reference_path, tokens, smoke_source, Path(__file__).resolve(),
             checkpoint_manifest]
    manifest = dict(scope=__doc__, command=command, explicit_environment=settings,
                    trace_mode="fresh_slot_after_untraced_primary" if args.trace_after else "primary" if args.trace else "none",
                    removed_override_names=sorted(set(removed)), gpu_processes_before=processes.stdout,
                    reference_provenance=dict(recorded_summary=reference_summary, layer_sequence=layers,
                                              trace_position_prefixes=dict(prefixes), embedding_shape=list(embedding.shape),
                                              logits_shape=list(reference_all.shape), original_argv=None,
                                              log_top10_and_norm_match=True, norm_tolerance=norm_tolerance,
                                              unrecorded_fields=["CPU threads", "matrix row-block size", "creation-source hash"]),
                    checkpoint_identity=dict(source=str(checkpoint_manifest), current_sizes_checked=True,
                                             current_weight_bytes_rehashed=False, retained_verification=checkpoint),
                    inputs={str(path): dict(sha256=sha(path), bytes=path.stat().st_size) for path in paths})
    shutil.copy2(smoke_source, output / "dsv41-smoke.py")
    shutil.copy2(Path(__file__).resolve(), output / "runner.py")
    shutil.copy2(reference_log, output / "reference.log")
    shutil.copy2(tokens, output / "tokens.json")
    shutil.copy2(checkpoint_manifest, output / "checkpoint-sha256.json")
    np.save(output / "reference-last-logits.npy", reference.astype(np.float32))
    (output / "settings.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    print(f"Running final6b3 smoke18; stdout/stderr retained in {output / 'native.log'}", flush=True)
    with (output / "native.log").open("w") as log:
        completed = subprocess.run(command, env=environment, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(native_exit_code=completed.returncode, elapsed_seconds=time.monotonic() - started)
    comparison = dict(scope="Native GGUF logits versus retained F32-matrix-input reference; strict tolerance is reported, not relaxed.")
    try:
        report = json.loads(report_path.read_text())
        if not report.get("passed") or report["library_sha256"] != NATIVE_SHA or report["prompt_tokens"] != PROMPT:
            raise RuntimeError("Native smoke failed or did not use the reviewed inputs")
        if report.get("first_forward_traced") is not args.trace:
            raise RuntimeError("Primary tracing mode does not match the requested control")
        native = np.load(output / "smoke.first-logits.npy").astype(np.float64)
        comparison.update(final_native=metrics(native, reference), generated_tokens=report["generated_tokens"],
                          expected_generated_tokens=[22, 1], generated_tokens_match=report["generated_tokens"] == [22, 1])
        if args.trace_after:
            diagnostic = report.get("trace_diagnostic", {})
            comparison["traced_diagnostic"] = {"execution": diagnostic,
                "scope": "Fresh-slot traced graph may change fusion and is not the primary production-path comparison."}
            if diagnostic.get("complete"):
                traced_path = output / "smoke.traced-first-logits.npy"
                traced = np.load(traced_path).astype(np.float64)
                coverage = trace_coverage(output / "trace-after", len(PROMPT), len(native))
                comparison["traced_diagnostic"].update(vs_primary=metrics(traced, native),
                                                       vs_reference=metrics(traced, reference),
                                                       logits_sha256=sha(traced_path), trace_coverage=coverage)
                if coverage["complete"]:
                    terminal = np.fromfile(output / "trace-after/p000000_v41.40.logits.f32", dtype="<f4")
                    same = np.array_equal(terminal.view(np.uint32), traced.astype(np.float32).view(np.uint32))
                    comparison["traced_diagnostic"]["terminal_trace_matches_returned_logits"] = bool(same)
                    if not same:
                        comparison["error"] = "Terminal trace differs from returned diagnostic logits; primary metrics remain retained"
                else:
                    comparison["error"] = "Requested trace has incomplete stage coverage; primary metrics remain retained"
            else:
                comparison["error"] = "Requested fresh-slot trace did not complete; primary metrics remain retained"
        if completed.returncode:
            comparison["error"] = f"Smoke process exited {completed.returncode}; primary metrics remain retained"
        old_path = work / "smoke-first-logits.f32"
        if old_path.exists():
            old = np.fromfile(old_path, dtype="<f4").astype(np.float64)
            comparison["old_native"] = metrics(old, reference)
            comparison["old_native_artifact"] = dict(path=str(old_path), sha256=sha(old_path))
    except Exception as error:
        comparison["error"] = str(error)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    manifest["artifacts"] = {str(path.relative_to(output)): dict(sha256=sha(path), bytes=path.stat().st_size)
                             for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(comparison, indent=2), flush=True)
    # Numerical strictness is preserved in comparison.json. The command status
    # checks execution/expected greedy output and must not be called parity.
    return int("error" in comparison or not comparison.get("generated_tokens_match", False))


if __name__ == "__main__":
    raise SystemExit(main())
