#!/usr/bin/env python3
"""Read a bounded, inclusive Muse Spark expert-layer range into the file cache.

This external preparation performs no inference, server launch or GPU work.
For GPU source prefetch on a CPU32 deployment, select layers 32..61. It may run
alongside startup, but BOTH must finish before quality/performance requests.
Report preparation separately from startup and inference timings. Residency is
observed, never guaranteed. The existing CPU-expert warmer remains unchanged.

Requires unchanged llama.cpp gguf-py on PYTHONPATH. All 62 layers' three expert
weights are validated before selecting any range. --check-only reads headers
and queries residency without reading tensor data. Model files are read-only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time


WARM_PATH = Path(__file__).with_name("warm-muse-spark-cpu-experts.py")
SPEC = importlib.util.spec_from_file_location("spark_prefetch_warmer", WARM_PATH)
WARM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WARM)
LAYERS = 62


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def range_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_ranges(paths, first_layer, last_layer, reader_factory):
    if not 0 <= first_layer <= last_layer < LAYERS:
        raise ValueError("Layer bounds must satisfy 0 <= first-layer <= last-layer < 62")

    def checked_reader(path):
        reader = reader_factory(path)
        fields = reader.fields
        if path == paths[0] or "general.architecture" in fields:
            arch = fields["general.architecture"].contents()
            if arch not in ("muse-spark", "muse_spark"):
                raise ValueError(f"Unexpected architecture: {arch}")
            if int(fields[arch + ".block_count"].contents()) != LAYERS:
                raise ValueError("This prefetch helper requires the complete 62-layer Muse Spark model")
        if "split.no" in fields and int(fields["split.no"].contents()) != paths.index(path):
            raise ValueError("GGUF split.no does not match its numbered shard")
        for tensor in reader.tensors:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
            if match and not 0 <= int(match[1]) < LAYERS:
                raise ValueError(f"Expert tensor has an invalid layer: {tensor.name}")
        return reader

    # Validate missing/duplicate/overlapping/out-of-bounds tensors across ALL
    # layers first. An invalid unselected CPU layer must not be hidden by slicing.
    all_ranges = WARM.plan_ranges(paths, LAYERS, checked_reader)
    selected = [item for item in all_ranges if first_layer <= item["layer"] <= last_layer]
    return all_ranges, selected


def recorded_verification(path, shards):
    """Carry prior checkpoint hashes; do not mislabel them as newly calculated."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass" or report.get("repo") != "unsloth/Muse-Spark-Open-816B-A42B-GGUF":
        raise ValueError("Checkpoint verification must be a passing Muse Spark download report")
    rows = report.get("shards", [])
    by_name = {Path(row["file"]).name: row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != {shard.name for shard in shards}:
        raise ValueError("Checkpoint verification must describe exactly these shards")
    for shard in shards:
        row = by_name[shard.name]
        if row.get("size_status") != "pass" or row.get("actual_bytes") != shard.stat().st_size:
            raise ValueError(f"Checkpoint verification size mismatch: {shard.name}")
        if row.get("sha256_status") == "fail":
            raise ValueError(f"Checkpoint verification hash failed: {shard.name}")
    return dict(path=str(path.resolve()), report_sha256=sha256(path), report=report,
                scope="Prior verification and current sizes only; checkpoint hashes are not recomputed by prefetch.")


def run(args, reader_factory=None):
    # Exclusive creation protects previous evidence and rejects existing model
    # paths/symlinks as output. Keep the same owned handle through failure writes.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        start = time.monotonic()
        before_io = WARM.process_io()
        report = dict(status="running", model=str(args.model.resolve()),
                      first_layer=args.first_layer, last_layer=args.last_layer,
                      expected_model_layers=LAYERS, workers=args.workers,
                      chunk_bytes=args.chunk_mib * 1024**2, read_mode=args.read_mode,
                      mode="check-only" if args.check_only else "prefetch",
                      timing_scope="External file-cache preparation; excluded from startup and inference timing. Wait for both server and helper before measured requests.",
                      started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      completed_ranges=0, completed_range_bytes=0,
                      progress_scope="Completed range counts are a lower bound after failure; in-flight/partial reads are not included. Process I/O is also recorded.")

        def save():
            output.seek(0)
            json.dump(report, output, indent=2)
            output.write("\n")
            output.truncate()
            output.flush()

        save()
        exit_code = 0
        try:
            report["tool_sha256"] = {str(Path(__file__).resolve()): sha256(__file__),
                                     str(WARM_PATH.resolve()): sha256(WARM_PATH)}
            if reader_factory is None:
                import gguf
                reader_factory = gguf.GGUFReader
            paths = WARM.discover_shards(args.model)
            if args.verification:
                report["checkpoint_verification"] = recorded_verification(args.verification, paths)
            all_ranges, ranges = select_ranges(paths, args.first_layer, args.last_layer, reader_factory)
            report.update(validated_tensor_count=len(all_ranges), validated_ranges_sha256=range_hash(all_ranges),
                          ranges=ranges, ranges_sha256=range_hash(ranges),
                          selected_bytes=sum(item["bytes"] for item in ranges),
                          before=WARM.residency(ranges))
            save()
            if not args.check_only:
                def progress(done, count, item, total, elapsed):
                    report.update(completed_ranges=done, completed_range_bytes=total,
                                  last_completed_tensor=item["tensor"], read_elapsed_seconds=elapsed)
                    save()
                    print(f"{done}/{count} {item['tensor']} total={total / 1024**3:.3f}GiB elapsed={elapsed:.1f}s", file=sys.stderr, flush=True)
                report["prefetch"] = WARM.warm_ranges(ranges, args.chunk_mib * 1024**2, progress,
                                                     args.workers, args.read_mode)
                report["after"] = WARM.residency(ranges)
            report["status"] = "checked" if args.check_only else "completed"
        except (Exception, KeyboardInterrupt) as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            exit_code = 1
        finally:
            after_io = WARM.process_io()
            report.update(duration_seconds=time.monotonic() - start,
                          finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          process_io_before=before_io, process_io_after=after_io,
                          process_io_delta={key: after_io[key] - before_io[key] for key in before_io if key in after_io}
                          if before_io is not None and after_io is not None else None)
            save()
        print(json.dumps({key: report[key] for key in ("status", "selected_bytes", "completed_ranges", "duration_seconds", "error") if key in report}))
        return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="First numbered GGUF shard, including metadata-only shard 00001")
    parser.add_argument("--first-layer", type=int, required=True, help="First decoder layer, inclusive (0..61)")
    parser.add_argument("--last-layer", type=int, required=True, help="Last decoder layer, inclusive (0..61)")
    parser.add_argument("--output", type=Path, required=True, help="New evidence JSON; existing paths are never overwritten")
    parser.add_argument("--verification", type=Path, help="Existing download-verification JSON to retain prior checkpoint hashes")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--read-mode", choices=("pread", "pread-mmap-touch"), default="pread")
    args = parser.parse_args(argv)
    if not 0 <= args.first_layer <= args.last_layer < LAYERS:
        parser.error("Layer bounds must satisfy 0 <= first-layer <= last-layer < 62")
    if not 1 <= args.chunk_mib <= 64 or not 1 <= args.workers <= 8:
        parser.error("chunk-mib must be 1..64 and workers must be 1..8")
    try:
        return run(args)
    except FileExistsError:
        parser.error("--output already exists; use a new evidence path")


if __name__ == "__main__":
    raise SystemExit(main())
