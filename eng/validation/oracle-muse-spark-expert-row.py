#!/usr/bin/env python3
"""CPU-only bounded dequantized oracle for one saved real-checkpoint MoE row.

Reads only that row's routed experts, dequantizing at most --chunk-rows matrix
rows at once. No native TensorSharp or CUDA calls. F64 matvec/normalization is an
independent reference; CPU Q8_K and CUDA MMQ Q8_1 variants additionally emulate
activation quantization to diagnose expected backend differences. These variants
do not exactly reproduce backend floating-point reduction order. Only Q3_K gate/
up and Q4_K down are supported. Run with OPENBLAS_NUM_THREADS=1 and no measured
inference; record this preparation separately from benchmark wall times.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

SPEC = importlib.util.spec_from_file_location("spark_expert_bench", Path(__file__).with_name("benchmark-muse-spark-cpu-experts.py"))
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def quantized_activation(values, mode, scale_half, np):
    if mode == "exact":
        return values.astype(np.float64)
    size = 256 if mode == "cpu_q8k" else 32
    blocks = values.astype(np.float32).reshape(-1, size)
    maximum = np.abs(blocks).max(axis=1, keepdims=True)
    inverse = np.divide(np.float32(127), maximum, out=np.zeros_like(maximum), where=maximum != 0)
    scaled = blocks * inverse
    # CPU nearest_int uses nearest-even; CUDA roundf rounds ties away from zero.
    quant = np.rint(scaled) if mode == "cpu_q8k" else np.copysign(np.floor(np.abs(scaled) + np.float32(.5)), scaled)
    scales = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse), where=inverse != 0)
    if scale_half:
        scales = scales.astype(np.float16).astype(np.float32)
    return (quant * scales).reshape(-1).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=115)
    parser.add_argument("--chunk-rows", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.chunk_rows <= 256:
        parser.error("chunk-rows must be1..256")
    import numpy as np
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize, Q4_K
    original = json.loads(args.diagnostic.read_text())
    plan = original["plan"]
    layer = plan["layers"][0]
    arrays = np.load(args.arrays)
    if not 0 <= args.row < len(arrays["inputs"]):
        parser.error("row outside saved inputs")
    h = plan["metadata"]["moe_latent_size"]
    ids, routing = arrays["ids"][args.row], arrays["routing"][args.row]
    modes = ["exact", "cpu_q8k", "cuda_q8_1"]
    output = np.zeros((h, 3), dtype=np.float64)
    start = time.monotonic()
    with bench.MappedLayer(plan, layer, np) as mapped:
        if [t["type_name"] for t in mapped.items[:3]] != ["Q3_K", "Q3_K", "Q4_K"]:
            raise ValueError("Oracle is restricted to Q3_K gate/up and Q4_K down")
        source = arrays["inputs"][args.row]
        input_modes = np.stack([quantized_activation(source, mode, False, np) for mode in modes], axis=1)
        def matvec(part, expert, values, down_original=None):
            item = mapped.items[part]
            width, rows, count = item["shape"]
            per_expert = item["bytes"] // count
            row_bytes = per_expert // rows
            source_bytes = mapped.entries[part][1]
            result = np.empty((rows, 3), dtype=np.float64)
            for first in range(0, rows, args.chunk_rows):
                n = min(args.chunk_rows, rows - first)
                offset = expert * per_expert + first * row_bytes
                encoded = source_bytes[offset:offset + n * row_bytes].reshape(n, row_bytes)
                decoded = dequantize(encoded, GGMLQuantizationType(item["ggml_type"]))
                result[first:first+n] = decoded.astype(np.float64) @ values
                if part == 2:
                    # CUDA's Q4_K MMQ DS4 uses half scale and half ORIGINAL
                    # input sums per32 for the affine minimum contribution.
                    blocks = encoded.reshape(-1, 144)
                    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
                    _, mins = Q4_K.get_scale_min(blocks[:, 4:16].copy())
                    factors = (dmin * mins).reshape(n, width // 32).astype(np.float64)
                    quant_sums = values[:, 2].reshape(-1, 32).sum(axis=1)
                    original_sums = down_original[:, 2].astype(np.float32).reshape(-1, 32).sum(axis=1).astype(np.float16).astype(np.float64)
                    result[first:first+n, 2] += factors @ (quant_sums - original_sums)
            return result
        for slot, expert in enumerate(ids):
            gate = matvec(0, int(expert), input_modes)
            up = matvec(1, int(expert), input_modes)
            exponent = np.exp(-np.abs(gate))
            sigmoid = np.where(gate >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
            active = gate * sigmoid * up
            for column in (1, 2):
                g, u = gate[:, column].astype(np.float32), up[:, column].astype(np.float32)
                ex = np.exp(-np.abs(g))
                sig = np.where(g >= 0, np.float32(1) / (np.float32(1) + ex), ex / (np.float32(1) + ex))
                active[:, column] = (g * sig) * u
            # Stored backend activations are F32 before down-input quantization.
            down_input = np.stack([quantized_activation(active[:, i], mode, i == 2, np)
                                   for i, mode in enumerate(modes)], axis=1)
            down = matvec(2, int(expert), down_input, active)
            scaled = down * mapped.norm[:, None]
            normed = scaled / np.sqrt(np.mean(scaled * scaled, axis=0) + plan["metadata"]["post_norm_epsilon"])
            output += float(routing[slot]) * normed
            print(f"expert={int(expert)} slot={slot} complete elapsed={time.monotonic()-start:.1f}s", flush=True)
    native_cpu, native_gpu = arrays["cpu_0"][args.row], arrays["stream_0"][args.row]
    comparisons = {}
    for index, mode in enumerate(modes):
        for label, native in (("cpu", native_cpu), ("gpu", native_gpu)):
            difference = np.abs(output[:, index] - native)
            comparisons[mode + "_vs_" + label] = {"rmse": float(np.sqrt(np.mean(difference ** 2))),
                "max_abs": float(difference.max()), "worst_channel": int(difference.argmax()),
                "channel715_reference": float(output[715, index]), "channel715_native": float(native[715])}
    report = {"status": "independent_diagnostic_no_parity_verdict", "row": args.row,
        "experts": ids.tolist(), "routing": routing.tolist(), "comparisons": comparisons,
        "seconds": time.monotonic() - start, "tool_sha256": bench.sha256_file(__file__),
        "dequantizer_source": str(Path(dequantize.__code__.co_filename).resolve()),
        "limitations": ["Backend F32 reduction order and exact transcendental implementation are not reproduced.",
            "CUDA Q8_1 emulation assumes the Q3_K D4 and Q4_K DS4 MMQ layouts from unchanged ggml a05c992.",
            "One synthetic latent row, not full-model or end-to-end quality equivalence."]}
    bench.save(args.output, report)
    np.savez_compressed(args.output.with_suffix(".npz"), oracle=output)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
