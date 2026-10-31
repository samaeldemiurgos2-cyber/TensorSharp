#!/usr/bin/env python3
"""Diff two DeepSeek V4 / V4.1 tensor traces layer by layer.

The direct-CUDA engine (TS_DSV4_CUDA_TRACE_DIR) and the pure-C# CPU executor (TS_DSV4_CPU_TRACE_DIR)
write the same stage tensors under the same names, p<first position>_<stage>.f32 (raw little-endian
float32). The CPU executor is held to the PyTorch reference, so running both on the same prompt and
diffing the directories shows the first stage where an executor departs from it, and by how much.

The native ggml executor (TS_DSV41_TRACE_DIR, on ggml_cuda or ggml_cpu) names its V4.1 stages
p<first position>_v41.<layer>.<stage>.f32 instead. Either directory may be a native one: its names are
mapped onto the managed ones (NATIVE_TO_MANAGED), so a direct-CUDA trace can be bisected against
ggml_cuda's on the same prompt. Run ggml with TS_DSV41_YOCO_KEEP=0: the managed executors have no YOCO
bounded replay. ggml's engram_gate has no managed tensor (the direct engine folds the gate into its
kernel), so it is compared with the gate a managed trace implies: per token and stream,
(engram_output - layer input) . value / |value|^2, from the layer's engram_output, its input (the
previous layer's hidden streams, or the embedding) and the value rows of its engram_kv.

    compare-dsv4-traces.py REFERENCE_DIR CANDIDATE_DIR [--threshold 1e-2] [--hc 4]

Prints one line per stage, in forward order: the relative error (max |a - b| over max |b|), the
cosine similarity, and a flag on the first stage whose relative error exceeds the threshold.
"""
import argparse
import os
import re
import sys

import numpy as np

STAGE_ORDER = ["embedding", "engram_kv", "engram_gate", "engram_output", "attn_input", "qr", "q", "raw_k",
               "latent", "attn_out", "ffn_input", "ffn_out", "hidden"]

# trace_v41 stage (ggml_ops_deepseek4.cpp, ggml_ops_deepseek41.inc) -> managed stage. The native stages
# the managed executors do not trace (engram_dot, logits) stay unpaired and are counted at the end, and
# so do index_scores and index_top: the direct engine traces them in its own layout (every cache row, the
# selection unordered, as .f32 and .i32), which InferenceWeb.Tests' Dsv41CudaEngineParityTests compares row
# by row instead (visible rows, selection rules, score error).
NATIVE_TO_MANAGED = {
    "engram_kv": "engram_kv",
    "engram_gate": "engram_gate",
    "engram_output": "engram_output",
    "attn_input": "attn_input",
    "qr": "qr",
    "query": "q",
    "raw_kv": "raw_k",
    "latent": "latent",
    "attn_output": "attn_out",
    "ffn_output": "ffn_out",
    "output": "hidden",
}

NATIVE_NAME = re.compile(r"p(\d+)_v41\.(\d+)\.(.+)\.f32$")
MANAGED_NAME = re.compile(r"p(\d+)_(?:blk(\d+)_)?(.+)\.f32$")


def managed_name(pos, layer, stage):
    return f"p{pos:06d}_{stage}.f32" if layer is None else f"p{pos:06d}_blk{layer:02d}_{stage}.f32"


def index_trace(directory):
    """{managed file name: path} for a trace directory written under either naming scheme."""
    entries = {}
    for name in os.listdir(directory):
        native = NATIVE_NAME.match(name)
        if native:
            pos, layer, stage = int(native.group(1)), int(native.group(2)), native.group(3)
            entries[managed_name(pos, layer, NATIVE_TO_MANAGED.get(stage, "native." + stage))] = \
                os.path.join(directory, name)
        elif name.endswith(".f32"):
            entries[name] = os.path.join(directory, name)
    return entries


def stage_key(name):
    m = MANAGED_NAME.match(name)
    if not m:
        return (sys.maxsize, sys.maxsize, sys.maxsize, name)
    pos, layer, stage = int(m.group(1)), m.group(2), m.group(3)
    layer = -1 if layer is None else int(layer)
    order = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
    return (pos, layer, order, stage)


def load(entry):
    return entry if isinstance(entry, np.ndarray) else np.fromfile(entry, dtype=np.float32).astype(np.float64)


def derive_engram_gates(entries, other, hc):
    """Adds to a managed trace the engram_gate it implies, wherever the other trace holds one to compare
    with (whose size, hc gates per token, gives the token count the managed sizes cannot: they all
    scale with tokens * n_embd) and this trace holds the layer's engram_output, engram_kv and input."""
    for gate in [n for n in other if n not in entries and n.endswith("_engram_gate.f32")]:
        m = MANAGED_NAME.match(gate)
        pos, layer = int(m.group(1)), int(m.group(2))
        source = managed_name(pos, None, "embedding") if layer == 0 else managed_name(pos, layer - 1, "hidden")
        names = (managed_name(pos, layer, "engram_output"), source, managed_name(pos, layer, "engram_kv"))
        if not all(n in entries for n in names):
            continue
        out, inp, kv = (load(entries[n]) for n in names)
        nt = load(other[gate]).size // hc
        if nt == 0 or out.shape != inp.shape or out.size % (nt * hc) or kv.size * hc != out.size * (hc + 1):
            continue
        # out and inp are [nt, hc, n_embd]; kv is [nt, (hc + 1) * n_embd], the value its last n_embd.
        embd = out.size // (nt * hc)
        update = (out - inp).reshape(nt, hc, embd)
        value = kv.reshape(nt, (hc + 1) * embd)[:, hc * embd:]
        norm = np.maximum((value * value).sum(axis=1), 1e-30)
        entries[gate] = (np.einsum("tse,te->ts", update, value) / norm[:, None]).reshape(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--threshold", type=float, default=1e-2)
    ap.add_argument("--hc", type=int, default=4, help="hyper-connection streams, for the derived Engram gate")
    args = ap.parse_args()

    reference, candidate = index_trace(args.reference), index_trace(args.candidate)
    derive_engram_gates(reference, candidate, args.hc)
    derive_engram_gates(candidate, reference, args.hc)
    names = sorted(set(reference) & set(candidate), key=stage_key)
    if not names:
        print("no stage both traces wrote")
        return 2
    first_bad = None
    for name in names:
        ref, cand = load(reference[name]), load(candidate[name])
        if ref.shape != cand.shape:
            print(f"{name:40s} shape {cand.shape} vs reference {ref.shape}")
            first_bad = first_bad or name
            continue
        scale = max(np.abs(ref).max(), 1e-30)
        rel = float(np.abs(cand - ref).max() / scale)
        denom = np.linalg.norm(ref) * np.linalg.norm(cand)
        cos = float(ref @ cand / denom) if denom > 0 else 1.0
        flag = ""
        if rel > args.threshold and first_bad is None:
            first_bad = name
            flag = "  <-- first above threshold"
        print(f"{name:40s} rel {rel:10.3e}  cos {cos:.6f}{flag}")
    only_ref = sorted(set(reference) - set(candidate))
    only_cand = sorted(set(candidate) - set(reference))
    if only_ref or only_cand:
        print(f"stages only in the reference: {len(only_ref)}, only in the candidate: {len(only_cand)}")
    return 1 if first_bad else 0


if __name__ == "__main__":
    sys.exit(main())
