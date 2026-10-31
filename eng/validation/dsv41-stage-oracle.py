#!/usr/bin/env python3
"""Per-stage float64 LOCAL oracle (gate v2, S8c-hardened) for a DeepSeek V4.1 trace pair: direct CUDA
(TS_DSV4_CUDA_TRACE_DIR) and ggml_cuda (TS_DSV41_TRACE_DIR, run with TS_DSV41_YOCO_KEEP=0) on the same prompt.
Every layer: raw-cache layers (compress ratio 0) and the compressed/sparse layers (ratio 1 or 2: compressor,
indexer, candidate mask, sparse selection and attention).

    dsv41-stage-oracle.py --ggml GGML_TRACE --cuda CUDA_TRACE --model MODEL.gguf
                          [--cuda-probe DIR --ggml-probe DIR]      (probe.json token ids: the Engram lookup)
                          [--layers 0,...,9] [--state-in S.npz] [--state-out S.npz]
                          [--moe-tokens 0:68] [--segments 0:64,64:68 | --segments calls]
                          [--rope-table exact|engine] [--json OUT]
    dsv41-stage-oracle.py --gate-from REPORT.json            (re-evaluate the gate on an earlier report)
    dsv41-stage-oracle.py --merge W1.json W2.json ... --json OUT (join layer windows, then gate)

Method. For every stage it recomputes the stage in float64 from the GGUF weights (gguf-py's numpy decoders, an
implementation independent of both engines) and from the SAME engine's own traced inputs to that stage:

    local_err(engine, stage) = || stage_engine - exact(stage; inputs_engine) || / || exact ||

which isolates the rounding each engine adds AT that stage from what it inherited. Where an engine applies a
format rounding BY DESIGN (the trained cache formats of ts_dsv41_quant_row / TSG_DSV41_FUSED_QUANT: mode 0 FP8 E4M3
raw K, mode 1 MXFP4 indexer q/k, mode 2 NVFP4 compressed KV; the F16 caches) the oracle applies the same rounding,
ported bit for bit in float32, to its exact recompute; the remaining error is what is judged. RoPE uses an exact
float64 table (--rope-table engine replays the direct engine's float32 table instead).

S8c hardening (review of S8b):
  1. Selections are checked as LISTS: count, uniqueness, visibility and (pruned) -inf rows; the attention core is
     recomputed from the EXACT selection (exact top-k of the engine's own traced scores) as well as from the engine's
     verbatim id list, so a duplicate id standing in for the boundary row fails.
  2. Two-sided: ggml's own local error must sit within a stated bound of a PRECISION MODEL of its arithmetic (the
     oracle's exact stage with ggml-cuda's activation roundings: q8_1 for quantized weights (MMQ above 8 rows, MMVQ
     at most 8; Q2_K's MMQ operand per 64 values), BF16/F16 for BF16/F16 weights above 3 rows, F32 below):
     ggml_local <= model_factor x model_local + slack, and the model must REPLAY ggml: rel(ggml, model) <=
     replay_factor x ggml_local; F32 stages <= 1e-6 for ggml too. ffn_out at layers whose MoE weights include a type
     the model does not replay (REPLAYED_MOE_TYPES) is reported, not bounded, by the model. Plus two semantic
     discrimination self-tests: each engine's attn_out must fit the oracle's attention semantics better than every
     informative perturbation (sinks toggled, window -1/+1, selection toggled, compressed rows rotated at the other
     end of the block), and each engine's ffn_out its MoE semantics (routing-weight normalization toggled, routed
     scale 1, the weakest pick dropped, the shared expert omitted, routed / shared swiglu clamp off).
  3. Layer windows: --state-in/--state-out carry the cross-layer state, so long prompts are traced and judged a few
     layers at a time (--merge joins the reports).
  4. Routing: the engine's picks must be a top-6 of its OWN traced router logits up to the derived F32 evaluation
     bound of sqrt(softplus(l)) + b, and of the exact logits from its own ffn_input up to that bound plus the
     derived F32 GEMM bound; no constant threshold.
  5. Every paired stage is gated per segment as well as in aggregate (--segments calls: one segment per call).
     A raw_k segment over the factor is FLAGGED (not failed) only when every differing element is a single FP8
     step and a binomial test does not reject "the direct engine flips at most factor x as often" (p >= 0.01).
  6. The Layer-1 Engram oracle uses each engine's OWN lookup rows: recomputed from its own token ids (the
     dsv41_engram.h hash and the host lookup's dequantization); the direct engine's traced rows are checked
     against them.
  7. The candidate-block pin (pos // block) is asserted to be used only for a ratio-1 candidate source.

Stages (direct name / ggml name; "-" = not traced by that engine):
  every layer   attn_input / attn_input; qr / qr; q / query; raw_k / raw_kv; attn_out / attn_output;
                ffn_input / - (ggml's derived in float64); router, selw, experts, shexp_out / -; ffn_out / ffn_output;
                hidden / output (or the next layer's engram_output alias)
  KV sources    comp_kv, comp_score / -; latent / latent; lid_pre / -; lid_k, comp_k / -
  index sources index_q, index_w / -; index_scores / index_scores; index_top(+count) / index_top
  candidate     cand_mask / - ; ggml's is read from its consumers' -inf rows
  attention     attn_core / - (sinks + softmax over [raw window | selected compressed rows], K = V)
Layer roles come from the GGUF (attention.kv_source_layers, indexer.source_layers, candidate_source_layer).

Needs gguf-py (pip install gguf, or GGUF_PY=<directory holding the gguf package>) except for --gate-from/--merge.
"""
import argparse
import json
import math
import os
import re
import struct
import sys

import numpy as np

HC = 4
F32 = np.float32
U = 2.0 ** -24          # float32 unit roundoff

_GGUF_PY = None


def gguf_py():
    """gguf-py, imported on first use (the gate, the merge and the unit tests do not need it)."""
    global _GGUF_PY
    if _GGUF_PY is None:
        if os.environ.get("GGUF_PY"):
            sys.path.insert(0, os.environ["GGUF_PY"])
        from gguf import quants as gq
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
        _GGUF_PY = (gq, GGML_QUANT_SIZES, GGMLQuantizationType)
    return _GGUF_PY


# --------------------------------------------------------------------------------------------- GGUF
class Gguf:
    NUMPY = {0: np.uint8, 1: np.int8, 2: np.uint16, 3: np.int16, 4: np.uint32, 5: np.int32, 6: np.float32,
             10: np.uint64, 11: np.int64, 12: np.float64}

    def __init__(self, path):
        self.path = path
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        with open(path, "rb") as f:
            def read(fmt):
                size = struct.calcsize("<" + fmt)
                return struct.unpack("<" + fmt, f.read(size))[0]

            def string():
                return f.read(read("Q")).decode("utf-8", "replace")

            def value(kind):
                if kind == 8:
                    return string()
                if kind == 9:
                    inner, count = read("I"), read("Q")
                    if inner in self.NUMPY:
                        a = np.frombuffer(f.read(np.dtype(self.NUMPY[inner]).itemsize * count), self.NUMPY[inner])
                        return a.tolist() if count < 4096 else a.copy()
                    return [value(inner) for _ in range(count)]
                return read(scalar[kind])

            assert f.read(4) == b"GGUF"
            read("I")
            n_tensors, n_kv = read("Q"), read("Q")
            self.kv = {}
            for _ in range(n_kv):
                name = string()
                self.kv[name] = value(read("I"))
            self.tensors = {}
            for _ in range(n_tensors):
                name = string()
                dims = [read("Q") for _ in range(read("I"))]
                kind, offset = read("I"), read("Q")
                self.tensors[name] = (dims, kind, offset)
            align = int(self.kv.get("general.alignment", 32))
            self.data = (f.tell() + align - 1) // align * align

    def row_bytes(self, name):
        _, sizes, qtype = gguf_py()
        dims, kind, _ = self.tensors[name]
        block, size = sizes[qtype(kind)]
        return dims[0] // block * size

    def _decode(self, raw, kind, rows, ne0):
        gq, _, qtype = gguf_py()
        qt = qtype(kind)
        if qt == qtype.F32:
            return np.frombuffer(raw, np.float32).reshape(rows, ne0).astype(np.float64)
        if qt == qtype.F16:
            return np.frombuffer(raw, np.float16).reshape(rows, ne0).astype(np.float64)
        if qt == qtype.BF16:
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(rows, ne0).astype(np.float64)
        arr = np.frombuffer(raw, np.uint8).reshape(rows, -1)
        return np.asarray(gq.dequantize(arr, qt), dtype=np.float64).reshape(rows, ne0)

    def get(self, name, first_row=0, rows=None):
        """Rows [first_row, first_row + rows) of a 2-D (or flattened 3-D) tensor, as float64 [rows, ne0]."""
        dims, kind, offset = self.tensors[name]
        total = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        rows = total - first_row if rows is None else rows
        rb = self.row_bytes(name)
        with open(self.path, "rb") as f:
            f.seek(self.data + offset + first_row * rb)
            raw = f.read(rows * rb)
        return self._decode(raw, kind, rows, dims[0])

    def gather(self, name, row_ids):
        """Arbitrary rows of a tensor (sparse reads through a memory map), float64 [len(row_ids), ne0]."""
        dims, kind, offset = self.tensors[name]
        rb = self.row_bytes(name)
        mm = np.memmap(self.path, np.uint8, mode="r")
        ids = np.asarray(row_ids, np.int64)
        idx = (self.data + offset + ids[:, None] * rb) + np.arange(rb)[None, :]
        raw = np.ascontiguousarray(mm[idx])
        del mm
        return self._decode(raw.tobytes(), kind, len(ids), dims[0])

    def vec(self, name):
        return self.get(name).reshape(-1)

    def expert(self, name, j):
        dims = self.tensors[name][0]
        return self.get(name, first_row=j * dims[1], rows=dims[1])

    def type_name(self, name):
        return gguf_py()[2](self.tensors[name][1]).name


# -------------------------------------------------------------------------------------------- traces
class Trace:
    """One engine's trace directory. Files with the same p0 were overwritten by the probe's later passes; its last
    passes are the reset-prefill of the prefix and the refill of the decode positions, so each position appears once.
    The chunk layout (p0, rows) is read from the attn_input of `layer` (the first layer of the window), which both
    engines write for every forward."""
    CUDA = re.compile(r"p(\d+)_(?:blk(\d+)_)?(.+)\.(f32|i32|f16|u8)$")
    GGML = re.compile(r"p(\d+)_v41\.(\d+)\.(.+)\.(f32|i32)$")

    def __init__(self, directory, engine, e, layer=0):
        self.dir, self.engine = directory, engine
        self.files = {}
        rx = self.CUDA if engine == "cuda" else self.GGML
        for name in os.listdir(directory):
            mm = rx.match(name)
            if mm:
                L = mm.group(2)
                self.files[(int(mm.group(1)), None if L is None else int(L), mm.group(3))] = os.path.join(directory, name)
        p0s = sorted(p0 for (p0, L, st) in self.files if L == layer and st == "attn_input")
        if not p0s:
            raise SystemExit(f"{directory}: no layer-{layer} attn_input trace")
        self.chunks = [(p0, os.path.getsize(self.files[(p0, layer, "attn_input")]) // (4 * e)) for p0 in p0s]
        pos = 0
        for p0, nt in self.chunks:
            if p0 != pos:
                raise SystemExit(f"{directory}: chunks do not tile the positions (p0 {p0}, expected {pos})")
            pos += nt
        self.T = pos
        self.calls = np.zeros(self.T, np.int64)          # rows of the call each position was computed in
        for p0, nt in self.chunks:
            self.calls[p0:p0 + nt] = nt

    def tok(self, L, stage, width, dtype=np.float32):
        """[T, width] over every chunk (None when any chunk lacks the file)."""
        parts = []
        for p0, nt in self.chunks:
            f = self.files.get((p0, L, stage))
            if f is None:
                return None
            parts.append(np.fromfile(f, dtype).reshape(nt, width))
        a = np.concatenate(parts, axis=0)
        return a.astype(np.float64) if dtype in (np.float32, np.float16) else a

    def per_chunk(self, L, stage, dtype=np.float32):
        for p0, nt in self.chunks:
            f = self.files.get((p0, L, stage))
            if f is not None:
                yield p0, nt, np.fromfile(f, dtype)

    def block_calls(self, ratio, n_rows):
        """Per compressed row: how many rows the call that completed it compressed (the indexer K GEMM's batch)."""
        out = np.zeros(n_rows, np.int64)
        for p0, nt in self.chunks:
            blocks = [p // ratio for p in range(p0, p0 + nt) if (p + 1) % ratio == 0 and p // ratio < n_rows]
            for b in blocks:
                out[b] = len(blocks)
        return out


def block_rows(tr, L, stage, width, ratio, n_rows, dtype=np.float32):
    """The rows of a per-block trace (latent, comp_k, ...) as [n_rows, width] float64, block r covering positions
    [r * ratio, (r + 1) * ratio). Each call traces the blocks it completes in position order; ggml's ratio-2 graph also
    compresses one masked scratch row on a step that completes none, which is skipped."""
    rows = {}
    for p0, nt, a in tr.per_chunk(L, stage, dtype):
        a = a.reshape(-1, width)
        blocks = [p // ratio for p in range(p0, p0 + nt) if (p + 1) % ratio == 0]
        if len(a) == len(blocks):
            for i, b in enumerate(blocks):
                rows[b] = a[i]
        elif not blocks and len(a) == 1 and tr.engine == "ggml":
            continue
        else:
            raise SystemExit(f"{tr.engine} L{L} {stage} p0 {p0}: {len(a)} rows for {len(blocks)} completed blocks")
    missing = [r for r in range(n_rows) if r not in rows]
    if missing:
        raise SystemExit(f"{tr.engine} L{L} {stage}: rows {missing[:5]}... missing")
    return np.stack([rows[r] for r in range(n_rows)]).astype(np.float64)


def per_position(tr, L, stage, dtype=np.float32):
    """{position: row} of a [nt, width] per-query trace whose width varies by chunk (index scores, masks, top-k)."""
    out = {}
    for p0, nt, a in tr.per_chunk(L, stage, dtype):
        a = a.reshape(nt, -1)
        for i in range(nt):
            out[p0 + i] = a[i]
    return out


# -------------------------------------------------------------------------------------------- math
def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def tok_rel(a, b):
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-300)


def rms_norm(x, w, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * w


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def hc_gates(xs, fn, scale, base, iters, eps, rms_eps):
    """xs [T, 4, e] -> pre [T,4], post [T,4], comb [T,16] (idst + 4*isrc), as ts_hc_gates_comb."""
    flat = xs.reshape(len(xs), -1)
    inv = 1.0 / np.sqrt((flat * flat).mean(axis=1, keepdims=True) + rms_eps)
    mixes = (flat @ fn.T) * inv
    pre = sigmoid(mixes[:, 0:4] * scale[0] + base[0:4]) + eps
    post = 2.0 * sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    c = (mixes[:, 8:24] * scale[2] + base[8:24]).reshape(-1, 4, 4)  # [T, isrc, idst]
    c = np.exp(c - c.max(axis=2, keepdims=True))
    c = c / c.sum(axis=2, keepdims=True) + eps
    for it in range(iters):
        if it > 0:
            c = c / (eps + c.sum(axis=2, keepdims=True))   # each source over its destinations
        c = c / (eps + c.sum(axis=1, keepdims=True))       # each destination over its sources
    return pre, post, c.reshape(-1, 16)


def hc_post(xs, out, post, comb):
    """xsOut[t, idst, e] = out[t, e] * post[t, idst] + sum_isrc xs[t, isrc, e] * comb[t, idst + 4 * isrc]."""
    c = comb.reshape(-1, 4, 4)  # [T, isrc, idst]
    return out[:, None, :] * post[:, :, None] + np.einsum("tse,tsd->tde", xs, c)


def collapse(xs, w):
    return np.einsum("tse,ts->te", xs, w)


class Rope:
    """RoPE table [n_pos, n_rot] (cos, sin interleaved per pair) plus the angles. exact: float64 angles
    theta = p * base^(-2i/n_rot), YaRN-mixed as ggml's rope_yarn (ramp over the pair index between the corr dims;
    the magnitude scale attn_factor * (1 + 0.1 ln(1/s)) is exactly 1). engine: the direct engine's float32 table
    (DeepSeek4CudaExecutor.BuildRopeTable: iterated float32 theta, float32 cos/sin)."""

    def __init__(self, n_pos, n_rot, base, freq_scale=1.0, ext=0.0, corr=(0.0, 0.0), mode="exact"):
        half = n_rot // 2
        i = np.arange(half, dtype=np.float64)
        if ext:
            y = (i - corr[0]) / max(0.001, corr[1] - corr[0])
            mix = (1.0 - np.clip(y, 0.0, 1.0)) * ext
        else:
            mix = np.ones(half)
        if mode == "exact":
            ext_th = np.arange(n_pos, dtype=np.float64)[:, None] * (base ** (-2.0 * i / n_rot))[None, :]
            th = ext_th * freq_scale * (1.0 - mix) + ext_th * mix if ext else ext_th
            c, s = np.cos(th), np.sin(th)
        else:
            attn = F32(1.0) if not ext else F32(1.0) / (F32(1.0) + F32(0.1) * F32(math.log(1.0 / freq_scale)))
            ts = F32(base) ** F32(-2.0 / n_rot)
            theta = np.arange(n_pos, dtype=F32)
            c = np.zeros((n_pos, half), F32)
            s = np.zeros((n_pos, half), F32)
            th = np.zeros((n_pos, half))
            for k in range(half):
                interp = F32(freq_scale) * theta
                t = interp
                ms = attn
                if ext:
                    ramp = F32(mix[k])
                    t = (interp * (F32(1) - ramp) + theta * ramp).astype(F32)
                    ms = F32(ms * (F32(1) + F32(0.1) * F32(math.log(1.0 / freq_scale))))
                c[:, k] = np.cos(t) * ms
                s[:, k] = np.sin(t) * ms
                th[:, k] = t
                theta = (theta * ts).astype(F32)
            c, s = c.astype(np.float64), s.astype(np.float64)
        self.tab = np.zeros((n_pos, n_rot))
        self.tab[:, 0::2], self.tab[:, 1::2] = c, s
        self.theta = np.abs(th)

    def angle_err(self, pos):
        """A bound on the direct engine's float32 table angle error at these positions, per pair [len(pos), n_rot/2]:
        the iterated theta (one product and the rounded per-pair factor per step) and the rounded argument."""
        k = np.arange(self.theta.shape[1])
        return self.theta[pos] * (2.0 * k[None, :] + 4.0) * U + 4.0 * U


def rope(x, tab, pos, n_rot, inverse=False):
    """Rotate the last n_rot dims of x [T, ..., D] (interleaved pairs) at positions pos."""
    y = x.copy()
    rb = x.shape[-1] - n_rot
    c = tab[pos][:, 0::2]
    s = tab[pos][:, 1::2]
    shape = (len(pos),) + (1,) * (x.ndim - 2) + (n_rot // 2,)
    c, s = c.reshape(shape), s.reshape(shape)
    x0, x1 = x[..., rb::2], x[..., rb + 1::2]
    if inverse:
        y[..., rb::2], y[..., rb + 1::2] = x0 * c + x1 * s, -x0 * s + x1 * c
    else:
        y[..., rb::2], y[..., rb + 1::2] = x0 * c - x1 * s, x0 * s + x1 * c
    return y


# ---------------------------------------------------------- the trained cache formats (dsv41_quant.h, bit for bit)
def bf16(v):
    b = np.asarray(v, F32).view(np.uint32).astype(np.uint64)
    fin = (b & 0x7F800000) != 0x7F800000
    r = np.where(fin, (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000, b)
    return r.astype(np.uint32).view(F32)


def round_even(v):
    base = np.floor(v)
    frac = v - base
    up = (frac > 0.5) | ((frac == 0.5) & ((base.astype(np.int64) & 1) != 0))
    return (base + up).astype(F32)


def e4m3(v):
    v = np.asarray(v, F32)
    mag = np.minimum(np.abs(v), F32(448.0))
    _, ex = np.frexp(np.maximum(mag, F32(2.0 ** -6)))
    step = np.ldexp(F32(1.0), ex - 4).astype(F32)
    return np.copysign((round_even((mag / step).astype(F32)) * step).astype(F32), v).astype(F32)


def e4m3_step(mag):
    """The E4M3 grid spacing at a magnitude (in the scaled domain): 2^(e - 3) for normals, 2^-9 below 2^-6."""
    mag = np.asarray(mag, np.float64)
    _, ex = np.frexp(np.maximum(mag, 2.0 ** -6))
    return np.ldexp(1.0, ex - 4)


def e2m1(v):
    v = np.asarray(v, F32)
    m = np.abs(v)
    r = np.where(m <= 0.25, 0.0, np.where(m < 0.75, 0.5, np.where(m <= 1.25, 1.0, np.where(m < 1.75, 1.5,
        np.where(m <= 2.5, 2.0, np.where(m < 3.5, 3.0, np.where(m <= 5.0, 4.0, 6.0))))))).astype(F32)
    return np.copysign(r, v).astype(F32)


def ceil_pow2(v):
    b = np.ascontiguousarray(v, F32).view(np.uint32)
    e = ((b >> 23) & 255).astype(np.int32) - 127 + ((b & 0x7FFFFF) != 0)
    return np.ldexp(F32(1.0), e).astype(F32)


def quant_scale(x, mode):
    """The per-group scale of dsv41_quant (mode 0/1: 32 values, power of two; mode 2: 16 values, E4M3)."""
    x = np.asarray(x, F32)
    blk = 16 if mode == 2 else 32
    v = bf16(x).reshape(-1, blk)
    amax = np.abs(v).max(axis=1, keepdims=True)
    if mode == 0:
        return ceil_pow2(np.maximum(amax, F32(1e-4)) * (F32(1.0) / F32(448.0)))
    if mode == 1:
        return ceil_pow2(np.maximum(amax, F32(6.0 * 2.0 ** -126)) * (F32(1.0) / F32(6.0)))
    return e4m3((np.maximum(amax, F32(6.0 * 2.0 ** -9)) / F32(6.0)).astype(F32))


def dsv41_quant(x, mode):
    """ts_dsv41_quant_row / TSG_DSV41_FUSED_QUANT on float32 rows: bf16, per-group amax scale (mode 0: 32, E4M3 with a
    power-of-two scale; mode 1: 32, E2M1 with a power-of-two scale; mode 2: 16, E2M1 with an E4M3 scale), bf16."""
    x = np.asarray(x, F32)
    shp = x.shape
    blk = 16 if mode == 2 else 32
    v = bf16(x).reshape(-1, blk)
    scale = quant_scale(x, mode)
    n = (v / scale).astype(F32)
    q = e4m3(n) if mode == 0 else e2m1(n)
    return bf16((q * scale).astype(F32)).reshape(shp)


def cache_row(x, mode):
    """A committed cache row: the format rounding, then the F16 cache."""
    return dsv41_quant(x, mode).astype(np.float16).astype(np.float64)


def format_check(engine_rows, pre, mode, tol, f16=True):
    """A format stage: the engine's stored rows against the replayed rounding of the exact pre-quantization rows.
    An element that differs is a tie when the replay reproduces the engine's value for some perturbation of the
    exact input within tol (elementwise absolute bound): +tol, -tol, +sign*tol, -sign*tol."""
    def q(x):
        r = dsv41_quant(x, mode)
        return r.astype(np.float16).astype(np.float64) if f16 else r.astype(np.float64)
    exact = q(pre)
    diff = engine_rows != exact
    explained = np.zeros_like(diff)
    sg = np.sign(pre)
    for variant in (pre + tol, pre - tol, pre + sg * tol, pre - sg * tol):
        explained |= q(variant) == engine_rows
    une = diff & ~explained
    out = {"elements": int(diff.size), "differing": int(diff.sum()), "tie_explained": int((diff & explained).sum()),
           "unexplained": int(une.sum()), "rel_l2": rel(engine_rows, exact)}
    if une.any():
        idx = np.argwhere(une)[:8]
        out["unexplained_examples"] = [{"row": int(r), "dim": int(d), "engine": float(engine_rows[r, d]),
                                        "replay": float(exact[r, d]), "pre": float(pre[r, d]), "tol": float(tol[r, d])}
                                       for r, d in idx]
    return out, exact


def fp8_flip_stats(k, k_exact, kv_pre, kv_pre_model):
    """raw_k (mode 0) against its replay, per row: the number of differing elements, and how many differ by MORE than
    the format noise envelope 2 x step + 4 x sigma. step: the E4M3 grid step at the replay's power-of-two group scale
    (a bin flip is one step; a group-scale change by one power of two, the group amax crossing a power of two,
    re-quantizes the group on a grid twice as coarse: up to 1.5 steps). sigma: the row's rms pre-quantization noise of
    the q8_1 GEMM (precision model minus exact), which an element whose magnitude is small against its group amax
    sees as several of its fine steps."""
    scale = np.repeat(quant_scale(kv_pre, 0).reshape(len(k), -1), 32, axis=1).astype(np.float64)
    sigma = np.sqrt(np.mean((kv_pre_model - kv_pre) ** 2, axis=1, keepdims=True))
    diff = k != k_exact
    big = np.maximum(np.abs(k), np.abs(k_exact)) / scale
    noise = np.abs(k - k_exact) <= (2.0 * e4m3_step(big) * scale + 4.0 * sigma) * (1.0 + 1e-6)
    return diff.sum(axis=1), (diff & ~noise).sum(axis=1)


# ------------------------------------------------------------- ggml-cuda's activation roundings (precision model)
DS_HALF_TYPES = {"Q4_0", "Q4_1", "Q5_1", "Q4_K", "Q5_K", "Q2_K", "IQ1_S"}   # MMQ q8_1 layouts with a half scale
D2S6_TYPES = {"Q2_K"}                     # MMQ_Q8_1_DS_LAYOUT_D2S6: one (half) scale per 64 values


def roundf(v):
    """C roundf (half away from zero), on float32 values evaluated exactly in float64."""
    v = np.asarray(v, np.float64)
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def q8_1(x, mmq, d_half, block=32):
    """ggml-cuda's q8_1 activation operand, per `block` values of the last axis (32; 64 in MMQ's D2S6 layout).
    MMQ (quantize_mmq_q8_1): d_inv = 127/amax, q = roundf(x d_inv), d = 1/d_inv (F32, or half in the DS4/D2S6
    layouts). MMVQ (quantize_q8_1): d = amax/127, q = roundf(x/d), d stored as half."""
    x = np.asarray(x, F32)
    v = x.reshape(-1, block)
    amax = np.abs(v).max(axis=1, keepdims=True)
    nz = amax > 0
    safe = np.where(nz, amax, F32(1.0)).astype(F32)
    if mmq:
        d_inv = (F32(127.0) / safe).astype(F32)
        q = roundf((v * d_inv).astype(F32))
        d = (F32(1.0) / d_inv).astype(F32)
        if d_half:
            d = d.astype(np.float16).astype(F32)
    else:
        d = (safe / F32(127.0)).astype(F32)
        q = roundf((v / d).astype(F32))
        d = d.astype(np.float16).astype(F32)
    return np.where(nz, q * d.astype(np.float64), 0.0).reshape(x.shape)


def act_model(x, wtype, calls):
    """The activation operand ggml-cuda's GEMM with a `wtype` weight actually multiplies, row by row; `calls` is the
    number of rows of the call each row was computed in. Quantized weights: q8_1 (MMVQ at <= 8 rows, MMQ above; MMQ
    quantizes Q2_K's operand per 64 values). The model ignores the layouts' half-precision partial sums (the IQ1_S
    delta and the Q2_K min terms use them in place of the quantized values' sums).
    BF16/F16 weights: rounded to the weight type above 3 rows (MMF / cuBLAS), F32 at <= 3 (MMVF). F32: unchanged."""
    x = np.asarray(x, np.float64)
    calls = np.asarray(calls)
    out = x.copy()
    if wtype in ("F32", "F64"):
        return out
    if wtype in ("BF16", "F16"):
        rnd = calls > 3
        if rnd.any():
            xr = x[rnd].astype(F32)
            out[rnd] = (bf16(xr) if wtype == "BF16" else xr.astype(np.float16)).astype(np.float64)
        return out
    for mmq in (False, True):
        sel = (calls > 8) if mmq else (calls <= 8)
        if sel.any():
            out[sel] = q8_1(x[sel], mmq, mmq and wtype in DS_HALF_TYPES, 64 if (mmq and wtype in D2S6_TYPES) else 32)
    return out


def softplus(x):
    return np.logaddexp(0.0, x)


def swiglu(g, u, limit):
    if limit > 0:
        u = np.clip(u, -limit, limit)
        g = np.minimum(g, limit)
    return u * (g / (1.0 + np.exp(-g)))


def _segments(cand, exact, segs, positions):
    if positions is None:
        return {f"{a}:{b}": rel(cand[a:b], exact[a:b]) for a, b in segs if a < min(b, len(cand))}
    pp = np.asarray(positions)
    return {f"{a}:{b}": rel(cand[(pp >= a) & (pp < b)], exact[(pp >= a) & (pp < b)])
            for a, b in segs if ((pp >= a) & (pp < b)).any()}


def summary(name, cand, exact, segs, extra=None, positions=None, model=None):
    r = {"stage": name, "rel_l2": rel(cand, exact)}
    t = tok_rel(cand, exact)
    r["tok_median"] = float(np.median(t))
    r["tok_max"] = float(t.max())
    r["tok_argmax"] = int(t.argmax()) if positions is None else int(positions[int(t.argmax())])
    r["segments"] = _segments(cand, exact, segs, positions)
    if model is not None:
        r["model_rel_l2"] = rel(model, exact)
        r["model_segments"] = _segments(model, exact, segs, positions)
        r["replay_rel_l2"] = rel(cand, model)
    if extra:
        r.update(extra)
    return r


def ragged_summary(name, cand, exact, positions, segs, extra=None, model=None):
    """rel-L2 of per-position vectors of different lengths (score rows): concatenated, per position, per segment."""
    cflat, eflat = np.concatenate(cand), np.concatenate(exact)
    pos_flat = np.concatenate([np.full(len(c), p) for c, p in zip(cand, positions)])
    per = np.array([np.linalg.norm(c - e) / max(np.linalg.norm(e), 1e-300) for c, e in zip(cand, exact)])

    def segd(a_flat):
        return {f"{a}:{b}": rel(a_flat[(pos_flat >= a) & (pos_flat < b)], eflat[(pos_flat >= a) & (pos_flat < b)])
                for a, b in segs if ((pos_flat >= a) & (pos_flat < b)).any()}
    r = {"stage": name, "rel_l2": rel(cflat, eflat), "tok_median": float(np.median(per)), "tok_max": float(per.max()),
         "tok_argmax": int(positions[int(per.argmax())]), "positions": [int(min(positions)), int(max(positions)), len(positions)],
         "values": int(len(cflat)), "segments": segd(cflat)}
    if model is not None:
        mflat = np.concatenate(model)
        r["model_rel_l2"] = rel(mflat, eflat)
        r["model_segments"] = segd(mflat)
        r["replay_rel_l2"] = rel(cflat, mflat)
    if extra:
        r.update(extra)
    return r


# ------------------------------------------------------------------------------------------- Engram
class Engram:
    """The V4.1 Engram lookup recomputed from token ids: dsv41_engram.h's hash_tokens (token map, rolling
    multiply-xor n-gram hash, per-column prime buckets) and the host lookup's to_float of the hashed table rows
    (TS_DSV41_ENGRAM_DEVICE=0). Both GGUF layouts: TensorSharp's (engram.*, blk.N.engram_embd) and llama.cpp's
    (ple.*, one combined per_layer_token_embd table whose per-layer offsets are rebased onto the layer's slice)."""

    def __init__(self, g):
        a = g.kv["general.architecture"]
        kv = g.kv
        self.g = g
        self.combined = f"{a}.ple.layers" in kv
        token_map = np.asarray(kv[f"{a}.engram.token_map"], np.int64)
        self.token_map = token_map
        if self.combined:
            self.ngram = int(kv[f"{a}.ple.ngram_size"])
            self.nheads = int(kv[f"{a}.ple.heads_per_ngram"])
            self.hd = int(kv[f"{a}.embedding_length_per_layer_input"])
            ids = list(kv[f"{a}.ple.layers"])
            mult = np.asarray(kv[f"{a}.ple.layer_multipliers"], np.uint64)
            primes = np.asarray(kv[f"{a}.ple.head_vocab_sizes"], np.uint64)
            offs = np.asarray(kv[f"{a}.ple.head_offsets"], np.uint64)
            self.pad = int(token_map[int(kv[f"{a}.engram.pad_token_id"])])
        else:
            self.ngram = int(kv[f"{a}.engram.max_ngram_size"])
            self.nheads = int(kv[f"{a}.engram.head_count"])
            self.hd = int(kv[f"{a}.engram.key_length"])
            ids = list(kv[f"{a}.engram.layer_ids"])
            mult = np.asarray(kv[f"{a}.engram.multipliers"], np.uint64)
            primes = np.asarray(kv[f"{a}.engram.primes"], np.uint64)
            offs = np.asarray(kv[f"{a}.engram.offsets"], np.uint64)
            self.pad = int(kv[f"{a}.engram.pad_id"])
        self.cols = (self.ngram - 1) * self.nheads
        self.layers = {}
        for i, L in enumerate(ids):
            o = offs[i * self.cols:(i + 1) * self.cols]
            first = int(o[0]) if self.combined else 0
            self.layers[int(L)] = {"mult": mult[i * self.ngram:(i + 1) * self.ngram],
                                   "primes": primes[i * self.cols:(i + 1) * self.cols],
                                   "offsets": (o - np.uint64(first)) if self.combined else o,
                                   "first": first,
                                   "table": "per_layer_token_embd.weight" if self.combined else f"blk.{int(L)}.engram_embd.weight"}

    def hash_tokens(self, L, tokens):
        """[T, cols] row ids into layer L's table slice (dsv41_engram.h hash_tokens, history starting at position 0)."""
        lay = self.layers[L]
        hist = self.token_map[np.asarray(tokens, np.int64)].astype(np.uint64)
        T = len(hist)
        out = np.zeros((T, self.cols), np.int64)
        rolling = np.zeros(T, np.uint64)
        pos = np.arange(T)
        for shift in range(self.ngram):
            blocked = pos < shift
            tok = np.where(blocked, np.uint64(self.pad), hist[np.maximum(pos - shift, 0)])
            with np.errstate(over="ignore"):
                rolling ^= tok * lay["mult"][shift]
            if not shift:
                continue
            for head in range(self.nheads):
                col = (shift - 1) * self.nheads + head
                out[:, col] = (rolling % lay["primes"][col] + lay["offsets"][col]).astype(np.int64)
        return out

    def lookup(self, L, tokens):
        """The F32 rows the lookup stages, [T, cols * head_dim] float64."""
        lay = self.layers[L]
        ids = self.hash_tokens(L, tokens) + lay["first"]
        uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
        rows = self.g.gather(lay["table"], uniq)
        return rows[inv].reshape(len(ids), self.cols * self.hd)


# ------------------------------------------------------------------------------------------- layers
class Model:
    def __init__(self, g):
        self.g = g
        a = g.kv["general.architecture"]
        self.e = int(g.kv[f"{a}.embedding_length"])
        self.nh = int(g.kv[f"{a}.attention.head_count"])
        self.hd = int(g.kv[f"{a}.attention.key_length"])
        self.nrot = int(g.kv[f"{a}.rope.dimension_count"])
        self.qlora = int(g.kv[f"{a}.attention.q_lora_rank"])
        self.og = int(g.kv[f"{a}.attention.output_group_count"])
        self.olora = int(g.kv[f"{a}.attention.output_lora_rank"])
        self.nswa = int(g.kv[f"{a}.attention.sliding_window"])
        self.eps = float(np.float32(g.kv.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)))
        self.nex = int(g.kv[f"{a}.expert_count"])
        self.used = int(g.kv[f"{a}.expert_used_count"])
        self.wscale = float(np.float32(g.kv.get(f"{a}.expert_weights_scale", 1.0)))
        self.wnorm = bool(g.kv.get(f"{a}.expert_weights_norm", False))
        self.iters = int(g.kv.get(f"{a}.hyper_connection.sinkhorn_iterations", 20))
        self.hceps = float(np.float32(g.kv.get(f"{a}.hyper_connection.epsilon", 1e-6)))
        self.clamp_exp = [float(v) for v in g.kv.get(f"{a}.swiglu_clamp_exp", [])]
        self.clamp_sh = [float(v) for v in g.kv.get(f"{a}.swiglu_clamp_shexp", self.clamp_exp)]
        self.ratios = [int(r) for r in g.kv[f"{a}.attention.compress_ratios"]]
        self.rope_base = float(g.kv.get(f"{a}.rope.freq_base", 10000.0))
        self.comp_base = float(g.kv.get(f"{a}.attention.compress_rope_freq_base", self.rope_base))
        factor = float(g.kv.get(f"{a}.rope.scaling.factor", 0.0) or 0.0)
        self.yarn_scale = 1.0 / factor if factor > 0 else 1.0
        self.yarn_ext = 1.0 if factor > 0 else 0.0
        self.n_ctx_orig = int(g.kv.get(f"{a}.rope.scaling.original_context_length", 0))
        self.beta_fast = float(g.kv.get(f"{a}.rope.scaling.yarn_beta_fast", 32.0))
        self.beta_slow = float(g.kv.get(f"{a}.rope.scaling.yarn_beta_slow", 1.0))
        el = g.kv.get(f"{a}.ple.layers", g.kv.get(f"{a}.engram.layer_ids", []))
        self.engram_layers = [int(v) for v in el]
        self.kv_sources = [int(v) for v in g.kv.get(f"{a}.attention.kv_source_layers", [])]
        self.index_sources = [int(v) for v in g.kv.get(f"{a}.attention.indexer.source_layers", [])]
        self.cand_source = int(g.kv.get(f"{a}.attention.indexer.candidate_source_layer", -1))
        self.cand_topk = int(g.kv.get(f"{a}.attention.indexer.candidate_top_k_blocks",
                                      g.kv.get(f"{a}.attention.indexer.candidate_top_k", 0)))
        self.cand_block = int(g.kv.get(f"{a}.attention.indexer.candidate_block_size", 1))
        self.idx_topk = int(g.kv.get(f"{a}.attention.indexer.top_k", 0))
        self.idx_nh = int(g.kv.get(f"{a}.attention.indexer.head_count", 0))
        self.idx_hd = int(g.kv.get(f"{a}.attention.indexer.key_length", 0))
        self.cache = {}

    def corr_dims(self):
        def dim(beta):
            return self.nrot * math.log(self.n_ctx_orig / (beta * 2 * math.pi)) / (2 * math.log(self.comp_base))
        return max(0.0, math.floor(dim(self.beta_fast))), min(self.nrot - 1.0, math.ceil(dim(self.beta_slow)))

    def kv_source_of(self, L):
        return max(s for s in self.kv_sources if s <= L and self.ratios[s] == self.ratios[L])

    def index_source_of(self, L):
        return max(s for s in self.index_sources if s <= L and self.ratios[s] == self.ratios[L])

    def w(self, name):
        if name not in self.cache:
            self.cache[name] = self.g.get(name)
        return self.cache[name]

    def v(self, name):
        return self.w(name).reshape(-1)

    def t(self, name):
        return self.g.type_name(name)


# The attention semantics the oracle implements (a --mutate test flag perturbs them on purpose).
BASE_SEM = {"window_delta": 0, "sinks": True, "select": True, "rot_end": False}
MUTATIONS = {"sinks-off": {"sinks": False}, "window-1": {"window_delta": -1}, "window+1": {"window_delta": 1},
             "rot-end": {"rot_end": True}, "select-all": {"select": False}}


def attention_exact(m, L, q, raw, pos, comp=None, rows_of=None, ratio=0, sem=None, sinks=None):
    """Attention with sinks over the engine's own rotated q [n, nh, hd] at positions pos: per query the raw window of
    its own committed keys raw [T, hd] (positions p - window + 1 .. p) plus, on a compressed layer, the compressed
    cache rows comp [R, hd] its selection names (rows_of(p): row ids, used verbatim; None = every visible row
    (p + 1) // ratio). K doubles as V. sem: the semantics (window, sinks, selection on/off). The core [n, nh, hd]."""
    sem = BASE_SEM if sem is None else sem
    sinks = m.v(f"blk.{L}.attn_sinks.weight") if sinks is None else sinks
    window = m.nswa + sem["window_delta"]
    scale = 1.0 / np.sqrt(m.hd)
    out = np.zeros_like(q)
    for t in range(len(q)):
        p = int(pos[t])
        lo = max(0, p - window + 1)
        keys = raw[lo:p + 1]
        if comp is not None:
            rows = rows_of(p) if (rows_of is not None and sem["select"]) else None
            rows = np.arange((p + 1) // ratio) if rows is None else rows
            if len(rows):
                keys = np.concatenate([keys, comp[rows]], axis=0)
        s = (q[t] @ keys.T) * scale                         # [nh, W]
        if sem["sinks"]:
            mx = np.maximum(s.max(axis=1), sinks)
            w = np.exp(s - mx[:, None])
            den = w.sum(axis=1) + np.exp(sinks - mx)
        else:
            mx = s.max(axis=1)
            w = np.exp(s - mx[:, None])
            den = w.sum(axis=1)
        out[t] = (w @ keys) / den[:, None]
    return out


def out_projection(m, L, core, pos, tab, calls=None):
    """Inverse RoPE plus the grouped LoRA out-projection; with calls: ggml's activation roundings (precision model)."""
    o = rope(core, tab, pos, m.nrot, inverse=True)        # [T, nh, hd]
    hpg = m.nh // m.og
    na, nb = f"blk.{L}.attn_output_a.weight", f"blk.{L}.attn_output_b.weight"
    woa = m.w(na)                                          # [og*olora, hpg*hd]
    wob = m.w(nb)                                          # [e, og*olora]
    if calls is not None:
        o = act_model(o.reshape(len(o), -1), m.t(na), calls).reshape(o.shape)
    og = np.zeros((len(o), m.og * m.olora))
    for g in range(m.og):
        og[:, g * m.olora:(g + 1) * m.olora] = o[:, g * hpg:(g + 1) * hpg, :].reshape(len(o), -1) @ woa[g * m.olora:(g + 1) * m.olora].T
    if calls is not None:
        og = act_model(og, m.t(nb), calls)
    return og @ wob.T


def index_scores(iq, iw, ik, nvis, keep=None, bound=False, chunk=64):
    """sum_h iw[t, h] * relu(iq[t, h] . ik[r]) for r < nvis[t] (float64), rows outside keep[t] -inf.
    With bound: also the F32 accumulation bound (ID + IH + 4) * 2^-24 * sum_h |iw| sum_d |iq k| per row."""
    T, H, D = iq.shape
    gamma = (D + H + 4) * U
    out, bnd = [], []
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        nv = int(max(nvis[t0:t1]))
        k = ik[:nv]
        d = (iq[t0:t1].reshape(-1, D) @ k.T).reshape(t1 - t0, H, nv)
        s = np.matmul(iw[t0:t1, None, :], np.maximum(d, 0.0))[:, 0, :]
        if bound:
            a = (np.abs(iq[t0:t1]).reshape(-1, D) @ np.abs(k).T).reshape(t1 - t0, H, nv)
            b = gamma * np.matmul(np.abs(iw[t0:t1])[:, None, :], a)[:, 0, :]
        for i in range(t1 - t0):
            n = int(nvis[t0 + i])
            row = s[i, :n].copy()
            if keep is not None and keep[t0 + i] is not None:
                row[~keep[t0 + i]] = -np.inf
            out.append(row)
            if bound:
                bnd.append(b[i, :n].copy())
    return (out, bnd) if bound else out


def topk_order(scores, k):
    """The exact top-k of a score row (ties to the lower row id) and the boundary gap s_(k) - s_(k+1)."""
    order = np.argsort(-scores, kind="stable")
    k = min(k, len(scores))
    gap = float(scores[order[k - 1]] - scores[order[k]]) if len(scores) > k else math.inf
    return set(int(r) for r in order[:k]), gap


def consistent(scores, sel, bound=None):
    """Whether the SET `sel` is a top-|sel| of `scores` after moving each score by at most its bound: the violation
    max_{r not in sel}(s_r - b_r) - min_{r in sel}(s_r + b_r) must be <= 0 (with bound 0: exact ties only).
    Set semantics only: the list checks (count, duplicates, visibility) are selection_integrity's."""
    n = len(scores)
    ins = np.zeros(n, bool)
    ins[list(sel)] = True
    b = np.zeros(n) if bound is None else bound
    if not ins.any() or ins.all():
        return 0.0
    hi_out = np.max(scores[~ins] - b[~ins])
    lo_in = np.min(scores[ins] + b[ins])
    if hi_out == -np.inf:
        return 0.0
    if lo_in == -np.inf:
        return math.inf
    return float(hi_out - lo_in)


def selection_integrity(ids, nvis, k, scores, engine):
    """The list checks of an engine's indexer top-k at one query (S8c finding 1): the id list itself, not its set.
    cuda: index_count must be min(k, nvis), every id visible (< nvis) and unique. ggml: its top list is
    min(k, n_kv) wide over a bucketed key range, so invisible (masked) ids pad it when nvis < k; its visible ids
    must be unique and min(k, nvis) many. Both: at most max(0, count - finite) selected rows may be -inf (pruned).
    Returns (problems, visible ids verbatim, their sorted unique set)."""
    ids = np.asarray(ids, np.int64)
    problems = []
    want = min(k, nvis)
    if len(ids) and ids.min() < 0:
        problems.append("negative id")
    if engine == "cuda":
        if len(ids) != want:
            problems.append(f"count {len(ids)} != {want}")
        if len(ids) and ids.max() >= nvis:
            problems.append(f"{int((ids >= nvis).sum())} invisible ids")
        vis = ids[(ids >= 0) & (ids < nvis)]
    else:
        vis = ids[(ids >= 0) & (ids < nvis)]
        if len(np.unique(vis)) != want:
            problems.append(f"visible unique ids {len(np.unique(vis))} != {want}")
    dup = len(ids) - len(np.unique(ids))
    if dup:
        problems.append(f"{dup} duplicate ids")
    uniq = np.unique(vis)
    if scores is not None and len(uniq):
        n_fin = int(np.isfinite(scores).sum())
        n_inf = int((~np.isfinite(scores[uniq])).sum())
        if n_inf > max(0, want - n_fin):
            problems.append(f"{n_inf} -inf (pruned) rows selected with {n_fin} finite")
    return problems, vis, uniq


def exact_selection(scores, k, engine_set=None):
    """The exact top-k of an engine's own traced score row (ties to the lower id). When the engine's (validated)
    set is itself an exact top-k it is returned instead: the two differ only by exact ties."""
    kk = min(k, len(scores))
    if engine_set is not None and len(engine_set) == kk and consistent(scores, engine_set) <= 0:
        return np.asarray(sorted(int(r) for r in engine_set), np.int64)
    order = np.argsort(-scores, kind="stable")
    return np.sort(order[:kk]).astype(np.int64)


def pinned_block(p, ratio, block):
    """The block both candidate kernels pin to +inf: pos // block (TSG_DSV41_CANDIDATE_SCORES, ts_dsv41_candidate_f32).
    That is the newest visible row's block only at ratio 1; no other candidate-source ratio is modelled."""
    if ratio != 1:
        raise SystemExit(f"candidate source with compress ratio {ratio}: the kernels pin block pos // block, which is "
                         "the newest compressed row's block only at ratio 1; the oracle does not model this ratio")
    return p // block


def block_scores(row_scores, p, block, ratio=1, row_bound=None):
    """TSG_DSV41_CANDIDATE_SCORES / ts_dsv41_candidate_f32: per block of `block` rows the max visible score, the
    pinned block +inf. Returns block scores (and the per-block bound: max row bound in the block)."""
    nv = len(row_scores)
    nb = (nv + block - 1) // block
    pad = np.full(nb * block, -np.inf)
    pad[:nv] = row_scores
    bs = pad.reshape(nb, block).max(axis=1)
    pb_ = pinned_block(p, ratio, block)
    if pb_ < nb:
        bs[pb_] = np.inf
    if row_bound is None:
        return bs, None
    pb = np.zeros(nb * block)
    pb[:nv] = row_bound
    return bs, pb.reshape(nb, block).max(axis=1)


def block_select(bs, budget):
    """The kept blocks: the best `budget` by score (ties to the lower id), never a -inf block."""
    order = np.argsort(-bs, kind="stable")
    keep = [int(b) for b in order[:budget] if bs[b] > -np.inf]
    return set(keep)


def expert_outputs(m, L, jobs):
    """jobs: {expert: [(key, x_row, calls)]} -> {(key, expert): (exact, ggml-model, exact with the clamp off)}, decoding one
    expert at a time (a float64 expert is ~280 MB, so none is kept)."""
    clamp = m.clamp_exp[L] if L < len(m.clamp_exp) else 0.0
    tg, td = m.t(f"blk.{L}.ffn_gate_exps.weight"), m.t(f"blk.{L}.ffn_down_exps.weight")
    out = {}
    for n, (ex, members) in enumerate(sorted(jobs.items())):
        wg = m.g.expert(f"blk.{L}.ffn_gate_exps.weight", ex)
        wu = m.g.expert(f"blk.{L}.ffn_up_exps.weight", ex)
        wd = m.g.expert(f"blk.{L}.ffn_down_exps.weight", ex)
        xs = np.stack([x for _, x, _ in members])
        calls = np.array([c for _, _, c in members])
        gx, ux = xs @ wg.T, xs @ wu.T
        y = swiglu(gx, ux, clamp) @ wd.T
        yn = swiglu(gx, ux, 0.0) @ wd.T if clamp > 0 else y
        xa = act_model(xs, tg, calls)
        ym = act_model(swiglu(xa @ wg.T, xa @ wu.T, clamp), td, calls) @ wd.T
        for r, (key, _, _) in enumerate(members):
            out[(key, ex)] = (y[r], ym[r], yn[r])
        if n % 64 == 0:
            print(f"  experts L{L}: {n}/{len(jobs)}", flush=True)
    return out


def shared_exact(m, L, x, calls):
    """The shared expert: exact, under ggml's activation roundings, and exact with the clamp off."""
    clamp = m.clamp_sh[L] if L < len(m.clamp_sh) else 0.0
    ng, nu, nd = (f"blk.{L}.ffn_{k}_shexp.weight" for k in ("gate", "up", "down"))
    gx, ux = x @ m.w(ng).T, x @ m.w(nu).T
    h = swiglu(gx, ux, clamp)
    hn = swiglu(gx, ux, 0.0)
    xa = act_model(x, m.t(ng), calls)
    hm = act_model(swiglu(xa @ m.w(ng).T, xa @ m.w(nu).T, clamp), m.t(nd), calls)
    return h @ m.w(nd).T, hm @ m.w(nd).T, hn @ m.w(nd).T


def route(m, L, logits):
    bias = m.v(f"blk.{L}.exp_probs_b.bias") if f"blk.{L}.exp_probs_b.bias" in m.g.tensors else np.zeros(m.nex)
    raw = np.sqrt(softplus(logits))
    score = raw + bias
    order = np.argsort(-score, axis=1, kind="stable")
    srt = np.take_along_axis(score, order, axis=1)
    return raw, score, order, srt[:, m.used - 1] - srt[:, m.used], bias


def score_eval_bound(logits, bias):
    """|F32 score - exact score| for the direct engine's router score s = sqrtf(logf(1 + expf(l))) + b
    (ts_dsv4_router_score, l <= 20; l itself above), fast-math off: expf <= 2 ulp (4u relative), the sum 1 + e
    one rounding (absolute u after the log), logf <= 1 ulp (2u relative), sqrtf correctly rounded (u), the add u:
      dsp = 4u e/(1+e) + u + 2u sp   (l > 20: log1p(e^-l), the replaced tail)
      dr  = dsp / max(r, sqrt(dsp)) + u r      ds = dr + u (|s| + dr)."""
    l = np.asarray(logits, np.float64)
    e = np.exp(np.minimum(l, 20.0))
    sp = softplus(l)
    dsp = np.where(l > 20.0, np.exp(-np.minimum(l, 700.0)) + U * l, 4 * U * e / (1 + e) + U + 2 * U * sp)
    r = np.sqrt(sp)
    dr = dsp / np.maximum(r, np.sqrt(dsp)) + U * r
    s = r + bias
    return dr + U * (np.abs(s) + dr)


def score_deriv(l):
    """d sqrt(softplus(l)) / dl = sigmoid(l) / (2 sqrt(softplus(l)))."""
    return sigmoid(l) / (2.0 * np.sqrt(np.maximum(softplus(l), 1e-300)))


def routing_checks(m, L, sel, rl_own, x, W, bias):
    """S8c finding 4: per token, the engine's picks must be (a) a top-k of its OWN traced router logits' scores up to
    score_eval_bound, and (b) a top-k of the exact scores of its traced ffn_input up to score_eval_bound plus the F32
    GEMM bound |dl| <= gamma_K sum_j |x_j W_ij| (gamma_K = K u / (1 - K u)) through the score's slope (max over
    l +- dl). Plus list integrity (k unique ids in range). Returns per-token violations and margins."""
    K = x.shape[1]
    gam = K * U / (1 - K * U)
    l_ex = x @ W.T
    dl = gam * (np.abs(x) @ np.abs(W).T)
    s_own = np.sqrt(softplus(rl_own)) + bias
    s_ex = np.sqrt(softplus(l_ex)) + bias
    b_own = score_eval_bound(rl_own, bias)
    slope = np.maximum(np.maximum(score_deriv(l_ex - dl), score_deriv(l_ex + dl)), score_deriv(l_ex))
    b_ex = slope * dl + score_eval_bound(l_ex, bias)
    own_bad, ex_bad, integ_bad, mism = [], [], [], []
    for t in range(len(x)):
        ids = np.asarray(sel[t], np.int64)
        if len(np.unique(ids)) != m.used or ids.min() < 0 or ids.max() >= m.nex:
            integ_bad.append(int(t))
            continue
        s = set(int(v) for v in ids)
        v_own = consistent(s_own[t], s, b_own[t])
        v_ex = consistent(s_ex[t], s, b_ex[t])
        if v_own > 0:
            own_bad.append((int(t), float(v_own)))
        if v_ex > 0:
            ex_bad.append((int(t), float(v_ex)))
        order = np.argsort(-s_ex[t], kind="stable")
        if set(int(r) for r in order[:m.used]) != s:
            mism.append({"position": int(t), "exact_margin": float(s_ex[t][order[m.used - 1]] - s_ex[t][order[m.used]]),
                         "violation_after_bound": float(v_ex), "bound_at_boundary": float(max(b_ex[t][order[m.used - 1]], b_ex[t][order[m.used]]))})
    return {"own_violations": own_bad, "exact_violations": ex_bad, "integrity_violations": integ_bad,
            "mismatches_vs_exact": mism, "own_bound_median": float(np.median(b_own)),
            "exact_bound_median": float(np.median(b_ex))}


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k, n + 1)]
    mx = max(terms)
    return float(min(1.0, math.exp(mx) * sum(math.exp(t - mx) for t in terms)))


def weights_for(m, raw, sel):
    w = np.take_along_axis(raw, sel, axis=1)
    if m.wnorm:
        w = w / w.sum(axis=1, keepdims=True)
    return w * m.wscale


def verdict_from(v):
    """The discrimination rule on stored distances (all normalized by the BASELINE exact output's norm, so a variant
    that rescales the output is judged on distances, not on its own norm): d_b = |engine - base|, d_v = |engine -
    variant|, effect = |variant - base|. Informative when effect >= 0.1 d_b; expected d_v / d_b for noise orthogonal
    to the move: sqrt(1 + (effect / d_b)^2); required: at least half that excess."""
    d_b, d_v, eff = v["dist_base"], v["dist_variant"], v["dist_effect"]
    rho = eff / max(d_b, 1e-300)
    expected = math.sqrt(1.0 + rho * rho)
    required = 1.0 + 0.5 * (expected - 1.0)
    informative = rho >= 0.1
    measured = d_v / max(d_b, 1e-300)
    v.update(measured_ratio=measured, expected_ratio=expected, required_ratio=required, informative=bool(informative),
             ok=bool((not informative) or measured >= required), base_err=d_b, err=d_v, effect=eff)
    return v


def discrimination_verdict(name, engine_out, ex_b, ex_v):
    """One semantic variant against the baseline (see discriminate and verdict_from)."""
    nb = max(float(np.linalg.norm(ex_b)), 1e-300)
    return verdict_from({"variant": name, "dist_base": float(np.linalg.norm(engine_out - ex_b)) / nb,
                         "dist_variant": float(np.linalg.norm(engine_out - ex_v)) / nb,
                         "dist_effect": float(np.linalg.norm(ex_v - ex_b)) / nb})


def discriminate(m, L, ratio, q_sub, pos_sub, raw_cache, comp_for, rows_of, ao_sub, project, base_sem, sinks=None):
    """S8c finding 2 (self-test of the oracle's attention semantics): the engine's traced attn_out against the oracle
    under its baseline semantics and under each perturbation (sinks toggled, window -1/+1; compressed layers:
    selection toggled; ratio 2: compressed rows rotated at the other end of their block). A variant is informative
    when its exact output moves by effect >= 0.1 x the baseline error; with noise orthogonal to that move its error
    is expected at sqrt(1 + (effect/base)^2) x base, and it must reach at least half that excess. A variant that fits
    better than the baseline means the baseline is not the semantics the engine computes."""
    variants = [("sinks " + ("off" if base_sem["sinks"] else "on"), dict(base_sem, sinks=not base_sem["sinks"])),
                ("window -1", dict(base_sem, window_delta=base_sem["window_delta"] - 1)),
                ("window +1", dict(base_sem, window_delta=base_sem["window_delta"] + 1))]
    if ratio:
        variants.append(("selection " + ("ignored (all visible)" if base_sem["select"] else "applied"),
                         dict(base_sem, select=not base_sem["select"])))
    if ratio > 1:
        variants.append(("compressed rows rotated at block " + ("start" if base_sem["rot_end"] else "end"),
                         dict(base_sem, rot_end=not base_sem["rot_end"])))

    def ao_for(sem):
        comp = comp_for(sem["rot_end"]) if ratio else None
        core = attention_exact(m, L, q_sub, raw_cache, pos_sub, comp, rows_of, ratio, sem, sinks)
        return project(core, pos_sub)
    ex_b = ao_for(base_sem)
    return [discrimination_verdict(name, ao_sub, ex_b, ao_for(sem)) for name, sem in variants]


# Stages the direct engine computes in F32 arithmetic end to end: their local error must be F32 rounding.
EXACT_STAGES = {"attn_input", "attn_core", "attn_core (own list)", "ffn_input", "router", "selw", "engram_output",
                "engram_update", "engram_lookup (own tokens)", "hidden", "latent (own comp_kv)",
                "index_scores (own iq/ik/iw)"}
# Format stages: a committed cache row against the replayed trained-format rounding of the engine's own input.
FORMAT_STAGES = {"comp_k (own latent)", "lid_k (own lid_pre)"}
# Paired stages whose ggml error is bounded two-sidedly by the precision model.
# MoE weight types whose ggml-cuda GEMMs the precision model replays (replay distance far below the error, measured on
# the 64-token calibration); with any other type in a layer's MoE (IQ1_S / Q2_K / Q5_K: half-precision partial sums of
# the DS4 / D2S6 operand layouts; IQ2_S) the model is a lower estimate, not a replay, and ffn_out's two-sided check is
# the MoE semantic discrimination alone.
REPLAYED_MOE_TYPES = {"Q8_0", "IQ2_XXS", "IQ3_XXS"}
MODEL_STAGES = {"engram_kv", "qr", "q (own qr)", "q rope tail", "q (own attn_in)", "raw_k", "latent (own attn_input)",
                "index_scores (own qr/x/latent)", "attn_out", "ffn_out (own sel)"}


def build_table(report):
    rows, order = {}, {}
    for r in report["stages"]:
        rows.setdefault((r["layer"], r["stage"]), {})[r["engine"]] = r
        order.setdefault((r["layer"], r["stage"]), len(order))
    table = []
    for (L, stage), v in sorted(rows.items(), key=lambda kv: (kv[0][0], order[kv[0]])):
        c, g = v.get("cuda"), v.get("ggml")
        cl, gl = (c or {}).get("rel_l2"), (g or {}).get("rel_l2")
        seg = {}
        for k in ((c or {}).get("segments") or {}):
            gs = ((g or {}).get("segments") or {}).get(k)
            seg[k] = {"cuda": c["segments"][k], "ggml": gs}
        table.append({"layer": L, "stage": stage, "cuda_local": cl, "ggml_local": gl,
                      "ratio": (cl / gl) if (cl is not None and gl) else None,
                      "ggml_model": (g or {}).get("model_rel_l2"), "cuda_model": (c or {}).get("model_rel_l2"),
                      "ggml_replay": (g or {}).get("replay_rel_l2"), "moe_types": (g or {}).get("moe_types"),
                      "segments": seg, "cuda_flips": (c or {}).get("flips_by_segment"),
                      "ggml_flips": (g or {}).get("flips_by_segment"),
                      "cuda_beyond": (c or {}).get("beyond_noise_by_segment"),
                      "ggml_beyond": (g or {}).get("beyond_noise_by_segment")})
    return table


def oracle_gate(report, factor, slack, exact_bound, model_factor, replay_factor=0.4, alpha=0.01):
    """Gate v2 (S8c) over a report: see the module docstring. Returns (fails, flags, lines)."""
    fails, flags, lines = [], [], []
    table = report.get("table") or build_table(report)
    for row in table:
        st, L, cl, gl = row["stage"], row["layer"], row["cuda_local"], row["ggml_local"]
        if st in FORMAT_STAGES:
            continue
        # ---- aggregate (direct engine)
        if cl is not None:
            if st in EXACT_STAGES:
                ok = cl <= exact_bound if gl is None else (cl <= exact_bound or cl <= factor * gl + slack)
                rule = f"<= {exact_bound:g}" + ("" if gl is None else f" or <= {factor:g} x ggml {gl:.3e}")
            elif gl is not None:
                ok, rule = cl <= factor * gl + slack, f"<= {factor:g} x ggml {gl:.3e} + {slack:g}"
            else:
                lines.append(f"  info L{L} {st:32s} cuda {cl:.3e} (no ggml trace, no bound)")
                ok, rule = None, None
            if ok is not None:
                lines.append(f"  {'PASS' if ok else 'FAIL'} L{L} {st:32s} cuda {cl:.3e} {rule}")
                if not ok:
                    fails.append(f"L{L} {st}")
        # ---- per segment (S8c finding 5): decode and 1-row tails on their own
        if cl is not None and (gl is not None or st in EXACT_STAGES):
            for seg, v in sorted((row.get("segments") or {}).items(), key=lambda kv: int(kv[0].split(":")[0])):
                cs, gs = v["cuda"], v["ggml"]
                if st in EXACT_STAGES:
                    ok = cs <= exact_bound or (gs is not None and cs <= factor * gs + slack)
                elif gs is None:
                    continue
                else:
                    ok = cs <= factor * gs + slack
                if ok:
                    continue
                if st == "raw_k" and row.get("cuda_flips") and row.get("ggml_flips"):
                    nc, ng = row["cuda_flips"].get(seg, 0), row["ggml_flips"].get(seg, 0)
                    multi = (row.get("cuda_beyond") or {}).get(seg, 0)
                    gmulti = (row.get("ggml_beyond") or {}).get(seg, 0)
                    p = binom_sf(nc, nc + ng, factor / (1.0 + factor))
                    detail = (f"cuda {cs:.3e} vs ggml {gs:.3e} (x{cs / gs:.2f}); cuda {nc} vs ggml {ng} elements differ from "
                              f"the replay, beyond the FP8 noise envelope cuda {multi} / ggml {gmulti}; "
                              f"P(>= {nc} of {nc + ng} | {factor:g}x rate) = {p:.3f}")
                    if multi == 0 and p >= alpha:
                        lines.append(f"  FLAG L{L} {st} segment {seg}: {detail}; explained: FP8 bin flips and "
                                     f"power-of-two scale changes within the counting noise (p >= {alpha:g})")
                        flags.append(f"L{L} {st} [{seg}]")
                        continue
                    lines.append(f"  FAIL L{L} {st} segment {seg}: {detail}")
                else:
                    lines.append(f"  FAIL L{L} {st} segment {seg}: cuda {cs:.3e} vs "
                                 + (f"ggml {gs:.3e} (x{cs / gs:.2f})" if gs else f"bound {exact_bound:g}"))
                fails.append(f"L{L} {st} [{seg}]")
        # ---- two-sided (S8c finding 2): ggml against the precision model of its own arithmetic
        if gl is not None:
            if st in EXACT_STAGES:
                ok = gl <= exact_bound
                lines.append(f"  {'PASS' if ok else 'FAIL'} L{L} {st:32s} two-sided: ggml {gl:.3e} <= {exact_bound:g} (F32 stage)")
                if not ok:
                    fails.append(f"L{L} {st} ggml two-sided")
            elif st in MODEL_STAGES and row.get("ggml_model") is not None:
                gm, rp = row["ggml_model"], row.get("ggml_replay")
                types = row.get("moe_types")
                if types is not None and not set(types) <= REPLAYED_MOE_TYPES:
                    lines.append(f"  info L{L} {st:32s} two-sided: ggml {gl:.3e}, model {gm:.3e} (ggml/model {gl / max(gm, 1e-300):.2f}"
                                 + ("" if rp is None else f", replay {rp:.2e}") + f"): weight types "
                                 f"{sorted(set(types) - REPLAYED_MOE_TYPES)} are not replayed; the MoE semantic "
                                 "discrimination is this stage's two-sided check")
                    continue
                ok = gl <= model_factor * gm + slack
                rp_ok = rp is None or rp <= replay_factor * gl + slack
                lines.append(f"  {'PASS' if ok and rp_ok else 'FAIL'} L{L} {st:32s} two-sided: ggml {gl:.3e} <= {model_factor:g} x model "
                             f"{gm:.3e} (ggml/model {gl / max(gm, 1e-300):.2f}); replay rel(ggml, model) "
                             + ("n/a" if rp is None else f"{rp:.2e} <= {replay_factor:g} x ggml"))
                if not ok:
                    fails.append(f"L{L} {st} ggml two-sided")
                if not rp_ok:
                    fails.append(f"L{L} {st} ggml replay")
    for key, f in sorted(report.get("format", {}).items(), key=lambda kv: (int(kv[0].split("|")[0]), kv[0])):
        L, st = key.split("|")
        ok = f["unexplained"] == 0
        lines.append(f"  {'PASS' if ok else 'FAIL'} L{L} {st:32s} {f['differing']}/{f['elements']} elements differ from the "
                     f"format replay, {f['tie_explained']} ties within the stated bound, {f['unexplained']} unexplained "
                     f"(rel-L2 {f['rel_l2']:.2e})")
        if not ok:
            fails.append(f"L{L} {st}")
    for L, r in sorted(report.get("routing", {}).items(), key=lambda kv: int(kv[0])):
        ck = r.get("checks")
        if ck is None:
            lines.append(f"  FAIL L{L} routing: report predates the S8c derived-bound checks")
            fails.append(f"L{L} routing")
            continue
        ok = not ck["own_violations"] and not ck["exact_violations"] and not ck["integrity_violations"]
        lines.append(f"  {'PASS' if ok else 'FAIL'} L{L} routing: picks = top-{r.get('used', 6)} of own logits up to the F32 "
                     f"score bound ({len(ck['own_violations'])} violations), of the exact logits up to the derived bound "
                     f"({len(ck['exact_violations'])}), list integrity ({len(ck['integrity_violations'])}); "
                     f"{len(ck['mismatches_vs_exact'])} picks differ from the exact pick, all within the bound unless a "
                     f"violation is listed")
        if not ok:
            fails.append(f"L{L} routing")
    for L, s in sorted(report.get("selections", {}).items(), key=lambda kv: int(kv[0])):
        for check in s.get("gated", []):
            ok = check["violations"] == 0
            lines.append(f"  {'PASS' if ok else 'FAIL'} L{L} {check['name']}: {check['checked']} positions, "
                         f"{check['violations']} outside the bound ({check['bound']}); {check.get('detail', '')}")
            if not ok:
                fails.append(f"L{L} {check['name']}")
    for name, pr in sorted(report.get("pruning", {}).items()):
        if not pr:
            continue
        if pr["late_positions"] == 0:
            ok = not pr["engaged"]
        else:
            ok = pr["engaged"] and pr["first_pruned"] == pr["expected_first"] and pr["all_late_pruned"] \
                and pr["max_blocks"] <= pr["budget"]
        lines.append(f"  {'PASS' if ok else 'FAIL'} pruning {name}: {pr['pairs_pruned']} (position, layer) pairs pruned, "
                     f"first at {pr['first_pruned']} (expected {pr['expected_first']}), max blocks {pr['max_blocks']} "
                     f"(budget {pr['budget']}), positions >= {pr['expected_first']} pruned at every consumer: "
                     f"{pr['all_late_pruned']}; dropped block fraction at the last position {pr.get('dropped_last')}")
        if not ok:
            fails.append(f"pruning {name}")
    disc = [(L, "", d) for L, d in report.get("discrimination", {}).items()] + \
        [(L, "moe ", d) for L, d in report.get("discrimination_moe", {}).items()]
    for L, kind, d in sorted(disc, key=lambda x: (int(x[0]), x[1])):
        for eng, vs in sorted(d.items()):
            for v in vs:
                if "dist_base" in v:
                    verdict_from(v)
                tag = "PASS" if v["ok"] else "FAIL"
                if not v["informative"]:
                    tag = "info"
                lines.append(f"  {tag} L{L} {kind}discrimination {eng}: {v['variant']}: error {v['measured_ratio']:.3f}x baseline "
                             f"(expected {v['expected_ratio']:.3f}x, required >= {v['required_ratio']:.3f}x; effect "
                             f"{v['effect']:.2e} vs baseline {v['base_err']:.2e})")
                if v["informative"] and not v["ok"]:
                    fails.append(f"L{L} {kind}discrimination {eng} {v['variant']}")
    return fails, flags, lines


# ------------------------------------------------------------------------------------------- state
def _ragged(d):
    ps = sorted(d)
    flat = [np.asarray(d[p]) for p in ps]
    off = np.cumsum([0] + [len(a) for a in flat])
    return np.asarray(ps, np.int64), (np.concatenate(flat) if flat else np.zeros(0)), off.astype(np.int64)


def _unragged(ps, flat, off, conv=lambda a: a):
    return {int(p): conv(flat[off[i]:off[i + 1]]) for i, p in enumerate(ps)}


def save_state(path, T, next_layer, xs_in, pre_prev, G, SEL, SEL_LIST, CAND, CAND_EXACT):
    """The cross-layer state a later layer window needs (S8c: long prompts are traced a few layers at a time)."""
    a = {"meta": np.frombuffer(json.dumps({"T": T, "next_layer": next_layer}).encode(), np.uint8)}
    for eng in ("cuda", "ggml"):
        a[f"{eng}|xs"] = xs_in[eng].astype(np.float32)
        a[f"{eng}|pre_prev"] = pre_prev[eng]
        for src, d in G[eng].items():
            for k, v in d.items():
                a[f"{eng}|G|{src}|{k}"] = v
        for nm, store in (("SEL", SEL), ("SELL", SEL_LIST)):
            for isrc, d in store[eng].items():
                ps, flat, off = _ragged(d)
                a[f"{eng}|{nm}|{isrc}|p"], a[f"{eng}|{nm}|{isrc}|f"], a[f"{eng}|{nm}|{isrc}|o"] = ps, flat.astype(np.int64), off
        ce = CAND_EXACT[eng]
        ps, flat, off = _ragged({p: v[0] for p, v in ce.items()})
        a[f"{eng}|CX|p"], a[f"{eng}|CX|f"], a[f"{eng}|CX|o"] = ps, flat.astype(np.float64), off
    ps, flat, off = _ragged({p: v[1].astype(np.uint8) for p, v in CAND["cuda"].items()})
    a["cuda|C|p"], a["cuda|C|f"], a["cuda|C|o"] = ps, flat.astype(np.uint8), off
    ps, flat, off = _ragged({p: np.asarray(sorted(v), np.int64) for p, v in CAND["ggml"].items()})
    a["ggml|C|p"], a["ggml|C|f"], a["ggml|C|o"] = ps, flat.astype(np.int64), off
    np.savez(path, **a)


def load_state(path, m):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    xs_in, pre_prev = {}, {}
    G = {"cuda": {}, "ggml": {}}
    SEL = {"cuda": {}, "ggml": {}}
    SEL_LIST = {"cuda": {}, "ggml": {}}
    CAND = {"cuda": {}, "ggml": {}}
    CAND_EXACT = {"cuda": {}, "ggml": {}}
    for eng in ("cuda", "ggml"):
        xs_in[eng] = z[f"{eng}|xs"].astype(np.float64)
        pre_prev[eng] = z[f"{eng}|pre_prev"]
        for key in z.files:
            parts = key.split("|")
            if parts[0] != eng:
                continue
            if parts[1] == "G":
                G[eng].setdefault(int(parts[2]), {})[parts[3]] = z[key]
            elif parts[1] in ("SEL", "SELL") and parts[3] == "p":
                store = SEL if parts[1] == "SEL" else SEL_LIST
                isrc = int(parts[2])
                store[eng][isrc] = _unragged(z[key], z[f"{eng}|{parts[1]}|{isrc}|f"], z[f"{eng}|{parts[1]}|{isrc}|o"])
        cx = _unragged(z[f"{eng}|CX|p"], z[f"{eng}|CX|f"], z[f"{eng}|CX|o"])
        CAND_EXACT[eng] = {p: (bs, block_select(bs, m.cand_topk)) for p, bs in cx.items()}
    for p, mk in _unragged(z["cuda|C|p"], z["cuda|C|f"], z["cuda|C|o"]).items():
        mk = mk.astype(bool)
        CAND["cuda"][p] = (set(int(r) // m.cand_block for r in np.nonzero(mk)[0]), mk)
    CAND["ggml"] = _unragged(z["ggml|C|p"], z["ggml|C|f"], z["ggml|C|o"], lambda a: set(int(v) for v in a))
    return meta, xs_in, pre_prev, G, SEL, SEL_LIST, CAND, CAND_EXACT


def merge_reports(paths):
    reps = [json.load(open(p)) for p in paths]
    out = dict(reps[0])
    out["windows"] = [{"path": p, "report_layers": r.get("report_layers")} for p, r in zip(paths, reps)]
    for key in ("stages", "notes"):
        out[key] = [x for r in reps for x in r.get(key, [])]
    for key in ("format", "selections", "pruning", "routing", "discrimination", "discrimination_moe", "selection_stats", "coverage"):
        out[key] = {}
        for r in reps:
            out[key].update(r.get(key, {}) or {})
    out["report_layers"] = sorted(set(L for r in reps for L in r.get("report_layers", [])))
    out["table"] = build_table(out)
    return out


def segments_from(arg, chunks, T):
    if not arg:
        return []
    if arg == "calls":
        return [(p0, p0 + nt) for p0, nt in chunks]
    return [tuple(int(v) for v in s.split(":")) for s in arg.split(",") if s]


def discrim_positions(T, calls, stride):
    if stride <= 1:
        return np.arange(T)
    keep = set(range(0, T, stride)) | set(int(p) for p in np.nonzero(calls <= 8)[0]) | {T - 1}
    return np.asarray(sorted(keep), np.int64)


# -------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ggml")
    ap.add_argument("--cuda")
    ap.add_argument("--model")
    ap.add_argument("--cuda-probe", help="the direct run's probe output directory (probe.json token ids)")
    ap.add_argument("--ggml-probe", help="the ggml run's probe output directory (probe.json token ids)")
    ap.add_argument("--layers", default=None, help="layers to report (default: all); without --state-in earlier layers are recomputed")
    ap.add_argument("--state-in", help="cross-layer state written by the previous window's --state-out")
    ap.add_argument("--state-out", help="write the cross-layer state after the last layer")
    ap.add_argument("--state-every", help="directory: write the state after every layer (after-L<L>.npz)")
    ap.add_argument("--moe-tokens", default=None, help="a:b[,c:d] token ranges for the MoE oracle (default all)")
    ap.add_argument("--segments", default="calls", help="a:b,c:d or 'calls' (one segment per call; default)")
    ap.add_argument("--rope-table", choices=("exact", "engine"), default="exact")
    ap.add_argument("--discrim-stride", type=int, default=0, help="positions of the discrimination self-test (0: auto)")
    ap.add_argument("--mutate", choices=sorted(MUTATIONS) + ["rope-base"], help="TEST ONLY: perturb the oracle's semantics")
    ap.add_argument("--json")
    ap.add_argument("--gate-from", help="evaluate the oracle gate on an earlier --json report and exit")
    ap.add_argument("--merge", nargs="+", help="join per-window --json reports (then gate; --json writes the merge)")
    ap.add_argument("--factor", type=float, default=1.25)
    ap.add_argument("--slack", type=float, default=1e-6)
    ap.add_argument("--exact-bound", type=float, default=1e-6)
    ap.add_argument("--model-factor", type=float, default=1.25)
    ap.add_argument("--replay-factor", type=float, default=0.4)
    args = ap.parse_args()
    gate_args = (args.factor, args.slack, args.exact_bound, args.model_factor, args.replay_factor)
    if args.gate_from or args.merge:
        report = json.load(open(args.gate_from)) if args.gate_from else merge_reports(args.merge)
        if args.merge:
            report["gate"] = None
        fails, flags, lines = oracle_gate(report, *gate_args)
        print(chr(10).join(lines))
        print(f"flags: {len(flags)}" + ("" if not flags else " (" + ", ".join(flags) + ")"))
        print("gate: PASS" if not fails else "gate: FAIL at " + ", ".join(fails))
        if args.merge and args.json:
            report["gate"] = {"factor": args.factor, "slack": args.slack, "exact_bound": args.exact_bound,
                              "model_factor": args.model_factor, "replay_factor": args.replay_factor, "fails": fails, "flags": flags, "pass": not fails}
            with open(args.json, "w") as f:
                json.dump(report, f, indent=1)
        return 1 if fails else 0
    if not (args.ggml and args.cuda and args.model):
        ap.error("--ggml, --cuda and --model are required")

    g = Gguf(args.model)
    m = Model(g)
    e, nh, hd = m.e, m.nh, m.hd
    n_layer = int(g.kv[f"{g.kv['general.architecture']}.block_count"])
    report_layers = sorted(int(v) for v in args.layers.split(",")) if args.layers else list(range(n_layer))
    state = load_state(args.state_in, m) if args.state_in else None
    if state is not None:
        if state[0]["next_layer"] != report_layers[0]:
            raise SystemExit(f"--state-in ends at layer {state[0]['next_layer'] - 1}; the window must start at {state[0]['next_layer']}")
        layers = list(range(report_layers[0], max(report_layers) + 1))
    else:
        layers = list(range(max(report_layers) + 1))
    TR = {"cuda": Trace(args.cuda, "cuda", e, layers[0]), "ggml": Trace(args.ggml, "ggml", e, layers[0])}
    if TR["cuda"].T != TR["ggml"].T:
        raise SystemExit(f"token counts differ: cuda {TR['cuda'].T}, ggml {TR['ggml'].T}")
    T = TR["cuda"].T
    if state is not None and state[0]["T"] != T:
        raise SystemExit(f"--state-in has {state[0]['T']} tokens, the traces {T}")
    pos = np.arange(T)
    segs = segments_from(args.segments, TR["cuda"].chunks, T)
    calls = {k: v.calls for k, v in TR.items()}
    sem0 = dict(BASE_SEM)
    if args.mutate in MUTATIONS:
        sem0.update(MUTATIONS[args.mutate])
    report = {"model": args.model, "ggml": args.ggml, "cuda": args.cuda, "rope_table": args.rope_table,
              "chunks": {k: v.chunks for k, v in TR.items()}, "stages": [], "notes": [], "format": {},
              "selections": {}, "pruning": {}, "report_layers": report_layers, "segments": [f"{a}:{b}" for a, b in segs],
              "discrimination": {}, "selection_stats": {}, "coverage": {}, "mutate": args.mutate,
              "semantics": sem0, "model_factor": args.model_factor}
    if args.mutate:
        report["notes"].append(f"TEST MUTATION {args.mutate}: the oracle's semantics are perturbed on purpose")

    def add(r):
        if r["layer"] not in report_layers:
            return
        report["stages"].append(r)
        seg = " ".join(f"{k}={v:.3e}" for k, v in r.get("segments", {}).items())
        mdl = f" model {r['model_rel_l2']:.3e}" if "model_rel_l2" in r else ""
        print(f"{r['engine']:5s} L{r['layer']} {r['stage']:32s} local rel-L2 {r['rel_l2']:.3e}{mdl}  tok median "
              f"{r['tok_median']:.3e}  max {r['tok_max']:.3e} @{r['tok_argmax']}  {seg}", flush=True)

    def tr(engine, L, stage, width, dtype=np.float32):
        return TR[engine].tok(L, stage, width, dtype)

    corr = m.corr_dims()
    comp_base = 10000.0 if args.mutate == "rope-base" else m.comp_base
    tab_raw = Rope(T + 2, m.nrot, m.rope_base, mode=args.rope_table)
    tab_comp = Rope(T + 2, m.nrot, comp_base, m.yarn_scale, m.yarn_ext, corr, mode=args.rope_table)
    report["rope"] = {"raw_base": m.rope_base, "comp_base": comp_base, "yarn_freq_scale": m.yarn_scale,
                      "yarn_ext": m.yarn_ext, "corr_dims": list(corr), "n_ctx_orig": m.n_ctx_orig}

    tokens = {}
    for eng, d in (("cuda", args.cuda_probe), ("ggml", args.ggml_probe)):
        if d:
            tokens[eng] = json.load(open(os.path.join(d, "probe.json")))["tokens"]
            if len(tokens[eng]) != T:
                raise SystemExit(f"{eng} probe has {len(tokens[eng])} tokens, the traces {T}")
    engram = Engram(g) if any(L in m.engram_layers for L in layers) else None
    if engram is not None and len(tokens) != 2:
        raise SystemExit("an Engram layer is in the window: --cuda-probe and --ggml-probe are required (S8c finding 6)")

    if state is not None:
        _, xs_in, pre_ffn_prev, G, SEL, SEL_LIST, CAND, CAND_EXACT = state
    else:
        xs_in = {"cuda": tr("cuda", None, "embedding", HC * e).reshape(-1, HC, e),
                 "ggml": tr("ggml", 0, "engram_output", HC * e).reshape(-1, HC, e)}
        report["embedding_cuda_vs_ggml"] = rel(xs_in["cuda"], xs_in["ggml"])
        pre_ffn_prev = {k: None for k in TR}
        G = {"cuda": {}, "ggml": {}}        # kv source -> {"comp": cache rows, "lid": exact from own latent, ...}
        SEL = {"cuda": {}, "ggml": {}}      # index source -> {p: EXACT selection (validated engine set or exact top-k)}
        SEL_LIST = {"cuda": {}, "ggml": {}}  # index source -> {p: the engine's visible ids verbatim}
        CAND = {"cuda": {}, "ggml": {}}     # p -> candidate blocks (cuda: its traced mask; ggml: from its consumers)
        CAND_EXACT = {"cuda": {}, "ggml": {}}
    report["tokens"] = T
    if args.moe_tokens:
        mt = []
        for r in args.moe_tokens.split(","):
            a, b = (int(v) for v in r.split(":"))
            mt.extend(range(a, min(b, T)))
        moe_tokens = sorted(set(mt))
    else:
        moe_tokens = list(range(T))
    report["moe_tokens"] = [moe_tokens[0], moe_tokens[-1], len(moe_tokens)]
    stride = args.discrim_stride or max(1, int(math.ceil(T / 768)))
    dpos = discrim_positions(T, calls["cuda"], stride)
    report["discrimination_positions"] = [int(len(dpos)), int(stride)]

    for L in layers:
        ratio = int(m.ratios[L])
        tabL = tab_comp if ratio else tab_raw
        rep = L in report_layers
        st = {}
        score_data = {}
        disc = {}

        for eng in ("cuda", "ggml"):
            T_ = TR[eng]
            cl_ = calls[eng]
            xs = xs_in[eng]
            s = {}
            # ---- Engram (before attention, in place on the streams): each engine's OWN lookup rows
            if L in m.engram_layers:
                wkv = f"blk.{L}.engram_kv.weight"
                kv = tr(eng, L, "engram_kv", (HC + 1) * e)
                look_exact = engram.lookup(L, tokens[eng])
                if eng == "cuda":
                    lookup = tr("cuda", L, "engram_lookup", g.tensors[wkv][0][0])
                    add(dict(summary("engram_lookup (own tokens)", lookup, look_exact, segs), engine=eng, layer=L,
                             table=g.type_name(engram.layers[L]["table"]),
                             rows_differing=int(np.sum(np.any(lookup != look_exact, axis=1)))))
                else:
                    lookup = look_exact
                    if rep:
                        report["notes"].append(f"L{L} ggml Engram lookup rows recomputed from ggml's own token ids")
                kv_exact = lookup @ m.w(wkv).T
                kv_model = act_model(lookup, m.t(wkv), cl_) @ m.w(wkv).T
                add(dict(summary("engram_kv", kv, kv_exact, segs, model=kv_model), engine=eng, layer=L, weight=m.t(wkv)))
                del kv_model
                qw = m.w(f"blk.{L}.engram_gate.weight")               # [HC, e]
                key = kv[:, :HC * e].reshape(-1, HC, e)
                val = kv[:, HC * e:]
                xn = xs / np.sqrt((xs * xs).mean(axis=2, keepdims=True) + m.eps)
                kn = key / np.sqrt((key * key).mean(axis=2, keepdims=True) + m.eps)
                dot = np.einsum("tse,se,tse->ts", xn, qw, kn) / np.sqrt(e)
                mag = np.sqrt(np.maximum(np.abs(dot), 1e-6))
                gate = sigmoid(np.where(dot >= 0, mag, -mag))
                eo_exact = xs + gate[:, :, None] * val[:, None, :]
                eo = tr(eng, L, "engram_output", HC * e).reshape(-1, HC, e)
                add(dict(summary("engram_output", eo, eo_exact, segs), engine=eng, layer=L))
                upd = eo - xs
                add(dict(summary("engram_update", upd, eo_exact - xs, segs), engine=eng, layer=L))
                xs = eo
            # ---- attention block: hc pre, qr, q, raw K (+ the precision model of each GEMM)
            pre_a, post_a, comb_a = hc_gates(xs, m.w(f"blk.{L}.hc_attn_fn.weight"), m.v(f"blk.{L}.hc_attn_scale.weight"),
                                             m.v(f"blk.{L}.hc_attn_base.weight"), m.iters, m.hceps, m.eps)
            cw = np.full((T, HC), 0.25) if L == 0 else pre_ffn_prev[eng]
            ai_exact = rms_norm(collapse(xs, cw), m.v(f"blk.{L}.attn_norm.weight"), m.eps)
            ai = tr(eng, L, "attn_input", e)
            add(dict(summary("attn_input", ai, ai_exact, segs), engine=eng, layer=L))
            nqa, nqb, nkv = f"blk.{L}.attn_q_a.weight", f"blk.{L}.attn_q_b.weight", f"blk.{L}.attn_kv.weight"
            qa_norm = m.v(f"blk.{L}.attn_q_a_norm.weight")
            qr_exact = rms_norm(ai @ m.w(nqa).T, qa_norm, m.eps)
            qr_model = rms_norm(act_model(ai, m.t(nqa), cl_) @ m.w(nqa).T, qa_norm, m.eps)
            qr = tr(eng, L, "qr", m.qlora)
            if qr is not None:
                add(dict(summary("qr", qr, qr_exact, segs, model=qr_model), engine=eng, layer=L, weight=m.t(nqa)))
            q = tr(eng, L, "q" if eng == "cuda" else "query", nh * hd).reshape(-1, nh, hd)
            content = hd - m.nrot
            if qr is not None:
                q_exact = rope((qr @ m.w(nqb).T).reshape(-1, nh, hd), tabL.tab, pos, m.nrot)
                q_model = rope((act_model(qr, m.t(nqb), cl_) @ m.w(nqb).T).reshape(-1, nh, hd), tabL.tab, pos, m.nrot)
                add(dict(summary("q (own qr)", q[:, :, :content], q_exact[:, :, :content], segs,
                                 model=q_model[:, :, :content]), engine=eng, layer=L, weight=m.t(nqb)))
                add(dict(summary("q rope tail", q[:, :, content:], q_exact[:, :, content:], segs,
                                 model=q_model[:, :, content:]), engine=eng, layer=L))
                del q_exact, q_model
            q_chain = rope((qr_exact @ m.w(nqb).T).reshape(-1, nh, hd), tabL.tab, pos, m.nrot)
            q_chain_m = rope((act_model(qr_model, m.t(nqb), cl_) @ m.w(nqb).T).reshape(-1, nh, hd), tabL.tab, pos, m.nrot)
            add(dict(summary("q (own attn_in)", q[:, :, :content], q_chain[:, :, :content], segs,
                             model=q_chain_m[:, :, :content]), engine=eng, layer=L))
            del q_chain, q_chain_m, qr_model
            kv_norm = m.v(f"blk.{L}.attn_kv_a_norm.weight")
            kv_pre = rope(rms_norm(ai @ m.w(nkv).T, kv_norm, m.eps)[:, None, :], tabL.tab, pos, m.nrot)[:, 0, :]
            k_exact = dsv41_quant(kv_pre, 0).astype(np.float64)
            kv_pre_m = rope(rms_norm(act_model(ai, m.t(nkv), cl_) @ m.w(nkv).T, kv_norm, m.eps)[:, None, :], tabL.tab, pos, m.nrot)[:, 0, :]
            k_model = dsv41_quant(kv_pre_m, 0).astype(np.float64)
            k = tr(eng, L, "raw_k" if eng == "cuda" else "raw_kv", hd)
            nflip, nmulti = fp8_flip_stats(k, k_exact, kv_pre, kv_pre_m)
            fl_seg = {f"{a}:{b}": int(nflip[a:b].sum()) for a, b in segs if a < min(b, T)}
            ms_seg = {f"{a}:{b}": int(nmulti[a:b].sum()) for a, b in segs if a < min(b, T)}
            add(dict(summary("raw_k", k, k_exact, segs, model=k_model), engine=eng, layer=L,
                     fp8_bins_differing=float(np.mean(k != k_exact)), flips_by_segment=fl_seg,
                     beyond_noise_by_segment=ms_seg, vs_unquantized=rel(k, kv_pre), weight=m.t(nkv),
                     beyond_noise=int(nmulti.sum())))
            del kv_pre_m
            del k_model
            raw_cache = k.astype(np.float16).astype(np.float64)     # the F16 raw ring the attention reads

            comp, rows_exact, rows_list, comp_for = None, None, None, None
            if ratio:
                src, isrc = m.kv_source_of(L), m.index_source_of(L)
                # ---- KV source: compressor, indexer K, the two committed caches
                if L == src:
                    nb = T // ratio
                    start = np.arange(nb) * ratio
                    bcalls = T_.block_calls(ratio, nb)
                    nckv, ncg, nik = (f"blk.{L}.attn_compressor_kv.weight", f"blk.{L}.attn_compressor_gate.weight",
                                      f"blk.{L}.indexer.attn_k.weight")
                    wkv = m.w(nckv)
                    wg = m.w(ncg) if ratio > 1 else None
                    cn = m.v(f"blk.{L}.attn_compressor_norm.weight")

                    def pool(kvp, sc):
                        if ratio == 1:
                            v = kvp[:nb]
                        else:
                            k3 = kvp[:nb * ratio].reshape(nb, ratio, hd)
                            s3 = sc[:nb * ratio].reshape(nb, ratio, hd)
                            w3 = np.exp(s3 - s3.max(axis=1, keepdims=True))
                            v = (w3 * k3).sum(axis=1) / w3.sum(axis=1)
                        return rms_norm(v, cn, m.eps)

                    kvp_exact = ai @ wkv.T
                    sc_exact = ai @ wg.T if wg is not None else None
                    lat_exact = pool(kvp_exact, sc_exact)
                    ai_m = act_model(ai, m.t(nckv), cl_)
                    lat_model = pool(ai_m @ wkv.T, (act_model(ai, m.t(ncg), cl_) @ wg.T) if wg is not None else None)
                    lat = block_rows(T_, L, "latent", hd, ratio, nb)
                    add(dict(summary("latent (own attn_input)", lat, lat_exact, segs, positions=start, model=lat_model),
                             engine=eng, layer=L, weight=m.t(nckv), rows=nb))
                    del ai_m, lat_model
                    wik = m.w(nik)
                    ikn = m.v(f"blk.{L}.indexer.k_norm.weight")
                    lid_pre_exact = rms_norm(lat @ wik.T, ikn, m.eps)
                    lid_exact = cache_row(rope(lid_pre_exact[:, None, :], tab_comp.tab, start, m.nrot)[:, 0, :], 1)
                    lid_model = cache_row(rope(rms_norm(act_model(lat, m.t(nik), bcalls) @ wik.T, ikn, m.eps)[:, None, :],
                                               tab_comp.tab, start, m.nrot)[:, 0, :], 1)
                    comp_pre = rope(lat[:, None, :], tab_comp.tab, start, m.nrot)[:, 0, :]
                    comp_end = cache_row(rope(lat[:, None, :], tab_comp.tab, start + ratio - 1, m.nrot)[:, 0, :], 2)
                    if eng == "cuda":
                        ckv = tr("cuda", L, "comp_kv", hd)
                        add(dict(summary("comp_kv (own attn_input)", ckv, kvp_exact, segs), engine=eng, layer=L,
                                 weight=m.t(nckv)))
                        csc = None
                        if ratio > 1:
                            csc = tr("cuda", L, "comp_score", hd)
                            add(dict(summary("comp_score (own attn_input)", csc, sc_exact, segs), engine=eng, layer=L))
                        add(dict(summary("latent (own comp_kv)", lat, pool(ckv, csc), segs, positions=start),
                                 engine=eng, layer=L))
                        lp = block_rows(T_, L, "lid_pre", m.idx_hd, ratio, nb)
                        add(dict(summary("lid_pre (own latent)", lp, lid_pre_exact, segs, positions=start), engine=eng,
                                 layer=L, weight=m.t(nik)))
                        lk = block_rows(T_, L, "lid_k", m.idx_hd, ratio, nb, np.float16)
                        ck = block_rows(T_, L, "comp_k", hd, ratio, nb, np.float16)
                        # format stages: tie tolerance = F32 rounding of the pre-quantization chain plus, on the rotated
                        # dims, the float32 RoPE-table angle error times the pair's radius
                        for nm, rows_eng, pre_unrot, mode in (("lid_k (own lid_pre)", lk, lp, 1),
                                                              ("comp_k (own latent)", ck, lat, 2)):
                            pre = rope(pre_unrot[:, None, :], tab_comp.tab, start, m.nrot)[:, 0, :]
                            tol = 8 * U * np.abs(pre) + 1e-30
                            dimn = pre.shape[1]
                            r0 = dimn - m.nrot
                            rad = np.sqrt(pre_unrot[:, r0::2] ** 2 + pre_unrot[:, r0 + 1::2] ** 2)
                            ang = tab_comp.angle_err(start) * rad + 4 * U * rad
                            tol[:, r0::2] += ang
                            tol[:, r0 + 1::2] += ang
                            fc, _ = format_check(rows_eng, pre, mode, tol)
                            fc["bound"] = "8u|x| + (float32 RoPE angle bound + 4u) x pair radius on rotated dims"
                            if rep:
                                report["format"][f"{L}|{nm}"] = fc
                            print(f"cuda  L{L} {nm:32s} {fc['differing']}/{fc['elements']} differ, {fc['tie_explained']} ties, "
                                  f"{fc['unexplained']} unexplained, rel-L2 {fc['rel_l2']:.3e}", flush=True)
                            add(dict(summary(nm, rows_eng, cache_row(pre, mode), segs, positions=start), engine=eng, layer=L,
                                     differing=fc["differing"], unexplained=fc["unexplained"]))
                        fl_l = float(np.mean(lk != lid_exact))
                        add(dict(summary("lid_k (own latent)", lk, lid_exact, segs, positions=start), engine=eng, layer=L,
                                 mxfp4_bins_differing=fl_l))
                        comp_main = ck if not sem0["rot_end"] else comp_end
                        G[eng][L] = {"comp": comp_main, "comp_start": ck, "comp_end": comp_end, "lid": lid_exact,
                                     "lid_model": lid_model, "lid_traced": lk}
                    else:
                        comp_start = cache_row(comp_pre, 2)
                        G[eng][L] = {"comp": comp_start if not sem0["rot_end"] else comp_end, "comp_start": comp_start,
                                     "comp_end": comp_end, "lid": lid_exact, "lid_model": lid_model}
                        if rep:
                            report["notes"].append(f"L{L} ggml caches rebuilt from its traced latent by the format replay")
                # ---- index source: queries, weights, scores, top-k (and the candidate blocks at the candidate source)
                if L == isrc:
                    IH, ID = m.idx_nh, m.idx_hd
                    nvis = (pos + 1) // ratio
                    niq, nip = f"blk.{L}.indexer.attn_q_b.weight", f"blk.{L}.indexer.proj.weight"
                    iq_rot = rope((qr @ m.w(niq).T).reshape(T, IH, ID), tab_comp.tab, pos, m.nrot)
                    iq_exact = dsv41_quant(iq_rot, 1).astype(np.float64)
                    iw_exact = (ai @ m.w(nip).T) / math.sqrt(ID * IH)
                    iq_model = dsv41_quant(rope((act_model(qr, m.t(niq), cl_) @ m.w(niq).T).reshape(T, IH, ID),
                                                tab_comp.tab, pos, m.nrot), 1).astype(np.float64)
                    iw_model = (act_model(ai, m.t(nip), cl_) @ m.w(nip).T) / math.sqrt(ID * IH)
                    sc_eng = per_position(T_, L, "index_scores")
                    sc_eng = {p: v[:(p + 1) // ratio].astype(np.float64) for p, v in sc_eng.items()}
                    keep = [None] * T
                    if L > m.cand_source >= 0:
                        for p, v in sc_eng.items():
                            keep[p] = np.isfinite(v)
                    lid = G[eng][src]["lid"]
                    comp_scores = index_scores(iq_exact, iw_exact, lid, nvis, keep)
                    model_scores = index_scores(iq_model, iw_model, G[eng][src]["lid_model"], nvis, keep)
                    del iq_model, iw_model
                    # the engine's id lists, verbatim (S8c finding 1)
                    lists, problems = {}, {}
                    if eng == "cuda":
                        for p0, nt, a in T_.per_chunk(L, "index_top", np.int32):
                            a = a.reshape(nt, -1)
                            cnt = np.fromfile(T_.files[(p0, L, "index_count")], np.int32)
                            for i in range(nt):
                                lists[p0 + i] = a[i, :cnt[i]].astype(np.int64)
                    else:
                        for p0, nt, a in T_.per_chunk(L, "index_top", np.int32):
                            a = a.reshape(nt, -1)
                            for i in range(nt):
                                lists[p0 + i] = a[i].astype(np.int64)
                    sel_exact, sel_list = {}, {}
                    for p, ids in lists.items():
                        sc = sc_eng.get(p)
                        prob, vis, uniq = selection_integrity(ids, (p + 1) // ratio, m.idx_topk, sc, eng)
                        if prob:
                            problems[p] = prob
                        sel_list[p] = vis
                        sel_exact[p] = exact_selection(sc, m.idx_topk, set(int(r) for r in uniq) if not prob else None) \
                            if sc is not None else uniq
                    SEL[eng][L] = sel_exact
                    SEL_LIST[eng][L] = sel_list
                    score_data[eng] = {"traced": sc_eng, "composite": comp_scores, "model": model_scores, "lists": lists,
                                       "problems": problems, "sel_exact": sel_exact, "sel_list": sel_list, "keep": keep}
                    if eng == "cuda":
                        iq_t = tr("cuda", L, "index_q", IH * ID).reshape(T, IH, ID)
                        iw_t = tr("cuda", L, "index_w", IH)
                        add(dict(summary("index_q (own qr)", iq_t, iq_exact, segs), engine=eng, layer=L,
                                 weight=m.t(niq), mxfp4_bins_differing=float(np.mean(iq_t != iq_exact))))
                        add(dict(summary("index_w (own attn_input)", iw_t, iw_exact, segs), engine=eng, layer=L,
                                 weight=m.t(nip)))
                        f32_scores, f32_bound = index_scores(iq_t, iw_t, G[eng][src]["lid_traced"], nvis, keep, bound=True)
                        score_data[eng].update(f32=f32_scores, f32_bound=f32_bound)
                        ps = sorted(sc_eng)
                        fin = [np.isfinite(sc_eng[p]) for p in ps]
                        add(dict(ragged_summary("index_scores (own iq/ik/iw)", [sc_eng[p][f] for p, f in zip(ps, fin)],
                                                [f32_scores[p][f] for p, f in zip(ps, fin)], ps, segs),
                                 engine=eng, layer=L))
                        del iq_t
                    if L == m.cand_source and eng == "cuda":
                        for p, mk in per_position(T_, L, "cand_mask", np.uint8).items():
                            nv = (p + 1) // ratio
                            rows_kept = np.nonzero(mk[:nv])[0]
                            CAND["cuda"][p] = (set(int(r) // m.cand_block for r in rows_kept), mk[:nv].astype(bool))
                # ---- the attention reads the group's compressed cache with its index source's selection
                comp = G[eng][src]["comp"]
                rows_exact = SEL[eng].get(isrc, {})
                rows_list = SEL_LIST[eng].get(isrc, {})
                gsrc = G[eng][src]

                def comp_for(rot_end, gsrc=gsrc):
                    return gsrc["comp_end"] if rot_end else gsrc["comp_start"]
            exact_of = (lambda p, d=rows_exact: d.get(p)) if ratio else None
            core_exact = attention_exact(m, L, q, raw_cache, pos, comp, exact_of, ratio, sem0)
            if eng == "cuda":
                core = tr("cuda", L, "attn_core", nh * hd)
                if core is not None:
                    core = core.reshape(-1, nh, hd)
                    add(dict(summary("attn_core", core, core_exact, segs), engine=eng, layer=L,
                             selection="exact top-k of the engine's own traced scores" if ratio else "-"))
                    if ratio:
                        differ = [p for p in rows_list if not np.array_equal(np.sort(rows_list[p]), rows_exact.get(p))]
                        if differ:
                            list_of = (lambda p, d=rows_list: d.get(p))
                            core_list = attention_exact(m, L, q, raw_cache, pos, comp, list_of, ratio, sem0)
                        else:
                            core_list = core_exact
                        add(dict(summary("attn_core (own list)", core, core_list, segs), engine=eng, layer=L,
                                 positions_where_list_differs_from_exact=len(differ)))
                        del core_list
                    s["ao_from_core"] = out_projection(m, L, core, pos, tabL.tab)
                    del core
            ao_exact = out_projection(m, L, core_exact, pos, tabL.tab)
            ao_model = out_projection(m, L, core_exact, pos, tabL.tab, calls=cl_)
            ao = tr(eng, L, "attn_out" if eng == "cuda" else "attn_output", e)
            add(dict(summary("attn_out", ao, ao_exact, segs, model=ao_model), engine=eng, layer=L,
                     weights=[m.t(f"blk.{L}.attn_output_a.weight"), m.t(f"blk.{L}.attn_output_b.weight")]))
            del ao_model
            if "ao_from_core" in s:
                add(dict(summary("out_proj (own core)", ao, s.pop("ao_from_core"), segs), engine=eng, layer=L))
            if rep:
                disc[eng] = discriminate(m, L, ratio, q[dpos], dpos, raw_cache, comp_for, exact_of, ao[dpos],
                                         lambda c, p, L=L, tab=tabL.tab: out_projection(m, L, c, p, tab), sem0)
                for v in disc[eng]:
                    print(f"  discrim {eng} L{L} {v['variant']}: {v['measured_ratio']:.3f}x (expected {v['expected_ratio']:.3f}, "
                          f"{'informative' if v['informative'] else 'not informative'}, {'ok' if v['ok'] else 'FAIL'})", flush=True)
            del core_exact, q
            # ---- FFN block
            xm = hc_post(xs, ao, post_a, comb_a)
            pre_f, post_f, comb_f = hc_gates(xm, m.w(f"blk.{L}.hc_ffn_fn.weight"), m.v(f"blk.{L}.hc_ffn_scale.weight"),
                                             m.v(f"blk.{L}.hc_ffn_base.weight"), m.iters, m.hceps, m.eps)
            fi_exact = rms_norm(collapse(xm, pre_a), m.v(f"blk.{L}.ffn_norm.weight"), m.eps)
            fi = tr(eng, L, "ffn_input", e) if eng == "cuda" else None
            if fi is not None:
                add(dict(summary("ffn_input", fi, fi_exact, segs), engine=eng, layer=L))
            else:
                fi = fi_exact     # ggml does not trace it: its own input, derived in float64 from its traces
                if rep:
                    report["notes"].append(f"L{L} ggml ffn_input derived from its own streams and attn_output (float64)")
            Wr = m.w(f"blk.{L}.ffn_gate_inp.weight")
            logits_exact = fi @ Wr.T
            raw, score, order, margin, bias = route(m, L, logits_exact)
            if eng == "cuda":
                rl = tr("cuda", L, "router", m.nex)
                add(dict(summary("router", rl, logits_exact, segs), engine=eng, layer=L))
                sel = tr("cuda", L, "experts", m.used, np.int32)
                sel_exact = np.sort(order[:, :m.used], axis=1)
                agree = (np.sort(sel, axis=1) == sel_exact).all(axis=1)
                selw = tr("cuda", L, "selw", m.used)
                raw_c, _, _, margin_c, _ = route(m, L, rl)
                wts = weights_for(m, raw_c, sel)
                if selw is not None:
                    add(dict(summary("selw", selw, wts, segs), engine=eng, layer=L))
                s.update(sel=sel, wts=wts, agree=agree, margin=margin, margin_own=margin_c)
                if rep:
                    s["route_checks"] = routing_checks(m, L, sel, rl, fi, Wr, bias)
            else:
                sel = order[:, :m.used]
                wts = weights_for(m, raw, sel)
                s.update(sel=sel, wts=wts, margin=margin)
            if rep:
                sh_exact, sh_model, sh_noclamp = shared_exact(m, L, fi[moe_tokens], cl_[moe_tokens])
                if eng == "cuda":
                    sho = tr("cuda", L, "shexp_out", e)
                    if sho is not None:
                        add(dict(summary("shexp_out", sho[moe_tokens], sh_exact, []), engine=eng, layer=L,
                                 weight=m.t(f"blk.{L}.ffn_gate_shexp.weight")))
                s["sh_exact"], s["sh_model"], s["sh_noclamp"] = sh_exact, sh_model, sh_noclamp
            s.update(fi=fi, raw=raw, order=order, xm=xm, pre_f=pre_f, post_f=post_f, comb_f=comb_f,
                     fo=tr(eng, L, "ffn_out" if eng == "cuda" else "ffn_output", e), calls=cl_)
            st[eng] = s

        # ---- the compressed path's cross-engine reductions: score stages on positions both engines traced, selections
        if ratio and L == m.index_source_of(L):
            sd_c, sd_g = score_data["cuda"], score_data["ggml"]
            common = sorted(set(sd_c["traced"]) & set(sd_g["traced"]))
            for eng, sd in (("cuda", sd_c), ("ggml", sd_g)):
                ps = common if common else sorted(sd["traced"])
                fin = [np.isfinite(sd["traced"][p]) for p in ps]
                if ps:
                    add(dict(ragged_summary("index_scores (own qr/x/latent)", [sd["traced"][p][f] for p, f in zip(ps, fin)],
                                            [sd["composite"][p][f] for p, f in zip(ps, fin)], ps, segs,
                                            model=[sd["model"][p][f] for p, f in zip(ps, fin)]),
                             engine=eng, layer=L, scope="positions both engines trace" if common else "this engine only"))
            if not common:
                report["notes"].append(f"L{L}: ggml does not rank this layer (every visible row is selected); "
                                       "index_scores (own qr/x/latent) is the direct engine's alone")
            stats = {}
            for eng, sd in (("cuda", sd_c), ("ggml", sd_g)):
                ranked = sorted(p for p in sd["lists"] if (p + 1) // ratio > m.idx_topk and p in sd["traced"])
                stats[eng] = {"positions_with_list": len(sd["lists"]), "positions_ranking_over_topk": len(ranked),
                              "first_over_topk": ranked[0] if ranked else None,
                              "max_visible_rows": int(max(((p + 1) // ratio for p in sd["lists"]), default=0)),
                              "visible_rows_at_last_ranked": int((ranked[-1] + 1) // ratio) if ranked else 0,
                              "finite_rows_at_last_ranked": int(np.isfinite(sd["traced"][ranked[-1]]).sum()) if ranked else 0}
            if rep:
                report["selection_stats"][str(L)] = stats
            # selections: (i) list integrity (count, duplicates, visibility, -inf), (a) the set on the engine's own traced
            # scores, exact ties only; (b) direct only, on the exact scores from its own traced q/k/w within their F32
            # bound; (c) info: against the composite exact scores
            sels = {"gated": [], "info": []}
            for eng, sd in (("cuda", sd_c), ("ggml", sd_g)):
                viol_a, viol_b, flips_b, flips_c, pos_c, checked = [], [], 0, 0, 0, 0
                integ = sorted(sd["problems"].items())
                for p, ids in sd["lists"].items():
                    if p not in sd["traced"] or p in sd["problems"]:
                        continue
                    sc = sd["traced"][p]
                    chosen = set(int(r) for r in np.unique(sd["sel_list"][p]))
                    checked += 1
                    v = consistent(sc, chosen)
                    if v > 0:
                        viol_a.append((p, v))
                    if eng == "cuda":
                        fs, fb = sd["f32"][p], sd["f32_bound"][p]
                        vb = consistent(fs, chosen, fb)
                        ex, _ = topk_order(fs, m.idx_topk)
                        flips_b += len(chosen - ex)
                        if vb > 0:
                            viol_b.append((p, vb))
                    ex_c, _ = topk_order(sd["composite"][p], m.idx_topk)
                    nfl = len(chosen - ex_c)
                    flips_c += nfl
                    pos_c += nfl > 0
                gate_list = sels["gated"] if eng == "cuda" else sels["info"]
                gate_list.append({"name": f"{eng} index top-{m.idx_topk} list integrity (count, unique, visible, not pruned)",
                                  "checked": len(sd["lists"]), "violations": len(integ), "bound": "exact",
                                  "detail": f"first {integ[:3]}"})
                gate_list.append({"name": f"{eng} index top-{m.idx_topk} = exact top-k of its own traced scores",
                                  "checked": checked, "violations": len(viol_a), "bound": "exact ties only",
                                  "detail": f"first {viol_a[:3]}"})
                if eng == "cuda":
                    sels["gated"].append({"name": f"cuda index top-{m.idx_topk} = exact top-k of the scores from its own "
                                          "index_q/index_w/lid_k", "checked": checked, "violations": len(viol_b),
                                          "bound": "F32 accumulation bound per row", "detail":
                                          f"rows differing from the exact pick {flips_b} (all within the bound unless "
                                          f"violations > 0); first {viol_b[:3]}"})
                sels["info"].append({"name": f"{eng} index top-{m.idx_topk} vs exact top-k of the composite scores "
                                     "(own qr/x/latent)", "checked": checked, "positions_with_flips": pos_c,
                                     "flipped_rows": flips_c})
            # candidate blocks (layer c) and the kept rows of the consumers (layers > c)
            if L == m.cand_source:
                for eng, sd in (("cuda", sd_c), ("ggml", sd_g)):
                    for p, sc in sd["traced"].items():
                        bs, _ = block_scores(sc, p, m.cand_block, ratio)
                        CAND_EXACT[eng][p] = (bs, block_select(bs, m.cand_topk))
                viol_a, viol_b, checked, nflip_b = [], [], 0, 0
                for p, (blocks, mk) in sorted(CAND["cuda"].items()):
                    if p not in sd_c["traced"]:
                        continue
                    checked += 1
                    bs, _ = CAND_EXACT["cuda"][p]
                    nv = len(mk)
                    full = np.zeros(nv, bool)
                    for b in blocks:
                        full[b * m.cand_block:min(nv, (b + 1) * m.cand_block)] = True
                    if not np.array_equal(full, mk):
                        viol_a.append((p, "mask is not whole blocks"))
                        continue
                    k_eff = min(m.cand_topk, int(np.sum(bs > -np.inf)))
                    if len(blocks) != k_eff or any(bs[b] == -np.inf for b in blocks) or consistent(bs, blocks) > 0:
                        viol_a.append((p, consistent(bs, blocks)))
                    fs, fb = sd_c["f32"][p], sd_c["f32_bound"][p]
                    bsf, bbf = block_scores(fs, p, m.cand_block, ratio, fb)
                    nflip_b += len(blocks - block_select(bsf, m.cand_topk))
                    vb = consistent(bsf, blocks, bbf)
                    if vb > 0:
                        viol_b.append((p, vb))
                sels["gated"].append({"name": f"cuda candidate blocks (cand_mask) = exact top-{m.cand_topk} blocks of its "
                                      "own traced scores", "checked": checked, "violations": len(viol_a),
                                      "bound": "exact ties only", "detail": f"first {viol_a[:3]}"})
                sels["gated"].append({"name": "cuda candidate blocks = exact blocks of the scores from its own "
                                      "index_q/index_w/lid_k", "checked": checked, "violations": len(viol_b),
                                      "bound": "F32 accumulation bound (block max of the row bounds)",
                                      "detail": f"blocks differing from the exact pick {nflip_b}; first {viol_b[:3]}"})
            if L > m.cand_source >= 0:
                for eng, sd in (("cuda", sd_c), ("ggml", sd_g)):
                    bad, checked, pruned, dropped = [], 0, [], {}
                    for p, sc in sorted(sd["traced"].items()):
                        kept = np.isfinite(sc)
                        nv = len(sc)
                        kb = set(int(r) // m.cand_block for r in np.nonzero(kept)[0])
                        if not kept.all():
                            pruned.append((p, len(kb)))
                            nbv = (nv + m.cand_block - 1) // m.cand_block
                            dropped[p] = 1.0 - len(kb) / nbv
                        whole = np.zeros(nv, bool)
                        for b in kb:
                            whole[b * m.cand_block:min(nv, (b + 1) * m.cand_block)] = True
                        if p in CAND_EXACT[eng]:
                            bs = CAND_EXACT[eng][p][0]
                            k_eff = min(m.cand_topk, int(np.sum(bs > -np.inf)))
                            ok_sel = len(kb) == k_eff and consistent(bs, kb) <= 0
                        elif nv <= m.cand_topk * m.cand_block:
                            ok_sel = bool(kept.all())
                        else:
                            continue
                        checked += 1
                        if eng == "cuda" and p in CAND["cuda"] and not np.array_equal(kept, CAND["cuda"][p][1]):
                            bad.append((p, "differs from its cand_mask"))
                        elif not np.array_equal(kept, whole):
                            bad.append((p, "kept rows are not whole blocks"))
                        elif not ok_sel:
                            bad.append((p, "kept blocks are not a top-k of its layer-c block scores"))
                        if eng == "ggml" and not kept.all():
                            CAND["ggml"][p] = kb
                    lst = sels["gated"] if eng == "cuda" else sels["info"]
                    lst.append({"name": f"{eng} kept rows = its own candidate blocks of layer {m.cand_source} (exact on its "
                                "own traced scores" + (", and its cand_mask)" if eng == "cuda" else ")"),
                                "checked": checked, "violations": len(bad), "bound": "exact ties only",
                                "detail": f"first {bad[:3]}"})
                    first = min((p for p, _ in pruned), default=None)
                    key = f"{eng} L{L}"
                    late = [p for p in sd["traced"] if (p + 1) // ratio > m.cand_topk * m.cand_block]
                    dv = np.array([dropped[p] for p in sorted(dropped)]) if dropped else np.zeros(0)
                    last = max(dropped) if dropped else None
                    if rep:
                        report["pruning"][key] = {
                            "engaged": bool(pruned), "pairs_pruned": len(pruned), "first_pruned": first,
                            "expected_first": (m.cand_topk * m.cand_block + 1) * ratio - 1,
                            "max_blocks": max((b for _, b in pruned), default=0), "budget": m.cand_topk,
                            "all_late_pruned": all(not np.isfinite(sd["traced"][p]).all() for p in late) if late else None,
                            "late_positions": len(late),
                            "dropped_median": float(np.median(dv)) if len(dv) else None,
                            "dropped_max": float(dv.max()) if len(dv) else None,
                            "dropped_last": None if last is None else round(dropped[last], 4), "last_pruned": last,
                            "blocks_visible_at_last": None if last is None else (len(sd["traced"][last]) + m.cand_block - 1) // m.cand_block,
                            "positions_dropping_over_half": int((dv > 0.5).sum())}
            if rep:
                report["selections"][str(L)] = sels
                for c in sels["gated"] + sels["info"]:
                    print(f"  sel L{L} {c['name']}: {c.get('checked')} checked, "
                          f"{c.get('violations', c.get('flipped_rows'))} {'violations' if 'violations' in c else 'flipped rows'}"
                          f" {c.get('detail', '')}", flush=True)
        if rep and disc:
            report["discrimination"][str(L)] = disc

        # ---- routed experts, expert-major over both engines' selections plus each token's next two candidates
        if rep:
            jobs = {}
            for eng in ("cuda", "ggml"):
                s = st[eng]
                for i, t in enumerate(moe_tokens):
                    need = set(int(x) for x in s["sel"][t]) | set(int(x) for x in s["order"][t, :m.used + 2])
                    for ex in need:
                        jobs.setdefault(ex, []).append(((eng, t), s["fi"][t], int(s["calls"][t])))
            print(f"  L{L}: {len(jobs)} experts for {len(moe_tokens)} tokens", flush=True)
            y = expert_outputs(m, L, jobs)

            def combo(eng, t, i, sel_t, w_t, which=0, shared=True, sh_which=None):
                shw = which if sh_which is None else sh_which
                sh_src = st[eng]["sh_exact"] if shw == 0 else st[eng]["sh_model"] if shw == 1 else st[eng]["sh_noclamp"]
                out = sh_src[i].copy() if shared else np.zeros(e)
                for ex, wt in zip(sel_t, w_t):
                    out += wt * y[((eng, t), int(ex))][which]
                return out

            moe_types = [m.t(f"blk.{L}.ffn_{k}.weight") for k in ("gate_exps", "up_exps", "down_exps", "gate_shexp",
                                                                   "up_shexp", "down_shexp")]
            dmoe = {}

            flips = []
            for eng in ("cuda", "ggml"):
                s = st[eng]
                fo_exact = np.stack([combo(eng, t, i, s["sel"][t], s["wts"][t]) for i, t in enumerate(moe_tokens)])
                fo_model = np.stack([combo(eng, t, i, s["sel"][t], s["wts"][t], 1) for i, t in enumerate(moe_tokens)])
                fo = s["fo"]
                r = summary("ffn_out (own sel)", fo[moe_tokens], fo_exact, [], model=fo_model)
                r["tok_argmax"] = int(moe_tokens[r["tok_argmax"]])
                loc = tok_rel(fo[moe_tokens], fo_exact)
                r["per_token"] = {int(t): float(v) for t, v in zip(moe_tokens, loc)}
                mt_ = np.asarray(moe_tokens)
                r["segments"] = _segments(fo[moe_tokens], fo_exact, segs, mt_)
                r["model_segments"] = _segments(fo_model, fo_exact, segs, mt_)
                r["replay_rel_l2"] = rel(fo[moe_tokens], fo_model)
                r["weights"] = [m.t(f"blk.{L}.ffn_gate_exps.weight"), m.t(f"blk.{L}.ffn_down_exps.weight")]
                r["moe_types"] = moe_types
                add(dict(r, engine=eng, layer=L))
                # MoE semantic discrimination (S8c finding 2, the MoE counterpart of the attention self-test): the
                # engine's ffn_out must fit the oracle's MoE semantics better than each informative perturbation
                fo_m = fo[moe_tokens]
                raw_t = s["raw"][moe_tokens]
                sel_t = np.asarray([s["sel"][t] for t in moe_tokens])
                wn = np.take_along_axis(raw_t, sel_t, axis=1)
                w_tog = (wn if m.wnorm else wn / wn.sum(axis=1, keepdims=True)) * m.wscale
                low = np.argmin(wn, axis=1)
                variants = {
                    "routing weights normalization toggled":
                        [combo(eng, t, i, s["sel"][t], w_tog[i]) for i, t in enumerate(moe_tokens)],
                    "shared expert omitted":
                        [combo(eng, t, i, s["sel"][t], s["wts"][t], shared=False) for i, t in enumerate(moe_tokens)],
                    f"top-{m.used - 1} (weakest pick dropped)":
                        [combo(eng, t, i, np.delete(np.asarray(s["sel"][t]), low[i]), np.delete(np.asarray(s["wts"][t]), low[i]))
                         for i, t in enumerate(moe_tokens)],
                    "routed swiglu clamp off":
                        [combo(eng, t, i, s["sel"][t], s["wts"][t], which=2, sh_which=0) for i, t in enumerate(moe_tokens)],
                    "shared swiglu clamp off":
                        [combo(eng, t, i, s["sel"][t], s["wts"][t], which=0, sh_which=2) for i, t in enumerate(moe_tokens)]}
                if m.wscale != 1.0:
                    variants["routed scale 1"] = [combo(eng, t, i, s["sel"][t], np.asarray(s["wts"][t]) / m.wscale)
                                                  for i, t in enumerate(moe_tokens)]
                dmoe[eng] = [discrimination_verdict(k, fo_m, fo_exact, np.stack(v)) for k, v in variants.items()]
                for v in dmoe[eng]:
                    print(f"  discrim-moe {eng} L{L} {v['variant']}: {v['measured_ratio']:.3f}x (expected {v['expected_ratio']:.3f}, "
                          f"{'informative' if v['informative'] else 'not informative'}, {'ok' if v['ok'] else 'FAIL'})", flush=True)
                s["loc"] = loc
                med = float(np.median(loc))
                for i, t in enumerate(moe_tokens):
                    if loc[i] < max(4 * med, 1e-2):
                        continue
                    cur = [int(x) for x in s["sel"][t]]
                    best = None
                    for drop in cur:
                        for add_ in (int(x) for x in s["order"][t, :m.used + 2]):
                            if add_ in cur:
                                continue
                            alt = np.array([[x for x in cur if x != drop] + [add_]])
                            w_alt = weights_for(m, s["raw"][[t]], alt)[0]
                            err = float(tok_rel(fo[[t]], combo(eng, t, i, alt[0], w_alt)[None, :])[0])
                            if best is None or err < best[0]:
                                best = (err, drop, add_)
                    if best is None:
                        continue
                    flips.append({"engine": eng, "position": int(t), "local_err": float(loc[i]), "median_local": med,
                                  "best_swap_err": best[0], "dropped": best[1], "added": best[2],
                                  "exact_margin": float(s["margin"][t]),
                                  "explained_by_flip": bool(best[0] < 0.25 * loc[i])})

        for eng in ("cuda", "ggml"):
            s = st[eng]
            hid_exact = hc_post(s["xm"], s["fo"], s["post_f"], s["comb_f"])
            hid = tr(eng, L, "hidden", HC * e) if eng == "cuda" else tr(eng, L, "output", HC * e)
            if hid is None and eng == "ggml":
                hid = tr("ggml", L + 1, "engram_output", HC * e)
            hid = hid.reshape(-1, HC, e)
            add(dict(summary("hidden", hid, hid_exact, segs), engine=eng, layer=L))
            s["hid"] = hid

        if rep:
            c, gg = st["cuda"], st["ggml"]
            ck = c["route_checks"]
            differ = int(sum(1 for t in moe_tokens if sorted(int(x) for x in c["order"][t, :m.used])
                             != sorted(int(x) for x in gg["order"][t, :m.used])))
            print(f"  L{L} routing: cuda pick = exact top-{m.used} on its own input for {int(c['agree'].sum())}/{T} tokens; "
                  f"own-logit violations {len(ck['own_violations'])}, exact-input violations {len(ck['exact_violations'])}, "
                  f"integrity {len(ck['integrity_violations'])}; exact picks differ between the engines' inputs at "
                  f"{differ}/{len(moe_tokens)} oracle tokens", flush=True)
            report.setdefault("discrimination_moe", {})[str(L)] = dmoe
            report.setdefault("routing", {})[str(L)] = {
                "used": m.used, "checks": ck,
                "cuda_selection_matches_exact_on_own_input": f"{int(c['agree'].sum())}/{T}",
                "exact_selections_differ": f"{differ}/{len(moe_tokens)}", "flip_checks": flips}
        for eng in ("cuda", "ggml"):
            xs_in[eng] = st[eng]["hid"]
            pre_ffn_prev[eng] = st[eng]["pre_f"]
        st.clear()
        m.cache.clear()
        if args.state_every:
            os.makedirs(args.state_every, exist_ok=True)
            save_state(os.path.join(args.state_every, f"after-L{L}.npz"), T, L + 1, xs_in, pre_ffn_prev, G, SEL, SEL_LIST,
                       CAND, CAND_EXACT)

    report["table"] = build_table(report)
    print("\nstage                             layer   cuda local   ggml local   cuda/ggml   ggml model  ggml/model")
    for r in report["table"]:
        cl, gl, gm, ra = r["cuda_local"], r["ggml_local"], r["ggml_model"], r["ratio"]
        cs = "" if cl is None else f"{cl:.3e}"
        gs = "" if gl is None else f"{gl:.3e}"
        rs = "" if ra is None else f"{ra:.2f}"
        ms = "" if gm is None else f"{gm:.3e}"
        qs = "" if (gm is None or gl is None) else f"{gl / gm:.2f}"
        print(f"{r['stage']:32s}  L{r['layer']}   {cs:>10s}   {gs:>10s}   {rs:>9s}   {ms:>10s}  {qs}")
    fails, flags, lines = oracle_gate(report, *gate_args)
    report["gate"] = {"factor": args.factor, "slack": args.slack, "exact_bound": args.exact_bound,
                      "model_factor": args.model_factor, "replay_factor": args.replay_factor, "fails": fails, "flags": flags, "pass": not fails}
    print(chr(10).join(lines))
    print(f"flags: {len(flags)}" + ("" if not flags else " (" + ", ".join(flags) + ")"))
    print("gate: PASS" if not fails else "gate: FAIL at " + ", ".join(fails))
    if args.state_out:
        save_state(args.state_out, T, layers[-1] + 1, xs_in, pre_ffn_prev,
                   {eng: {s: {k: v for k, v in d.items()} for s, d in G[eng].items()} for eng in G},
                   SEL, SEL_LIST, CAND, CAND_EXACT)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
