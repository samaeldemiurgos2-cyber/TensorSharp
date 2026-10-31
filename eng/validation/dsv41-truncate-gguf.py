#!/usr/bin/env python3
"""Write a self-consistent DeepSeek V4.1 GGUF that keeps a subset of transformer layers.

The "1017-lite" model for the S8 real-weight gate keeps original layers
0,1,2,3,20,21,22,23,24 (renumbered 0..8), token_embd, output_norm, output, and
the Engram rows of every kept Engram layer. Tensor payloads are byte copies,
streamed in chunks; nothing is requantized.

Every metadata key is classified from the source header. Per-layer keys are
filtered, layer-id keys are renumbered, and the combined Engram table
(deepseek41.ple.*, per_layer_token_embd.weight) is cut to the kept layers' row
slices with head_offsets rebased onto the new table. A key that looks per-layer
but is not understood is a hard error, never a silent copy. Kept layers that
read a dropped layer's shared KV cache, indexer state or candidate mask are
refused. The output is one unsplit GGUF, written to <output>.partial, verified
by sampled hashing against the source and renamed only after it passes. The
manifest records the whole file's sha256 (computed while writing), which is the
--weights-id to give DeepSeek41DistributedProbe for runs on the output.

Failures never leave a partial output behind: any error before the output is
published (a write, close or fsync error such as EDQUOT, a failed verification,
Ctrl-C, SIGTERM) removes the <output>.partial and manifest .partial this run
created; --keep-failed-partial keeps a fully written one for inspection. A
.partial that already exists belongs to someone else and is refused, never
removed. SIGKILL cannot be caught; the next run then refuses the stale file.

Quota-limited volumes: a pod's network volume (MooseFS/FUSE) reports the whole
cluster's free space to statvfs, so the free-space check cannot see the pod
quota. --quota-bytes N --quota-root DIR adds a du-style walk of DIR and refuses
before writing when used + output + reserve (--min-free-gib) exceeds N.

Candidate pruning: the source budget (candidate_top_k_blocks x
candidate_block_size = 2048 x 8 rows for 1017) prunes nothing below about 16K
tokens, so a 2K-token gate never runs it. --candidate-top-k-blocks N rewrites
the budget and records the original in tensorsharp.truncate.*. The loaders read
it under two spellings: the current ones candidate_top_k_blocks, else
candidate_top_k (a file with both must agree); the PR #306-era ones (the
ggml_cuda reference build, /root/ts-final) candidate_top_k only, else 2048. A
file with candidate_top_k_blocks alone therefore pruned on the current loaders
and never on the reference (S8 round 4 on VM1). The output carries
candidate_top_k equal to the budget whenever 2048 would be wrong, so every
loader generation reads the same value and both engines compute the same
function. N must keep at
least indexer.top_k candidate rows ((N-1) x block + 1, the pinned newest block
may hold one row); for 1017 that is N >= 65, which engages pruning from token
521 at the ratio-1 candidate source and keeps ggml_cuda's gathered decode.

Modes:
  default    plan, write, verify, publish (refuses to overwrite)
  --dry-run  print the plan (needs headers only; missing shards are reported)
  --verify   check an existing output against the source and print a JSON report

The tool reads GGUF headers with plain file reads (no mmap), so it only needs
the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import random
import re
import shutil
import signal
import stat as _stat
import struct
import sys
import threading
import time
from pathlib import Path

TOOL_VERSION = "dsv41-truncate-gguf/1"
DEFAULT_LAYERS = (0, 1, 2, 3, 20, 21, 22, 23, 24)
COMBINED_TABLE = "per_layer_token_embd.weight"
ROOT_TENSORS = ("token_embd.weight", "output_norm.weight", "output.weight",
                "output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight")
REQUIRED_ROOTS = ("token_embd.weight", "output_norm.weight", "output.weight")
MAX_RATIO_TAIL = 8          # trailing non-trunk entries tolerated in compress_ratios
PROVENANCE = "tensorsharp.truncate."
GGML_MAX_NAME = 64          # ggml.h; gguf_init_from_file refuses names of this length or longer
GGML_TYPE_COUNT = 43        # ggml facdc44 enum ggml_type
CANDIDATE_BLOCKS_KEY = "deepseek41.attention.indexer.candidate_top_k_blocks"
CANDIDATE_TOPK_KEY = "deepseek41.attention.indexer.candidate_top_k"
CANDIDATE_BLOCK_SIZE_KEY = "deepseek41.attention.indexer.candidate_block_size"
SHARD_RE = re.compile(r"^(?P<prefix>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# ---------------------------------------------------------------------------
# GGUF primitives (little-endian GGUF v2/v3)
# ---------------------------------------------------------------------------

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
TYPE_NAMES = {UINT8: "UINT8", INT8: "INT8", UINT16: "UINT16", INT16: "INT16", UINT32: "UINT32",
              INT32: "INT32", FLOAT32: "FLOAT32", BOOL: "BOOL", STRING: "STRING", ARRAY: "ARRAY",
              UINT64: "UINT64", INT64: "INT64", FLOAT64: "FLOAT64"}
_FMT = {UINT8: "B", INT8: "b", UINT16: "H", INT16: "h", UINT32: "I", INT32: "i", FLOAT32: "f",
        BOOL: "B", UINT64: "Q", INT64: "q", FLOAT64: "d"}
INT_TYPES = {UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, INT64}
_RANGE = {UINT8: (0, 2**8 - 1), INT8: (-2**7, 2**7 - 1), UINT16: (0, 2**16 - 1), INT16: (-2**15, 2**15 - 1),
          UINT32: (0, 2**32 - 1), INT32: (-2**31, 2**31 - 1), UINT64: (0, 2**64 - 1), INT64: (-2**63, 2**63 - 1)}

# ggml_type -> (name, block elements, block bytes), from ggml facdc44 ggml.h / ggml-common.h.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
GGML_TYPE_IDS = {name: tid for tid, (name, _, _) in GGML_TYPES.items()}


class GgufError(ValueError):
    """The file is not a GGUF this tool can read."""


class TruncateError(ValueError):
    """The requested truncation is refused; the message says why."""


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def tensor_nbytes(ttype: int, shape) -> int:
    if ttype not in GGML_TYPES:
        raise GgufError(f"unknown ggml tensor type {ttype}")
    _, block, size = GGML_TYPES[ttype]
    if not shape or shape[0] % block:
        raise GgufError(f"tensor row of {shape[0] if shape else 0} elements is not a multiple of the "
                        f"{GGML_TYPES[ttype][0]} block size {block}")
    nbytes = shape[0] // block * size
    for dim in shape[1:]:
        nbytes *= dim
    return nbytes


class Kv:
    """One metadata entry. raw is the on-disk value bytes (type tag included) when unchanged."""
    __slots__ = ("key", "vtype", "value", "atype", "raw")

    def __init__(self, key, vtype, value, atype=None, raw=None):
        self.key, self.vtype, self.value, self.atype, self.raw = key, vtype, value, atype, raw

    def encoded(self) -> bytes:
        return self.raw if self.raw is not None else encode_value(self.vtype, self.value, self.atype)

    def type_name(self) -> str:
        if self.vtype == ARRAY:
            return f"ARRAY[{TYPE_NAMES.get(self.atype, self.atype)}]"
        return TYPE_NAMES.get(self.vtype, str(self.vtype))

    def is_int_array(self) -> bool:
        return self.vtype == ARRAY and self.atype in INT_TYPES

    def is_numeric_array(self) -> bool:
        return self.vtype == ARRAY and self.atype in _FMT


def _encode_string(text: str) -> bytes:
    data = text.encode("utf-8", "surrogateescape")
    return struct.pack("<Q", len(data)) + data


def _check_range(vtype, value, key="value"):
    lo, hi = _RANGE[vtype]
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise GgufError(f"{key}: {value!r} does not fit {TYPE_NAMES[vtype]}")


def encode_value(vtype: int, value, atype=None) -> bytes:
    out = [struct.pack("<I", vtype)]
    if vtype == STRING:
        out.append(_encode_string(value))
    elif vtype == ARRAY:
        out.append(struct.pack("<IQ", atype, len(value)))
        if atype == STRING:
            out.extend(_encode_string(item) for item in value)
        elif atype in _FMT:
            if atype in INT_TYPES:
                for item in value:
                    _check_range(atype, item)
            items = [1 if item else 0 for item in value] if atype == BOOL else list(value)
            out.append(struct.pack(f"<{len(items)}{_FMT[atype]}", *items))
        else:
            raise GgufError(f"unsupported array element type {atype}")
    elif vtype == BOOL:
        out.append(struct.pack("<B", 1 if value else 0))
    elif vtype in _FMT:
        if vtype in INT_TYPES:
            _check_range(vtype, value)
        out.append(struct.pack("<" + _FMT[vtype], value))
    else:
        raise GgufError(f"unsupported value type {vtype}")
    return b"".join(out)


class _HeaderReader:
    """Incremental reader over the front of a file; keeps every byte it consumed."""

    def __init__(self, stream, size):
        self.stream, self.size = stream, size
        self.data = bytearray()
        self.pos = 0

    def need(self, count):
        if self.pos + count > self.size:
            raise GgufError("header runs past the end of the file")
        while len(self.data) - self.pos < count:
            chunk = self.stream.read(max(1 << 20, count))
            if not chunk:
                raise GgufError("header runs past the end of the file")
            self.data += chunk

    def unpack(self, fmt):
        size = struct.calcsize(fmt)
        self.need(size)
        values = struct.unpack_from(fmt, self.data, self.pos)
        self.pos += size
        return values

    def u32(self):
        return self.unpack("<I")[0]

    def u64(self):
        return self.unpack("<Q")[0]

    def string(self):
        length = self.u64()
        self.need(length)
        text = bytes(self.data[self.pos:self.pos + length]).decode("utf-8", "surrogateescape")
        self.pos += length
        return text

    def value(self, vtype):
        if vtype == STRING:
            return self.string(), None
        if vtype == ARRAY:
            atype, count = self.unpack("<IQ")
            if atype == STRING:
                return [self.string() for _ in range(count)], atype
            if atype not in _FMT:
                raise GgufError(f"unsupported array element type {atype}")
            item = struct.calcsize(_FMT[atype])
            if count * item > self.size:
                raise GgufError("array runs past the end of the file")
            values = list(self.unpack(f"<{count}{_FMT[atype]}")) if count else []
            if atype == BOOL:
                values = [bool(v) for v in values]
            return values, atype
        if vtype not in _FMT:
            raise GgufError(f"unsupported value type {vtype}")
        value = self.unpack("<" + _FMT[vtype])[0]
        return (bool(value) if vtype == BOOL else value), None


class TensorInfo:
    __slots__ = ("name", "shape", "type", "offset", "nbytes", "shard", "abs_offset")

    def __init__(self, name, shape, ttype, offset):
        self.name, self.shape, self.type, self.offset = name, list(shape), ttype, offset
        self.nbytes = tensor_nbytes(ttype, self.shape)
        self.shard, self.abs_offset = 0, None

    @property
    def type_name(self):
        return GGML_TYPES[self.type][0]


class GgufHeader:
    """Header of one GGUF file: metadata, tensor table, data offset. Payloads are not read."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            self.file_size = stream.tell()
            stream.seek(0)
            reader = _HeaderReader(stream, self.file_size)
            reader.need(4)
            if bytes(reader.data[:4]) != b"GGUF":
                raise GgufError(f"{self.path} is not a GGUF file")
            reader.pos = 4
            self.version = reader.u32()
            if self.version not in (2, 3):
                raise GgufError(f"{self.path}: unsupported GGUF version {self.version} (little-endian v2/v3 only)")
            n_tensors, n_kv = reader.unpack("<QQ")
            if n_tensors > 1 << 24 or n_kv > 1 << 24:
                raise GgufError(f"{self.path}: implausible tensor/metadata counts")
            self.kvs: dict[str, Kv] = {}
            for _ in range(n_kv):
                key = reader.string()
                start = reader.pos
                vtype = reader.u32()
                value, atype = reader.value(vtype)
                if key in self.kvs:
                    raise GgufError(f"{self.path}: duplicate metadata key {key}")
                self.kvs[key] = Kv(key, vtype, value, atype, bytes(reader.data[start:reader.pos]))
            self.tensors: list[TensorInfo] = []
            for _ in range(n_tensors):
                name = reader.string()
                n_dims = reader.u32()
                if not 1 <= n_dims <= 4:
                    raise GgufError(f"{self.path}: tensor {name} has {n_dims} dimensions")
                dims = reader.unpack(f"<{n_dims}Q")
                ttype, offset = reader.unpack("<IQ")
                self.tensors.append(TensorInfo(name, dims, ttype, offset))
            self.table_end = reader.pos
        alignment = self.kvs.get("general.alignment")
        if alignment is not None and alignment.vtype != UINT32:
            raise GgufError(f"{self.path}: general.alignment must be UINT32")
        self.alignment = alignment.value if alignment is not None else 32
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise GgufError(f"{self.path}: general.alignment {self.alignment} is not a power of two")
        self.data_offset = align(self.table_end, self.alignment)
        for tensor in self.tensors:
            if tensor.offset % self.alignment:
                raise GgufError(f"{self.path}: tensor {tensor.name} offset {tensor.offset} is not aligned")
            tensor.abs_offset = self.data_offset + tensor.offset

    def required_size(self):
        return max([self.table_end] + [t.abs_offset + t.nbytes for t in self.tensors])

    def value(self, key, default=None):
        kv = self.kvs.get(key)
        return default if kv is None else kv.value


class SourceModel:
    """A (possibly split) GGUF checkpoint: shard-1 metadata plus one tensor table across shards."""

    def __init__(self, path, allow_missing=False):
        self.path = Path(path)
        first = GgufHeader(self.path)
        self.kvs = first.kvs
        count_kv = first.kvs.get("split.count")
        count = int(count_kv.value) if count_kv is not None else 1
        self.paths = [self.path]
        if count > 1:
            match = SHARD_RE.match(self.path.name)
            if not match or int(match["count"]) != count:
                raise GgufError(f"{self.path} declares split.count {count} but is not named <prefix>-NNNNN-of-{count:05d}.gguf")
            if int(match["no"]) != 1 or int(first.value("split.no", 0)) != 0:
                raise GgufError(f"{self.path} is not the first shard; pass shard 00001 (it carries the metadata)")
            self.paths = [self.path.with_name(f"{match['prefix']}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
        self.headers, self.missing = [], []
        for index, shard_path in enumerate(self.paths):
            if index == 0:
                header = first
            elif not shard_path.exists():
                if not allow_missing:
                    raise GgufError(f"shard {shard_path} is missing; every shard must sit beside shard 1")
                self.missing.append(str(shard_path))
                self.headers.append(None)
                continue
            else:
                header = GgufHeader(shard_path)
            if count > 1 and (int(header.value("split.no", -1)) != index or int(header.value("split.count", -1)) != count):
                raise GgufError(f"{shard_path} has split.no/split.count that do not match its position {index + 1} of {count}")
            self.headers.append(header)
        self.tensors: dict[str, TensorInfo] = {}
        self.incomplete = []
        for index, header in enumerate(self.headers):
            if header is None:
                continue
            for tensor in header.tensors:
                if tensor.name in self.tensors:
                    raise GgufError(f"tensor {tensor.name} appears in more than one shard")
                tensor.shard = index
                self.tensors[tensor.name] = tensor
            if header.required_size() > header.file_size:
                self.incomplete.append(dict(path=str(header.path), bytes=header.file_size, required=header.required_size()))
        self.complete = not self.missing
        expected = first.value("split.tensors.count")
        if self.complete and count > 1 and expected is not None and int(expected) != len(self.tensors):
            raise GgufError(f"split.tensors.count is {expected} but the shards hold {len(self.tensors)} tensors")

    def files(self):
        result = []
        for path, header in zip(self.paths, self.headers):
            if header is not None:
                stat = path.stat()
                result.append(dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                                   tensors=len(header.tensors)))
        return result


# ---------------------------------------------------------------------------
# Metadata classification
# ---------------------------------------------------------------------------

# deepseek41.<suffix> keys that hold no per-layer data. Copied unchanged.
GLOBAL_KEYS = {
    "context_length", "embedding_length", "embedding_length_out", "vocab_size",
    "rope.scaling.type", "rope.scaling.factor", "rope.scaling.original_context_length",
    "rope.scaling.yarn_beta_fast", "rope.scaling.yarn_beta_slow", "rope.freq_base", "rope.dimension_count",
    "attention.layer_norm_rms_epsilon", "attention.key_length", "attention.value_length",
    "attention.q_lora_rank", "attention.output_group_count", "attention.output_lora_rank",
    "attention.compress_rope_freq_base", "expert_count", "expert_used_count", "expert_gating_func",
    "expert_feed_forward_length", "expert_shared_count", "expert_weights_scale", "expert_weights_norm",
    "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k",
    "attention.indexer.candidate_top_k_blocks", "attention.indexer.candidate_top_k",
    "attention.indexer.candidate_block_size",
    "hyper_connection.count", "hyper_connection.sinkhorn_iterations", "hyper_connection.epsilon",
    # Engram hash geometry shared by every Engram layer (head dim, n-gram size, heads, vocab).
    "ple.ngram_size", "ple.heads_per_ngram", "embedding_length_per_layer_input",
    "engram.vocab_size", "engram.pad_token_id", "engram.compressed_vocab_size", "engram.token_map",
    "engram.pad_id", "engram.max_ngram_size", "engram.head_count", "engram.key_length",
}
# Scalars here are global; an array of block_count values is per-layer (llama.cpp convention).
SCALAR_OR_PER_LAYER = {"attention.head_count", "attention.head_count_kv", "feed_forward_length",
                       "attention.sliding_window"}
PER_LAYER_VALUES = {"swiglu_clamp_exp", "swiglu_clamp_shexp", "attention.sliding_window_pattern"}
PER_LAYER_WITH_TAIL = {"attention.compress_ratios"}
LAYER_ID_LISTS = {"attention.kv_source_layers": "kv", "attention.indexer.source_layers": "index",
                  "ple.layers": "engram", "engram.layer_ids": "engram", "dspark.target_layers": "dspark"}
LAYER_ID_SCALARS = {"attention.indexer.candidate_source_layer": "candidate"}
# Arrays laid out [engram layer][width]; the width names the per-layer stride.
ENGRAM_MAJOR = {"ple.layer_multipliers": "ngram", "ple.head_vocab_sizes": "columns", "ple.head_offsets": "columns",
                "engram.multipliers": "ngram", "engram.primes": "columns", "engram.offsets": "columns"}
DSPARK_SCALARS = {"dspark.block_size", "dspark.noise_token_id", "dspark.markov_rank",
                  "dspark.expert_count", "dspark.expert_used_count"}
COPY_NAMESPACES = ("general.", "tokenizer.", "quantize.")
LAYERISH_WORDS = {"layer", "layers", "block", "blocks", "blk", "source", "sources", "nextn", "mtp", "pattern",
                  "ratios", "target", "targets", "swa", "hc", "ple", "engram", "dspark", "per"}


def looks_per_layer(key: str, kv: Kv, n_layer: int) -> str | None:
    """A reason this key may carry per-layer data, or None."""
    words = set(re.split(r"[._]", key.lower()))
    hit = sorted(words & LAYERISH_WORDS)
    if hit:
        return f"name mentions {', '.join(hit)}"
    if kv.is_numeric_array():
        count = len(kv.value)
        if count and (count % n_layer == 0 or n_layer <= count <= n_layer + MAX_RATIO_TAIL):
            return f"array of {count} values relates to block_count {n_layer}"
        if kv.is_int_array() and 0 < count < n_layer and all(0 <= v < n_layer for v in kv.value) and \
                all(a < b for a, b in zip(kv.value, kv.value[1:])):
            return "increasing integers inside [0, block_count) read like layer ids"
    return None


def _int(kv: Kv, key: str) -> int:
    if kv.vtype not in INT_TYPES:
        raise TruncateError(f"{key} must be an integer, found {kv.type_name()}")
    return int(kv.value)


def _layer_ids(kv: Kv, key: str, n_layer: int) -> list[int]:
    if not kv.is_int_array():
        raise TruncateError(f"{key} must be an integer array, found {kv.type_name()}")
    values = [int(v) for v in kv.value]
    if any(v < 0 or v >= n_layer for v in values) or any(a >= b for a, b in zip(values, values[1:])):
        raise TruncateError(f"{key} = {values} is not an increasing list of layer ids below {n_layer}")
    return values


def parse_layers(spec: str) -> list[int]:
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise TruncateError(f"bad layer range {part}")
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    if not result:
        raise TruncateError("--layers selects no layer")
    return sorted(result)


def source_of(owners: list[int], layer: int):
    best = None
    for owner in owners:
        if owner <= layer:
            best = owner
    return best


# ---------------------------------------------------------------------------
# Loader ports (dsv41_engram_gguf.h / Dsv41EngramData.cs)
# ---------------------------------------------------------------------------

def _array_u64(values: dict, key: str, max_count: int) -> list[int]:
    """detail::gguf_array: an integer array with 1..max_count non-negative entries."""
    kv = values.get(key)
    if kv is None or not kv.is_int_array() or not 0 < len(kv.value) <= max_count or any(v < 0 for v in kv.value):
        raise TruncateError(f"Missing or invalid DeepSeek V4.1 GGUF metadata: {key}")
    return [int(v) for v in kv.value]


def _u32(values: dict, key: str, fallback=None) -> int:
    kv = values.get(key)
    if kv is None:
        if fallback is None:
            raise TruncateError(f"Missing or invalid DeepSeek V4.1 GGUF metadata: {key}")
        return fallback
    if kv.vtype not in (UINT32, INT32, UINT64, INT64) or not 0 <= int(kv.value) <= 2**32 - 1:
        raise TruncateError(f"Missing or invalid DeepSeek V4.1 GGUF metadata: {key}")
    return int(kv.value)


def combined_engram_slices(values: dict, table_rows) -> list[dict]:
    """Port of tsg_dsv41::combined_engram_slices: one contiguous row range per Engram layer."""
    ngram = _u32(values, "deepseek41.ple.ngram_size")
    heads = _u32(values, "deepseek41.ple.heads_per_ngram")
    if not 2 <= ngram <= 16 or not 1 <= heads <= 128:
        raise TruncateError("Missing or invalid DeepSeek V4.1 GGUF metadata: deepseek41.ple dimensions")
    columns = (ngram - 1) * heads
    ids = _array_u64(values, "deepseek41.ple.layers", 128)
    primes = _array_u64(values, "deepseek41.ple.head_vocab_sizes", len(ids) * columns)
    offsets = _array_u64(values, "deepseek41.ple.head_offsets", len(ids) * columns)
    if len(primes) != len(ids) * columns or len(offsets) != len(ids) * columns:
        raise TruncateError("deepseek41.ple flattened array lengths")
    slices = []
    for i, layer in enumerate(ids):
        first, rows = offsets[i * columns], 0
        for j in range(columns):
            if primes[i * columns + j] < 2 or offsets[i * columns + j] != first + rows:
                raise TruncateError("deepseek41.ple bucket layout is not contiguous per layer")
            rows += primes[i * columns + j]
        if layer > 2**31 - 1 or rows > 2**31 - 1 or (
                table_rows is not None and (first > table_rows or rows > table_rows - first)):
            raise TruncateError(f"deepseek41.ple bucket rows exceed {COMBINED_TABLE}")
        slices.append(dict(id=layer, first_row=first, rows=rows, index=i))
    return slices


def engram_layout(values: dict, shapes: dict) -> dict:
    """Port of load_engram_gguf + Dsv41EngramData.Load/Validate (the Engram part).

    shapes maps tensor names to GGML-order shapes, with the combined table present
    as itself. Returns per-layer hash parameters (offsets rebased onto each layer's
    own table) and the row slice each layer occupies in its table tensor.
    """
    tokens = values.get("tokenizer.ggml.tokens")
    if tokens is None or tokens.vtype != ARRAY or tokens.atype != STRING or not 0 < len(tokens.value) <= 1048576:
        raise TruncateError("Missing or invalid DeepSeek V4.1 GGUF metadata: tokenizer.ggml.tokens")
    vocab = len(tokens.value)
    embedding = shapes.get("token_embd.weight")
    if embedding is not None and (len(embedding) < 2 or embedding[1] != vocab):
        raise TruncateError("token embedding vocabulary does not match tokenizer")
    combined = "deepseek41.engram.layer_ids" not in values and "deepseek41.ple.layers" in values
    p = "deepseek41.ple." if combined else "deepseek41.engram."
    if combined:
        ngram, heads = _u32(values, p + "ngram_size"), _u32(values, p + "heads_per_ngram")
        dim = _u32(values, "deepseek41.embedding_length_per_layer_input")
    else:
        ngram, heads = _u32(values, p + "max_ngram_size"), _u32(values, p + "head_count")
        dim = _u32(values, p + "key_length")
    if not 2 <= ngram <= 16 or not 1 <= heads <= 128 or not 1 <= dim <= 65536:
        raise TruncateError("deepseek41.engram dimensions")
    columns = (ngram - 1) * heads
    ids = _array_u64(values, p + ("layers" if combined else "layer_ids"), 128)
    multipliers = _array_u64(values, p + ("layer_multipliers" if combined else "multipliers"), len(ids) * ngram)
    primes = _array_u64(values, p + ("head_vocab_sizes" if combined else "primes"), len(ids) * columns)
    offsets = _array_u64(values, p + ("head_offsets" if combined else "offsets"), len(ids) * columns)
    if len(multipliers) != len(ids) * ngram or len(primes) != len(ids) * columns or len(offsets) != len(ids) * columns:
        raise TruncateError("deepseek41.engram flattened array lengths")
    slices = None
    if combined:
        table = shapes.get(COMBINED_TABLE, "absent") if shapes else None
        if table == "absent":
            raise TruncateError(f"DeepSeek V4.1 Engram metadata names {COMBINED_TABLE} but no shard holds it")
        if table is not None and (len(table) < 2 or any(d != 1 for d in table[2:])):
            raise TruncateError(f"{COMBINED_TABLE} must be two-dimensional")
        slices = combined_engram_slices(values, table[1] if table is not None else None)
        for i in range(len(ids)):
            first = offsets[i * columns]
            for j in range(columns):
                offsets[i * columns + j] -= first
    token_map = _array_u64(values, "deepseek41.engram.token_map", vocab)
    if any(v >= vocab for v in token_map):
        raise TruncateError("Missing or invalid DeepSeek V4.1 GGUF metadata: deepseek41.engram.token_map")
    compressed = max(token_map) + 1
    if combined:
        pad_token = _u32(values, "deepseek41.engram.pad_token_id")
        if pad_token >= len(token_map):
            raise TruncateError("deepseek41.engram.pad_token_id")
        pad = token_map[pad_token]
        if "deepseek41.engram.compressed_vocab_size" in values and \
                _u32(values, "deepseek41.engram.compressed_vocab_size") != compressed:
            raise TruncateError("deepseek41.engram.compressed_vocab_size does not match the token map")
    else:
        pad = _u32(values, "deepseek41.engram.pad_id")
    n_layer = _u32(values, "deepseek41.block_count")
    if not 1 <= n_layer <= 128:
        raise TruncateError("deepseek41.block_count")
    if len(token_map) != vocab or not pad < compressed or set(token_map) != set(range(compressed)):
        raise TruncateError("DeepSeek V4.1 GGUF Engram token map / pad id is invalid")
    layers = []
    for i, layer in enumerate(ids):
        if layer >= n_layer:
            raise TruncateError("Missing or invalid DeepSeek V4.1 GGUF metadata: deepseek41.engram.layer_ids")
        if i and layer <= ids[i - 1]:
            raise TruncateError("Invalid DeepSeek V4.1 Engram table dimensions (layer ids must increase)")
        if combined:
            rows = slices[i]["rows"]
            if table is not None and table[0] != dim:
                raise TruncateError(f"Engram embedding tensor dimensions at layer {layer}")
        else:
            shape = shapes.get(f"blk.{layer}.engram_embd.weight") if shapes else None
            if shapes and (shape is None or len(shape) < 2 or shape[0] != dim or any(d != 1 for d in shape[2:])):
                raise TruncateError(f"Engram embedding tensor dimensions at layer {layer}")
            rows = shape[1] if shape is not None else None
        mult = multipliers[i * ngram:(i + 1) * ngram]
        prime = primes[i * columns:(i + 1) * columns]
        offs = offsets[i * columns:(i + 1) * columns]
        if any(m % 2 == 0 or m > (2**63 - 1) // compressed for m in mult):
            raise TruncateError("Invalid DeepSeek V4.1 Engram hash multiplier")
        total = 0
        for j in range(columns):
            if prime[j] < 2 or prime[j] > 2**32 - 1 or offs[j] != total:
                raise TruncateError("Invalid DeepSeek V4.1 Engram bucket layout")
            total += prime[j]
        if rows is not None and (total != rows or not 0 < rows <= 2**31 - 1):
            raise TruncateError("DeepSeek V4.1 Engram bucket sizes do not match table rows")
        layers.append(dict(id=layer, multipliers=mult, primes=prime, offsets=offs, rows=total,
                           first_row=slices[i]["first_row"] if combined else 0))
    return dict(combined=combined, vocab=vocab, compressed=compressed, pad=pad, ngram=ngram, heads=heads,
                dim=dim, columns=columns, token_map=token_map, layers=layers)


def engram_hash_rows(layout: dict, layer: dict, tokens) -> list[int]:
    """Port of Dsv41EngramData.HashTokens / engram_data::hash_tokens for one layer.

    Returns layer-relative rows laid out [token][hash column]; add layer["first_row"]
    for the row of the table tensor (the slice view's start in the combined layout).
    """
    mask = (1 << 64) - 1
    history = [-1 if t < 0 else layout["token_map"][t] for t in tokens]
    rows = []
    for pos in range(len(history)):
        rolling, blocked = 0, False
        for shift in range(layout["ngram"]):
            blocked = blocked or pos < shift or history[pos - shift] < 0
            token = layout["pad"] if blocked else history[pos - shift]
            rolling ^= (token * layer["multipliers"][shift]) & mask
            if shift == 0:
                continue
            for head in range(layout["heads"]):
                column = (shift - 1) * layout["heads"] + head
                rows.append(rolling % layer["primes"][column] + layer["offsets"][column])
    return rows


def ggml_gguf_layout(header: GgufHeader) -> dict:
    """Port of the tensor-table rules of gguf_init_from_file (ggml facdc44 src/gguf.cpp:630-818).

    Both V4.1 loaders open the file through it. Beyond what GgufHeader already
    refuses (version, duplicate keys, alignment type and power of two, 1..4 dims,
    known types, rows divisible by the block size), ggml rejects a name of
    GGML_MAX_NAME bytes or more, a duplicate name, a dimension >= 2^63, and any
    tensor whose offset is not exactly the running sum of GGML_PAD(nbytes,
    alignment) over the tensors before it in table order. Aligned, increasing
    offsets with gaps (which gguf-py and this tool's reader accept) are refused.
    """
    problems, seen, expected = [], set(), 0
    for tensor in header.tensors:
        size = len(tensor.name.encode("utf-8", "surrogateescape"))
        if size >= GGML_MAX_NAME:
            problems.append(f"tensor name {tensor.name!r} is too long: {size} >= {GGML_MAX_NAME}")
        if tensor.name in seen:
            problems.append(f"duplicate tensor name {tensor.name!r}")
        seen.add(tensor.name)
        if any(d >= 2**63 for d in tensor.shape):
            problems.append(f"tensor {tensor.name!r} has a negative (int64) dimension")
        if not 0 <= tensor.type < GGML_TYPE_COUNT:
            problems.append(f"tensor {tensor.name!r} has invalid ggml type {tensor.type}")
        if tensor.offset != expected:
            problems.append(f"tensor {tensor.name!r} has offset {tensor.offset}, expected {expected}")
        expected += align(tensor.nbytes, header.alignment)
    if header.data_offset != align(header.table_end, header.alignment):
        problems.append(f"data section starts at {header.data_offset}, expected {align(header.table_end, header.alignment)}")
    return dict(tensors=len(header.tensors), alignment=header.alignment, data_offset=header.data_offset,
                data_size=expected, problems=problems[:20], problem_count=len(problems))


def loader_contract(values: dict, shapes: dict | None) -> list[dict]:
    """The load-time checks of the native loader (ggml_ops_deepseek4.cpp, dsv41_engram_gguf.h)
    and the managed one (Dsv41EngramData.Load, DeepSeek4CudaExecutor.LoadEngramMetadata),
    evaluated against a header. shapes=None means no tensor table is available; checks that
    need tensors are then reported as unchecked, never as passed."""
    checks = []

    def check(name, fn):
        try:
            detail = fn()
            checks.append(dict(check=name, status="pass", detail=detail))
        except (TruncateError, KeyError) as error:
            checks.append(dict(check=name, status="FAIL", detail=str(error)))
        except _Unchecked as reason:
            checks.append(dict(check=name, status="unchecked", detail=str(reason)))

    def need_tensors():
        if shapes is None:
            raise _Unchecked("no tensor table (header-only plan)")

    n_layer = _u32(values, "deepseek41.block_count")
    ratios_kv = values.get("deepseek41.attention.compress_ratios")

    def arch():
        kv = values.get("general.architecture")
        if kv is None or kv.value != "deepseek41":
            raise TruncateError("general.architecture must be deepseek41")
        if not 1 <= n_layer <= 128:
            raise TruncateError("deepseek41.block_count must be 1..128")
        return f"deepseek41, block_count {n_layer}"
    check("architecture and block_count", arch)

    def ratios():
        if ratios_kv is None or not ratios_kv.is_int_array() or len(ratios_kv.value) < n_layer:
            raise TruncateError("deepseek41.attention.compress_ratios must hold >= block_count integers")
        bad = [r for r in ratios_kv.value[:n_layer] if not 0 <= r <= 2]
        if bad:
            raise TruncateError(f"V4.1 compression ratios must be 0..2, found {bad}")
        for key in ("deepseek41.swiglu_clamp_exp", "deepseek41.swiglu_clamp_shexp"):
            if key in values and len(values[key].value) < n_layer:
                raise TruncateError(f"{key} is shorter than block_count")
        return f"ratios[:{n_layer}] = {ratios_kv.value[:n_layer]} (+{len(ratios_kv.value) - n_layer} tail)"
    check("compress_ratios / swiglu lengths", ratios)

    layout_box = {}
    topo_box = {}

    def engram():
        layout = engram_layout(values, shapes or {})
        layout_box["layout"] = layout
        return dict(combined=layout["combined"], pad=layout["pad"], compressed_vocab=layout["compressed"],
                    layers=[dict(id=l["id"], first_row=l["first_row"], rows=l["rows"]) for l in layout["layers"]],
                    table_checked=shapes is not None)
    check("Engram metadata (load_engram_gguf / Dsv41EngramData.Load)", engram)

    def owners():
        need_tensors()
        kv = [i for i in range(n_layer) if f"blk.{i}.attn_compressor_kv.weight" in shapes]
        index = [i for i in range(n_layer) if f"blk.{i}.indexer.attn_q_b.weight" in shapes]
        if not kv or not index:
            raise TruncateError("kv/indexer source layer lists derived from tensors must be non-empty")
        for key, found in (("deepseek41.attention.kv_source_layers", kv), ("deepseek41.attention.indexer.source_layers", index)):
            if key in values and [int(v) for v in values[key].value] != found:
                raise TruncateError(f"{key} = {values[key].value} disagrees with tensor-derived owners {found}")
        return dict(kv_sources=kv, index_sources=index)
    check("kv/index source owners (tensor presence == metadata)", owners)

    def topology():
        if shapes is None:
            kv = [int(v) for v in values["deepseek41.attention.kv_source_layers"].value]
            index = [int(v) for v in values["deepseek41.attention.indexer.source_layers"].value]
        else:
            kv = [i for i in range(n_layer) if f"blk.{i}.attn_compressor_kv.weight" in shapes]
            index = [i for i in range(n_layer) if f"blk.{i}.indexer.attn_q_b.weight" in shapes]
        ratios = [int(r) for r in ratios_kv.value[:n_layer]]
        rows = []
        kv_src = index_src = -1
        for il in range(n_layer):
            if il in kv:
                kv_src = il
            if il in index:
                index_src = il
            ratio = ratios[il]
            if ratio and (kv_src < 0 or index_src < 0 or ratios[kv_src] != ratio or ratios[index_src] != ratio):
                raise TruncateError(f"V4.1 cache-sharing topology does not match compression ratios at layer {il}")
            rows.append((il, ratio, kv_src if ratio else -1, index_src if ratio else -1))
        candidate = next((i for i in index if ratios[i] == 1), -1)
        key = "deepseek41.attention.indexer.candidate_source_layer"
        if key in values:
            if values[key].vtype not in (UINT32, INT32, UINT64, INT64):
                raise TruncateError(f"{key} must be an integer")
            candidate = int(values[key].value)
        if candidate < -1 or candidate >= n_layer:
            raise TruncateError(f"{key} = {candidate} is out of range")
        if candidate >= 0:
            if candidate not in index:
                raise TruncateError("candidate source must own an indexer")
            blocks_key, topk_key = "deepseek41.attention.indexer.candidate_top_k_blocks", "deepseek41.attention.indexer.candidate_top_k"
            topk = _u32(values, blocks_key if blocks_key in values else topk_key, 2048)
            if blocks_key in values and topk_key in values and _u32(values, topk_key) != topk:
                raise TruncateError(f"{blocks_key} disagrees with {topk_key}")
            if not 1 <= topk <= 2**31 - 1 or not 1 <= _u32(values, "deepseek41.attention.indexer.candidate_block_size", 8) <= 2**31 - 1:
                raise TruncateError("Invalid candidate indexer geometry")
            topo_box["budget"] = topk
        topo_box["candidate"] = candidate
        return dict(kv_sources=kv, index_sources=index, candidate=candidate,
                    per_layer=[dict(layer=l, ratio=r, kv_source=k, index_source=x) for l, r, k, x in rows],
                    owners_from="tensors" if shapes is not None else "metadata lists")
    check("cache-sharing topology and candidate source", topology)

    def budget_agreement():
        # The PR #306-era loaders (dsv41_engram_gguf.h and Dsv41EngramData.cs before the 1017 work,
        # i.e. the ggml_cuda reference build) read candidate_top_k only, with default 2048. The
        # current ones read candidate_top_k_blocks first. Both must arrive at the same budget, or
        # the reference prunes differently from the engine it validates.
        if "candidate" not in topo_box:
            raise _Unchecked("the cache-sharing topology check did not pass")
        if topo_box["candidate"] < 0:
            return "no candidate source layer: no loader reads the budget"
        current, legacy = topo_box["budget"], _u32(values, CANDIDATE_TOPK_KEY, 2048)
        if current != legacy:
            raise TruncateError(
                f"the loaders disagree on the candidate budget: current ones (candidate_top_k_blocks, else "
                f"candidate_top_k) read {current} blocks, PR #306-era ones (candidate_top_k only, default 2048) "
                f"read {legacy}; the ggml_cuda reference build would prune differently. Write candidate_top_k = "
                f"{current} as well")
        return dict(blocks=current, current_loaders=current, pr306_loaders=legacy)
    check("candidate budget read alike by current and PR #306-era loaders", budget_agreement)

    def tensors():
        need_tensors()
        layout = layout_box.get("layout")
        hash_layers = _u32(values, "deepseek41.hash_layer_count", 0)
        engram_ids = {l["id"] for l in layout["layers"]} if layout else set()
        ratios = [int(r) for r in ratios_kv.value[:n_layer]]
        missing = []
        for name in REQUIRED_ROOTS:
            if name not in shapes:
                missing.append(name)
        for il in range(n_layer):
            names = ["attn_norm.weight", "attn_sinks.weight", "attn_q_a.weight", "attn_q_a_norm.weight",
                     "attn_q_b.weight", "attn_kv.weight", "attn_kv_a_norm.weight", "attn_output_a.weight",
                     "attn_output_b.weight", "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                     "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight", "ffn_gate_inp.weight",
                     "ffn_norm.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight", "ffn_up_exps.weight",
                     "ffn_gate_shexp.weight", "ffn_down_shexp.weight", "ffn_up_shexp.weight",
                     "ffn_gate_tid2eid.weight" if il < hash_layers else "exp_probs_b.bias"]
            if f"blk.{il}.attn_compressor_kv.weight" in shapes:
                names += ["attn_compressor_norm.weight", "indexer.attn_k.weight", "indexer.k_norm.weight"]
                if ratios[il] > 1:
                    names.append("attn_compressor_gate.weight")
            if f"blk.{il}.indexer.attn_q_b.weight" in shapes:
                names.append("indexer.proj.weight")
            if il in engram_ids:
                if f"blk.{il}.engram_gate.weight" in shapes:   # the native loader's own test
                    names += ["engram_gate.weight", "engram_kv.weight"]
                else:
                    names += ["engram_q.weight", "engram_k.weight", "engram_wkv.weight"]
            elif f"blk.{il}.engram_embd.weight" in shapes:
                raise TruncateError(f"Engram embedding tensor at layer {il} is absent from the Engram layer list")
            missing += [f"blk.{il}.{n}" for n in names if f"blk.{il}.{n}" not in shapes]
            if layout and layout["combined"] and f"blk.{il}.engram_embd.weight" in shapes:
                raise TruncateError(f"GGUF has both blk.{il}.engram_embd.weight and a combined Engram table")
        if missing:
            raise TruncateError(f"tensors the loaders look up are missing: {missing[:12]}{' ...' if len(missing) > 12 else ''}")
        stray = sorted({int(m[1]) for m in (BLK_RE.match(n) for n in shapes) if m and int(m[1]) >= n_layer})
        if stray:
            raise TruncateError(f"tensors exist for blocks {stray} beyond block_count {n_layer}")
        return f"{n_layer} layers complete; hash layers {hash_layers}; engram layers {sorted(engram_ids)}"
    check("per-layer tensors looked up by the native loader", tensors)
    return checks


class _Unchecked(Exception):
    pass


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class Plan:
    def __init__(self):
        self.kvs: list[Kv] = []
        self.changes: list[dict] = []
        self.unclassified: list[dict] = []
        self.tensors: list[dict] = []
        self.notes: list[str] = []
        self.candidate_budget: dict | None = None
        self.candidate_override: int | None = None


def _change(plan, key, action, old, new, note=""):
    plan.changes.append(dict(key=key, action=action, old=old, new=new, note=note))


def _candidate_budget(kvs, plan, new_ratios, override) -> dict:
    """The candidate-pruning budget both loaders will read, and the first token it prunes at.

    Native load_engram_gguf (dsv41_engram_gguf.h:246-252) and managed
    Dsv41EngramData.Load read candidate_top_k_blocks, else candidate_top_k, else
    2048, and candidate_block_size, else 8. Their PR #306-era versions (the
    ggml_cuda reference build) read candidate_top_k only, else 2048; make_plan
    writes candidate_top_k whenever that default would be wrong, so both
    generations read the budget returned here. ggml_cuda builds the candidate mask
    only once the candidate source holds more than blocks x block_size compressed
    rows (ggml_ops_deepseek41.inc:311); the direct engine ranks blocks at every
    length but keeps all of them until then (ts_dsv41_candidate_f32). Shorter
    contexts therefore never exercise pruning.
    """
    def read(key):
        return _int(kvs[key], key) if key in kvs else None

    blocks, topk = read(CANDIDATE_BLOCKS_KEY), read(CANDIDATE_TOPK_KEY)
    source = blocks if blocks is not None else topk if topk is not None else 2048
    block = read(CANDIDATE_BLOCK_SIZE_KEY)
    block = 8 if block is None else block
    index_topk = read("deepseek41.attention.indexer.top_k")
    origin_key = PROVENANCE + "candidate_top_k_blocks_source"
    original = _int(kvs[origin_key], origin_key) if origin_key in kvs else source
    # Fewest blocks that still leave indexer.top_k candidate rows when the pinned
    # newest block holds a single visible row: (N - 1) * block + 1 >= top_k.
    min_blocks = None if index_topk is None or block < 1 else max(1, -(-(index_topk - 1) // block) + 1)
    budget = source
    if override is not None:
        override = int(override)
        if plan.new_candidate < 0:
            raise TruncateError("--candidate-top-k-blocks: the truncated model keeps no candidate source layer, so no "
                                "loader reads the budget")
        if min_blocks is None:
            raise TruncateError("--candidate-top-k-blocks needs deepseek41.attention.indexer.top_k and a positive "
                                "candidate_block_size to check the budget")
        if not min_blocks <= override <= 2**31 - 1:
            raise TruncateError(
                f"--candidate-top-k-blocks {override} can leave {max(0, (override - 1) * block + 1)} candidate rows, "
                f"fewer than indexer.top_k {index_topk}: the indexer would then select pruned rows (the released "
                f"checkpoint never does), and ggml_cuda turns its gathered decode off (dsv41_use_gather). "
                f"Use {min_blocks} or more")
        budget = override
    new_candidate = plan.new_candidate if plan.new_candidate >= 0 else None
    # V4.1 ratio code r compresses r tokens per row (dsv4_acquire_graph: cr = 2, hr = 1), so the candidate
    # source holds (pos + 1) // r rows and pruning first engages at token number (budget * block + 1) * r.
    ratio = new_ratios[new_candidate] if new_candidate is not None else None
    engages = (budget * block + 1) * ratio if ratio else None
    result = dict(source=source, new=budget, original=original, overridden=budget != source, block_size=block,
                  indexer_top_k=index_topk, min_blocks=min_blocks, candidate_layer_new=new_candidate,
                  candidate_ratio_code=ratio, prunes_from_token=engages)
    if engages is None:
        plan.notes.append("no candidate source layer is kept: candidate pruning is off in the truncated model")
    elif result["overridden"]:
        plan.notes.append(f"candidate budget {source} -> {budget} blocks x {block} rows: pruning engages from token "
                          f"{engages} at candidate source layer {new_candidate} (ratio code {ratio}); the original "
                          f"budget {original} is recorded in {origin_key}")
    else:
        hint = (f"; --candidate-top-k-blocks {min_blocks} engages it from token {(min_blocks * block + 1) * ratio}"
                if min_blocks is not None and min_blocks < budget else "")
        plan.notes.append(f"candidate pruning engages only from token {engages} ({budget} blocks x {block} rows at "
                          f"candidate source layer {new_candidate}, ratio code {ratio}); shorter contexts run the "
                          f"candidate path with every block kept{hint}")
    return result


def make_plan(src: SourceModel, kept, header_only=False, layout_json=None, candidate_top_k_blocks=None) -> Plan:
    """Classify every metadata key and tensor of src for keeping the given layers.

    header_only plans work from metadata alone (shards may be missing); tensor
    ownership then comes from the metadata source lists. candidate_top_k_blocks
    rewrites the candidate-pruning budget (see the module docstring).
    """
    kvs = src.kvs
    plan = Plan()
    arch_kv = kvs.get("general.architecture")
    if arch_kv is None or arch_kv.value != "deepseek41":
        raise TruncateError("this tool writes DeepSeek V4.1 (general.architecture = deepseek41) files only")
    A = "deepseek41."
    if A + "block_count" not in kvs:
        raise TruncateError("deepseek41.block_count is missing")
    n_layer = _int(kvs[A + "block_count"], A + "block_count")
    kept = sorted(set(int(k) for k in kept))
    if not kept or kept[0] < 0 or kept[-1] >= n_layer:
        raise TruncateError(f"kept layers {kept} must lie in [0, {n_layer})")
    new_of = {old: new for new, old in enumerate(kept)}
    plan.kept, plan.new_of, plan.n_layer = kept, new_of, n_layer
    use_tensors = src.complete and not header_only
    plan.header_only = not use_tensors
    shapes = {name: t.shape for name, t in src.tensors.items()} if use_tensors else None

    # --- compress ratios -----------------------------------------------------
    ratios_kv = kvs.get(A + "attention.compress_ratios")
    if ratios_kv is None or not ratios_kv.is_int_array() or not n_layer <= len(ratios_kv.value) <= n_layer + MAX_RATIO_TAIL:
        raise TruncateError(f"deepseek41.attention.compress_ratios must hold block_count..block_count+{MAX_RATIO_TAIL} integers")
    ratios = [int(r) for r in ratios_kv.value[:n_layer]]

    # --- shared-cache owners -------------------------------------------------
    meta_kv = [int(v) for v in kvs[A + "attention.kv_source_layers"].value] if A + "attention.kv_source_layers" in kvs else None
    meta_index = [int(v) for v in kvs[A + "attention.indexer.source_layers"].value] if A + "attention.indexer.source_layers" in kvs else None
    if use_tensors:
        kv_owners = [i for i in range(n_layer) if f"blk.{i}.attn_compressor_kv.weight" in src.tensors]
        index_owners = [i for i in range(n_layer) if f"blk.{i}.indexer.attn_q_b.weight" in src.tensors]
        for key, meta, found in (("attention.kv_source_layers", meta_kv, kv_owners),
                                 ("attention.indexer.source_layers", meta_index, index_owners)):
            if meta is not None and meta != found:
                raise TruncateError(f"deepseek41.{key} = {meta} disagrees with the tensors, which put owners at {found}")
    else:
        if meta_kv is None or meta_index is None:
            raise TruncateError("a header-only plan needs deepseek41.attention.kv_source_layers and "
                                "deepseek41.attention.indexer.source_layers (tensor shards are missing)")
        kv_owners, index_owners = meta_kv, meta_index
    plan.kv_owners, plan.index_owners = kv_owners, index_owners

    # --- topology: kept layers must keep the layers whose state they read ------
    problems, per_layer = [], []
    for layer in kept:
        ratio = ratios[layer]
        kv_src = source_of(kv_owners, layer) if ratio else None
        index_src = source_of(index_owners, layer) if ratio else None
        if ratio and (kv_src is None or index_src is None):
            raise TruncateError(f"source model is inconsistent: layer {layer} has ratio {ratio} but no earlier kv/index owner")
        if ratio and kv_src not in new_of:
            problems.append(f"layer {layer} (compress ratio {ratio}) reads the shared compressed KV cache owned by "
                            f"layer {kv_src}, which is dropped")
        if ratio and index_src not in new_of:
            problems.append(f"layer {layer} (compress ratio {ratio}) reads the lightning-indexer keys owned by "
                            f"layer {index_src}, which is dropped")
        per_layer.append(dict(new=new_of[layer], old=layer, ratio=ratio,
                              kv_source_old=kv_src, kv_source_new=new_of.get(kv_src),
                              index_source_old=index_src, index_source_new=new_of.get(index_src)))
    cand_key = A + "attention.indexer.candidate_source_layer"
    cand_default = next((i for i in index_owners if ratios[i] == 1), -1)
    candidate = int(kvs[cand_key].value) if cand_key in kvs else cand_default
    new_candidate = -1
    if candidate >= 0:
        if candidate in new_of:
            new_candidate = new_of[candidate]
        else:
            dependents = [r["old"] for r in per_layer if r["ratio"] and r["index_source_old"] > candidate]
            if dependents:
                problems.append(f"layers {dependents} index with the candidate block mask built by layer {candidate} "
                                "(deepseek41.attention.indexer.candidate_source_layer), which is dropped")
            else:
                plan.notes.append(f"candidate source layer {candidate} is dropped and no kept indexer runs after it: "
                                  "candidate pruning is disabled (-1), which is what the source does for these layers")
    new_kv_owners = [new_of[o] for o in kv_owners if o in new_of]
    new_index_owners = [new_of[o] for o in index_owners if o in new_of]
    if not new_kv_owners or not new_index_owners:
        problems.append("no kv-cache or indexer owner layer is kept; both V4.1 loaders require at least one of each")
    if problems:
        raise TruncateError("refusing to drop layers that kept layers depend on:\n  - " + "\n  - ".join(problems) +
                            f"\n  kv owners {kv_owners}, indexer owners {index_owners}, candidate source {candidate}; "
                            "keep those layers or drop their dependents")
    # Re-derive each kept layer's sources inside the new model exactly as the loaders do.
    new_ratios = [ratios[o] for o in kept]
    for row in per_layer:
        if row["ratio"]:
            assert source_of(new_kv_owners, row["new"]) == row["kv_source_new"]
            assert source_of(new_index_owners, row["new"]) == row["index_source_new"]
    plan.per_layer, plan.candidate, plan.new_candidate = per_layer, candidate, new_candidate
    plan.new_kv_owners, plan.new_index_owners = new_kv_owners, new_index_owners
    plan.candidate_budget = _candidate_budget(kvs, plan, new_ratios, candidate_top_k_blocks)

    # --- Engram ----------------------------------------------------------------
    has_ple, has_ts = A + "ple.layers" in kvs, A + "engram.layer_ids" in kvs
    if has_ple and has_ts:
        raise TruncateError("the source carries both deepseek41.ple.* and deepseek41.engram.layer_ids Engram metadata")
    if not has_ple and not has_ts:
        raise TruncateError("the source has no Engram layer list (deepseek41.ple.layers or deepseek41.engram.layer_ids)")
    combined = has_ple
    table_rows, table = None, None
    if combined:
        if use_tensors:
            table = src.tensors.get(COMBINED_TABLE)
            if table is None:
                raise TruncateError(f"deepseek41.ple.* names {COMBINED_TABLE}, but no shard holds it")
            if len(table.shape) > 2 and any(d != 1 for d in table.shape[2:]):
                raise TruncateError(f"{COMBINED_TABLE} must be two-dimensional")
            table_rows = table.shape[1]
        elif layout_json:
            for name, tname, shape, _ in layout_json.get("special", []):
                if name == COMBINED_TABLE:
                    table_rows = int(shape[1])
        slices = combined_engram_slices(kvs, table_rows)
        engram_ids = [s["id"] for s in slices]
    else:
        engram_ids = _layer_ids(kvs[A + "engram.layer_ids"], A + "engram.layer_ids", n_layer)
        slices = None
    kept_engram = [i for i, layer in enumerate(engram_ids) if layer in new_of]
    if not kept_engram:
        raise TruncateError(f"no Engram layer is kept (Engram layers are {engram_ids}); both V4.1 loaders require at "
                            "least one, so keep one of them")
    ngram = _int(kvs[A + ("ple.ngram_size" if combined else "engram.max_ngram_size")], "ngram")
    heads = _int(kvs[A + ("ple.heads_per_ngram" if combined else "engram.head_count")], "heads")
    columns = (ngram - 1) * heads
    plan.engram = dict(combined=combined, source_layers=engram_ids, kept_layers=[engram_ids[i] for i in kept_engram],
                       new_layers=[new_of[engram_ids[i]] for i in kept_engram], ngram=ngram, heads=heads, columns=columns)
    new_first, rebased = 0, {}
    if combined:
        plan.engram["slices"] = []
        for i in kept_engram:
            s = slices[i]
            rebased[i] = new_first - s["first_row"]
            plan.engram["slices"].append(dict(layer_old=s["id"], layer_new=new_of[s["id"]], source_first_row=s["first_row"],
                                              new_first_row=new_first, rows=s["rows"]))
            new_first += s["rows"]
        plan.engram["source_rows"] = table_rows
        plan.engram["new_rows"] = new_first

    # --- DSpark target contract --------------------------------------------------
    dspark_targets = None
    dspark_kept = True
    if A + "dspark.target_layers" in kvs:
        dspark_targets = _layer_ids(kvs[A + "dspark.target_layers"], A + "dspark.target_layers", n_layer)
        dspark_kept = all(t in new_of for t in dspark_targets)
        if not dspark_kept:
            plan.notes.append(f"DSpark target layers {dspark_targets} are not all kept: every deepseek41.dspark.* key is "
                              "removed, so no drafter can attach to the truncated model")

    # --- metadata ------------------------------------------------------------------
    for key, kv in kvs.items():
        if key.startswith("split."):
            _change(plan, key, "removed", kv.value, None, "single-file output")
            continue
        if key.startswith(PROVENANCE):
            _change(plan, key, "replaced", kv.value, None, "provenance is rewritten below")
            continue
        if key.startswith(COPY_NAMESPACES):
            plan.kvs.append(kv)
            continue
        if not key.startswith(A):
            reason = looks_per_layer(key, kv, n_layer)
            if reason:
                raise TruncateError(f"metadata key {key} ({kv.type_name()}) looks per-layer ({reason}) and this tool does "
                                    "not know how to rewrite it; classify it in dsv41-truncate-gguf.py before truncating")
            plan.kvs.append(kv)
            plan.unclassified.append(dict(key=key, type=kv.type_name()))
            continue
        sub = key[len(A):]
        new = None
        if sub == "block_count":
            new = Kv(key, kv.vtype, len(kept))
            _change(plan, key, "renumber", kv.value, new.value)
        elif sub in PER_LAYER_VALUES or (sub in SCALAR_OR_PER_LAYER and kv.vtype == ARRAY):
            if kv.vtype != ARRAY or len(kv.value) != n_layer:
                raise TruncateError(f"{key} is expected to hold one value per layer ({n_layer}), found "
                                    f"{kv.type_name()} of {len(kv.value) if kv.vtype == ARRAY else 1}")
            new = Kv(key, ARRAY, [kv.value[o] for o in kept], kv.atype)
            _change(plan, key, "filter", kv.value, new.value)
        elif sub in PER_LAYER_WITH_TAIL:
            tail = kv.value[n_layer:]
            new = Kv(key, ARRAY, [kv.value[o] for o in kept] + list(tail), kv.atype)
            _change(plan, key, "filter", kv.value, new.value,
                    f"{len(tail)} trailing non-trunk entries kept as-is" if tail else "")
        elif sub in LAYER_ID_LISTS:
            role = LAYER_ID_LISTS[sub]
            ids = _layer_ids(kv, key, n_layer)
            if role == "dspark":
                if not dspark_kept:
                    _change(plan, key, "removed", kv.value, None, "DSpark target layers dropped")
                    continue
                values = [new_of[i] for i in ids]
            elif role == "engram":
                values = [new_of[engram_ids[i]] for i in kept_engram]
            elif role == "kv":
                values = new_kv_owners
            else:
                values = new_index_owners
            new = Kv(key, ARRAY, values, kv.atype)
            _change(plan, key, "renumber", kv.value, new.value)
        elif sub in LAYER_ID_SCALARS:
            vtype = kv.vtype
            if new_candidate < 0 and vtype not in (INT32, INT64):
                vtype = INT32
            new = Kv(key, vtype, new_candidate)
            _change(plan, key, "renumber", kv.value, new.value,
                    "type changed to INT32 to hold -1" if vtype != kv.vtype else "")
        elif sub in ENGRAM_MAJOR:
            width = ngram if ENGRAM_MAJOR[sub] == "ngram" else columns
            if not kv.is_int_array() or len(kv.value) != len(engram_ids) * width:
                raise TruncateError(f"{key} must hold {len(engram_ids)} x {width} integers")
            values = []
            for i in kept_engram:
                chunk = list(kv.value[i * width:(i + 1) * width])
                if sub == "ple.head_offsets":
                    chunk = [v + rebased[i] for v in chunk]
                values.extend(chunk)
            new = Kv(key, ARRAY, values, kv.atype)
            _change(plan, key, "filter+rebase" if sub == "ple.head_offsets" else "filter", kv.value, new.value)
        elif sub == "hash_layer_count":
            count = _int(kv, key)
            new = Kv(key, kv.vtype, sum(1 for o in kept if o < count))
            _change(plan, key, "recount", kv.value, new.value, "kept layers below the source count")
        elif sub == "nextn_predict_layers":
            if _int(kv, key) != 0:
                raise TruncateError(f"{key} = {kv.value}: MTP/NextN blocks are not supported by this tool")
            plan.kvs.append(kv)
            continue
        elif sub.startswith("dspark."):
            if not dspark_kept:
                _change(plan, key, "removed", kv.value, None, "DSpark target layers dropped")
                continue
            if sub not in DSPARK_SCALARS:
                reason = looks_per_layer(key, kv, n_layer)
                if reason:
                    raise TruncateError(f"metadata key {key} looks per-layer ({reason}) and is not understood")
                plan.unclassified.append(dict(key=key, type=kv.type_name()))
            plan.kvs.append(kv)
            continue
        elif key in (CANDIDATE_BLOCKS_KEY, CANDIDATE_TOPK_KEY) and plan.candidate_budget["overridden"]:
            new = Kv(key, kv.vtype, plan.candidate_budget["new"])
            _change(plan, key, "override", kv.value, new.value,
                    f"--candidate-top-k-blocks: pruning from token {plan.candidate_budget['prunes_from_token']}")
        elif sub in GLOBAL_KEYS or sub in SCALAR_OR_PER_LAYER:
            plan.kvs.append(kv)
            continue
        else:
            reason = looks_per_layer(key, kv, n_layer)
            if reason:
                raise TruncateError(f"metadata key {key} ({kv.type_name()}) looks per-layer ({reason}) and this tool does "
                                    "not know how to rewrite it; classify it in dsv41-truncate-gguf.py before truncating")
            plan.kvs.append(kv)
            plan.unclassified.append(dict(key=key, type=kv.type_name()))
            continue
        plan.kvs.append(new)
    if cand_key not in kvs:
        derived = next((i for i in new_index_owners if new_ratios[i] == 1), -1)
        if derived != new_candidate:
            plan.kvs.append(Kv(cand_key, INT32, new_candidate))
            _change(plan, cand_key, "added", None, new_candidate, "the loaders' default would pick a different layer")
    budget = plan.candidate_budget
    if budget["overridden"] and CANDIDATE_BLOCKS_KEY not in kvs and CANDIDATE_TOPK_KEY not in kvs:
        plan.kvs.append(Kv(CANDIDATE_BLOCKS_KEY, UINT32, budget["new"]))
        _change(plan, CANDIDATE_BLOCKS_KEY, "added", None, budget["new"],
                "--candidate-top-k-blocks (the source relied on the loaders' default 2048)")
    if plan.new_candidate >= 0 and CANDIDATE_TOPK_KEY not in kvs and budget["new"] != 2048:
        # The PR #306-era loaders (the ggml_cuda reference build) read only this spelling and fall
        # back to 2048; without it they keep every block while the current loaders prune.
        plan.kvs.append(Kv(CANDIDATE_TOPK_KEY, UINT32, budget["new"]))
        _change(plan, CANDIDATE_TOPK_KEY, "added", None, budget["new"],
                "PR #306-era loaders read only candidate_top_k (default 2048); written so every loader "
                "generation reads the same budget")
    plan.candidate_override = budget["new"] if budget["overridden"] else None

    # --- provenance ------------------------------------------------------------
    origin = kept
    if PROVENANCE + "origin_layers" in kvs:
        earlier = [int(v) for v in kvs[PROVENANCE + "origin_layers"].value]
        if len(earlier) != n_layer:
            raise TruncateError(f"{PROVENANCE}origin_layers does not match block_count")
        origin = [earlier[o] for o in kept]
    provenance = [(PROVENANCE + "kept_layers", ARRAY, kept, UINT32),
                  (PROVENANCE + "origin_layers", ARRAY, origin, UINT32),
                  (PROVENANCE + "source_block_count", UINT32, n_layer, None),
                  (PROVENANCE + "tool", STRING, TOOL_VERSION, None)]
    # The original model's budget travels with every descendant whose budget differs;
    # candidate_top_k_blocks names the override this run applied (--verify replays it).
    if budget["overridden"] or PROVENANCE + "candidate_top_k_blocks_source" in kvs:
        provenance.append((PROVENANCE + "candidate_top_k_blocks_source", UINT32, budget["original"], None))
    if budget["overridden"]:
        provenance.append((PROVENANCE + "candidate_top_k_blocks", UINT32, budget["new"], None))
    for key, vtype, value, atype in provenance:
        plan.kvs.append(Kv(key, vtype, value, atype))
        _change(plan, key, "added", None, value, "provenance (immediate source layer ids / original ids)")
    alignment = kvs["general.alignment"].value if "general.alignment" in kvs else 32
    plan.alignment = alignment

    # --- tensors -----------------------------------------------------------------
    plan.tensor_bytes = None
    if use_tensors:
        layer_tensors = {o: 0 for o in kept}
        for name, tensor in src.tensors.items():
            match = BLK_RE.match(name)
            if match:
                old = int(match[1])
                if old >= n_layer:
                    raise TruncateError(f"tensor {name} belongs to block {old} >= block_count {n_layer}; "
                                        "MTP/NextN blocks are not supported")
                if old not in new_of:
                    continue
                layer_tensors[old] += 1
                plan.tensors.append(dict(name=f"blk.{new_of[old]}.{match[2]}", source=name, type=tensor.type,
                                         shape=list(tensor.shape), nbytes=tensor.nbytes, layer_old=old,
                                         segments=[(tensor.shard, tensor.abs_offset, tensor.nbytes)]))
            elif name == COMBINED_TABLE:
                if not combined:
                    raise TruncateError(f"{COMBINED_TABLE} is present but the Engram metadata is TensorSharp's per-layer layout")
                row_bytes = tensor_nbytes(tensor.type, tensor.shape[:1])
                segments = [(tensor.shard, tensor.abs_offset + s["source_first_row"] * row_bytes, s["rows"] * row_bytes)
                            for s in plan.engram["slices"]]
                plan.engram["row_bytes"] = row_bytes
                plan.tensors.append(dict(name=name, source=name, type=tensor.type, shape=[tensor.shape[0], new_first],
                                         nbytes=new_first * row_bytes, layer_old=None, segments=segments))
            elif name in ROOT_TENSORS:
                plan.tensors.append(dict(name=name, source=name, type=tensor.type, shape=list(tensor.shape),
                                         nbytes=tensor.nbytes, layer_old=None,
                                         segments=[(tensor.shard, tensor.abs_offset, tensor.nbytes)]))
            else:
                raise TruncateError(f"non-layer tensor {name} is not known to be layer-independent; add it to "
                                    "ROOT_TENSORS once you have checked it carries no per-layer data")
        empty = [o for o, count in layer_tensors.items() if not count]
        if empty:
            raise TruncateError(f"kept layers {empty} have no tensors in the source")
        absent = [n for n in REQUIRED_ROOTS if n not in src.tensors]
        if absent:
            raise TruncateError(f"the source lacks {absent}")
        if not combined:
            listed = {f"blk.{i}.engram_embd.weight" for i in engram_ids}
            present = {n for n in src.tensors if n.endswith(".engram_embd.weight")}
            if listed != present:
                raise TruncateError("deepseek41.engram.layer_ids does not match the blk.N.engram_embd.weight tensors")
        plan.tensor_bytes = sum(t["nbytes"] for t in plan.tensors)
    plan.estimate = _estimate(plan, layout_json) if layout_json else None
    return plan


def _estimate(plan, layout):
    """Byte estimate from model-layout-metadata.json when tensor shards are absent."""
    layer_bytes = layout.get("layer_bytes", {})
    special = {name: (tname, shape, nbytes) for name, tname, shape, nbytes in layout.get("special", [])}
    layers = {o: int(layer_bytes[str(o)]) for o in plan.kept if str(o) in layer_bytes}
    roots = {n: int(special[n][2]) for n in REQUIRED_ROOTS if n in special}
    engram = 0
    if plan.engram["combined"] and COMBINED_TABLE in special:
        tname, shape, _ = special[COMBINED_TABLE]
        engram = plan.engram["new_rows"] * tensor_nbytes(GGML_TYPE_IDS[tname], [int(shape[0])])
    total = sum(layers.values()) + sum(roots.values()) + engram
    return dict(layer_bytes={str(k): v for k, v in layers.items()}, kept_layer_bytes=sum(layers.values()),
                root_bytes=roots, engram_slice_bytes=engram, total_bytes=total,
                total_gib=round(total / 2**30, 3), weights_gib=round((total - engram) / 2**30, 3),
                engram_gib=round(engram / 2**30, 3), source="model-layout-metadata.json (estimate; shards absent)")


def planned_values(plan) -> dict:
    return {kv.key: kv for kv in plan.kvs}


def planned_shapes(plan):
    if plan.header_only:
        return None
    return {t["name"]: t["shape"] for t in plan.tensors}


# ---------------------------------------------------------------------------
# Output encoding, copy, verify
# ---------------------------------------------------------------------------

def encode_header(plan) -> tuple[bytes, int]:
    parts = [b"GGUF", struct.pack("<IQQ", 3, len(plan.tensors), len(plan.kvs))]
    for kv in plan.kvs:
        parts.append(_encode_string(kv.key))
        parts.append(kv.encoded())
    offset = 0
    for tensor in plan.tensors:
        tensor["offset"] = offset
        shape = tensor["shape"]
        parts.append(_encode_string(tensor["name"]) + struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape,
                                                                   tensor["type"], offset))
        offset = align(offset + tensor["nbytes"], plan.alignment)
    header = b"".join(parts)
    header += b"\0" * (align(len(header), plan.alignment) - len(header))
    return header, offset


class _Files:
    """Read handles opened once per path and closed together (no mmap)."""

    def __init__(self):
        self.handles = {}

    def read(self, path, offset, length):
        handle = self.handles.get(path)
        if handle is None:
            handle = self.handles[path] = open(path, "rb")
        handle.seek(offset)
        data = handle.read(length)
        if len(data) != length:
            raise IOError(f"short read from {path} at {offset}: wanted {length}, got {len(data)}")
        return data

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _stream_segments(src_paths, segments, chunk, sink):
    """Read segments with a background thread; sink(bytes) hashes and writes."""
    chunks: queue.Queue = queue.Queue(maxsize=4)
    stop = threading.Event()

    def produce():
        try:
            with _Files() as files:
                for shard, offset, length in segments:
                    done = 0
                    while done < length and not stop.is_set():
                        size = min(chunk, length - done)
                        data = files.read(src_paths[shard], offset + done, size)
                        while not stop.is_set():
                            try:
                                chunks.put(data, timeout=0.5)
                                break
                            except queue.Full:
                                continue
                        done += size
            if not stop.is_set():
                chunks.put(None)
        except BaseException as error:  # handed to the consumer
            if not stop.is_set():
                chunks.put(error)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = chunks.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            sink(item)
    finally:
        stop.set()
        thread.join(timeout=5)


def _sample_positions(nbytes, block, samples, seed, name, extra=()):
    if nbytes == 0:
        return []
    block = min(block, nbytes)
    rng = random.Random(f"{seed}:{name}")
    positions = {0, nbytes - block}
    for _ in range(samples):
        positions.add(rng.randrange(0, nbytes - block + 1))
    for p in extra:
        positions.add(min(max(0, p), nbytes - block))
    return sorted(positions)


def _read_mapped(files, src_paths, segments, position, length):
    """Bytes [position, position+length) of the concatenation of segments."""
    out, base = [], 0
    for shard, offset, size in segments:
        lo, hi = max(position, base), min(position + length, base + size)
        if lo < hi:
            out.append(files.read(src_paths[shard], offset + lo - base, hi - lo))
        base += size
    return b"".join(out)


def verify_output(src: SourceModel, out_path: Path, kept=None, samples=6, sample_bytes=1 << 20, full=False,
                  seed=0, engram_sequences=8, chunk=64 << 20, log=None, candidate_top_k_blocks=None) -> dict:
    """Re-open out_path and check it against the source; returns a JSON-able report.

    kept and candidate_top_k_blocks default to what the output's tensorsharp.truncate.*
    provenance says this tool applied."""
    out = GgufHeader(out_path)
    if kept is None:
        kv = out.kvs.get(PROVENANCE + "kept_layers")
        if kv is None:
            raise TruncateError(f"{out_path} has no {PROVENANCE}kept_layers; pass --layers")
        kept = [int(v) for v in kv.value]
    if candidate_top_k_blocks is None and PROVENANCE + "candidate_top_k_blocks" in out.kvs:
        candidate_top_k_blocks = int(out.kvs[PROVENANCE + "candidate_top_k_blocks"].value)
    plan = make_plan(src, kept, candidate_top_k_blocks=candidate_top_k_blocks)
    header, _ = encode_header(plan)
    checks = []

    def record(name, ok, detail):
        checks.append(dict(check=name, status="pass" if ok else "FAIL", detail=detail))

    # metadata: same keys, order, types and bytes as the plan derives from the source
    expected = [(kv.key, kv.encoded()) for kv in plan.kvs]
    actual = [(key, kv.raw) for key, kv in out.kvs.items()]
    diff = [k for (k, v), (k2, v2) in zip(expected, actual) if k != k2 or v != v2]
    record("metadata equals the plan derived from the source", expected == actual,
           dict(keys=len(actual), expected_keys=len(expected), differing=diff[:20]))
    # tensor table
    want = [(t["name"], t["type"], t["shape"], t["offset"]) for t in plan.tensors]
    have = [(t.name, t.type, t.shape, t.offset) for t in out.tensors]
    record("tensor table equals the plan (names, types, shapes, offsets)", want == have,
           dict(tensors=len(have), expected=len(want),
                differing=[w[0] for w, h in zip(want, have) if w != h][:20]))
    record("header bytes equal the planned header", out.data_offset == len(header),
           dict(data_offset=out.data_offset, planned=len(header)))
    # The plan above is rebuilt by the same encoder that wrote the file, so it cannot
    # catch an encoder bug; this is ggml's own rule, evaluated on the file alone.
    layout = ggml_gguf_layout(out)
    record("ggml gguf_init_from_file accepts the tensor table (offsets = running padded sizes)",
           not layout["problems"], layout)
    required = out.required_size()
    record("file holds every tensor and ends at the data (no surplus)",
           required <= out.file_size <= align(required, out.alignment),
           dict(file_size=out.file_size, required=required))
    record("no split.* keys (single-file output)", not any(k.startswith("split.") for k in out.kvs), None)
    # sampled (or full) byte comparison against the source
    by_name = {t.name: t for t in out.tensors}
    mismatches, compared, ranges = [], 0, 0
    src_paths = [str(p) for p in src.paths]
    with _Files() as files:
        for tensor in plan.tensors:
            out_tensor = by_name.get(tensor["name"])
            if out_tensor is None:
                mismatches.append(dict(tensor=tensor["name"], error="missing"))
                continue
            nbytes = tensor["nbytes"]
            if full:
                position = 0
                while position < nbytes:
                    size = min(chunk, nbytes - position)
                    a = files.read(str(out_path), out_tensor.abs_offset + position, size)
                    b = _read_mapped(files, src_paths, tensor["segments"], position, size)
                    compared += size
                    ranges += 1
                    if a != b:
                        mismatches.append(dict(tensor=tensor["name"], position=position, length=size))
                        break
                    position += size
                continue
            bounds, base = [], 0
            for _, _, size in tensor["segments"][:-1]:
                base += size
                bounds += [base, base - sample_bytes]
            for position in _sample_positions(nbytes, sample_bytes, samples, seed, tensor["name"], bounds):
                size = min(sample_bytes, nbytes - position)
                a = hashlib.sha256(files.read(str(out_path), out_tensor.abs_offset + position, size)).hexdigest()
                b = hashlib.sha256(_read_mapped(files, src_paths, tensor["segments"], position, size)).hexdigest()
                compared += size
                ranges += 1
                if a != b:
                    mismatches.append(dict(tensor=tensor["name"], source=tensor["source"], position=position,
                                           length=size, output_sha256=a, source_sha256=b))
        record(("full" if full else "sampled") + " tensor bytes equal the source bytes", not mismatches,
               dict(tensors=len(plan.tensors), ranges=ranges, bytes_compared=compared, mismatches=mismatches[:20],
                    sample_bytes=None if full else sample_bytes, random_samples_per_tensor=None if full else samples))
        # loader contract on the written header
        out_values = out.kvs
        out_shapes = {t.name: t.shape for t in out.tensors}
        contract = loader_contract(out_values, out_shapes)
        for item in contract:
            checks.append(dict(check="loader: " + item["check"], status=item["status"], detail=item["detail"]))
        # Engram: same rows and bytes as the full model for every kept Engram layer
        try:
            checks.append(_engram_equivalence(src, plan, out, files, engram_sequences, seed))
        except TruncateError as error:
            record("Engram rows hash to the same table bytes as the source", False, str(error))
    status = "pass" if all(c["status"] == "pass" for c in checks) else "FAIL"
    return dict(tool=TOOL_VERSION, mode="verify", status=status, output=str(out_path), output_bytes=out.file_size,
                kept_layers=kept, layer_map={str(o): n for o, n in plan.new_of.items()},
                candidate_budget=plan.candidate_budget, checks=checks, source_files=src.files())


def _engram_equivalence(src, plan, out, files, sequences, seed, token_sequences=None):
    """Hash token sequences through the source's and the output's Engram metadata and
    compare every distinct row they reach, at every position and hash column, byte for
    byte. token_sequences replaces the seeded random sequences (tests)."""
    src_shapes = {name: t.shape for name, t in src.tensors.items()}
    src_layout = engram_layout(src.kvs, src_shapes)
    out_layout = engram_layout(out.kvs, {t.name: t.shape for t in out.tensors})
    rng = random.Random(f"{seed}:engram")
    src_paths = [str(p) for p in src.paths]
    out_tensors = {t.name: t for t in out.tensors}
    compared_rows, details = 0, []
    for layer in src_layout["layers"]:
        if layer["id"] not in plan.new_of:
            continue
        new_id = plan.new_of[layer["id"]]
        mirror = next((l for l in out_layout["layers"] if l["id"] == new_id), None)
        if mirror is None:
            raise TruncateError(f"kept Engram layer {layer['id']} (new {new_id}) is missing from the output")
        for key in ("multipliers", "primes", "offsets", "rows"):
            if layer[key] != mirror[key]:
                raise TruncateError(f"Engram layer {layer['id']}: {key} differ after truncation")
        if src_layout["combined"]:
            src_table = src.tensors[COMBINED_TABLE]
            out_table = out_tensors[COMBINED_TABLE]
        else:
            src_table = src.tensors[f"blk.{layer['id']}.engram_embd.weight"]
            out_table = out_tensors[f"blk.{new_id}.engram_embd.weight"]
        row_bytes = tensor_nbytes(src_table.type, src_table.shape[:1])
        vocab = src_layout["vocab"]
        if token_sequences is None:
            batch = []
            for s in range(sequences):
                length = rng.randint(1, 12)
                tokens = [rng.randrange(vocab) for _ in range(length)]
                if s % 3 == 2:
                    tokens[rng.randrange(length)] = -1          # image position blocks lookbacks
                batch.append(tokens)
        else:
            batch = [list(tokens) for tokens in token_sequences]
        for tokens in batch:
            rows_src = engram_hash_rows(src_layout, layer, tokens)
            rows_out = engram_hash_rows(out_layout, mirror, tokens)
            if rows_src != rows_out:
                raise TruncateError(f"Engram layer {layer['id']}: hashed rows differ for tokens {tokens}")
            # Every distinct row, not a prefix: the list is [position][column], so any
            # prefix shorter than a position's columns only ever reads position 0.
            for row in sorted(set(rows_src)):
                a = files.read(src_paths[src_table.shard], src_table.abs_offset + (layer["first_row"] + row) * row_bytes, row_bytes)
                b = files.read(str(out.path), out_table.abs_offset + (mirror["first_row"] + row) * row_bytes, row_bytes)
                if a != b:
                    raise TruncateError(f"Engram layer {layer['id']}: row {row} bytes differ")
                compared_rows += 1
        details.append(dict(layer_old=layer["id"], layer_new=new_id, source_first_row=layer["first_row"],
                            new_first_row=mirror["first_row"], rows=layer["rows"]))
    return dict(check="Engram rows hash to the same table bytes as the source", status="pass",
                detail=dict(layers=details, sequences_per_layer=len(batch) if details else 0,
                            rows_compared=compared_rows, rows="every distinct row each sequence hashes to"))


_BYTE_UNITS = {"": 1, "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
               "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}


def byte_count(text: str) -> int:
    """A positive byte count: an integer, optionally with B, KB/MB/GB/TB (powers of 10) or
    KiB/MiB/GiB/TiB (powers of 2). A bare K/M/G/T is refused: '500G' is exactly the
    GB-or-GiB ambiguity that hid 37 GB of the pod quota."""
    match = re.fullmatch(r"\s*(\d+)\s*([A-Za-z]*)\s*", str(text))
    if not match or match[2].lower() not in _BYTE_UNITS or int(match[1]) <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive byte count (an integer, optionally with "
                                         "B, KB, MB, GB, TB or KiB, MiB, GiB, TiB)")
    return int(match[1]) * _BYTE_UNITS[match[2].lower()]


def _allocated_bytes(st):
    """st_blocks x 512 (du's default measure), or None where stat has no st_blocks (Windows)."""
    blocks = getattr(st, "st_blocks", None)
    return None if blocks is None else int(blocks) * 512


def tree_usage(root) -> dict:
    """du-style usage of the tree under root, staying on root's filesystem (du -x).

    Every file, directory and symlink is counted once, hard links once per inode (as du
    does), and symlinks are not followed. Two totals are kept: apparent bytes (st_size,
    what `du -sb` prints) and allocated bytes (st_blocks x 512, du's default). The pod
    quota on a MooseFS/FUSE volume cannot be read from inside the pod, and MooseFS can
    enforce either a 'length' limit (apparent bytes) or a 'size' limit (allocated,
    rounded up to its 64 KiB blocks). used_bytes is therefore the larger of the two
    totals, so the check holds under either rule: sparse files make the allocated total
    smaller and block rounding makes it larger, and neither can let the check pass when
    the stricter rule would fail. (A 'realsize' limit, size x replica goal, is not
    visible from inside the pod.) A directory that cannot be listed or an entry that
    cannot be stat'ed is an error, because skipping it would undercount; an entry that
    vanishes during the walk is skipped, as du does."""
    root = os.path.abspath(root)
    try:
        top = os.lstat(root)
    except OSError as error:
        raise TruncateError(f"--quota-root {root}: {error.strerror or error}")
    if not _stat.S_ISDIR(top.st_mode):
        raise TruncateError(f"--quota-root {root} is not a directory")
    started = time.time()
    apparent = allocated = entries = linked = 0
    allocated_known = True
    seen, other_filesystems = set(), []

    def add(st):
        nonlocal apparent, allocated, entries, linked, allocated_known
        if not _stat.S_ISDIR(st.st_mode) and st.st_nlink > 1:
            inode = (st.st_dev, st.st_ino)
            if inode in seen:
                linked += 1
                return
            seen.add(inode)
        apparent += st.st_size
        size = _allocated_bytes(st)
        if size is None:
            allocated_known = False
        else:
            allocated += size
        entries += 1

    add(top)
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as listing:
                names = [entry.path for entry in listing]
        except OSError as error:
            raise TruncateError(f"cannot list {directory} ({error.strerror or error}); the quota check would "
                                "undercount, so it refuses")
        for path in names:
            try:
                st = os.lstat(path)          # DirEntry.stat() leaves st_ino/st_nlink zero on Windows
            except FileNotFoundError:
                continue
            except OSError as error:
                raise TruncateError(f"cannot stat {path} ({error.strerror or error}); the quota check would "
                                    "undercount, so it refuses")
            reparse = getattr(st, "st_file_attributes", 0) & getattr(_stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if _stat.S_ISDIR(st.st_mode) and not reparse:
                if st.st_dev != top.st_dev:
                    other_filesystems.append(path)       # another mount: not charged to this quota
                    continue
                pending.append(path)
            add(st)
    used = max(apparent, allocated) if allocated_known else apparent
    if not allocated_known:
        basis = "apparent bytes (this platform's stat has no st_blocks)"
    elif allocated >= apparent:
        basis = "allocated bytes (st_blocks x 512), the larger of allocated and apparent"
    else:
        basis = "apparent bytes (st_size), the larger of apparent and allocated"
    return dict(root=root, used_bytes=used, basis=basis, apparent_bytes=apparent,
                allocated_bytes=allocated if allocated_known else None, entries=entries,
                hard_links_counted_once=linked, other_filesystems_skipped=other_filesystems[:20],
                walk_seconds=round(time.time() - started, 2))


def check_quota(quota_bytes: int, root, out_path, planned_bytes: int, reserve: int) -> dict:
    """Usage under root plus the planned output and the reserve, against quota_bytes.
    Returns a report whose status is "pass" or "FAIL"; a root that does not contain the
    output, or a tree that cannot be walked completely, raises TruncateError."""
    root_real = os.path.realpath(root)
    out_real = os.path.realpath(out_path)
    try:
        inside = os.path.commonpath([root_real, out_real]) == root_real
    except ValueError:                      # different drives on Windows
        inside = False
    if not inside:
        raise TruncateError(f"--output {out_real} is not under --quota-root {root_real}; the quota there would "
                            "not see this write")
    usage = tree_usage(root_real)
    need = usage["used_bytes"] + planned_bytes + reserve
    return dict(status="pass" if need <= quota_bytes else "FAIL", quota_bytes=quota_bytes, **usage,
                planned_output_bytes=planned_bytes, reserve_bytes=reserve, needed_bytes=need,
                headroom_bytes=quota_bytes - need)


def quota_line(q: dict) -> str:
    return (f"quota check [{q['status']}]: {q['root']} uses {q['used_bytes']:,} B ({q['basis']}); "
            f"+ output {q['planned_output_bytes']:,} B + reserve {q['reserve_bytes']:,} B = {q['needed_bytes']:,} B "
            f"vs --quota-bytes {q['quota_bytes']:,} B (headroom after the write {q['headroom_bytes']:,} B)")


def _discard(paths, error, log):
    """Remove files this run created; never raises (the original error is what matters)."""
    for path in paths:
        try:
            os.unlink(path)
            log(f"[truncate] {type(error).__name__}: removed {path}")
        except FileNotFoundError:
            pass
        except OSError as failure:
            print(f"[truncate] WARNING could not remove {path} ({failure}); delete it by hand, it holds "
                  "space on the volume", file=sys.stderr)


def write_output(src: SourceModel, plan: Plan, out_path: Path, chunk=64 << 20, min_free=2 << 30,
                 samples=6, sample_bytes=1 << 20, seed=0, log=print, quota=None, keep_failed=False) -> dict:
    """Write <out>.partial, fsync, verify, then publish it and its manifest.

    Any failure before publishing (an OSError from write/close/fsync such as EDQUOT, a
    failed verification, KeyboardInterrupt, SystemExit from the CLI's SIGTERM handler)
    removes the files this run created and re-raises, so a ~100 GB partial never stays
    pinned on a quota-limited volume. keep_failed keeps a fully written and fsynced
    .partial whose verification or publish failed. Paths that exist before the run are
    refused and never removed. quota=(bytes, root) refuses before writing when the
    du-style usage of root plus the output and min_free exceeds bytes."""
    partial = out_path.with_name(out_path.name + ".partial")
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")
    manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
    for path in (out_path, partial, manifest_path, manifest_partial):
        if path.exists():
            raise TruncateError(f"{path} already exists; refusing to overwrite (remove a stale .partial yourself)")
    header, data_size = encode_header(plan)
    total = len(header) + data_size
    quota_report = None
    if quota is not None:
        quota_report = check_quota(quota[0], quota[1], out_path, total, min_free)
        log("[truncate] " + quota_line(quota_report))
        if quota_report["status"] != "pass":
            raise TruncateError(
                f"quota: {quota_report['root']} already uses {quota_report['used_bytes']:,} B "
                f"({quota_report['basis']}); the output needs {total:,} B plus a {min_free:,} B reserve "
                f"(--min-free-gib), {quota_report['needed_bytes']:,} B in all, which exceeds --quota-bytes "
                f"{quota_report['quota_bytes']:,} by {-quota_report['headroom_bytes']:,} B. Nothing was written.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_path.parent).free
    if free < total + min_free:
        raise TruncateError(f"{out_path.parent} has {free / 2**30:.1f} GiB free; the output needs "
                            f"{total / 2**30:.1f} GiB plus a {min_free / 2**30:.1f} GiB reserve")
    src_paths = [str(p) for p in src.paths]
    started = time.time()
    copied = [0, 0]

    def progress(n):
        copied[0] += n
        if copied[0] - copied[1] >= (4 << 30) or copied[0] == plan.tensor_bytes:
            copied[1] = copied[0]
            elapsed = max(time.time() - started, 1e-6)
            log(f"[truncate] {copied[0] / 2**30:8.2f} / {plan.tensor_bytes / 2**30:.2f} GiB  "
                f"{copied[0] / elapsed / 2**20:7.1f} MiB/s")

    whole = hashlib.sha256()         # the file's sha256, without a second pass over ~100 GB
    created = []                     # files this run created and has not published
    written = False                  # the .partial is complete, closed and fsynced
    try:
        stream = open(partial, "xb")   # exclusive: a .partial that appeared since the check is not ours
        created.append(partial)
        with stream:
            def put(data):
                whole.update(data)
                stream.write(data)
            put(header)
            for tensor in plan.tensors:
                here = stream.tell() - len(header)
                if here > tensor["offset"]:
                    raise AssertionError("tensor offsets overlap")
                put(b"\0" * (tensor["offset"] - here))
                digest = hashlib.sha256()

                def sink(data, digest=digest):
                    digest.update(data)
                    put(data)
                    progress(len(data))
                _stream_segments(src_paths, tensor["segments"], chunk, sink)
                tensor["sha256"] = digest.hexdigest()
            put(b"\0" * (len(header) + data_size - stream.tell()))
            stream.flush()
            os.fsync(stream.fileno())
        written = True
        copy_seconds = time.time() - started
        log(f"[truncate] wrote {partial} ({total} bytes) in {copy_seconds:.1f}s; verifying")
        report = verify_output(src, partial, plan.kept, samples=samples, sample_bytes=sample_bytes, seed=seed,
                               candidate_top_k_blocks=plan.candidate_override)
        if report["status"] != "pass":
            raise TruncateError(f"verification of {partial} failed:\n" +
                                json.dumps([c for c in report["checks"] if c["status"] != "pass"], indent=2))
        manifest = dict(tool=TOOL_VERSION, status="verified", output=str(out_path.resolve()), output_bytes=total,
                        output_sha256=whole.hexdigest(),
                        weights_id_note="output_sha256 equals sha256sum of the file; pass it as "
                                        "DeepSeek41DistributedProbe --weights-id so every probe names the lite file",
                        kept_layers=plan.kept, layer_map={str(o): n for o, n in plan.new_of.items()},
                        source_files=src.files(), metadata_changes=_jsonable(plan.changes), notes=plan.notes,
                        engram=plan.engram, candidate_budget=plan.candidate_budget,
                        copy_seconds=round(copy_seconds, 1), quota=quota_report,
                        tensors=[dict(name=t["name"], source=t["source"], type=GGML_TYPES[t["type"]][0],
                                      shape=t["shape"], bytes=t["nbytes"], offset=t["offset"], sha256=t["sha256"])
                                 for t in plan.tensors],
                        verification=report)
        stream = open(manifest_partial, "x", encoding="utf-8")
        created.append(manifest_partial)
        with stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        _publish(partial, out_path)
        created.remove(partial)
        _publish(manifest_partial, manifest_path)
        created.remove(manifest_partial)
    except BaseException as error:
        if keep_failed and written:
            log(f"[truncate] {type(error).__name__}: keeping {', '.join(map(str, created))} for inspection "
                "(--keep-failed-partial); delete it before retrying")
        else:
            _discard(created, error, log)
        raise
    log(f"[truncate] published {out_path}")
    return manifest


def _publish(temp: Path, final: Path):
    """Rename temp to final without replacing an existing final."""
    try:
        os.link(temp, final)
    except FileExistsError:
        raise TruncateError(f"{final} appeared while writing; it was not replaced")
    except OSError:
        if final.exists():
            raise TruncateError(f"{final} appeared while writing; it was not replaced")
        os.rename(temp, final)           # no hard links here; Windows refuses an existing target
        return
    os.unlink(temp)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _jsonable(changes):
    return [dict(c) for c in changes]


def _show(value, limit=12):
    if isinstance(value, list):
        if len(value) <= limit:
            return json.dumps(value)
        head = ", ".join(json.dumps(v) for v in value[:8])
        tail = ", ".join(json.dumps(v) for v in value[-3:])
        return f"[{head}, ..., {tail}] (n={len(value)})"
    return json.dumps(value)


def plan_report(src: SourceModel, plan: Plan) -> dict:
    values = planned_values(plan)
    contract = loader_contract(values, planned_shapes(plan))
    tensors = None
    if not plan.header_only:
        by_layer = {}
        for t in plan.tensors:
            key = f"blk.{plan.new_of[t['layer_old']]} (old {t['layer_old']})" if t["layer_old"] is not None else t["name"]
            by_layer[key] = by_layer.get(key, 0) + t["nbytes"]
        types = {}
        for t in plan.tensors:
            types[GGML_TYPES[t["type"]][0]] = types.get(GGML_TYPES[t["type"]][0], 0) + t["nbytes"]
        tensors = dict(count=len(plan.tensors), bytes=plan.tensor_bytes, gib=round(plan.tensor_bytes / 2**30, 3),
                       bytes_by_layer=by_layer, bytes_by_type=types,
                       list=[dict(name=t["name"], source=t["source"], type=GGML_TYPES[t["type"]][0],
                                  shape=t["shape"], bytes=t["nbytes"]) for t in plan.tensors])
    return dict(tool=TOOL_VERSION, mode="dry-run", source=str(src.path), source_files=src.files(),
                missing_shards=src.missing, incomplete_shards=src.incomplete, header_only=plan.header_only,
                block_count=dict(old=plan.n_layer, new=len(plan.kept)), kept_layers=plan.kept,
                layer_map={str(o): n for o, n in plan.new_of.items()},
                topology=dict(kv_owners=dict(old=plan.kv_owners, new=plan.new_kv_owners),
                              index_owners=dict(old=plan.index_owners, new=plan.new_index_owners),
                              candidate=dict(old=plan.candidate, new=plan.new_candidate),
                              candidate_budget=plan.candidate_budget,
                              per_layer=plan.per_layer),
                engram=plan.engram, metadata_changes=_jsonable(plan.changes),
                copied_keys=sum(1 for kv in plan.kvs if kv.raw is not None),
                unclassified_copied=plan.unclassified, notes=plan.notes, tensors=tensors,
                estimate=plan.estimate, planned_header_loader_contract=contract)


def print_plan(report, out=None):
    w = (out or sys.stdout).write
    w(f"source: {report['source']}\n")
    if report["missing_shards"]:
        w(f"missing shards (header-only plan, no tensor data needed): {len(report['missing_shards'])}\n")
        for path in report["missing_shards"]:
            w(f"  {path}\n")
    w(f"block_count {report['block_count']['old']} -> {report['block_count']['new']}; kept layers "
      f"{report['kept_layers']}\n")
    w("layer map (old -> new): " + " ".join(f"{o}->{n}" for o, n in report["layer_map"].items()) + "\n\n")
    w(f"{'new':>3} {'old':>3} {'ratio':>5} {'kv src old->new':>16} {'idx src old->new':>17}\n")
    for row in report["topology"]["per_layer"]:
        kv = f"{row['kv_source_old']}->{row['kv_source_new']}" if row["ratio"] else "-"
        ix = f"{row['index_source_old']}->{row['index_source_new']}" if row["ratio"] else "-"
        w(f"{row['new']:>3} {row['old']:>3} {row['ratio']:>5} {kv:>16} {ix:>17}\n")
    t = report["topology"]
    w(f"kv owners {t['kv_owners']['old']} -> {t['kv_owners']['new']}; indexer owners {t['index_owners']['old']} -> "
      f"{t['index_owners']['new']}; candidate {t['candidate']['old']} -> {t['candidate']['new']}\n")
    b = t["candidate_budget"]
    if b["prunes_from_token"] is not None:
        w(f"candidate pruning: {b['source']} -> {b['new']} blocks x {b['block_size']} rows (indexer top_k "
          f"{b['indexer_top_k']}, minimum {b['min_blocks']} blocks); prunes from token {b['prunes_from_token']}\n")
    e = report["engram"]
    w(f"Engram ({'combined ple.*' if e['combined'] else 'per-layer'}): layers {e['source_layers']} -> kept "
      f"{e['kept_layers']} as {e['new_layers']}\n")
    for s in e.get("slices", []):
        w(f"  layer {s['layer_old']}->{s['layer_new']}: source rows [{s['source_first_row']}, "
          f"{s['source_first_row'] + s['rows']}) -> new rows [{s['new_first_row']}, {s['new_first_row'] + s['rows']})\n")
    w("\nrewritten metadata keys (old -> new):\n")
    width = max([len(c["key"]) for c in report["metadata_changes"]] + [10])
    for c in report["metadata_changes"]:
        note = f"  ({c['note']})" if c["note"] else ""
        w(f"  {c['key']:<{width}}  {c['action']:<13} {_show(c['old'])} -> {_show(c['new'])}{note}\n")
    w(f"copied unchanged: {report['copied_keys']} keys\n")
    for item in report["unclassified_copied"]:
        w(f"  WARNING unclassified key copied unchanged (does not look per-layer): {item['key']} {item['type']}\n")
    for note in report["notes"]:
        w(f"note: {note}\n")
    if report["tensors"]:
        tens = report["tensors"]
        w(f"\ntensors: {tens['count']} ({tens['gib']} GiB)\n")
        for key, nbytes in tens["bytes_by_layer"].items():
            w(f"  {key:<40} {nbytes / 2**30:10.3f} GiB\n")
    else:
        w("\ntensors: unavailable (tensor shards absent; the dry-run reads headers only)\n")
    if report["estimate"]:
        est = report["estimate"]
        w(f"estimate ({est['source']}): weights {est['weights_gib']} GiB + Engram slice {est['engram_gib']} GiB = "
          f"{est['total_gib']} GiB\n")
    w("\nloader checks on the planned header:\n")
    for c in report["planned_header_loader_contract"]:
        w(f"  [{c['status']:>9}] {c['check']}\n")
    q = report.get("quota")
    if q:
        w("\n" + (f"quota check [unchecked]: {q['reason']}" if q["status"] == "unchecked" else quota_line(q)) + "\n")


def _raise_on_sigterm(signum, frame):
    """SIGTERM (kill, or a pod stop) becomes SystemExit, so write_output's cleanup runs."""
    raise SystemExit(128 + signum)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="source GGUF, or shard 00001 of a split checkpoint")
    parser.add_argument("--output", "-o", type=Path, help="output GGUF (single file)")
    parser.add_argument("--layers", help="layers to keep, e.g. 0-3,20-24 (default: %s)" % ",".join(map(str, DEFAULT_LAYERS)))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan; reads headers only")
    mode.add_argument("--verify", action="store_true", help="verify an existing --output against the source")
    parser.add_argument("--report", type=Path, help="also write the JSON report here (must not exist)")
    parser.add_argument("--layout-json", type=Path, help="model-layout-metadata.json for byte estimates in a header-only dry-run")
    parser.add_argument("--chunk-mib", type=int, default=64, help="copy chunk size")
    parser.add_argument("--samples", type=int, default=6, help="random sampled blocks per tensor (plus first/last)")
    parser.add_argument("--sample-kib", type=int, default=1024, help="sampled block size")
    parser.add_argument("--full", action="store_true", help="--verify compares every byte instead of samples")
    parser.add_argument("--engram-sequences", type=int, default=8, help="token sequences hashed per Engram layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-top-k-blocks", type=int, metavar="N",
                        help="rewrite the candidate-pruning budget so pruning engages on short gate texts "
                             "(1017: 65 prunes from token 521; default keeps the source budget)")
    parser.add_argument("--min-free-gib", type=float, default=2.0,
                        help="reserve that must stay free after the write, both on the filesystem and under "
                             "--quota-bytes (default 2)")
    parser.add_argument("--quota-bytes", type=byte_count, metavar="N",
                        help="quota of the volume the output goes to (e.g. 500GB or 500GiB; a bare G is refused). "
                             "Before writing, refuse when the du-style usage of --quota-root plus the output and "
                             "the --min-free-gib reserve exceeds N. For FUSE volumes (MooseFS pods) whose statvfs "
                             "reports the whole cluster, not the quota")
    parser.add_argument("--quota-root", type=Path, metavar="DIR",
                        help="directory the quota covers (e.g. /workspace); must contain --output. Usage is the "
                             "larger of the apparent (du -sb) and allocated (st_blocks x 512) totals")
    parser.add_argument("--keep-failed-partial", action="store_true",
                        help="keep a fully written .partial whose verification or publish failed (default: "
                             "remove it, so it never holds a quota-limited volume full); an incomplete one is "
                             "always removed")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if args.report is not None and args.report.exists():
        parser.error(f"{args.report} exists; keep earlier evidence")
    if (args.quota_bytes is None) != (args.quota_root is None):
        parser.error("--quota-bytes and --quota-root go together")
    if args.quota_bytes is not None and (args.verify or args.output is None):
        parser.error("--quota-bytes checks a write: it needs --output and does not apply to --verify")
    quota = None if args.quota_bytes is None else (args.quota_bytes, args.quota_root)
    reserve = int(args.min_free_gib * 2**30)
    try:
        if args.verify:
            if args.output is None:
                parser.error("--verify needs --output")
            src = SourceModel(args.source)
            if src.incomplete:
                raise TruncateError(f"source shards are incomplete: {src.incomplete}")
            kept = parse_layers(args.layers) if args.layers else None
            report = verify_output(src, args.output, kept, samples=args.samples, sample_bytes=args.sample_kib << 10,
                                   full=args.full, seed=args.seed, engram_sequences=args.engram_sequences,
                                   candidate_top_k_blocks=args.candidate_top_k_blocks,
                                   chunk=args.chunk_mib << 20)
            text = json.dumps(report, indent=2)
            print(text)
            if args.report:
                args.report.write_text(text + "\n", encoding="utf-8")
            return 0 if report["status"] == "pass" else 1
        kept = parse_layers(args.layers) if args.layers else list(DEFAULT_LAYERS)
        src = SourceModel(args.source, allow_missing=args.dry_run)
        if src.incomplete and not args.dry_run:
            raise TruncateError(f"source shards are incomplete: {src.incomplete}")
        layout = json.loads(args.layout_json.read_text(encoding="utf-8")) if args.layout_json else None
        plan = make_plan(src, kept, header_only=args.dry_run and not src.complete, layout_json=layout,
                         candidate_top_k_blocks=args.candidate_top_k_blocks)
        report = plan_report(src, plan)
        if args.dry_run:
            if args.output is not None:
                report["output"] = dict(path=str(args.output), exists=args.output.exists())
            if quota is not None:
                if plan.header_only:
                    report["quota"] = dict(status="unchecked", reason="header-only plan: the output size is "
                                                                      "unknown until every shard is present")
                else:
                    header, data_size = encode_header(plan)
                    report["quota"] = check_quota(quota[0], quota[1], args.output, len(header) + data_size, reserve)
            print_plan(report)
            text = json.dumps(report, indent=2)
            print("\n" + text)
            if args.report:
                args.report.write_text(text + "\n", encoding="utf-8")
            bad = [c for c in report["planned_header_loader_contract"] if c["status"] == "FAIL"]
            return 1 if bad or report.get("quota", {}).get("status") == "FAIL" else 0
        if args.output is None:
            parser.error("writing needs --output (or pass --dry-run)")
        print_plan(report)
        bad = [c for c in report["planned_header_loader_contract"] if c["status"] != "pass"]
        if bad:
            raise TruncateError(f"the planned header fails loader checks: {bad}")
        installed, previous = False, None
        try:
            previous = signal.signal(signal.SIGTERM, _raise_on_sigterm)
            installed = True
        except ValueError:                  # not the main thread: no handler, no cleanup on SIGTERM
            pass
        try:
            manifest = write_output(src, plan, args.output, chunk=args.chunk_mib << 20, min_free=reserve,
                                    samples=args.samples, sample_bytes=args.sample_kib << 10, seed=args.seed,
                                    quota=quota, keep_failed=args.keep_failed_partial)
        finally:
            if installed:
                signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)
        summary = dict(status=manifest["status"], output=manifest["output"], output_bytes=manifest["output_bytes"],
                       output_sha256=manifest["output_sha256"], verification=manifest["verification"]["status"])
        print(json.dumps(summary, indent=2))
        if args.report:
            args.report.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return 0
    except (TruncateError, GgufError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
