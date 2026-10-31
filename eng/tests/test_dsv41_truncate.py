"""Contract tests for eng/validation/dsv41-truncate-gguf.py on small synthetic V4.1 GGUFs.

The synthetic sources are written with gguf-py's GGUFWriter and the outputs are read
back with gguf-py's GGUFReader, so the tool's own GGUF reader/writer is checked
against an independent implementation. The Engram oracle below follows
eng/dsv41-reference.py (XOR-reduced products, absolute head offsets), not the
tool's port of the native loader.
"""
import builtins
import contextlib
import errno
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import stat
import struct
import tempfile
import unittest
from unittest import mock
import zlib

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFReader, GGUFValueType as V, GGUFWriter
from gguf.constants import GGML_QUANT_SIZES


def load_tool():
    path = Path(__file__).parents[1] / "validation" / "dsv41-truncate-gguf.py"
    spec = importlib.util.spec_from_file_location("dsv41_truncate_gguf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()
REAL_HEADER = (Path(__file__).parents[2] / "artifacts" / "dsv41-1017-header" /
               "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf")

NGRAM, HEADS, DIM = 3, 2, 256           # 4 hash columns per Engram layer, Q5_K rows of 176 bytes
TOKENS = ["<s>", "</s>", "<pad>", "a", "b", "c", "A", "B", "x", "y", "z", "!"]
TOKEN_MAP = [0, 1, 2, 3, 4, 5, 3, 4, 6, 7, 8, 9]
ENGRAM_PRIMES = {1: [5, 7, 11, 13], 3: [3, 5, 7, 11], 5: [17, 19, 23, 29]}
ENGRAM_MULTS = {1: [3, 5, 7], 3: [9, 11, 13], 5: [15, 17, 19]}

LAYER_TENSORS = [
    ("attn_norm.weight", Q.F32, [32]), ("attn_sinks.weight", Q.F32, [4]),
    ("attn_q_a.weight", Q.Q8_0, [32, 8]), ("attn_q_a_norm.weight", Q.F32, [8]),
    ("attn_q_b.weight", Q.Q6_K, [256, 2]), ("attn_kv.weight", Q.Q8_0, [32, 4]),
    ("attn_kv_a_norm.weight", Q.F32, [4]), ("attn_output_a.weight", Q.Q8_0, [32, 2]),
    ("attn_output_b.weight", Q.Q8_0, [64, 8]),
    ("hc_attn_fn.weight", Q.F32, [16, 6]), ("hc_attn_base.weight", Q.F32, [6]), ("hc_attn_scale.weight", Q.F32, [3]),
    ("hc_ffn_fn.weight", Q.F32, [16, 6]), ("hc_ffn_base.weight", Q.F32, [6]), ("hc_ffn_scale.weight", Q.F32, [3]),
    ("ffn_gate_inp.weight", Q.F32, [32, 4]), ("ffn_norm.weight", Q.F32, [32]),
    ("ffn_gate_exps.weight", Q.IQ1_S, [256, 2, 4]), ("ffn_up_exps.weight", Q.IQ2_XXS, [256, 2, 4]),
    ("ffn_down_exps.weight", Q.IQ2_S, [256, 1, 4]),
    ("ffn_gate_shexp.weight", Q.Q5_K, [256, 2]), ("ffn_up_shexp.weight", Q.Q5_K, [256, 2]),
    ("ffn_down_shexp.weight", Q.Q8_0, [64, 4]),
]


def payload(name, qtype, shape):
    block, size = GGML_QUANT_SIZES[qtype]
    nbytes = shape[0] // block * size * int(np.prod(shape[1:], dtype=np.int64))
    data = np.random.default_rng(zlib.crc32(name.encode())).integers(0, 256, nbytes, dtype=np.uint8)
    return data.reshape(tuple(reversed(shape[1:])) + (shape[0] // block * size,))


def build(path, n_layer=10, ratios=(0, 0, 2, 2, 2, 2, 1, 1, 1, 1), tail=(0, 0), kv_owners=(2, 4, 6),
          index_owners=(2, 4, 6, 8), candidate=6, engram=(1, 5), hash_layers=1, dspark=(7, 8, 9),
          extra_kv=(), split=0):
    """A small V4.1 GGUF in llama.cpp's combined-Engram layout. split>0 writes shards
    of that many tensors after a metadata-only first shard (unsloth's layout)."""
    writer = GGUFWriter(path, "deepseek41", split_max_tensors=split, small_first_shard=bool(split))
    p = "deepseek41."
    writer.add_name("Synthetic V4.1")
    writer.add_key_value("general.version", "1017", V.STRING)
    writer.add_key_value(p + "block_count", n_layer, V.UINT32)
    writer.add_key_value(p + "context_length", 4096, V.UINT32)
    writer.add_key_value(p + "embedding_length", 32, V.UINT32)
    writer.add_key_value(p + "attention.head_count", 4, V.UINT32)
    writer.add_key_value(p + "attention.sliding_window", 128, V.UINT32)
    writer.add_key_value(p + "rope.freq_base", 10000.0, V.FLOAT32)
    writer.add_key_value(p + "expert_weights_norm", True, V.BOOL)
    writer.add_key_value(p + "swiglu_clamp_exp", [10.0 + i for i in range(n_layer)], V.ARRAY, V.FLOAT32)
    writer.add_key_value(p + "swiglu_clamp_shexp", [20.0 + i for i in range(n_layer)], V.ARRAY, V.FLOAT32)
    writer.add_key_value(p + "attention.indexer.top_k", 64, V.UINT32)
    writer.add_key_value(p + "attention.compress_ratios", list(ratios) + list(tail), V.ARRAY, V.INT32)
    writer.add_key_value(p + "hyper_connection.count", 4, V.UINT32)
    writer.add_key_value(p + "attention.kv_source_layers", list(kv_owners), V.ARRAY, V.UINT32)
    writer.add_key_value(p + "attention.indexer.source_layers", list(index_owners), V.ARRAY, V.UINT32)
    if candidate is not None:
        writer.add_key_value(p + "attention.indexer.candidate_source_layer", candidate, V.INT32)
    writer.add_key_value(p + "attention.indexer.candidate_top_k_blocks", 16, V.UINT32)
    writer.add_key_value(p + "attention.indexer.candidate_block_size", 8, V.UINT32)
    writer.add_key_value(p + "ple.layers", list(engram), V.ARRAY, V.UINT32)
    writer.add_key_value(p + "ple.ngram_size", NGRAM, V.UINT32)
    writer.add_key_value(p + "ple.heads_per_ngram", HEADS, V.UINT32)
    writer.add_key_value(p + "embedding_length_per_layer_input", DIM, V.UINT32)
    writer.add_key_value(p + "ple.layer_multipliers", [m for e in engram for m in ENGRAM_MULTS[e]], V.ARRAY, V.UINT64)
    offsets, primes, total = [], [], 0
    for e in engram:
        for prime in ENGRAM_PRIMES[e]:
            offsets.append(total)
            primes.append(prime)
            total += prime
    writer.add_key_value(p + "ple.head_offsets", offsets, V.ARRAY, V.UINT64)
    writer.add_key_value(p + "ple.head_vocab_sizes", primes, V.ARRAY, V.UINT64)
    writer.add_key_value(p + "hash_layer_count", hash_layers, V.UINT32)
    writer.add_key_value(p + "engram.vocab_size", 1000, V.UINT32)
    writer.add_key_value(p + "engram.pad_token_id", 2, V.UINT32)
    writer.add_key_value(p + "engram.compressed_vocab_size", max(TOKEN_MAP) + 1, V.UINT32)
    writer.add_key_value(p + "engram.token_map", TOKEN_MAP, V.ARRAY, V.INT32)
    if dspark:
        writer.add_key_value(p + "dspark.block_size", 5, V.UINT32)
        writer.add_key_value(p + "dspark.target_layers", list(dspark), V.ARRAY, V.UINT32)
    for key, value, vtype, sub in extra_kv:
        writer.add_key_value(key, value, vtype, sub)
    writer.add_token_list(TOKENS)
    writer.add_key_value("tokenizer.ggml.model", "gpt2", V.STRING)

    def tensor(name, qtype, shape):
        writer.add_tensor(name, payload(name, qtype, shape), raw_dtype=qtype)

    tensor("token_embd.weight", Q.Q8_0, [32, len(TOKENS)])
    for il in range(n_layer):
        specs = list(LAYER_TENSORS)
        specs.append(("ffn_gate_tid2eid.weight", Q.I32, [6, len(TOKENS)]) if il < hash_layers
                     else ("exp_probs_b.bias", Q.F32, [4]))
        if il in kv_owners:
            specs += [("attn_compressor_kv.weight", Q.Q8_0, [32, 4]), ("attn_compressor_norm.weight", Q.F32, [4]),
                      ("indexer.attn_k.weight", Q.Q8_0, [32, 2]), ("indexer.k_norm.weight", Q.F32, [2])]
            if ratios[il] > 1:
                specs.append(("attn_compressor_gate.weight", Q.Q8_0, [32, 4]))
        if il in index_owners:
            specs += [("indexer.proj.weight", Q.Q8_0, [32, 2]), ("indexer.attn_q_b.weight", Q.Q6_K, [256, 2])]
        if il in engram:
            specs += [("engram_kv.weight", Q.Q8_0, [256, 4]), ("engram_gate.weight", Q.F32, [32, 4])]
        for name, qtype, shape in specs:
            tensor(f"blk.{il}.{name}", qtype, shape)
    tensor("output_norm.weight", Q.F32, [32])
    tensor("output.weight", Q.Q6_K, [256, len(TOKENS)])
    tensor("per_layer_token_embd.weight", Q.Q5_K, [DIM, total])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if split:
        shards = sorted(path.parent.glob(path.stem + "-*-of-*.gguf"))
        assert len(shards) > 2, shards
        return shards[0]
    return path


def read_gguf(path):
    """Fields and tensor bytes via gguf-py; the mapping is released before returning
    (Windows refuses to delete or rewrite a file that is still mapped)."""
    reader = GGUFReader(str(path))
    fields = {name: (list(field.types), field.contents()) for name, field in reader.fields.items()
              if not name.startswith("GGUF.")}
    tensors = {t.name: (t.tensor_type.name, [int(x) for x in t.shape], np.asarray(t.data).tobytes())
               for t in reader.tensors}
    del reader
    gc.collect()
    return fields, tensors


def oracle_rows(fields, layer_index, tokens):
    """Absolute rows of the combined table, in the style of eng/dsv41-reference.py."""
    p = "deepseek41."
    ngram = fields[p + "ple.ngram_size"][1]
    heads = fields[p + "ple.heads_per_ngram"][1]
    columns = (ngram - 1) * heads
    mult = np.asarray(fields[p + "ple.layer_multipliers"][1][layer_index * ngram:(layer_index + 1) * ngram], dtype=np.int64)
    primes = np.asarray(fields[p + "ple.head_vocab_sizes"][1][layer_index * columns:(layer_index + 1) * columns],
                        dtype=np.int64).reshape(ngram - 1, heads)
    offsets = np.asarray(fields[p + "ple.head_offsets"][1][layer_index * columns:(layer_index + 1) * columns], dtype=np.int64)
    token_map = fields[p + "engram.token_map"][1]
    pad = token_map[fields[p + "engram.pad_token_id"][1]]
    out = []
    for pos in range(len(tokens)):
        history, blocked = [], False
        for shift in range(ngram):
            blocked = blocked or pos < shift or tokens[pos - shift] < 0
            history.append(pad if blocked else token_map[tokens[pos - shift]])
        products = np.asarray(history, dtype=np.int64) * mult
        rows = [np.bitwise_xor.reduce(products[:n]) % primes[n - 2] for n in range(2, ngram + 1)]
        out.extend((np.concatenate(rows) + offsets).tolist())
    return out


ROW = 176   # Q5_K bytes per 256-element row
SEQUENCES = [[3], [0, 4, 5], [11, 10, 9, 8, 7], [2, 2, 2, 6], [5, -1, 4, 3, 1], [7, 3, 11, 0, 4, 9, 1, 2]]


def gapped_encode_header(plan):
    """The tool's header encoding with one alignment unit of slack before every tensor
    after the first. Each offset stays aligned and increasing, so gguf-py and the tool's
    own reader accept the file, but gguf_init_from_file (ggml src/gguf.cpp:795) requires
    every offset to equal the running sum of GGML_PAD(nbytes, alignment)."""
    parts = [b"GGUF", struct.pack("<IQQ", 3, len(plan.tensors), len(plan.kvs))]
    for kv in plan.kvs:
        parts += [tool._encode_string(kv.key), kv.encoded()]
    offset = 0
    for index, tensor in enumerate(plan.tensors):
        if index:
            offset += plan.alignment
        tensor["offset"] = offset
        shape = tensor["shape"]
        parts.append(tool._encode_string(tensor["name"]) +
                     struct.pack(f"<I{len(shape)}QIQ", len(shape), *shape, tensor["type"], offset))
        offset = tool.align(offset + tensor["nbytes"], plan.alignment)
    header = b"".join(parts)
    return header + b"\0" * (tool.align(len(header), plan.alignment) - len(header)), offset


EDQUOT = getattr(errno, "EDQUOT", 122)


def quota_exceeded():
    raise OSError(EDQUOT, "Disk quota exceeded")


class FailingFile:
    """Stands in for a file the tool opens in exclusive-create mode. Writes pass through
    until `budget` bytes (or characters) have been written; the write that crosses it
    lands its in-budget prefix on disk and then calls fail(), the way a write on a volume
    at its quota fails partway. close_fail fails the close instead (FUSE can report EDQUOT
    there)."""

    def __init__(self, handle, path, budget=None, fail=None, close_fail=None, state=None):
        self.handle, self.path, self.budget, self.fail, self.close_fail = handle, path, budget, fail, close_fail
        self.state = {} if state is None else state
        self.written = 0

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.close()
        if self.close_fail is not None and exc[0] is None:
            self.close_fail()
        return False

    def write(self, data):
        if self.budget is not None and self.written + len(data) > self.budget:
            keep = self.budget - self.written
            self.handle.write(data[:keep])
            self.handle.flush()
            self.written += keep
            self.state.update(failed_after=self.written, size_on_disk=os.path.getsize(self.path))
            self.fail()
        self.written += len(data)
        return self.handle.write(data)


def failing_open(suffix, **failure):
    """A replacement for the tool's open(): exclusive creates of a path ending in suffix
    get a FailingFile; everything else (the source reads, verify) is the real open."""
    def fake(path, mode="r", *args, **kwargs):
        handle = builtins.open(path, mode, *args, **kwargs)
        if "x" in mode and str(path).endswith(suffix):
            return FailingFile(handle, path, **failure)
        return handle
    return fake


def racing_open(suffix, foreign):
    """Another writer creates `suffix` between the tool's existence check and its
    exclusive create; the file is not the tool's to remove."""
    def fake(path, mode="r", *args, **kwargs):
        if "x" in mode and str(path).endswith(suffix):
            with builtins.open(path, "wb") as other:
                other.write(foreign)
        return builtins.open(path, mode, *args, **kwargs)
    return fake


def st_blocks_bytes(st):
    blocks = getattr(st, "st_blocks", None)
    return None if blocks is None else blocks * 512


def du_oracle(root, allocated_of=st_blocks_bytes):
    """du -x by os.walk: the root, every directory and file once (hard links once per
    inode), no symlinks followed; the larger of the apparent and allocated totals, or
    the apparent total where stat has no st_blocks."""
    top = os.lstat(root)
    stats = [top]
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            stats.append(os.lstat(os.path.join(dirpath, name)))
    seen, apparent, allocated, known = set(), 0, 0, True
    for st in stats:
        if not stat.S_ISDIR(st.st_mode) and st.st_nlink > 1:
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
        apparent += st.st_size
        size = allocated_of(st)
        if size is None:
            known = False
        else:
            allocated += size
    return max(apparent, allocated) if known else apparent, apparent, allocated if known else None


class TruncateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def source(self, name="src", **spec):
        folder = self.dir / name
        folder.mkdir()
        return build(folder / "model.gguf", **spec)

    def run_tool(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = tool.main([str(a) for a in args])
        return code, out.getvalue(), err.getvalue()

    def truncate(self, source, layers, name="out.gguf"):
        output = self.dir / name
        code, out, err = self.run_tool(source, "--output", output, "--layers", layers)
        self.assertEqual(code, 0, out + err)
        self.assertTrue(output.exists() and output.with_name(name + ".manifest.json").exists())
        self.assertFalse(output.with_name(name + ".partial").exists())
        return output

    def refuse(self, source, layers, *patterns):
        output = self.dir / "refused.gguf"
        code, out, err = self.run_tool(source, "--output", output, "--layers", layers)
        self.assertEqual(code, 2, out + err)
        for pattern in patterns:
            self.assertRegex(err, pattern)
        self.assertFalse(output.exists() or output.with_name("refused.gguf.partial").exists())
        return err

    def test_renumbers_every_per_layer_key_and_copies_bytes(self):
        src = self.source()
        out = self.truncate(src, "0-3,6-8")          # 0,1,2,3,6,7,8 -> 0..6
        new_of = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5, 8: 6}
        sf, st = read_gguf(src)
        of, ot = read_gguf(out)
        p = "deepseek41."
        expect = {
            p + "block_count": 7,
            p + "swiglu_clamp_exp": [10.0, 11.0, 12.0, 13.0, 16.0, 17.0, 18.0],
            p + "swiglu_clamp_shexp": [20.0, 21.0, 22.0, 23.0, 26.0, 27.0, 28.0],
            p + "attention.compress_ratios": [0, 0, 2, 2, 1, 1, 1, 0, 0],      # trunk + preserved tail
            p + "attention.kv_source_layers": [2, 4],
            p + "attention.indexer.source_layers": [2, 4, 6],
            p + "attention.indexer.candidate_source_layer": 4,
            p + "ple.layers": [1],
            p + "ple.layer_multipliers": ENGRAM_MULTS[1],
            p + "ple.head_vocab_sizes": ENGRAM_PRIMES[1],
            p + "ple.head_offsets": [0, 5, 12, 23],
            p + "hash_layer_count": 1,
            # the fixture's budget is 16 blocks in candidate_top_k_blocks alone; PR #306-era loaders
            # read only candidate_top_k (default 2048), so the output spells it both ways
            p + "attention.indexer.candidate_top_k": 16,
            "tensorsharp.truncate.kept_layers": [0, 1, 2, 3, 6, 7, 8],
            "tensorsharp.truncate.origin_layers": [0, 1, 2, 3, 6, 7, 8],
            "tensorsharp.truncate.source_block_count": 10,
        }
        for key, value in expect.items():
            self.assertEqual(of[key][1], value, key)
        for key in expect:
            if key in sf:
                self.assertEqual(of[key][0], sf[key][0], f"{key} changed type")
        self.assertEqual(of[p + "attention.indexer.candidate_top_k"][0], [V.UINT32])
        self.assertFalse([k for k in of if k.startswith((p + "dspark.", "split."))])
        untouched = set(sf) - set(expect) - {k for k in sf if k.startswith(p + "dspark.")}
        for key in untouched:
            self.assertEqual(of[key], sf[key], key)
        self.assertEqual(set(of) - set(sf), {k for k in expect if k.startswith("tensorsharp.")} |
                         {"tensorsharp.truncate.tool", p + "attention.indexer.candidate_top_k"})
        # every kept tensor is a renamed byte copy; nothing else is present
        expected_names = set()
        for name, (qtype, shape, data) in st.items():
            parts = name.split(".", 2)
            if parts[0] == "blk":
                old = int(parts[1])
                if old not in new_of:
                    continue
                new_name = f"blk.{new_of[old]}.{parts[2]}"
                self.assertEqual(ot[new_name], (qtype, shape, data), name)
                expected_names.add(new_name)
            elif name == "per_layer_token_embd.weight":
                self.assertEqual(ot[name][:2], (qtype, [DIM, 36]))
                self.assertEqual(ot[name][2], data[:36 * ROW])
                expected_names.add(name)
            else:
                self.assertEqual(ot[name], (qtype, shape, data), name)
                expected_names.add(name)
        self.assertEqual(set(ot), expected_names)
        code, text, err = self.run_tool(src, "--verify", "--output", out)
        self.assertEqual(code, 0, text + err)
        report = json.loads(text)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(any(c["check"].startswith("loader: per-layer tensors") and c["status"] == "pass"
                            for c in report["checks"]))

    def test_engram_slice_is_rebased_and_hashes_to_the_same_bytes(self):
        src = self.source()
        out = self.truncate(src, "0,2-6")             # keeps Engram layer 5 only, as new layer 4
        sf, st = read_gguf(src)
        of, ot = read_gguf(out)
        p = "deepseek41."
        self.assertEqual(of[p + "ple.layers"][1], [4])
        self.assertEqual(of[p + "ple.head_offsets"][1], [0, 17, 36, 59])   # [36, 53, 72, 95] rebased
        self.assertEqual(of[p + "attention.indexer.candidate_source_layer"][1], 5)
        self.assertEqual(ot["per_layer_token_embd.weight"][1], [DIM, 88])
        source_table, table = st["per_layer_token_embd.weight"][2], ot["per_layer_token_embd.weight"][2]
        self.assertEqual(table, source_table[36 * ROW:124 * ROW])
        self.assert_same_rows(src, out, sf, of, source_table, table, [(1, 0)])

    def test_dropping_a_middle_engram_layer_concatenates_the_kept_slices(self):
        src = self.source(engram=(1, 3, 5))
        out = self.truncate(src, "0-2,4-6")           # drops Engram layer 3
        sf, st = read_gguf(src)
        of, ot = read_gguf(out)
        p = "deepseek41."
        self.assertEqual(of[p + "ple.layers"][1], [1, 4])
        self.assertEqual(of[p + "ple.head_offsets"][1], [0, 5, 12, 23, 36, 53, 72, 95])
        self.assertEqual(of[p + "ple.layer_multipliers"][1], ENGRAM_MULTS[1] + ENGRAM_MULTS[5])
        source_table, table = st["per_layer_token_embd.weight"][2], ot["per_layer_token_embd.weight"][2]
        self.assertEqual(table, source_table[:36 * ROW] + source_table[62 * ROW:150 * ROW])
        self.assert_same_rows(src, out, sf, of, source_table, table, [(0, 0), (2, 1)])

    def assert_same_rows(self, src, out, sf, of, source_table, table, layer_pairs):
        """Independent oracle and the tool's native-loader port must both map every
        sequence to rows whose bytes are identical in the full and truncated tables."""
        src_header, out_header = tool.SourceModel(src), tool.GgufHeader(out)
        src_layout = tool.engram_layout(src_header.kvs, {n: t.shape for n, t in src_header.tensors.items()})
        out_layout = tool.engram_layout(out_header.kvs, {t.name: t.shape for t in out_header.tensors})
        rows = 0
        for source_index, output_index in layer_pairs:
            for tokens in SEQUENCES:
                full = oracle_rows(sf, source_index, tokens)
                lite = oracle_rows(of, output_index, tokens)
                self.assertEqual(len(full), len(tokens) * (NGRAM - 1) * HEADS)
                for a, b in zip(full, lite):
                    self.assertEqual(source_table[a * ROW:(a + 1) * ROW], table[b * ROW:(b + 1) * ROW])
                    rows += 1
                layer, mirror = src_layout["layers"][source_index], out_layout["layers"][output_index]
                port_full = [layer["first_row"] + r for r in tool.engram_hash_rows(src_layout, layer, tokens)]
                port_lite = [mirror["first_row"] + r for r in tool.engram_hash_rows(out_layout, mirror, tokens)]
                self.assertEqual(port_full, full)
                self.assertEqual(port_lite, lite)
        self.assertGreater(rows, 100)
        slices = tool.combined_engram_slices(out_header.kvs, len(table) // ROW)
        self.assertEqual(sum(s["rows"] for s in slices), len(table) // ROW)

    def test_refuses_to_drop_a_layer_that_kept_layers_read(self):
        src = self.source()
        self.refuse(src, "0,1,3", r"layer 3 \(compress ratio 2\) reads the shared compressed KV cache owned by layer 2")
        self.refuse(src, "0-7,9", r"layer 9 \(compress ratio 1\) reads the lightning-indexer keys owned by layer 8")
        self.refuse(src, "0,2,3", r"no Engram layer is kept")
        other = self.source("candidate", index_owners=(2, 4, 6, 7, 8), candidate=7)
        self.refuse(other, "0-2,6,8", r"layers \[8\] index with the candidate block mask built by layer 7")

    def test_dropped_candidate_without_dependents_disables_pruning(self):
        src = self.source()
        out = self.truncate(src, "0-3")
        of, _ = read_gguf(out)
        types, value = of["deepseek41.attention.indexer.candidate_source_layer"]
        self.assertEqual((types, value), ([V.INT32], -1))
        code, text, err = self.run_tool(src, "--verify", "--output", out)
        self.assertEqual(code, 0, text + err)

    def test_refuses_unknown_per_layer_keys(self):
        cases = [
            ("deepseek41.attention.mystery_scale", [1.0] * 10, V.ARRAY, V.FLOAT32, "relates to block_count"),
            ("deepseek41.attention.foo_source_layers", [2, 6], V.ARRAY, V.UINT32, "mentions"),
            ("deepseek41.swa_window_pattern", [True, False] * 5, V.ARRAY, V.BOOL, "mentions"),
            ("deepseek41.leading_dense_block_count", 1, V.UINT32, None, "mentions block"),
            ("deepseek41.attention.gain_ids", [1, 4, 7], V.ARRAY, V.INT32, "layer ids"),
            ("acme.per_layer_gain", [0.5] * 10, V.ARRAY, V.FLOAT32, "mentions"),
        ]
        for index, (key, value, vtype, sub, reason) in enumerate(cases):
            with self.subTest(key=key):
                src = self.source(f"unknown{index}", extra_kv=[(key, value, vtype, sub)])
                self.refuse(src, "0-3,6-8", rf"metadata key {key} .*looks per-layer \(.*{reason}")
        src = self.source("benign", extra_kv=[("deepseek41.attention.temperature", 0.5, V.FLOAT32, None)])
        out = self.truncate(src, "0-3,6-8")
        of, _ = read_gguf(out)
        self.assertEqual(of["deepseek41.attention.temperature"][1], 0.5)

    def test_split_shard_input_matches_single_file_input(self):
        single = self.source("single")
        first = self.source("split", split=40)
        self.assertEqual(tool.GgufHeader(first).tensors, [])            # metadata-only shard 1
        a = self.truncate(single, "0-3,6-8", "from-single.gguf")
        b = self.truncate(first, "0-3,6-8", "from-split.gguf")
        self.assertEqual(hashlib.sha256(a.read_bytes()).digest(), hashlib.sha256(b.read_bytes()).digest())
        code, text, err = self.run_tool(first, "--verify", "--output", b)
        self.assertEqual(code, 0, text + err)
        shards = sorted(first.parent.glob("model-*-of-*.gguf"))
        shards[-1].unlink()
        self.refuse(first, "0-3,6-8", r"is missing")

    def test_dry_run_reads_headers_only(self):
        first = self.source("split", split=40)
        full_code, full_text, _ = self.run_tool(first, "--dry-run", "--layers", "0-3,6-8")
        self.assertEqual(full_code, 0)
        for shard in sorted(first.parent.glob("model-*-of-*.gguf"))[1:]:
            shard.unlink()
        code, text, err = self.run_tool(first, "--dry-run", "--layers", "0-3,6-8", "--output", self.dir / "never.gguf")
        self.assertEqual(code, 0, text + err)
        self.assertIn("header-only plan, no tensor data needed", text)
        self.assertIn("tensors: unavailable", text)
        report = json.loads(text[text.index("\n{") + 1:])
        full = json.loads(full_text[full_text.index("\n{") + 1:])
        self.assertTrue(report["header_only"])
        self.assertEqual(report["metadata_changes"], full["metadata_changes"])
        self.assertEqual(report["topology"], full["topology"])
        self.assertFalse((self.dir / "never.gguf").exists())
        statuses = {c["check"]: c["status"] for c in report["planned_header_loader_contract"]}
        self.assertEqual(statuses["per-layer tensors looked up by the native loader"], "unchecked")
        self.assertNotIn("FAIL", statuses.values())

    def test_verify_detects_changed_bytes_and_wrong_layers(self):
        src = self.source()
        out = self.truncate(src, "0-3,6-8")
        header = tool.GgufHeader(out)
        table = next(t for t in header.tensors if t.name == "per_layer_token_embd.weight")
        with open(out, "r+b") as stream:
            stream.seek(table.abs_offset + 5 * ROW + 3)
            byte = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([byte[0] ^ 0xFF]))
        code, text, _ = self.run_tool(src, "--verify", "--output", out)
        self.assertEqual(code, 1)
        report = json.loads(text)
        failed = [c for c in report["checks"] if c["status"] != "pass"]
        self.assertEqual([c["detail"]["mismatches"][0]["tensor"] for c in failed if "tensor bytes" in c["check"]],
                         ["per_layer_token_embd.weight"])
        self.assertTrue(all("tensor bytes" in c["check"] or "Engram" in c["check"] for c in failed), failed)
        code, text, _ = self.run_tool(src, "--verify", "--output", out, "--layers", "0-3,6-7")
        self.assertEqual(code, 1)
        self.assertIn("metadata equals the plan derived from the source", [
            c["check"] for c in json.loads(text)["checks"] if c["status"] != "pass"])

    def test_small_chunks_and_full_verify_match_the_default_copy(self):
        # The fixtures' tensors are smaller than the default 64 MiB chunk; force the
        # multi-chunk reader/writer pipeline and the chunked full comparison.
        src = self.source()
        reference = self.truncate(src, "0,2-6", "reference.gguf")
        model = tool.SourceModel(src)
        plan = tool.make_plan(model, [0, 2, 3, 4, 5, 6])
        chunked = self.dir / "chunked.gguf"
        manifest = tool.write_output(model, plan, chunked, chunk=97, min_free=0, samples=3, sample_bytes=61,
                                     log=lambda *_: None)
        self.assertEqual(chunked.read_bytes(), reference.read_bytes())
        # the in-stream whole-file digest is what sha256sum prints (the probe's --weights-id)
        self.assertEqual(manifest["output_sha256"], hashlib.sha256(chunked.read_bytes()).hexdigest())
        table = next(t for t in manifest["tensors"] if t["name"] == "per_layer_token_embd.weight")
        with open(src, "rb") as stream:
            source = tool.GgufHeader(src)
            info = next(t for t in source.tensors if t.name == "per_layer_token_embd.weight")
            stream.seek(info.abs_offset + 36 * ROW)
            self.assertEqual(table["sha256"], hashlib.sha256(stream.read(88 * ROW)).hexdigest())
        report = tool.verify_output(model, chunked, full=True, chunk=113)
        self.assertEqual(report["status"], "pass", json.dumps(report, indent=1))
        full = next(c for c in report["checks"] if c["check"].startswith("full tensor bytes"))
        self.assertEqual(full["detail"]["bytes_compared"], plan.tensor_bytes)

    def test_refuses_to_overwrite(self):
        src = self.source()
        out = self.truncate(src, "0-3,6-8")
        before = out.read_bytes()
        code, _, err = self.run_tool(src, "--output", out, "--layers", "0-3,6-8")
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(out.read_bytes(), before)
        stale = self.dir / "other.gguf.manifest.json.partial"
        stale.write_text("{}")
        code, _, err = self.run_tool(src, "--output", self.dir / "other.gguf", "--layers", "0-3,6-8")
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite", err)
        self.assertFalse((self.dir / "other.gguf.partial").exists())

    def test_tool_reader_agrees_with_gguf_py(self):
        src = self.source()
        fields, tensors = read_gguf(src)
        header = tool.GgufHeader(src)
        self.assertEqual(set(header.kvs), set(fields))
        for key, kv in header.kvs.items():
            self.assertEqual(kv.value, fields[key][1], key)
            self.assertEqual(kv.raw, tool.encode_value(kv.vtype, kv.value, kv.atype), key)
        with open(src, "rb") as stream:
            for t in header.tensors:
                stream.seek(t.abs_offset)
                self.assertEqual(stream.read(t.nbytes), tensors[t.name][2], t.name)

    def test_write_refuses_a_tensor_table_ggml_would_reject(self):
        # verify_output rebuilds the expected tensor table with the same encoder that
        # wrote the file, so an encoder bug cannot show up there; only ggml's own
        # offset rule, evaluated on the file alone, catches it.
        src = self.source()
        model = tool.SourceModel(src)
        plan = tool.make_plan(model, [0, 1, 2, 3, 6, 7, 8])
        out = self.dir / "gapped.gguf"
        partial = out.with_name("gapped.gguf.partial")
        with mock.patch.object(tool, "encode_header", gapped_encode_header):
            # a failed verification removes the complete .partial by default (it would
            # otherwise hold ~100 GB of a quota-limited volume) ...
            with self.assertRaisesRegex(tool.TruncateError, r"gguf_init_from_file(.|\n)*has offset \d+, expected \d+"):
                tool.write_output(model, plan, out, min_free=0, log=lambda *_: None)
            self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["src"])
            # ... and keep_failed (--keep-failed-partial) keeps it for inspection
            with self.assertRaisesRegex(tool.TruncateError, r"gguf_init_from_file(.|\n)*has offset \d+, expected \d+"):
                tool.write_output(model, plan, out, min_free=0, log=lambda *_: None, keep_failed=True)
        self.assertFalse(out.exists() or out.with_name("gapped.gguf.manifest.json").exists())
        fields, tensors = read_gguf(partial)          # an independent reader is blind to the gaps
        self.assertEqual(len(tensors), len(plan.tensors))
        problems = tool.ggml_gguf_layout(tool.GgufHeader(partial))["problems"]
        self.assertTrue(problems and problems[0].startswith(f"tensor '{plan.tensors[1]['name']}' has offset"), problems)
        good = self.truncate(src, "0-3,6-8", "good.gguf")
        layout = tool.ggml_gguf_layout(tool.GgufHeader(good))
        self.assertEqual(layout["problems"], [], layout)
        self.assertEqual(layout["data_size"], tool.GgufHeader(good).file_size - layout["data_offset"])

    def planned_bytes(self, src, kept):
        header, data_size = tool.encode_header(tool.make_plan(tool.SourceModel(src), kept))
        return len(header), len(header) + data_size

    def test_write_failure_mid_copy_removes_the_partial(self):
        # On a pod volume at its quota a write (or, on FUSE, the close) fails with EDQUOT
        # partway through a ~100 GB copy. The .partial must not stay behind holding the
        # volume at its quota; an incomplete one goes even with --keep-failed-partial.
        src = self.source()
        header_bytes, total = self.planned_bytes(src, [0, 1, 2, 3, 6, 7, 8])
        middle = header_bytes + (total - header_bytes) // 2
        cases = [
            ("write", ".gguf.partial", dict(budget=middle, fail=quota_exceeded), ()),
            ("write-keep", ".gguf.partial", dict(budget=middle, fail=quota_exceeded), ("--keep-failed-partial",)),
            ("close", ".gguf.partial", dict(close_fail=quota_exceeded), ()),
            ("manifest", ".manifest.json.partial", dict(budget=10, fail=quota_exceeded), ()),
        ]
        for name, suffix, failure, extra in cases:
            with self.subTest(failure=name):
                folder = self.dir / name
                state = {}
                with mock.patch.object(tool, "open", failing_open(suffix, state=state, **failure), create=True):
                    code, out, err = self.run_tool(src, "--output", folder / "lite.gguf", "--layers", "0-3,6-8", *extra)
                self.assertEqual(code, 2, out + err)
                self.assertIn("Disk quota exceeded", err)
                if name.startswith("write"):     # it failed mid-copy, with tensor bytes already on disk
                    self.assertTrue(header_bytes < state["size_on_disk"] == middle < total, state)
                self.assertEqual(list(folder.iterdir()), [], f"{name} left files behind")
                self.assertRegex(out, r"OSError: removed .*lite\.gguf\.partial")
        with self.subTest(failure="fsync"):
            model = tool.SourceModel(src)
            out = self.dir / "fsync" / "lite.gguf"
            with mock.patch.object(tool.os, "fsync", side_effect=OSError(errno.EIO, "Input/output error")):
                with self.assertRaisesRegex(OSError, "Input/output error"):
                    tool.write_output(model, tool.make_plan(model, [0, 1, 2, 3, 6, 7, 8]), out, min_free=0,
                                      log=lambda *_: None, keep_failed=True)
            self.assertEqual(list(out.parent.iterdir()), [])

    def test_interrupted_write_removes_the_partial(self):
        # Ctrl-C, and kill/pod stop (SIGTERM, which the CLI turns into SystemExit), mid-copy.
        src = self.source()
        header_bytes, total = self.planned_bytes(src, [0, 1, 2, 3, 6, 7, 8])
        middle = header_bytes + (total - header_bytes) // 2

        def interrupt():
            raise KeyboardInterrupt

        model = tool.SourceModel(src)
        out = self.dir / "ctrl-c" / "lite.gguf"
        state = {}
        with mock.patch.object(tool, "open", failing_open(".gguf.partial", budget=middle, fail=interrupt,
                                                          state=state), create=True):
            with self.assertRaises(KeyboardInterrupt):
                tool.write_output(model, tool.make_plan(model, [0, 1, 2, 3, 6, 7, 8]), out, min_free=0,
                                  log=lambda *_: None)
        self.assertEqual(state["size_on_disk"], middle)
        self.assertEqual(list(out.parent.iterdir()), [])

        def sigterm():                     # the signal arriving mid-write runs whatever handler is installed
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        before = signal.getsignal(signal.SIGTERM)
        out = self.dir / "sigterm" / "lite.gguf"
        state = {}
        with mock.patch.object(tool, "open", failing_open(".gguf.partial", budget=middle, fail=sigterm,
                                                          state=state), create=True):
            with self.assertRaises(SystemExit) as raised:
                self.run_tool(src, "--output", out, "--layers", "0-3,6-8")
        self.assertEqual(raised.exception.code, 128 + signal.SIGTERM)
        self.assertEqual(state["size_on_disk"], middle)
        self.assertEqual(list(out.parent.iterdir()), [])
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)    # the CLI restores the handler

    def test_cleanup_only_removes_files_this_run_created(self):
        src = self.source()
        foreign = b"another run's bytes"
        # a .partial left before the run is refused and kept
        stale = self.dir / "stale" / "lite.gguf.partial"
        stale.parent.mkdir()
        stale.write_bytes(foreign)
        code, out, err = self.run_tool(src, "--output", self.dir / "stale" / "lite.gguf", "--layers", "0-3,6-8")
        self.assertEqual(code, 2, out + err)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(stale.read_bytes(), foreign)
        # one that appears between the existence check and the exclusive create is not ours either
        out_path = self.dir / "race" / "lite.gguf"
        with mock.patch.object(tool, "open", racing_open(".gguf.partial", foreign), create=True):
            code, out, err = self.run_tool(src, "--output", out_path, "--layers", "0-3,6-8")
        self.assertEqual(code, 2, out + err)
        self.assertEqual(sorted(p.name for p in out_path.parent.iterdir()), ["lite.gguf.partial"])
        self.assertEqual(out_path.with_name("lite.gguf.partial").read_bytes(), foreign)
        # a racing manifest .partial: our complete output .partial goes, theirs stays
        out_path = self.dir / "race-manifest" / "lite.gguf"
        with mock.patch.object(tool, "open", racing_open(".manifest.json.partial", foreign), create=True):
            code, out, err = self.run_tool(src, "--output", out_path, "--layers", "0-3,6-8")
        self.assertEqual(code, 2, out + err)
        self.assertEqual(sorted(p.name for p in out_path.parent.iterdir()), ["lite.gguf.manifest.json.partial"])
        self.assertEqual(out_path.with_name("lite.gguf.manifest.json.partial").read_bytes(), foreign)

    def test_quota_check_refuses_and_accepts_at_the_boundary(self):
        src = self.source()
        volume = self.dir / "volume"
        (volume / "models" / "other").mkdir(parents=True)
        (volume / "out").mkdir()
        (volume / "models" / "a.bin").write_bytes(b"a" * 1000)
        (volume / "models" / "other" / "b.bin").write_bytes(b"b" * 5000)
        os.link(volume / "models" / "other" / "b.bin", volume / "models" / "b-link.bin")   # one inode
        used, apparent, allocated = du_oracle(volume)
        usage = tool.tree_usage(volume)
        self.assertEqual((usage["used_bytes"], usage["apparent_bytes"], usage["allocated_bytes"]),
                         (used, apparent, allocated))
        self.assertEqual(usage["hard_links_counted_once"], 1)
        _, planned = self.planned_bytes(src, [0, 1, 2, 3, 6, 7, 8])
        reserve = 1 << 30                                       # --min-free-gib 1
        quota = used + planned + reserve
        output = volume / "out" / "lite.gguf"
        args = [src, "--output", output, "--layers", "0-3,6-8", "--quota-root", volume, "--min-free-gib", "1"]

        # one byte over: refused before anything is written; the dry-run says the same
        code, out, err = self.run_tool(*args, "--quota-bytes", quota - 1)
        self.assertEqual(code, 2, out + err)
        self.assertRegex(err, rf"quota: .* already uses {used:,} B \(.*\); the output needs {planned:,} B plus a "
                              rf"{reserve:,} B reserve .* exceeds --quota-bytes {quota - 1:,} by 1 B")
        self.assertEqual(list((volume / "out").iterdir()), [])
        code, out, _ = self.run_tool(*args, "--dry-run", "--quota-bytes", quota - 1)
        self.assertEqual(code, 1)
        self.assertIn("quota check [FAIL]", out)
        code, out, _ = self.run_tool(*args, "--dry-run", "--quota-bytes", quota)
        self.assertEqual(code, 0)
        self.assertIn("quota check [pass]", out)

        # exactly at the quota: written, at the planned size, with the check in the manifest
        code, out, err = self.run_tool(*args, "--quota-bytes", quota)
        self.assertEqual(code, 0, out + err)
        self.assertEqual(output.stat().st_size, planned)
        q = json.loads(output.with_name("lite.gguf.manifest.json").read_text(encoding="utf-8"))["quota"]
        self.assertEqual((q["status"], q["used_bytes"], q["planned_output_bytes"], q["reserve_bytes"],
                          q["headroom_bytes"]), ("pass", used, planned, reserve, 0))

        # the output must sit under the root the quota covers; both flags go together
        (self.dir / "elsewhere").mkdir()
        code, out, err = self.run_tool(src, "--output", self.dir / "elsewhere" / "x.gguf", "--layers", "0-3,6-8",
                                       "--quota-root", volume, "--quota-bytes", "1TB")
        self.assertEqual(code, 2, out + err)
        self.assertIn("is not under --quota-root", err)
        self.assertEqual(list((self.dir / "elsewhere").iterdir()), [])
        with self.assertRaises(SystemExit):
            self.run_tool(src, "--output", self.dir / "x.gguf", "--quota-bytes", "1TB")
        # units are explicit: GB and GiB differ by 7 %, the gap that hid the pod's headroom
        self.assertEqual([tool.byte_count(t) for t in ("1234", "500GB", "500GiB", "2 TiB")],
                         [1234, 500 * 10**9, 500 * 2**30, 2 * 2**40])
        for bad in ("500G", "0", "-5", "1.5GB", "GB"):
            with self.assertRaises(tool.argparse.ArgumentTypeError, msg=bad):
                tool.byte_count(bad)

    def test_quota_usage_is_the_larger_of_apparent_and_allocated(self):
        volume = self.dir / "volume"
        (volume / "sub").mkdir(parents=True)
        (volume / "small.bin").write_bytes(b"s" * 100)
        (volume / "sub" / "big.bin").write_bytes(b"b" * 50000)

        def rounded(st):                   # 4 KiB blocks: never less than the length, small.bin far more
            return -(-st.st_size // 4096) * 4096

        def sparse(st):                    # big.bin mostly holes: it allocates less than its length
            return 4096 if st.st_size == 50000 else rounded(st)

        for name, fake, expect in (("block rounding", rounded, "allocated"), ("sparse", sparse, "apparent"),
                                   ("no st_blocks", lambda st: None, "apparent")):
            with self.subTest(name):
                used, apparent, allocated = du_oracle(volume, fake)
                with mock.patch.object(tool, "_allocated_bytes", fake):
                    usage = tool.tree_usage(volume)
                self.assertEqual((usage["used_bytes"], usage["apparent_bytes"], usage["allocated_bytes"]),
                                 (used, apparent, allocated))
                self.assertTrue(usage["basis"].startswith(expect), usage["basis"])
                self.assertEqual(used, apparent if expect == "apparent" else allocated)
        # a directory it cannot list would be undercounted: refuse instead
        real_scandir = os.scandir

        def unreadable(path):
            if Path(path) == volume / "sub":
                raise PermissionError(errno.EACCES, "Permission denied", str(path))
            return real_scandir(path)

        with mock.patch.object(tool.os, "scandir", unreadable):
            with self.assertRaisesRegex(tool.TruncateError, r"cannot list .*sub \(Permission denied\)"):
                tool.tree_usage(volume)

    def test_candidate_budget_override_engages_pruning_on_short_texts(self):
        # Fixture: indexer.top_k 64, candidate budget 16 blocks x 8 rows, candidate
        # source 6 (ratio 1) -> new layer 4. Unchanged, pruning starts at token 129.
        p = "deepseek41.attention.indexer."
        src = self.source()
        plan = tool.make_plan(tool.SourceModel(src), [0, 1, 2, 3, 6, 7, 8])
        self.assertEqual((plan.candidate_budget["new"], plan.candidate_budget["prunes_from_token"]), (16, 129))
        self.assertTrue(any("--candidate-top-k-blocks 9 engages it from token 73" in n for n in plan.notes), plan.notes)
        err = self.refuse_with(src, "0-3,6-8", "--candidate-top-k-blocks", "8")
        self.assertRegex(err, r"can leave 57 candidate rows, fewer than indexer.top_k 64.*Use 9 or more")
        self.assertRegex(self.refuse_with(src, "0-3", "--candidate-top-k-blocks", "9"), r"keeps no candidate source")
        code, text, err = self.run_tool(src, "--dry-run", "--layers", "0-3,6-8", "--candidate-top-k-blocks", "9")
        self.assertEqual(code, 0, text + err)
        self.assertIn("candidate pruning: 16 -> 9 blocks x 8 rows (indexer top_k 64, minimum 9 blocks); "
                      "prunes from token 73", text)

        output = self.dir / "pruned.gguf"
        code, text, err = self.run_tool(src, "--output", output, "--layers", "0-3,6-8", "--candidate-top-k-blocks", "9")
        self.assertEqual(code, 0, text + err)
        sf, _ = read_gguf(src)
        of, _ = read_gguf(output)
        self.assertEqual(of[p + "candidate_top_k_blocks"], ([V.UINT32], 9))
        # the spelling the PR #306-era loaders (the ggml_cuda reference build) read
        self.assertEqual(of[p + "candidate_top_k"], ([V.UINT32], 9))
        self.assertEqual(of["tensorsharp.truncate.candidate_top_k_blocks_source"], ([V.UINT32], 16))
        self.assertEqual(of["tensorsharp.truncate.candidate_top_k_blocks"], ([V.UINT32], 9))
        self.assertEqual(of[p + "candidate_block_size"], sf[p + "candidate_block_size"])
        self.assertEqual(of[p + "top_k"], sf[p + "top_k"])
        # --verify replays the override from the provenance; contradicting it fails
        code, text, err = self.run_tool(src, "--verify", "--output", output)
        self.assertEqual(code, 0, text + err)
        self.assertEqual(json.loads(text)["candidate_budget"]["new"], 9)
        code, text, _ = self.run_tool(src, "--verify", "--output", output, "--candidate-top-k-blocks", "10")
        self.assertEqual(code, 1)
        # a descendant keeps the budget it inherits and still names the original one
        again = self.dir / "again.gguf"
        code, text, err = self.run_tool(output, "--output", again, "--layers", "0-4,6")
        self.assertEqual(code, 0, text + err)
        af, _ = read_gguf(again)
        self.assertEqual(af[p + "candidate_top_k_blocks"][1], 9)
        self.assertEqual(af[p + "candidate_top_k"][1], 9)
        self.assertEqual(af["tensorsharp.truncate.candidate_top_k_blocks_source"][1], 16)
        self.assertNotIn("tensorsharp.truncate.candidate_top_k_blocks", af)
        self.assertEqual(af["tensorsharp.truncate.origin_layers"][1], [0, 1, 2, 3, 6, 8])

        # both spellings present (TensorSharp's own candidate_top_k too): both move, or a loader refuses
        both = self.source("both", extra_kv=[(p + "candidate_top_k", 16, V.UINT32, None)])
        out2 = self.dir / "both.gguf"
        code, text, err = self.run_tool(both, "--output", out2, "--layers", "0-3,6-8", "--candidate-top-k-blocks", "12")
        self.assertEqual(code, 0, text + err)
        bf, _ = read_gguf(out2)
        self.assertEqual((bf[p + "candidate_top_k_blocks"][1], bf[p + "candidate_top_k"][1]), (12, 12))

    def test_candidate_budget_is_read_alike_by_pr306_loaders(self):
        # The PR #306-era loaders (dsv41_engram_gguf.h / Dsv41EngramData.cs; the ggml_cuda reference
        # build on the pods) read candidate_top_k only, default 2048; the current ones read
        # candidate_top_k_blocks first. S8 round 4 wrote the override into candidate_top_k_blocks
        # alone: the reference kept every block while the current loaders pruned.
        p = "deepseek41.attention.indexer."
        name = "candidate budget read alike by current and PR #306-era loaders"

        def pr306(values):
            return int(values[p + "candidate_top_k"].value) if p + "candidate_top_k" in values else 2048

        def current(values):
            key = p + "candidate_top_k_blocks" if p + "candidate_top_k_blocks" in values else p + "candidate_top_k"
            return int(values[key].value) if key in values else 2048

        src = self.source()                                  # candidate_top_k_blocks 16 alone
        model = tool.SourceModel(src)
        for override, budget in ((None, 16), (9, 9), (2048, 2048)):
            plan = tool.make_plan(model, [0, 1, 2, 3, 6, 7, 8], candidate_top_k_blocks=override)
            values = tool.planned_values(plan)
            self.assertEqual((current(values), pr306(values)), (budget, budget), override)
            statuses = {c["check"]: c["status"] for c in tool.loader_contract(values, tool.planned_shapes(plan))}
            self.assertEqual(statuses[name], "pass", override)
        self.assertNotIn(p + "candidate_top_k", values)       # 2048 is what both read without it
        # the round-4 shape: the override in candidate_top_k_blocks alone fails the check
        values = dict(tool.planned_values(tool.make_plan(model, [0, 1, 2, 3, 6, 7, 8], candidate_top_k_blocks=9)))
        del values[p + "candidate_top_k"]
        checks = {c["check"]: c for c in tool.loader_contract(values, None)}
        self.assertEqual(checks[name]["status"], "FAIL")
        self.assertIn("read 9 blocks, PR #306-era ones (candidate_top_k only, default 2048) read 2048",
                      checks[name]["detail"])
        # written and verified: the reference reads the fixture's 16 too
        out = self.truncate(src, "0-3,6-8")
        of, _ = read_gguf(out)
        self.assertEqual(of[p + "candidate_top_k"], ([V.UINT32], 16))
        code, text, err = self.run_tool(src, "--verify", "--output", out)
        self.assertEqual(code, 0, text + err)
        self.assertTrue(any(c["check"] == "loader: " + name and c["status"] == "pass" for c in json.loads(text)["checks"]))
        # no candidate source kept: no loader reads the budget, nothing is added
        out = self.truncate(src, "0-3", name="nocand.gguf")
        of, _ = read_gguf(out)
        self.assertNotIn(p + "candidate_top_k", of)

    def refuse_with(self, source, layers, *extra):
        output = self.dir / "refused.gguf"
        code, out, err = self.run_tool(source, "--output", output, "--layers", layers, *extra)
        self.assertEqual(code, 2, out + err)
        self.assertFalse(output.exists() or output.with_name("refused.gguf.partial").exists())
        return err

    def test_engram_check_reads_rows_from_every_position(self):
        # The row list is [position][hash column]; a fixed-length prefix of it only
        # ever reaches the first positions (on 1017's 24 columns, position 0 alone).
        src = self.source()
        out = self.truncate(src, "0,2-6")               # Engram layer 5 -> new layer 4, rows [0, 88)
        of, _ = read_gguf(out)
        columns = (NGRAM - 1) * HEADS
        tokens = late = None
        for candidate in ([7, 3, 11, 0, 4, 9, 1, 2, 5, 6, 8, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]):
            rows = oracle_rows(of, 0, candidate)
            only_late = sorted(set(rows[4 * columns:]) - set(rows[:4 * columns]))
            if only_late:
                tokens, late = candidate, only_late[0]
                break
        self.assertIsNotNone(late)
        header = tool.GgufHeader(out)
        table = next(t for t in header.tensors if t.name == "per_layer_token_embd.weight")
        with open(out, "r+b") as stream:
            stream.seek(table.abs_offset + late * ROW + 7)
            byte = stream.read(1)
            stream.seek(-1, 1)
            stream.write(bytes([byte[0] ^ 0x5A]))
        model = tool.SourceModel(src)
        plan = tool.make_plan(model, [0, 2, 3, 4, 5, 6])
        with tool._Files() as files:
            with self.assertRaisesRegex(tool.TruncateError, rf"row {late} bytes differ"):
                tool._engram_equivalence(model, plan, tool.GgufHeader(out), files, 0, 0, token_sequences=[tokens])
            ok = tool._engram_equivalence(model, plan, tool.GgufHeader(out), files, 0, 0, token_sequences=[tokens[:4]])
        self.assertEqual(ok["detail"]["rows_compared"], len(set(oracle_rows(of, 0, tokens[:4]))))

    @unittest.skipUnless(REAL_HEADER.exists(), "the real 1017 header is not staged in artifacts/")
    def test_real_1017_header_candidate_budget(self):
        model = tool.SourceModel(REAL_HEADER, allow_missing=True)
        plan = tool.make_plan(model, tool.DEFAULT_LAYERS, header_only=True)
        self.assertEqual({k: plan.candidate_budget[k] for k in ("source", "block_size", "indexer_top_k", "min_blocks",
                                                                 "candidate_layer_new", "prunes_from_token")},
                         dict(source=2048, block_size=8, indexer_top_k=512, min_blocks=65, candidate_layer_new=4,
                              prunes_from_token=16385))
        # 1017 spells the budget candidate_top_k_blocks = 2048 alone; the PR #306-era loaders' default
        # is also 2048, so the default plan adds nothing
        self.assertNotIn("deepseek41.attention.indexer.candidate_top_k", {c["key"] for c in plan.changes})
        plan = tool.make_plan(model, tool.DEFAULT_LAYERS, header_only=True, candidate_top_k_blocks=65)
        changes = {c["key"]: c for c in plan.changes}
        self.assertEqual(changes["deepseek41.attention.indexer.candidate_top_k_blocks"]["new"], 65)
        self.assertEqual({k: changes["deepseek41.attention.indexer.candidate_top_k"][k] for k in ("action", "new")},
                         dict(action="added", new=65))
        self.assertEqual(plan.candidate_budget["prunes_from_token"], 521)
        statuses = {c["check"]: c["status"] for c in tool.loader_contract(tool.planned_values(plan), None)}
        self.assertNotIn("FAIL", statuses.values())
        self.assertEqual(statuses["candidate budget read alike by current and PR #306-era loaders"], "pass")
        with self.assertRaisesRegex(tool.TruncateError, "Use 65 or more"):
            tool.make_plan(model, tool.DEFAULT_LAYERS, header_only=True, candidate_top_k_blocks=64)

    @unittest.skipUnless(REAL_HEADER.exists(), "the real 1017 header is not staged in artifacts/")
    def test_real_1017_header_default_plan(self):
        plan = tool.make_plan(tool.SourceModel(REAL_HEADER, allow_missing=True), tool.DEFAULT_LAYERS, header_only=True)
        changes = {c["key"]: c for c in plan.changes}
        p = "deepseek41."
        self.assertEqual(changes[p + "block_count"]["new"], 9)
        self.assertEqual(changes[p + "attention.kv_source_layers"]["new"], [2, 4])
        self.assertEqual(changes[p + "attention.indexer.source_layers"]["new"], [2, 4, 8])
        self.assertEqual(changes[p + "attention.indexer.candidate_source_layer"]["new"], 4)
        self.assertEqual(changes[p + "ple.layers"]["new"], [1])
        self.assertEqual(changes[p + "attention.compress_ratios"]["new"], [0, 0, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0])
        self.assertEqual(plan.engram["slices"], [dict(layer_old=1, layer_new=1, source_first_row=0,
                                                      new_first_row=0, rows=384006168)])
        self.assertEqual(plan.unclassified, [])


if __name__ == "__main__":
    unittest.main()
