"""Embedded V4.1 metadata contract tests using real, tiny GGUF files."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


metadata = module("dsv41_gguf", "dsv41-gguf.py")


class EmbeddedEngramTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def weights(self, changes=None, omit=(), graph_changes=None):
        values = {"layer_ids": [1], "token_map": [2, 0, 1, 2, 0], "pad_id": 1,
                  "max_ngram_size": 3, "head_count": 2, "key_length": 4,
                  "multipliers": [3, 5, 7], "primes": [5, 7, 11, 13], "offsets": [0, 5, 12, 23]}
        values.update(changes or {})
        path = Path(self.directory.name) / "test.gguf"
        writer = GGUFWriter(path, "deepseek41")
        graph = {"block_count": 3, "attention.compress_ratios": [0, 1, 1]}
        graph.update(graph_changes or {})
        for key, value in graph.items():
            name = "deepseek41." + key
            if isinstance(value, list):
                kind = GGUFValueType.FLOAT32 if any(isinstance(x, float) for x in value) else GGUFValueType.INT64
                writer.add_key_value(name, value, GGUFValueType.ARRAY, kind)
            elif isinstance(value, float):
                writer.add_float32(name, value)
            else:
                writer.add_int64(name, value)
        writer.add_token_list(["A", "B", "<pad>", "a", "b"])
        for key, value in values.items():
            if key in omit:
                continue
            name = "deepseek41.engram." + key
            if isinstance(value, list):
                kind = GGUFValueType.FLOAT32 if any(isinstance(x, float) for x in value) else GGUFValueType.UINT64
                writer.add_key_value(name, value, GGUFValueType.ARRAY, kind)
            else:
                writer.add_uint32(name, value)
        writer.add_tensor("blk.1.engram_embd.weight", np.zeros((36, 4), dtype=np.float32))
        writer.add_tensor("token_embd.weight", np.zeros((5, 4), dtype=np.float32))
        writer.add_tensor("blk.1.attn_compressor_kv.weight", np.zeros((4, 4), dtype=np.float32))
        writer.add_tensor("blk.1.indexer.attn_q_b.weight", np.zeros((4, 4), dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reader = GGUFReader(path)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        return SimpleNamespace(readers=[reader], tensors=tensors,
                               shape=lambda name: tuple(int(x) for x in tensors[name].shape))

    def test_embedded_map_layout_and_checkpoint_defaults(self):
        result = metadata.load_engram(self.weights())
        self.assertEqual(result["token_map"].tolist(), [2, 0, 1, 2, 0])
        self.assertEqual((result["compressed"], result["pad"]), (3, 1))
        self.assertEqual((result["kv_sources"], result["index_sources"]), ([1], [1]))
        self.assertEqual((result["candidate"], result["candidate_topk"], result["candidate_block"]), (1, 2048, 8))
        self.assertEqual(result["layouts"][0]["rows"], 36)

    def test_missing_arrays_and_malformed_values_are_rejected(self):
        for name in ("token_map", "pad_id", "multipliers", "primes", "offsets"):
            with self.subTest(missing=name), self.assertRaises(ValueError):
                metadata.load_engram(self.weights(omit=(name,)))
        for changed in ({"token_map": [2, 0, 1, 2]}, {"token_map": [2, 0, 1, 4, 0]},
                        {"token_map": [0, 0, 0, 0, 0], "pad_id": 1}, {"pad_id": 3},
                        {"multipliers": [2, 5, 7]}, {"multipliers": [3.0, 5.0, 7.0]},
                        {"multipliers": [3, 5]}, {"offsets": [0, 5, 13, 23]},
                        {"primes": [5, 7, 11, 17]}, {"layer_ids": [2]},
                        {"max_ngram_size": 17}, {"head_count": 129}, {"key_length": 65537}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                metadata.load_engram(self.weights(changed))

    def test_vocab_tensor_and_candidate_integer_contract(self):
        weights = self.weights()
        del weights.tensors["token_embd.weight"]
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            metadata.load_engram(weights)
        weights = self.weights()
        weights.tensors["blk.2.engram_embd.weight"] = weights.tensors["blk.1.engram_embd.weight"]
        with self.assertRaisesRegex(ValueError, "layer IDs"):
            metadata.load_engram(weights)
        weights = self.weights()
        for kind, value in ((GGUFValueType.FLOAT32, 2.5), (GGUFValueType.UINT64, 2**32)):
            weights.readers[0].fields["deepseek41.attention.indexer.candidate_top_k"] = SimpleNamespace(types=[kind], contents=lambda: value)
            with self.assertRaises(ValueError):
                metadata.load_engram(weights)

    def test_candidate_budget_is_read_under_either_name(self):
        # Unsloth's 1017 GGUFs name the budget candidate_top_k_blocks; the native and
        # managed loaders read either name, and a file with both must agree.
        weights = self.weights()
        fields = weights.readers[0].fields

        def budget(**values):
            for name in ("candidate_top_k", "candidate_top_k_blocks"):
                fields.pop("deepseek41.attention.indexer." + name, None)
            for name, value in values.items():
                fields["deepseek41.attention.indexer." + name] = SimpleNamespace(
                    types=[GGUFValueType.UINT32], contents=lambda value=value: value)
            return metadata.load_engram(weights)["candidate_topk"]
        self.assertEqual(budget(candidate_top_k_blocks=5), 5)
        self.assertEqual(budget(candidate_top_k=6), 6)
        self.assertEqual(budget(candidate_top_k_blocks=5, candidate_top_k=5), 5)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            budget(candidate_top_k_blocks=5, candidate_top_k=6)

    def test_block_count_and_compression_metadata_match_runtime_contract(self):
        for changed in ({"block_count": 3.0}, {"block_count": 3.5}, {"block_count": -1},
                        {"attention.compress_ratios": [0.0, 1.0, 1.0]},
                        {"attention.compress_ratios": [-1, 1, 1]},
                        {"attention.compress_ratios": [0, 1, 1, -1]},
                        {"attention.compress_ratios": [0, 1]},
                        {"attention.compress_ratios": [0, 1, 1] + [0] * 254}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                metadata.load_engram(self.weights(graph_changes=changed))
        # The published checkpoint includes trailing MTP ratios; preserve them
        # as valid metadata while inferring sources only for target blocks.
        for count in (4, 256):
            result = metadata.load_engram(self.weights(graph_changes={
                "attention.compress_ratios": [0, 1, 1] + [0] * (count - 3)}))
            self.assertEqual((result["candidate"], result["candidate_topk"], result["candidate_block"]), (1, 2048, 8))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed (eng/dsv41-reference.py imports it)")
    def test_padding_and_media_barrier_use_already_compressed_id(self):
        reference = module("dsv41_reference", "dsv41-reference.py")
        layout = metadata.load_engram(self.weights())
        # A mapped padding ID of 1 must remain 1; token_map[1] would be 0.
        for history, position in (([0], 0), ([0, -1, 0], 2)):
            oracle = reference.Reference.__new__(reference.Reference)
            oracle.engram, oracle.history, oracle.position = layout, history, position
            oracle.w = SimpleNamespace(rows=Mock(side_effect=RuntimeError("rows captured")))
            with self.assertRaisesRegex(RuntimeError, "rows captured"):
                oracle.eng_inject(1, None, [0])
            oracle.w.rows.assert_called_once_with("blk.1.engram_embd.weight", [3, 8, 16, 27])

    def test_vision_binding_matches_raw_gguf_tokenizer(self):
        weights = self.weights()
        prepare = module("dsv41_prepare_vision", "dsv41-prepare-vision.py")
        expected = prepare.tokenizer_fingerprint({"model": {"vocab": {"A": 0, "B": 1, "<pad>": 2, "a": 3, "b": 4}}, "added_tokens": []}, 5)
        self.assertEqual(metadata.tokenizer_identity(weights.readers[0]), (5, expected))
        self.assertEqual(prepare.parent_tokenizer_identity(Path(self.directory.name) / "test.gguf"), (5, expected))


if __name__ == "__main__":
    unittest.main()
