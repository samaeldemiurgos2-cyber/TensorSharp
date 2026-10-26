from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gguf import GGUFReader


class FixturePreparationTests(unittest.TestCase):
    def test_optional_http_vocabulary_preserves_every_tensor_and_default_tokens(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            root = Path(directory)
            repo = Path(__file__).parents[3]
            subprocess.run([sys.executable, str(repo / "eng/dsv41-fixture.py"), str(root / "source"), "--f32"],
                           capture_output=True, check=True)
            source = root / "source/deepseek41-fixture.gguf"
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            original = GGUFReader(source)
            cleanup.callback(original.data._mmap.close)
            for mode in ("numerical", "http"):
                command = [sys.executable, str(repo / "eng/validation/prepare-dsv41-managed-fixture.py"),
                    "--source", str(source), "--source-sha256", source_sha, "--output-dir", str(root / mode)]
                if mode == "http": command.append("--byte-tokenizer")
                subprocess.run(command, capture_output=True, check=True)
                manifest = json.loads((root / mode / "manifest.json").read_text())
                self.assertEqual(142, manifest["unchanged_tensor_count"])
                self.assertEqual(mode == "http", manifest["http_only_byte_tokenizer"])
                self.assertEqual(142, len(manifest["tensors"]))
                derived = GGUFReader(root / mode / "deepseek41-fixture.gguf")
                cleanup.callback(derived.data._mmap.close)
                self.assertEqual([], derived.fields["tokenizer.ggml.merges"].contents())
                tokens = derived.fields["tokenizer.ggml.tokens"].contents()
                if mode == "numerical":
                    self.assertEqual(original.fields["tokenizer.ggml.tokens"].contents(), tokens)
                    self.assertEqual({"tokenizer.ggml.merges": []}, manifest["metadata_change"])
                else:
                    self.assertEqual(256, len(set(tokens)))
                    self.assertEqual("A", tokens[65])
                    self.assertEqual("Ġ", tokens[32])
                    self.assertEqual("Ā", tokens[0])
                    self.assertEqual(tokens, manifest["metadata_change"]["tokenizer.ggml.tokens"])
                for left, right in zip(original.tensors, derived.tensors):
                    self.assertEqual(left.name, right.name)
                    self.assertEqual(left.data.tobytes(), right.data.tobytes())


if __name__ == "__main__": unittest.main()
