"""Fixture expert dimensions and quantization must match its published metadata."""
import json
from contextlib import ExitStack
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from gguf import GGUFReader


class FixtureGeometryTests(unittest.TestCase):
    def generate(self, output, *extra):
        return subprocess.run([sys.executable, str(Path(__file__).parents[1] / "dsv41-fixture.py"),
            str(output), *extra], capture_output=True, text=True)

    def check_geometry(self, width, extra):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            output = Path(directory)
            result = self.generate(output, *extra)
            self.assertEqual(0, result.returncode, result.stderr)
            config = json.loads((output / "deepseek41.config.json").read_text())["config"]["text_config"]
            self.assertEqual(width, config["moe_intermediate_size"])
            reader = GGUFReader(output / "deepseek41-fixture.gguf")
            cleanup.callback(reader.data._mmap.close)
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            self.assertEqual(width, reader.fields["deepseek41.expert_feed_forward_length"].contents())
            for layer in range(config["num_hidden_layers"]):
                prefix = f"blk.{layer}.ffn_"
                for projection in ("gate", "up", "down"):
                    tensor = tensors[prefix + projection + "_exps.weight"]
                    expected = [width, 256, 4] if projection == "down" else [256, width, 4]
                    self.assertEqual(expected, list(tensor.shape))
                    shared = tensors[prefix + projection + "_shexp.weight"]
                    self.assertEqual(expected[:2], list(shared.shape))
                    expected_type = "Q3_K" if projection == "down" and "--k-mix" in extra else "Q2_K"
                    self.assertEqual(expected_type, tensor.tensor_type.name)
                    self.assertEqual(expected_type, shared.tensor_type.name)

    def test_default_width_remains_256(self):
        self.check_geometry(256, [])

    def test_nine_block_mixed_quantization_width_matches_all_routed_and_shared_tensors(self):
        self.check_geometry(2304, ["--k-mix", "--cuda-index", "--moe-intermediate-size", "2304"])

    def test_nonpositive_or_unaligned_width_fails_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in ("0", "-256", "257"):
                output = Path(directory) / value
                result = self.generate(output, "--moe-intermediate-size", value)
                self.assertNotEqual(0, result.returncode)
                self.assertIn("positive multiple of 256", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__": unittest.main()
