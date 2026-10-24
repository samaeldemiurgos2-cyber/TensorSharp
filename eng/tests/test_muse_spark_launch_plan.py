"""Incomplete downloads and metadata-only first shards must not yield false plans."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'validation/muse-spark-launch-plan.py'
SPEC = importlib.util.spec_from_file_location('spark_launch_plan', SOURCE)
plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plan)


class LaunchPlanTests(unittest.TestCase):
    def shards(self, root, count):
        directory = root / 'UD-Q3_K_XL'
        directory.mkdir()
        paths = []
        for index in range(1, count + 1):
            path = directory / f'Muse-Spark-Open-816B-A42B-UD-Q3_K_XL-{index:05d}-of-{count:05d}.gguf'
            path.write_bytes(b'fixture')
            paths.append(path)
        return paths

    def test_q3_shard_count_is_inferred_and_first_shard_alone_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.shards(root, 10)
            self.assertEqual(paths, plan.discover_shards(root, 'UD-Q3_K_XL'))
            paths[-1].unlink()
            with self.assertRaisesRegex(ValueError, 'all 10'):
                plan.discover_shards(root, 'UD-Q3_K_XL')

    def test_manifest_sizes_and_filenames_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.shards(root, 2)
            manifest = {'siblings': [{'rfilename': f'UD-Q3_K_XL/{p.name}', 'size': 7} for p in paths]}
            self.assertEqual(paths, plan.discover_shards(root, 'UD-Q3_K_XL', manifest))
            manifest['siblings'][0]['size'] = 8
            with self.assertRaisesRegex(ValueError, 'wrong-size'):
                plan.discover_shards(root, 'UD-Q3_K_XL', manifest)
            manifest['siblings'].pop()
            with self.assertRaisesRegex(ValueError, 'filenames'):
                plan.discover_shards(root, 'UD-Q3_K_XL', manifest)

    def test_mixed_models_and_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = self.shards(root, 2)
            paths[1].rename(paths[1].with_name('Other-00002-of-00002.gguf'))
            with self.assertRaisesRegex(ValueError, 'Mixed'):
                plan.discover_shards(root, 'UD-Q3_K_XL')
            with self.assertRaisesRegex(ValueError, 'single directory'):
                plan.discover_shards(root, '../UD-Q3_K_XL')

    def test_kv_and_rank_zero_reserve_change_placement(self):
        # Weights alone favor three tiny offloaded layers on rank0. Charging KV
        # and its projector moves a layer away before planning GPU admission.
        self.assertEqual([3, 1], plan.partition([1, 1, 1, 9], 0, 0, 2))
        self.assertEqual([2, 2], plan.partition([5, 5, 5, 13], 8, 0, 2))
        with self.assertRaises(ValueError):
            plan.partition([1], 0, 0, 2)

    def test_stream_stack_is_reserved_on_the_rank_that_owns_the_layer(self):
        self.assertEqual([1, 3], plan.partition([1, 1, 1, 9], 0, 0, 2, [9, 0, 0, 0]))
        with self.assertRaises(ValueError):
            plan.partition([1, 1], 0, 0, 2, [1])


if __name__ == '__main__':
    unittest.main()
