"""Resource limits and shard manifests must reflect the selected checkpoint."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys


ROOT = Path(__file__).resolve().parents[1] / 'validation'


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


artifacts = module('muse_spark_artifacts')
resources = module('inference_resource_limits')
sys.path.insert(0, str(ROOT))
try:
    sampler = module('sample-inference-hardware')
finally:
    sys.path.remove(str(ROOT))


class MuseSparkResourceTests(unittest.TestCase):
    def manifest(self, quantization, count):
        return {'siblings': [{'rfilename': f'{quantization}/Muse-Spark-Open-816B-A42B-'
                              f'{quantization}-{index:05d}-of-{count:05d}.gguf'}
                             for index in range(1, count + 1)]}

    def test_quantization_selects_its_complete_variable_shard_set(self):
        q3, q4 = self.manifest('UD-Q3_K_XL', 10), self.manifest('UD-Q4_K_XL', 12)
        together = {'siblings': q3['siblings'] + q4['siblings']}
        self.assertEqual(q3['siblings'], artifacts.shard_entries(together, 'UD-Q3_K_XL'))
        self.assertEqual(q4['siblings'], artifacts.shard_entries(together, 'UD-Q4_K_XL'))

    def test_missing_duplicate_or_mixed_count_shards_are_rejected(self):
        rows = self.manifest('UD-Q3_K_XL', 10)['siblings']
        for invalid in (rows[1:], rows + [rows[0]],
                        rows[:-1] + [{'rfilename': rows[-1]['rfilename'].replace('of-00010', 'of-00011')}]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    artifacts.shard_entries({'siblings': invalid}, 'UD-Q3_K_XL')

    def files(self, root, values):
        for name, value in values.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(value))

    def test_v1_container_limits_are_independent_of_host_capacity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.files(root, {'memory/memory.limit_in_bytes': 250999996416,
                              'memory/memory.usage_in_bytes': 200000000000,
                              'cpu/cpu.cfs_quota_us': 4080000, 'cpu/cpu.cfs_period_us': 100000,
                              'cpuset/cpuset.cpus': '0-95'})
            actual = resources.cgroup_snapshot(root)
            self.assertEqual(250999996416, actual['memory_limit_bytes'])
            self.assertEqual(40.8, actual['cpu_quota_cores'])
            self.assertNotIn('memory_current_bytes', resources.stable_limits(actual))

    def test_v2_unlimited_and_finite_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.files(root, {'memory.max': 'max', 'memory.current': 4096,
                              'cpu.max': 'max 100000', 'cpuset.cpus.effective': '0-3'})
            actual = resources.cgroup_snapshot(root)
            self.assertIsNone(actual['memory_limit_bytes'])
            self.assertIsNone(actual['cpu_quota_cores'])
            self.files(root, {'memory.max': '8192', 'cpu.max': '200000 100000'})
            actual = resources.cgroup_snapshot(root)
            self.assertEqual(8192, actual['memory_limit_bytes'])
            self.assertEqual(2, actual['cpu_quota_cores'])

    def test_missing_resource_files_are_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual('unavailable', resources.cgroup_snapshot(Path(folder))['status'])

    def test_process_stat_separates_own_major_faults_from_child_faults(self):
        fields = ['0'] * 22
        for index, value in {0: 'R', 7: '100', 9: '7', 10: '999', 11: '200',
                             12: '100', 17: '5', 19: '123456', 21: '2048'}.items():
            fields[index] = value
        actual = sampler.parse_process_stat('123 (a worker (name)) ' + ' '.join(fields), 100, 4096)
        self.assertEqual({'start_ticks': 123456, 'cpu_seconds': 3.0, 'rss_mib': 8,
                          'threads': 5, 'minor_faults': 100, 'major_faults': 7}, actual)


if __name__ == '__main__':
    unittest.main()
