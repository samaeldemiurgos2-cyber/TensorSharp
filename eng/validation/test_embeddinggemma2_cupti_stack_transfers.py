import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('stack_transfers', Path(__file__).with_name('embeddinggemma2-cupti-stack-transfers.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class StackTransferTests(unittest.TestCase):
    def fixture(self):
        def event(start, end, **extra):
            return dict(start_ns=start, end_ns=end, device=0, context=1, stream=13, **extra)
        def copy_event(start, direction):
            return event(start, start + 1, direction=direction, bytes=36 * 768 * 4, copy_count=1)
        copies = [copy_event(1, 'HtoD'), copy_event(3, 'DtoH'), copy_event(10, 'HtoD'), copy_event(180, 'DtoH')]
        # Hidden-sized internal device copies are explicitly excluded.
        copies.append(copy_event(20, 'DtoD'))
        kernels = [event(12 + i * 10, 13 + i * 10, name='vision_kernel<64>') for i in range(16)]
        report = dict(schema_version=2, completed=True, diagnostic=True, hardware_counters=False,
            cupti_api_version=130401, errors=[], dropped=0, incomplete=0, callback_exceptions=0,
            truncated=0, invalid_buffers=0, kernel_count=16, memcpy_record_count=5,
            kernel_events=kernels, memcpy_events=copies)
        probe = dict(completed=True, backend='GGML_CUDA', require_validated_float32_precision=True,
            require_ggml_cuda_stack=True, ggml_cuda_stack_observation_available=True,
            ggml_cuda_stack_encode_count=1, retained_block_loop_encode_count=0,
            raw_tower_benchmark=None, cupti_activity_output='/capture.json', width=96, height=96)
        return report, probe

    def test_one_pair_brackets_all_blocks_and_excludes_patch_and_d2d(self):
        report, probe = self.fixture()
        result = module.attribute(report, probe, 36)
        self.assertEqual(result['total_stack_copy_bytes'], 2 * 36 * 768 * 4)
        self.assertEqual(result['stack_transfers']['HtoD']['count'], 1)
        self.assertEqual(len(result['stack']['attention_markers']), 16)
        self.assertEqual(result['excluded_hidden_sized_d2d_records'], 1)
        self.assertFalse(result['paired_performance_claim'])

    def test_route_observation_shape_and_quality_scope_cannot_be_inferred(self):
        _, original = self.fixture()
        for change in (dict(ggml_cuda_stack_encode_count=0, retained_block_loop_encode_count=1),
                       dict(ggml_cuda_stack_observation_available=False),
                       dict(require_ggml_cuda_stack=False), dict(backend='GGML_CPU'),
                       dict(raw_tower_benchmark={}), dict(vision_profile='1'),
                       dict(width=112), dict(completed=False), dict(ggml_cuda_stack_encode_count=2)):
            report, _ = self.fixture(); probe = copy.deepcopy(original); probe.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                module.attribute(report, probe, 36)

    def test_non_pooling_multiple_patch_grid_matches_production(self):
        report, probe = self.fixture()
        probe.update(width=192, height=208)
        for event in report['memcpy_events']:
            event['bytes'] = 156 * 768 * 4
        result = module.attribute(report, probe, 156)
        self.assertEqual(result['total_stack_copy_bytes'], 2 * 156 * 768 * 4)

    def test_extra_missing_batched_cross_context_and_reordered_events_reject(self):
        for mutation in (lambda r: r['memcpy_events'].pop(3),
                         lambda r: r['memcpy_events'].append(copy.deepcopy(r['memcpy_events'][2])),
                         lambda r: r['memcpy_events'][2].update(copy_count=2),
                         lambda r: r['memcpy_events'][0].update(context=2),
                         lambda r: r['kernel_events'][7].update(context=2),
                         lambda r: r['kernel_events'][7].update(stream=2),
                         lambda r: r['kernel_events'][7].update(name='vision_kernel<72>'),
                         lambda r: r['kernel_events'][7].update(start_ns=10),
                         lambda r: r['memcpy_events'][3].update(start_ns=150),
                         lambda r: r['memcpy_events'][2].update(start_ns=2),
                         lambda r: r['kernel_events'].pop(),
                         lambda r: r.update(incomplete=1)):
            report, probe = self.fixture(); mutation(report)
            # Keep record counts truthful to ensure semantic gates are tested.
            report['kernel_count'] = len(report['kernel_events'])
            report['memcpy_record_count'] = len(report['memcpy_events'])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                module.attribute(report, probe, 36)

    def test_cuda12_api26_capture_is_rejected(self):
        report, probe = self.fixture()
        report['cupti_api_version'] = 26
        with self.assertRaisesRegex(ValueError, 'CUDA 13'):
            module.attribute(report, probe, 36)


if __name__ == '__main__':
    unittest.main()
