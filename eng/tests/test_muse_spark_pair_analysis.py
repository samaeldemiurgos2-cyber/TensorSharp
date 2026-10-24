import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('spark_pair_analysis', ROOT / 'eng/validation/analyze-muse-spark-paired.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def fixture(engine):
    flag = '--no-prefix-cache' if engine == 'tensorsharp' else '--no-cache-prompt'
    def case(name, repeat):
        return {'scenario': name, 'tag': f'{name}-r{repeat}', 'repeat': repeat, 'status': 'ok',
            'input_sha256': name, 'total_wall_ms': 1000,
            'turns': [{'request': {'stream': True, 'cache_prompt': False}, 'metrics': {
                'prompt_tokens': 100, 'completion_tokens': 512, 'raw_usage': {'prompt_tokens_details': {'cached_tokens': 0}},
                't_start_abs': 0, 't_first_abs': .1, 't_last_abs': .9, 't_end_abs': 1,
                'finish_reason': 'length', 'output_text': 'synthetic test only'}}]}
    return {'engine': engine, 'weights_id': 'synthetic', 'profile': 'synthetic', 'thinking': False,
        'sampling': {}, 'scenarios': ['short', 'decode', 'long_8k'], 'repeats': 3, 'concurrency': 4,
        'harness_sha256': {'synthetic': 'test'}, 'cache_policy': 'disabled', 'run_complete': True,
        'provenance': {'hardware_id': 'synthetic', 'weights_sha256': {'synthetic': 'test'},
            'settings': {'prefix_cache': False}, 'upstream_ggml_revision': 'synthetic',
            'upstream_ggml_clean': True, 'server_revision': 'synthetic', 'server_command': flag,
            'server_log': 'synthetic', 'launch_observed': True},
        'warmup': case('warmup', 0), 'cases': [case(n, i) for i in range(3) for n in ['short', 'decode', 'long_8k']]}


class MuseSparkPairAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.candidate, self.reference = fixture('tensorsharp'), fixture('llamacpp')

    def test_complete_equal_work_preserves_all_repeats_and_original_verdict(self):
        before = copy.deepcopy(self.candidate)
        result = analysis.analyze(self.candidate, self.reference)
        self.assertEqual('pass', result['original_harness_campaign_verdict']['status'])
        self.assertTrue(result['scenarios']['decode']['equal_work_eligible'])
        self.assertEqual(3, result['scenarios']['decode']['task_latency']['conversation_wall_ms']['candidate']['count'])
        self.assertEqual(before, self.candidate)

    def test_unequal_completion_keeps_campaign_inconclusive_and_task_ratio_descriptive(self):
        self.reference['cases'][2]['turns'][0]['metrics']['completion_tokens'] = 513
        result = analysis.analyze(self.candidate, self.reference)
        self.assertEqual('inconclusive', result['original_harness_campaign_verdict']['status'])
        self.assertTrue(result['scenarios']['decode']['equal_work_eligible'])
        item = result['scenarios']['long_8k']
        self.assertFalse(item['equal_work_eligible'])
        self.assertEqual('descriptive_only_unequal_completion_work', item['task_latency']['status'])
        self.assertFalse(item['task_latency']['parity_claim'])
        self.assertIsNone(item['common_client_metrics']['client_first_generated_event_ms']['candidate_speed_ratio'])

    def test_quality_cache_prompt_and_provenance_failures_never_receive_task_ratios(self):
        for mutation in ('quality', 'cache', 'prompt', 'provenance', 'request', 'warmup'):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.reference)
                metrics = bad['cases'][2]['turns'][0]['metrics']
                if mutation == 'quality': bad['cases'][2]['status'] = 'fail'
                elif mutation == 'cache': metrics['raw_usage']['prompt_tokens_details']['cached_tokens'] = 1
                elif mutation == 'prompt': metrics['prompt_tokens'] = 101
                elif mutation == 'provenance': bad['provenance']['launch_observed'] = False
                elif mutation == 'request': bad['cases'][2]['input_sha256'] = 'different'
                elif mutation == 'warmup': bad['warmup']['status'] = 'fail'
                result = analysis.analyze(self.candidate, bad)['scenarios']['long_8k']
                self.assertIsNone(result['task_latency']['reference_over_candidate_median_wall_ratio'])
                self.assertFalse(result['equal_work_eligible'])

    def test_incomplete_duplicate_plan_and_engine_errors_block_all_projections(self):
        for mutation in ('incomplete', 'duplicate', 'plan', 'engine'):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(self.reference)
                if mutation == 'incomplete': bad['run_complete'] = False
                elif mutation == 'duplicate': bad['cases'][1] = copy.deepcopy(bad['cases'][0])
                elif mutation == 'plan': bad['scenarios'].reverse()
                else: bad['engine'] = 'tensorsharp'
                result = analysis.analyze(self.candidate, bad)
                for case in result['scenarios'].values():
                    self.assertFalse(case['equal_work_eligible'])
                    self.assertIsNone(case['task_latency']['reference_over_candidate_median_wall_ratio'])

    def test_client_metric_recomputed_without_mixing_server_timers(self):
        for report in (self.candidate, self.reference):
            for case in report['cases']:
                case['turns'][0]['metrics']['decode_tps'] = 999999
                case['turns'][0]['metrics']['decode_timing_source'] = report['engine']
        item = analysis.analyze(self.candidate, self.reference)['scenarios']['decode']
        metric = item['common_client_metrics']['client_event_window_tokens_per_second_estimate']
        self.assertAlmostEqual(511/.8, metric['observed_distributions']['candidate']['median'])
        self.assertEqual(1, metric['candidate_speed_ratio'])
        self.assertFalse(metric['parity_claim'])

    def test_short_nonstreaming_or_invalid_clocks_have_no_decode_window_ratio(self):
        for mutation in ('short', 'nonstreaming', 'clock'):
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(self.candidate)
                for case in candidate['cases']:
                    turn = case['turns'][0]
                    if mutation == 'short': turn['metrics']['completion_tokens'] = 1
                    elif mutation == 'nonstreaming': turn['request']['stream'] = False
                    else: turn['metrics']['t_last_abs'] = -1
                item = analysis.analyze(candidate, self.reference)['scenarios']['decode']
                self.assertIsNone(item['common_client_metrics']['client_event_window_tokens_per_second_estimate']['candidate_speed_ratio'])

    def test_source_mismatch_suppresses_ratios_without_changing_original_comparator(self):
        result = analysis.analyze(self.candidate, self.reference, additional_global_errors=['source mismatch'])
        self.assertEqual('pass', result['original_harness_campaign_verdict']['status'])
        self.assertFalse(result['scenarios']['decode']['equal_work_eligible'])

    def test_one_bad_repeat_is_not_dropped_from_scenario(self):
        self.reference['cases'][4]['status'] = 'fail'
        item = analysis.analyze(self.candidate, self.reference)['scenarios']['decode']
        self.assertEqual(3, len(item['observations']['reference']))
        self.assertFalse(item['equal_work_eligible'])
        self.assertIsNone(item['task_latency']['reference_over_candidate_median_wall_ratio'])

    def test_cli_preserves_source_bytes_hashes_sources_and_refuses_output_replacement(self):
        current = {Path(path).name: analysis.checksum(Path(path)) for path in
            (analysis.HARNESS_PATH, analysis.harness.validation.__file__, analysis.harness.engines.__file__)}
        self.candidate['harness_sha256'] = current
        self.reference['harness_sha256'] = current
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left, right, output = (root / name for name in ('candidate.json', 'reference.json', 'analysis.json'))
            for path,report in ((left,self.candidate),(right,self.reference)):
                path.write_text(json.dumps(report), encoding='utf-8')
            original = [left.read_bytes(), right.read_bytes()]
            command = [sys.executable, str(ROOT / 'eng/validation/analyze-muse-spark-paired.py'),
                       '--candidate',str(left),'--reference',str(right),'--output',str(output)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(0, process.returncode, process.stderr)
            saved = output.read_bytes()
            report = json.loads(saved)
            self.assertEqual(analysis.checksum(left), report['sources']['candidate']['sha256'])
            self.assertTrue(report['scenarios']['decode']['equal_work_eligible'])
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(0, process.returncode)
            self.assertEqual(saved, output.read_bytes())
            self.assertEqual(original, [left.read_bytes(), right.read_bytes()])


if __name__ == '__main__':
    unittest.main()
