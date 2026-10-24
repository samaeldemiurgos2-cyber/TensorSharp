"""Reject configured-only, stale, failed, and numerically impossible telemetry."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'validation/muse-spark-runtime-probe.py'
SPEC = importlib.util.spec_from_file_location('muse_spark_runtime_probe', SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

SPEC_LOG = ('Speculative decoding armed for request (algorithm=ngram(n=2..8), maxDraft=8)\n'
            'Speculative decoding stats for request: drafted=40 accepted=31 acceptance=78% '
            'verifySteps=5 plainSteps=2 rollbacks=3\n')


class MuseSparkRuntimeProbeTests(unittest.TestCase):
    def test_configuration_alone_does_not_prove_either_mechanism(self):
        observed = probe.telemetry('enabled=True algorithm=ngram batchedPagedAttention=yes')
        self.assertTrue(probe.mechanism_failures('batch', observed))
        self.assertTrue(probe.mechanism_failures('ngram', observed))

    def test_batch_requires_successful_multiple_served_sequences(self):
        for line in ('[cb] step#8 BatchedPaged scheduled=3 served=1',
                     '[cb] step#8 BatchedPaged scheduled=3 served=3 | a:D error=InvalidOperationException',
                     '[cb] step#8 PerSequence scheduled=3 served=3',
                     '[cb] step#8 BatchedPaged scheduled=1 served=3'):
            with self.subTest(line=line):
                self.assertTrue(probe.mechanism_failures('batch', probe.telemetry(line)))
        observed = probe.telemetry('[cb] step#8 BatchedPaged scheduled=3 served=3 | a:P fwd=4 | b:D fwd=1')
        self.assertFalse(probe.mechanism_failures('batch', observed))
        self.assertTrue(observed['batch_steps'][0]['mixed_prefill_decode'])

    def test_ngram_needs_actual_valid_acceptance(self):
        self.assertFalse(probe.mechanism_failures('ngram', probe.telemetry(SPEC_LOG)))
        for invalid in (SPEC_LOG.replace('accepted=31', 'accepted=0'),
                        SPEC_LOG.replace('drafted=40', 'drafted=0'),
                        SPEC_LOG.replace('accepted=31', 'accepted=41'),
                        SPEC_LOG.replace('verifySteps=5', 'verifySteps=0'),
                        SPEC_LOG.replace('algorithm=ngram', 'algorithm=draft-head')):
            with self.subTest(invalid=invalid):
                self.assertTrue(probe.mechanism_failures('ngram', probe.telemetry(invalid)))

    def test_copy_requires_complete_visible_answer_and_valid_usage(self):
        metrics = {'assistant_message': {'content': 'exact'}, 'usage_present': True,
                   'finish_reason': 'stop', 'prompt_tokens': 12, 'completion_tokens': 4}
        wire = {'sse_lines': ['data: [DONE]']}
        self.assertTrue(probe.valid_copy(metrics, wire, 'exact'))
        for override in ({'finish_reason': 'length'}, {'usage_present': False},
                         {'prompt_tokens': 0}, {'completion_tokens': 0},
                         {'assistant_message': {'reasoning_content': 'exact'}}):
            with self.subTest(override=override):
                self.assertFalse(probe.valid_copy(metrics | override, wire, 'exact'))
        self.assertFalse(probe.valid_copy(metrics, {}, 'exact'))

    def test_previous_request_telemetry_is_never_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'server.log'
            output = Path(directory) / 'result.json'
            log.write_text(SPEC_LOG)
            args = types.SimpleNamespace(mode='ngram', model='spark', url='http://unused',
                timeout=1, server_log=str(log), reference=None, out=str(output))

            def completion(url, body, timeout, wire):
                self.assertEqual([], body['skills'])
                self.assertIs(False, body['skills_discovery'])
                self.assertIs(False, body['multi_agent'])
                expected = body['messages'][0]['content'].split('<copy>\n')[1].split('\n</copy>')[0]
                wire['sse_lines'] = ['data: [DONE]']
                return {'assistant_message': {'content': expected}, 'usage_present': True,
                        'finish_reason': 'stop', 'prompt_tokens': 120, 'completion_tokens': 96}

            with patch.object(probe.engines, '_run_streaming', side_effect=completion), \
                    patch.object(probe.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(1, probe.run(args))
            result = json.loads(output.read_text())
            self.assertEqual({'skills': [], 'skills_discovery': False, 'multi_agent': False},
                             result['request_adapter'])
            self.assertTrue(result['requests'][0]['correct'])
            self.assertFalse(result['passed'])
            self.assertEqual([], result['telemetry']['speculation_stats'])


if __name__ == '__main__':
    unittest.main()
