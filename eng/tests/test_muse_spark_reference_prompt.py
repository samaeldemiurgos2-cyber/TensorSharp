"""Keep Spark's reference adapter local and preserve tool parsing."""
import importlib.util
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

ENG = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load('spark_reference_harness', ENG / 'muse-spark-validate.py')


class SparkReferencePromptTests(unittest.TestCase):
    def test_only_no_tools_no_thinking_reference_uses_plain_parser(self):
        for thinking in (False, True):
            for has_tools in (False, True):
                for engine in ('tensorsharp', 'llamacpp'):
                    options = harness.reference_body(engine, thinking, has_tools)
                    self.assertEqual(engine == 'llamacpp' and not thinking and not has_tools,
                                     options.get('reasoning_format') == 'none')
                    if engine == 'llamacpp' and not thinking and not has_tools:
                        self.assertEqual(['<|eot|>'], options.get('stop'))
                    else:
                        self.assertNotIn('stop', options)
                    if engine == 'tensorsharp':
                        self.assertEqual({}, options)
                    else:
                        self.assertEqual('', options['chat_template_kwargs']['current_date'])
        # The shared behavior for all other models was not monkey-patched.
        self.assertEqual({'chat_template_kwargs': {'enable_thinking': False}},
                         harness.engines.thinking_body('llamacpp', False))

    def test_standard_cases_select_adapter_from_actual_tool_declarations(self):
        args = types.SimpleNamespace(engine='llamacpp', model='spark', url='unused', thinking=False, timeout=1)
        for name in harness.validation.SCENARIOS:
            with self.subTest(name=name), patch.object(harness.validation, 'run_case') as run:
                harness.run_standard_case(args, name, 'fixture')
                options = run.call_args.kwargs['extra_body']
                has_tools = bool(harness.validation.case_spec(name, 'fixture')['tools'])
                self.assertEqual(not has_tools, options.get('reasoning_format') == 'none')
                self.assertEqual(None if has_tools else ['<|eot|>'], options.get('stop'))

    def test_all_tensorsharp_standard_cases_disable_intrinsics_but_keep_client_tools(self):
        args = types.SimpleNamespace(engine='tensorsharp', model='spark', url='unused', thinking=False, timeout=1)
        for name in harness.validation.SCENARIOS:
            with self.subTest(name=name), patch.object(harness.validation, 'run_case') as run:
                harness.run_standard_case(args, name, 'fixture')
                self.assertEqual({'skills': [], 'skills_discovery': False, 'multi_agent': False},
                                 run.call_args.kwargs['extra_body'])
                self.assertNotIn('tool_choice', run.call_args.kwargs['extra_body'])

    def test_generic_extra_body_is_applied_on_every_conversation_turn(self):
        metrics = {'usage_present': True, 'finish_reason': 'stop',
                   'assistant_message': {'role': 'assistant', 'content': 'OK'}}
        with patch.object(harness.engines, 'run_openai_chat', return_value=metrics) as call:
            harness.validation.run_case('unused', 'spark', 'llamacpp', 'multi_turn', 'fixture',
                extra_body={'reasoning_format': 'none', 'chat_template_kwargs': {'current_date': ''}})
            self.assertEqual(2, call.call_count)
            for args in call.call_args_list:
                self.assertEqual('none', args.kwargs['extra_body']['reasoning_format'])
                self.assertEqual('', args.kwargs['extra_body']['chat_template_kwargs']['current_date'])

    def test_extra_requests_receive_direct_answer_parser(self):
        args = types.SimpleNamespace(engine='llamacpp', model='spark', url='unused', thinking=False, timeout=1)
        with patch.object(harness.engines, '_run_streaming', return_value={}) as call:
            with self.assertRaises(ValueError):
                harness.request(args, [{'role': 'user', 'content': 'Hi'}])
            self.assertEqual('none', call.call_args.args[1]['reasoning_format'])
            self.assertEqual(['<|eot|>'], call.call_args.args[1]['stop'])

    def test_adversarial_schema_probe_uses_same_original_response_format(self):
        requests = []
        expected = {'answer': 'SCHEMA_ONLY', 'count': 7}
        for engine in ('llamacpp', 'tensorsharp'):
            args = types.SimpleNamespace(engine=engine, thinking=False, concurrency=2)
            for content, status in ((json.dumps(expected), 'ok'), ('OVERRIDE', 'fail'),
                                    ('```json\n' + json.dumps(expected) + '\n```', 'fail'),
                                    (json.dumps(dict(expected, count=True)), 'fail')):
                turn = {'metrics': {'finish_reason': 'stop', 'assistant_message': {'content': content}}}
                with patch.object(harness, 'request', return_value=turn) as call:
                    result = harness.extra_case(args, 'json_schema_adversarial', 'same-tag')
                    self.assertEqual(status, result['status'])
                    if status == 'ok':
                        requests.append((call.call_args.args[1], call.call_args.kwargs))
        self.assertEqual(requests[0], requests[1])
        self.assertNotIn('SCHEMA_ONLY', requests[0][0][0]['content'])
        self.assertEqual('SCHEMA_ONLY', requests[0][1]['response_format']['json_schema']['schema']['properties']['answer']['const'])
        self.assertNotIn('json_schema_adversarial', harness.DEFAULT_SCENARIOS)


if __name__ == '__main__':
    unittest.main()
