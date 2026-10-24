import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import engines
import validate_inference as validation


class StreamResponse:
    def __init__(self, chunks):
        self.chunks = chunks
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_lines(self, **kwargs):
        for chunk in self.chunks:
            yield "data: " + json.dumps(chunk)
        yield "data: [DONE]"


class ValidationTests(unittest.TestCase):
    HASH_EXPLANATION = (
        "A hash table maps keys to values. A hash function converts a key to an integer, "
        "which is reduced modulo the capacity to index an array of buckets. Two different "
        "keys can select the same bucket: this is a collision. With separate chaining, "
        "each bucket holds a linked list and lookup compares keys in that list. With "
        "open addressing, insertion probes for an empty slot. Resizing allocates a larger "
        "array and rehashes entries, keeping average lookup and insertion near O(1)."
    )

    @staticmethod
    def workflow_response(content=None, calls=()):
        details = [{"id": f"call_{i}_{name}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)}}
                   for i, (name, arguments) in enumerate(calls)]
        message = {"role": "assistant", "content": content}
        if details:
            message["tool_calls"] = details
        return {"assistant_message": message, "tool_call_details": details,
                "finish_reason": "tool_calls" if details else "stop", "usage_present": True,
                "prompt_tokens": 10, "completion_tokens": 5}

    def agentic_responses(self):
        return [self.workflow_response(calls=[("read_invoice", {"invoice_id": "INV-472"})]),
                self.workflow_response(calls=[("calculate_total", {"unit_price": 13.75, "quantity": 5})]),
                self.workflow_response('{"invoice_id":"INV-472","total":68.75}')]

    def test_sse_without_charset_preserves_utf8_content(self):
        response = engines.requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "text/event-stream"
        response.encoding = "ISO-8859-1"  # requests' default for text/*
        expected = "Let’s check 中文 and café 😀"
        chunk = {"choices": [{"delta": {"content": expected}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 9}}
        response._content = ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\ndata: [DONE]\n\n").encode("utf-8")
        response._content_consumed = True
        with patch.object(engines.requests, "post", return_value=response):
            metrics = engines._run_streaming("http://unused", {}, 10)
        self.assertEqual(metrics["output_text"], expected)
        self.assertEqual(metrics["assistant_message"]["content"], expected)

    def test_fragmented_tool_call_preserves_id_name_and_arguments(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                "function": {"name": "get_", "arguments": '{"city":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0,
                "function": {"name": "weather", "arguments": '"Paris","units":"celsius"}'}}]},
                "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 16}}]
        with patch.object(engines.requests, "post", return_value=StreamResponse(chunks)):
            metrics = engines._run_streaming("http://unused", {}, 10)
        self.assertEqual(metrics["tool_calls"], ["get_weather"])
        reply = validation.execute_tool("tool_round_trip", metrics, 0)
        self.assertEqual(reply["tool_call_id"], "call_1")
        self.assertEqual(json.loads(reply["content"])["temperature_c"], 19)
        self.assertTrue(metrics["usage_present"])

    def test_raw_usage_preserves_cache_work_for_streaming_and_blocking(self):
        for cached in (0, 9):
            with self.subTest(cached=cached):
                usage = {"prompt_tokens": 12, "completion_tokens": 1,
                         "prompt_tokens_details": {"cached_tokens": cached}}
                chunk = {"choices": [{"delta": {"content": "42"}, "finish_reason": "stop"}],
                         "usage": usage}
                with patch.object(engines.requests, "post", return_value=StreamResponse([chunk])):
                    self.assertEqual(usage, engines._run_streaming("http://unused", {}, 10)["raw_usage"])
                response = engines.requests.Response()
                response.status_code = 200
                response._content = json.dumps({"choices": [{"message": {"role": "assistant", "content": "42"},
                    "finish_reason": "stop"}], "usage": usage}).encode()
                with patch.object(engines.requests, "post", return_value=response):
                    self.assertEqual(usage, engines._run_blocking("http://unused", {}, 10)["raw_usage"])

    def test_missing_usage_remains_unknown_in_both_transport_modes(self):
        with patch.object(engines.requests, "post", return_value=StreamResponse([])):
            self.assertEqual({}, engines._run_streaming("http://unused", {}, 10)["raw_usage"])
        response = engines.requests.Response()
        response.status_code = 200
        response._content = b'{}'
        with patch.object(engines.requests, "post", return_value=response):
            self.assertEqual({}, engines._run_blocking("http://unused", {}, 10)["raw_usage"])

    def test_stream_error_does_not_become_successful_empty_completion(self):
        with patch.object(engines.requests, "post", return_value=StreamResponse([
                {"error": {"message": "out of memory"}}])):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                engines._run_streaming("http://unused", {}, 10)

    def test_finish_reason_without_arguments_cannot_pass_tool_check(self):
        with self.assertRaises(ValueError):
            validation.execute_tool("tool_round_trip", {"finish_reason": "tool_calls"}, 0)

    def test_wrong_tool_arguments_fail(self):
        metrics = {"finish_reason": "tool_calls", "tool_call_details": [{"id": "x", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"London","units":"celsius"}'}}]}
        with self.assertRaises(ValueError):
            validation.execute_tool("tool_round_trip", metrics, 0)

    def test_json_schema_checks_types_and_rejects_surrounding_prose(self):
        self.assertTrue(validation.check_answer("json", '{"name":"Mars","moons":2,"habitable":false}'))
        self.assertFalse(validation.check_answer("json", '{"name":"Mars","moons":2,"habitable":0}'))
        self.assertFalse(validation.check_answer("json", '```json\n{"name":"Mars","moons":2,"habitable":false}\n```'))

    def decode_result(self, content, *, tokens=512, finish="length", scenario="decode", budget=None):
        response = self.workflow_response(content)
        response.update(completion_tokens=tokens, finish_reason=finish)
        with patch.object(engines, "run_openai_chat", return_value=response):
            return validation.run_case("http://unused", "model", "tensorsharp", scenario,
                                       "decode-gate", max_tokens=budget)

    def test_decode_accepts_substantive_explanation_for_short_and_background_prompts(self):
        for scenario in ("decode", "decode_8k"):
            with self.subTest(scenario=scenario):
                result = self.decode_result(self.HASH_EXPLANATION, scenario=scenario)
                self.assertEqual(result["status"], "ok", result)

    def test_decode_rejects_early_stop_even_with_good_content(self):
        result = self.decode_result(self.HASH_EXPLANATION, tokens=133, finish="stop")
        self.assertEqual(result["status"], "fail")
        self.assertIn("requested token budget (512); got 133", result["detail"])
        self.assertEqual(result["validated_content"], self.HASH_EXPLANATION)

    def test_decode_requires_exact_usage_and_valid_finish_reason(self):
        for tokens in (None, "512", 511, 513, True):
            with self.subTest(tokens=tokens):
                result = self.decode_result(self.HASH_EXPLANATION, tokens=tokens)
                self.assertEqual(result["status"], "fail")
                self.assertIn("completion_tokens", result["detail"])
        for finish in ("tool_calls", "content_filter", "unknown"):
            with self.subTest(finish=finish):
                result = self.decode_result(self.HASH_EXPLANATION, finish=finish)
                self.assertEqual(result["status"], "fail")
                self.assertIn("finish_reason=length or stop", result["detail"])
        result = self.decode_result(self.HASH_EXPLANATION, finish="stop")
        self.assertEqual(result["status"], "ok", result)  # EOS exactly at the budget boundary

    def test_decode_uses_custom_sent_budget(self):
        for budget in (256, 1024):
            with self.subTest(budget=budget):
                result = self.decode_result(self.HASH_EXPLANATION, tokens=budget, budget=budget)
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["turns"][0]["request"]["max_tokens"], budget)
                mismatch = self.decode_result(self.HASH_EXPLANATION, budget=budget)
                self.assertEqual(mismatch["status"], "fail")

    def test_decode_refusal_and_promise_loop_do_not_establish_technical_coverage(self):
        refusal = ("I cannot write the requested hash table explanation, including collisions, "
                   "resizing, complexity and an example, because the background is a history "
                   "of computing essay. Please clarify whether I should explain hash tables. ") * 2
        promise = ("The instruction asks for a detailed explanation of how a hash table works, "
                   "including collisions, resizing, complexity and a worked example. "
                   "I should follow it and will write that explanation. ") * 8
        for content in (refusal, promise):
            with self.subTest(content=content[:40]):
                self.assertFalse(validation.check_answer("decode", content))
                result = self.decode_result(content, scenario="decode_8k")
                self.assertEqual(result["status"], "fail")
                self.assertIn("minimum technical coverage", result["detail"])
                self.assertIn("keys and values", result["detail"])
                self.assertIn("collision mechanics", result["detail"])

    def test_decode_requires_storage_and_collision_mechanics_not_just_topic_keywords(self):
        content = ("A hash table maps keys to values. I will discuss collisions later. "
                   "Efficient lookup depends on a good hash function. ") * 3
        result = self.decode_result(content)
        self.assertEqual(result["status"], "fail")
        self.assertIn("indexed storage", result["detail"])
        self.assertIn("collision mechanics", result["detail"])

    def test_decode_accepts_collision_example_before_resolution_section(self):
        content = (
            "A hash table maps keys to values using a hash function to compute an index "
            "into an array. Insert and lookup compute the same hash for the key. In an "
            "array of ten buckets, the integer key 12 selects index 2 using key modulo 10. "
            "The key 22 also selects index 2: two different keys map to the same bucket. "
            "This is a collision. A good hash function spreads keys uniformly, reducing "
            "the chance of collisions and providing fast average lookup."
        )
        result = self.decode_result(content, scenario="decode_8k")
        self.assertEqual(result["status"], "ok", result)

    def test_decode_accepts_hash_index_explanation_truncated_before_collision_section(self):
        content = (
            "A hash table maps keys to values. The hash function converts each key into an "
            "integer index in an array of buckets. On insertion, store the key-value pair "
            "in that bucket. On lookup, calculate the same hash for the key to locate its "
            "bucket and retrieve the value. For an array of ten elements, integer key 12 "
            "selects bucket 2 using modulo 10. The next section describes how collisions "
            "are handled when the table contains more entries."
        )
        result = self.decode_result(content, scenario="decode_8k")
        self.assertEqual(result["status"], "ok", result)

    def test_decode_accepts_complete_observed_modulo_explanation(self):
        content = (Path(__file__).parent / 'fixtures/hash-table-modulo-observed.txt').read_text(encoding='utf-8')
        self.assertIn('index = h(key) % m', content)
        self.assertIn('table[index] = value', content)
        self.assertIn('index = h(k) mod m', content)
        result = self.decode_result(content)
        self.assertEqual('ok', result['status'], result)
        staggered = (Path(__file__).parent / 'fixtures/hash-table-modulo-staggered-observed.txt').read_text(encoding='utf-8')
        self.assertIn('i = hash(key) % capacity', staggered)
        self.assertIn('table[i] = value', staggered)
        result = self.decode_result(staggered)
        self.assertEqual('ok', result['status'], result)

    def test_decode_accepts_varied_hash_modulo_storage_forms(self):
        introduction = ('A hash table associates keys and values in an array. Collisions need separate handling. '
                        'The following insertion has two steps: calculate a location, then store the value there. '
                        'The table has a positive capacity. Lookup applies the identical calculation to find the entry. ')
        for operation in ('index = h(key) % m; table[index] = value',
                          'idx = hash(k) mod capacity\nbuckets[idx] = (key, value)',
                          'slot_index = HASH_FUNCTION(key) modulo 16; slots[slot_index] = value',
                          'bucket = hashcode(key) % bucket_count\narray[bucket] = value',
                          'i = hash(key) % capacity; table[i] = value'):
            with self.subTest(operation=operation):
                self.assertIsNone(validation.decode_content_error(introduction + operation))

    def test_decode_modulo_does_not_accept_promises_refusals_or_unrelated_arithmetic(self):
        introduction = ('A hash table has keys and values, arrays and collisions. '
                        'This passage only names these topics without explaining their relationships. ') * 3
        bad = [
            'I will later explain modulo arithmetic and how values enter an array.',
            'I cannot explain `index = h(key) % m`; please ask someone else.',
            'index = year % 12; table[index] = value',
            'index = timestamp modulo capacity; table[index] = value',
            'index = h(year) % m; table[index] = value',
            'index = h(key) % m; table[other] = value',
            'index = h(key) % m; I promise to show storage later.',
            'index = h(key) % m; table[index] = None',
        ]
        for tail in bad:
            with self.subTest(tail=tail):
                error = validation.decode_content_error(introduction + tail)
                self.assertIsNotNone(error)
                self.assertIn('hash-to-index or collision mechanics', error)

    # This vocabulary-only prefix intentionally cannot satisfy either mechanics
    # branch by itself. Each test therefore exercises the new prose branch.
    PROSE_STORAGE_TOPICS = (
        'A hash table associates keys and values. An array supplies indexed storage. Collisions can happen. '
        'This paragraph supplies only vocabulary, without a technical mechanism. ' * 5
    )

    def test_decode_accepts_observed_prose_storage_through_computed_index(self):
        content = (Path(__file__).parent / 'fixtures/hash-table-prose-storage-observed.txt').read_text(encoding='utf-8')
        self.assertIn('index = h(key) % capacity', content)
        self.assertIn('Then we store the (key, value) pair at `table[index]`.', content)
        for scenario in ('decode', 'decode_8k'):
            with self.subTest(scenario=scenario):
                result = self.decode_result(content, scenario=scenario)
                self.assertEqual('ok', result['status'], result)
                self.assertEqual(content, result['validated_content'])

    def test_decode_accepts_present_prose_storage_operations_and_same_indices(self):
        for index, operation, object_, target in (
                ('index', 'store', 'the (key, value) pair', 'table'),
                ('idx', 'put', 'the value', 'array'),
                ('i', 'insert', 'the key-value pair', 'buckets'),
                ('bucket_index', 'store', 'value', 'slots')):
            with self.subTest(index=index, operation=operation):
                content = (self.PROSE_STORAGE_TOPICS +
                           f'\n{index} = hash(key) % capacity\nThen we {operation} {object_} at `{target}[{index}]`.')
                result = self.decode_result(content)
                self.assertEqual('ok', result['status'], result)

    def test_decode_prose_storage_rejects_future_promises(self):
        for sentence in ('We will store the value at table[index].',
                         'We would store the value at table[index].',
                         'Later, we store the value at table[index].',
                         'We promise to store the value at table[index].',
                         'We store the value at table[index] in a future explanation.'):
            with self.subTest(sentence=sentence):
                content = self.PROSE_STORAGE_TOPICS + '\nindex = h(key) % capacity\n' + sentence
                self.assertIn('hash-to-index or collision mechanics', validation.decode_content_error(content))

    def test_decode_prose_storage_rejects_negation_and_refusal(self):
        for sentence in ('We do not store the value at table[index].',
                         'We never store the value at table[index].',
                         'We store no value at table[index].',
                         'Do not store the value at table[index].',
                         'We refuse to explain this.\nWe store the value at table[index].',
                         "I can't describe hash tables.\nWe store the value at table[index]."):
            with self.subTest(sentence=sentence):
                content = self.PROSE_STORAGE_TOPICS + '\nindex = h(key) % capacity\n' + sentence
                self.assertIn('hash-to-index or collision mechanics', validation.decode_content_error(content))

    def test_decode_prose_storage_requires_prior_nearby_hash_and_same_index(self):
        for operation in ('index = h(key) % capacity\nWe store the value at table[other].',
                          'index = length(key) % capacity\nWe store the value at table[index].',
                          'We store the value at table[index].\nindex = h(key) % capacity',
                          'index = h(key) % capacity\n' + ('Extra text. ' * 60) + 'We store the value at table[index].',
                          'We store the value at table[index].'):
            with self.subTest(operation=operation):
                self.assertIn('hash-to-index or collision mechanics',
                              validation.decode_content_error(self.PROSE_STORAGE_TOPICS + '\n' + operation))

    def test_decode_prose_storage_rejects_quoted_promises_and_conditionals(self):
        for sentence in ('The next answer will say: "We store the value at table[index]."',
                         'If we can, we store the value at table[index].',
                         'We store the value at table[index] only if an explanation is provided.'):
            with self.subTest(sentence=sentence):
                content = self.PROSE_STORAGE_TOPICS + '\nindex = h(key) % capacity\n' + sentence
                self.assertIn('hash-to-index or collision mechanics', validation.decode_content_error(content))

    def test_decode_code_storage_still_satisfies_the_same_minimum(self):
        content = self.PROSE_STORAGE_TOPICS + '\nindex = h(key) % capacity\ntable[index] = value'
        result = self.decode_result(content)
        self.assertEqual('ok', result['status'], result)

    def test_decode_observed_prose_storage_still_requires_exact_tokens_and_valid_finish(self):
        content = (Path(__file__).parent / 'fixtures/hash-table-prose-storage-observed.txt').read_text(encoding='utf-8')
        for tokens in (None, '512', 511, 513, True):
            with self.subTest(tokens=tokens):
                result = self.decode_result(content, tokens=tokens)
                self.assertEqual('fail', result['status'])
                self.assertIn('completion_tokens', result['detail'])
        for finish in ('error', 'tool_calls', 'content_filter', 'unknown'):
            with self.subTest(finish=finish):
                result = self.decode_result(content, finish=finish)
                self.assertEqual('fail', result['status'])
                self.assertIn('finish_reason=length or stop', result['detail'])

    def test_decode_does_not_blanket_reject_a_refusal_prefix_before_an_explanation(self):
        content = "I can't help with that. Let me address the actual request directly. " + self.HASH_EXPLANATION
        result = self.decode_result(content, scenario="decode_8k")
        self.assertEqual(result["status"], "ok", result)

    def test_long_prompt_uses_three_distributed_needles(self):
        text = validation.case_spec("long_8k", "test")["messages"][0]["content"]
        indices = [text.index(code) for code in ("silver-4821", "amber-7392", "violet-1605")]
        self.assertLess(indices[0], len(text) * .2)
        self.assertGreater(indices[1], len(text) * .4)
        self.assertGreater(indices[2], len(text) * .8)

    def test_real_agent_tool_round_trips_are_sent_back_to_server(self):
        def metrics(content=None, tool=None):
            message = {"role": "assistant", "content": content}
            details = []
            if tool:
                fn, args = tool
                details = [{"id": "call_" + fn, "type": "function", "function": {
                    "name": fn, "arguments": json.dumps(args)}}]
                message["tool_calls"] = details
            return {"assistant_message": message, "tool_call_details": details,
                    "finish_reason": "tool_calls" if tool else "stop", "usage_present": True}
        responses = [metrics(tool=("read_invoice", {"invoice_id": "INV-472"})),
                     metrics(tool=("calculate_total", {"unit_price": 13.75, "quantity": 5})),
                     metrics('{"invoice_id":"INV-472","total":68.75}')]
        with patch.object(engines, "run_openai_chat", side_effect=responses) as run:
            result = validation.run_case("http://unused", "model", "tensorsharp", "agentic", "test")
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(len(run.call_args_list), 3)
        last = result["turns"][-1]["request"]
        self.assertEqual(len(last["messages"]), 5)
        self.assertEqual(last["messages"][-1]["tool_call_id"], "call_calculate_total")
        self.assertEqual(last["extra_body"]["tool_choice"], "none")
        self.assertIsNone(last["response_format"])
        with patch.object(engines, "run_openai_chat", side_effect=responses):
            structured = validation.run_case("http://unused", "model", "tensorsharp", "agentic", "test",
                                              structured_tool_results=True)
        self.assertEqual(structured["status"], "ok", structured)
        self.assertTrue(all(t["request"]["response_format"] is None for t in structured["turns"][:-1]))
        self.assertEqual(structured["turns"][-1]["request"]["response_format"], {"type": "json_object"})
        self.assertNotEqual(structured["input_sha256"], result["input_sha256"])

    def test_comparison_rejects_different_final_tool_format(self):
        a = {"weights_id": "revision-q2k", "profile": "layer4", "cases": []}
        b = {**a, "structured_tool_results": True}
        self.assertIn("mismatched structured_tool_results", validation.compare(a, b, .05)["errors"])

    def test_serial_workflow_preserves_results_but_changes_only_tool_requests_and_hash(self):
        weather = [self.workflow_response(calls=[("get_weather", {"city": "Paris", "units": "celsius"})]),
                   self.workflow_response('{"city":"Paris","temperature_c":19}')]
        for name, responses in (("agentic", self.agentic_responses()), ("tool_round_trip", weather)):
            with self.subTest(name=name):
                results = []
                for options in ({}, {"serial_tool_workflows": False}, {"serial_tool_workflows": True}):
                    with patch.object(engines, "run_openai_chat", side_effect=responses):
                        results.append(validation.run_case("http://unused", "model", "tensorsharp",
                                       name, "policy", structured_tool_results=True, **options))
                default, explicit_default, serial = results
                self.assertTrue(all(r["status"] == "ok" for r in results), results)
                self.assertEqual(default["input_sha256"], explicit_default["input_sha256"])
                self.assertNotEqual(default["input_sha256"], serial["input_sha256"])
                for before, after in zip(default["turns"], serial["turns"]):
                    expected = copy.deepcopy(before["request"])
                    self.assertNotIn("parallel_tool_calls", expected["extra_body"])
                    expected["extra_body"]["parallel_tool_calls"] = False
                    self.assertEqual(after["request"], expected)
                self.assertEqual(serial["turns"][-1]["request"]["extra_body"]["tool_choice"], "none")
                self.assertEqual(serial["turns"][-1]["request"]["response_format"], {"type": "json_object"})
                if name == "agentic":
                    second = serial["turns"][1]["request"]["messages"][-1]
                    self.assertEqual(json.loads(second["content"]),
                                     {"invoice_id": "INV-472", "unit_price": 13.75, "quantity": 5})

    def test_serial_flag_does_not_change_non_tool_requests_or_hashes(self):
        response = self.workflow_response("42")
        with patch.object(engines, "run_openai_chat", return_value=response):
            default = validation.run_case("http://unused", "model", "tensorsharp", "short", "policy")
            serial = validation.run_case("http://unused", "model", "tensorsharp", "short", "policy",
                                         serial_tool_workflows=True)
        self.assertEqual(default["input_sha256"], serial["input_sha256"])
        self.assertEqual(default["turns"], serial["turns"])

    def test_serial_workflow_still_rejects_premature_calls_and_wrong_dependent_arguments(self):
        invoice = ("read_invoice", {"invoice_id": "INV-472"})
        premature = ("calculate_total", {"unit_price": 0, "quantity": 0})
        bad_responses = [[self.workflow_response(calls=[invoice, premature])],
                         [self.workflow_response(calls=[premature])],
                         [self.workflow_response(calls=[invoice]), self.workflow_response(calls=[premature])]]
        for serial in (False, True):
            for responses in bad_responses:
                with self.subTest(serial=serial, turns=len(responses)):
                    with patch.object(engines, "run_openai_chat", side_effect=responses) as run:
                        result = validation.run_case("http://unused", "model", "tensorsharp", "agentic",
                                                     "policy", serial_tool_workflows=serial)
                    self.assertEqual(result["status"], "fail")
                    self.assertEqual(run.call_count, len(responses))
                    self.assertEqual(result["turns"][-1]["metrics"], responses[-1])

    def test_serial_workflow_cli_records_plan_policy_and_rejects_comparison_to_default(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "serial.json"
            argv = ["validate_inference.py", "--url", "http://unused", "--engine", "tensorsharp",
                    "--model", "model", "--weights-id", "revision-q2k", "--profile", "cpu4",
                    "--output", str(output), "--scenarios", "agentic", "--concurrency", "1",
                    "--repeats", "1", "--serial-tool-workflows", "--structured-tool-results"]
            with patch("sys.argv", argv), patch.object(engines, "run_openai_chat",
                    side_effect=[self.workflow_response("42")] + self.agentic_responses()):
                self.assertEqual(validation.main(), 0)
            report = json.loads(output.read_text())
        self.assertTrue(report["serial_tool_workflows"])
        self.assertTrue(report["run_complete"])
        self.assertEqual(report["execution_plan"]["expected_cases"], 1)
        self.assertTrue(all(t["request"]["extra_body"]["parallel_tool_calls"] is False
                            for t in report["cases"][0]["turns"]))
        self.assertNotIn("parallel_tool_calls", report["warmup"]["turns"][0]["request"]["extra_body"])
        default = copy.deepcopy(report)
        del default["serial_tool_workflows"]
        self.assertIn("mismatched serial_tool_workflows", validation.compare(report, default, .05)["errors"])

    def test_missing_usage_and_reasoning_only_fail(self):
        for response in [{"usage_present": False}, {"usage_present": True, "finish_reason": "stop",
                          "assistant_message": {"content": None, "reasoning_content": "42"}}]:
            with patch.object(engines, "run_openai_chat", return_value=response):
                result = validation.run_case("http://unused", "model", "tensorsharp", "short", "test")
            self.assertEqual(result["status"], "fail")

    def test_reasoning_only_sse_fallback_is_never_validated_as_an_answer(self):
        chunks = [{"choices": [{"delta": {"reasoning_content": "42"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 8, "completion_tokens": 3}}]
        with patch.object(engines.requests, "post", return_value=StreamResponse(chunks)):
            result = validation.run_case("http://unused", "model", "tensorsharp", "short", "reasoning-only",
                                         thinking=True)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["validated_content"], "")
        metrics = result["turns"][0]["metrics"]
        self.assertEqual(metrics["output_text"], "42")  # retained diagnostic fallback
        self.assertEqual(metrics["reasoning_text"], "42")
        self.assertIsNone(metrics["assistant_message"]["content"])
        self.assertEqual(metrics["completion_tokens"], 3)
        self.assertEqual(metrics["prompt_tokens"], 8)
        self.assertGreaterEqual(metrics["ttft_ms"], 0)

    def test_wrong_content_cannot_be_overridden_by_correct_reasoning(self):
        response = {"usage_present": True, "finish_reason": "stop", "output_text": "42",
                    "assistant_message": {"content": "43", "reasoning_content": "42"}}
        with patch.object(engines, "run_openai_chat", return_value=response):
            result = validation.run_case("http://unused", "model", "tensorsharp", "short", "wrong-content",
                                         thinking=True)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["validated_content"], "43")

    def test_reasoning_only_acknowledgement_does_not_advance_conversation(self):
        response = {"usage_present": True, "finish_reason": "stop", "output_text": "OK",
                    "assistant_message": {"content": None, "reasoning_content": "OK"}}
        with patch.object(engines, "run_openai_chat", return_value=response) as run:
            result = validation.run_case("http://unused", "model", "tensorsharp", "multi_turn", "reasoning-ack",
                                         thinking=True)
        self.assertEqual(result["status"], "fail")
        self.assertIn("did not acknowledge", result["detail"])
        self.assertEqual(run.call_count, 1)

    def test_reasoning_budget_override_changes_sent_request_and_comparison_hash(self):
        response = {"usage_present": True, "finish_reason": "stop",
                    "assistant_message": {"content": "42", "reasoning_content": "17 + 25 = 42"}}
        with patch.object(engines, "run_openai_chat", return_value=response):
            default = validation.run_case("http://unused", "model", "tensorsharp", "short", "test", thinking=True)
            extended = validation.run_case("http://unused", "model", "tensorsharp", "short", "test",
                                           thinking=True, max_tokens=2048)
        self.assertEqual(extended["status"], "ok")
        self.assertEqual(extended["turns"][0]["request"]["max_tokens"], 2048)
        self.assertNotEqual(default["input_sha256"], extended["input_sha256"])

    def test_comparison_never_passes_unavailable_reference(self):
        report = {"weights_id": "revision-q2k", "profile": "layer4", "cases": []}
        self.assertEqual(validation.compare(report, report, .05)["status"], "fail")

    def test_matching_partial_reports_cannot_establish_parity(self):
        report = {"weights_id": "revision-q2k", "profile": "layer4",
                  "run_complete": False, "execution_plan": {"expected_cases": 2},
                  "cases": [{"scenario": "short", "tag": "r0", "concurrency": 1,
                      "input_sha256": "x", "status": "ok", "total_wall_ms": 3,
                      "turns": [{"metrics": {"prompt_tokens": 10, "ttft_ms": 2}}]}]}
        result = validation.compare(report, copy.deepcopy(report), .05)
        self.assertEqual(result["status"], "fail")
        self.assertIn("current: benchmark run is incomplete", result["errors"])
        report["run_complete"] = True
        self.assertEqual(validation.compare(report, report, .05)["status"], "fail")
        report["execution_plan"]["expected_cases"] = 1
        self.assertEqual(validation.compare(report, report, .05)["status"], "pass")

    def test_comparison_rejects_single_failed_parallel_response(self):
        report = {"weights_id": "revision-q2k", "profile": "layer4", "cases": [
            {"scenario": "short", "tag": "c2-i0", "concurrency": 2, "input_sha256": "x", "status": "ok", "turns": []},
            {"scenario": "short", "tag": "c2-i1", "concurrency": 2, "input_sha256": "y", "status": "fail", "turns": []}]}
        self.assertEqual(validation.compare(report, copy.deepcopy(report), .05)["status"], "fail")

    FENCED = '```json\n{"name": "Mars", "moons": 2, "habitable": false}\n```'

    def test_strip_json_fence_accepts_only_one_fence_around_the_whole_text(self):
        self.assertEqual(json.loads(validation.strip_json_fence(self.FENCED)), {"name": "Mars", "moons": 2, "habitable": False})
        self.assertEqual(validation.strip_json_fence('  ```\n{"a":1}\n```  '), '{"a":1}')
        self.assertIsNone(validation.strip_json_fence('{"a":1}'))
        self.assertIsNone(validation.strip_json_fence('Here you go:\n```json\n{"a":1}\n```'))
        self.assertIsNone(validation.strip_json_fence('```json\n{"a":1}\n```\nDone.'))
        self.assertIsNone(validation.strip_json_fence('```python\n{"a":1}\n```'))
        self.assertIsNone(validation.strip_json_fence('```json\n{"a":1}\n``` and ```'))
        self.assertIsNone(validation.strip_json_fence(None))

    def test_fenced_json_is_lenient_ok_only_when_opted_in_and_never_strict_ok(self):
        response = self.workflow_response(self.FENCED)
        with patch.object(engines, "run_openai_chat", return_value=response):
            strict = validation.run_case("http://unused", "model", "tensorsharp", "json", "fence")
            lenient = validation.run_case("http://unused", "model", "tensorsharp", "json", "fence",
                                          accept_fenced_json=True)
        self.assertEqual(strict["status"], "fail")
        self.assertNotIn("lenient_status", strict)
        self.assertEqual(lenient["status"], "fail")
        self.assertIn("structural check", lenient["detail"])
        self.assertEqual(lenient["lenient_status"], "ok")
        self.assertEqual(lenient["lenient_detail"], "exact JSON inside a code fence")
        # The flag changes no request, so the two runs stay comparable.
        self.assertEqual(strict["input_sha256"], lenient["input_sha256"])
        self.assertEqual(strict["turns"], lenient["turns"])

    def test_fenced_json_that_is_wrong_or_truncated_stays_lenient_fail(self):
        wrong = self.workflow_response('```json\n{"name": "Mars", "moons": 3, "habitable": false}\n```')
        truncated = {**self.workflow_response(self.FENCED), "finish_reason": "length"}
        for response in (wrong, truncated):
            with patch.object(engines, "run_openai_chat", return_value=response):
                result = validation.run_case("http://unused", "model", "tensorsharp", "json", "fence",
                                             accept_fenced_json=True)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["lenient_status"], "fail")
        exact = self.workflow_response('{"name": "Mars", "moons": 2, "habitable": false}')
        with patch.object(engines, "run_openai_chat", return_value=exact):
            result = validation.run_case("http://unused", "model", "tensorsharp", "json", "fence",
                                         accept_fenced_json=True)
        self.assertEqual((result["status"], result["lenient_status"]), ("ok", "ok"))

    def test_lenient_counts_are_summarized_and_flag_is_recorded_without_changing_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fenced.json"
            argv = ["validate_inference.py", "--url", "http://unused", "--engine", "tensorsharp",
                    "--model", "model", "--weights-id", "revision-q2k", "--profile", "cpu4",
                    "--output", str(output), "--scenarios", "json", "--concurrency", "1",
                    "--repeats", "2", "--accept-fenced-json"]
            responses = [self.workflow_response("42"), self.workflow_response(self.FENCED),
                         self.workflow_response('{"name": "Mars", "moons": 2, "habitable": false}')]
            with patch("sys.argv", argv), patch.object(engines, "run_openai_chat", side_effect=responses):
                self.assertEqual(validation.main(), 1)  # strict failure still fails the run
            report = json.loads(output.read_text())
        self.assertTrue(report["accept_fenced_json"])
        self.assertEqual(report["summary"]["json@c1"]["passed"], 1)
        self.assertEqual(report["summary"]["json@c1"]["lenient_passed"], 2)
        self.assertEqual([c["lenient_status"] for c in report["cases"]], ["ok", "ok"])
        self.assertEqual([c["status"] for c in report["cases"]], ["fail", "ok"])

    def test_summary_omits_lenient_counts_when_not_opted_in(self):
        with patch.object(engines, "run_openai_chat", return_value=self.workflow_response("42")):
            case = validation.run_case("http://unused", "model", "tensorsharp", "short", "plain")
        case.update(concurrency=1)
        self.assertNotIn("lenient_passed", validation.summarize([case])["short@c1"])


if __name__ == "__main__":
    unittest.main()
