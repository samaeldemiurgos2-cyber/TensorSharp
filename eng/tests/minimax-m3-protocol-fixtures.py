#!/usr/bin/env python3
"""Regenerate the MiniMax-M3.1 text-protocol fixtures in InferenceWeb.Tests/Fixtures/MiniMaxM3/.

Usage: python minimax-m3-protocol-fixtures.py <hf-model-dir> <out-dir>

<hf-model-dir> holds MiniMaxAI/MiniMax-M3.1-Flash's tokenizer.json, tokenizer_config.json and
chat_template.jinja (the template is byte-identical to the unsloth GGUF's tokenizer.chat_template).
Needs HF `transformers` (apply_chat_template) and `tokenizers`. Writes:

  tokenizer_reference.json  ids = tokenizer.json encode(add_special_tokens=False) over code, digits,
                            CJK, Hangul jamo, decomposed accents and other NFC-changing text, emoji,
                            every added token and chat renders; pieces = the pre-tokenizer split of
                            text without added tokens (offsets into the raw text, no normalizer);
                            decoded = decode(ids, skip_special_tokens=False).
  chat_reference.json       expected = apply_chat_template(messages, tools, tokenize=False, ...) for
                            the cases below; the C# test rebuilds the same request as TensorSharp
                            ChatMessage / ToolFunction objects and asserts byte equality.
"""
import json, os, random, sys, unicodedata

import tokenizers
import transformers
from tokenizers import Tokenizer
from transformers import AutoTokenizer

model_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
hf = AutoTokenizer.from_pretrained(model_dir)
template = open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf-8").read()
added = [t.content for t in hf.added_tokens_decoder.values()]

# ---------------------------------------------------------------- chat renders
weather = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
search = {"type": "function", "function": {
    "name": "search_docs", "description": "搜索文档 — returns <b>matches</b> & \"quotes\"\nline2\ttab",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
        "score": {"type": "number", "default": 0.5, "exclusiveMinimum": 0.0},
        "exact": {"type": "boolean", "default": False},
        "tags": {"type": "array", "items": {"type": "string"}},
        "filter": {"type": "object", "properties": {
            "field": {"type": "string"}, "op": {"type": "string", "enum": ["eq", "ne", "gt"]},
            "value": {"type": ["string", "number", "null"]}}, "required": ["field"]},
        "big": {"type": "number", "default": 1e20}, "tiny": {"type": "number", "default": 1e-7}},
        "required": ["query"]}}}
no_desc = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}
create_order = {"type": "function", "function": {
    "name": "create_order", "description": "Create an order.",
    "parameters": {"type": "object", "properties": {
        "user_id": {"type": "integer"}, "express": {"type": "boolean"}, "weight": {"type": "number"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "sku": {"type": "string"}, "qty": {"type": "integer"}}}},
        "address": {"type": "object", "properties": {"city": {"type": "string"}, "zip": {"type": "string"}}},
        "note": {"type": "string"}}, "required": ["user_id", "items"]}}}

def call(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}

SYS = {"role": "system", "content": "You are a concise assistant."}
U = lambda t: {"role": "user", "content": t}
A = lambda t, **kw: dict({"role": "assistant", "content": t}, **kw)
T = lambda t: {"role": "tool", "content": t}

paris_call = call("get_weather", {"city": "Paris", "unit": "celsius"})
cases = [
    ("user_only", [U("Hello!")], None, True, None),
    ("system_user", [SYS, U("What is the capital of France?")], None, True, None),
    ("developer_user", [{"role": "developer", "content": "Answer in French."}, U("Hi")], None, True, None),
    ("root_user", [{"role": "root", "content": "You are MiniMax, deployed by ACME."}, U("Who are you?")], None, True, None),
    ("root_system_mid_system", [{"role": "root", "content": "ROOT"}, {"role": "system", "content": "SYS"},
                                U("q1"), A("a1"), {"role": "system", "content": "MID"}, U("q2")], None, True, None),
    ("effort_high", [U("Prove that sqrt(2) is irrational.")], None, True, "high"),
    ("effort_low_system_tools", [SYS, U("Weather in Oslo?")], [weather], True, "low"),
    ("effort_medium_root", [{"role": "root", "content": "ROOT"}, U("x")], None, True, "medium"),
    ("tools_declared", [U("What's the weather in Paris?")], [weather], True, None),
    ("tools_unicode_nested_schema", [SYS, U("找一下 tokenizer 的文档")], [search], True, None),
    ("tool_without_description", [U("ping it")], [no_desc], True, None),
    ("three_tools", [U("go")], [weather, search, create_order], True, None),
    ("tool_round_trip", [U("What's the weather in Paris?"),
                         A("", reasoning_content="The user wants the weather in Paris. I should call get_weather.",
                           tool_calls=[paris_call]),
                         T('{"temperature": 18, "condition": "cloudy"}')], [weather], True, None),
    ("reasoning_split_from_content", [U("2+2?"), A("<mm:think>Simple arithmetic.</mm:think>4"), U("And 3+3?"),
                                      A("6"), U("And 4+4?")], None, True, None),
    ("reasoning_split_strips_newlines", [U("q"), A("\n<mm:think>\nfirst line\nsecond\n\n</mm:think>\n\nanswer\n")],
     None, False, None),
    ("empty_reasoning_content_keeps_content", [U("q"), A("x</mm:think>y", reasoning_content="")], None, True, None),
    ("reasoning_content_wins_over_content", [U("q"), A("<mm:think>ignored</mm:think>shown", reasoning_content="R")],
     None, True, None),
    ("bare_closer_history", [U("hi"), A("Hello! How can I help?"), U("Tell me a joke.")], None, True, None),
    ("two_tool_results_grouped", [U("Weather in Paris and Rome?"),
                                  A("", tool_calls=[paris_call, call("get_weather", {"city": "Rome"})]),
                                  T("r1"), T("r2")], [weather], True, None),
    ("parallel_calls_mixed_args", [U("order"),
        A("Placing the order now.", reasoning_content="Need create_order and weather.", tool_calls=[
            call("create_order", {"user_id": 42, "express": True, "weight": 1.5, "note": None,
                                  "items": [{"sku": "book-001", "qty": 2}, {"sku": "pen", "qty": 10}],
                                  "address": {"city": "Zürich", "zip": "8001", "extra": None}}),
            call("get_weather", {"city": "a <b> & \"c\"\nnext line", "unit": "fahrenheit"}),
            call("ping", {})]),
        T("ok"), T("{\"temp\": 3}"), T("pong")], [create_order, weather, no_desc], True, None),
    ("no_arg_call", [U("ping"), A("", tool_calls=[call("ping", {})]), T("pong")], [no_desc], True, None),
    ("tool_then_user", [U("weather?"), A("", tool_calls=[paris_call]), T("sunny"), A("It is sunny."),
                        U("Thanks! And tomorrow?")], [weather], True, None),
    ("image_user", [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}],
     None, True, None),
    ("two_images_user", [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                                       {"type": "text", "text": "Compare them."}]}], None, True, None),
    ("image_only_user", [{"role": "user", "content": [{"type": "image"}]}], None, True, None),
    ("no_generation_prompt_ends_assistant", [SYS, U("hi"), A("hello")], None, False, None),
    ("no_generation_prompt_ends_user", [U("hi")], None, False, None),
    ("all_turn_reasoning", [U("a"), A("b", reasoning_content="r1"), U("c"), A("d"), U("e"),
                            A("f", reasoning_content="r3\nmultiline"), U("g")], None, True, None),
    ("literal_markup_in_user", [U("Is <tool_call> or </mm:think> or ]<]minimax[>[ special?")], None, True, None),
    ("empty_system_defaults", [{"role": "system", "content": ""}, U("hi"), {"role": "system", "content": ""}, U("again")],
     None, True, None),
    ("unicode_everywhere", [SYS, U("你好 🌍 — Grüße! 한국어 😀"), A("Bonjour 🌸", reasoning_content="思考中…"),
                            U("ok")], None, True, None),
    ("everything_header", [{"role": "root", "content": "ROOT"}, {"role": "developer", "content": "DEV"}, U("go")],
     [weather, search], True, "high"),
    ("null_content_with_calls", [U("w"), {"role": "assistant", "content": None, "tool_calls": [paris_call]},
                                 T("rain")], [weather], True, None),
    ("agent_loop", [SYS, U("Plan a trip."),
                    A("", reasoning_content="Check weather first.", tool_calls=[paris_call]), T("18C"),
                    A("Now Rome.", reasoning_content="Then Rome.", tool_calls=[call("get_weather", {"city": "Rome"})]),
                    T("25C"), A("Paris is 18C and Rome is 25C.", reasoning_content="Summarize."),
                    U("Which is warmer?")], [weather], True, None),
    ("float_args", [U("f"), A("", tool_calls=[call("create_order", {"user_id": -7, "weight": 0.1, "items": [],
                                                                     "express": False})]), T("done")],
     [create_order], True, None),
    ("float_args_wide", [U("f"), A("", tool_calls=[call("create_order", {"user_id": 0, "weight": 2.0, "items": [1e20, 1e-07, -3.5, 123456789.125, 1e16, 0.0001]})]), T("done")],
     [create_order], True, None),
    ("developer_then_developer", [{"role": "developer", "content": "D1"}, U("u"), {"role": "developer", "content": "D2"},
                                  A("x"), U("y")], None, True, None),
    ("nested_lists_and_none_items", [U("n"), A("", tool_calls=[call("create_order", {
        "user_id": 1, "items": [[1, 2], [], {"a": None, "b": [True, None]}], "note": "multi\nline"})]), T("ok")],
     [create_order], True, None),
]

chat = []
for name, messages, tools, gen, effort in cases:
    kwargs = {}
    if effort is not None:
        kwargs["reasoning_effort"] = effort
    rendered = hf.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=gen,
                                      chat_template=template, **kwargs)
    chat.append({"name": name, "messages": messages, "tools": tools, "add_generation_prompt": gen,
                 "reasoning_effort": effort, "expected": rendered})

# ---------------------------------------------------------------- tokenizer corpus
coder = "\U0001F9D1\U0001F3FD‍\U0001F4BB"
hand = [
    "INV-472", "silver-4821", " is 17 + 25", "1", "12", "123", "1234", "12345", "1234567", "3.14159", "-0.000123",
    "1e10", "6.02E+23", "$1,234,567.89", "+1 (415) 555-0199", "2026-10-23T12:34:56Z", "0x1F4A9", "v10.0.401",
    "192.168.0.1:8080", "１２３", "٤٥٦٧", "²³¹⁴", "ⅫⅩⅠ",
    "你好，世界！", "价格是12345元。", "日本語のテキスト123です",
    "한국어 텍스트 2026년", "مرحبا بالعالم 123",
    "Привет, мир! 42", "हिन्दी पाठ १२३",
    "ไทย ๑๒๓", "don't", "DON'T", "we'll", "I'm", "THEY'RE", "O'Neil's", "'s", "rock 'n' roll",
    "def f(x):\n    return x**2 + 1\n", "for (int i = 0; i < 100; i++) {\n\tsum += a[i];\n}", "{\"name\": \"Mars\", \"moons\": 2}",
    "public static async Task<int> RunAsync(CancellationToken ct = default) => await Task.FromResult(42);",
    "  leading", "trailing  ", "a  b   c    d", "\n\n\n", "\r\n\r\n", " \n \n ", "\t\t\tx", "x y", "x　y",
    "\U0001F600", "hello\U0001F600world", " \U0001F600", coder + " coding", "\U0001F1FA\U0001F1F8\U0001F1E8\U0001F1F3",
    "\U0001F468‍\U0001F469‍\U0001F467 family", "❤️‍\U0001F525", "\U0001F44D\U0001F3FF",
    "a/b\\c|d", "!!!???", "...", "--flag=value", "email@example.com", "https://example.com/a?b=1&c=22",
    "mixed中文English123数字", "x1y22z333w4444", "SELECT * FROM t WHERE id = 42;",
    "#include <stdio.h>\nint main(){return 0;}", "λx.x+1", "∑_{i=1}^{n} i² = n(n+1)/2",
    "​zero​width", "private", "control\u0001char", "­ soft hyphen", "a b c",
    "\U0001D7CE\U0001D7CF\U0001D7D0", "\U00020000\U00020001abc", "\U0001D400\U0001D401 math bold",
    # NFC: composition, canonical singletons, Hangul jamo, ordering of marks, composition exclusions
    "école", "Café naı̈ve", "Ångström", "Ångström", "Ω ohm", "Kelvin K",
    "한글", "가 뎌", "Việt Nam", "Việt Nam",
    "ą́", "ą́", "क़ क़", "豈更車", " x y", ";·",
    "ȫ", "é́", "Ǻ", "ñõ", "̣̇ lonely marks",
    "שּׁ", "آ", "ゔパ", "각", "ẛ̣",
    # MiniMax control / user-defined tokens, alone, adjacent and inside text
    "]~!b[]~b]system\nhello[e~[\n", "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"f\">]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
    "<mm:think>reasoning</mm:think>answer", "</mm:think>direct", "x<think>y</think>z", "]~b]~!b[", "[e~[[e~[",
    "]<]image[>[]<]image[>[Describe", "]<]start of image[>[]<]image[>[]<]end of image[>[", "]<]video[>[",
    "<|content_altered_placeholder|>", "<fim_prefix>a<fim_suffix>b<fim_middle>", "]!p~[", "]!d~[",
    "<tool_call> without namespace", "] < ] minimax [ > [", "<mm:think", "mm:think>", "]<]minimax[>",
]
hand += added
frags = ["INV-", "472", "12345678", "3.5", " ", "  ", "\n", "\r\n", "\t", "中文", "テスト", "한국",
         "한", "é", "Å", "豈", "\U0001F600", coder, "'s", "'LL", "don't", "Hello", "world",
         "_x", "{", "}", "(", ")", "[", "]", ":", ";", ",", ".", "!", "?", "\"", "'", "٣", "²", "é",
         " ", "　", "return", "def", "=", "==", "->", "/", "\\", "0", "000", "#", "<", ">", "~",
         "]<]minimax[>[", "<tool_call>", "</tool_call>", "<mm:think>", "</mm:think>", "]~b]", "[e~[", "]<]image[>[",
         "\U0001D7CE", "\U00020000"]
rng = random.Random(20261023)
gen = ["".join(rng.choice(frags) for _ in range(rng.randint(1, 12))) for _ in range(260)]
renders = [c["expected"] for c in chat if c["name"] in (
    "tool_round_trip", "parallel_calls_mixed_args", "unicode_everywhere", "image_user", "everything_header")]

texts, seen = [], set()
for t in hand + gen + renders:
    if t not in seen:
        seen.add(t); texts.append(t)

cases_out = []
for t in texts:
    ids = tok.encode(t, add_special_tokens=False).ids
    entry = {"text": t, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)}
    if not any(a in t for a in added):
        entry["pieces"] = [t[a:b] for _, (a, b) in tok.pre_tokenizer.pre_tokenize_str(t)]
        assert "".join(entry["pieces"]) == t, (t, entry["pieces"])
    entry["nfc_changes"] = unicodedata.normalize("NFC", t) != t
    cases_out.append(entry)

versions = f"transformers {transformers.__version__}, tokenizers {tokenizers.__version__}"
with open(os.path.join(out_dir, "tokenizer_reference.json"), "w", encoding="utf-8") as f:
    json.dump({"source": "MiniMaxAI/MiniMax-M3.1-Flash tokenizer.json via " + versions +
               ": ids = encode(add_special_tokens=False) (NFC normalizer + minimax-m2 split + byte-level BPE), "
               "pieces = pre_tokenizer.pre_tokenize_str offsets (text without added tokens), "
               "decoded = decode(ids, skip_special_tokens=False). Regenerate with "
               "eng/tests/minimax-m3-protocol-fixtures.py.", "cases": cases_out},
              f, ensure_ascii=False, indent=0)
with open(os.path.join(out_dir, "chat_reference.json"), "w", encoding="utf-8") as f:
    json.dump({"source": "MiniMaxAI/MiniMax-M3.1-Flash chat_template.jinja (byte-identical to the unsloth GGUF "
               "tokenizer.chat_template) rendered by apply_chat_template, " + versions +
               ". Regenerate with eng/tests/minimax-m3-protocol-fixtures.py.", "cases": chat},
              f, ensure_ascii=False, indent=1)
print(len(cases_out), "tokenizer cases;", len(chat), "chat cases")
