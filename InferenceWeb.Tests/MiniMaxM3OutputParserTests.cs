// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="MiniMaxM3OutputParser"/>: the outcomes of llama.cpp's MiniMax-M3 parser tests
/// (llama.cpp minimax-m31 branch, tests/test-chat.cpp, "MiniMax-M3 tests") with the same tool
/// schemas, streamed whole, one character at a time and in random pieces, plus the adaptive
/// openers, malformed and truncated blocks, repeated tags and the JSON-mode trigger.
/// </summary>
public class MiniMaxM3OutputParserTests
{
    private const string Ns = "]<]minimax[>[";
    private const string Open = Ns + "<tool_call>\n";
    private const string Close = Ns + "</tool_call>";

    private static string Invoke(string name, string body) => $"{Ns}<invoke name=\"{name}\">{body}{Ns}</invoke>\n";
    private static string Arg(string name, string value) => $"{Ns}<{name}>{value}{Ns}</{name}>";

    // llama.cpp tests/test-chat.cpp tool declarations used by the MiniMax-M3 cases.
    private static readonly List<ToolFunction> Tools = ToolFunction.ParseList("""
        [
          {"name": "special_function", "description": "I'm special", "parameters": {"type": "object",
            "properties": {"arg1": {"type": "integer", "description": "The arg."}}, "required": ["arg1"]}},
          {"name": "empty_args", "description": "A tool that takes no arguments", "parameters": {"type": "object", "properties": {}}},
          {"name": "get_time", "description": "Get the current time in a city", "parameters": {"type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}},
          {"name": "get_weather", "description": "Get the current weather in a city", "parameters": {"type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}},
          {"name": "magic_int", "description": "Magic tool that takes a hash", "parameters": {"type": "object",
            "properties": {"ref": {"type": "integer"}, "name": {"type": "string"}}, "required": ["ref"]}},
          {"name": "amount", "description": "Amount converter", "parameters": {"type": "object",
            "properties": {"orig": {"type": "number"}}, "required": ["orig"]}},
          {"name": "toggle", "description": "Toggle a feature", "parameters": {"type": "object",
            "properties": {"enabled": {"type": "boolean", "description": "Whether to enable the feature"}}, "required": ["enabled"]}},
          {"name": "imaginary_number", "description": "Imaginary number converter", "parameters": {"type": "object",
            "properties": {"number": {"type": "object", "properties": {"real": {"type": "number"}, "imaginary": {"type": "number"}},
              "required": ["real", "imaginary"]}}, "required": ["number"]}},
          {"name": "nested_args", "description": "Tool with nested array arguments", "parameters": {"type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}},
              "entries": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"},
                "label": {"type": "string"}}, "required": ["id", "label"]}}}, "required": ["tags", "entries"]}},
          {"name": "union_args", "description": "Tool with union arguments", "parameters": {"type": "object",
            "properties": {
              "filter": {"anyOf": [{"type": "array", "items": {"type": "string"}},
                {"type": "object", "properties": {"field": {"type": "string"}, "op": {"type": "string"}}, "required": ["field", "op"]}]},
              "label": {"oneOf": [{"type": "string"}, {"type": "object", "properties": {"text": {"type": "string"}}}]},
              "limit": {"oneOf": [{"type": "integer"}, {"type": "object", "properties": {"max": {"type": "integer"}}, "required": ["max"]}]}}}},
          {"name": "config", "description": "Settings", "parameters": {"type": "object", "properties": {
            "meta": {"type": "object"}, "list": {"type": "array"}, "any": {}, "maybe": {"type": ["integer", "null"]},
            "ref": {"$ref": "#/$defs/Point"}, "kind": {"enum": ["a", "b"]}, "level": {"enum": [1, 2, 3]}},
            "$defs": {"Point": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}}}}
        ]
        """);

    public static IEnumerable<object[]> LlamaCases() =>
    [
        // name, model output, reasoning, content, calls as "name:{json}" joined by '|'
        ["content_only", "</mm:think>Hello, world!\nWhat's up?", "", "Hello, world!\nWhat's up?", ""],
        ["thinking_content", "<mm:think>I'm\nthinking</mm:think>Hello, world!\nWhat's up?", "I'm\nthinking", "Hello, world!\nWhat's up?", ""],
        ["thinking_call", "<mm:think>Let me check the time</mm:think>" + Open + Invoke("get_time", Arg("city", "Tokyo")) + Close,
            "Let me check the time", "", "get_time:{\"city\":\"Tokyo\"}"],
        ["call_without_reasoning_int", "</mm:think>" + Open + Invoke("special_function", Arg("arg1", "1")) + Close,
            "", "", "special_function:{\"arg1\":1}"],
        ["no_parameters", "<mm:think>Let's call a tool:</mm:think>" + Open + Invoke("empty_args", "") + Close,
            "Let's call a tool:", "", "empty_args:{}"],
        ["parallel", "<mm:think>Calling both</mm:think>" + Open + Invoke("get_time", Arg("city", "Paris")) +
            Invoke("get_weather", Arg("city", "Paris")) + Close,
            "Calling both", "", "get_time:{\"city\":\"Paris\"}|get_weather:{\"city\":\"Paris\"}"],
        ["content_before_call", "<mm:think>Thinking about it</mm:think>Let me call the function." + Open +
            Invoke("special_function", Arg("arg1", "1")) + Close,
            "Thinking about it", "Let me call the function.", "special_function:{\"arg1\":1}"],
        ["negative", "<mm:think>Test negative</mm:think>" + Open + Invoke("magic_int", Arg("ref", "-14")) + Close,
            "Test negative", "", "magic_int:{\"ref\":-14}"],
        ["decimal", "<mm:think>Test decimal</mm:think>" + Open + Invoke("amount", Arg("orig", "3.14")) + Close,
            "Test decimal", "", "amount:{\"orig\":3.14}"],
        ["boolean", "<mm:think>Test boolean</mm:think>" + Open + Invoke("toggle", Arg("enabled", "true")) + Close,
            "Test boolean", "", "toggle:{\"enabled\":true}"],
        ["mixed", "<mm:think>Multi-arg call</mm:think>" + Open + Invoke("magic_int", Arg("ref", "42") + Arg("name", "foo bar")) + Close,
            "Multi-arg call", "", "magic_int:{\"ref\":42,\"name\":\"foo bar\"}"],
        ["nested_object", "<mm:think>Nested object</mm:think>" + Open +
            Invoke("imaginary_number", Arg("number", Arg("real", "1.5") + Arg("imaginary", "-2.5"))) + Close,
            "Nested object", "", "imaginary_number:{\"number\":{\"real\":1.5,\"imaginary\":-2.5}}"],
        ["nested_arrays", "<mm:think>Nested arrays</mm:think>" + Open + Invoke("nested_args",
            Arg("tags", Arg("item", "alpha") + Arg("item", "beta")) +
            Arg("entries", Arg("item", Arg("id", "1") + Arg("label", "one")))) + Close,
            "Nested arrays", "", "nested_args:{\"tags\":[\"alpha\",\"beta\"],\"entries\":[{\"id\":1,\"label\":\"one\"}]}"],
        ["union_array", "<mm:think>Union array</mm:think>" + Open +
            Invoke("union_args", Arg("filter", Arg("item", "alpha") + Arg("item", "beta"))) + Close,
            "Union array", "", "union_args:{\"filter\":[\"alpha\",\"beta\"]}"],
        ["union_scalar", "<mm:think>Union scalar</mm:think>" + Open + Invoke("union_args", Arg("limit", "5")) + Close,
            "Union scalar", "", "union_args:{\"limit\":5}"],
        ["union_nested", "<mm:think>Union nested</mm:think>" + Open + Invoke("union_args", Arg("limit", Arg("max", "10"))) + Close,
            "Union nested", "", "union_args:{\"limit\":{\"max\":10}}"],
        ["union_string", "<mm:think>Union string</mm:think>" + Open + Invoke("union_args", Arg("label", "hi")) + Close,
            "Union string", "", "union_args:{\"label\":\"hi\"}"],
        ["union_string_markup", "<mm:think>Union string</mm:think>" + Open + Invoke("union_args", Arg("label", Arg("text", "hi"))) + Close,
            "Union string", "", "union_args:{\"label\":\"]<]minimax[>[<text>hi]<]minimax[>[</text>\"}"],
        ["empty_reasoning", "<mm:think></mm:think>" + Open + Invoke("get_time", Arg("city", "XYZCITY")) + Close,
            "", "", "get_time:{\"city\":\"XYZCITY\"}"],
    ];

    private static (string Thinking, string Content, List<ToolCall> Calls) Parse(
        IEnumerable<string> pieces, bool thinking = false, string? suffix = null, List<ToolFunction>? tools = null)
    {
        var parser = new MiniMaxM3OutputParser();
        parser.Init(thinking, tools ?? Tools);
        if (suffix != null)
            parser.SetGenerationPromptSuffix(suffix);
        var reasoning = new StringBuilder();
        var content = new StringBuilder();
        var calls = new List<ToolCall>();
        var all = pieces.ToList();
        for (int i = 0; i < all.Count; i++)
        {
            ParsedOutput output = parser.Add(all[i], i == all.Count - 1);
            reasoning.Append(output.Thinking);
            content.Append(output.Content);
            if (output.ToolCalls != null)
                calls.AddRange(output.ToolCalls);
        }
        if (all.Count == 0)
        {
            ParsedOutput output = parser.Add("", true);
            content.Append(output.Content);
        }
        return (reasoning.ToString(), content.ToString(), calls);
    }

    private static readonly JsonSerializerOptions Relaxed =
        new() { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private static string Calls(List<ToolCall> calls)
        => string.Join("|", calls.Select(c => c.Name + ":" + JsonSerializer.Serialize(c.Arguments, Relaxed)));

    /// <summary>The same text whole, one character at a time and in seeded random pieces.</summary>
    private static IEnumerable<IEnumerable<string>> Chunkings(string text)
    {
        yield return [text];
        yield return text.Select(c => c.ToString());
        var random = new Random(text.Length);
        for (int round = 0; round < 8; round++)
        {
            var pieces = new List<string>();
            for (int at = 0; at < text.Length;)
            {
                int length = Math.Min(text.Length - at, random.Next(1, 9));
                pieces.Add(text.Substring(at, length));
                at += length;
            }
            yield return pieces;
        }
    }

    [Theory]
    [MemberData(nameof(LlamaCases))]
    public void LlamaCppCases_ParseTheSameAtEveryChunking(string name, string output, string reasoning, string content, string calls)
    {
        foreach (var pieces in Chunkings(output))
        {
            var parsed = Parse(pieces);
            Assert.True((reasoning, content, calls) == (parsed.Thinking, parsed.Content, Calls(parsed.Calls)),
                $"{name}: got reasoning={JsonSerializer.Serialize(parsed.Thinking)} content={JsonSerializer.Serialize(parsed.Content)} " +
                $"calls={Calls(parsed.Calls)}");
        }
    }

    [Fact]
    public void ThinkingPrompt_StartsInsideTheBlock()
    {
        // Init(thinking: true): the prompt ended `]~b]ai\n<mm:think>`.
        foreach (var pieces in Chunkings("Plan it.</mm:think>Done."))
            Assert.Equal(("Plan it.", "Done."), Pick(Parse(pieces, thinking: true)));
        // The model repeating the opener the prompt already wrote is not reasoning text.
        Assert.Equal(("Plan.", "Done."), Pick(Parse(["<mm:think>Plan.</mm:think>Done."], thinking: true)));
        // The recorded suffix wins over Init, in either direction.
        Assert.Equal(("Plan.", "Done."), Pick(Parse(["Plan.</mm:think>Done."], thinking: false, suffix: "<mm:think>")));
        Assert.Equal(("", "Done."), Pick(Parse(["</mm:think>Done."], thinking: true, suffix: "")));
        Assert.Equal(("", "Done.</mm:think>"), Pick(Parse(["Done.</mm:think>"], thinking: true, suffix: "]~b]ai\n</mm:think>")));
        // Cut off while reasoning: everything so far is reasoning.
        Assert.Equal(("Still going", ""), Pick(Parse(["Still go", "ing"], thinking: true)));
    }

    private static (string, string) Pick((string Thinking, string Content, List<ToolCall> Calls) parsed)
        => (parsed.Thinking, parsed.Content);

    [Fact]
    public void AdaptiveStart_HoldsOnlyWhatCouldStillBeAnOpener()
    {
        var parser = new MiniMaxM3OutputParser();
        parser.Init(false, null);
        Assert.Equal("", parser.Add("\n</mm:", false).Content);
        ParsedOutput closed = parser.Add("think>Answer", false);
        Assert.Equal(("Answer", ""), (closed.Content, closed.Thinking));

        parser.Init(false, null);
        Assert.Equal("", parser.Add("<mm:th", false).Content);
        Assert.True(!parser.IsThinkingChannelOpen);
        Assert.Equal("Why", parser.Add("ink>Why", false).Thinking);
        Assert.True(parser.IsThinkingChannelOpen);

        // No opener: the reply is the answer from its first character, whitespace included.
        Assert.Equal(("", "  Plain answer <mm:think> later"), Pick(Parse(["  Plain", " answer <mm:think> later"])));
        Assert.Equal(("", "<mm:th"), Pick(Parse(["<mm:th"])));
        Assert.Equal(("", ""), Pick(Parse([])));
    }

    [Fact]
    public void Content_HoldsAPartialCallOpenerAcrossPieces()
    {
        var parser = new MiniMaxM3OutputParser();
        parser.Init(false, Tools);
        Assert.Equal("Calling", parser.Add("</mm:think>Calling]<]minimax[>[", false).Content);
        ParsedOutput call = parser.Add("<tool_call>\n" + Invoke("get_time", Arg("city", "Oslo")) + Close, true);
        Assert.Equal("", call.Content);
        Assert.Equal("get_time:{\"city\":\"Oslo\"}", Calls(call.ToolCalls!));

        // A namespace token that does not open a block is the answer's text.
        Assert.Equal(("", "x ]<]minimax[>[ y"), Pick(Parse(["</mm:think>x ]<]minimax[>[", " y"])));
    }

    [Fact]
    public void Arguments_TypedBySchema()
    {
        var (_, content, calls) = Parse(["</mm:think>" + Open + Invoke("config",
            Arg("meta", "{\"k\": [1, 2]}") + Arg("list", "") + Arg("any", "5") + Arg("maybe", "null") +
            Arg("ref", Arg("x", "1.5") + Arg("y", "2")) + Arg("kind", " a ") + Arg("level", "2") +
            Arg("unknown", "7") + Arg("unknown_obj", Arg("p", "q"))) + Close]);
        Assert.Equal("", content);
        Assert.Equal("config:{\"meta\":{\"k\":[1,2]},\"list\":[],\"any\":5,\"maybe\":null,\"ref\":{\"x\":1.5,\"y\":2}," +
                     "\"kind\":\" a \",\"level\":2,\"unknown\":\"7\",\"unknown_obj\":{\"p\":\"q\"}}", Calls(calls));

        // Repeated elements become a list; text that does not fit the type stays the model's text.
        (_, _, calls) = Parse(["</mm:think>" + Open + Invoke("magic_int", Arg("name", "a") + Arg("name", "b") + Arg("name", "c")) +
                               Invoke("magic_int", Arg("ref", "forty-two")) + Close]);
        Assert.Equal("magic_int:{\"name\":[\"a\",\"b\",\"c\"]}|magic_int:{\"ref\":\"forty-two\"}", Calls(calls));
        Assert.Equal(new[] { 0, 1 }, calls.Select(c => c.Index));

        // A string keeps every byte the model wrote, newlines and markup-like text included.
        (_, _, calls) = Parse(["</mm:think>" + Open + Invoke("get_time", Arg("city", "\n a <b> & \"c\"\n")) + Close]);
        Assert.Equal("\n a <b> & \"c\"\n", calls.Single().Arguments["city"]);

        // A tool the request did not declare: text stays text, nested elements are objects.
        (_, _, calls) = Parse(["</mm:think>" + Open + Invoke("undeclared", Arg("n", "1") + Arg("o", Arg("k", "v"))) + Close], tools: []);
        Assert.Equal("undeclared:{\"n\":\"1\",\"o\":{\"k\":\"v\"}}", Calls(calls));
    }

    [Fact]
    public void FlatToolDeclaration_StillTypesItsParameters()
    {
        var flat = new List<ToolFunction>
        {
            new() { Name = "f", Parameters = new() { ["n"] = new ToolParameter { Type = "integer" }, ["s"] = new ToolParameter { Type = "string" } } },
        };
        var (_, _, calls) = Parse(["</mm:think>" + Open + Invoke("f", Arg("n", "3") + Arg("s", "3")) + Close], tools: flat);
        Assert.Equal("f:{\"n\":3,\"s\":\"3\"}", Calls(calls));
    }

    [Fact]
    public void MalformedBlocks_DegradeToContentWithoutThrowing()
    {
        string[] malformed =
        [
            Open + "not a call" + Close,
            Open + Ns + "<invoke name=\"get_time\">" + Ns + "<city>Paris" + Ns + "</invoke>\n" + Close,
            Open + Ns + "<invoke name=\"\">" + Ns + "</invoke>\n" + Close,
            Open + Ns + "<invoke name=\"get_time\">stray" + Ns + "</invoke>\n" + Close,
            Open + Ns + "<invoke name=\"get_time\">" + Ns + "</city>" + Ns + "</invoke>\n" + Close,
        ];
        foreach (string block in malformed)
        {
            foreach (var pieces in Chunkings("</mm:think>Text " + block + " tail"))
            {
                var (_, content, calls) = Parse(pieces);
                Assert.Empty(calls);
                Assert.Equal("Text " + block + " tail", content);
            }
        }

        // One bad invoke beside a good one: the good one is a call, the bad one stays visible.
        string bad = Ns + "<invoke name=\"get_time\">oops" + Ns + "</invoke>\n";
        var mixed = Parse(["</mm:think>" + Open + Invoke("get_time", Arg("city", "Rome")) + bad + Close]);
        Assert.Equal("get_time:{\"city\":\"Rome\"}", Calls(mixed.Calls));
        Assert.Equal(bad.TrimEnd('\n'), mixed.Content);
    }

    [Fact]
    public void TruncatedBlock_DispatchesOnlyCompleteCalls()
    {
        // Stopped after the last </invoke> (EOS without </tool_call>), or mid-way through
        // the next one (max_tokens).
        string complete = "</mm:think>" + Open + Invoke("get_time", Arg("city", "Rome"));
        foreach (string reply in new[] { complete, complete + Ns + "<invoke name=\"get_weather\">" + Ns + "<city>Ro" })
        {
            foreach (var pieces in Chunkings(reply))
            {
                var parsed = Parse(pieces);
                Assert.Equal("get_time:{\"city\":\"Rome\"}", Calls(parsed.Calls));
                Assert.Equal("", parsed.Content);
            }
        }
        // Nothing complete at all: the text is shown, not swallowed.
        string cut = "</mm:think>" + Open + Ns + "<invoke name=\"get_ti";
        Assert.Equal(cut.Substring("</mm:think>".Length), Parse([cut]).Content);
    }

    [Fact]
    public void OpenBlock_StreamsProgressButNoCallUntilItCloses()
    {
        var parser = new MiniMaxM3OutputParser();
        parser.Init(false, Tools);
        string reply = "</mm:think>" + Open + Invoke("get_time", Arg("city", "Rome")) + Close;
        var progress = new StringBuilder();
        var names = new List<string?>();
        List<ToolCall>? calls = null;
        foreach (char c in reply)
        {
            ParsedOutput output = parser.Add(c.ToString(), false);
            progress.Append(output.ToolCallText);
            names.Add(output.ToolCallName);
            if (output.ToolCalls != null)
            {
                Assert.Null(calls);
                calls = output.ToolCalls;
            }
        }
        Assert.Equal("get_time:{\"city\":\"Rome\"}", Calls(calls!));
        // Everything between the block's opener and its close, as it was written.
        Assert.Equal("\n" + Invoke("get_time", Arg("city", "Rome")), progress.ToString());
        Assert.Contains("get_time", names);
    }

    [Fact]
    public void TextAfterTheBlock_IsContent()
    {
        var parsed = Parse(["</mm:think>Before." + Open + Invoke("get_time", Arg("city", "Rome")) + Close + "After."]);
        Assert.Equal(("Before.After.", "get_time:{\"city\":\"Rome\"}"), (parsed.Content, Calls(parsed.Calls)));
    }

    [Fact]
    public void RenderedToolRound_ParsesBackToTheSameCall()
    {
        // What the renderer writes for a call is what the parser reads back.
        var call = new ToolCall
        {
            Name = "nested_args",
            Arguments = new Dictionary<string, object?>
            {
                ["tags"] = new List<object?> { "alpha", "beta" },
                ["entries"] = new List<object?> { new Dictionary<string, object?> { ["id"] = 1L, ["label"] = "one" } },
            },
        };
        string prompt = ChatTemplate.RenderMiniMaxM3(
            [new() { Role = "user", Content = "u" }, new() { Role = "assistant", Content = "", Thinking = "R", ToolCalls = [call] }],
            addGenerationPrompt: false, tools: Tools);
        string turn = prompt.Substring(prompt.LastIndexOf("]~b]ai\n", StringComparison.Ordinal) + "]~b]ai\n".Length);
        turn = turn.Substring(0, turn.Length - "[e~[\n".Length);
        var parsed = Parse([turn]);
        Assert.Equal("R", parsed.Thinking);
        Assert.Equal(Calls([call]), Calls(parsed.Calls));
    }

    /// <summary>response_format: the JSON grammar armed after <c>&lt;/mm:think&gt;</c> leaves
    /// the model's trained prefix free in both thinking modes, and the parser returns the JSON
    /// as the content.</summary>
    [Fact]
    public void JsonMode_GrammarAllowsTheThinkPrefixAndParserReturnsTheJson()
    {
        // 1 = </mm:think> and 2 = <mm:think> as control tokens; the rest is text.
        var tokens = new BytesTokenizer("</mm:think>", "<mm:think>", "Weigh it.", "{", "\"ok\"", ":", "true", "}", "Sure", " ");
        tokens.Special = [0, 1, 2];
        GrammarMaskCache cache = GrammarLibrary.ForJsonObject(tokens);
        foreach (bool thinking in new[] { false, true })
        {
            int[][] replies = thinking
                ? [[3, 1, 10, 4, 5, 6, 7, 8]]
                : [[1, 4, 5, 6, 7, 8], [2, 3, 1, 4, 5, 6, 7, 8]];
            foreach (int[] reply in replies)
            {
                GrammarConstraint constraint = GrammarLibrary.NewConstraint(cache, tokens);
                constraint.ActivateAfter(OutputParserFactory.GrammarActivationTrigger("minimax-m3", thinking)!, skipLeadingWhitespace: true);
                var parser = OutputParserFactory.Create("minimax-m3");
                parser.Init(thinking, null);
                parser.SetGenerationPromptSuffix(KVCachePromptRenderer.GetAssistantGenerationSuffix("minimax-m3", thinking));
                var content = new StringBuilder();
                foreach (int id in reply)
                {
                    Assert.True(Allowed(constraint, id), $"thinking={thinking}: token {tokens.Vocab[id]} masked");
                    constraint.Accept(id);
                    if (id == 1)
                    {
                        Assert.True(constraint.IsActive);
                        Assert.False(Allowed(constraint, 9));
                    }
                    content.Append(parser.Add(tokens.Vocab[id], false).Content);
                }
                content.Append(parser.Add("", true).Content);
                Assert.True(constraint.IsComplete && !constraint.IsDead);
                Assert.Equal("{\"ok\":true}", content.ToString().Trim());
            }
        }
    }

    private static bool Allowed(GrammarConstraint constraint, int id)
        => (constraint.CurrentMask()[id >> 6] & (1UL << (id & 63))) != 0;

    /// <summary>
    /// Tool schemas are client input. A union that refers back to itself used to recurse in
    /// ContainerSchema until the stack overflowed (killing the process), and a union of two refs to
    /// the same target fanned out 2^64 calls in MayBeString/Types (a hung thread). Both must parse
    /// promptly, with the argument kept as text because the schema says nothing usable.
    /// </summary>
    [Theory]
    [InlineData("""{"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}""")]
    [InlineData("""{"anyOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/A"}]}""")]
    public void HostileRecursiveSchemas_ParseWithBoundedWork(string definition)
    {
        string json = "[{\"name\": \"deep\", \"description\": \"Self-referential parameter\", \"parameters\": "
            + "{\"type\": \"object\", \"properties\": {\"p\": {\"$ref\": \"#/$defs/A\"}}, \"required\": [\"p\"], "
            + "\"$defs\": {\"A\": " + definition + "}}}]";
        var tools = ToolFunction.ParseList(json);
        string text = "</mm:think>" + $"{Ns}<tool_call>\n" + Invoke("deep", Arg("p", $"{Ns}<x>1{Ns}</x>")) + $"{Ns}</tool_call>";
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var parsed = Parse([text], tools: tools);
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5), $"parse took {watch.Elapsed}");
        Assert.Equal("deep", Assert.Single(parsed.Calls).Name);
    }

    private sealed class BytesTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        public int[] Special = [0];
        public BytesTokenizer(params string[] pieces) => Vocab = new[] { "[e~[" }.Concat(pieces).ToArray();
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => [0];
        public int VocabSize => Vocab.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => Special;
        public List<int> Encode(string text, bool addSpecial = true) => [LookupToken(text)];
        public string Decode(List<int> ids) => string.Concat(ids.Select(id => Vocab[id]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }
}
