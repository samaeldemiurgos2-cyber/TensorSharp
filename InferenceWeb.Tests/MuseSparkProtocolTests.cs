// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

public sealed class MuseSparkProtocolTests
{
    private const string AnswerSuffix = " to=user<|message|>";
    private static readonly string Template = ReadTemplate();
    private static readonly List<ToolFunction> Tools =
    [
        new() { Name = "weather.get", Description = "Get weather.",
            Parameters = new() { ["city"] = new() { Type = "string" } } },
        new() { Name = "time.get", Description = "Get time.", Parameters = new() },
    ];

    [Theory]
    [InlineData(null, 512)]
    [InlineData("low", 8)]
    [InlineData("medium", 32)]
    [InlineData("high", 256)]
    [InlineData("xhigh", 512)]
    public void ReleasedGgufTemplateRendersItsOwnReasoningScale(string? effort, int expected)
    {
        string prompt = ChatTemplate.RenderFromGgufTemplate(ReadFixture("unsloth_chat_template.jinja"),
            [new() { Role = "user", Content = "Describe <|eot|>.", ImagePaths = ["image.png"] }],
            architecture: "muse-spark", tools: Tools, enableThinking: true, reasoningEffort: effort);
        Assert.Contains($"Reasoning strength: {expected}.", prompt);
        Assert.Contains("<|patch|>Describe < |eot|>.", prompt);
        Assert.Contains("[func_name] (e.g., weather.get)", prompt);
        Assert.Contains("\"additionalProperties\": false", prompt);
    }

    [Fact]
    public void MissingTemplateFailsRatherThanSendingChatMlToSpark()
        => Assert.Throws<InvalidOperationException>(() => ChatTemplate.RenderFromGgufTemplate("",
            [new() { Role = "user", Content = "Hello" }], architecture: "muse-spark"));

    [Fact]
    public void ReleasedTemplateThinkingOffPrimesJsonAnswerDespiteItsReasoningDefault()
    {
        string prompt = ChatTemplate.RenderFromGgufTemplate(ReadFixture("unsloth_chat_template.jinja"),
            [new() { Role = "user", Content = "Return JSON." }], architecture: "muse-spark", enableThinking: false);
        // This release ignores disable_reasoning. Completing the user channel in
        // the prompt still makes token-zero JSON grammar compatible with it.
        Assert.Contains("Reasoning strength: 512.", prompt);
        Assert.EndsWith("<|start|>assistant" + AnswerSuffix, prompt);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void RenderMatchesIndependentPythonJinjaReference(int index)
    {
        using var data = JsonDocument.Parse(ReadFixture("jinja_reference.json"));
        var fixture = data.RootElement[index];
        var context = (Dictionary<string, object>)ChatMlOutputParser.JsonElementToObject(fixture.GetProperty("context"))!;
        var template = new Jinja2Template(ReadFixture(fixture.GetProperty("template").GetString()!));
        Assert.Equal(fixture.GetProperty("expected").GetString(), template.Render(context));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void ReleasedRequestRenderingMatchesUnchangedLlamaJinja(int index)
    {
        using var data = JsonDocument.Parse(ReadFixture("llama_jinja_matched_reference.json"));
        var fixture = data.RootElement.GetProperty("cases")[index];
        var context = fixture.GetProperty("context");
        var content = context.GetProperty("messages")[0].GetProperty("content");
        ChatMessage message = content.ValueKind == JsonValueKind.String
            ? new() { Role = "user", Content = content.GetString()! }
            : new() { Role = "user", ImagePaths = ["fixture.png"], Content = content[1].GetProperty("text").GetString()! };
        List<ToolFunction>? tools = fixture.GetProperty("has_tools").GetBoolean()
            ? [new() { Name = "weather.get", Description = "Get weather.",
                Parameters = new() { ["city"] = new() { Type = "string" } }, Required = ["city"] }]
            : null;
        string actual = ChatTemplate.RenderFromGgufTemplate(ReadFixture("unsloth_chat_template.jinja"),
            [message], architecture: "muse-spark", tools: tools,
            enableThinking: fixture.GetProperty("thinking").GetBoolean(),
            reasoningEffort: context.TryGetProperty("reasoning_effort", out var effort) ? effort.GetString() : null);
        Assert.True(fixture.GetProperty("llama_jinja_matches").GetBoolean());
        Assert.Equal(fixture.GetProperty("expected").GetString(), actual);
    }

    [Theory]
    [InlineData("muse-spark")]
    [InlineData("muse_spark")]
    public void ProtocolSupportsRoutedReasoningAndStructuredAnswers(string architecture)
    {
        var protocol = ChatProtocolRegistry.For(architecture)!;
        Assert.Equal("muse-spark", protocol.Id);
        Assert.IsType<MuseGlimmerOutputParser>(OutputParserFactory.Create(architecture));
        Assert.True(protocol.OutputParserAlwaysRequired);
        Assert.True(protocol.ThinkingBudgetUsesOutputParser);
        Assert.Equal("to=user<|message|>", protocol.ThinkingGrammarActivationTrigger);
        Assert.Equal("<|eom|><|start|>assistant", protocol.ThinkingBudgetToolTransitionText);
        Assert.True(protocol.RendersReasoningEffort);
    }

    [Theory]
    [InlineData("minimal", 1)]
    [InlineData(" LOW ", 16)]
    [InlineData("medium", 64)]
    [InlineData("high", 256)]
    public void UpstreamTemplateUsesNumericReasoningStrength(string effort, int strength)
    {
        string prompt = Render([new() { Role = "user", Content = "What is 2 + 3?" }], true, effort: effort);
        Assert.Contains($"Reasoning strength: {strength}.", prompt);
        Assert.Contains("\"self\": private reasoning", prompt);
        Assert.EndsWith("<|start|>assistant", prompt);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ThinkingOffRemovesSelfRecipientAndOnlyPrimesToolFreeReplies(bool withTools)
    {
        string prompt = Render([new() { Role = "user", Content = "Hello" }],
            false, withTools ? Tools : null, "high");
        Assert.DoesNotContain("Reasoning strength:", prompt);
        Assert.DoesNotContain("\"self\": private reasoning", prompt);
        Assert.Contains("\"commentary\": user-visible intermediate updates", prompt);
        Assert.EndsWith("<|start|>assistant" + (withTools ? "" : AnswerSuffix), prompt);
    }

    [Fact]
    public void ControlTextIsEscapedWithoutLosingActualImagePlaceholders()
    {
        string prompt = Render([new() { Role = "user", ImagePaths = ["one.png", "two.png"],
            Content = "Describe these. Literal <|eot|> <image> <audio> <video>." }]);
        Assert.Contains("<|start|>user<|message|><|patch|><|patch|>Describe these. Literal < |eot|> < image> < audio> < video>.<|eot|>", prompt);
        Assert.DoesNotContain("< |patch|>", prompt);
        Assert.DoesNotContain("<|im_start|>", prompt);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ToolHistoryUsesCommentaryAndAssociatesParallelResultsById(bool releasedTemplate)
    {
        string prompt = Render([
            new() { Role = "user", Content = "Weather and time in Paris?" },
            new() { Role = "assistant", Content = "Checking both.", Thinking = "Use two tools.", ToolCalls = [
                new() { Id = "a", Name = "weather.get", Arguments = new() { ["city"] = "Paris" } },
                new() { Id = "b", Name = "time.get", Arguments = new() },
            ] },
            new() { Role = "tool", ToolCallId = "b", Content = "12:00" },
            new() { Role = "tool", ToolCallId = "a", Content = "Sunny" },
        ], tools: Tools, template: releasedTemplate ? ReadFixture("unsloth_chat_template.jinja") : Template);
        Assert.Contains("<|start|>assistant to=self<|message|>Use two tools.<|eom|>", prompt);
        Assert.Contains("<|start|>assistant to=commentary<|message|>Checking both.<|eom|>", prompt);
        Assert.Contains("<|start|>assistant to=weather.get<|message|><atem:function_calls>", prompt);
        Assert.Contains("<atem:parameter name=\"city\">Paris</atem:parameter>", prompt);
        Assert.Contains("<|start|>tool time.get<|message|><tool_output name=\"time.get\">\n12:00", prompt);
        Assert.Contains("<|start|>tool weather.get<|message|><tool_output name=\"weather.get\">\nSunny", prompt);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(1000)]
    public void ChunkedAgenticOutputKeepsCommentaryVisibleAndParsesOnlyToolRecipients(int chunkSize)
    {
        const string generated = " to=self<|message|>Plan.<|eom|>"
            + "<|start|>assistant to=commentary<|message|>Checking both.<|eom|>"
            + "<|start|>assistant to=weather.get<|message|><atem:function_calls><atem:invoke name=\"weather.get\">"
            + "<atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke></atem:function_calls><|eom|>"
            + "<|start|>assistant to=time.get<|message|><atem:function_calls><atem:invoke name=\"time.get\">"
            + "</atem:invoke></atem:function_calls><|eot|>";
        var parser = OutputParserFactory.Create("muse-spark");
        parser.Init(true, Tools);
        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var calls = new List<ToolCall>();
        for (int i = 0; i < generated.Length; i += chunkSize)
        {
            int count = Math.Min(chunkSize, generated.Length - i);
            var parsed = parser.Add(generated.Substring(i, count), i + count == generated.Length);
            content.Append(parsed.Content);
            thinking.Append(parsed.Thinking);
            if (parsed.ToolCalls != null) calls.AddRange(parsed.ToolCalls);
        }
        Assert.Equal("Checking both.", content.ToString());
        Assert.Equal("Plan.", thinking.ToString());
        Assert.Equal(["weather.get", "time.get"], calls.Select(c => c.Name));
        Assert.Equal("Paris", calls[0].Arguments!["city"]);
        Assert.Equal([0, 1], calls.Select(c => c.Index));
    }

    [Fact]
    public void AtemArgumentsPreserveJsonNullQuotedStringsAndRawWhitespace()
    {
        var parser = OutputParserFactory.Create("muse-spark");
        parser.Init(false, Tools);
        var parsed = parser.Add(" to=weather.get<|message|><atem:function_calls><atem:invoke name=\"weather.get\">"
            + "<atem:parameter name=\"nullable\">null</atem:parameter>"
            + "<atem:parameter name=\"quoted\">\"Paris\"</atem:parameter>"
            + "<atem:parameter name=\"raw\">  New York  </atem:parameter>"
            + "</atem:invoke></atem:function_calls><|eot|>", true);
        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Null(call.Arguments!["nullable"]);
        Assert.Equal("Paris", call.Arguments["quoted"]);
        Assert.Equal("  New York  ", call.Arguments["raw"]);
    }

    [Fact]
    public void DirectStructuredAnswerStreamsWithoutWaitingForHeader()
    {
        const string json = "{\"city\":\"Paris\",\"ok\":true}";
        var tokenizer = new ByteTokenizer();
        var grammar = new GrammarConstraint(Grammar.JsonObject(), tokenizer);
        var parser = OutputParserFactory.Create("muse-spark");
        parser.Init(false, null);
        parser.SetGenerationPromptSuffix(AnswerSuffix);
        var content = new StringBuilder();
        foreach (int token in tokenizer.Encode(json))
        {
            grammar.Accept(token);
            content.Append(parser.Add(tokenizer.Decode([token]), false).Content);
        }
        Assert.True(grammar.IsComplete);
        Assert.False(grammar.IsDead);
        Assert.Equal(json, content.ToString());
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void MultiTurnRawOutputPreservesExactCachedPrefix(bool thinking, bool releasedTemplate)
    {
        var tokenizer = new ByteTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        string template = releasedTemplate ? ReadFixture("unsloth_chat_template.jinja") : Template;
        List<ChatMessage> messages = [new() { Role = "user", Content = "What is 2 + 3?" }];
        var first = renderer.RenderToTokens(tokenizer, template, messages, "muse-spark", true,
            out _, out string trailing, tools: null, enableThinking: thinking);
        string suffix = thinking ? "" : AnswerSuffix;
        var raw = tokenizer.Encode(thinking
            ? " to=self<|message|>Add.<|eom|><|start|>assistant to=user<|message|>5."
            : "5.");
        messages.Add(new() { Role = "assistant", Content = "5.", RawOutputTokens = raw,
            RawGenerationSuffix = suffix, RawPromptTrailingWhitespace = trailing });
        messages.Add(new() { Role = "user", Content = "And 2 + 4?" });
        var next = renderer.RenderToTokens(tokenizer, template, messages, "muse-spark", true,
            tools: null, enableThinking: thinking);
        var prefix = first.Concat(raw).Concat(tokenizer.Encode("<|eot|>")).ToArray();
        Assert.Equal(prefix, next.Take(prefix.Length));
    }

    private static string Render(List<ChatMessage> messages, bool thinking = false,
        List<ToolFunction>? tools = null, string? effort = null, string? template = null) =>
        ChatTemplate.RenderFromGgufTemplate(template ?? Template, messages, architecture: "muse-spark",
            tools: tools, enableThinking: thinking, reasoningEffort: effort);

    private static string ReadTemplate()
        => ReadFixture("chat_template.jinja");

    private static string ReadFixture(string name)
    {
        using var stream = typeof(MuseSparkProtocolTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.MuseSpark." + name)!;
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private sealed class ByteTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 256).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 256;
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public bool IsEos(int id) => false;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList();
        public string Decode(List<int> tokens) => Encoding.UTF8.GetString(tokens.Select(t => (byte)t).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
