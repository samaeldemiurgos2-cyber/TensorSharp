// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-M3.1's chat protocol: the purpose-built renderer against HF renders of the GGUF's
/// own template, the registry entry, the thinking modes and multi-turn KV reuse.
/// Fixtures/MiniMaxM3/chat_reference.json holds <c>apply_chat_template</c> output of
/// MiniMaxAI/MiniMax-M3.1-Flash's chat_template.jinja (byte-identical to the unsloth GGUF's
/// tokenizer.chat_template); regenerate with eng/tests/minimax-m3-protocol-fixtures.py.
/// </summary>
public class MiniMaxM3ProtocolTests
{
    private const string Arch = "minimax-m3";
    private const string Ns = "]<]minimax[>[";

    internal sealed record RenderCase(string Name, List<ChatMessage> Messages, List<ToolFunction>? Tools,
        bool AddGenerationPrompt, string? ReasoningEffort, string Expected);

    private static readonly Lazy<IReadOnlyDictionary<string, RenderCase>> Cases = new(LoadCases);

    internal static IReadOnlyDictionary<string, RenderCase> LoadCases()
    {
        using var stream = typeof(MiniMaxM3ProtocolTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.MiniMaxM3.chat_reference.json")!;
        using var doc = JsonDocument.Parse(stream);
        var cases = new Dictionary<string, RenderCase>(StringComparer.Ordinal);
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            JsonElement tools = c.GetProperty("tools");
            JsonElement effort = c.GetProperty("reasoning_effort");
            var rendered = new RenderCase(
                c.GetProperty("name").GetString()!,
                c.GetProperty("messages").EnumerateArray().Select(ToMessage).ToList(),
                tools.ValueKind == JsonValueKind.Null ? null : ToolFunction.ParseList(tools.GetRawText()),
                c.GetProperty("add_generation_prompt").GetBoolean(),
                effort.ValueKind == JsonValueKind.Null ? null : effort.GetString(),
                c.GetProperty("expected").GetString()!);
            cases.Add(rendered.Name, rendered);
        }
        return cases;
    }

    /// <summary>An HF chat message as the OpenAI request parser would hand it over:
    /// content parts joined, images as attachments (rendered before the text), reasoning
    /// in Thinking and tool arguments as JsonElement values.</summary>
    private static ChatMessage ToMessage(JsonElement m)
    {
        var message = new ChatMessage { Role = m.GetProperty("role").GetString()! };
        JsonElement content = m.GetProperty("content");
        if (content.ValueKind == JsonValueKind.Null)
            message.Content = null;
        else if (content.ValueKind == JsonValueKind.String)
            message.Content = content.GetString();
        else
        {
            var text = new StringBuilder();
            foreach (JsonElement part in content.EnumerateArray())
            {
                string type = part.GetProperty("type").GetString()!;
                if (type == "image")
                {
                    Assert.True(text.Length == 0, "TensorSharp renders a message's images before its text.");
                    (message.ImagePaths ??= new List<string>()).Add($"image{message.ImagePaths.Count}.png");
                }
                else
                    text.Append(part.GetProperty("text").GetString());
            }
            message.Content = text.ToString();
        }
        if (m.TryGetProperty("reasoning_content", out JsonElement reasoning) && reasoning.ValueKind == JsonValueKind.String)
            message.Thinking = reasoning.GetString();
        if (m.TryGetProperty("tool_calls", out JsonElement calls))
        {
            message.ToolCalls = calls.EnumerateArray().Select((call, i) =>
            {
                JsonElement function = call.GetProperty("function");
                return new ToolCall
                {
                    Name = function.GetProperty("name").GetString()!,
                    Arguments = JsonSerializer.Deserialize<Dictionary<string, object?>>(
                        function.GetProperty("arguments").GetRawText())!,
                    Index = i,
                };
            }).ToList();
        }
        return message;
    }

    public static IEnumerable<object[]> CaseNames() => LoadCases().Keys.Select(name => new object[] { name });

    [Fact]
    public void Fixture_CoversTheTemplateFeatures()
        => Assert.True(Cases.Value.Count >= 30, $"only {Cases.Value.Count} reference renders");

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Render_MatchesTheHfTemplateByteForByte(string name)
    {
        RenderCase c = Cases.Value[name];
        string actual = ChatTemplate.RenderFromGgufTemplate("{{ messages }}", c.Messages, c.AddGenerationPrompt,
            Arch, c.Tools, enableThinking: false, reasoningEffort: c.ReasoningEffort);
        if (actual != c.Expected)
        {
            int at = 0;
            while (at < actual.Length && at < c.Expected.Length && actual[at] == c.Expected[at])
                at++;
            Assert.Fail($"{name}: first difference at {at}\nexpected: {JsonSerializer.Serialize(Window(c.Expected, at))}\n" +
                        $"actual:   {JsonSerializer.Serialize(Window(actual, at))}");
        }
    }

    private static string Window(string text, int at)
        => text.Substring(Math.Max(0, at - 60), Math.Min(text.Length, at + 60) - Math.Max(0, at - 60));

    [Fact]
    public void ThinkingModes_OnlyChangeTheGenerationPrompt()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Hi" } };
        string adaptive = ChatTemplate.RenderMiniMaxM3(messages);
        Assert.EndsWith("]~b]ai\n", adaptive);
        Assert.Equal(adaptive + "<mm:think>", ChatTemplate.RenderMiniMaxM3(messages, thinking: MiniMaxM3ThinkingMode.Enabled));
        Assert.Equal(adaptive + "</mm:think>", ChatTemplate.RenderMiniMaxM3(messages, thinking: MiniMaxM3ThinkingMode.Disabled));
        Assert.Equal(adaptive, ChatTemplate.RenderMiniMaxM3(messages, addGenerationPrompt: false,
            thinking: MiniMaxM3ThinkingMode.Enabled) + "]~b]ai\n");

        // The protocol: think=false is the template as shipped, think=true opens the block.
        Assert.Equal(adaptive, ChatTemplate.RenderFromGgufTemplate("", messages, true, Arch, null, enableThinking: false));
        Assert.Equal(adaptive + "<mm:think>",
            ChatTemplate.RenderFromGgufTemplate("", messages, true, "minimax_m3", null, enableThinking: true));
    }

    [Fact]
    public void ReasoningEffort_IsTheEffortTagOfTheSystemBlock()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Hi" } };
        string high = ChatTemplate.RenderFromGgufTemplate("", messages, true, Arch, null, false, ReasoningEffort.High);
        Assert.Contains("<model_id>8bb8149fe</model_id>\n\n<effort>high</effort>[e~[\n]~b]developer\n", high);
        Assert.DoesNotContain("<effort>", ChatTemplate.RenderFromGgufTemplate("", messages, true, Arch, null, false, null));
    }

    [Fact]
    public void Registry_DeclaresTheWholeProtocol()
    {
        ChatProtocol protocol = ChatProtocolRegistry.For(Arch)!;
        Assert.Same(protocol, ChatProtocolRegistry.For("minimax_m3"));
        Assert.Equal("minimax-m3", protocol.Id);
        Assert.True(protocol.PreferOwnRenderer!(new ChatRenderRequest([], true, Arch, null, false)));
        Assert.IsType<MiniMaxM3OutputParser>(OutputParserFactory.Create(Arch));
        Assert.True(OutputParserFactory.IsAlwaysRequired(Arch));
        Assert.Equal("</mm:think>", OutputParserFactory.GrammarActivationTrigger(Arch, enableThinking: false));
        Assert.Equal("</mm:think>", OutputParserFactory.GrammarActivationTrigger(Arch, enableThinking: true));
        Assert.Equal("<mm:think>", KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, true));
        Assert.Equal("", KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, false));
        Assert.True(protocol.RendersReasoningEffort && protocol.SharesTemplatePreamble && protocol.RendersAssistantReasoning);
        Assert.True(protocol.RendersToolDeclarations && protocol.RendersToolResultMessages && protocol.CapsVideoFrames);
        Assert.Equal(ToolCallRawSplicing.Always, protocol.ToolCallRawSplicing);
        // No opener: declaring `<mm:think>` would cap the model's own adaptive reasoning at the
        // unrequested-channel budget (~64 tokens), which HF and llama.cpp never truncate.
        Assert.Equal("</mm:think>", protocol.ThinkingBudgetEndToken);
        Assert.Null(protocol.ThinkingBudgetOpenToken);
    }

    [Fact]
    public void MediaPlaceholders_OneImageTokenPerImageAndTimedFrames()
    {
        var image = new ChatMessage { Role = "user", Content = "What is this?", ImagePaths = ["a.png", "b.png"] };
        Assert.Contains("]~b]user\n]<]image[>[]<]image[>[What is this?[e~[\n", ChatTemplate.RenderMiniMaxM3([image]));

        var video = new ChatMessage
        {
            Role = "user", Content = "Summarize.", IsVideo = true,
            ImagePaths = ["f0.png", "f1.png"], ImageTimestamps = [0.0, 2.5],
        };
        Assert.Contains("]~b]user\n]<]0.0 seconds[>[]<]image[>[]<]2.5 seconds[>[]<]image[>[Summarize.[e~[\n",
            ChatTemplate.RenderMiniMaxM3([video]));
    }

    [Fact]
    public void UnknownRole_IsRefusedRatherThanDropped()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "a" },
            new() { Role = "root", Content = "late root" },
        };
        var error = Assert.Throws<ArgumentException>(() => ChatTemplate.RenderMiniMaxM3(messages));
        Assert.Contains("'root' at position 1", error.Message);
    }

    [Fact]
    public void ToolResult_WithoutStructuredCalls_StillRenders()
    {
        // Raw-token splicing clears the round's ToolCalls; the template would raise here.
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "w" },
            new() { Role = "assistant", Content = "x" },
            new() { Role = "tool", Content = "r1" },
            new() { Role = "tool", Content = "r2" },
        };
        Assert.EndsWith("]~b]ai\n</mm:think>x[e~[\n]~b]tool\n<response>r1</response>\n<response>r2</response>[e~[\n]~b]ai\n",
            ChatTemplate.RenderMiniMaxM3(messages));
    }

    [Fact]
    public void ToolArguments_RenderPythonValuesFromDotNetTypes()
    {
        var call = new ToolCall
        {
            Name = "f",
            Arguments = new Dictionary<string, object?>
            {
                ["s"] = "a <b>", ["i"] = 3L, ["d"] = 2.0, ["e"] = 1e-5, ["b"] = false, ["n"] = null,
                ["l"] = new List<object?> { 1L, null, "x" }, ["m"] = new Dictionary<string, object?> { ["k"] = true, ["z"] = null },
                ["arr"] = new[] { "p", "q" },
            },
        };
        string rendered = ChatTemplate.RenderMiniMaxM3(
            [new() { Role = "user", Content = "u" }, new() { Role = "assistant", Content = "", ToolCalls = [call] }],
            addGenerationPrompt: false);
        Assert.Contains($"</mm:think>{Ns}<tool_call>\n{Ns}<invoke name=\"f\">{Ns}<s>a <b>{Ns}</s>{Ns}<i>3{Ns}</i>" +
            $"{Ns}<d>2.0{Ns}</d>{Ns}<e>1e-05{Ns}</e>{Ns}<b>false{Ns}</b>{Ns}<l>{Ns}<item>1{Ns}</item>{Ns}<item>{Ns}</item>" +
            $"{Ns}<item>x{Ns}</item>{Ns}</l>{Ns}<m>{Ns}<k>true{Ns}</k>{Ns}</m>{Ns}<arr>{Ns}<item>p{Ns}</item>{Ns}<item>q{Ns}</item>" +
            $"{Ns}</arr>{Ns}</invoke>\n{Ns}</tool_call>[e~[\n", rendered);
    }

    [Theory]
    [InlineData(0.1, "0.1")]
    [InlineData(2.0, "2.0")]
    [InlineData(-3.5, "-3.5")]
    [InlineData(1e16, "1e+16")]
    [InlineData(1e15, "1000000000000000.0")]
    [InlineData(1e-5, "1e-05")]
    [InlineData(0.0001, "0.0001")]
    [InlineData(123456789.125, "123456789.125")]
    [InlineData(1.5e300, "1.5e+300")]
    [InlineData(-0.0, "-0.0")]
    [InlineData(0.30000000000000004, "0.30000000000000004")]
    [InlineData(5e-324, "5e-324")]
    public void PythonFloatRepr_MatchesPython(double value, string expected)
        => Assert.Equal(expected, ChatTemplate.PythonFloatRepr(value, json: false));

    // ---- KV reuse ---------------------------------------------------------------------

    /// <summary>One token per character, so a common prefix is a character count.</summary>
    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => [];
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(i => (char)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
            => buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }

    public static IEnumerable<object[]> ReplyShapes() =>
    [
        // think, generated text (after the generation prompt), parsed content, parsed reasoning
        [false, "</mm:think>Hello there.", "Hello there.", ""],
        [false, "<mm:think>The user greets me.</mm:think>Hello there.", "Hello there.", "The user greets me."],
        [true, "The user greets me.</mm:think>Hello there.", "Hello there.", "The user greets me."],
        [false, $"<mm:think>Look it up.</mm:think>{Ns}<tool_call>\n{Ns}<invoke name=\"get_weather\">{Ns}<city>Paris{Ns}</city>{Ns}</invoke>\n{Ns}</tool_call>",
            "", "Look it up."],
    ];

    /// <summary>
    /// Turn N+1's prompt must extend turn N's prompt plus what the model generated, through
    /// both history paths: the generated tokens spliced back (what the hosts record), and the
    /// parsed content + reasoning re-rendered by the template (what an API client sends back).
    /// </summary>
    [Theory]
    [MemberData(nameof(ReplyShapes))]
    public void NextTurn_ExtendsThePreviousPromptAndReply(bool think, string generated, string content, string reasoning)
    {
        var tokenizer = new CharacterTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var tools = ToolFunction.ParseList("""
            [{"type":"function","function":{"name":"get_weather","description":"Weather.",
              "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
            """);
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "Be brief." },
            new() { Role = "user", Content = "Hi, weather in Paris?" },
        };
        List<int> prompt = renderer.RenderToTokens(tokenizer, "", history, Arch, true, out _,
            out string boundary, tools, think);
        string suffix = KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, think);
        Assert.Equal(think ? "<mm:think>" : "\n", think ? suffix : boundary);

        var parser = new MiniMaxM3OutputParser();
        parser.Init(think, tools);
        parser.SetGenerationPromptSuffix(suffix);
        ParsedOutput parsed = parser.Add(generated, done: true);
        Assert.Equal((content, reasoning), (parsed.Content, parsed.Thinking));

        List<int> cached = prompt.Concat(tokenizer.Encode(generated)).ToList();
        foreach (bool raw in new[] { true, false })
        {
            var assistant = new ChatMessage
            {
                Role = "assistant", Content = parsed.Content, Thinking = parsed.Thinking, ToolCalls = parsed.ToolCalls,
            };
            if (raw)
            {
                assistant.RawOutputTokens = tokenizer.Encode(generated);
                assistant.RawPromptTrailingWhitespace = boundary;
                assistant.RawGenerationSuffix = suffix;
            }
            var next = new List<ChatMessage>(history) { assistant };
            if (parsed.ToolCalls != null)
                next.Add(new ChatMessage { Role = "tool", Content = "18C" });
            else
                next.Add(new ChatMessage { Role = "user", Content = "Thanks!" });
            List<int> follow = renderer.RenderToTokens(tokenizer, "", next, Arch, true, tools, think);

            // A thinking-on turn re-rendered from parsed text cannot know the prompt opened the
            // block; its reasoning renders as <mm:think>R</mm:think> exactly as the cache holds it.
            int common = 0;
            while (common < follow.Count && common < cached.Count && follow[common] == cached[common])
                common++;
            Assert.True(common == cached.Count,
                $"raw={raw}: diverged at {common}/{cached.Count}: cache {JsonSerializer.Serialize(tokenizer.Decode(cached.Skip(common).Take(40).ToList()))}" +
                $" vs render {JsonSerializer.Serialize(tokenizer.Decode(follow.Skip(common).Take(40).ToList()))}");
        }
    }

    [Fact]
    public void SharedPreamble_IsThePrefixOfEveryConversation()
    {
        var tokenizer = new CharacterTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<int> preamble = renderer.RenderToTokens(tokenizer, "", [], Arch, addGenerationPrompt: false);
        List<int> chat = renderer.RenderToTokens(tokenizer, "", [new() { Role = "user", Content = "x" }], Arch, true);
        Assert.Equal(preamble, chat.Take(preamble.Count));
        Assert.EndsWith("]~b]developer\nYou are a helpful assistant.[e~[\n", tokenizer.Decode(preamble));
    }
}
