// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="MiniMaxM3ToolGrammar"/> (tool_choice required / named / auto) over a vocabulary
/// of single characters plus the MiniMax markers as control tokens, so every byte of a reply
/// is a sampling decision: what the grammar admits, what it masks, and that the replies it
/// admits parse back with <see cref="MiniMaxM3OutputParser"/>.
/// </summary>
public class MiniMaxM3ProtocolToolGrammarTests
{
    private const string Ns = "]<]minimax[>[";
    private const string Open = Ns + "<tool_call>\n";
    private const string Close = Ns + "</tool_call>";
    private static string Invoke(string name, string body) => $"{Ns}<invoke name=\"{name}\">{body}{Ns}</invoke>\n";
    private static string Arg(string name, string value) => $"{Ns}<{name}>{value}{Ns}</{name}>";

    private static readonly List<ToolFunction> Tools = ToolFunction.ParseList("""
        [
          {"name": "magic_int", "parameters": {"type": "object",
            "properties": {"ref": {"type": "integer"}, "name": {"type": "string"}}, "required": ["ref"]}},
          {"name": "get_time", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}}, "required": ["city"]}},
          {"name": "empty_args", "parameters": {"type": "object", "properties": {}}},
          {"name": "nested_args", "parameters": {"type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}},
              "entries": {"type": "array", "items": {"type": "object", "properties": {"id": {"type": "integer"},
                "label": {"type": "string"}}, "required": ["id", "label"]}}}, "required": ["tags", "entries"]}},
          {"name": "union_args", "parameters": {"type": "object", "properties": {
            "filter": {"anyOf": [{"type": "array", "items": {"type": "string"}},
              {"type": "object", "properties": {"field": {"type": "string"}, "op": {"type": "string"}}, "required": ["field", "op"]}]},
            "label": {"oneOf": [{"type": "string"}, {"type": "object", "properties": {"text": {"type": "string"}}}]},
            "limit": {"oneOf": [{"type": "integer"}, {"type": "object", "properties": {"max": {"type": "integer"}}, "required": ["max"]}]},
            "mode": {"enum": ["fast", "slow"]}, "maybe": {"type": ["integer", "null"]}, "flag": {"type": "boolean"}}}}
        ]
        """);

    /// <summary>EOS, the five markers as control tokens, then one token per character.</summary>
    private sealed class CharTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        private static readonly string[] Markers = { Ns, "<tool_call>", "</tool_call>", "<mm:think>", "</mm:think>" };
        public CharTokenizer()
        {
            var vocab = new List<string> { "[e~[" };
            vocab.AddRange(Markers);
            vocab.Add("\n");
            for (char c = ' '; c <= '~'; c++) vocab.Add(c.ToString());
            vocab.Add("ü");
            Vocab = vocab.ToArray();
        }
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => [0];
        public int VocabSize => Vocab.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => [0, 1, 2, 3, 4, 5];
        public List<int> Encode(string text, bool addSpecial = true)
        {
            var ids = new List<int>();
            for (int at = 0; at < text.Length;)
            {
                int marker = Array.FindIndex(Markers, m => string.CompareOrdinal(text, at, m, 0, m.Length) == 0);
                string piece = marker >= 0 ? Markers[marker] : text[at].ToString();
                ids.Add(LookupToken(piece));
                at += piece.Length;
            }
            return ids;
        }
        public string Decode(List<int> ids) => string.Concat(ids.Select(id => Vocab[id]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    private static readonly CharTokenizer Tokens = new();

    private static bool Allowed(GrammarConstraint constraint, int id)
        => (constraint.CurrentMask()[id >> 6] & (1UL << (id & 63))) != 0;

    /// <summary>Feed <paramref name="reply"/>; return the index of the first masked token, or -1.</summary>
    private static int FirstRejected(GrammarConstraint constraint, string reply)
    {
        List<int> ids = Tokens.Encode(reply);
        for (int i = 0; i < ids.Count; i++)
        {
            if (!Allowed(constraint, ids[i]))
                return i;
            constraint.Accept(ids[i]);
        }
        return -1;
    }

    private static void AssertAccepted(MiniMaxM3ToolGrammar plan, string reply, bool thinking = false)
    {
        GrammarConstraint constraint = plan.NewConstraint(Tokens, thinking);
        int rejected = FirstRejected(constraint, reply);
        Assert.True(rejected < 0, $"masked at token {rejected}: {JsonSerializer.Serialize(Tokens.Decode(Tokens.Encode(reply).Take(rejected + 1).ToList()))}");
        Assert.True(constraint.IsComplete && !constraint.IsDead, "the reply does not complete the grammar");
    }

    private static void AssertRejected(MiniMaxM3ToolGrammar plan, string reply, string acceptedPrefix)
    {
        GrammarConstraint constraint = plan.NewConstraint(Tokens, false);
        int rejected = FirstRejected(constraint, reply);
        Assert.True(rejected >= 0, $"{JsonSerializer.Serialize(reply)} was accepted");
        Assert.Equal(acceptedPrefix, Tokens.Decode(Tokens.Encode(reply).Take(rejected).ToList()));
    }

    public static IEnumerable<object[]> LlamaReplies() =>
    [
        [Invoke("magic_int", Arg("ref", "42") + Arg("name", "foo bar"))],
        [Invoke("magic_int", Arg("ref", "-14"))],
        [Invoke("get_time", Arg("city", "a <b> & \"c\"\nZürich"))],
        [Invoke("empty_args", "")],
        [Invoke("nested_args", Arg("tags", Arg("item", "alpha") + Arg("item", "beta")) +
            Arg("entries", Arg("item", Arg("id", "1") + Arg("label", "one"))))],
        [Invoke("union_args", Arg("filter", Arg("item", "alpha") + Arg("item", "beta")))],
        [Invoke("union_args", Arg("filter", Arg("field", "f") + Arg("op", "eq")) + Arg("limit", "5"))],
        [Invoke("union_args", Arg("limit", Arg("max", "10")) + Arg("label", Arg("text", "hi")))],
        [Invoke("union_args", Arg("mode", "slow") + Arg("maybe", "null") + Arg("flag", "true") + Arg("maybe", "3"))],
        [Invoke("get_time", Arg("city", "Paris")) + Invoke("magic_int", Arg("ref", "1"))],
    ];

    [Theory]
    [MemberData(nameof(LlamaReplies))]
    public void Required_AdmitsWellFormedCallsAndTheyParseBack(string calls)
    {
        var plan = MiniMaxM3ToolGrammar.Compile(Tools, MiniMaxM3ToolChoice.Required);
        _ = Grammar.Parse(plan.Source);
        foreach (string prefix in new[] { "</mm:think>", "<mm:think>Think it through.</mm:think>\n" })
        {
            string reply = prefix + Open + calls + Close;
            AssertAccepted(plan, reply);
            var parser = new MiniMaxM3OutputParser();
            parser.Init(false, Tools);
            ParsedOutput parsed = parser.Add(reply, true);
            // The line break the grammar lets through before the block is the answer's own.
            Assert.Equal(prefix.EndsWith('\n') ? "\n" : "", parsed.Content);
            Assert.Equal(calls.Split(Ns + "</invoke>").Length - 1, parsed.ToolCalls!.Count);
        }
        // Thinking on: the prompt opened the block, the reply closes it.
        AssertAccepted(plan, "Reasoning.</mm:think>" + Open + calls + Close, thinking: true);
    }

    [Fact]
    public void Required_MasksWhatIsNotAWellFormedCall()
    {
        var plan = MiniMaxM3ToolGrammar.Compile(Tools, MiniMaxM3ToolChoice.Required);
        // Prose instead of the block.
        AssertRejected(plan, "</mm:think>Sure, " + Open, "</mm:think>");
        // An undeclared function.
        AssertRejected(plan, "</mm:think>" + Open + Invoke("rm_rf", ""), "</mm:think>" + Open + Ns + "<invoke name=\"");
        // Text where an integer goes.
        AssertRejected(plan, "</mm:think>" + Open + Invoke("magic_int", Arg("ref", "x")),
            "</mm:think>" + Open + Ns + "<invoke name=\"magic_int\">" + Ns + "<ref>");
        // A required parameter left out.
        AssertRejected(plan, "</mm:think>" + Open + Invoke("magic_int", Arg("name", "n")),
            "</mm:think>" + Open + Ns + "<invoke name=\"magic_int\">" + Ns + "<");
        // A string enum outside its values.
        AssertRejected(plan, "</mm:think>" + Open + Invoke("union_args", Arg("mode", "medium")),
            "</mm:think>" + Open + Ns + "<invoke name=\"union_args\">" + Ns + "<mode>");
        // A raw string ends at its own close tag; text after it is not a value.
        AssertRejected(plan, "</mm:think>" + Open + Invoke("get_time", Arg("city", "Rome") + "x"),
            "</mm:think>" + Open + Ns + "<invoke name=\"get_time\">" + Ns + "<city>Rome" + Ns + "</city>");
        // Nothing after the block but the end of the turn.
        GrammarConstraint constraint = plan.NewConstraint(Tokens, false);
        Assert.Equal(-1, FirstRejected(constraint, "</mm:think>" + Open + Invoke("empty_args", "") + Close));
        Assert.False(Allowed(constraint, Tokens.LookupToken("x")));
        Assert.True(constraint.IsComplete);
    }

    [Fact]
    public void NamedAndSingleCall_RestrictTheBlock()
    {
        var named = MiniMaxM3ToolGrammar.Compile(Tools, MiniMaxM3ToolChoice.Named, "get_time");
        AssertAccepted(named, "</mm:think>" + Open + Invoke("get_time", Arg("city", "Rome")) + Close);
        AssertRejected(named, "</mm:think>" + Open + Invoke("magic_int", Arg("ref", "1")),
            "</mm:think>" + Open + Ns + "<invoke name=\"");

        var single = MiniMaxM3ToolGrammar.Compile(Tools, MiniMaxM3ToolChoice.Required, parallelToolCalls: false);
        string first = "</mm:think>" + Open + Invoke("get_time", Arg("city", "Rome"));
        AssertAccepted(single, first + Close);
        AssertRejected(single, first + Invoke("get_time", Arg("city", "Oslo")), first + Ns + "<");

        Assert.Throws<NotSupportedException>(() => MiniMaxM3ToolGrammar.Compile(Tools, MiniMaxM3ToolChoice.Named, "missing"));
        Assert.Throws<NotSupportedException>(() => MiniMaxM3ToolGrammar.Compile([], MiniMaxM3ToolChoice.Required));
    }

    [Fact]
    public void Auto_LeavesTheAnswerFreeUntilTheModelOpensABlock()
    {
        var plan = MiniMaxM3ToolGrammar.Compile(Tools, MiniMaxM3ToolChoice.Auto);
        AssertAccepted(plan, "<mm:think>Hmm.</mm:think>No tool needed, the answer is 4.");
        AssertAccepted(plan, "</mm:think>Checking. " + Open + Invoke("get_time", Arg("city", "Rome")) + Close);
        // A block quoted in the reasoning does not arm it; one opened in the answer does.
        AssertAccepted(plan, "<mm:think>I could write " + Open + " here.</mm:think>Fine.");
        AssertRejected(plan, "</mm:think>" + Open + Invoke("magic_int", Arg("ref", "x")),
            "</mm:think>" + Open + Ns + "<invoke name=\"magic_int\">" + Ns + "<ref>");
    }

    /// <summary>The same through the real 200k vocabulary, where the markers are
    /// USER_DEFINED tokens and the rest is byte-level BPE (TS_TEST_MINIMAX_M3_GGUF: the first
    /// GGUF shard, of which only the tokenizer metadata is read).</summary>
    [ModelFact("TS_TEST_MINIMAX_M3_GGUF")]
    public void Required_AdmitsTheRealTokenizationOfWellFormedCalls()
    {
        ITokenizer tokenizer;
        using (var gguf = GgufFile.OpenWithoutSiblingShards(Environment.GetEnvironmentVariable("TS_TEST_MINIMAX_M3_GGUF")!))
            tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        var plan = MiniMaxM3ToolGrammar.Compile(Tools, MiniMaxM3ToolChoice.Required);
        foreach (object[] calls in LlamaReplies().Take(5))
        {
            string reply = "<mm:think>Think it through.</mm:think>" + Open + (string)calls[0] + Close;
            GrammarConstraint constraint = plan.NewConstraint(tokenizer, false);
            foreach (int id in tokenizer.Encode(reply, addSpecial: false))
            {
                Assert.True(Allowed(constraint, id), $"token {id} {JsonSerializer.Serialize(tokenizer.Decode([id]))} masked in {JsonSerializer.Serialize(reply)}");
                constraint.Accept(id);
            }
            Assert.True(constraint.IsComplete && !constraint.IsDead);
        }
        GrammarConstraint prose = plan.NewConstraint(tokenizer, false);
        prose.Accept(200060);
        Assert.False(Allowed(prose, tokenizer.Encode("Sure", false)[0]));
        Assert.True(Allowed(prose, 200058));
    }

    [Fact]
    public void Adapter_CompilesOnlyRequiredOrNamedChoices()
    {
        using var auto = JsonDocument.Parse("""{"tool_choice": "auto"}""");
        Assert.Null(TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareMiniMaxM3ToolGrammar(auto.RootElement, Tools, null));
        using var required = JsonDocument.Parse("""{"tool_choice": "required", "parallel_tool_calls": false}""");
        Assert.Equal(MiniMaxM3ToolChoice.Required,
            TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareMiniMaxM3ToolGrammar(required.RootElement, Tools, null)!.Choice);
        using var named = JsonDocument.Parse("""{"tool_choice": {"type": "function", "function": {"name": "get_time"}}}""");
        Assert.Equal(MiniMaxM3ToolChoice.Named,
            TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareMiniMaxM3ToolGrammar(named.RootElement, Tools, null)!.Choice);
        Assert.Throws<NotSupportedException>(() => TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareMiniMaxM3ToolGrammar(
            required.RootElement, Tools, StructuredOutputFormat.JsonObject()));
    }
}
