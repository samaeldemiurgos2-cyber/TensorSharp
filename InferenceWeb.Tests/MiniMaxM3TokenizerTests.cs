// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-M3.1's tokenizer (tokenizer.ggml.pre "minimax-m2", NFC normalizer, USER_DEFINED
/// protocol markers) against the HF tokenizer. Fixtures/MiniMaxM3/tokenizer_reference.json was
/// generated from MiniMaxAI/MiniMax-M3.1-Flash's tokenizer.json with HF tokenizers: <c>ids</c>
/// are <c>encode(add_special_tokens=False)</c> over code, digits, CJK, Hangul jamo, decomposed
/// accents and other text NFC changes, emoji, every added token and full chat renders;
/// <c>pieces</c> are the pre-tokenizer's split of text without added tokens. Regenerate with
/// eng/tests/minimax-m3-protocol-fixtures.py. The id test needs the GGUF's tokenizer
/// metadata: point TS_TEST_MINIMAX_M3_GGUF at the first shard of MiniMax-M3.1-Flash (only its
/// header is read; a header-only copy of it is enough).
/// </summary>
public class MiniMaxM3TokenizerTests
{
    private sealed record Case(string Text, int[] Ids, string Decoded, string[]? Pieces, bool NfcChanges);

    private static readonly Lazy<IReadOnlyList<Case>> Cases = new(() =>
    {
        using var stream = typeof(MiniMaxM3TokenizerTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.MiniMaxM3.tokenizer_reference.json")!;
        using var doc = JsonDocument.Parse(stream);
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(
            c.GetProperty("text").GetString()!,
            c.GetProperty("ids").EnumerateArray().Select(p => p.GetInt32()).ToArray(),
            c.GetProperty("decoded").GetString()!,
            c.TryGetProperty("pieces", out JsonElement pieces)
                ? pieces.EnumerateArray().Select(p => p.GetString()!).ToArray() : null,
            c.GetProperty("nfc_changes").GetBoolean())).ToList();
    });

    [Fact]
    public void PreTokenizer_MatchesTheHfSplitOnEveryCase()
    {
        Assert.Equal(BpeTokenizer.ResolvePreTokenizerPattern("gpt-4o"), BpeTokenizer.ResolvePreTokenizerPattern("minimax-m2"));
        var tokenizer = new BpeTokenizer([], [], [], -1, [], false, false, "minimax-m2");
        var cases = Cases.Value.Where(c => c.Pieces != null).ToList();
        Assert.True(cases.Count > 200, $"only {cases.Count} split cases");
        var mismatches = cases
            .Where(c => !c.Pieces!.SequenceEqual(tokenizer.SplitForBpe(c.Text)))
            .Select(c => $"{JsonSerializer.Serialize(c.Text)}: expected {JsonSerializer.Serialize(c.Pieces)}, " +
                         $"got {JsonSerializer.Serialize(tokenizer.SplitForBpe(c.Text))}")
            .ToList();
        Assert.True(mismatches.Count == 0, string.Join("\n", mismatches));
    }

    /// <summary>A byte-level vocabulary with no merges: one token per UTF-8 byte, plus one
    /// USER_DEFINED marker. Enough to see what NFC does to the bytes.</summary>
    private static BpeTokenizer ByteTokenizer(bool nfc)
    {
        var vocab = new List<string>();
        for (int b = 0; b < 256; b++)
        {
            char c = (char)b;
            if (c == 0x00ad) c = (char)0x0143;
            else if (c <= 0x0020) c = (char)(c + 0x0100);
            else if (c >= 0x007f && c <= 0x00a0) c = (char)(c + 0x00a2);
            vocab.Add(c.ToString());
        }
        vocab.Add("<mm:think>");
        var types = Enumerable.Repeat(1, 256).Append(4).ToArray();
        return new BpeTokenizer(vocab.ToArray(), types, [], -1, [], false, false, "minimax-m2", nfc);
    }

    [Fact]
    public void Nfc_ComposesTextFragmentsBeforePreTokenization()
    {
        var nfc = ByteTokenizer(nfc: true);
        var raw = ByteTokenizer(nfc: false);
        foreach (var (decomposed, composed) in new[]
        {
            ("école", "école"), ("한", "한"), ("Å", "Å"), ("豈", "豈"),
            ("ą́", "ą́"),
        })
        {
            Assert.Equal(nfc.Encode(composed, false), nfc.Encode(decomposed, false));
            Assert.NotEqual(raw.Encode(composed, false), raw.Encode(decomposed, false));
            Assert.Equal(composed, nfc.Decode(nfc.Encode(decomposed, false)));
        }
        // The marker stays one token, and text on both sides of it is normalized.
        List<int> ids = nfc.Encode("é<mm:think>Å", false);
        Assert.Equal(nfc.Encode("é", false).Append(256).Concat(nfc.Encode("Å", false)), ids);
        // ASCII is already NFC and takes the fast path; a lone surrogate is kept as is.
        const string ascii = "plain ASCII text";
        Assert.Same(ascii, BpeTokenizer.NormalizeNfc(ascii));
        Assert.Equal("a\ud800b", BpeTokenizer.NormalizeNfc("a\ud800b"));
    }

    private static ITokenizer GgufTokenizer()
    {
        using var gguf = GgufFile.OpenWithoutSiblingShards(Environment.GetEnvironmentVariable("TS_TEST_MINIMAX_M3_GGUF")!);
        Assert.Equal("minimax-m3", gguf.GetString("general.architecture"));
        Assert.Equal("minimax-m2", gguf.GetString("tokenizer.ggml.pre", null));
        Assert.True(gguf.GetBool("tokenizer.ggml.normalizer.nfc", false));
        return ModelBase.CreateTokenizerFromGguf(gguf);
    }

    [ModelFact("TS_TEST_MINIMAX_M3_GGUF")]
    public void Encode_FromTheGguf_MatchesTheHfIdsAndDecode()
    {
        ITokenizer tokenizer = GgufTokenizer();
        Assert.Contains(Cases.Value, c => c.NfcChanges);
        var mismatches = new List<string>();
        foreach (Case c in Cases.Value)
        {
            List<int> ids = tokenizer.Encode(c.Text, addSpecial: false);
            if (!c.Ids.SequenceEqual(ids))
                mismatches.Add($"{JsonSerializer.Serialize(c.Text)}: expected [{string.Join(",", c.Ids)}], got [{string.Join(",", ids)}]");
            else if (tokenizer.Decode(ids) != c.Decoded)
                mismatches.Add($"{JsonSerializer.Serialize(c.Text)}: decoded {JsonSerializer.Serialize(tokenizer.Decode(ids))}, " +
                               $"expected {JsonSerializer.Serialize(c.Decoded)}");
        }
        Assert.True(mismatches.Count == 0, $"{mismatches.Count}/{Cases.Value.Count} mismatches:\n" + string.Join("\n", mismatches));
    }

    [ModelFact("TS_TEST_MINIMAX_M3_GGUF")]
    public void Gguf_SpecialTokensBosAndEog()
    {
        ITokenizer tokenizer = GgufTokenizer();
        Assert.Equal(200064, tokenizer.VocabSize);
        // The template writes `]~!b[]~b]` itself: no BOS may be prepended.
        Assert.Equal(tokenizer.Encode("]~!b[]~b]system\nhi", addSpecial: false), tokenizer.Encode("]~!b[]~b]system\nhi", addSpecial: true));
        Assert.Equal(new[] { 200034, 200019 }, tokenizer.Encode("]~!b[]~b]", false));
        Assert.Equal(new[] { 200020 }, tokenizer.EosTokenIds);
        Assert.True(tokenizer.IsEos(200020) && !tokenizer.IsEos(200019));
        // USER_DEFINED markers tokenize atomically and survive detokenization for the parser.
        Assert.Equal(new[] { 200058, 200052 }, tokenizer.Encode("]<]minimax[>[<tool_call>", false));
        Assert.Equal(new[] { 200059, 200060 }, tokenizer.Encode("<mm:think></mm:think>", false));
        Assert.Equal("<mm:think>x</mm:think>]<]minimax[>[<tool_call>",
            tokenizer.Decode([200059, .. tokenizer.Encode("x", false), 200060, 200058, 200052]));
        Assert.Equal(new[] { 200025 }, tokenizer.Encode(ChatTemplate.MiniMaxM3ImagePlaceholder, false));
        var special = ((ISpecialTokenVocabulary)tokenizer).SpecialTokenIds;
        Assert.Contains(200060, special);
        Assert.DoesNotContain(200061, special);
    }

    /// <summary>
    /// response_format through the real vocabulary: the JSON grammar armed after
    /// <c>&lt;/mm:think&gt;</c> lets the model write its trained prefix - the bare closer or a
    /// whole reasoning block - and then enforces the object; the parser hands back the JSON.
    /// </summary>
    [ModelFact("TS_TEST_MINIMAX_M3_GGUF")]
    public void JsonMode_ArmsAfterTheThinkCloserWithTheRealVocabulary()
    {
        ITokenizer tokenizer = GgufTokenizer();
        GrammarMaskCache cache = GrammarLibrary.ForJsonObject(tokenizer);
        foreach (string reply in new[] { "</mm:think>{\"a\": 1}", "<mm:think>Think.</mm:think>{\"a\": 1}" })
        {
            GrammarConstraint constraint = GrammarLibrary.NewConstraint(cache, tokenizer);
            constraint.ActivateAfter(OutputParserFactory.GrammarActivationTrigger("minimax-m3")!, skipLeadingWhitespace: true);
            foreach (int id in tokenizer.Encode(reply, addSpecial: false))
            {
                Assert.True(Allowed(constraint, id), $"{reply}: token {id} {JsonSerializer.Serialize(tokenizer.Decode([id]))} masked");
                constraint.Accept(id);
                if (id == 200060)
                {
                    Assert.True(constraint.IsActive);
                    Assert.False(Allowed(constraint, 200060));
                    Assert.False(Allowed(constraint, tokenizer.Encode("Sure", false)[0]));
                }
            }
            Assert.True(constraint.IsActive && constraint.IsComplete && !constraint.IsDead);

            var parser = OutputParserFactory.Create("minimax-m3");
            parser.Init(false, null);
            ParsedOutput parsed = parser.Add(reply, true);
            Assert.Equal("{\"a\": 1}", parsed.Content);
        }
    }

    /// <summary>
    /// think=false is M3.1's adaptive mode (the template as shipped): the model opens
    /// <c>&lt;mm:think&gt;</c> itself and HF and llama.cpp never truncate that reasoning, so the
    /// unrequested-channel cap (~64 tokens) must not arm. think=true opens the block in the
    /// prompt and keeps its normal budget.
    /// </summary>
    [ModelFact("TS_TEST_MINIMAX_M3_GGUF")]
    public void AdaptiveReasoning_IsNotCappedButRequestedThinkingIsBudgeted()
    {
        ITokenizer tokenizer = GgufTokenizer();
        SamplingConfig original = SamplingConfig.Greedy;
        SamplingConfig adaptive = ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, "minimax-m3",
            ChatGenerationPipeline.UnrequestedThinkingBudgetFor(2048), out bool adaptiveInstalled, enableThinking: false);
        Assert.False(adaptiveInstalled);
        Assert.Same(original, adaptive);

        SamplingConfig requested = ChatGenerationPipeline.WithThinkingBudget(original, tokenizer, "minimax-m3",
            ChatGenerationPipeline.ThinkingBudgetFor(2048, true), out bool requestedInstalled, enableThinking: true);
        Assert.True(requestedInstalled);
        Assert.True(requested.ThinkingBudget!.OpenAtStart);
        Assert.Equal(tokenizer.Encode("</mm:think>", addSpecial: false).Single(), requested.ThinkingBudget.EndTokenId);
    }

    private static bool Allowed(GrammarConstraint constraint, int id)
        => (constraint.CurrentMask()[id >> 6] & (1UL << (id & 63))) != 0;
}
