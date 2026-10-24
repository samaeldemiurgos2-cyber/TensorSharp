// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// MiniMax-M3.1's DSpark drafter through the managed speculative stack (the
// registry's block speculator, SpeculativeExecution's draft/verify/rollback loop,
// the model's ISpeculativeModel surface) over the real native executor, on
// eng/minimax-m3-fixture.py's small fixture and its drafter (dspark.gguf beside
// model.gguf). The fixture's target follows a successor chain the drafter's
// Markov code predicts, so on a prompt that ends in 100000 the drafts are mostly
// accepted and greedy tokens sit far from any near-tie: speculative greedy must
// equal plain greedy token for token.
// Requires TS_TEST_MINIMAX_M3_FIXTURE (the fixture GGUF) and TS_TEST_GGML_BACKEND.
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MiniMaxM3DsparkFactAttribute : FactAttribute
{
    private BackendType _ggmlBackend;

    public MiniMaxM3DsparkFactAttribute()
    {
        Skip = TestGates.ModelSkip(MiniMaxM3ModelNativeExecutorTests.FixtureVariable);
        if (Skip == null && !File.Exists(MiniMaxM3DsparkExecutorTests.DrafterPath))
            Skip = $"Requires the fixture's DSpark drafter ({MiniMaxM3DsparkExecutorTests.DrafterPath} not found).";
    }

    /// <summary>The GGML backend the test constructs; skips unless the process pins it.</summary>
    public BackendType GgmlBackend
    {
        get => _ggmlBackend;
        set { _ggmlBackend = value; Skip ??= TestGates.GgmlPinSkip(value); }
    }
}

[Collection(MiniMaxM3EnvironmentCollection.Name)]
[Trait("Requires", "Models")]
public sealed class MiniMaxM3DsparkExecutorTests : IDisposable
{
    private const int ChainFirst = 100000, ChainLength = 4096;

    internal static string DrafterPath => Path.Combine(
        Path.GetDirectoryName(Environment.GetEnvironmentVariable(MiniMaxM3ModelNativeExecutorTests.FixtureVariable) ?? "") ?? "",
        "dspark.gguf");

    private static readonly string[] Saved = ["MAX_CONTEXT", "TS_MINIMAX_M3_UBATCH", MiniMaxM3Architecture.MsaModeVariable];
    private readonly Dictionary<string, string> _old = Saved.ToDictionary(n => n, Environment.GetEnvironmentVariable);
    private readonly ITestOutputHelper _output;

    public MiniMaxM3DsparkExecutorTests(ITestOutputHelper output)
    {
        _output = output;
        MoeCpuOffloadConfig.Reset();
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "2048");
        Environment.SetEnvironmentVariable("TS_MINIMAX_M3_UBATCH", "256");
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, null);
    }

    public void Dispose()
    {
        foreach (var (name, value) in _old) Environment.SetEnvironmentVariable(name, value);
        MoeCpuOffloadConfig.Reset();
    }

    /// <summary>Random ordinary-text ids (none from the chain), then the chain's first id.</summary>
    private static int[] ChainPrompt(int count, int seed)
    {
        var tokens = new List<int>();
        uint state = (uint)seed * 2654435761u + 7;
        while (tokens.Count < count)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            int t = (int)(state % 190000) + 1000;
            if (t < ChainFirst || t >= ChainFirst + ChainLength) tokens.Add(t);
        }
        tokens.Add(ChainFirst);
        return tokens.ToArray();
    }

    private static int Argmax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    [MiniMaxM3DsparkFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaSpeculativeGreedyEqualsPlainGreedy() => SpeculativeGreedyEqualsPlainGreedy(BackendType.GgmlCuda, 160);

    [MiniMaxM3DsparkFact(GgmlBackend = BackendType.GgmlCpu)]
    public void CpuSpeculativeGreedyEqualsPlainGreedy() => SpeculativeGreedyEqualsPlainGreedy(BackendType.GgmlCpu, 48);

    private void SpeculativeGreedyEqualsPlainGreedy(BackendType backend, int newTokens)
    {
        using var model = new MiniMaxM3Model(Environment.GetEnvironmentVariable(MiniMaxM3ModelNativeExecutorTests.FixtureVariable),
            backend, 1, null, 1, DrafterPath);
        Assert.True(model.HasDraftHead);
        Assert.Equal(DraftHeadKind.Block, model.DraftHeadKind);
        Assert.Equal(7, model.DraftBlockSize);
        Assert.True(model.DraftSelfCatchUp);
        Assert.Equal(1, ((ISpeculativeTarget)model).SpecFeatureSize);

        // The registry picks the block speculator for the checkpoint's drafter.
        var options = new SpeculationOptions { Enabled = true, MaxDraftTokens = 7, MinDraftProb = 0f };
        ISpeculator speculator = SpeculatorRegistry.Create(model, options, out string decline);
        Assert.True(speculator != null, decline);
        Assert.IsType<BlockDraftSpeculator>(speculator);
        var decoder = new SpeculativeDecoder(model, speculator) { AdaptiveSpeculation = false, MinDraftProb = 0f };

        foreach (int seed in new[] { 3, 5 })
        {
            int[] prompt = ChainPrompt(seed == 3 ? 100 : 400, seed);
            List<int> spec = decoder.GenerateGreedy(prompt, newTokens);

            model.ResetKVCache();
            var plain = new List<int>();
            float[] logits = model.Forward(prompt);
            for (int i = 0; i < newTokens; i++)
            {
                int tok = Argmax(logits);
                plain.Add(tok);
                if (i + 1 < newTokens) logits = model.Forward(new[] { tok });
            }
            _output.WriteLine($"{backend} prompt {prompt.Length}: drafted {decoder.TokensDrafted}, accepted " +
                $"{decoder.TokensAccepted} ({100 * decoder.AcceptanceRate:F1} %), {decoder.VerifySteps} verify steps, " +
                $"{decoder.PlainSteps} plain steps");
            Assert.Equal(plain, spec);
            Assert.True(decoder.AcceptanceRate > 0.5, $"acceptance {decoder.AcceptanceRate:P1}");
        }
        long[] stats = model.DsparkStatsForTests();
        Assert.NotNull(stats);
        Assert.True(stats[2] > 0, "no native drafts ran");
    }
}
