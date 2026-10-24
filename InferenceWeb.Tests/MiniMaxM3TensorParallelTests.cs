// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// MiniMax-M3's --tp N through the managed surface, on eng/minimax-m3-fixture.py's
// small fixture: the native executor on ggml's meta backend (attention and caches
// mirrored, the FFN split along n_ff) against the same executor on one GPU, for
// prompts below position 64 (where the fixture's sparse attention is dense) and past
// it, one-token decodes, a rewind, per-request slots and a batched decode.
// Requires TS_TEST_MINIMAX_M3_FIXTURE, TS_TEST_GGML_BACKEND=cuda and N CUDA GPUs
// (TS_TEST_MINIMAX_M3_TP=N, default 2).
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MiniMaxM3TensorParallelFactAttribute : FactAttribute
{
    public MiniMaxM3TensorParallelFactAttribute()
    {
        Skip = TestGates.ModelSkip(MiniMaxM3ModelNativeExecutorTests.FixtureVariable)
            ?? TestGates.GgmlPinSkip(BackendType.GgmlCuda)
            ?? TestGates.CudaDevicesSkip(MiniMaxM3TensorParallelTests.Degree);
    }
}

[Collection(MiniMaxM3EnvironmentCollection.Name)]
[Trait("Requires", "Models")]
public sealed class MiniMaxM3TensorParallelTests : IDisposable
{
    internal const string DegreeVariable = "TS_TEST_MINIMAX_M3_TP";

    internal static int Degree =>
        int.TryParse(Environment.GetEnvironmentVariable(DegreeVariable), out int n) && n >= 2 ? n : 2;

    private static readonly string[] Saved = ["MAX_CONTEXT", "TS_MINIMAX_M3_UBATCH", MiniMaxM3Architecture.MsaModeVariable];
    private readonly Dictionary<string, string> _old = Saved.ToDictionary(n => n, Environment.GetEnvironmentVariable);
    private readonly ITestOutputHelper _output;

    // An all-reduce sums the FFN's partial products in another order than one GPU's matmul
    // (and ggml-cuda's NCCL path reduces prompt-sized tensors in BF16); the random-weight
    // fixture amplifies that like any change of graph shape. A wrong split, slot or row moves
    // logits by whole units, far past this bound.
    private const float Tolerance = 0.5f;

    public MiniMaxM3TensorParallelTests(ITestOutputHelper output)
    {
        _output = output;
        MoeCpuOffloadConfig.Reset();
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "1024");
        Environment.SetEnvironmentVariable("TS_MINIMAX_M3_UBATCH", "64");
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, null);
    }

    public void Dispose()
    {
        foreach (var (name, value) in _old) Environment.SetEnvironmentVariable(name, value);
        MoeCpuOffloadConfig.Reset();
    }

    private static string Fixture => Environment.GetEnvironmentVariable(MiniMaxM3ModelNativeExecutorTests.FixtureVariable);

    private static int[] Tokens(int count, int seed)
    {
        var tokens = new int[count];
        uint state = (uint)seed * 2654435761u + 7;
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            tokens[i] = (int)(state % 199000) + 1000;
        }
        return tokens;
    }

    private static int Argmax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    private void Close(string what, float[] reference, ReadOnlySpan<float> actual)
    {
        Assert.Equal(reference.Length, actual.Length);
        float diff = 0, a = float.NegativeInfinity, b = float.NegativeInfinity;
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(float.IsFinite(reference[i]) && float.IsFinite(actual[i]), $"{what}: non-finite logit at {i}");
            diff = MathF.Max(diff, MathF.Abs(reference[i] - actual[i]));
            if (reference[i] > a) { b = a; a = reference[i]; } else if (reference[i] > b) b = reference[i];
        }
        _output.WriteLine($"{what}: max |dlogit| {diff:G4}, reference top-1 margin {a - b:G4}");
        Assert.True(diff <= Tolerance, $"{what}: max |dlogit| {diff} > {Tolerance}");
        if (a - b > 2 * diff) Assert.Equal(Argmax(reference), Argmax(actual));
    }

    /// <summary>A prompt, then eight greedy decodes; every row's logits.</summary>
    private static List<float[]> Run(MiniMaxM3Model model, int[] prompt, int[] chain)
    {
        var rows = new List<float[]>();
        model.ResetKVCache();
        rows.Add((float[])model.Forward(prompt).Clone());
        foreach (int t in chain) rows.Add((float[])model.Forward([t]).Clone());
        return rows;
    }

    [MiniMaxM3TensorParallelFact]
    public void TensorParallelMatchesOneGpu()
    {
        int tp = Degree;
        var prompts = new[] { Tokens(40, 40), Tokens(150, 150) };
        var chains = new int[prompts.Length][];
        var reference = new List<float[]>[prompts.Length];
        using (var one = new MiniMaxM3Model(Fixture, BackendType.GgmlCuda))
        {
            for (int p = 0; p < prompts.Length; p++)
            {
                // The greedy chain of the one-GPU model, so both feed identical tokens.
                one.ResetKVCache();
                int next = Argmax(one.Forward(prompts[p]));
                chains[p] = new int[8];
                for (int i = 0; i < 8; i++) { chains[p][i] = next; next = Argmax(one.Forward([next])); }
                reference[p] = Run(one, prompts[p], chains[p]);
            }
        }

        using var model = new MiniMaxM3Model(Fixture, BackendType.GgmlCuda, tpDegree: tp);
        long[] stats = model.NativeStatsForTests();
        Assert.Equal(tp, stats[11]);   // tensor-parallel degree
        Assert.Equal(0, stats[3]);     // no host experts
        Assert.Equal(0, stats[9]);     // no pipelined prefill
        for (int p = 0; p < prompts.Length; p++)
        {
            var rows = Run(model, prompts[p], chains[p]);
            for (int i = 0; i < rows.Count; i++)
                Close($"tp {tp} prompt {prompts[p].Length} row {i}", reference[p][i], rows[i]);
        }

        // An exact rewind: the kept rows were written by the same graphs either way.
        int[] seq = Tokens(174, 7);
        model.ResetKVCache();
        model.Forward(seq);
        Assert.True(model.TryTruncateKVCache(150));
        float[] rewound = (float[])model.Forward([seq[150]]).Clone();
        model.ResetKVCache();
        model.Forward(seq[..150]);
        float[] fresh = model.Forward([seq[150]]);
        Close($"tp {tp} rewind to 150", fresh, rewound);

        // Per-request slots and one batched decode step over a dense and a sparse slot.
        int[] a = Tokens(40, 11), b = Tokens(150, 12);
        Assert.True(model.BindSequenceCache("a"));
        model.Forward(a);
        Assert.True(model.BindSequenceCache("b"));
        model.Forward(b);
        int x = 4242, y = 7777;
        model.BindSequenceCache("a");
        float[] singleA = (float[])model.Forward([x]).Clone();
        Assert.True(model.TryTruncateKVCache(a.Length));
        model.BindSequenceCache("b");
        float[] singleB = (float[])model.Forward([y]).Clone();
        Assert.True(model.TryTruncateKVCache(b.Length));
        var outLogits = new float[2][];
        Assert.True(model.TryForwardBatchedFusedDecode(["a", "b"], [x, y], [a.Length, b.Length], outLogits));
        Close($"tp {tp} batched decode, slot a (dense)", singleA, outLogits[0]);
        Close($"tp {tp} batched decode, slot b (sparse)", singleB, outLogits[1]);
        model.RestorePrimaryCache();
        model.OnSequenceReleased("a");
        model.OnSequenceReleased("b");
    }
}
