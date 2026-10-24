// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The managed MiniMax-M3 contracts over the real native executor
// (ggml_ops_minimax_m3_exec.cpp), on eng/minimax-m3-fixture.py's small fixture:
// all-row speculative verification against one-token decodes, an exact rewind,
// per-request slots, batched decode and batched prefill against their single-slot
// equivalents. Each case runs once below position 64 (where the fixture's
// sparse attention equals dense) and once past it (top-4 of 16-row blocks).
// Two failure paths too: a batched decode step whose later chunk fails leaves no
// slot advanced, and a prefix-cache clone that must free memory frees no live slot.
//
// Numerical parity with llama.cpp is eng/validation/minimax-m3-parity.cpp's job;
// these tests compare the executor with itself through the managed surface.
// Requires TS_TEST_MINIMAX_M3_FIXTURE (the fixture GGUF) and TS_TEST_GGML_BACKEND.
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MiniMaxM3FixtureFactAttribute : FactAttribute
{
    private BackendType _ggmlBackend;

    public MiniMaxM3FixtureFactAttribute()
    {
        Skip = TestGates.ModelSkip(MiniMaxM3ModelNativeExecutorTests.FixtureVariable);
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
public sealed class MiniMaxM3ModelNativeExecutorTests : IDisposable
{
    internal const string FixtureVariable = "TS_TEST_MINIMAX_M3_FIXTURE";

    private static readonly string[] Saved =
        ["MAX_CONTEXT", "TS_MINIMAX_M3_UBATCH", MiniMaxM3Architecture.MsaModeVariable, "TS_MINIMAX_M3_SLOT_INIT_ROWS"];
    private readonly Dictionary<string, string> _old = Saved.ToDictionary(n => n, Environment.GetEnvironmentVariable);
    private readonly ITestOutputHelper _output;

    public MiniMaxM3ModelNativeExecutorTests(ITestOutputHelper output)
    {
        _output = output;
        MoeCpuOffloadConfig.Reset();
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "1024");
        // Small ubatches so a prompt crosses several native graphs.
        Environment.SetEnvironmentVariable("TS_MINIMAX_M3_UBATCH", "64");
        Environment.SetEnvironmentVariable(MiniMaxM3Architecture.MsaModeVariable, null);
        Environment.SetEnvironmentVariable("TS_MINIMAX_M3_SLOT_INIT_ROWS", null);
    }

    public void Dispose()
    {
        foreach (var (name, value) in _old) Environment.SetEnvironmentVariable(name, value);
        MoeCpuOffloadConfig.Reset();
    }

    private static MiniMaxM3Model Load(BackendType backend)
        => new(Environment.GetEnvironmentVariable(FixtureVariable), backend);

    /// <summary>Ordinary-text ids of the real vocabulary (below the 200000 special-token range).</summary>
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

    /// <summary>Top-1 minus top-2 logit.</summary>
    private static float Margin(ReadOnlySpan<float> row)
    {
        float a = float.NegativeInfinity, b = float.NegativeInfinity;
        foreach (float x in row)
        {
            if (x > a) { b = a; a = x; }
            else if (x > b) b = x;
        }
        return a - b;
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        Assert.Equal(x.Length, y.Length);
        float d = 0;
        for (int i = 0; i < x.Length; i++)
        {
            Assert.True(float.IsFinite(x[i]) && float.IsFinite(y[i]));
            d = MathF.Max(d, MathF.Abs(x[i] - y[i]));
        }
        return d;
    }

    /// <summary>Two logit rows from different graph shapes (one-token decode against a multi-row
    /// graph) agree to <paramref name="tolerance"/> and pick the same token wherever the reference
    /// is not a near-tie.</summary>
    private void Close(string what, float[] reference, ReadOnlySpan<float> actual, float tolerance)
    {
        float diff = MaxAbsDiff(reference, actual);
        float margin = Margin(reference);
        _output.WriteLine($"{what}: max |dlogit| {diff:G4}, reference top-1 margin {margin:G4}");
        Assert.True(diff <= tolerance, $"{what}: max |dlogit| {diff} > {tolerance}");
        if (margin > 2 * diff) Assert.Equal(Argmax(reference), Argmax(actual));
    }

    // Different graph shapes reduce in different orders over F16 K/V and Q8_0/NVFP4 weights,
    // and the random-weight fixture amplifies that: llama.cpp's own one-token decode and its
    // prefill of the same tokens differ by 0.24 nats in a top-2 log-probability gap on it.
    // A wrong slot, row or position moves logits by whole units, far past this bound.
    private const float ShapeTolerance = 0.5f;

    // The CPU kernels compute a row the same way whatever the graph's shape, so there a
    // verification row must equal the one-token decode it stands for (with dense attention
    // forced they agree to 2e-6). That is what makes speculation output-preserving: the sparse
    // selection of a position may not depend on how many tokens its graph carried.
    private const float CpuExactTolerance = 1e-3f;

    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaVerifyRowsMatchOneTokenDecodes() => VerifyRowsMatchOneTokenDecodes(BackendType.GgmlCuda, ShapeTolerance);

    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCpu)]
    public void CpuVerifyRowsMatchOneTokenDecodes() => VerifyRowsMatchOneTokenDecodes(BackendType.GgmlCpu, CpuExactTolerance);

    private void VerifyRowsMatchOneTokenDecodes(BackendType backend, float tolerance)
    {
        using var model = Load(backend);
        int vocab = model.Config.VocabSize;
        foreach (int length in new[] { 40, 150 })
        {
            int[] prompt = Tokens(length, length);
            model.ResetKVCache();
            int next = Argmax(model.Forward(prompt));
            var chain = new int[8];
            var decoded = new float[chain.Length][];
            for (int i = 0; i < chain.Length; i++)
            {
                chain[i] = next;
                decoded[i] = (float[])model.Forward([next]).Clone();
                next = Argmax(decoded[i]);
            }
            Assert.Equal(length + chain.Length, model.CacheSeqLen);

            // The same rows as one all-row verification over the chain.
            model.ResetKVCache();
            model.Forward(prompt);
            var rows = new float[chain.Length * vocab];
            model.SpecForward(chain, null, rows, allLogitsRows: true);
            Assert.Equal(length + chain.Length, model.CacheSeqLen);
            for (int i = 0; i < chain.Length; i++)
                Close($"{backend} prompt {length} verify row {i}", decoded[i], rows.AsSpan(i * vocab, vocab), tolerance);

            // Reject the last four rows; the next decode continues from the kept prefix.
            model.SpecRewindCache(length + 4);
            Assert.Equal(length + 4, model.CacheSeqLen);
            Close($"{backend} prompt {length} decode after rejecting 4", decoded[4], model.Forward([chain[4]]), tolerance);
        }
    }

    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCuda)]
    public void RewindIsExactAtAnyDepth()
    {
        using var model = Load(BackendType.GgmlCuda);
        foreach (int length in new[] { 40, 150 })
        {
            int[] tokens = Tokens(length + 24, length + 1);
            // A: the whole sequence, then a rewind deep into it (across MSA blocks past 64).
            model.ResetKVCache();
            model.Forward(tokens);
            Assert.True(model.TryTruncateKVCache(length));
            Assert.Equal(length, model.CacheSeqLen);
            float[] rewound = (float[])model.Forward([tokens[length]]).Clone();
            // B: only the kept prefix.
            model.ResetKVCache();
            model.Forward(tokens[..length]);
            float[] fresh = model.Forward([tokens[length]]);
            // The kept rows were written by prefill graphs over the same positions either way.
            Close($"rewind to {length}", fresh, rewound, 1e-3f);
            // Past the head is refused and changes nothing.
            Assert.False(model.TryTruncateKVCache(length + 5));
            Assert.Equal(length + 1, model.CacheSeqLen);
        }
    }

    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCuda)]
    public void SlotsAreIndependentAndBatchedDecodeMatchesOneSlotDecodes()
    {
        using var model = Load(BackendType.GgmlCuda);
        int vocab = model.Config.VocabSize;
        Assert.True(model.SupportsPerSequenceFusedForward);
        int[] a = Tokens(40, 11), b = Tokens(150, 12);
        Assert.True(model.BindSequenceCache("a"));
        model.Forward(a);
        Assert.True(model.BindSequenceCache("b"));
        Assert.Equal(0, model.CacheSeqLen);
        model.Forward(b);
        Assert.False(model.BindSequenceCache("a"));
        Assert.Equal(a.Length, model.CacheSeqLen);

        int x = 4242, y = 7777;
        float[] single = (float[])model.Forward([x]).Clone();
        Assert.True(model.TryTruncateKVCache(a.Length));
        model.BindSequenceCache("b");
        float[] singleB = (float[])model.Forward([y]).Clone();
        Assert.True(model.TryTruncateKVCache(b.Length));

        var outLogits = new float[2][];
        Assert.True(model.TryForwardBatchedFusedDecode(["a", "b"], [x, y], [a.Length, b.Length], outLogits));
        Close("batched decode, slot a (dense)", single, outLogits[0], ShapeTolerance);
        Close("batched decode, slot b (sparse)", singleB, outLogits[1], ShapeTolerance);
        Assert.Equal(vocab, outLogits[0].Length);
        model.BindSequenceCache("a");
        Assert.Equal(a.Length + 1, model.CacheSeqLen);

        model.RestorePrimaryCache();
        model.OnSequenceReleased("a");
        model.OnSequenceReleased("b");
        Assert.False(model.HasFusedSequenceCache("a"));
    }

    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCuda)]
    public void BatchedPrefillMatchesOneSlotPrefills()
    {
        using var model = Load(BackendType.GgmlCuda);
        int[] c = Tokens(40, 21), d = Tokens(150, 22);
        model.BindSequenceCache("c-single");
        float[] singleC = (float[])model.Forward(c).Clone();
        model.BindSequenceCache("d-single");
        float[] singleD = (float[])model.Forward(d).Clone();
        model.RestorePrimaryCache();

        var outLogits = new float[2][];
        Assert.True(model.TryForwardBatchedFusedPrefill(["c", "d"], [c, d], [0, 0], outLogits));
        Close("batched prefill, sequence c", singleC, outLogits[0], ShapeTolerance);
        Close("batched prefill, sequence d", singleD, outLogits[1], ShapeTolerance);
        model.BindSequenceCache("d");
        Assert.Equal(d.Length, model.CacheSeqLen);
        model.RestorePrimaryCache();
        foreach (string id in new[] { "c", "d", "c-single", "d-single" }) model.OnSequenceReleased(id);
    }

    /// <summary>A holder with rows past the position the batched prefill continues from (a cached state
    /// cloned for a request that reuses less of it) is rewound there, and its prompt tail then gives the
    /// logits of one prefill of the whole prompt.</summary>
    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCuda)]
    public void BatchedPrefillRewindsAHolderPastItsPosition()
    {
        using var model = Load(BackendType.GgmlCuda);
        Assert.True(model.BatchedPrefillRewindsHolders);
        int[] e = Tokens(120, 23), f = Tokens(60, 24);
        model.BindSequenceCache("e-single");
        float[] singleE = (float[])model.Forward(e).Clone();
        model.BindSequenceCache("f-single");
        float[] singleF = (float[])model.Forward(f).Clone();
        // "x" holds e's first 100 rows; its request reuses only 80 of them.
        model.BindSequenceCache("x");
        model.Forward(e[..100]);
        model.RestorePrimaryCache();

        var outLogits = new float[2][];
        Assert.True(model.TryForwardBatchedFusedPrefill(["x", "y"], [e[80..], f], [80, 0], outLogits));
        Close("batched prefill after a rewind, sequence x", singleE, outLogits[0], ShapeTolerance);
        Close("batched prefill, sequence y", singleF, outLogits[1], ShapeTolerance);
        model.BindSequenceCache("x");
        Assert.Equal(e.Length, model.CacheSeqLen);
        model.RestorePrimaryCache();
        foreach (string id in new[] { "x", "y", "e-single", "f-single" }) model.OnSequenceReleased(id);
    }

    /// <summary>More decodes than one native call carries (16) run as several calls. When a later one fails,
    /// the earlier calls' slots already hold this step's token: the step must report that nothing advanced
    /// with those slots back at the step's positions, or the executor's serial fallback forwards their
    /// tokens a second time (a duplicated row, and a head one past the sequence).</summary>
    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCuda)]
    public void ABatchedDecodeWhoseLaterChunkFailsLeavesNoSlotAdvanced()
    {
        using var model = Load(BackendType.GgmlCuda);
        const int n = 17;   // two native calls: sequences 0-8, then 9-16
        var ids = new string[n];
        var tokens = new int[n];
        var positions = new int[n];
        for (int i = 0; i < n; i++)
        {
            ids[i] = "d" + i;
            Assert.True(model.BindSequenceCache(ids[i]));
            model.Forward(Tokens(20 + i, 300 + i));
            tokens[i] = 4242 + i;
            positions[i] = 20 + i;
        }
        model.RestorePrimaryCache();
        // The second call's last sequence names a position its slot is not at, so that call fails.
        positions[n - 1] += 3;
        Assert.False(model.TryForwardBatchedFusedDecode(ids, tokens, positions, new float[n][]));
        for (int i = 0; i < n; i++)
        {
            model.BindSequenceCache(ids[i]);
            Assert.Equal(20 + i, model.CacheSeqLen);
        }
        model.RestorePrimaryCache();
        foreach (string id in ids) model.OnSequenceReleased(id);
    }

    /// <summary>A prefix-cache clone whose copy must grow while memory is short (every admission refused
    /// through a huge TS_MINIMAX_M3_SLOT_HEADROOM_MB) frees only slots that are still retained: not the
    /// slot an earlier admission of the same scheduling pass donated to a live request (no graph call ran
    /// in between to tell the executor), and not the clone's own source.</summary>
    [MiniMaxM3FixtureFact(GgmlBackend = BackendType.GgmlCuda)]
    public void ACloneUnderMemoryPressureFreesNoLiveSlotAndKeepsItsSource()
    {
        // New slots start at 256 rows, so the clone of a 300-token checkpoint has to grow.
        Environment.SetEnvironmentVariable("TS_MINIMAX_M3_SLOT_INIT_ROWS", "256");
        using var model = Load(BackendType.GgmlCuda);
        int[] shared = Tokens(300, 31), turn = Tokens(40, 32);
        model.BindSequenceCache("a");
        model.Forward(shared);
        Assert.True(model.TryCaptureCopy("a", "pc:checkpoint", out _));
        model.BindSequenceCache("b");
        model.Forward(turn);
        Assert.True(model.TryCaptureDonate("b", "pc:end-b", turn.Length, out _));
        // A slot that never grew, freed: pooled, it becomes the clone's fresh slot without an admission check.
        model.BindSequenceCache("pooled");
        model.Forward(turn[..4]);
        model.OnSequenceReleased("pooled");
        // A graph call hands the executor both retained slots (end state first).
        model.BindSequenceCache("a");
        model.Forward([turn[0]]);
        model.RestorePrimaryCache();

        // One scheduling pass: b2 continues b (b's slot is donated: live), then c clones the checkpoint.
        Assert.True(model.TryMaterialize(new MaterializeRequest(MaterializeOp.Donate, "pc:end-b", "b2", turn.Length, turn.Length)));
        GgmlMiniMaxM3Native.SetOption("TS_MINIMAX_M3_SLOT_HEADROOM_MB", "100000000");
        bool cloned;
        try
        {
            cloned = model.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "pc:checkpoint", "c", shared.Length, shared.Length));
        }
        finally
        {
            GgmlMiniMaxM3Native.SetOption("TS_MINIMAX_M3_SLOT_HEADROOM_MB", null);
        }
        Assert.True(cloned);
        Assert.Contains("pc:checkpoint", model.RetainedPayloadKeys);
        model.BindSequenceCache("b2");   // the donated slot is still there
        Assert.Equal(turn.Length, model.CacheSeqLen);
        model.BindSequenceCache("c");
        Assert.Equal(shared.Length, model.CacheSeqLen);
        model.RestorePrimaryCache();
        foreach (string id in new[] { "a", "b2", "c" }) model.OnSequenceReleased(id);
    }
}
