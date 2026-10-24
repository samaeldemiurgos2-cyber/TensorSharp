// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The native whole-model executor (ggml_ops_muse_spark_exec.cpp) against the
/// independent double-precision decoder: chunked prefill, decode across the
/// sliding-window ring's wrap, exact and refused rewinds, batched decode over
/// several sequence slots, all-row speculative verification and image rows.
/// </summary>
[Collection("Muse Spark model")]
public sealed class MuseSparkNativeExecutorTests : IDisposable
{
    private static readonly string[] Saved =
    [
        "TS_MUSE_SPARK_NATIVE", "MAX_CONTEXT", "TS_MUSE_SPARK_KV_PAD", "TS_MUSE_SPARK_UBATCH",
        "TS_MUSE_SPARK_SWA_REWIND", "TS_MUSE_SPARK_FUSE_QKVG", "TS_MUSE_SPARK_SWA_FULL", "TS_MUSE_SPARK_SLOT_INIT_ROWS",
        "TS_MUSE_SPARK_HOST_MOE_TOKENS", "TS_MUSE_SPARK_PIPELINE", "TS_MUSE_SPARK_PIPELINE_UBATCH",
        "TS_MUSE_SPARK_LAYER_SPLIT",
    ];
    private readonly Dictionary<string, string> _old = Saved.ToDictionary(n => n, Environment.GetEnvironmentVariable);
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-muse-spark-native-" + Guid.NewGuid().ToString("N"));
    private readonly MuseSparkSyntheticModelBuilder _fixture = new();
    private const int Vocab = MuseSparkSyntheticModelBuilder.Vocab;
    // Native K/V rows are F16; the oracle keeps them in double precision.
    private const float Tolerance = 6e-3f;

    public MuseSparkNativeExecutorTests()
    {
        Directory.CreateDirectory(_dir);
        MoeCpuOffloadConfig.Reset();
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_NATIVE", "1");
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "64");
        // A 4-row pad, 4-token ubatches and no rewind margin give a 24-row ring
        // (window 3 + ubatch 4 + batched-decode allowance 16) that 40 tokens wrap.
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", "4");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_UBATCH", "4");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_SWA_REWIND", "0");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG", "1");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_SWA_FULL", null);
        // Global K/V rows start at 8 and double while sequences grow, so every
        // test also crosses several row reallocations (and their graph rebuilds).
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_SLOT_INIT_ROWS", "8");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_LAYER_SPLIT", null);
    }

    public void Dispose()
    {
        foreach (var (name, value) in _old) Environment.SetEnvironmentVariable(name, value);
        MoeCpuOffloadConfig.Reset();
        Directory.Delete(_dir, true);
    }

    private static int[] Tokens(int count, int seed)
    {
        var tokens = new int[count];
        uint state = (uint)seed * 2654435761u + 7;
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            tokens[i] = (int)(state % (Vocab - 2)) + 2;
        }
        return tokens;
    }

    private static void Close(float[] expected, ReadOnlySpan<float> actual, float tolerance = Tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        float error = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]));
            error = MathF.Max(error, MathF.Abs(expected[i] - actual[i]));
        }
        Assert.True(error <= tolerance, $"maximum logit error {error} > {tolerance}");
    }

    private MuseSparkModel Model(BackendType backend) => new(_fixture.Write(Path.Combine(_dir, Guid.NewGuid() + ".gguf")), backend);

    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuPrefillDecodeAcrossRingWrapAndRewinds() => RingCheck(BackendType.GgmlCpu);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaPrefillDecodeAndRewinds() => RingCheck(BackendType.GgmlCuda);

    private void RingCheck(BackendType backend)
    {
        if (backend != BackendType.GgmlCpu) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", null);
        using var model = Model(backend);
        int[] tokens = Tokens(40, 1);
        // 10 tokens arrive as ubatches of 4, 4 and 2.
        Close(_fixture.Reference(tokens[..10]), model.Forward(tokens[..10]));
        for (int end = 11; end <= tokens.Length; end++)
            Close(_fixture.Reference(tokens[..end]), model.Forward([tokens[end - 1]]));
        Assert.Equal(tokens.Length, model.CacheSeqLen);

        // A short rewind keeps the window the next query reads: exact.
        Assert.True(model.TryTruncateKVCache(37));
        Close(_fixture.Reference(tokens[..38]), model.Forward([tokens[37]]));
        if (backend == BackendType.GgmlCpu)
        {
            // Deeper than the ring retains: refused, nothing changes.
            Assert.False(model.TryTruncateKVCache(10));
            Assert.Equal(38, model.CacheSeqLen);
            Close(_fixture.Reference(tokens[..39]), model.Forward([tokens[38]]));
        }

        // A fresh sequence from position 0 sees nothing of the previous one.
        model.ResetKVCache();
        Close(_fixture.Reference(tokens[5..12]), model.Forward(tokens[5..12]));
    }

    /// <summary>Pipelined prefill (forward_pipelined): the equal micro-batches of one call are
    /// enqueued back to back on one scheduler with rotating input copies, the last one padded
    /// past the call's end. Logits after the prefill, and every decode step that reads the K/V
    /// rows it wrote (over the rows the padding touched, and on the CPU, whose 4-row pad gives a
    /// 24-row ring, across the sliding ring's wrap), must match the independent decoder. On GPUs
    /// the 256-row pad keeps every window and the ring at 256 rows here; the growing-window
    /// variants below cover windows that change inside one call.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuPipelinedPrefillMatchesReference() => PipelineCheck(BackendType.GgmlCpu, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaPipelinedPrefillMatchesReference() => PipelineCheck(BackendType.GgmlCuda, 1);

    [MultiCudaFact(2, GgmlBackend = BackendType.GgmlCuda)]
    public void MultiCudaPipelinedPrefillAcrossGpusMatchesReference() => PipelineCheck(BackendType.GgmlCuda, 2);

    private void PipelineCheck(BackendType backend, int gpus)
    {
        if (backend != BackendType.GgmlCpu) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", null);
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PIPELINE", "1");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PIPELINE_UBATCH", "4");
        // One layer per GPU: a stage on each, whatever the devices' free memory.
        if (gpus > 1) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_LAYER_SPLIT", OneLayerPerGpu(gpus));
        using var model = new MuseSparkModel(_fixture.Write(Path.Combine(_dir, Guid.NewGuid() + ".gguf")), backend,
            layerSplitDegree: gpus);
        // On GPUs the load already ran one pipelined warm-up; count from here.
        long[] before = model.NativeStatsForTests();
        Assert.NotNull(before);
        int[] tokens = Tokens(48, 9);
        // 14 tokens: four 4-row micro-batches, the last padded with two copies of
        // the final token; the decode steps then rewrite the padded rows.
        Close(_fixture.Reference(tokens[..14]), model.Forward(tokens[..14]));
        for (int end = 15; end <= 18; end++)
            Close(_fixture.Reference(tokens[..end]), model.Forward([tokens[end - 1]]));

        // 32 tokens from position 0: eight micro-batches, and the logits come from the
        // pipeline itself (on the CPU the 24-row sliding ring wraps during the call).
        model.ResetKVCache();
        Close(_fixture.Reference(tokens[..32]), model.Forward(tokens[..32]));
        for (int end = 33; end <= 40; end++)
            Close(_fixture.Reference(tokens[..end]), model.Forward([tokens[end - 1]]));

        // A pipelined call that resumes a sequence mid-window.
        Assert.True(model.TryTruncateKVCache(36));
        Close(_fixture.Reference(tokens[..48]), model.Forward(tokens[36..48]));

        // All three prefills really ran pipelined: 4 + 8 + 3 micro-batches, none declined.
        long[] stats = model.NativeStatsForTests();
        Assert.NotNull(stats);
        Assert.Equal(3, stats[6] - before[6]);
        Assert.Equal(15, stats[7] - before[7]);
        Assert.Equal(0, stats[8] - before[8]);
        Assert.Equal(backend == BackendType.GgmlCpu ? 0 : gpus, stats[10]);
    }

    private static string OneLayerPerGpu(int gpus)
        => string.Join(",", Enumerable.Repeat(MuseSparkSyntheticModelBuilder.Layers / gpus, gpus));

    /// <summary>Pipelined prefill whose K/V windows grow inside one call: with the GPUs' 256-row
    /// pad, 64-row ubatches and a 1024-token context, a 650-token prompt runs as 11 micro-batches
    /// whose global window grows 256 -> 512 -> 768 within the scheduler reserved for the last one,
    /// while the 256-row sliding ring wraps and later micro-batches wait on earlier ones' copy
    /// slots (more than the scheduler's input copies are in flight). A second call with smaller
    /// windows reuses the reservation. With <paramref name="globalFirst"/> the first GPU holds
    /// only the global (NoPE) layer, so it never reads positions.</summary>
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaPipelinedPrefillGrowingWindowsMatchesReference() => GrowingWindowCheck(1, globalFirst: false);

    [MultiCudaFact(2, GgmlBackend = BackendType.GgmlCuda)]
    public void MultiCudaPipelinedPrefillGrowingWindowsMatchesReference() => GrowingWindowCheck(2, globalFirst: false);

    [MultiCudaFact(2, GgmlBackend = BackendType.GgmlCuda)]
    public void MultiCudaPipelinedPrefillGlobalOnlyFirstStageMatchesReference() => GrowingWindowCheck(2, globalFirst: true);

    private void GrowingWindowCheck(int gpus, bool globalFirst)
    {
        const int context = 1024;
        var fixture = new MuseSparkSyntheticModelBuilder(contextLength: context, slidingFirst: !globalFirst);
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", null);
        Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_UBATCH", "64");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PIPELINE", "1");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PIPELINE_UBATCH", "64");
        if (gpus > 1) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_LAYER_SPLIT", OneLayerPerGpu(gpus));
        using var model = new MuseSparkModel(fixture.Write(Path.Combine(_dir, Guid.NewGuid() + ".gguf")), BackendType.GgmlCuda,
            layerSplitDegree: gpus);
        long[] before = model.NativeStatsForTests();
        Assert.NotNull(before);
        int[] tokens = Tokens(700, 11);

        // 650 tokens: 11 micro-batches of 60 rows, the last padded with 10 copies of token 649.
        Close(fixture.Reference(tokens[..650]), model.Forward(tokens[..650]), LongTolerance);
        for (int end = 651; end <= 654; end++)
            Close(fixture.Reference(tokens[..end]), model.Forward([tokens[end - 1]]), LongTolerance);

        // Resume at 600 with 100 tokens: 2 micro-batches of 50 rows, windows within the reservation.
        Assert.True(model.TryTruncateKVCache(600));
        Close(fixture.Reference(tokens[..700]), model.Forward(tokens[600..700]), LongTolerance);

        long[] stats = model.NativeStatsForTests();
        Assert.NotNull(stats);
        Assert.Equal(2, stats[6] - before[6]);
        Assert.Equal(13, stats[7] - before[7]);
        Assert.Equal(0, stats[8] - before[8]);
        Assert.Equal(64, stats[9]);
        Assert.Equal(gpus, stats[10]);
    }

    // Hundreds of positions of F16 K/V against the double-precision oracle.
    private const float LongTolerance = 2e-2f;

    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuBatchedDecodeOverSlotsMatchesIndependentSequences() => BatchedCheck(BackendType.GgmlCpu);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaBatchedDecodeOverSlotsMatchesIndependentSequences() => BatchedCheck(BackendType.GgmlCuda);

    private void BatchedCheck(BackendType backend)
    {
        if (backend != BackendType.GgmlCpu) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", null);
        using var model = Model(backend);
        Assert.True(model.SupportsPerSequenceFusedForward);
        int[][] prompts = [Tokens(30, 2), Tokens(30, 3), Tokens(30, 4)];
        int[] lengths = [6, 9, 2];
        string[] ids = ["a", "b", "c"];
        for (int s = 0; s < ids.Length; s++)
        {
            Assert.True(model.BindSequenceCache(ids[s]));
            Close(_fixture.Reference(prompts[s][..lengths[s]]), model.Forward(prompts[s][..lengths[s]]));
        }
        // 20 batched steps carry every sequence across its ring boundary.
        for (int step = 0; step < 20; step++)
        {
            var tokens = new int[ids.Length];
            var positions = new int[ids.Length];
            for (int s = 0; s < ids.Length; s++) { tokens[s] = prompts[s][lengths[s]]; positions[s] = lengths[s]; }
            var logits = new float[ids.Length][];
            Assert.True(model.TryForwardBatchedFusedDecode(ids, tokens, positions, logits));
            for (int s = 0; s < ids.Length; s++)
            {
                lengths[s]++;
                float[] expected = _fixture.Reference(prompts[s][..lengths[s]]);
                if (Environment.GetEnvironmentVariable("TS_TEST_VERBOSE") == "1")
                    Console.WriteLine($"step {step} slot {ids[s]} len {lengths[s]} err {expected.Zip(logits[s], (x, y) => MathF.Abs(x - y)).Max()}");
                Close(expected, logits[s]);
            }
        }
        // A position that disagrees with its slot declines the step without advancing anything.
        var bad = new float[2][];
        Assert.False(model.TryForwardBatchedFusedDecode(["a", "b"], [1, 2], [0, lengths[1]], bad));
        model.BindSequenceCache("b");
        Close(_fixture.Reference(prompts[1][..(lengths[1] + 1)]), model.Forward([prompts[1][lengths[1]]]));
        foreach (string id in ids) model.OnSequenceReleased(id);
    }

    /// <summary>Host experts for a few tokens run through the executor's static-row kernel
    /// (host_moe_matmul); with it switched off they run through ggml's mul_mat_id. Both
    /// quantize and reduce with ggml's own CPU type traits, so the logits must agree.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuHostExpertKernelMatchesMulMatId()
    {
        float[][] Run(string hostMoeTokens)
        {
            Environment.SetEnvironmentVariable("TS_MUSE_SPARK_HOST_MOE_TOKENS", hostMoeTokens);
            using var model = Model(BackendType.GgmlCpu);
            int[][] prompts = [Tokens(16, 5), Tokens(16, 6), Tokens(16, 7), Tokens(16, 8)];
            string[] ids = ["a", "b", "c", "d"];
            var outputs = new List<float[]>();
            for (int s = 0; s < ids.Length; s++)
            {
                Assert.True(model.BindSequenceCache(ids[s]));
                outputs.Add(model.Forward(prompts[s][..5]).ToArray());   // ubatches of 4 and 1 tokens
            }
            for (int step = 0; step < 6; step++)
            {
                var logits = new float[ids.Length][];
                Assert.True(model.TryForwardBatchedFusedDecode(ids, ids.Select((_, s) => prompts[s][5 + step]).ToArray(),
                    ids.Select(_ => 5 + step).ToArray(), logits));
                outputs.AddRange(logits.Select(l => l.ToArray()));
            }
            foreach (string id in ids) model.OnSequenceReleased(id);
            return outputs.ToArray();
        }

        float[][] kernel = Run(null);
        float[][] mulMatId = Run("0");
        Assert.Equal(mulMatId.Length, kernel.Length);
        for (int i = 0; i < kernel.Length; i++) Close(mulMatId[i], kernel[i], 1e-5f);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuBatchedPrefillOverNewSlotsMatchesIndependentSequences() => BatchedPrefillCheck(BackendType.GgmlCpu);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaBatchedPrefillOverNewSlotsMatchesIndependentSequences() => BatchedPrefillCheck(BackendType.GgmlCuda);

    private void BatchedPrefillCheck(BackendType backend)
    {
        if (backend != BackendType.GgmlCpu) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", null);
        using var model = Model(backend);
        int[][] prompts = [Tokens(30, 7), Tokens(30, 8), Tokens(30, 9)];
        string[] ids = ["p", "q", "r"];
        // 6 + 9 + 2 tokens through 4-token graphs: segments split across calls.
        int[] lengths = [6, 9, 2];
        var logits = new float[3][];
        Assert.True(model.TryForwardBatchedFusedPrefill(ids, [prompts[0][..6], prompts[1][..9], prompts[2][..2]], [0, 0, 0], logits));
        for (int s = 0; s < ids.Length; s++)
        {
            Assert.True(model.HasFusedSequenceCache(ids[s]));
            Close(_fixture.Reference(prompts[s][..lengths[s]]), logits[s]);
        }
        // A second chunk continues each slot from its head.
        Assert.True(model.TryForwardBatchedFusedPrefill(ids, [prompts[0][6..11], prompts[1][9..10], prompts[2][2..7]],
            [6, 9, 2], logits));
        lengths = [11, 10, 7];
        for (int s = 0; s < ids.Length; s++) Close(_fixture.Reference(prompts[s][..lengths[s]]), logits[s]);
        // A start that disagrees with a slot's head declines without advancing anything.
        Assert.False(model.TryForwardBatchedFusedPrefill(ids, [[1], [2], [3]], [0, 10, 7], logits));
        var step = new float[3][];
        Assert.True(model.TryForwardBatchedFusedDecode(ids, [prompts[0][11], prompts[1][10], prompts[2][7]], [11, 10, 7], step));
        for (int s = 0; s < ids.Length; s++) Close(_fixture.Reference(prompts[s][..(lengths[s] + 1)]), step[s]);
        foreach (string id in ids) model.OnSequenceReleased(id);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuCheckpointCopyAndClonesContinueIndependently() => CloneCheck(BackendType.GgmlCpu);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaCheckpointCopyAndClonesContinueIndependently() => CloneCheck(BackendType.GgmlCuda);

    private void CloneCheck(BackendType backend)
    {
        if (backend != BackendType.GgmlCpu) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", null);
        using var model = Model(backend);
        var prefixCache = (TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel)model;
        int[] shared = Tokens(30, 10);   // 30 > the 24-row CPU ring: the copy must carry the wrapped labels
        int[] a = shared.Concat(Tokens(6, 11)).ToArray(), b = shared.Concat(Tokens(6, 12)).ToArray();
        Assert.True(model.BindSequenceCache("owner"));
        Close(_fixture.Reference(shared), model.Forward(shared));
        Assert.True(prefixCache.TryCaptureCopy("owner", "pc:test:1", out _));
        // Two clones of the checkpoint each continue with their own suffix.
        foreach (var (id, tokens) in new[] { ("cloneA", a), ("cloneB", b) })
        {
            Assert.True(prefixCache.TryMaterialize(new TensorSharp.Runtime.Scheduling.PrefixCache.MaterializeRequest(
                TensorSharp.Runtime.Scheduling.PrefixCache.MaterializeOp.Clone, "pc:test:1", id, shared.Length, shared.Length)));
            Assert.False(model.BindSequenceCache(id));
            Assert.Equal(shared.Length, model.CacheSeqLen);
            Close(_fixture.Reference(tokens), model.Forward(tokens[shared.Length..]));
        }
        // The source and the checkpoint are untouched by the clones' continuations.
        model.BindSequenceCache("owner");
        Close(_fixture.Reference(shared.Concat(new[] { 5 }).ToArray()), model.Forward([5]));
        Assert.Contains("pc:test:1", ((TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModelDiagnostics)model).RetainedPayloadKeys);
        foreach (string id in new[] { "owner", "cloneA", "cloneB" }) model.OnSequenceReleased(id);
        prefixCache.ReleasePayloads(new[] { "pc:test:1" }, TensorSharp.Runtime.Scheduling.PrefixCache.ReleaseReason.Invalidated);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuSpeculativeAllRowsAndShortRewindMatchPrefixes()
    {
        using var model = Model(BackendType.GgmlCpu);
        int[] tokens = Tokens(12, 5);
        var logits = new float[tokens.Length * Vocab];
        model.SpecForward(tokens, null, logits, true);
        for (int end = 1; end <= tokens.Length; end++)
            Close(_fixture.Reference(tokens[..end]), logits.AsSpan((end - 1) * Vocab, Vocab));
        model.SpecRewindCache(9);
        Close(_fixture.Reference(tokens[..10]), model.Forward([tokens[9]]));
        Assert.Throws<NotSupportedException>(() => model.SpecForward([1], new float[MuseSparkSyntheticModelBuilder.Hidden], new float[Vocab], false));
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuVisionRowsSurviveUbatchBoundaries() => VisionCheck(BackendType.GgmlCpu);

    // The token gather runs on the host while the layers run on the GPU; the
    // image-row override must stay with the gathered rows (it once ran on the
    // GPU and wrote into host memory).
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void CudaVisionRowsOverrideHostGatheredEmbeddings() => VisionCheck(BackendType.GgmlCuda);

    private void VisionCheck(BackendType backend)
    {
        if (backend != BackendType.GgmlCpu) Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_PAD", null);
        using var model = Model(backend);
        int[] tokens = Tokens(9, 6);
        var replacements = new Dictionary<int, float[]>();
        float[] values = Enumerable.Range(0, 5 * MuseSparkSyntheticModelBuilder.Hidden).Select(x => MathF.Sin(x * 0.37f)).ToArray();
        var embeddings = new Tensor(new CpuAllocator(BlasEnum.DotNet), DType.Float32, 5, MuseSparkSyntheticModelBuilder.Hidden);
        embeddings.SetElementsAsFloat(values);
        // Rows 2..6 straddle the 4-token ubatch boundary.
        model.SetVisionEmbeddings(embeddings, 2);
        for (int row = 0; row < 5; row++)
            replacements[row + 2] = values.AsSpan(row * MuseSparkSyntheticModelBuilder.Hidden, MuseSparkSyntheticModelBuilder.Hidden).ToArray();
        Close(_fixture.Reference(tokens, replacements: replacements), model.Forward(tokens));
        Assert.False(model.HasPendingVisionEmbeddings);
    }
}
