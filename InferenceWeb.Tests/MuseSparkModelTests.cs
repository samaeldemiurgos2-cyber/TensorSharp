// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[CollectionDefinition("Muse Spark model", DisableParallelization = true)]
public sealed class MuseSparkModelCollection { }

[Collection("Muse Spark model")]
public sealed class MuseSparkModelTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-muse-spark-" + Guid.NewGuid().ToString("N"));
    private readonly MuseSparkSyntheticModelBuilder _fixture = new();
    private readonly string _oldChunk = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK");
    private readonly string _oldContext = Environment.GetEnvironmentVariable("MAX_CONTEXT");
    private readonly string _oldInitial = Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS");
    private readonly string _oldFusion = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG");
    private readonly string _oldCopies = Environment.GetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES");
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldDtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    public MuseSparkModelTests()
    {
        Directory.CreateDirectory(_dir);
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        MoeCpuOffloadConfig.Reset();
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "64");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK", "2");
        Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "2");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG", "1");
        Environment.SetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES", "1");
    }
    public void Dispose()
    {
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK", _oldChunk);
        Environment.SetEnvironmentVariable("MAX_CONTEXT", _oldContext);
        Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", _oldInitial);
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG", _oldFusion);
        Environment.SetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES", _oldCopies);
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldDtypeExplicit); MoeCpuOffloadConfig.Reset();
        Directory.Delete(_dir, true);
    }
    private MuseSparkModel Model() => new(_fixture.Write(Path.Combine(_dir, Guid.NewGuid() + ".gguf")), BackendType.Cpu);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuHalfWeightsMatchIndependentDecoder() => NativeCheck(BackendType.GgmlCpu, false, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaHalfWeightsMatchIndependentDecoder() => NativeCheck(BackendType.GgmlCuda, false, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaCpuExpertsMatchIndependentDecoder() => NativeCheck(BackendType.GgmlCuda, true, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaTwoLayerPlacementMatchesIndependentDecoder() => NativeCheck(BackendType.GgmlCuda, false, 2);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaPackedMixedLengthSequencesMatchIndependentDecoder() => PackedCheck(BackendType.GgmlCuda, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaPackedTwoLayerSequencesMatchIndependentDecoder() => PackedCheck(BackendType.GgmlCuda, 2);

    [Fact]
    public void FailedCacheGrowthDisposesUncommittedStorageAndPreservesPrefix() => FailedGrowthCheck(BackendType.Cpu, 1);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuFailedCacheGrowthKeepsPhysicalF16Strides() => FailedGrowthCheck(BackendType.GgmlCpu, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaFailedCacheGrowthKeepsPhysicalF16Strides() => FailedGrowthCheck(BackendType.GgmlCuda, 2);

    private void FailedGrowthCheck(BackendType backend, int split)
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        using var model = new MuseSparkModel(_fixture.Write(Path.Combine(_dir, "growth.gguf"), true), backend, layerSplitDegree: split);
        int[] tokens = [7, 21, 42, 11, 9, 86];
        model.Forward(tokens[..1]);
        var field = typeof(ModelBase).GetField("_allocator", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
        var original = (IAllocator)field.GetValue(model)!;
        var failing = new GrowthFailingAllocator(original);
        field.SetValue(model, failing);
        try { Assert.Throws<OutOfMemoryException>(() => model.PrepareForPrefill(4)); }
        finally { field.SetValue(model, original); }
        Assert.Equal(1, model.CacheSeqLen);
        Assert.Equal(3, failing.Successful.Count); // Layer0 K/V committed; layer1 K must be freed.
        Assert.Throws<InvalidOperationException>(() => failing.Successful[2].AddRef());
        var caches = (Tensor[])typeof(MuseSparkModel).GetField("_kvCacheK", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;
        Assert.Equal(4, caches[0].Sizes[1]);
        Assert.Equal(2, caches[1].Sizes[1]);
        // Fits the old common floor: no new allocation may hide the mixed
        // physical head strides left by the failed larger reservation.
        Close(_fixture.Reference(tokens[..2], true), model.Forward(tokens[1..2]), 3e-3f);
        // A later successful growth must retain both the old and new prefix rows.
        Close(_fixture.Reference(tokens, true), model.Forward(tokens[2..]), 3e-3f);
    }

    private sealed class GrowthFailingAllocator(IAllocator inner) : IAllocator
    {
        internal List<Storage> Successful { get; } = new();
        public BlasEnum BlasEnum => inner.BlasEnum;
        public int DeviceId => inner.DeviceId;
        public float GetAllocatedMemoryRatio() => inner.GetAllocatedMemoryRatio();
        public Storage Allocate(DType type, long count)
        {
            if (Successful.Count == 3) throw new OutOfMemoryException("Injected failure after one complete cache layer and the next K allocation.");
            var storage = inner.Allocate(type, count);
            Successful.Add(storage);
            return storage;
        }
    }

    private void NativeCheck(BackendType backend, bool offload, int split)
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        if (offload) MoeCpuOffloadConfig.SetAllLayers();
        string path = _fixture.Write(Path.Combine(_dir, "half.gguf"), true);
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        float[] separateLogits;
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG", "0");
        using (var separate = new MuseSparkModel(path, backend, layerSplitDegree: split))
            separateLogits = separate.Forward(tokens[..5]).ToArray();
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG", "1");
        using var model = new MuseSparkModel(path, backend, layerSplitDegree: split);
        Assert.Equal(MuseSparkSyntheticModelBuilder.Layers, model.FusedAttentionLayerCount);
        // Half storage plus GPU matmul rounding is checked against a double
        // oracle that first rounds each matrix to the exact GGUF F16 values.
        float[] prefill = model.Forward(tokens[..5]);
        Close(_fixture.Reference(tokens[..5], true), prefill, 3e-3f);
        Close(separateLogits, prefill, 3e-3f);
        for (int end = 6; end <= tokens.Length; end++)
            Close(_fixture.Reference(tokens[..end], true), model.Forward([tokens[end - 1]]), 3e-3f);
        var snapshot = new byte[checked((int)model.ComputeKVBlockByteSize(4))];
        Assert.True(model.TryExtractKVBlock(0, 4, snapshot));
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 4, snapshot));
        Close(_fixture.Reference(tokens, true), model.Forward(tokens[4..]), 3e-3f);
    }
    private static void Close(float[] expected, float[] actual, float tolerance = 3e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        float error = expected.Zip(actual, (x, y) => MathF.Abs(x - y)).Max();
        Assert.True(actual.All(float.IsFinite));
        Assert.True(error <= tolerance, $"maximum logit error {error} > {tolerance}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FusedProjectionMatchesSeparateWithFloatAndHalfMatrices(bool half)
    {
        string path = _fixture.Write(Path.Combine(_dir, "fusion.gguf"), half);
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG", "0");
        using var separate = new MuseSparkModel(path, BackendType.Cpu);
        Assert.Equal(0, separate.FusedAttentionLayerCount);
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_FUSE_QKVG", "1");
        using var fused = new MuseSparkModel(path, BackendType.Cpu);
        Assert.Equal(MuseSparkSyntheticModelBuilder.Layers, fused.FusedAttentionLayerCount);
        Close(separate.Forward(tokens[..5]), fused.Forward(tokens[..5]));
        for (int end = 6; end <= tokens.Length; end++)
        {
            float[] actual = fused.Forward([tokens[end - 1]]);
            Close(separate.Forward([tokens[end - 1]]), actual);
            Close(_fixture.Reference(tokens[..end], half), actual);
        }
    }

    [Fact]
    public void ChunkedPrefillAndEveryDecodeMatchIndependentDoublePrecisionDecoder()
    {
        using var model = Model();
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        Close(_fixture.Reference(tokens[..5]), model.Forward(tokens[..5]));
        for (int end = 6; end <= tokens.Length; end++)
            Close(_fixture.Reference(tokens[..end]), model.Forward([tokens[end - 1]]));
        Assert.Equal(tokens.Length, model.CacheSeqLen);
        string export = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_FIXTURE_OUTPUT");
        if (!string.IsNullOrWhiteSpace(export))
        {
            _fixture.Write(export);
            File.WriteAllText(export + ".reference.json", System.Text.Json.JsonSerializer.Serialize(new
            {
                tokens, logits = Enumerable.Range(1, tokens.Length).Select(n => _fixture.Reference(tokens[..n])).ToArray()
            }));
            model.ResetKVCache();
            File.WriteAllText(export + ".managed.json", System.Text.Json.JsonSerializer.Serialize(new
            {
                tokens, logits = tokens.Select(t => model.Forward([t]).ToArray()).ToArray()
            }));
        }
    }

    [Fact]
    public void TruncationAndPrefixSnapshotRestorePreserveLogits()
    {
        using var model = Model();
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        model.Forward(tokens[..6]);
        var snapshot = new byte[checked((int)model.ComputeKVBlockByteSize(4))];
        Assert.True(model.TryExtractKVBlock(0, 4, snapshot));
        model.TruncateKVCache(4);
        Close(_fixture.Reference(tokens[..5]), model.Forward([tokens[4]]));
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 4, snapshot));
        Close(_fixture.Reference(tokens), model.Forward(tokens[4..]));
        Assert.False(model.TryInjectKVBlock(0, 4, snapshot));
    }

    [Fact]
    public void VisionRowsSurviveChunkBoundariesAndAreNormalizedWithText()
    {
        using var model = Model();
        int[] tokens = [7, 21, 42, 11, 9, 86];
        var replacements = new Dictionary<int, float[]>();
        float[] values = Enumerable.Range(0, 4 * MuseSparkSyntheticModelBuilder.Hidden).Select(x => MathF.Sin(x * 0.37f)).ToArray();
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        var embeddings = new Tensor(allocator, DType.Float32, 4, MuseSparkSyntheticModelBuilder.Hidden);
        embeddings.SetElementsAsFloat(values);
        model.SetVisionEmbeddings(embeddings, 1);
        for (int row = 0; row < 4; row++) replacements[row + 1] = values.AsSpan(row * MuseSparkSyntheticModelBuilder.Hidden, MuseSparkSyntheticModelBuilder.Hidden).ToArray();
        Close(_fixture.Reference(tokens, replacements: replacements), model.Forward(tokens));
        Assert.False(model.HasPendingVisionEmbeddings);
        using var wrong = new Tensor(allocator, DType.Float32, 1, 3);
        Assert.Throws<ArgumentException>(() => model.SetVisionEmbeddings(wrong, 0));
    }

    [Fact]
    public void RoutingBiasOnlySelectsAndTinySigmoidsRespectF16Floor()
    {
        int[] selected = new int[2]; float[] weights = new float[2];
        MuseSparkModel.Route([1, 0, -1], [0, 10, 20], 2, true, 2, selected, weights);
        Assert.Equal(new[] { 2, 1 }, selected);
        Assert.InRange(weights[0], 0.6994f, 0.6996f);
        Assert.InRange(weights.Sum(), 1.99999f, 2.00001f);
        MuseSparkModel.Route([-100, -100, -100], [0, 0, 0], 2, true, 1, selected, weights);
        Assert.Equal(new[] { 0, 1 }, selected);
        Assert.True(weights.Sum() < 1e-30f);
    }

    [Fact]
    public void PackedMixedLengthPrefillDecodeAndNewArrivalMatchIndependentSequences()
        => PackedCheck(BackendType.Cpu, 1);

    private void PackedCheck(BackendType backend, int split, Action<MuseSparkModel> beforeDispose = null)
    {
        bool half = backend != BackendType.Cpu;
        using var model = new MuseSparkModel(_fixture.Write(Path.Combine(_dir, "packed.gguf"), half), backend,
            layerSplitDegree: split);
        int[][] prompts = [[7, 21, 42, 11, 9, 86, 102, 68], [81, 53, 29, 40, 16, 12, 63, 5], [17, 2, 199, 7]];
        const int blockSize = 2;
        var pool = new BlockPool(32, blockSize, model.ComputeKVBlockByteSize(blockSize));
        var sequences = prompts.Select((tokens, index) => new SequenceState($"seq{index}", tokens, 8, blockSize, SamplingConfig.Default)).ToArray();
        foreach (var sequence in sequences)
            foreach (var block in pool.AllocateNew(4)) sequence.BlockTable.AppendBlock(block);
        void Step(int[] indices, int[] counts)
        {
            var starts = new List<int> { 0 }; var positions = new List<int>(); var slots = new List<int>();
            var tables = new int[indices.Length][];
            for (int s = 0; s < indices.Length; s++)
            {
                var sequence = sequences[indices[s]];
                tables[s] = sequence.BlockTable.Blocks.Select(b => b.Id).ToArray();
                starts.Add(starts[^1] + counts[s]);
                for (int row = sequence.NumComputedTokens; row < sequence.NumComputedTokens + counts[s]; row++)
                {
                    positions.Add(row);
                    slots.Add(tables[s][row / blockSize] * blockSize + row % blockSize);
                }
            }
            var capture = new float[positions.Count * model.Config.VocabSize];
            var context = new BatchedForwardContext
            {
                Sequences = indices.Select(i => sequences[i]).ToList(), NumScheduledTokens = counts.ToList(),
                QueryStartLoc = starts, Positions = positions, SlotMapping = slots, BlockTables = tables,
                MaxQueryLen = counts.Max(), MaxSeqLen = positions.Max() + 1, CaptureLogitsAll = capture,
            };
            var actual = model.ForwardBatch(context);
            for (int s = 0; s < indices.Length; s++)
            {
                var sequence = sequences[indices[s]];
                for (int row = 0; row < counts[s]; row++)
                    Close(_fixture.Reference(prompts[indices[s]][..(sequence.NumComputedTokens + row + 1)], half),
                        capture.AsSpan((starts[s] + row) * model.Config.VocabSize, model.Config.VocabSize).ToArray(), half ? 3e-3f : 3e-5f);
                sequence.AdvanceComputedTokens(counts[s]);
                Close(_fixture.Reference(prompts[indices[s]][..sequence.NumComputedTokens], half), actual[s], half ? 3e-3f : 3e-5f);
            }
        }
        Step([0, 1], [3, 5]);
        Step([1, 0], [1, 2]);
        Step([2, 0, 1], [4, 1, 1]);
        Assert.Equal(0, model.CacheSeqLen); // Packed state is separate from the single-stream cache.
        beforeDispose?.Invoke(model);
    }

    [Fact]
    public void DisposedModelReleasesPagedCachesWhileModelRemainsReferenced()
    {
        MuseSparkModel retained = null;
        WeakReference[] buffers = null;
        PackedCheck(BackendType.Cpu, 1, model =>
        {
            retained = model;
            buffers = new[] { "_pagedKeys", "_pagedValues" }.Select(name => new WeakReference(
                typeof(MuseSparkModel).GetField(name, System.Reflection.BindingFlags.Instance |
                    System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!)).ToArray();
        });
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        Assert.All(buffers, buffer => Assert.False(buffer.IsAlive));
        GC.KeepAlive(retained);
    }

    [Fact]
    public void SpeculativeAllRowsAndDeepRewindMatchIndependentPrefixes()
    {
        using var model = Model();
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        float[] logits = new float[tokens.Length * MuseSparkSyntheticModelBuilder.Vocab];
        float[] hidden = new float[tokens.Length * MuseSparkSyntheticModelBuilder.Hidden];
        model.SpecForward(tokens, hidden, logits, true);
        Assert.True(hidden.All(float.IsFinite));
        for (int end = 1; end <= tokens.Length; end++)
            Close(_fixture.Reference(tokens[..end]), logits.AsSpan((end - 1) * MuseSparkSyntheticModelBuilder.Vocab, MuseSparkSyntheticModelBuilder.Vocab).ToArray());
        model.SpecRewindCache(2); // Deeper than SWA window; the uniform cache retains this prefix.
        Close(_fixture.Reference(tokens[..3]), model.Forward([tokens[2]]));
        int before = model.CacheSeqLen;
        Assert.Throws<ArgumentException>(() => model.SpecForward([1, 2], null, new float[2], true));
        Assert.Throws<ArgumentException>(() => model.SpecForward([1, 2], new float[2], logits, true));
        Assert.Equal(before, model.CacheSeqLen);
        model.ResetKVCache();
        for (int i = 0; i < tokens.Length; i++)
        {
            float[] oneHidden = new float[MuseSparkSyntheticModelBuilder.Hidden];
            float[] oneLogits = new float[MuseSparkSyntheticModelBuilder.Vocab];
            model.SpecForward([tokens[i]], oneHidden, oneLogits, false);
            Close(hidden.AsSpan(i * oneHidden.Length, oneHidden.Length).ToArray(), oneHidden);
        }
    }

    [Fact]
    public void InvalidExpertGeometryIsRejectedBeforeAllocatingWeights()
    {
        _fixture.Weights.Remove("blk.0.ffn_post_exp_norm.weight");
        Assert.Throws<InvalidDataException>(() => Model());
    }
}

internal sealed class MuseSparkLayerSplitFactAttribute : FactAttribute
{
    public MuseSparkLayerSplitFactAttribute()
        => Skip = TestGates.CudaDevicesSkip(2) ?? TestGates.GgmlPinSkip(BackendType.GgmlCuda);
}
