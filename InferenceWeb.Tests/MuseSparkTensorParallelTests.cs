// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkTensorParallelTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-tp-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public MuseSparkTensorParallelTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        _environment.Set("TS_KV_INITIAL_TOKENS", "2");
        _environment.Set("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
        _environment.Set("TS_MUSE_SPARK_FUSE_QKVG", "0");
        _environment.Set("TS_MUSE_SPARK_LAYER_SPLIT", null);
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        Directory.Delete(_directory, true);
    }

    [Theory]
    [InlineData(2, false)]
    [InlineData(3, false)]
    [InlineData(5, false)]
    [InlineData(8, false)]
    [InlineData(9, false)]
    [InlineData(3, true)]
    [InlineData(9, true)]
    public void WholeAndUnevenPartitionsMatchScalarOracleAcrossChunksSnapshotsAndSpeculation(int degree, bool halfKv)
    {
        KvCacheDtypeConfig.Set(halfKv ? KvCacheDtype.F16 : KvCacheDtype.F32);
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16, experts: 36, used: 4);
        string path = fixture.Write(Path.Combine(_directory, "model.gguf"));
        using var group = new CpuGroup(degree);
        using var model = new MuseSparkModel(path, BackendType.Cpu, tpDegree: degree, tpGroup: group);
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        float tolerance = halfKv ? 2e-3f : 4e-5f;
        Close(fixture.Reference(tokens[..5]), model.Forward(tokens[..5]), tolerance);
        for (int length = 6; length <= tokens.Length; length++)
            Close(fixture.Reference(tokens[..length]), model.Forward([tokens[length - 1]]), tolerance);
        Assert.True(group.Reductions > 0);
        AssertShardedAttention(model, degree, fixture);
        var snapshot = new byte[checked((int)model.ComputeKVBlockByteSize(4))];
        Assert.True(model.TryExtractKVBlock(0, 4, snapshot));
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 4, snapshot));
        Close(fixture.Reference(tokens), model.Forward(tokens[4..]), tolerance);
        model.ResetKVCache();
        float[] logits = new float[tokens.Length * model.Config.VocabSize];
        float[] hidden = new float[tokens.Length * model.Config.HiddenSize];
        model.SpecForward(tokens, hidden, logits, true);
        for (int row = 0; row < tokens.Length; row++)
            Close(fixture.Reference(tokens[..(row + 1)]), logits.AsSpan(row * model.Config.VocabSize, model.Config.VocabSize).ToArray(), tolerance);
        model.SpecRewindCache(2);
        Close(fixture.Reference(tokens[..4]), model.Forward(tokens[2..4]), tolerance);
        Assert.All(hidden, value => Assert.True(float.IsFinite(value)));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantizedAttentionShardsAndExpertSlicesPreserveExactStoredWeights(bool cpuExperts)
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16, experts: 36, used: 2,
            headDim: 16, latent: 32, ffn: 32);
        string path = MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture, Path.Combine(_directory, "q8.gguf"));
        if (cpuExperts) MoeCpuOffloadConfig.SetLayers(1);
        int[] tokens = [7, 21, 42, 11, 9];
        float[] unsharded;
        using (var single = new MuseSparkModel(path, BackendType.Cpu))
            unsharded = single.Forward(tokens);
        using var model = new MuseSparkModel(path, BackendType.Cpu, 9, new CpuGroup(9));
        // Q8 kernels quantize activations as well as weights. Compare the exact
        // same backend arithmetic here; the independent F32 decoder checks the
        // full mathematical graph in the other cases.
        Close(unsharded, model.Forward(tokens), 1e-4f);
        var quant = Field<Dictionary<string, QuantizedWeight[]>>(model, "_tpQuantWeights", typeof(ModelBase));
        foreach (var (name, shards) in quant)
        {
            var (shape, decoded) = fixture.Weights[name];
            var rebuilt = new float[decoded.Length];
            int offset = 0;
            foreach (var shard in shards)
            {
                var local = new float[checked((int)(shard.Ne0 * shard.Ne1))];
                unsafe
                {
                    fixed (float* destination = local)
                        ManagedQuantizedOps.DequantizeRowToFloat32(shard.GgmlType, shard.Data, destination, shard.Ne0 * shard.Ne1);
                }
                if (name.EndsWith("attn_output.weight"))
                {
                    for (int row = 0; row < (int)shape[1]; row++)
                        local.AsSpan(row * (int)shard.Ne0, (int)shard.Ne0).CopyTo(rebuilt.AsSpan(row * (int)shape[0] + offset));
                    offset += (int)shard.Ne0;
                }
                else { local.CopyTo(rebuilt, offset); offset += local.Length; }
            }
            Assert.Equal(decoded, rebuilt);
        }
    }

    [Fact]
    public void PackedUnevenRanksRetainIndependentPagedSequences()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 14, kvHeads: 7, experts: 13, used: 4);
        using var model = new MuseSparkModel(fixture.Write(Path.Combine(_directory, "packed.gguf")), BackendType.Cpu, 3, new CpuGroup(3));
        int[][] prompts = [[7, 21, 42, 11, 9, 86], [81, 53, 29, 40, 16, 12], [17, 2, 199, 7]];
        const int blockSize = 2;
        var pool = new BlockPool(32, blockSize, model.ComputeKVBlockByteSize(blockSize));
        var sequences = prompts.Select((tokens, index) => new SequenceState($"tp{index}", tokens, 8, blockSize, SamplingConfig.Default)).ToArray();
        foreach (var sequence in sequences) foreach (var block in pool.AllocateNew(4)) sequence.BlockTable.AppendBlock(block);
        void Step(int[] indices, int[] counts)
        {
            var starts = new List<int> { 0 }; var positions = new List<int>(); var slots = new List<int>();
            int[][] tables = indices.Select(i => sequences[i].BlockTable.Blocks.Select(b => b.Id).ToArray()).ToArray();
            for (int s = 0; s < indices.Length; s++)
            {
                starts.Add(starts[^1] + counts[s]);
                for (int row = sequences[indices[s]].NumComputedTokens; row < sequences[indices[s]].NumComputedTokens + counts[s]; row++)
                { positions.Add(row); slots.Add(tables[s][row / blockSize] * blockSize + row % blockSize); }
            }
            var capture = new float[positions.Count * model.Config.VocabSize];
            var context = new BatchedForwardContext
            {
                Sequences = indices.Select(i => sequences[i]).ToList(), NumScheduledTokens = counts.ToList(),
                QueryStartLoc = starts, Positions = positions, SlotMapping = slots, BlockTables = tables,
                MaxQueryLen = counts.Max(), MaxSeqLen = positions.Max() + 1, CaptureLogitsAll = capture,
            };
            var output = model.ForwardBatch(context);
            for (int s = 0; s < indices.Length; s++)
            {
                var sequence = sequences[indices[s]];
                for (int row = 0; row < counts[s]; row++)
                    Close(fixture.Reference(prompts[indices[s]][..(sequence.NumComputedTokens + row + 1)]),
                        capture.AsSpan((starts[s] + row) * model.Config.VocabSize, model.Config.VocabSize).ToArray(), 4e-5f);
                sequence.AdvanceComputedTokens(counts[s]);
                Close(fixture.Reference(prompts[indices[s]][..sequence.NumComputedTokens]), output[s], 4e-5f);
            }
        }
        Step([0, 1], [3, 4]);
        Step([1, 0, 2], [1, 2, 2]);
        Step([2, 0, 1], [2, 1, 1]);
        Assert.Equal(0, model.CacheSeqLen);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(8)]
    [InlineData(9)]
    public void PublishedGeometryPartitionsCoverEveryHeadAndExpertExactlyOnce(int degree)
    {
        foreach (int total in new[] { 16, 256 })
        {
            var parts = Enumerable.Range(0, degree).Select(r => MuseSparkModel.SparkPartition(total, degree, r)).ToArray();
            Assert.Equal(0, parts[0].First);
            Assert.Equal(total, parts.Sum(p => p.Count));
            for (int rank = 1; rank < degree; rank++) Assert.Equal(parts[rank - 1].First + parts[rank - 1].Count, parts[rank].First);
            Assert.InRange(parts.Max(p => p.Count) - parts.Min(p => p.Count), 0, 1);
        }
    }

    [Fact]
    public void RouteCompactionAvoidsForeignWorkAndDuplicateZeroFillers()
    {
        int[] selected = [0, 4, 7, 8, 5, 4, 1, 8, 0, 1, 8, 9];
        float[] weights = Enumerable.Range(1, selected.Length).Select(i => i / 20f).ToArray();
        var local = MuseSparkModel.SparkCompactRoutes(selected, weights, 3, 4, 4, 4);
        Assert.Equal(2, local.Used);
        Assert.Equal(new[] { 0, 3, 1, 0, 0, 1 }, local.Selected);
        Assert.Equal(new[] { weights[1], weights[2], weights[4], weights[5], 0f, 0f }, local.Weights);
        for (int row = 0; row < 3; row++) Assert.Equal(2, local.Selected.AsSpan(row * 2, 2).ToArray().Distinct().Count());
        var unused = MuseSparkModel.SparkCompactRoutes(selected, weights, 3, 4, 12, 4);
        Assert.Equal(0, unused.Used); Assert.Empty(unused.Selected); Assert.Empty(unused.Weights);
        Assert.Throws<ArgumentException>(() => MuseSparkModel.SparkCompactRoutes([4, 4], [1, 1], 1, 2, 4, 4));
    }

    [Fact]
    public void DescriptorCapacityCountsUnevenShardsAndRankZeroTrunk()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 14, kvHeads: 7, experts: 13, used: 2);
        using var model = new MuseSparkModel(fixture.Write(Path.Combine(_directory, "capacity.gguf")), BackendType.Cpu, 3, new CpuGroup(3));
        long[] expected = new long[3];
        foreach (var (name, (shape, data)) in fixture.Weights)
        {
            long bytes = data.LongLength * sizeof(float);
            if (name.Contains("_exps."))
                for (int rank = 0; rank < 3; rank++) expected[rank] += bytes / 13 * (rank == 0 ? 5 : 4);
            else if (new[] { "attn_q.", "attn_k.", "attn_v.", "attn_gate.", "attn_output." }.Any(name.Contains))
                for (int rank = 0; rank < 3; rank++) expected[rank] += bytes / 7 * (rank == 0 ? 3 : 2);
            else if (name.Contains("attn_q_norm.") || name.Contains("attn_k_norm."))
                for (int rank = 0; rank < 3; rank++) expected[rank] += bytes;
            else expected[0] += bytes;
        }
        Assert.Equal(expected, model.TensorParallelPlannedWeightBytes);
        Assert.True(expected[0] > expected[1]); Assert.Equal(expected[1], expected[2]);
    }

    [Fact]
    public void RejectsMoreRanksThanKvGroupsBeforeWeightMaterialization()
    {
        var fixture = new MuseSparkSyntheticModelBuilder();
        string path = fixture.Write(Path.Combine(_directory, "invalid.gguf"));
        var error = Assert.Throws<NotSupportedException>(() => new MuseSparkModel(path, BackendType.Cpu, 3, new CpuGroup(3)));
        Assert.Contains("KV head", error.Message);
    }

    [Fact]
    public void MixedCheckpointKvDefaultAndOwnedViewsSurviveTensorSharding()
    {
        var fixture = new MuseSparkSyntheticModelBuilder();
        string path = fixture.WriteMixedHalf(Path.Combine(_directory, "mixed.gguf"),
            name => name.Contains("attn_") && !name.Contains("_norm.") && !name.Contains("_resid_"));
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, false);
        using (var single = new MuseSparkModel(path, BackendType.Cpu))
            Assert.Equal(KvCacheDtype.F16, single.KvCacheDtype);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, false);
        var model = new MuseSparkModel(path, BackendType.Cpu, 2, new CpuGroup(2));
        try
        {
            Assert.Equal(KvCacheDtype.F16, model.KvCacheDtype);
            Assert.Empty(Field<Dictionary<string, QuantizedWeight>>(model, "_quantWeights", typeof(ModelBase)));
            var views = Field<Dictionary<string, QuantizedWeight[]>>(model, "_tpQuantWeights", typeof(ModelBase)).Values.SelectMany(x => x).ToArray();
            var owners = Field<List<QuantizedWeight>>(model, "_sparkTpColumnOwners").ToArray();
            Assert.NotEmpty(views); Assert.NotEmpty(owners);
            float[] before = model.Forward([7, 21, 42]);
            model.ReleaseGgmlDeviceResidency(); // CPU no-op preserves the same public contract.
            model.ResetKVCache();
            Close(before, model.Forward([7, 21, 42]), 0f);
            model.Dispose();
            Assert.All(views, view => Assert.Equal(IntPtr.Zero, view.Data));
            Assert.All(owners, owner => Assert.Equal(IntPtr.Zero, owner.Data));
            Assert.Empty(Field<Dictionary<string, QuantizedWeight[]>>(model, "_tpQuantWeights", typeof(ModelBase)));
            Assert.Empty(Field<List<QuantizedWeight>>(model, "_sparkTpColumnOwners"));
        }
        finally { model.Dispose(); }
    }

    private static void AssertShardedAttention(MuseSparkModel model, int degree, MuseSparkSyntheticModelBuilder fixture)
    {
        var shards = Field<Dictionary<string, Tensor[]>>(model, "_tpWeights", typeof(ModelBase));
        Assert.Equal(MuseSparkSyntheticModelBuilder.Layers * 5, shards.Count);
        foreach (var (name, ranks) in shards)
        {
            Assert.Equal(degree, ranks.Length);
            var original = fixture.Weights[name];
            int axis = name.EndsWith("attn_output.weight") ? 1 : 0;
            Assert.Equal((long)original.Shape[1 - axis], ranks.Sum(t => t.Sizes[axis]));
            Assert.All(ranks, t => Assert.True(t.Sizes[axis] < (long)original.Shape[1 - axis]));
        }
    }

    private static T Field<T>(object model, string name, Type type = null)
        => (T)(type ?? model.GetType()).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static void Close(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(expected[i] - actual[i]) <= tolerance,
                $"Logit {i}: expected {expected[i]:R}, actual {actual[i]:R}, tolerance {tolerance:R}");
    }

    internal sealed class CpuGroup(int degree) : ITensorParallelGroup
    {
        private readonly IAllocator[] _allocators = Enumerable.Range(0, degree).Select(_ => (IAllocator)new CpuAllocator(BlasEnum.DotNet)).ToArray();
        public int Degree => degree;
        public bool IsActive => degree > 1;
        public int GlobalDegree => degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public int Reductions { get; private set; }
        public IAllocator GetAllocator(int rank) => _allocators[rank];
        public void RunPerRank(Action<int> body) => Parallel.For(0, degree, body);
        public void AllReduce(Tensor[] tensors)
        {
            var sum = new float[checked((int)tensors[0].ElementCount())];
            var local = new float[sum.Length];
            foreach (var tensor in tensors)
            {
                tensor.CopyToArray(local);
                for (int i = 0; i < sum.Length; i++) sum[i] += local[i];
            }
            foreach (var tensor in tensors) tensor.CopyFrom(sum);
            Reductions++;
        }
        public void Dispose() { }
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
    }
}
