// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Actual CUDA rank execution, including uneven partitions, quantized expert
/// slices and quantized attention input-column slices. Missing devices are
/// reported as skipped. Quantized weights use an actual same-backend oracle;
/// F32 weights/F32 KV separately exercise the independent scalar decoder.
/// </summary>
[Collection("Muse Spark model")]
public sealed class MuseSparkTensorParallelGpuTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-tp-gpu-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly ITestOutputHelper _output;

    public MuseSparkTensorParallelGpuTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        _environment.Set("TS_KV_INITIAL_TOKENS", "2");
        _environment.Set("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
        _environment.Set("TS_MUSE_SPARK_STREAM_EXPERTS", "0");
        _environment.Set("TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES", "1");
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

    [MuseSparkTensorParallelGpuFact(2)]
    public void TwoRanksMatchNativeQuantizedOracleAcrossCacheModes() => CheckRanks(2);

    [MuseSparkTensorParallelGpuFact(3)]
    public void ThreeUnevenRanksMatchNativeQuantizedOracleAcrossCacheModes() => CheckRanks(3);

    [MuseSparkTensorParallelGpuFact(5)]
    public void FiveUnevenRanksMatchNativeQuantizedOracleAcrossCacheModes() => CheckRanks(5);

    [MuseSparkTensorParallelGpuFact(8)]
    public void EightRanksMatchNativeQuantizedOracleAcrossCacheModes() => CheckRanks(8);

    [MuseSparkTensorParallelGpuFact(9)]
    public void NineUnevenRanksMatchNativeQuantizedOracleAcrossCacheModes() => CheckRanks(9);

    [MuseSparkOptimizedTensorParallelGpuFact]
    public void NineRanksOptimizedProfileMatchesNativeQuantizedOracleAcrossCacheModes()
    {
        // The native CPU arena flag is latched at library initialization. The
        // fact gate therefore requires it in the fresh testhost environment;
        // setting it only after discovery would not prove the selected path.
        Assert.Equal("1", Environment.GetEnvironmentVariable("TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH"));
        _environment.Set("TS_MUSE_SPARK_REUSE_NORM_SCRATCH", "1");
        _environment.Set("TS_MUSE_SPARK_KV_UPLOAD_2D", "1");
        _environment.Set("TS_MUSE_SPARK_FUSE_RESIDUALS", "1");
        _environment.Set("TS_MUSE_SPARK_FUSE_QK_PREP", "0");
        CheckRanks(9, "optimized-");
    }

    [MuseSparkTensorParallelGpuFact(2)]
    public void TwoRanksFloatWeightsMatchIndependentScalarOracle() => CheckFloatRanks(2);

    [MuseSparkTensorParallelGpuFact(3)]
    public void ThreeUnevenRanksFloatWeightsMatchIndependentScalarOracle() => CheckFloatRanks(3);

    [MuseSparkTensorParallelGpuFact(5)]
    public void FiveUnevenRanksFloatWeightsMatchIndependentScalarOracle() => CheckFloatRanks(5);

    [MuseSparkTensorParallelGpuFact(8)]
    public void EightRanksFloatWeightsMatchIndependentScalarOracle() => CheckFloatRanks(8);

    [MuseSparkTensorParallelGpuFact(9)]
    public void NineUnevenRanksFloatWeightsMatchIndependentScalarOracle() => CheckFloatRanks(9);

    [MuseSparkTensorParallelGpuFact(3)]
    public void VisionEmbeddingRowsCrossTpPrefillChunks()
    {
        var fixture = Fixture();
        string path = WriteQ8Fixture(fixture, Path.Combine(_directory, "vision-rows.gguf"));
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        var replacements = new Dictionary<int, float[]>();
        float[] values = Enumerable.Range(0, 4 * MuseSparkSyntheticModelBuilder.Hidden)
            .Select(index => MathF.Sin(index * 0.37f)).ToArray();
        for (int row = 0; row < 4; row++)
            replacements[row + 1] = values.AsSpan(row * MuseSparkSyntheticModelBuilder.Hidden,
                MuseSparkSyntheticModelBuilder.Hidden).ToArray();
        float[][] Run(string checkpoint, int degree)
        {
            using var model = Assert.IsType<MuseSparkModel>(ModelBase.Create(checkpoint, BackendType.GgmlCuda, tpDegree: degree));
            var embeddings = new Tensor(new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet),
                DType.Float32, 4, MuseSparkSyntheticModelBuilder.Hidden);
            embeddings.SetElementsAsFloat(values);
            model.SetVisionEmbeddings(embeddings, 1);
            float[] first = model.Forward(tokens[..6]).ToArray();
            Assert.False(model.HasPendingVisionEmbeddings);
            return [first, model.Forward(tokens[6..]).ToArray()];
        }
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        float[][] native = Run(path, 1), parallel = Run(path, 3);
        var checks = new MuseSparkGpuChecks(_output);
        for (int row = 0; row < native.Length; row++)
            checks.Compare($"Q8 vision row {row}, TP/native", native[row], parallel[row], 1e-4, 1e-4);

        // Stored F32 weights remove activation quantization from the independent
        // oracle. Keep the same image injection and chunk-boundary schedule.
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        float[][] unquantized = Run(fixture.Write(Path.Combine(_directory, "vision-f32.gguf")), 3);
        checks.Compare("F32 vision prefix/scalar", fixture.Reference(tokens[..6], replacements: replacements), unquantized[0], 3e-5, 3e-5);
        checks.Compare("F32 vision continuation/scalar", fixture.Reference(tokens, replacements: replacements), unquantized[1], 3e-5, 3e-5);
        checks.Complete("vision");
    }

    [MuseSparkTensorParallelGpuFact(2)]
    public void LargeCollectivePreservesFloat32ValuesBelowBfloat16Resolution()
        => CheckLargeCollective(2);

    [MuseSparkTensorParallelGpuFact(9)]
    public void NineRankLargeCollectivePreservesFloat32ValuesBelowBfloat16Resolution()
        => CheckLargeCollective(9);

    private static void CheckLargeCollective(int degree)
    {
        const int count = 1_048_580; // Large prefill payload, with a short final chunk.
        Assert.InRange(GgmlTensorParallelGroup.DeviceAllReduceThreshold, 0, count);
        var context = new GgmlContext(Enumerable.Range(0, degree).ToArray(), GgmlBackendType.Cuda);
        using var group = new GgmlTensorParallelGroup(context);
        var tensors = new Tensor[degree];
        try
        {
            Assert.True(context.HasDeviceAllReduce, "This test requires a working device collective.");
            float[] expected = new float[count];
            for (int rank = 0; rank < degree; rank++)
            {
                float[] values = new float[count];
                for (int index = 0; index < count; index++)
                {
                    // These dyadic values sum exactly in any F32 reduction
                    // order at the tested degrees, but lose their small terms
                    // if the transport silently narrows partials to BF16.
                    values[index] = rank % 2 == 0
                        ? 1f + ((index + rank) % 7 + 1) / 4096f
                        : -1f + ((index + rank) % 5) / 8192f;
                    expected[index] += values[index];
                }
                tensors[rank] = new Tensor(group.GetAllocator(rank), DType.Float32, count);
                tensors[rank].SetElementsAsFloat(values);
            }
            group.AllReduceFloat32(tensors);
            foreach (var tensor in tensors)
                Assert.Equal(expected, TensorToArray(tensor));
        }
        finally
        {
            foreach (var tensor in tensors) tensor?.Dispose();
            group.Dispose();
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
            context.ReleasePoolForDisposal();
        }
    }

    private static float[] TensorToArray(Tensor tensor)
    {
        var values = new float[checked((int)tensor.ElementCount())];
        tensor.CopyToArray(values);
        return values;
    }

    private void CheckRanks(int degree, string evidencePrefix = "")
    {
        // Each KV group spans exactly one Q8 block in the output projection.
        // Native activation quantization is part of the same-backend control;
        // a scalar float-activation decoder is not an exact Q8 kernel oracle.
        var fixture = Fixture();
        string path = WriteQ8Fixture(fixture, Path.Combine(_directory, "quantized.gguf"));
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        var checks = new MuseSparkGpuChecks(_output);
        foreach (KvCacheDtype dtype in new[] { KvCacheDtype.F32, KvCacheDtype.F16 })
        foreach (bool cpuExperts in new[] { false, true })
        {
            KvCacheDtypeConfig.Set(dtype);
            if (cpuExperts) MoeCpuOffloadConfig.SetLayers(1); else MoeCpuOffloadConfig.Reset();
            ScenarioRow[] native = RunCacheScenario(path, 1, tokens, requireAdvice: false);
            ScenarioRow[] parallel = RunCacheScenario(path, degree, tokens, requireAdvice: true);
            Assert.Equal(native.Select(row => (row.Label, row.PrefixLength)), parallel.Select(row => (row.Label, row.PrefixLength)));
            for (int row = 0; row < native.Length; row++)
            {
                string label = $"degree={degree}, KV={dtype}, CPU experts={cpuExperts}, {native[row].Label}";
                checks.Compare(label + " TP/native", native[row].Values, parallel[row].Values, 1e-4, 1e-4);
                // Preserve the original scalar verdict as diagnostic evidence.
                // It does not gate correctness of activation-quantized kernels.
                checks.Compare(label + " native/scalar diagnostic", fixture.Reference(tokens[..native[row].PrefixLength]),
                    native[row].Values, 0.006, 0.002, required: false);
            }
        }
        checks.Complete($"{evidencePrefix}quantized-cache-ranks-{degree}");
    }

    private void CheckFloatRanks(int degree)
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        var fixture = Fixture();
        string path = fixture.Write(Path.Combine(_directory, "float.gguf"));
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        ScenarioRow[] parallel = RunCacheScenario(path, degree, tokens, requireAdvice: false);
        var checks = new MuseSparkGpuChecks(_output);
        foreach (var row in parallel)
            checks.Compare($"degree={degree}, F32 {row.Label}/scalar", fixture.Reference(tokens[..row.PrefixLength]), row.Values, 3e-5, 3e-5);
        checks.Complete($"float-cache-ranks-{degree}");
    }

    [MuseSparkTensorParallelGpuFact(2)]
    public void FloatExpertPlaneResidencyReleasesAndReloadsAcrossPlacementModes()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        var fixture = Fixture();
        string path = fixture.Write(Path.Combine(_directory, "float-expert-residency.gguf"));
        int[] tokens = [7, 21, 42, 11, 9];
        var checks = new MuseSparkGpuChecks(_output);
        foreach (var (degree, split) in new[] { (1, 1), (2, 1), (1, 2) })
        {
            using var model = new MuseSparkModel(path, BackendType.GgmlCuda, tpDegree: degree, layerSplitDegree: split);
            var weights = (Dictionary<string, Tensor>)typeof(ModelBase)
                .GetField("_weights", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                .GetValue(model)!;
            var keys = new List<(string Name, long Expert, IntPtr Key, long Bytes)>();
            foreach (var (name, tensor) in weights)
                if (tensor.DimensionCount == 3 && name.Contains("_exps.", StringComparison.Ordinal))
                    for (long expert = 0; expert < tensor.Sizes[0]; expert++)
                        keys.Add((name, expert, tensor.Storage.PtrAtElement(
                            tensor.StorageOffset + expert * tensor.Strides[0]),
                            tensor.Sizes[1] * tensor.Sizes[2] * sizeof(float)));

            HashSet<(int Rank, IntPtr Key)> Resident()
            {
                var found = new HashSet<(int, IntPtr)>();
                int priorRank = GgmlBasicOps.GetActiveRank();
                bool priorPin = GgmlBasicOps.PinRank(0);
                try
                {
                    for (int rank = 0; rank < Math.Max(degree, split); rank++)
                    {
                        GgmlBasicOps.SetActiveRank(rank);
                        foreach (var key in keys)
                            if (GgmlBasicOps.HasResidentDeviceCopy(key.Key, key.Bytes)) found.Add((rank, key.Key));
                    }
                }
                finally { GgmlBasicOps.RestoreRankPin(priorPin, priorRank); }
                return found;
            }

            float[] before = model.Forward(tokens);
            var residentBefore = Resident();
            Assert.Contains(keys, key => key.Expert > 0 && residentBefore.Any(item => item.Key == key.Key));
            model.ReleaseGgmlDeviceResidency();
            Assert.Empty(Resident());
            model.ResetKVCache();
            float[] after = model.Forward(tokens);
            Assert.True(residentBefore.SetEquals(Resident()), "The same F32 expert planes did not reload on their original ranks.");
            _output.WriteLine($"TP={degree}, layer split={split}: {residentBefore.Count} actual expert-plane device copies released and reloaded.");
            checks.Compare($"TP={degree}, split={split}, release/reload", before, after, 3e-5, 3e-5);
            checks.Compare($"TP={degree}, split={split}, scalar", fixture.Reference(tokens), after, 3e-5, 3e-5);
        }
        checks.Complete("float-expert-plane-residency");
    }

    private static ScenarioRow[] RunCacheScenario(string path, int degree, int[] tokens, bool requireAdvice)
    {
        using var model = Assert.IsType<MuseSparkModel>(ModelBase.Create(path, BackendType.GgmlCuda, tpDegree: degree));
        if (requireAdvice && OperatingSystem.IsLinux())
            Assert.True(model.AdvisedGpuSourceBytes > 0, "Resident TP shards did not release any original mapped source pages.");
        var rows = new List<ScenarioRow>();
        void Capture(string label, int length, float[] values) => rows.Add(new ScenarioRow(label, length, values.ToArray()));
        Capture("prefill", 5, model.Forward(tokens[..5]));
        for (int end = 6; end <= tokens.Length; end++)
            Capture($"decode-{end}", end, model.Forward([tokens[end - 1]]));

        var snapshot = new byte[checked((int)model.ComputeKVBlockByteSize(4))];
        Assert.True(model.TryExtractKVBlock(0, 4, snapshot));
        model.TruncateKVCache(4);
        Capture("truncate", 5, model.Forward([tokens[4]]));
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 4, snapshot));
        Capture("restore", tokens.Length, model.Forward(tokens[4..]));
        Assert.Equal(tokens.Length, model.CacheSeqLen);

        model.ResetKVCache();
        float[] logits = new float[tokens.Length * model.Config.VocabSize];
        float[] hidden = new float[tokens.Length * model.Config.HiddenSize];
        model.SpecForward(tokens, hidden, logits, allLogitsRows: true);
        for (int row = 0; row < tokens.Length; row++)
            Capture($"spec-{row}", row + 1, logits.AsSpan(row * model.Config.VocabSize, model.Config.VocabSize).ToArray());
        Assert.All(hidden, value => Assert.True(float.IsFinite(value)));
        model.SpecRewindCache(4);
        float[] beforeRelease = model.Forward(tokens[4..6]);
        Capture("rewind", 6, beforeRelease);
        model.TruncateKVCache(4);
        model.ReleaseGgmlDeviceResidency();
        float[] afterRelease = model.Forward(tokens[4..6]);
        Capture("release", 6, afterRelease);
        Assert.All(afterRelease, value => Assert.True(float.IsFinite(value)));
        Assert.True(beforeRelease.Zip(afterRelease, (a, b) => MathF.Abs(a - b)).Max() < 5e-5f,
            "Releasing device residency changed the same cached-prefix continuation.");
        MuseSparkGpuSchedules.AssertQKPreparationUsed(model, degree);
        if (KvCacheDtypeConfig.Current == KvCacheDtype.F32)
            Assert.True(model.OwnedF32AttentionCalls >= degree * model.Config.NumLayers,
                "F32 cache scenario did not confirm owned CUDA attention.");
        else
            Assert.Equal(0, model.OwnedF32AttentionCalls);
        Assert.Equal(0, model.OwnedF32PagedAttentionCalls);
        return rows.ToArray();
    }

    private sealed record ScenarioRow(string Label, int PrefixLength, float[] Values);

    [MuseSparkTensorParallelGpuFact(5)]
    public void QuantizedPackedPrefillDecodeAndNewArrivalMatchSameScheduleOnOneGpu()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
        var fixture = Fixture();
        string path = WriteQ8Fixture(fixture, Path.Combine(_directory, "packed.gguf"));
        int[][] prompts = [[7, 21, 42, 11, 9, 86, 102, 68], [81, 53, 29, 40, 16, 12, 63, 5], [17, 2, 199, 7]];
        (int[], int[])[] schedule = [([0, 1], [3, 5]), ([1, 0], [1, 2]), ([2, 0, 1], [4, 1, 1])];
        MuseSparkGpuRow[] native = MuseSparkGpuSchedules.Packed(path, 1, prompts, 2, schedule, requireOwnedAttention: false);
        MuseSparkGpuRow[] parallel = MuseSparkGpuSchedules.Packed(path, 5, prompts, 2, schedule, requireOwnedAttention: false);
        string f32Path = fixture.Write(Path.Combine(_directory, "packed-f32.gguf"));
        var returned = new List<MuseSparkGpuRow>();
        // Full and gathered heads have different GEMM row counts. Require both
        // to match the independent F32 oracle, not bit identity across kernels.
        MuseSparkGpuRow[] unquantized = MuseSparkGpuSchedules.Packed(f32Path, 5, prompts, 2, schedule, requireOwnedAttention: false,
            captureBatch: (_, batch) => returned.AddRange(MuseSparkGpuSchedules.ReturnedRows(batch)), requireExactHeadRows: false);
        var checks = new MuseSparkGpuChecks(_output);
        Assert.Equal(native.Select(row => (row.Sequence, row.Position)), parallel.Select(row => (row.Sequence, row.Position)));
        Assert.Equal(native.Select(row => (row.Sequence, row.Position)), unquantized.Select(row => (row.Sequence, row.Position)));
        for (int row = 0; row < native.Length; row++)
        {
            var position = native[row];
            float[] scalar = fixture.Reference(prompts[position.Sequence][..(position.Position + 1)]);
            checks.Compare($"packed seq{position.Sequence} row{position.Position} Q8 TP/native", position.Values, parallel[row].Values, 1e-4, 1e-4);
            checks.Compare($"packed seq{position.Sequence} row{position.Position} Q8/scalar diagnostic", scalar, parallel[row].Values, 0.006, 0.002, required: false);
            checks.Compare($"packed seq{position.Sequence} row{position.Position} F32/scalar", scalar, unquantized[row].Values, 3e-5, 3e-5);
        }
        Assert.Equal(schedule.Sum(batch => batch.Item1.Length), returned.Count);
        foreach (var row in returned)
            checks.Compare($"returned seq{row.Sequence} row{row.Position} F32/scalar",
                fixture.Reference(prompts[row.Sequence][..(row.Position + 1)]), row.Values, 3e-5, 3e-5);
        checks.Complete("packed-ranks-5");
    }

    private static MuseSparkSyntheticModelBuilder Fixture()
        => new(heads: 32, kvHeads: 16, experts: 36, used: 2, headDim: 16, latent: 32, ffn: 32);

    internal static string WriteQ8Fixture(MuseSparkSyntheticModelBuilder fixture, string path)
    {
        var metadata = fixture.CreateMetadata().Select(pair =>
        {
            SyntheticGguf.Kv value = pair.Value switch
            {
                uint number => new SyntheticGguf.U32 { V = number },
                float number => new SyntheticGguf.F32 { V = number },
                bool flag => new SyntheticGguf.Bool { V = flag },
                string text => new SyntheticGguf.Str { V = text },
                string[] array => new SyntheticGguf.StrArr { V = array },
                int[] array => new SyntheticGguf.I32Arr { V = array },
                bool[] array => new BoolArray { Values = array },
                _ => throw new InvalidOperationException($"Unsupported fixture metadata: {pair.Key}"),
            };
            value.Key = pair.Key;
            return value;
        }).ToList();
        var tensors = new List<SyntheticGguf.Tensor>();
        foreach (var (name, (shape, data)) in fixture.Weights)
        {
            var tensor = new SyntheticGguf.Tensor { Name = name, Dims = shape, Data = data };
            if (shape.Length > 1)
            {
                Assert.Equal(0UL, shape[0] % 32);
                tensor.Type = SyntheticGguf.GgmlType.Q8_0;
                tensor.RawBlocks = SyntheticGguf.QuantizeQ8_0(data, (int)shape[0]);
                // Give the independent scalar oracle exactly the represented
                // values; it does not call TensorSharp's dequantizer or ops.
                for (int block = 0; block < data.Length / 32; block++)
                {
                    float scale = (float)BitConverter.UInt16BitsToHalf(
                        BitConverter.ToUInt16(tensor.RawBlocks, block * 34));
                    for (int j = 0; j < 32; j++)
                        data[block * 32 + j] = scale * unchecked((sbyte)tensor.RawBlocks[block * 34 + 2 + j]);
                }
            }
            tensors.Add(tensor);
        }
        SyntheticGguf.Write(path, metadata, tensors);
        return path;
    }

    private sealed class BoolArray : SyntheticGguf.Kv
    {
        internal bool[] Values;
        public override void Write(BinaryWriter writer)
        {
            writer.Write(9u); writer.Write(7u); writer.Write((ulong)Values.Length);
            foreach (bool value in Values) writer.Write(value);
        }
    }


}

internal sealed class MuseSparkTensorParallelGpuFactAttribute : FactAttribute
{
    public MuseSparkTensorParallelGpuFactAttribute(int devices)
        => Skip = TestGates.CudaDevicesSkip(devices) ?? TestGates.GgmlPinSkip(BackendType.GgmlCuda);
}

internal sealed class MuseSparkOptimizedTensorParallelGpuFactAttribute : FactAttribute
{
    public MuseSparkOptimizedTensorParallelGpuFactAttribute()
        => Skip = TestGates.CudaDevicesSkip(9) ?? TestGates.GgmlPinSkip(BackendType.GgmlCuda) ??
            (Environment.GetEnvironmentVariable("TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH") == "1" ? null :
                "Requires a fresh testhost with TS_MUSE_SPARK_REUSE_CPU_MOE_SCRATCH=1 before native library initialization.");
}
