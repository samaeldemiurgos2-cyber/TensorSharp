// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkExpertPlacementTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-expert-placement-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly ITestOutputHelper _output;

    public MuseSparkExpertPlacementTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        _environment.Set("TS_KV_INITIAL_TOKENS", "2");
        _environment.Set("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
        _environment.Set("TS_MUSE_SPARK_SCRATCH_RESERVE_MIB", "16");
        _environment.Set("TS_MUSE_SPARK_FUSE_QKVG", "0");
        _environment.Set("TS_MUSE_SPARK_STREAM_EXPERTS", "0");
        _environment.Set("TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES", "1");
        _environment.Set("TS_MUSE_SPARK_LAYER_SPLIT", null);
        MoeCpuOffloadConfig.Reset();
    }

    [Fact]
    public void UniformFitIsPreservedDespiteDifferentFreeMemory()
    {
        var plan = MuseSparkModel.PlanSparkExperts(13, [700, 100, 100], 100, [1400, 2000, 9000], 100);
        Assert.Equal(new[] { 5, 4, 4 }, plan.Counts);
        Assert.Equal("uniform (fits measured capacity)", plan.Policy);
        Assert.Equal(new long[] { 1200, 500, 500 }, plan.WeightBytes);
    }

    [Fact]
    public void AsymmetricSaturatedCapacityUsesEveryExpertExactlyOnce()
    {
        var plan = MuseSparkModel.PlanSparkExperts(13, [700, 400, 100], 100, [1000, 1000, 800], 100);
        Assert.Equal(new[] { 2, 5, 6 }, plan.Counts);
        Assert.Equal("capacity-balanced", plan.Policy);
        Assert.Equal(13, plan.Counts.Sum());
        Assert.All(plan.Counts, count => Assert.True(count > 0));
        Assert.Null(MuseSparkModel.CapacityRefusal(plan.WeightBytes.ToArray(), [1000, 1000, 800], 100));
    }

    [Fact]
    public void BalancedPlanMaximizesMinimumRemainingBytesAgainstExhaustiveOracle()
    {
        int tested = 0;
        const int experts = 9;
        foreach (long firstFree in new long[] { 200, 250, 300 })
        foreach (long secondFree in new long[] { 400, 550, 700 })
        foreach (long thirdFree in new long[] { 500, 650, 800 })
        {
            long[] fixedBytes = [100, 50, 0], free = [firstFree, secondFree, thirdFree];
            var candidates = new List<long>();
            for (int a = 1; a < experts - 1; a++)
            for (int b = 1; b < experts - a; b++)
            {
                int[] counts = [a, b, experts - a - b];
                long minimum = Enumerable.Range(0, 3).Min(rank => free[rank] - fixedBytes[rank] - 50 - 100 * counts[rank]);
                if (minimum >= 0) candidates.Add(minimum);
            }
            if (candidates.Count == 0)
            {
                Assert.Throws<ModelLoadRefusedException>(() => MuseSparkModel.PlanSparkExperts(experts, fixedBytes, 100, free, 50));
                continue;
            }
            var plan = MuseSparkModel.PlanSparkExperts(experts, fixedBytes, 100, free, 50);
            Assert.Equal("capacity-balanced", plan.Policy); // Rank 0 cannot hold its uniform three experts.
            long actual = Enumerable.Range(0, 3).Min(rank => free[rank] - plan.WeightBytes[rank] - 50);
            Assert.Equal(candidates.Max(), actual);
            Assert.Equal(plan.Counts, MuseSparkModel.PlanSparkExperts(experts, fixedBytes, 100, free, 50).Counts);
            tested++;
        }
        Assert.True(tested > 10);
    }

    [Fact]
    public void UnknownRankFallsBackUniformlyAndPreservesKnownRankRefusal()
    {
        var plan = MuseSparkModel.PlanSparkExperts(13, [700, 400, 100], 100, [1000, -1, 800], 100);
        Assert.Equal(new[] { 5, 4, 4 }, plan.Counts);
        Assert.Contains("unavailable", plan.Policy);
        Assert.Contains("GPU rank 0", MuseSparkModel.CapacityRefusal(plan.WeightBytes.ToArray(), [1000, -1, 800], 100));
    }

    [Fact]
    public void InsufficientCapacityRefusesWithoutReturningPartialOwnership()
    {
        Assert.Throws<ModelLoadRefusedException>(() => MuseSparkModel.PlanSparkExperts(13, [700, 400, 100], 100, [999, 1000, 800], 100));
        Assert.Throws<ModelLoadRefusedException>(() => MuseSparkModel.PlanSparkExperts(13, [700, 400, 100], 100, [799, 5000, 5000], 100));
        Assert.Throws<ArgumentException>(() => MuseSparkModel.PlanSparkExperts(13, [1, 2], 1, [-2, 10], 0));
        Assert.Throws<ArgumentException>(() => MuseSparkModel.PlanSparkExperts(1, [1, 2], 1, [10, 10], 0));
    }

    [Fact]
    public void NoGpuExpertsUsesUniformPlanAndDoesNotInventExpertBytes()
    {
        var plan = MuseSparkModel.PlanSparkExperts(13, [700, 400, 100], 0, [1000, 1000, 1000], 100);
        Assert.Equal(new[] { 5, 4, 4 }, plan.Counts);
        Assert.Equal(new long[] { 700, 400, 100 }, plan.WeightBytes);
        Assert.Contains("no partitioned", plan.Policy);
    }

    [Fact]
    public void PlanCopiesInputsAndExposesReadOnlyCounts()
    {
        long[] fixedBytes = [700, 400, 100], free = [1000, 1000, 800];
        var plan = MuseSparkModel.PlanSparkExperts(13, fixedBytes, 100, free, 100);
        fixedBytes[0] = free[0] = 0;
        Assert.Equal(700, plan.FixedBytes[0]);
        Assert.Equal(900, plan.WeightBytes[0]);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)plan.Counts)[0] = 1);
        Assert.Throws<OverflowException>(() => MuseSparkModel.PlanSparkExperts(2, [long.MaxValue], 1, [-1], 0));
    }

    [Fact]
    public void PublishedQ3CpuEightSnapshotKeepsRankZeroHeadroom()
    {
        // Actual Q3 descriptor sums and the recorded initial nine-A40 free
        // memory snapshot. Runtime must still requery devices before loading.
        long[] fixedBytes = [10270471168, 1795980288, 1795980288, 1795980288, 1795980288,
            1795980288, 1795980288, 898006016, 898006016];
        long[] free = Enumerable.Repeat(45490L << 20, 9).ToArray();
        free[7] = 48544L << 20;
        const long expertBytes = 1288175616, reserve = 2560L << 20;
        var plan = MuseSparkModel.PlanSparkExperts(256, fixedBytes, expertBytes, free, reserve);
        Assert.Equal(new[] { 22, 29, 29, 29, 29, 29, 28, 32, 29 }, plan.Counts);
        Assert.Null(MuseSparkModel.CapacityRefusal(plan.WeightBytes.ToArray(), free, reserve));
        long fullKvRankZero = 62L * 2 * 64 * 2 * 2 * 16384;
        long rankZeroAfterAdditionalVisionAndKv = free[0] - plan.WeightBytes[0] - reserve - (2560L << 20) - fullKvRankZero;
        Assert.True(rankZeroAfterAdditionalVisionAndKv > (3000L << 20));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void AsymmetricCpuPlacementMatchesOracleAndPreservesOffload(bool quantized, bool cpuExperts)
        => CheckModel(BackendType.Cpu, quantized, cpuExperts, KvCacheDtype.F32);

    [MuseSparkTensorParallelGpuFact(3)]
    public void AsymmetricGpuFloatPlacementMatchesIndependentScalar()
    {
        CheckModel(BackendType.GgmlCuda, false, false, KvCacheDtype.F32);
        CheckModel(BackendType.GgmlCuda, false, true, KvCacheDtype.F32);
    }

    [MuseSparkTensorParallelGpuFact(3)]
    public void AsymmetricGpuQuantizedPlacementMatchesNativeAcrossOffloadAndKvModes()
    {
        foreach (var dtype in new[] { KvCacheDtype.F32, KvCacheDtype.F16 })
        foreach (bool cpuExperts in new[] { false, true })
            CheckModel(BackendType.GgmlCuda, true, cpuExperts, dtype);
    }

    private void CheckModel(BackendType backend, bool quantized, bool cpuExperts, KvCacheDtype dtype)
    {
        const int degree = 3, experts = 13;
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 14, kvHeads: 7, experts: experts,
            used: 4, headDim: 16, latent: 32, ffn: 32);
        // Force routes across both asymmetric boundaries, including the final
        // global expert. Routing scores remain genuine model outputs.
        foreach (var (name, (_, data)) in fixture.Weights)
            if (name.EndsWith("exp_probs_b.bias", StringComparison.Ordinal))
            {
                Array.Fill(data, -10f);
                foreach (int expert in new[] { 0, 2, 6, 12 }) data[expert] = 10f;
            }
        string nameSuffix = $"{backend}-{quantized}-{cpuExperts}-{dtype}";
        string path = quantized
            ? MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture, Path.Combine(_directory, nameSuffix + ".gguf"))
            : fixture.Write(Path.Combine(_directory, nameSuffix + ".gguf"));
        KvCacheDtypeConfig.Set(dtype);
        if (cpuExperts) MoeCpuOffloadConfig.SetLayers(1); else MoeCpuOffloadConfig.Reset();
        int configuredOffload = MoeCpuOffloadConfig.CpuMoeLayers;
        using var cpuGroup = new MuseSparkTensorParallelTests.CpuGroup(degree);
        long[] fixedBytes;
        long bytesPerExpert;
        using (var descriptor = new MuseSparkModel(path, BackendType.Cpu, degree, cpuGroup))
        {
            fixedBytes = descriptor.TensorParallelFixedWeightBytes.ToArray();
            bytesPerExpert = descriptor.TensorParallelBytesPerExpert;
        }
        Assert.True(bytesPerExpert > 0);
        int[] desired = [2, 5, 6];
        long[] free = Enumerable.Range(0, degree)
            .Select(rank => fixedBytes[rank] + (16L << 20) + bytesPerExpert * desired[rank]).ToArray();
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        float[][] reference;
        using (var single = new MuseSparkModel(path, backend))
            reference = Run(single, tokens);
        using var group = backend == BackendType.Cpu ? new MuseSparkTensorParallelTests.CpuGroup(degree) : null;
        using var model = new MuseSparkModel(path, backend, degree, group, 1, free);
        free[0] = 0; // Caller changes cannot alter the plan used by initialization.
        Assert.Equal(desired, model.TensorParallelExpertCounts);
        Assert.Equal(desired, Field<int[]>(model, "_tpExpertCount"));
        Assert.Equal(new[] { 0, 2, 7 }, Field<int[]>(model, "_tpExpertFirst"));
        Assert.Equal(configuredOffload, MoeCpuOffloadConfig.CpuMoeLayers);
        Assert.Equal(cpuExperts, MoeCpuOffloadConfig.IsLayerOnCpu(0));
        Assert.False(MoeCpuOffloadConfig.IsLayerOnCpu(1));
        for (int rank = 0; rank < degree; rank++)
            Assert.Equal(fixedBytes[rank] + desired[rank] * bytesPerExpert, model.TensorParallelPlannedWeightBytes[rank]);
        float[][] actual = Run(model, tokens);
        var checks = new MuseSparkGpuChecks(_output);
        int[] lengths = [3, 4, 5, 8, 8, 5, 8];
        for (int row = 0; row < actual.Length; row++)
        {
            checks.Compare(nameSuffix + $" row {row} TP/native", reference[row], actual[row], quantized ? 1e-4 : 3e-5, quantized ? 1e-4 : 3e-5);
            if (!quantized)
                checks.Compare(nameSuffix + $" row {row} TP/scalar", fixture.Reference(tokens[..lengths[row]]), actual[row], 3e-5, 3e-5);
        }
        checks.Complete("asymmetric-" + nameSuffix);
    }

    private static float[][] Run(MuseSparkModel model, int[] tokens)
    {
        var rows = new List<float[]>
        {
            model.Forward(tokens[..3]).ToArray(),
            model.Forward(tokens[3..4]).ToArray(),
            model.Forward(tokens[4..5]).ToArray(),
            model.Forward(tokens[5..]).ToArray(),
        };
        var snapshot = new byte[checked((int)model.ComputeKVBlockByteSize(4))];
        Assert.True(model.TryExtractKVBlock(0, 4, snapshot));
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, 4, snapshot));
        rows.Add(model.Forward(tokens[4..]).ToArray());
        model.SpecRewindCache(3);
        rows.Add(model.Forward(tokens[3..5]).ToArray());
        model.ReleaseGgmlDeviceResidency();
        rows.Add(model.Forward(tokens[5..]).ToArray());
        return rows.ToArray();
    }

    private static T Field<T>(MuseSparkModel model, string name)
        => (T)typeof(MuseSparkModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        Directory.Delete(_directory, true);
    }
}
