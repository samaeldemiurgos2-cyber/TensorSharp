// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkTensorParallelQKPrepTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    [InlineData(9, 1)]
    [InlineData(128, 2)]
    [InlineData(256, 1)]
    public void RankPackingInitializesEveryUploadedByteAndPreservesPaddedInputs(int rows, int kvHeads)
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        const int hd = 64, padding = 7;
        int qWidth = kvHeads * 8 * hd, kWidth = kvHeads * hd;
        int[] widths = [qWidth, kWidth, kWidth, qWidth];
        var owners = new Tensor[4];
        var views = new Tensor[4];
        var original = new float[4][];
        try
        {
            for (int part = 0; part < 4; part++)
            {
                int stride = widths[part] + padding;
                original[part] = Enumerable.Range(0, rows * stride)
                    .Select(i => i % stride < widths[part] ? (part + 1) * 100000 + i : -99999f).ToArray();
                owners[part] = new Tensor(allocator, DType.Float32, rows, stride);
                owners[part].SetElementsAsFloat(original[part]);
                views[part] = owners[part].Narrow(1, 0, widths[part]);
            }
            using var packed = MuseSparkModel.PackSparkRankQKVG(allocator, views[0], views[1], views[2], views[3]);
            int totalWidth = widths.Sum();
            float[] actual = packed.GetElementsAsFloat(rows * totalWidth);
            Assert.Equal(new long[] { rows, totalWidth }, packed.Sizes);
            for (int row = 0; row < rows; row++)
            {
                int offset = 0;
                for (int part = 0; part < 4; part++)
                {
                    Assert.Equal(original[part].AsSpan(row * (widths[part] + padding), widths[part]).ToArray(),
                        actual.AsSpan(row * totalWidth + offset, widths[part]).ToArray());
                    offset += widths[part];
                }
            }
            for (int part = 0; part < 4; part++)
                Assert.Equal(original[part], owners[part].GetElementsAsFloat(original[part].Length));
        }
        finally
        {
            foreach (var view in views) view?.Dispose();
            foreach (var owner in owners) owner?.Dispose();
        }
    }

    [Fact]
    public void InvalidProjectionGeometryFailsBeforeAnyInputMutation()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var query = new Tensor(allocator, DType.Float32, 2, 16);
        using var key = new Tensor(allocator, DType.Float32, 2, 8);
        using var badRows = new Tensor(allocator, DType.Float32, 1, 8);
        using var badWidth = new Tensor(allocator, DType.Float32, 2, 7);
        query.SetElementsAsFloat(Enumerable.Repeat(17f, 32).ToArray());
        key.SetElementsAsFloat(Enumerable.Repeat(29f, 16).ToArray());
        Assert.Throws<ArgumentException>(() => MuseSparkModel.PackSparkRankQKVG(allocator, query, key, badRows, query));
        Assert.Throws<ArgumentException>(() => MuseSparkModel.PackSparkRankQKVG(allocator, query, key, badWidth, query));
        Assert.Throws<ArgumentException>(() => MuseSparkModel.PackSparkRankQKVG(allocator, query, key, key, key));
        Assert.All(query.GetElementsAsFloat(32), value => Assert.Equal(17f, value));
        Assert.All(key.GetElementsAsFloat(16), value => Assert.Equal(29f, value));
    }

    [Fact]
    public void UnsupportedCpuTensorParallelRetainsIndependentPackedScalarPath()
    {
        using var environment = new NativeEnvScope();
        environment.Set("TS_MUSE_SPARK_FUSE_QK_PREP", "1");
        using var controls = new MuseSparkTensorParallelTests();
        controls.PackedUnevenRanksRetainIndependentPagedSequences();
        controls.WholeAndUnevenPartitionsMatchScalarOracleAcrossChunksSnapshotsAndSpeculation(9, false);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuLocalHeadGroupsRetainPrimitiveOracleAcrossRowsAndPositions()
    {
        MuseSparkQKPrepTests.GraphCheck(GgmlBackendType.Cpu, 1, 8, 1);
        MuseSparkQKPrepTests.GraphCheck(GgmlBackendType.Cpu, 1, 16, 2);
    }

    [MuseSparkTensorParallelGpuFact(2)]
    public void NativeCudaLocalHeadGroupsRetainPrimitiveOracleAcrossRanksRowsAndPositions()
    {
        MuseSparkQKPrepTests.GraphCheck(GgmlBackendType.Cuda, 2, 8, 1);
        MuseSparkQKPrepTests.GraphCheck(GgmlBackendType.Cuda, 2, 16, 2);
    }

    [MuseSparkTensorParallelGpuFact(9)]
    public void FusedRankPreparationPreservesQuantizedCacheOffloadAndScalarControls()
    {
        using var environment = new NativeEnvScope();
        environment.Set("TS_MUSE_SPARK_FUSE_QK_PREP", "1");
        using var controls = new MuseSparkTensorParallelGpuTests(output);
        controls.TwoRanksMatchNativeQuantizedOracleAcrossCacheModes();
        controls.ThreeUnevenRanksMatchNativeQuantizedOracleAcrossCacheModes();
        controls.FiveUnevenRanksMatchNativeQuantizedOracleAcrossCacheModes();
        controls.EightRanksMatchNativeQuantizedOracleAcrossCacheModes();
        controls.NineUnevenRanksMatchNativeQuantizedOracleAcrossCacheModes();
        controls.NineUnevenRanksFloatWeightsMatchIndependentScalarOracle();
    }

    [MuseSparkTensorParallelGpuFact(9)]
    public void FusedRankPreparationPreservesPublishedHeadGeometryAndPackedSchedules()
    {
        using var environment = new NativeEnvScope();
        environment.Set("TS_MUSE_SPARK_FUSE_QK_PREP", "1");
        using var controls = new MuseSparkTensorParallelFlashAttentionTests(output);
        controls.FiveRanksPublishedHeadGeometryPackedF32MatchesOrdinaryAndScalar();
        controls.NineRanksPublishedHeadGeometryPackedF32MatchesOrdinaryAndScalar();
    }
}
