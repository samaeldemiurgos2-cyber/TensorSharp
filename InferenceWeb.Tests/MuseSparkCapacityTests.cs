using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class MuseSparkCapacityTests
{
    private const long MiB = 1L << 20;

    [Fact]
    public void ReportedTwentyFourOffloadPlacementFailsEvenWithoutScratch()
    {
        long[] weights = [45538, 48970, 48970, 48970, 50647];
        string reason = MuseSparkModel.CapacityRefusal(weights.Select(x => x * MiB).ToArray(),
            Enumerable.Repeat(45490 * MiB, 5).ToArray(), 0);
        Assert.NotNull(reason);
        foreach (int rank in Enumerable.Range(0, 5))
            Assert.Contains($"GPU rank {rank}: weights", reason);
        Assert.DoesNotContain("estimated", reason);
        Assert.True(ModelLoadRefusal.TryDescribe(new ModelLoadRefusedException(reason), out string flattened));
        Assert.Equal(reason, flattened);
    }

    [Fact]
    public void WorkspaceShortfallIsDistinguishedFromImpossibleWeights()
    {
        long reserve = MuseSparkModel.ResolveScratchReserveBytes(8192, 4096, 4096, 256, null);
        Assert.Equal(2560 * MiB, reserve);
        string reason = MuseSparkModel.CapacityRefusal([44000 * MiB], [45490 * MiB], reserve);
        Assert.Contains("estimated compute/active-KV", reason);
        Assert.DoesNotContain("weights 44000 MiB exceed", reason);
        Assert.Null(MuseSparkModel.CapacityRefusal([42000 * MiB], [45490 * MiB], reserve));
    }

    [Fact]
    public void ExplicitHeadroomDoesNotDisableTheWeightRequirement()
    {
        long reserve = MuseSparkModel.ResolveScratchReserveBytes(8192, 4096, 4096, 256, "0");
        Assert.Equal(0, reserve);
        Assert.Null(MuseSparkModel.CapacityRefusal([44000 * MiB], [45490 * MiB], reserve));
        Assert.NotNull(MuseSparkModel.CapacityRefusal([45538 * MiB], [45490 * MiB], reserve));
    }

    [Theory]
    [InlineData("-1")]
    [InlineData("NaN")]
    [InlineData("1.5")]
    [InlineData("9223372036854775807")]
    public void InvalidReserveIsAnActionableRefusal(string value)
        => Assert.Throws<ModelLoadRefusedException>(() =>
            MuseSparkModel.ResolveScratchReserveBytes(8192, 4096, 4096, 256, value));

    [Fact]
    public void TinyModelsDoNotInheritFullModelScratch()
    {
        long tiny = MuseSparkModel.ResolveScratchReserveBytes(32, 16, 16, 4, null);
        Assert.Equal(16 * MiB, tiny);
        Assert.Equal(5120 * MiB, MuseSparkModel.ResolveScratchReserveBytes(8192, 4096, 4096, 512, null));
    }

    [Fact]
    public void RecordedFiveA40Cpu32PlacementKeepsItsMeasuredFit()
    {
        // Header-derived resident bytes from the completed Q3/F16 profile,
        // explicit decoder counts 35,7,7,7,6. Do not invent a scratch policy
        // that rejects a configuration which completed actual inference.
        long[] weights = [30988521984, 44930181632, 44930181632, 44930181632, 40270242816];
        Assert.Null(MuseSparkModel.CapacityRefusal(weights,
            Enumerable.Repeat(45490 * MiB, 5).ToArray(),
            MuseSparkModel.ResolveScratchReserveBytes(8192, 4096, 4096, 256, null)));
    }

    [Fact]
    public void UnavailableDeviceMemoryIsNotTreatedAsZero()
    {
        Assert.Null(MuseSparkModel.CapacityRefusal([50000 * MiB], [-1], 3072 * MiB));
        Assert.Contains("GPU rank 1", MuseSparkModel.CapacityRefusal(
            [50000 * MiB, 50000 * MiB], [-1, 45490 * MiB], 3072 * MiB));
        Assert.NotNull(MuseSparkModel.CapacityRefusal([1], [0], 0));
    }

    [Fact]
    public void ExactHeadroomBoundaryFitsAndDoesNotOverflow()
    {
        Assert.Null(MuseSparkModel.CapacityRefusal([long.MaxValue - 7], [long.MaxValue], 7));
        Assert.NotNull(MuseSparkModel.CapacityRefusal([long.MaxValue - 7], [long.MaxValue], 8));
    }

    [Fact]
    public void DescriptorBytesFollowActualWeightMaterialization()
    {
        var norm = new GgufTensorInfo { Name = "blk.0.attn_norm.weight", Type = GgmlTensorType.F16, Shape = [8192] };
        var matrix = new GgufTensorInfo { Name = "blk.0.attn_q.weight", Type = GgmlTensorType.F16, Shape = [8192, 8192] };
        var expert = new GgufTensorInfo { Name = "blk.0.ffn_gate_exps.weight", Type = GgmlTensorType.Q3_K, Shape = [4096, 4096, 256] };
        Assert.Equal(8192 * 4, MuseSparkModel.PlacementWeightBytes(norm, 8192 * 2, BackendType.GgmlCuda));
        Assert.Equal(8192L * 8192 * 2, MuseSparkModel.PlacementWeightBytes(matrix, 8192L * 8192 * 2, BackendType.GgmlCuda));
        const long quantizedBytes = 1761607680;
        Assert.Equal(quantizedBytes, MuseSparkModel.PlacementWeightBytes(expert, quantizedBytes, BackendType.GgmlCuda));
    }

    [Fact]
    public void RootsAndUnequalLayersAreChargedToTheirOwningRank()
    {
        Assert.Equal(new long[] { 13, 23, 35 }, MuseSparkModel.SumPlacementBytes([3, 5, 7, 11, 13],
            10, 22, [0, 1, 1, 1, 2], 3));
        Assert.Equal(new long[] { 35 }, MuseSparkModel.SumPlacementBytes([3], 10, 22, [0], 1));
    }
}
