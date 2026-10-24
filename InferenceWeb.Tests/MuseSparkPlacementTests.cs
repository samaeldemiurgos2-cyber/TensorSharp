using System;
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class MuseSparkPlacementTests
{
    [Fact]
    public void OffloadedPrefixBalancesResidentBytesInsteadOfLayerCounts()
    {
        long[] layers = Enumerable.Range(0, 62).Select(l => l < 42 ? 1L : 80L).ToArray();
        int[] map = MuseSparkModel.PlanLayerPlacement(layers, 10, 10, 5);
        long[] loads = new long[5]; loads[0] = 10; loads[4] = 10;
        for (int l = 0; l < layers.Length; l++) loads[map[l]] += layers[l];
        Assert.Equal(Enumerable.Range(0,5), map.Distinct());
        Assert.True(map.Zip(map.Skip(1)).All(pair => pair.First <= pair.Second));
        Assert.True(loads.Max() - loads.Min() <= 80, string.Join(",", loads));
        Assert.True(map.Count(rank => rank == 0) > 40);
    }

    [Theory]
    [InlineData("2,3", 5, 2)]
    [InlineData("1,1,3", 5, 3)]
    public void ExplicitCountsAreHonored(string counts, int layers, int devices)
    {
        var map = MuseSparkModel.PlanLayerPlacement(new long[layers], 0, 0, devices, counts);
        Assert.Equal(counts.Split(',').Select(int.Parse), Enumerable.Range(0,devices).Select(rank => map.Count(x => x == rank)));
    }

    [Theory]
    [InlineData("2,2")]
    [InlineData("0,5")]
    [InlineData("1,1,3")]
    [InlineData("-1,6")]
    public void InvalidPlacementDoesNotSilentlyIgnoreTheRequest(string counts)
        => Assert.Throws<ArgumentException>(() => MuseSparkModel.PlanLayerPlacement(new long[5], 0, 0, 2, counts));
}
