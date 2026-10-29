using TensorSharp;

namespace InferenceWeb.Tests;

public sealed class LayerSplitPlanTests
{
    [Fact]
    public void MistralPreservesItsExistingCompiledConstructorContract()
    {
        Assert.NotNull(typeof(TensorSharp.Models.Mistral3Model).GetConstructor(
            [typeof(string), typeof(BackendType), typeof(int), typeof(ITensorParallelGroup)]));
    }

    [Fact]
    public void WeightedPartitionBalancesUnevenLayersAndEndpointWeights()
    {
        Assert.Equal(new[] { 0, 6, 8 }, LayerSplitPlan.Partition([1, 1, 1, 1, 6, 6, 6, 6], 2, 2));
        Assert.Equal(new[] { 0, 3, 10 }, LayerSplitPlan.Partition(Enumerable.Repeat(10L, 10).ToArray(), 2, 2, 40));
        Assert.Equal(new[] { 0, 7, 10 }, LayerSplitPlan.Partition(Enumerable.Repeat(10L, 10).ToArray(), 2, 2, lastBytes: 40));
        Assert.Equal(new[] { 0, 2, 4, 9 }, LayerSplitPlan.Partition(new long[9], 3, 2));
    }

    [Fact]
    public void WeightedPartitionMatchesExhaustiveOptimalCost()
    {
        var random = new Random(1017);
        for (int length = 1; length <= 12; length++)
        for (int stages = 1; stages <= Math.Min(4, length); stages++)
        for (int minimum = 1; minimum <= length / stages; minimum++)
        for (int sample = 0; sample < 8; sample++)
        {
            long[] bytes = Enumerable.Range(0, length).Select(_ => (long)random.Next(0, 101)).ToArray();
            long first = random.Next(0, 201), last = random.Next(0, 201);
            var cuts = LayerSplitPlan.Partition(bytes, stages, minimum, first, last);
            Assert.Equal(0, cuts[0]);
            Assert.Equal(length, cuts[^1]);
            long actual = 0;
            for (int stage = 0; stage < stages; stage++)
            {
                Assert.True(cuts[stage + 1] - cuts[stage] >= minimum);
                long cost = bytes.Skip(cuts[stage]).Take(cuts[stage + 1] - cuts[stage]).Sum();
                if (stage == 0) cost += first;
                if (stage == stages - 1) cost += last;
                actual = Math.Max(actual, cost);
            }
            long Enumerate(int stage, int begin)
            {
                long optimal = long.MaxValue;
                int finalEnd = length - (stages - stage - 1) * minimum;
                for (int end = begin + minimum; end <= finalEnd; end++)
                {
                    if (stage == stages - 1 && end != length) continue;
                    long cost = bytes.Skip(begin).Take(end - begin).Sum();
                    if (stage == 0) cost += first;
                    if (stage == stages - 1) cost += last;
                    else cost = Math.Max(cost, Enumerate(stage + 1, end));
                    optimal = Math.Min(optimal, cost);
                }
                return optimal;
            }
            Assert.Equal(Enumerate(0, 0), actual);
        }
    }

    [Fact]
    public void WeightedPlansOwnEveryLayerOnceAndUseEveryDevice()
    {
        var random = new Random(41);
        for (int nodes = 1; nodes <= 4; nodes++)
        for (int devices = 1; devices <= 4; devices++)
        {
            long[] bytes = Enumerable.Range(0, nodes * devices + 9).Select(_ => (long)random.Next(0, 1000)).ToArray();
            var owners = new int[bytes.Length];
            for (int node = 0; node < nodes; node++)
            {
                var plan = LayerSplitPlan.CreateWeighted(bytes, devices, node, nodes, 4000, 1000);
                var perDevice = new int[devices];
                for (int layer = plan.FirstLayer; layer < plan.EndLayer; layer++)
                {
                    owners[layer]++;
                    perDevice[plan.LocalRank(layer)]++;
                }
                Assert.All(perDevice, count => Assert.True(count > 0));
            }
            Assert.All(owners, count => Assert.Equal(1, count));
        }
    }

    [Theory]
    [InlineData(2, 1)]
    [InlineData(2, 3)]
    [InlineData(3, 2)]
    [InlineData(4, 1)]
    public void DriverLastOrderRunsTheSameStagesOnRotatedNodes(int nodes, int devices)
    {
        long[] bytes = Enumerable.Range(0, 40).Select(i => (long)(i * 37 % 11 + 1)).ToArray();
        for (int node = 0; node < nodes; node++)
        {
            var rotated = LayerSplitPlan.CreateWeighted(bytes, devices, node, nodes, 50, 70, true);
            int stage = (node + nodes - 1) % nodes;
            // A stage owns the same layers and device split whichever node runs it.
            var reference = LayerSplitPlan.CreateWeighted(bytes, devices, stage, nodes, 50, 70);
            Assert.Equal(stage, rotated.StageIndex);
            Assert.Equal(reference.FirstLayer, rotated.FirstLayer);
            Assert.Equal(reference.EndLayer, rotated.EndLayer);
            for (int layer = rotated.FirstLayer; layer < rotated.EndLayer; layer++)
                Assert.Equal(reference.LocalRank(layer), rotated.LocalRank(layer));
            Assert.Equal(stage == 0, rotated.IsFirst);
            Assert.Equal(stage == nodes - 1, rotated.IsLast);
            Assert.Equal(node == 0, rotated.IsLast);
            if (!rotated.IsFirst) Assert.Equal((node + nodes - 1) % nodes, rotated.PreviousNode);
            if (!rotated.IsLast) Assert.Equal((node + 1) % nodes, rotated.NextNode);
        }
        Assert.Equal(1, LayerSplitPlan.NodeOfStage(0, nodes, true));
        Assert.Equal(0, LayerSplitPlan.NodeOfStage(nodes - 1, nodes, true));
        Assert.Equal(nodes - 1, LayerSplitPlan.StageOf(0, nodes, true));
    }

    [Fact]
    public void LegacyOrderAndSingleNodePlansAreUnchanged()
    {
        var legacy = new LayerSplitPlan(12, 2, 1, 3);
        Assert.False(legacy.DriverRunsLastStage);
        Assert.Equal(1, legacy.StageIndex);
        Assert.Equal(0, legacy.PreviousNode);
        Assert.Equal(2, legacy.NextNode);
        var single = new LayerSplitPlan(12, 3, 0, 1, true);
        Assert.False(single.DriverRunsLastStage);
        Assert.True(single.IsFirst && single.IsLast);
    }

    [Fact]
    public void PartitionRejectsInvalidCostsAndPreservesMaximumRepresentableCost()
    {
        Assert.Throws<ArgumentException>(() => LayerSplitPlan.Partition([1, 2, 3], 2, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => LayerSplitPlan.Partition([1, -1], 2, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => LayerSplitPlan.Partition([1], 1, 1, -1));
        Assert.Throws<OverflowException>(() => LayerSplitPlan.Partition([long.MaxValue, 1], 1, 1));
        Assert.Equal(new[] { 0, 2 }, LayerSplitPlan.Partition([long.MaxValue, 0], 1, 1));
        Assert.Equal(new[] { 0, 1, 2 }, LayerSplitPlan.Partition([0, long.MaxValue], 2, 1));
    }
}
