// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

[CollectionDefinition("Scheduler block size environment", DisableParallelization = true)]
public sealed class SchedulerBlockSizeEnvironmentCollection { }

/// <summary>
/// A family whose only cross-request reuse is its pages reuses whole blocks only. At the old fixed 256
/// tokens every turn recomputed its last partial block, and a turn shorter than a block reused nothing:
/// eight parallel Hunyuan Dense conversations reused 0 tokens on their second turn. Unless the block size is
/// configured, the engine now picks 16 for such a family and keeps 256 for the ones with end states.
/// </summary>
[Collection("Scheduler block size environment")]
public sealed class PoolBlockSizeTests : IDisposable
{
    private const string Variable = "TS_SCHED_BLOCK_SIZE";
    private readonly string? _restore = Environment.GetEnvironmentVariable(Variable);

    public PoolBlockSizeTests() => Environment.SetEnvironmentVariable(Variable, null);

    public void Dispose() => Environment.SetEnvironmentVariable(Variable, _restore);

    public static IEnumerable<object[]> Families() => OracleFakes.All.Select(f => new object[] { f.Name });

    [Theory]
    [MemberData(nameof(Families))]
    public void AnUnsetBlockSize_IsSixteenForAPageOnlyFamily_AndTheDefaultOtherwise(string family)
    {
        using OracleModel model = OracleFakes.All.Single(f => f.Name == family).Create();
        bool pageOnly = model.Traits.EndState == EndStateSupport.None;
        int expected = pageOnly ? SchedulerConfig.PageFamilyBlockSize : SchedulerConfig.DefaultBlockSize;

        Assert.Equal(0, SchedulerConfig.FromEnvironment().BlockSize);
        Assert.Equal(expected, InferenceEngine.PreferredBlockSize(model));
        using (var engine = new InferenceEngine(model, SchedulerConfig.FromEnvironment(), NullLogger.Instance))
            Assert.Equal(expected, engine.PoolStats.blockSize);
    }

    [Fact]
    public void AConfiguredBlockSize_IsUsedAsGiven()
    {
        using OracleModel model = OracleFakes.P2();
        Environment.SetEnvironmentVariable(Variable, "64");
        using (var fromEnvironment = new InferenceEngine(model, SchedulerConfig.FromEnvironment(), NullLogger.Instance))
            Assert.Equal(64, fromEnvironment.PoolStats.blockSize);
        using (var inCode = new InferenceEngine(model, new SchedulerConfig { BlockSize = 256 }, NullLogger.Instance))
            Assert.Equal(256, inCode.PoolStats.blockSize);
    }
}
