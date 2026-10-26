// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

[Collection(EngineEnvironmentCollection.Name)]
public class CloneReservationTests
{
    [Theory]
    [InlineData(0, false)]
    [InlineData(32, false)]
    [InlineData(0, true)]
    public void Admission_BudgetsAndCopiesTheSameFittedReservation_OrAllocatesNothing(int generationCap, bool refuse)
    {
        string? previous = Environment.GetEnvironmentVariable("TS_KV_GENERATION_RESERVE_MAX");
        Environment.SetEnvironmentVariable("TS_KV_GENERATION_RESERVE_MAX", generationCap.ToString());
        using var model = new ReservationOracle { RefuseClone = refuse };
        var pool = new BlockPool(256, 8, 64);
        var scheduler = new ContinuousBatchScheduler(new SchedulerConfig { BlockSize = 8, NumBlocks = 256 }, pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        var seed = new SequenceState("seed", Enumerable.Range(1, 16).ToList(), 3, 8,
            SamplingConfig.Greedy, sharedPrefixTokens: 16);
        foreach (var block in pool.AllocateNew(2)!) seed.BlockTable.AppendBlock(block);
        try
        {
            model.Forward(seed.PromptTokens.ToArray());
            seed.SetComputedTokensForPrefixAdoption(16);
            cache.CaptureCheckpoint(seed);
            Assert.Single(model.RetainedPayloadKeys);
            var request = new SequenceState("clone", Enumerable.Range(1, 64).ToList(), 100, 8,
                SamplingConfig.Greedy, cacheScope: "other", sharedPrefixTokens: 16);
            Assert.Equal(16, cache.ComputeReusablePrefix(request));
            bool adopted = cache.TryAdopt(request, 16, (_, _) => throw new InvalidOperationException("No truncation expected"));
            Assert.Equal(!refuse, adopted);
            Assert.Equal(!refuse, request.PrefillReservationTaken);
            Assert.Equal(BatchExecutor.ResolvePrefillReservation(64, 100, model.MaxContextLength, generationCap), model.Requested);
            Assert.Equal(model.Requested - 1, model.Estimated);
            Assert.Equal(1, model.FitCalls);
            if (refuse)
            {
                Assert.Empty(model.Materializations);
                Assert.Equal(0, model.PrivateHolderCount);
                Assert.Equal(0, request.BlockTable.NumBlocks);
            }
            else
            {
                Assert.Equal(model.Estimated, Assert.Single(model.Materializations).RequiredContextTokens);
                Assert.Equal(1, model.PrivateHolderCount);
                pool.Free(request.BlockTable.Clear());
                model.OnSequenceReleased(request.RequestId);
            }
        }
        finally
        {
            cache.Reset();
            pool.Free(seed.BlockTable.Clear());
            Environment.SetEnvironmentVariable("TS_KV_GENERATION_RESERVE_MAX", previous);
        }
    }

    private sealed class ReservationOracle() : OracleModel(new OracleTraits
    {
        Name = "reservation", Class = FamilyClass.R, EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true, Truncation = TruncationKind.None,
    }, 8), IPrefixCacheModel
    {
        internal bool RefuseClone;
        internal int Requested, Estimated, FitCalls;
        internal List<MaterializeRequest> Materializations = new();
        public int ResolveCloneReservationTokens(string key, int requiredContextTokens)
        {
            FitCalls++;
            Requested = requiredContextTokens;
            return requiredContextTokens - 1;
        }
        public ResourceVector EstimateCloneBytes(string key, int tokens, int requiredContextTokens)
        {
            Estimated = requiredContextTokens;
            // Limit memory only after matching, so this exercises admission's budget gate.
            if (RefuseClone) SpareBytes = 0;
            return new ResourceVector { HostKv = requiredContextTokens * 16L };
        }
        public new bool TryMaterialize(in MaterializeRequest request)
        {
            Materializations.Add(request);
            return base.TryMaterialize(request);
        }
    }
}
