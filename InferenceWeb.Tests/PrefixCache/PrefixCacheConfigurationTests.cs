// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests.PrefixCache;

[Collection(EngineEnvironmentCollection.Name)]

public class PrefixCacheConfigurationTests
{
    [Fact]
    public void PrefixCachingIsOnByDefaultAndSurvivesASpeculationChange()
    {
        Assert.True(SchedulerConfig.Default.EnablePrefixCaching);
        Assert.True(SchedulerConfig.Default.WithSpeculation(SpeculationOptions.Disabled).EnablePrefixCaching);
        Assert.False(new SchedulerConfig { EnablePrefixCaching = false }
            .WithSpeculation(SpeculationOptions.Disabled).EnablePrefixCaching);
    }

    /// <summary>Only a value the user gave counts as a chosen lone-prompt step: unset, a model with its own
    /// steps (Muse Spark's pipelined prefill) sizes them.</summary>
    [Fact]
    public void TheSoloPrefillChunkIsChosenOnlyWhenTheEnvironmentSetsIt()
    {
        using var env = new EnvScope();
        env.Set("TS_SCHED_SOLO_PREFILL_CHUNK", null);
        var unset = SchedulerConfig.FromEnvironment();
        Assert.False(unset.SoloPrefillChunkSizeChosen);
        Assert.Equal(8192, unset.SoloPrefillChunkSize);
        env.Set("TS_SCHED_SOLO_PREFILL_CHUNK", "32768");
        var set = SchedulerConfig.FromEnvironment();
        Assert.True(set.SoloPrefillChunkSizeChosen);
        Assert.Equal(32768, set.SoloPrefillChunkSize);
    }

    /// <summary>Likewise the mixed-step prompt bound: unset, a model with its own bound (the
    /// direct-CUDA Qwen engine's fused mixed step) sets it; an explicit 0 chooses none.</summary>
    [Fact]
    public void TheMixedStepPrefillBudgetIsChosenOnlyWhenTheEnvironmentSetsIt()
    {
        using var env = new EnvScope();
        env.Set("TS_SCHED_MIXED_PREFILL_TOKENS", null);
        var unset = SchedulerConfig.FromEnvironment();
        Assert.False(unset.MixedStepPrefillBudgetChosen);
        Assert.Equal(0, unset.MixedStepPrefillBudget);
        Assert.False(unset.WithSpeculation(SpeculationOptions.Disabled).MixedStepPrefillBudgetChosen);
        var modelBound = unset.WithMixedStepPrefillBudget(1024).WithSpeculation(SpeculationOptions.Disabled);
        Assert.True(modelBound.MixedStepPrefillBudgetChosen);
        Assert.Equal(1024, modelBound.MixedStepPrefillBudget);

        env.Set("TS_SCHED_MIXED_PREFILL_TOKENS", "0");
        var off = SchedulerConfig.FromEnvironment();
        Assert.True(off.MixedStepPrefillBudgetChosen);
        Assert.Equal(0, off.MixedStepPrefillBudget);
        env.Set("TS_SCHED_MIXED_PREFILL_TOKENS", "512");
        Assert.Equal(512, SchedulerConfig.FromEnvironment().MixedStepPrefillBudget);
        Assert.True(new SchedulerConfig { MixedStepPrefillBudget = 8 }.MixedStepPrefillBudgetChosen);
    }

    [Fact]
    public void TheOffSwitchTurnsPrefixCachingOff()
    {
        using var env = new EnvScope();
        env.Set("TS_SCHED_PREFIX_CACHE", "0");
        Assert.False(SchedulerConfig.FromEnvironment().EnablePrefixCaching);
    }

    /// <summary>The radix cache is the only prefix cache. A deployment still asking for the
    /// removed legacy mode hears so instead of running on something it did not ask for.</summary>
    [Theory]
    [InlineData("legacy")]
    [InlineData("tree")]
    public void TheRemovedModeVariable_ErrorsNamingWhatReplacesIt(string value)
    {
        using var env = new EnvScope();
        env.Set("TS_PREFIX_CACHE_MODE", value);
        var ex = Assert.Throws<ArgumentException>(() => SchedulerConfig.FromEnvironment());
        Assert.Contains("TS_PREFIX_CACHE_MODE", ex.Message);
        Assert.Contains("TS_SCHED_PREFIX_CACHE", ex.Message);
    }
}
