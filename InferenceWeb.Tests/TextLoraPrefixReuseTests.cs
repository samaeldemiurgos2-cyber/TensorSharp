// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// K/V computed under a LoRA adapter is not the base model's. Every reuse path is keyed by
// tokens and conversation scope, so a LoRA sequence takes the explicit no-cache policy: on
// the oracle fakes - where a warm prompt is otherwise reused in full - a LoRA request reuses
// nothing the base model cached, and leaves nothing the base model would reuse.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cli;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class TextLoraPrefixReuseTests
{
    private static SchedulerConfig Config => new()
    {
        BlockSize = 16,
        NumBlocks = 64,
        MaxNumRunningSequences = 1,
        MaxNumBatchedTokens = 128,
        SoloPrefillChunkSize = 128,
        StopRepetition = false,
    };

    private sealed record Adapter(string Name) : ILoraAdapterHandle;

    /// <summary>An oracle fake that accepts an adapter. Its outputs do not depend on it; the
    /// reuse counts are what is under test.</summary>
    private sealed class LoraOracleModel(OracleTraits traits) : OracleModel(traits), ILoraAdapterModel
    {
        public void ActivateLoraAdapter(ILoraAdapterHandle adapter) { }
        public IReadOnlyCollection<string> GetLoraTargets(ILoraAdapterHandle adapter) => Array.Empty<string>();
        public IReadOnlyCollection<string> TakeAppliedLoraTargets() => Array.Empty<string>();
        public void MarkLoraUnsupported(ILoraAdapterHandle adapter, IReadOnlyCollection<string> missing) { }
    }

    private static OracleTraits TraitsOf(bool recurrent)
    {
        using var probe = recurrent ? OracleFakes.R() : OracleFakes.P();
        return probe.Traits;
    }

    private static SamplingConfig Lora()
    {
        var sampling = SamplingConfig.Greedy;
        sampling.LoraAdapter = new Adapter("style");
        return sampling;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LoraRequest_NeitherReusesNorSeedsTheBaseModelsPrefix(bool recurrent)
    {
        using var model = new LoraOracleModel(TraitsOf(recurrent));
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] warm = Enumerable.Range(1, 64).ToArray();
        int[] prompt = warm.Concat(new[] { 81, 82, 83 }).ToArray();
        int expectedBaseReuse = recurrent ? 63 : 64;

        // Base warm-up, then the same extension under an adapter: nothing is reused.
        session.Generate(warm, 1, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        var lora = session.Generate(prompt, 4, Lora(), cancellationToken: timeout.Token);
        Assert.Equal(0, lora.Completion.PrefixCacheReusedTokens);

        // The base model's cache is still the base model's: the extension reuses the warm-up.
        var baseAgain = session.Generate(prompt, 4, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        Assert.Equal(expectedBaseReuse, baseAgain.Completion.PrefixCacheReusedTokens);

        // A LoRA warm-up leaves nothing behind for the base model either.
        using var model2 = new LoraOracleModel(TraitsOf(recurrent));
        using var session2 = new CliInferenceSession(model2, Config, NullLogger.Instance);
        session2.Generate(warm, 1, Lora(), cancellationToken: timeout.Token);
        var baseAfterLora = session2.Generate(prompt, 4, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        Assert.Equal(0, baseAfterLora.Completion.PrefixCacheReusedTokens);
    }
}
