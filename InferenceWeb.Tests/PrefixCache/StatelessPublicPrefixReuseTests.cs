using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// A tensor-parallel Nemotron serves the server's stateless requests on the per-sequence route with A1
/// pages that resume only where a page ends with its recurrent state, i.e. where a prefill chunk ended.
/// The engine gave such a family 16-token blocks, so the only restore points were the last 16-token
/// boundaries before each prompt's end: past the prefix a later request can reuse (its prompt leaves the
/// warm-up's a little before that), so every request re-prefilled its ~3.6K-token system prompt, where the
/// single-device batched route (256-token blocks) reused 3,328 tokens of it. A recurrent page family keeps
/// 256-token blocks: the chunk aligner then leaves a restore point at the last 256 boundary.
/// </summary>
[Collection("Scheduler block size environment")]
public sealed class StatelessPublicPrefixReuseTests : IDisposable
{
    private const string Variable = "TS_SCHED_BLOCK_SIZE";
    private readonly string? _restore = Environment.GetEnvironmentVariable(Variable);
    private const int WarmUp = 600;     // the startup warm-up: the public prefix alone
    private const int Shared = 580;     // what a real request shares of it, like the server's

    public StatelessPublicPrefixReuseTests() => Environment.SetEnvironmentVariable(Variable, null);

    public void Dispose() => Environment.SetEnvironmentVariable(Variable, _restore);

    private static List<int> Prefix(int length) => Enumerable.Range(0, length).Select(k => 10 + k % 200).ToList();

    private static List<int> Request(int salt)
        => Prefix(Shared).Concat(Enumerable.Range(0, 30).Select(k => 220 + (salt * 7 + k * 3) % 30)).ToList();

    [Fact]
    public async Task ARecurrentPageFamily_ReusesThePublicPrefix_OnTheBlockSizeTheEngineChooses()
    {
        using var model = OracleFakes.R2();
        // The block size is the engine's choice, as on the server.
        using var engine = new InferenceEngine(model, SchedulerConfig.FromEnvironment(), NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        int block = engine.PoolStats.blockSize;

        async Task<int> Run(string id, List<int> prompt, int shared)
        {
            var seq = new SequenceState(id, prompt, 4, block, SamplingConfig.Greedy, sharedPrefixTokens: shared);
            return (await engine.SubmitRequest(seq).Completion.WaitAsync(timeout.Token)).PrefixCacheReusedTokens;
        }

        await Run("warm-up", Prefix(WarmUp), WarmUp);
        var reused = new List<int>();
        foreach (int salt in new[] { 1, 2, 1, 3 })
            reused.Add(await Run($"r{reused.Count}", Request(salt), Shared));

        // The last whole block inside what the requests share.
        int expected = Shared / block * block;
        Assert.True(expected > 0, $"block size {block}");
        Assert.All(reused, r => Assert.Equal(expected, r));
    }
}
