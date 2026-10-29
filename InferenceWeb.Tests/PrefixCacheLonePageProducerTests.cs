// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron-H's prefix cache keeps a shared prompt prefix only as recurrent pages, and a page
// is a restore point only where a lone (per-sequence) forward ended: a step of two prompts
// runs the batched route, whose model-paged rows carry no Mamba2 state, and publishes no
// page. Concurrent requests that share a system prompt therefore shared it only when the
// first of them happened to reach an idle engine alone. A warm server's burst arrives before
// the engine's next step, so all of it was admitted at once, every request forwarded the
// whole prompt, nothing was kept, and every later burst did the same: kvReused=0 and a full
// re-prefill (TTFT ~40 s against ~10 s) on every request of an A40 parallel-server run,
// in some runs and not others. The scheduler now lets the first prompt forward the shared
// prefix alone while the requests that share it wait that one step, then resume from it.
//
// Two more ways the prefix went unshared: a first request that reached the engine alone with a
// prompt running past the shared prefix split its forward only before its own last block, so
// every page of the shared prefix was cached without state (its forward now also ends at the
// shared page); and a restore point that arrived after such a stateless page was dropped as a
// duplicate of it (it now replaces the stateless page).
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class PrefixCacheLonePageProducerTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int PeakToken = 3;
    // The system prompt every request starts with, and the shared prefix the chat layer marks.
    private const int SharedTokens = 42;
    // The page boundary the requests can share: the shared prefix, block-aligned.
    private const int SharedPage = SharedTokens - SharedTokens % BlockSize;
    private const int TailTokens = 6;

    [Fact]
    public async Task Engine_BurstSharingASystemPrompt_ResumesFromTheFirstPromptsLonePage()
    {
        var model = new LonePageStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var gate = new ComputeGate();
        engine.ComputeGate = gate;

        // The burst reaches the engine before its next step, as a warm server's concurrent
        // requests do: the gate holds the step loop until every request is queued.
        gate.Close();
        var sequences = Enumerable.Range(0, 4).Select(i => Shared($"burst{i}", i)).ToList();
        var handles = sequences.Select(s => engine.SubmitRequest(s)).ToList();
        gate.Open();

        Task all = Task.WhenAll(handles.Select(h => h.Completion));
        Assert.Same(all, await Task.WhenAny(all, Task.Delay(TimeSpan.FromSeconds(30))));
        foreach (var handle in handles)
        {
            var completion = await handle.Completion;
            Assert.Null(handle.Sequence.Error);
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        }

        // The first request forwarded the shared prefix alone, ending on its page; the others
        // resumed from that page instead of forwarding it again.
        Assert.Equal(0, sequences[0].PrefixCacheReusedTokens);
        Assert.All(sequences.Skip(1), s => Assert.Equal(SharedPage, s.PrefixCacheReusedTokens));
        Assert.Equal(SharedPage, model.LoneForwards[0]);
        int promptTokens = SharedTokens + TailTokens;
        Assert.Equal(promptTokens + 3 * (promptTokens - SharedPage), model.PromptTokensForwarded);
    }

    [Fact]
    public async Task Engine_ALoneFirstPromptPastTheSharedPrefix_LeavesARestorePointThere()
    {
        // The first request reached the idle engine alone, so nothing waited for it, and its prompt
        // ran past the shared prefix. Its forward was split only before its own last block (56), so
        // it cached every page of the shared prefix without state: the burst behind it found no page
        // to resume (an A40 run whose first request was the 2067-token one, behind a 1890-token
        // system prompt: kvReused=0 on every request of every repeat). Its forward now also ends at
        // the shared page, and the whole burst resumes there.
        var model = new LonePageStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var gate = new ComputeGate();
        engine.ComputeGate = gate;

        var first = Long("long", SharedTokens);
        await RunAlone(engine, first);
        int tailSplit = LongPromptTokens - LongPromptTokens % BlockSize;
        Assert.Equal(new[] { SharedPage, tailSplit - SharedPage, LongPromptTokens - tailSplit }, model.LoneForwards);

        var burst = await RunBurst(engine, gate, 3);
        Assert.All(burst, s => Assert.Equal(SharedPage, s.PrefixCacheReusedTokens));
        int promptTokens = SharedTokens + TailTokens;
        Assert.Equal(LongPromptTokens + 3 * (promptTokens - SharedPage), model.PromptTokensForwarded);
    }

    [Fact]
    public async Task Engine_ABurstArrivingDuringALongFirstPromptsFirstStep_ResumesFromThePageThatStepPublished()
    {
        // As the A40 run met it: the long first request reached the idle engine alone, and the seven
        // requests behind it reached the engine during its first step, so they were admitted beside
        // it while it still had prompt left. The lone-producer hold cannot act on a producer that is
        // already running; the only page that can serve them is the one its first step published
        // (OnBlocksCommitted -> CapturePages), and that page is a restore point only when the step
        // ended at the shared page. Its forward used to end before its own last block instead, so
        // every request of the burst forwarded the whole system prompt again.
        var model = new LonePageStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var gate = new ComputeGate();
        engine.ComputeGate = gate;

        // Park the step loop after exactly one step: the gate closes inside the first forward, and
        // the loop checks it before it schedules the next step.
        model.OnForward = () =>
        {
            model.OnForward = null;
            gate.Close();
        };
        var first = Long("long", SharedTokens);
        var firstHandle = engine.SubmitRequest(first);
        Assert.True(SpinWait.SpinUntil(() => engine.StepsHeldByGate > 0, TimeSpan.FromSeconds(30)),
            "the step loop never parked after the long request's first step");
        int[] firstStepForwards = model.LoneForwards.ToArray();
        int computedAfterFirstStep = first.NumComputedTokens;
        Assert.InRange(computedAfterFirstStep, 1, LongPromptTokens - 1);

        const int Followers = 7;
        var burst = Enumerable.Range(0, Followers).Select(i => Shared($"burst{i}", i)).ToList();
        var handles = burst.Select(s => engine.SubmitRequest(s)).Prepend(firstHandle).ToList();
        gate.Open();
        foreach (var handle in handles)
        {
            var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
            Assert.Null(handle.Sequence.Error);
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        }

        // Every request of the burst resumed from the shared page, not only the first.
        Assert.All(burst.Select(s => s.PrefixCacheReusedTokens), reused => Assert.Equal(SharedPage, reused));
        Assert.Equal(new[] { SharedPage }, firstStepForwards);
        Assert.Equal(SharedPage, computedAfterFirstStep);
        int promptTokens = SharedTokens + TailTokens;
        Assert.Equal(LongPromptTokens + Followers * (promptTokens - SharedPage), model.PromptTokensForwarded);
    }

    [Fact]
    public async Task Engine_ARestorePointAtTheSharedPage_ReplacesTheStatelessPageCachedFirst()
    {
        // A request of a longer system prompt variant (a 50-token shared prefix, page 48) cached the
        // page of the 42-token ancestor without state. The next lone request of the shorter variant
        // ended its forward exactly at that page, but the tree kept the stateless page that arrived
        // first and dropped the restore point as a duplicate - and no request reused the prefix
        // again until the page was evicted. The restore point now replaces it.
        var model = new LonePageStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int variantShared = SharedPage + BlockSize + 2;
        await RunAlone(engine, Long("variant", variantShared));
        Assert.True(model.LoneForwards[0] > SharedPage, "the variant's restore point lies past the shared page");

        var producer = Shared("r1", 0);
        await RunAlone(engine, producer);
        Assert.Equal(0, producer.PrefixCacheReusedTokens);
        var later = Shared("r2", 1);
        await RunAlone(engine, later);
        Assert.Equal(SharedPage, later.PrefixCacheReusedTokens);
    }

    [Fact]
    public void Schedule_FirstStepOfABurst_IsTheLoneProducerCutAtTheSharedPage()
    {
        var (sched, executor, _) = NewScheduler(new LonePageStubModel());
        var burst = Enumerable.Range(0, 3).Select(i => Shared($"b{i}", i)).ToList();
        foreach (var seq in burst) sched.Submit(seq);

        var first = sched.Schedule();
        var work = Assert.Single(first.ScheduledWork);
        Assert.Same(burst[0], work.Sequence);
        Assert.Equal(SharedPage, work.NumScheduledTokens);
        Assert.Equal(2, sched.WaitingCount);
        executor.ExecuteStep(first);

        var second = sched.Schedule();
        Assert.Equal(3, second.ScheduledWork.Count);
        Assert.Equal(SharedPage, burst[1].PrefixCacheReusedTokens);
        Assert.Equal(SharedPage, burst[2].PrefixCacheReusedTokens);
    }

    [Fact]
    public void Schedule_ARequestThatSharesNothing_KeepsTheBurstTogether()
    {
        // Holding the others would not make the producer's step lone: the unrelated request
        // joins it, the step runs batched and publishes nothing. Everything is admitted at once,
        // as before.
        var (sched, _, _) = NewScheduler(new LonePageStubModel());
        sched.Submit(Shared("s0", 0));
        sched.Submit(Shared("s1", 1));
        var unrelated = Enumerable.Range(500, SharedTokens + TailTokens).ToList();
        sched.Submit(new SequenceState("other", unrelated, 2, BlockSize, SamplingConfig.Greedy,
            sharedPrefixTokens: SharedTokens));

        Assert.Equal(3, sched.Schedule().ScheduledWork.Count);
        Assert.Equal(0, sched.WaitingCount);
    }

    [Fact]
    public void Schedule_AnUnrelatedRequestPastTheRunningCap_StillKeepsTheBurstTogether()
    {
        // Held requests take no running slot, so admission visits every waiter: one that shares
        // nothing, however far back, would join the producer's step. Admission stays as before.
        var cfg = Config(maxRunning: 2);
        var (sched, _, _) = NewScheduler(new LonePageStubModel(), cfg);
        var s0 = Shared("s0", 0);
        var s1 = Shared("s1", 1);
        sched.Submit(s0);
        sched.Submit(s1);
        sched.Submit(Shared("s2", 2));
        sched.Submit(new SequenceState("other", Enumerable.Range(500, SharedTokens + TailTokens).ToList(), 2,
            BlockSize, SamplingConfig.Greedy, sharedPrefixTokens: SharedTokens));

        var step = sched.Schedule();
        Assert.Equal(new[] { s0, s1 }, step.ScheduledWork.Select(w => w.Sequence));
    }

    [Fact]
    public void Schedule_AdapterRequests_NeitherProduceNorWaitForALonePage()
    {
        // A LoRA request is cache-isolated: it neither publishes nor reuses prefix state, so a
        // burst of them has nothing to wait for and is admitted together.
        var (sched, _, _) = NewScheduler(new LonePageStubModel());
        var adapter = new StubAdapter();
        for (int i = 0; i < 3; i++)
            sched.Submit(Shared($"a{i}", i, adapter));

        var step = sched.Schedule();
        Assert.Equal(3, step.ScheduledWork.Count);
        Assert.Same(adapter, step.Adapter);
        Assert.All(step.ScheduledWork, w => Assert.Equal(0, w.Sequence.PrefixCacheReusedTokens));
    }

    [Fact]
    public void Schedule_AHeldRequest_KeepsItsTurnAheadOfALaterAdapterRequest()
    {
        // Adapter epochs are first-come-first-served. A request held for the producer's page
        // keeps its place in the queue: it used to be moved to the tail, behind an adapter
        // request that arrived after it, and then waited out that adapter's whole epoch.
        var (sched, executor, _) = NewScheduler(new LonePageStubModel());
        var producer = Shared("p", 0);
        var held = Shared("w1", 1);
        var adapterRequest = Shared("a", 2, new StubAdapter());
        var later = Shared("w2", 3);
        foreach (var seq in new[] { producer, held, adapterRequest, later })
            sched.Submit(seq);

        var first = sched.Schedule();
        var work = Assert.Single(first.ScheduledWork);
        Assert.Same(producer, work.Sequence);
        Assert.Equal(SharedPage, work.NumScheduledTokens);
        executor.ExecuteStep(first);

        var second = sched.Schedule();
        Assert.Null(second.Adapter);
        Assert.Equal(new[] { producer, held }, second.ScheduledWork.Select(w => w.Sequence));
        Assert.Equal(SharedPage, held.PrefixCacheReusedTokens);
        // The adapter request and the base request that arrived after it keep their order.
        Assert.Equal(2, sched.WaitingCount);
    }

    [Fact]
    public void Schedule_AFamilyWithoutTheBatchedRoute_IsNotHeld()
    {
        // Every step of a model without the batched route runs per sequence and writes each
        // prompt's pages, so its burst is admitted together exactly as before.
        var (sched, _, _) = NewScheduler(new LonePageStubModel { BatchedRoute = false });
        for (int i = 0; i < 3; i++)
            sched.Submit(Shared($"p{i}", i));

        Assert.Equal(3, sched.Schedule().ScheduledWork.Count);
    }

    private static SchedulerConfig Config(int maxRunning = 8) => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = maxRunning,
        MaxPrefillChunkSize = 64,
        NumBlocks = 128,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = BlockSize,
    };

    private static (ContinuousBatchScheduler, BatchExecutor, BlockPool) NewScheduler(
        LonePageStubModel model, SchedulerConfig? config = null)
    {
        var cfg = config ?? Config();
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var sched = new ContinuousBatchScheduler(cfg, pool, NullLogger.Instance,
            requiresPerBlockCapture: true);
        var executor = new BatchExecutor(model, pool, sched, NullLogger.Instance);
        executor.InitializeRadixCache(cfg);
        return (sched, executor, pool);
    }

    // A prompt whose own message runs past the shared page: alone, its last full forward ends at 56.
    private const int LongPromptTokens = SharedTokens + 20;

    /// <summary>A chat turn with a long message after the system prompt, which this request
    /// declares <paramref name="sharedPrefixTokens"/> tokens long.</summary>
    private static SequenceState Long(string id, int sharedPrefixTokens)
    {
        var tokens = Enumerable.Range(1, SharedTokens)
            .Concat(Enumerable.Range(300, LongPromptTokens - SharedTokens))
            .ToList();
        return new SequenceState(id, tokens, 2, BlockSize, SamplingConfig.Greedy,
            sharedPrefixTokens: sharedPrefixTokens);
    }

    /// <summary>Run <paramref name="seq"/> on the idle engine, alone, to completion.</summary>
    private static async Task RunAlone(InferenceEngine engine, SequenceState seq)
    {
        var handle = engine.SubmitRequest(seq);
        var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Null(seq.Error);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
    }

    /// <summary>A burst of <paramref name="count"/> chat turns that reaches the engine before its
    /// next step, run to completion.</summary>
    private static async Task<List<SequenceState>> RunBurst(InferenceEngine engine, ComputeGate gate, int count)
    {
        gate.Close();
        var burst = Enumerable.Range(0, count).Select(i => Shared($"burst{i}", i)).ToList();
        var handles = burst.Select(s => engine.SubmitRequest(s)).ToList();
        gate.Open();
        foreach (var handle in handles)
        {
            var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
            Assert.Null(handle.Sequence.Error);
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        }
        return burst;
    }

    /// <summary>A chat turn: the shared system prompt, then this request's own message.</summary>
    private static SequenceState Shared(string id, int index, IRequestAdapter? adapter = null)
    {
        var tokens = Enumerable.Range(1, SharedTokens)
            .Concat(Enumerable.Range(100 + index * TailTokens, TailTokens))
            .ToList();
        return new SequenceState(id, tokens, 2, BlockSize, SamplingConfig.Greedy,
            sharedPrefixTokens: SharedTokens, adapter: adapter);
    }

    private sealed class StubAdapter : IRequestAdapter
    {
        public string Id => "style#1";
        public bool IsRetired => false;
    }

    /// <summary>
    /// A recurrent page family with a batched route, as Nemotron-H: the linear cache holds token
    /// rows and one running state, so an extracted block carries the state at the end of the
    /// forward that preceded it, and a forward refuses to run on a state that does not match the
    /// rows (a resume from a block that is no restore point). The batched route keeps its rows
    /// in model-paged storage, which carries no state, so it publishes no page. It refuses the
    /// same resume: a request reaches it with cached pages injected into the linear cache and
    /// then migrated (<see cref="TryMigrateLinearKVToPaged"/>), and a batched forward resumes a
    /// request only where that migration, or its own last batched forward, left it.
    /// </summary>
    private sealed class LonePageStubModel : IModelArchitecture, IBatchedPagedModel, IPageOnlyPrefixCacheModel
    {
        private readonly List<int> _rows = new();
        private int _statePosition;
        // Where each request's rows in the batched route's storage end, by request id as the
        // model keys its per-request state; released with the request.
        private readonly Dictionary<string, int> _batchedRowsEnd = new(StringComparer.Ordinal);

        public bool BatchedRoute { get; init; } = true;
        public List<int> LoneForwards { get; } = new();
        public int PromptTokensForwarded { get; private set; }
        /// <summary>Called on the engine thread after each per-sequence forward.</summary>
        public Action? OnForward { get; set; }

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new NumberTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => true;
        public bool RequiresPerBlockCapture => true;
        public string KVStateFingerprint => "lone-page-stub";

        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
        {
            Class = FamilyClass.R,
            NamespaceFingerprint = KVStateFingerprint,
            EndState = EndStateSupport.None,
            PrimaryResident = true,
            Truncation = TruncationKind.None,
            Pages = PageSupport.A1HostSlab,
            PagesNeedStateAtEnd = true,
        };

        public long QuerySpareBytes(ResourceClass cls) => -1;

        public float[] ForwardPrompt(int[] tokens)
        {
            PromptTokensForwarded += tokens.Length;
            return Forward(tokens);
        }

        public float[] Forward(int[] tokens)
        {
            RequireStateAtRowsEnd();
            if (tokens.Length > 1)
                LoneForwards.Add(tokens.Length);
            _rows.AddRange(tokens);
            _statePosition = _rows.Count;
            OnForward?.Invoke();
            return Logits();
        }

        private void RequireStateAtRowsEnd()
        {
            if (_statePosition != _rows.Count)
                throw new InvalidOperationException(
                    $"The running state is at {_statePosition}, the rows end at {_rows.Count}: resumed from no restore point.");
        }

        public void ResetKVCache()
        {
            _rows.Clear();
            _statePosition = 0;
        }

        public void TruncateKVCache(int tokenCount) => throw new NotSupportedException();
        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (tokenCount + 1L) * sizeof(int);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (startToken < 0 || startToken + tokenCount > _rows.Count
                || destination.Length != ComputeKVBlockByteSize(tokenCount))
                return false;
            for (int i = 0; i < tokenCount; i++)
                BitConverter.TryWriteBytes(destination.Slice(i * sizeof(int), sizeof(int)), _rows[startToken + i]);
            BitConverter.TryWriteBytes(destination.Slice(tokenCount * sizeof(int), sizeof(int)), _statePosition);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _rows.Count || source.Length != ComputeKVBlockByteSize(tokenCount))
                return false;
            for (int i = 0; i < tokenCount; i++)
                _rows.Add(BitConverter.ToInt32(source.Slice(i * sizeof(int), sizeof(int))));
            _statePosition = BitConverter.ToInt32(source.Slice(tokenCount * sizeof(int), sizeof(int)));
            return true;
        }

        // ---- IBatchedPagedModel: a batched step keeps its rows where no page is cut from them.

        public bool BatchedForwardAvailable => BatchedRoute;
        public bool SupportsLinearKVMigration => true;
        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize)
        {
            // The linear cache holds what the executor injected from the owner's pages (or what
            // its own forwards left), and the batched route carries on from that state: it must
            // be a restore point, exactly as for a per-sequence Forward.
            RequireStateAtRowsEnd();
            _batchedRowsEnd[owner.RequestId] = _rows.Count;
            return true;
        }

        public void OnSequenceReleased(string requestId) => _batchedRowsEnd.Remove(requestId);

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            var logits = new float[ctx.Sequences.Count][];
            for (int i = 0; i < logits.Length; i++)
            {
                var seq = ctx.Sequences[i];
                // A request with computed tokens resumes from batched rows a migration checked or
                // an earlier batched forward of it wrote, ending exactly where it resumes.
                if (seq.NumComputedTokens > 0)
                {
                    bool held = _batchedRowsEnd.TryGetValue(seq.RequestId, out int rowsEnd);
                    if (!held || rowsEnd != seq.NumComputedTokens)
                        throw new InvalidOperationException(
                            $"{seq.RequestId} resumes the batched route at {seq.NumComputedTokens}, but its batched rows " +
                            (held ? $"end at {rowsEnd}" : "were never migrated") + ": resumed from no restore point.");
                }
                _batchedRowsEnd[seq.RequestId] = seq.NumComputedTokens + ctx.NumScheduledTokens[i];
                if (seq.NumComputedTokens < seq.PromptTokens.Count)
                    PromptTokensForwarded += Math.Min(ctx.NumScheduledTokens[i], seq.PromptTokens.Count - seq.NumComputedTokens);
                logits[i] = Logits();
            }
            return logits;
        }

        private static float[] Logits()
        {
            var logits = new float[VocabSize];
            logits[PeakToken] = 10f;
            return logits;
        }

        private sealed class NumberTokenizer : ITokenizer
        {
            public string[] Vocab { get; } = Enumerable.Range(0, PrefixCacheLonePageProducerTests.VocabSize)
                .Select(i => i.ToString()).ToArray();
            public int BosTokenId => -1;
            public int[] EosTokenIds => Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
