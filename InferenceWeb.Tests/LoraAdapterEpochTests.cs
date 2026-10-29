// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>
/// Per-request LoRA adapters under continuous batching ("adapter epochs", PR #282 review
/// items 1 and 7): the scheduler never mixes adapters in a step, the executor selects the
/// step's adapter under the GPU lock and always clears it, adapter sequences are
/// cache-isolated, speculation is off for adapter steps, and a retired adapter is released
/// only after its last request. A stub model whose logits peak at a different token for each
/// adapter makes any mixing visible in the sampled tokens.
/// </summary>
public sealed class LoraAdapterEpochTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int BaseToken = 3;

    // ------------------------------------------------------------------ SequenceState

    [Fact]
    public void AdapterSequence_IsCacheIsolated_WhateverTheCallerAsked()
    {
        var adapter = new StubAdapter("a", 1);
        var seq = new SequenceState("s", Enumerable.Range(1, 40).ToList(), 4, BlockSize, SamplingConfig.Greedy,
            cacheBreakpoints: new[] { 16, 32 }, sharedPrefixTokens: 24, cacheScope: "conv",
            publicCheckpointBoundaries: new[] { 8, 24 }, adapter: adapter);

        Assert.Same(adapter, seq.Adapter);
        Assert.NotNull(seq.CacheBreakpoints);
        Assert.Empty(seq.CacheBreakpoints!);
        Assert.Equal(0, seq.CacheBreakpointLimit);
        Assert.Equal(0, seq.SharedPrefixTokens);
        Assert.Empty(seq.PublicCheckpointBoundaries);

        // The base model's sequence keeps the caller's cache policy.
        var plain = new SequenceState("p", Enumerable.Range(1, 40).ToList(), 4, BlockSize, SamplingConfig.Greedy,
            cacheBreakpoints: new[] { 16, 32 }, sharedPrefixTokens: 24, publicCheckpointBoundaries: new[] { 8, 24 });
        Assert.Null(plain.Adapter);
        Assert.Equal(32, plain.CacheBreakpointLimit);
        Assert.Equal(24, plain.SharedPrefixTokens);
        Assert.Equal(new[] { 8, 24 }, plain.PublicCheckpointBoundaries);
    }

    // ------------------------------------------------------------------ scheduler

    [Fact]
    public void Schedule_DoesNotAdmitADifferentAdapterWhileTheRunningSetHasAnother()
    {
        var sched = NewScheduler();
        var lora = new StubAdapter("a", 1);
        var base1 = Seq("base1", null);
        sched.Submit(base1);
        var first = sched.Schedule();
        Assert.Null(first.Adapter);
        Assert.Equal(new[] { base1 }, first.ScheduledWork.Select(w => w.Sequence));

        var adapted = Seq("lora1", lora);
        sched.Submit(adapted);
        for (int i = 0; i < 3; i++)
        {
            var step = sched.Schedule();
            Assert.Null(step.Adapter);
            Assert.DoesNotContain(step.ScheduledWork, w => w.Sequence == adapted);
            Assert.Equal(SequenceStatus.Waiting, adapted.Status);
        }

        sched.NotifyStop(base1, SequenceStatus.FinishedStopped, "stop", new SchedulerOutput());
        var next = sched.Schedule();
        Assert.Same(lora, next.Adapter);
        Assert.Equal(new[] { adapted }, next.ScheduledWork.Select(w => w.Sequence));
        Assert.True(sched.ReferencesAdapter(lora));
    }

    [Fact]
    public void Schedule_CoAdmitsRequestsOfTheSameAdapter()
    {
        var sched = NewScheduler();
        var lora = new StubAdapter("a", 1);
        var a = Seq("a1", lora);
        var b = Seq("a2", lora);
        sched.Submit(a);
        sched.Submit(b);

        var step = sched.Schedule();

        Assert.Same(lora, step.Adapter);
        Assert.Equal(new[] { a, b }, step.ScheduledWork.Select(w => w.Sequence));
    }

    [Fact]
    public void Schedule_AdapterSwitchIsFirstComeFirstServed_SoNeitherSideStarves()
    {
        var sched = NewScheduler();
        var lora = new StubAdapter("a", 1);
        var base1 = Seq("base1", null);
        var adapted = Seq("lora1", lora);
        var base2 = Seq("base2", null);
        sched.Submit(base1);
        sched.Schedule();
        sched.Submit(adapted);
        sched.Submit(base2);

        // base2 shares base1's (base) epoch, but it arrived after the LoRA request: it waits.
        var step = sched.Schedule();
        Assert.Equal(new[] { base1 }, step.ScheduledWork.Select(w => w.Sequence));
        Assert.Equal(SequenceStatus.Waiting, base2.Status);

        sched.NotifyStop(base1, SequenceStatus.FinishedStopped, "stop", new SchedulerOutput());
        step = sched.Schedule();
        Assert.Same(lora, step.Adapter);
        Assert.Equal(new[] { adapted }, step.ScheduledWork.Select(w => w.Sequence));
        Assert.Equal(SequenceStatus.Waiting, base2.Status);

        sched.NotifyStop(adapted, SequenceStatus.FinishedStopped, "stop", new SchedulerOutput());
        step = sched.Schedule();
        Assert.Null(step.Adapter);
        Assert.Equal(new[] { base2 }, step.ScheduledWork.Select(w => w.Sequence));
        Assert.False(sched.ReferencesAdapter(lora));
    }

    [Fact]
    public void Schedule_AReloadedAdapterIsANewEpoch_EvenUnderTheSameName()
    {
        var sched = NewScheduler();
        var old = new StubAdapter("style", 1);
        var reloaded = new StubAdapter("style", 2);
        var a = Seq("old", old);
        var b = Seq("new", reloaded);
        sched.Submit(a);
        sched.Submit(b);

        var step = sched.Schedule();
        Assert.Same(old, step.Adapter);
        Assert.Equal(new[] { a }, step.ScheduledWork.Select(w => w.Sequence));

        sched.NotifyStop(a, SequenceStatus.FinishedStopped, "stop", new SchedulerOutput());
        step = sched.Schedule();
        Assert.Same(reloaded, step.Adapter);
        Assert.Equal(new[] { b }, step.ScheduledWork.Select(w => w.Sequence));
    }

    [Fact]
    public void Schedule_AParkedVictimRejoinsItsEpochBeforeAnotherAdapter()
    {
        var sched = NewScheduler();
        var loraA = new StubAdapter("a", 1);
        var loraB = new StubAdapter("b", 2);
        var a1 = Seq("a1", loraA);
        var a2 = Seq("a2", loraA);
        var b1 = Seq("b1", loraB);
        sched.Submit(a1);
        sched.Submit(a2);
        sched.Submit(b1);
        var step = sched.Schedule();
        Assert.Equal(new[] { a1, a2 }, step.ScheduledWork.Select(w => w.Sequence));

        // a2 loses its place (no device slot) and is parked at the front of the queue.
        var parked = new SchedulerOutput();
        sched.DeferForSequenceSlot(a2, parked, contextTokens: 64);
        sched.NotifyStop(a1, SequenceStatus.FinishedStopped, "stop", new SchedulerOutput());

        step = sched.Schedule();
        Assert.Same(loraA, step.Adapter);
        Assert.Equal(new[] { a2 }, step.ScheduledWork.Select(w => w.Sequence));
        Assert.Equal(SequenceStatus.Waiting, b1.Status);

        sched.NotifyStop(a2, SequenceStatus.FinishedStopped, "stop", new SchedulerOutput());
        step = sched.Schedule();
        Assert.Same(loraB, step.Adapter);
        Assert.Equal(new[] { b1 }, step.ScheduledWork.Select(w => w.Sequence));
    }

    [Theory]
    [InlineData("a", "b")]
    [InlineData("a", null)]
    [InlineData(null, "a")]
    public void Schedule_WaitersTheEpochHoldsBack_DoNotShrinkTheRunningPromptsStep(string? running, string? waiting)
    {
        var sched = NewScheduler(numBlocks: 512);
        var adapters = new Dictionary<string, IRequestAdapter> { ["a"] = new StubAdapter("a", 1), ["b"] = new StubAdapter("b", 2) };
        var longPrompt = new SequenceState("long", Prompt(2000), 4, BlockSize, SamplingConfig.Greedy,
            adapter: running == null ? null : adapters[running]);
        sched.Submit(longPrompt);
        Assert.Equal(256, Assert.Single(sched.Schedule().ScheduledWork).NumScheduledTokens);

        var waiters = Enumerable.Range(0, 3).Select(i => Seq($"w{i}", waiting == null ? null : adapters[waiting])).ToList();
        foreach (var w in waiters) sched.Submit(w);

        // Nothing else can run until the long prompt drains, so it keeps the whole budget
        // (the solo cap), not a quarter of it shared with requests that cannot be admitted.
        for (int i = 0; i < 3; i++)
        {
            var work = Assert.Single(sched.Schedule().ScheduledWork);
            Assert.Same(longPrompt, work.Sequence);
            Assert.Equal(256, work.NumScheduledTokens);
        }
        Assert.All(waiters, w => Assert.Equal(SequenceStatus.Waiting, w.Status));
    }

    [Fact]
    public void Schedule_WaitersOfTheRunningEpoch_StillShareTheStep()
    {
        var sched = NewScheduler(numBlocks: 512);
        var lora = new StubAdapter("a", 1);
        var longPrompt = new SequenceState("long", Prompt(2000), 4, BlockSize, SamplingConfig.Greedy, adapter: lora);
        sched.Submit(longPrompt);
        sched.Schedule();
        var waiters = Enumerable.Range(0, 3).Select(i => Seq($"w{i}", lora)).ToList();
        foreach (var w in waiters) sched.Submit(w);

        var step = sched.Schedule();

        // Four contenders: the even share of the 256-token budget.
        Assert.Equal(new[] { longPrompt }.Concat(waiters), step.ScheduledWork.Select(w => w.Sequence));
        Assert.Equal(64, step.ScheduledWork[0].NumScheduledTokens);
    }

    [Fact]
    public void Schedule_BaseOnlyTraffic_IsAdmittedTogetherWithNoAdapter()
    {
        var sched = NewScheduler();
        var seqs = Enumerable.Range(0, 4).Select(i => Seq($"r{i}", null)).ToList();
        foreach (var s in seqs) sched.Submit(s);

        for (int i = 0; i < 3; i++)
        {
            var step = sched.Schedule();
            Assert.Null(step.Adapter);
            Assert.Equal(seqs, step.ScheduledWork.Select(w => w.Sequence));
        }
    }

    // ------------------------------------------------------------------ planner

    [Fact]
    public void Planner_DoesNotSpeculateForAnAdapterStep_ButStillDoesForBase()
    {
        var caps = new ExecutionCapabilities
        {
            SupportsSpeculativeTrunk = true,
            HasDraftHead = true,
            SpeculationProfitable = true,
        };
        var config = new SchedulerConfig { Speculation = new SpeculationOptions { Enabled = true } };

        var basePlan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, config,
            new ExecutionStepFeatures { SequenceCount = 1 });
        var adapterPlan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, config,
            new ExecutionStepFeatures { SequenceCount = 1, AdapterActive = true });

        Assert.Equal(ExecutionPathKind.SpeculativePerSequence, basePlan.Selected);
        Assert.DoesNotContain(ExecutionPathKind.SpeculativePerSequence, adapterPlan.Candidates);
        Assert.Contains(adapterPlan.Rejections, r => r.Path == ExecutionPathKind.SpeculativePerSequence
            && r.Reason.Contains("LoRA adapter", StringComparison.Ordinal));
    }

    // ------------------------------------------------------------------ executor

    [Fact]
    public void Executor_RefusesAStepThatMixesAdapters_BeforeAnythingRuns()
    {
        var model = new AdapterStubModel();
        var lora = new StubAdapter("a", 1);
        var (executor, pool) = NewExecutor(model);
        var output = new SchedulerOutput { Adapter = lora };
        output.ScheduledWork.Add(Prefill(Seq("lora", lora), pool));
        output.ScheduledWork.Add(Prefill(Seq("base", null), pool));

        var ex = Assert.Throws<InvalidOperationException>(() => executor.ExecuteStep(output));

        Assert.Contains("one request adapter", ex.Message);
        Assert.Equal(0, model.ForwardCount);
        Assert.Equal(0, model.SelectCalls);
        Assert.Null(model.StepAdapter);
    }

    [Fact]
    public void Executor_BindsSelectsAndClearsTheAdapter_EvenWhenTheForwardFails()
    {
        var model = new AdapterStubModel { ThrowWhenAdapterSelected = true };
        var lora = new StubAdapter("a", 1);
        var (executor, pool) = NewExecutor(model);
        var output = new SchedulerOutput { Adapter = lora };
        output.ScheduledWork.Add(Prefill(Seq("lora", lora), pool));

        try { executor.ExecuteStep(output); }
        catch (InvalidOperationException) { }

        Assert.True(model.ForwardCount > 0);
        Assert.Contains(lora, model.BoundAdapters);   // bound lazily on the worker, once
        Assert.Equal(1, model.BindCount);
        Assert.All(model.Forwards, f => Assert.Same(lora, f.Adapter));
        Assert.All(model.Forwards, f => Assert.True(f.LockHeld));
        Assert.Null(model.StepAdapter);
    }

    [Fact]
    public void Executor_BaseStep_NeverTouchesAdapterSelection()
    {
        var model = new AdapterStubModel();
        var (executor, pool) = NewExecutor(model);
        var output = new SchedulerOutput();
        output.ScheduledWork.Add(Prefill(Seq("base", null), pool));

        var results = executor.ExecuteStep(output);

        Assert.Single(results);
        Assert.Null(results[0].Error);
        Assert.Equal(0, model.SelectCalls);
        Assert.Equal(0, model.BindCount);
        Assert.All(model.Forwards, f => Assert.Null(f.Adapter));
    }

    // ------------------------------------------------------------------ engine

    [Fact]
    public async Task Engine_ConcurrentBaseAndAdapterRequests_NeverMix()
    {
        var model = new AdapterStubModel { ForwardDelayMs = 1 };
        var loraA = new StubAdapter("a", 1);
        var loraB = new StubAdapter("b", 2);
        model.PeakFor[loraA] = 5;
        model.PeakFor[loraB] = 7;
        using var engine = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance);

        // Another GPU user (a media encoder between steps) must never see a selection.
        int observedSelections = 0;
        using var stop = new CancellationTokenSource();
        var observer = Task.Run(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                lock (model.GpuComputeLock)
                {
                    if (model.StepAdapter != null)
                        Interlocked.Increment(ref observedSelections);
                }
                Thread.Sleep(0);
            }
        });

        var adapters = new IRequestAdapter?[] { null, loraA, null, loraB, loraA, null, loraB, null };
        var seqs = adapters.Select((a, i) => new SequenceState($"r{i}", Prompt(10 + i), 6, BlockSize,
            SamplingConfig.Greedy, cacheScope: $"c{i}", adapter: a)).ToList();
        var handles = seqs.Select(s => engine.SubmitRequest(s)).ToList();
        foreach (var handle in handles)
            await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
        stop.Cancel();
        await observer;

        for (int i = 0; i < seqs.Count; i++)
        {
            int expected = adapters[i] == null ? BaseToken : model.PeakFor[adapters[i]!];
            Assert.Equal(6, seqs[i].OutputTokens.Count);
            Assert.All(seqs[i].OutputTokens, t => Assert.Equal(expected, t));
        }
        Assert.Equal(0, observedSelections);
        Assert.All(model.Forwards, f => Assert.True(f.Adapter == null || f.LockHeld));
        Assert.Null(model.StepAdapter);
    }

    [Fact]
    public async Task Engine_ReleasesARetiredAdapter_OnlyAfterItsLastRequest_AndOnce()
    {
        using var entered = new ManualResetEventSlim();
        using var proceed = new ManualResetEventSlim();
        var model = new AdapterStubModel { AdapterForwardEntered = entered, AdapterForwardGate = proceed };
        var lora = new StubAdapter("a", 1);
        model.PeakFor[lora] = 5;
        using var engine = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance);

        var seq = new SequenceState("held", Prompt(12), 4, BlockSize, SamplingConfig.Greedy, adapter: lora);
        var handle = engine.SubmitRequest(seq);
        Assert.True(entered.Wait(TimeSpan.FromSeconds(10)));

        // Unloaded mid-request: the request keeps its adapter to the end.
        lora.IsRetired = true;
        engine.ReleaseRetiredAdapters();
        Assert.Equal(0, model.ReleaseCount(lora));
        proceed.Set();

        var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        Assert.All(seq.OutputTokens, t => Assert.Equal(5, t));

        var until = DateTime.UtcNow.AddSeconds(10);
        while (model.ReleaseCount(lora) == 0 && DateTime.UtcNow < until)
            await Task.Delay(5);
        await Task.Delay(50);
        Assert.Equal(1, model.ReleaseCount(lora));
        Assert.False(model.ForwardAfterRelease);
        Assert.DoesNotContain(lora, model.BoundAdapters);
    }

    [Fact]
    public async Task Engine_AdapterRequestNeitherReusesNorPublishesPrefixCacheState()
    {
        var model = new AdapterStubModel();
        var lora = new StubAdapter("a", 1);
        model.PeakFor[lora] = 5;
        using var engine = new InferenceEngine(model, Config(prefixCaching: true), NullLogger.Instance);
        Assert.NotNull(engine.RadixCache);
        var prompt = Prompt(40);

        var first = await Run(engine, "base1", prompt, null);
        Assert.All(first.Seq.OutputTokens, t => Assert.Equal(BaseToken, t));
        ResourceVector cachedBefore = engine.RadixCache!.Tree.Cached;

        // Same conversation scope, same prompt: still nothing reused, nothing published.
        var adapted = await Run(engine, "lora1", prompt, lora);
        Assert.Equal(0, adapted.Completion.PrefixCacheReusedTokens);
        Assert.All(adapted.Seq.OutputTokens, t => Assert.Equal(5, t));
        Assert.Equal(cachedBefore, engine.RadixCache.Tree.Cached);

        // The next base request reuses only what the base model produced.
        var second = await Run(engine, "base2", prompt, null);
        Assert.True(second.Completion.PrefixCacheReusedTokens > 0,
            "a base request should still reuse the base request's pages");
        Assert.All(second.Seq.OutputTokens, t => Assert.Equal(BaseToken, t));
    }

    // ------------------------------------------------------------------ helpers

    private static async Task<(SequenceState Seq, InferenceCompletion Completion)> Run(
        InferenceEngine engine, string id, List<int> prompt, IRequestAdapter? adapter)
    {
        var seq = new SequenceState(id, prompt, 3, BlockSize, SamplingConfig.Greedy, cacheScope: "conv", adapter: adapter);
        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        return (seq, completion);
    }

    private static List<int> Prompt(int length) => Enumerable.Range(0, length).Select(i => 1 + i % (VocabSize - 2)).ToList();

    private static SequenceState Seq(string id, IRequestAdapter? adapter)
        => new(id, Prompt(12), 4, BlockSize, SamplingConfig.Greedy, adapter: adapter);

    private static SchedulerConfig Config(bool prefixCaching, int numBlocks = 256) => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 64,
        NumBlocks = numBlocks,
        BlockSize = BlockSize,
        EnablePrefixCaching = prefixCaching,
        DecodeQuantumTokens = 2,
        StopRepetition = false,
        Speculation = SpeculationOptions.Disabled,
    };

    private static ContinuousBatchScheduler NewScheduler(int numBlocks = 256)
    {
        var cfg = Config(prefixCaching: false, numBlocks);
        return new ContinuousBatchScheduler(cfg, new BlockPool(cfg.NumBlocks, BlockSize, 4L * BlockSize), NullLogger.Instance);
    }

    private static (BatchExecutor Executor, BlockPool Pool) NewExecutor(AdapterStubModel model)
    {
        var cfg = Config(prefixCaching: false);
        var pool = new BlockPool(cfg.NumBlocks, BlockSize, model.ComputeKVBlockByteSize(BlockSize));
        var scheduler = new ContinuousBatchScheduler(cfg, pool, NullLogger.Instance);
        return (new BatchExecutor(model, pool, scheduler, NullLogger.Instance), pool);
    }

    private static ScheduledSequenceWork Prefill(SequenceState seq, BlockPool pool)
    {
        foreach (var block in pool.AllocateNew((seq.PromptTokens.Count + BlockSize - 1) / BlockSize)!)
            seq.BlockTable.AppendBlock(block);
        seq.Status = SequenceStatus.Running;
        return new ScheduledSequenceWork(seq, seq.PromptTokens.Count, isNewAdmission: true, isPrefill: true);
    }

    private sealed class StubAdapter(string name, long generation) : IRequestAdapter
    {
        public string Id { get; } = $"{name}#{generation}";
        public bool IsRetired { get; set; }
        public override string ToString() => Id;
    }

    /// <summary>
    /// One linear cache of token rows (a full-attention page family). Its logits peak at
    /// <see cref="BaseToken"/> for the base model and at <see cref="PeakFor"/> for the adapter
    /// selected when it forwards, so a forward that ran with the wrong adapter samples a
    /// visibly wrong token. Records the selection and whether the GPU lock was held.
    /// </summary>
    private sealed class AdapterStubModel : IModelArchitecture, IAdapterAwareModel, IPageOnlyPrefixCacheModel
    {
        private readonly List<int> _rows = new();
        private readonly HashSet<IRequestAdapter> _bound = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<IRequestAdapter, int> _releases = new(ReferenceEqualityComparer.Instance);
        private readonly List<(IRequestAdapter? Adapter, bool LockHeld)> _forwards = new();
        private IRequestAdapter? _selected;
        private bool _gateUsed;

        public Dictionary<IRequestAdapter, int> PeakFor { get; } = new(ReferenceEqualityComparer.Instance);
        public bool ThrowWhenAdapterSelected { get; init; }
        public int ForwardDelayMs { get; init; }
        public ManualResetEventSlim? AdapterForwardEntered { get; init; }
        public ManualResetEventSlim? AdapterForwardGate { get; init; }
        public int SelectCalls { get; private set; }
        public int BindCount { get; private set; }
        public bool ForwardAfterRelease { get; private set; }

        public object GpuComputeLock { get; } = new();
        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new NumberTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "lora-adapter-stub";

        public IReadOnlyList<(IRequestAdapter? Adapter, bool LockHeld)> Forwards
        {
            get { lock (_forwards) return _forwards.ToArray(); }
        }

        public int ForwardCount
        {
            get { lock (_forwards) return _forwards.Count; }
        }

        public int ReleaseCount(IRequestAdapter adapter)
        {
            lock (_releases) return _releases.TryGetValue(adapter, out int n) ? n : 0;
        }

        public float[] Forward(int[] tokens)
        {
            var selected = _selected;
            lock (_forwards)
                _forwards.Add((selected, Monitor.IsEntered(GpuComputeLock)));
            if (selected != null)
            {
                lock (_releases)
                    if (_releases.ContainsKey(selected)) ForwardAfterRelease = true;
                if (ThrowWhenAdapterSelected)
                    throw new InvalidOperationException("forward failed with the adapter selected");
                if (AdapterForwardGate != null && !_gateUsed)
                {
                    _gateUsed = true;
                    AdapterForwardEntered?.Set();
                    if (!AdapterForwardGate.Wait(TimeSpan.FromSeconds(10)))
                        throw new TimeoutException("The test never released the adapter forward.");
                }
            }
            if (ForwardDelayMs > 0) Thread.Sleep(ForwardDelayMs);
            _rows.AddRange(tokens);
            var logits = new float[VocabSize];
            logits[selected == null ? BaseToken : PeakFor[selected]] = 10f;
            return logits;
        }

        public void ResetKVCache() => _rows.Clear();

        public void TruncateKVCache(int tokenCount)
        {
            if (tokenCount < _rows.Count) _rows.RemoveRange(tokenCount, _rows.Count - tokenCount);
        }

        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (long)tokenCount * sizeof(int);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (startToken < 0 || startToken + tokenCount > _rows.Count || destination.Length < tokenCount * sizeof(int)) return false;
            for (int i = 0; i < tokenCount; i++)
                BitConverter.TryWriteBytes(destination.Slice(i * sizeof(int), sizeof(int)), _rows[startToken + i]);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _rows.Count) return false;
            for (int i = 0; i < tokenCount; i++)
                _rows.Add(BitConverter.ToInt32(source.Slice(i * sizeof(int), sizeof(int))));
            return true;
        }

        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
        {
            Class = FamilyClass.P,
            NamespaceFingerprint = KVStateFingerprint,
            EndState = EndStateSupport.None,
            PrimaryResident = true,
            Truncation = TruncationKind.Any,
            RewindCapTokens = 16,
            Pages = PageSupport.A1HostSlab,
        };

        public long QuerySpareBytes(ResourceClass cls) => -1;

        // ---- IAdapterAwareModel

        public bool TryBindAdapter(IRequestAdapter adapter, out string? error)
        {
            error = null;
            lock (_bound)
            {
                if (_bound.Add(adapter)) BindCount++;
            }
            return true;
        }

        public void SelectStepAdapter(IRequestAdapter? adapter)
        {
            SelectCalls++;
            if (adapter != null)
            {
                lock (_bound)
                    if (!_bound.Contains(adapter)) throw new InvalidOperationException("not bound");
            }
            _selected = adapter;
        }

        public IRequestAdapter? StepAdapter => _selected;

        public IReadOnlyCollection<IRequestAdapter> BoundAdapters
        {
            get { lock (_bound) return _bound.ToArray(); }
        }

        public void ReleaseAdapter(IRequestAdapter adapter)
        {
            if (ReferenceEquals(_selected, adapter))
                throw new InvalidOperationException("selected");
            lock (_bound) _bound.Remove(adapter);
            lock (_releases) _releases[adapter] = ReleaseCount(adapter) + 1;
        }

        private sealed class NumberTokenizer : ITokenizer
        {
            public string[] Vocab { get; } = Enumerable.Range(0, LoraAdapterEpochTests.VocabSize).Select(i => i.ToString()).ToArray();
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
