// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The first text-LoRA server made the adapter a field of the shared model, set when a request
// was parsed and cleared when it ended. The engine batches concurrent requests, so a plain
// request running beside a LoRA one answered with the adapter (3 of 3 trials on Nemotron-H 8B),
// a second adapter swapped the first mid-generation, the device tensors were disposed while
// another sequence's forward could read them, and prefix-cache reuse ignored the adapter.
// The adapter now rides on each request's sequence: these tests pin that sequences under
// different adapters never share a forward, that the adapter is active for its own forwards
// only, that a LoRA sequence takes no cached prefix, and that a request whose forward skipped
// an adapter target fails instead of answering with the base model.
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
using Xunit;

namespace InferenceWeb.Tests;

public class TextLoraEngineIsolationTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int BaseToken = 7;

    private sealed record Adapter(string Name, int Token) : ILoraAdapterHandle;

    private static readonly Adapter AdapterA = new("a", 3);
    private static readonly Adapter AdapterB = new("b", 5);

    private static SchedulerConfig Config() => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 64,
        NumBlocks = 64,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = BlockSize,
    };

    private static SequenceState Sequence(string id, ILoraAdapterHandle adapter, int promptLen = 12, int maxNew = 6)
    {
        var sampling = SamplingConfig.Greedy;
        sampling.LoraAdapter = adapter;
        return new SequenceState(id, Enumerable.Range(1, promptLen).ToList(), maxNew, BlockSize, sampling);
    }

    [Fact]
    public void LoraSequence_TakesTheNoCachePolicy()
    {
        var sampling = SamplingConfig.Greedy;
        sampling.LoraAdapter = AdapterA;
        var seq = new SequenceState("lora", Enumerable.Range(1, 20).ToList(), 4, BlockSize, sampling,
            sharedPrefixTokens: 10, cacheScope: "session", publicCheckpointBoundaries: new[] { 5, 10 });

        Assert.Same(AdapterA, seq.LoraAdapter);
        Assert.NotNull(seq.CacheBreakpoints);
        Assert.Empty(seq.CacheBreakpoints);
        Assert.Equal(0, seq.CacheBreakpointLimit);
        Assert.Equal(0, seq.SharedPrefixTokens);
        Assert.Empty(seq.PublicCheckpointBoundaries);

        // The clone a request's sampling goes through keeps the adapter.
        Assert.Same(AdapterA, sampling.Clone().LoraAdapter);
    }

    [Fact]
    public void Scheduler_RunsOneAdapterPerStep_FirstComeFirstServed()
    {
        var cfg = Config();
        var pool = new BlockPool(cfg.NumBlocks, BlockSize, 64);
        var sched = new ContinuousBatchScheduler(cfg, pool, NullLogger.Instance);
        var a1 = Sequence("a1", AdapterA);
        var a2 = Sequence("a2", AdapterA);
        var plain = Sequence("plain", null);
        var a3 = Sequence("a3", AdapterA);
        sched.Submit(a1);
        sched.Submit(a2);
        sched.Submit(plain);
        sched.Submit(a3);

        var step = sched.Schedule();
        // a1 and a2 share the adapter and run together; the plain request waits for them, and
        // a3 - same adapter, but behind the plain one - does not jump the queue.
        Assert.Equal(new[] { "a1", "a2" }, step.ScheduledWork.Select(w => w.Sequence.RequestId).ToArray());
        Assert.True(sched.LoraBarrierHolds(plain));
        Assert.False(sched.LoraBarrierHolds(a3));
        Assert.Equal(2, sched.WaitingCount);
    }

    [Fact]
    public async Task Engine_ConcurrentRequests_EachRunOnlyUnderItsOwnAdapter()
    {
        var model = new LoraStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

        var requests = new (string Id, Adapter Adapter)[]
        {
            ("a-0", AdapterA), ("plain-0", null), ("b-0", AdapterB), ("a-1", AdapterA),
            ("plain-1", null), ("b-1", AdapterB), ("plain-2", null), ("a-2", AdapterA),
        };
        var handles = requests.Select(r => (r, engine.SubmitRequest(Sequence(r.Id, r.Adapter)))).ToList();

        foreach (var ((id, adapter), handle) in handles)
        {
            var tokens = await ReadAllAsync(handle);
            int expected = adapter?.Token ?? BaseToken;
            Assert.Equal(6, tokens.Count);
            Assert.All(tokens, t => Assert.Equal(expected, t));
            var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        }

        // A forward under another adapter would have sampled that adapter's token (or, over
        // K/V computed under another, the token after it). Nothing stays active past a step.
        Assert.True(model.LoraForwards > 0);
        Assert.Null(model.Active);
    }

    [Fact]
    public async Task Engine_LoraRequest_ReusesNoBaseModelPrefix()
    {
        var model = new LoraStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

        var baseSeq = Sequence("base", null, promptLen: 32);
        var baseHandle = engine.SubmitRequest(baseSeq);
        await ReadAllAsync(baseHandle);
        await baseHandle.Completion.WaitAsync(TimeSpan.FromSeconds(10));

        var loraSeq = Sequence("lora", AdapterA, promptLen: 32);
        var loraHandle = engine.SubmitRequest(loraSeq);
        var tokens = await ReadAllAsync(loraHandle);
        await loraHandle.Completion.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(0, loraSeq.PrefixCacheReusedTokens);
        Assert.All(tokens, t => Assert.Equal(AdapterA.Token, t));

        // And the base model does not pick up the K/V the LoRA request left behind.
        var again = Sequence("base-again", null, promptLen: 32);
        var againHandle = engine.SubmitRequest(again);
        Assert.All(await ReadAllAsync(againHandle), t => Assert.Equal(BaseToken, t));
        await againHandle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task Engine_FailsALoraRequestWhoseForwardSkippedATarget_BeforeAnyToken()
    {
        var model = new LoraStubModel { SkipTarget = "blk.0.ffn_down.weight" };
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

        var handle = engine.SubmitRequest(Sequence("lora", AdapterA));
        var tokens = await ReadAllAsync(handle);
        var ex = await Assert.ThrowsAnyAsync<Exception>(() => handle.Completion.WaitAsync(TimeSpan.FromSeconds(10)));

        Assert.Empty(tokens);
        Assert.Contains("blk.0.ffn_down.weight", ex.Message);
        Assert.Contains("'a'", ex.Message);
        Assert.Equal(new[] { "blk.0.ffn_down.weight" }, model.MarkedUnsupported);

        // A plain request is unaffected.
        var plain = engine.SubmitRequest(Sequence("plain", null));
        Assert.All(await ReadAllAsync(plain), t => Assert.Equal(BaseToken, t));
    }

    private static async Task<List<int>> ReadAllAsync(InferenceRequestHandle handle)
    {
        var list = new List<int>();
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        try
        {
            while (await handle.Tokens.WaitToReadAsync(cts.Token))
                while (handle.Tokens.TryRead(out int t))
                    list.Add(t);
        }
        catch (OperationCanceledException) { }
        catch (Exception) { /* a failed request completes its token stream with the error */ }
        return list;
    }

    /// <summary>
    /// A KV-snapshot stub whose logits peak at the active adapter's token (or <see cref="BaseToken"/>),
    /// and whose "K/V" records the adapter it was computed under, so a forward under the wrong
    /// adapter - or K/V reused across adapters - shows up in the sampled tokens.
    /// </summary>
    private sealed class LoraStubModel : IModelArchitecture, ILoraAdapterModel
    {
        private static readonly string[] Targets = { "blk.0.attn_qkv.weight", "blk.0.ffn_down.weight" };
        private readonly List<byte> _state = new();
        private readonly HashSet<string> _applied = new();
        private int _cacheSeqLen;

        public ILoraAdapterHandle Active { get; private set; }
        public string SkipTarget { get; init; }
        public int LoraForwards { get; private set; }
        public List<string> MarkedUnsupported { get; } = new();

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new StubTokenizer();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "lora-stub";

        public float[] Forward(int[] tokens)
        {
            // K/V "computed" under an adapter carries its marker; a token sampled after any
            // position computed under a different adapter is off by one.
            byte marker = (byte)((Active as Adapter)?.Token ?? BaseToken);
            for (int t = 0; t < tokens.Length; t++)
            {
                if (_state.Count > _cacheSeqLen) _state[_cacheSeqLen] = marker; else _state.Add(marker);
                _cacheSeqLen++;
            }
            bool mixed = false;
            for (int i = 0; i < _cacheSeqLen; i++) mixed |= _state[i] != marker;
            if (Active != null) LoraForwards++;
            foreach (string target in Targets)
                if (Active != null && target != SkipTarget) _applied.Add(target);

            var logits = new float[VocabSize];
            logits[mixed ? (marker + 1) % VocabSize : marker] = 10f;
            return logits;
        }

        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int tokenCount) => _cacheSeqLen = Math.Min(_cacheSeqLen, tokenCount);
        public void Dispose() { }
        public long ComputeKVBlockByteSize(int tokenCount) => tokenCount;

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (destination.Length != tokenCount || startToken + tokenCount > _cacheSeqLen) return false;
            for (int i = 0; i < tokenCount; i++) destination[i] = _state[startToken + i];
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _cacheSeqLen || source.Length != tokenCount) return false;
            for (int i = 0; i < tokenCount; i++)
            {
                if (_state.Count > destToken + i) _state[destToken + i] = source[i]; else _state.Add(source[i]);
            }
            _cacheSeqLen = destToken + tokenCount;
            return true;
        }

        public void ActivateLoraAdapter(ILoraAdapterHandle adapter) => Active = adapter;

        public IReadOnlyCollection<string> GetLoraTargets(ILoraAdapterHandle adapter) => Targets;

        public IReadOnlyCollection<string> TakeAppliedLoraTargets()
        {
            var applied = _applied.ToArray();
            _applied.Clear();
            return applied;
        }

        public void MarkLoraUnsupported(ILoraAdapterHandle adapter, IReadOnlyCollection<string> missing)
            => MarkedUnsupported.AddRange(missing);

        private sealed class StubTokenizer : ITokenizer
        {
            public string[] Vocab { get; } = Enumerable.Range(0, TextLoraEngineIsolationTests.VocabSize).Select(i => i.ToString()).ToArray();
            public int BosTokenId => -1;
            public int[] EosTokenIds => Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer)
            {
                foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString())) buffer.Add(b);
            }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
