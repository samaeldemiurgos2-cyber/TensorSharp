// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// DeepSeek V4.1 multi-turn THINKING reuse through the path the CLI and the server take today: the
// inference engine with the radix prefix tree (the default since 2026-09-17). The older
// DeepSeek41MultiTurnKvReuseTests pin KVCache.PlanReuse, which the CLI stopped calling that day,
// so they kept passing while the engine reused nothing.
//
// The report: `TensorSharp.Cli --interactive --think --backend ggml_cuda --tp 6` on
// DeepSeek-V4.1-Flash, "Please introduce Final Fantasy 7 in details", then "continue" - and the
// second turn showed KV reuse 0. V4.1 re-renders a past answer without its reasoning, so the second
// prompt diverges from the cache one token after the previous <|Assistant|> (the cache holds <think>,
// the render </think>), and keeping the previous prompt means rewinding the primary past the whole
// answer. The tree's donation slack refused any rewind over 16 tokens, even for a primary that a
// decline could not keep: the next step invalidates it either way. Nothing about it was TP-specific.
//
// The model is OracleFakes.NLive - DeepSeek V4.1's prefix-cache capabilities without retained slots
// (a DSpark drafter loaded), including the checkpoint its native slot takes at every prompt boundary -
// behind the real deepseek41 chat template and a one-token-per-character tokenizer. The oracle's state
// hashes every (token, position), so a reuse that is not exact changes the greedy output, and every
// turn is compared with a cold run.
using System.Text;
using TensorSharp;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Cli;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41ThinkingTurnReuseTests
{
    private const int Vocab = char.MaxValue + 1;
    private const int MaxNew = 24;
    private const string ThinkOpen = "<think>";

    private static readonly string[] Asks = ["Please introduce Final Fantasy 7 in details", "continue", "continue"];
    private static readonly string[] Reasoning = ["The user wants an overview.", "Keep going.", "More detail."];
    private static readonly string[] Answers =
        ["Final Fantasy VII is a 1997 role-playing game by Square.", "Its story follows Cloud Strife.", "It sold over 14 million copies."];

    /// <summary>One token per character, so a token index is a character index.</summary>
    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => [];
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(i => (char)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
            => buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }

    private sealed class RecordingLogger : Microsoft.Extensions.Logging.ILogger
    {
        private readonly List<string> _messages = new();
        public IReadOnlyList<string> Messages { get { lock (_messages) return _messages.ToArray(); } }
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
        public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, Microsoft.Extensions.Logging.EventId eventId,
            TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            lock (_messages) _messages.Add(formatter(state, exception));
        }
    }

    private static SchedulerConfig Config => new()
    {
        BlockSize = 16,
        NumBlocks = 128,
        MaxNumRunningSequences = 1,
        MaxNumBatchedTokens = 128,
        SoloPrefillChunkSize = 128,
        EnablePrefixCaching = true,
        StopRepetition = false,
    };

    private static List<int> Render(ITokenizer tokenizer, List<ChatMessage> history)
        => new KVCachePromptRenderer(new GgufPromptRenderer())
            .RenderToTokens(tokenizer, null, history, "deepseek41", addGenerationPrompt: true, enableThinking: true);

    /// <summary>The answer a fresh model with no cache gives for <paramref name="prompt"/>.</summary>
    private static List<int> Cold(ITokenizer tokenizer, List<int> prompt, CancellationToken token)
    {
        using var model = OracleFakes.NLive(vocab: Vocab, tokenizer: tokenizer);
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        return session.Generate(prompt, MaxNew, SamplingConfig.Greedy, cancellationToken: token,
            enablePrefixCache: false).Tokens;
    }

    private static int CommonPrefix(IReadOnlyList<int> a, IReadOnlyList<int> b)
    {
        int n = 0;
        while (n < a.Count && n < b.Count && a[n] == b[n]) n++;
        return n;
    }

    [Fact]
    public void EveryThinkingTurnReusesThePreviousPrompt_AndAnswersLikeAColdRun()
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var tokenizer = new CharacterTokenizer();
        var logger = new RecordingLogger();
        using var model = OracleFakes.NLive(vocab: Vocab, tokenizer: tokenizer);
        using var session = new CliInferenceSession(model, Config, logger);

        var history = new List<ChatMessage>();
        var cached = new List<int>();   // what the primary holds after each turn: prompt + every forwarded token
        for (int turn = 0; turn < Asks.Length; turn++)
        {
            history.Add(new ChatMessage { Role = "user", Content = Asks[turn] });
            List<int> prompt = Render(tokenizer, history);
            var result = session.Generate(prompt, MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);

            if (turn == 0)
            {
                Assert.Equal(0, result.Completion.PrefixCacheReusedTokens);
            }
            else
            {
                // The render diverges from the cache at the think marker after the previous prompt's
                // last <|Assistant|> ("<" is shared here because this tokenizer is per character; in
                // V4.1's vocabulary <think> and </think> are single tokens). Everything before it is
                // reused, aligned down to the compressor's 2 - that is the whole previous prompt.
                int common = CommonPrefix(cached, prompt);
                Assert.Equal(cached.Count - MaxNew - ThinkOpen.Length + 1, common);
                Assert.Equal(common - common % 2, result.Completion.PrefixCacheReusedTokens);
            }
            Assert.Equal(Cold(tokenizer, prompt, timeout.Token), result.Tokens);

            cached = prompt.Concat(result.Tokens).ToList();
            // What InteractiveSession keeps. The oracle's tokens are not language, so the history carries
            // fixed text; V4.1 never splices raw assistant tokens (AllowRawAssistantTokenSplicing=false).
            history.Add(new ChatMessage
            {
                Role = "assistant",
                Thinking = Reasoning[turn],
                Content = Answers[turn],
                RawOutputTokens = result.Tokens,
            });
        }

        // Each reuse came from the primary kept past the slack, which the model vouched for at admission,
        // and no admission promised reuse that the execution then took back.
        Assert.True(model.PrimaryRewindChecks >= Asks.Length - 1, $"asked {model.PrimaryRewindChecks} time(s)");
        Assert.DoesNotContain(logger.Messages, m => m.Contains("declined at execution time", StringComparison.Ordinal)
                                                    || m.Contains("was revoked before execution", StringComparison.Ordinal));
    }

    [Fact]
    public void ARewindTheSlotCannotReach_IsDeclinedAtAdmission_AndStillAnswersLikeAColdRun()
    {
        // Editing the first message moves the divergence far below the prompt-boundary checkpoint (the
        // model reaches NativeRewindSpan tokens below it), so the model refuses at admission and the turn
        // re-prefills - it is not promised reuse that the execution-time truncate would then retract.
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var tokenizer = new CharacterTokenizer();
        var logger = new RecordingLogger();
        using var model = OracleFakes.NLive(vocab: Vocab, tokenizer: tokenizer);
        using var session = new CliInferenceSession(model, Config, logger);

        var first = new List<ChatMessage> { new() { Role = "user", Content = Asks[0] } };
        session.Generate(Render(tokenizer, first), MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);

        var edited = new List<ChatMessage> { new() { Role = "user", Content = "Please introduce Final Fantasy 8 in details" } };
        List<int> prompt = Render(tokenizer, edited);
        var result = session.Generate(prompt, MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);

        Assert.Equal(0, result.Completion.PrefixCacheReusedTokens);
        Assert.True(model.PrimaryRewindChecks >= 1, "the model was never asked about the rewind");
        Assert.Equal(Cold(tokenizer, prompt, timeout.Token), result.Tokens);
        Assert.DoesNotContain(logger.Messages, m => m.Contains("declined at execution time", StringComparison.Ordinal));
        // And the zero says why, where it used to say only "0/N tokens".
        Assert.Contains(logger.Messages, m => m.StartsWith("Radix prompt reuse for ", StringComparison.Ordinal)
                                              && m.Contains("(rewinding the cached conversation is declined by the model)", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("Please introduce Final Fantasy 7 in details")]
    [InlineData("Please introduce Final Fantasy 7 in detail")]   // one token shorter: the other parity
    public void ARegeneratedTurn_LeavesACheckpoint_SoTheNextThinkingTurnStillReuses(string ask)
    {
        // A regenerate re-sends the same prompt. When the aligned reuse left ONE prompt token to forward,
        // that forward recorded no checkpoint and the rewind had dropped the old one, so the following
        // thinking turn - a rewind past the regenerated answer - could not be served and re-prefilled.
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var tokenizer = new CharacterTokenizer();
        var logger = new RecordingLogger();
        using var model = OracleFakes.NLive(vocab: Vocab, tokenizer: tokenizer);
        using var session = new CliInferenceSession(model, Config, logger);

        var history = new List<ChatMessage> { new() { Role = "user", Content = ask } };
        List<int> prompt = Render(tokenizer, history);
        session.Generate(prompt, MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        var regenerated = session.Generate(prompt, MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        Assert.InRange(regenerated.Completion.PrefixCacheReusedTokens, prompt.Count - 3, prompt.Count - 2);
        Assert.Equal(Cold(tokenizer, prompt, timeout.Token), regenerated.Tokens);

        history.Add(new ChatMessage { Role = "assistant", Thinking = Reasoning[0], Content = Answers[0], RawOutputTokens = regenerated.Tokens });
        history.Add(new ChatMessage { Role = "user", Content = "continue" });
        List<int> next = Render(tokenizer, history);
        var continued = session.Generate(next, MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        int common = prompt.Count - ThinkOpen.Length + 1;
        Assert.Equal(common - common % 2, continued.Completion.PrefixCacheReusedTokens);
        Assert.Equal(Cold(tokenizer, next, timeout.Token), continued.Tokens);
    }

    [Fact]
    public async Task ARequestWithCacheBreakpoints_IsNotPromisedTheLiveCacheItCannotContinue()
    {
        // The executor never continues the live cache for a request that carries explicit cache
        // breakpoints. Admission used to plan it anyway - "Radix prompt reuse ...: 186/277" - and execution
        // then retracted it, blaming "another sequence" that did not exist.
        var tokenizer = new CharacterTokenizer();
        var logger = new RecordingLogger();
        using var model = OracleFakes.NLive(vocab: Vocab, tokenizer: tokenizer);
        using var engine = new InferenceEngine(model, Config, logger);
        var history = new List<ChatMessage> { new() { Role = "user", Content = Asks[0] } };
        List<int> first = Render(tokenizer, history);
        var turn1 = new SequenceState("t1", first, MaxNew, 16, SamplingConfig.Greedy, cacheScope: "conversation");
        await engine.SubmitRequest(turn1).Completion.WaitAsync(TimeSpan.FromSeconds(30));

        history.Add(new ChatMessage { Role = "assistant", Thinking = Reasoning[0], Content = Answers[0], RawOutputTokens = turn1.OutputTokens.ToList() });
        history.Add(new ChatMessage { Role = "user", Content = "continue" });
        List<int> next = Render(tokenizer, history);
        var turn2 = new SequenceState("t2", next, MaxNew, 16, SamplingConfig.Greedy,
            cacheBreakpoints: new[] { next.Count - 1 }, cacheScope: "conversation");
        var done = await engine.SubmitRequest(turn2).Completion.WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(0, done.PrefixCacheReusedTokens);
        Assert.DoesNotContain(logger.Messages, m => m.Contains("no longer valid at execution time", StringComparison.Ordinal));
        Assert.Contains(logger.Messages, m => m.StartsWith("Radix prompt reuse for t2", StringComparison.Ordinal)
                                              && m.Contains("explicit cache breakpoints", StringComparison.Ordinal));
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        Assert.Equal(Cold(tokenizer, next, timeout.Token), turn2.OutputTokens);
    }

    [Fact]
    public async Task RetainedSlots_OverlappingThinkingConversations_EachReusesItsOwnPreviousPrompt()
    {
        // A server whose conversations overlap (retention is always on): every turn runs on its own
        // slot, retained when it finishes. The next thinking turn has to rewind that slot past the previous
        // answer, which the donation slack refused, so even with retention on overlapping conversations
        // re-prefilled every thinking turn. Both conversations' turns are admitted in the same step.
        var tokenizer = new CharacterTokenizer();
        OracleTraits traits = OracleFakes.N().Traits with { RewindCheckpoint = true, MinTailPrefillTokens = 2 };
        using var model = new OracleModel(traits, 16, Vocab, tokenizer);
        var config = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 256, MaxNumRunningSequences = 4, MaxNumBatchedTokens = 128,
            SoloPrefillChunkSize = 128, EnablePrefixCaching = true,
            StopRepetition = false,
        };
        var logger = new RecordingLogger();
        using var engine = new InferenceEngine(model, config, logger);
        var histories = new[]
        {
            new List<ChatMessage> { new() { Role = "user", Content = Asks[0] } },
            new List<ChatMessage> { new() { Role = "user", Content = "Please introduce Chrono Trigger in details" } },
        };
        var previous = new List<int>[histories.Length];
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        for (int turn = 0; turn < 3; turn++)
        {
            List<int>[] prompts = histories.Select(h => Render(tokenizer, h)).ToArray();
            SequenceState[] requests = prompts.Select((p, i) => new SequenceState($"c{i}t{turn}", p, MaxNew, 16,
                SamplingConfig.Greedy, cacheScope: $"conversation{i}")).ToArray();
            var gate = new ComputeGate();
            gate.Close();
            engine.ComputeGate = gate;
            var handles = requests.Select(r => engine.SubmitRequest(r)).ToArray();
            gate.Open();
            InferenceCompletion[] done = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(timeout.Token);

            for (int i = 0; i < requests.Length; i++)
            {
                if (turn > 0)
                {
                    int common = previous[i].Count - ThinkOpen.Length + 1;
                    Assert.Equal(common - common % 2, done[i].PrefixCacheReusedTokens);
                }
                Assert.Equal(Cold(tokenizer, prompts[i], timeout.Token), requests[i].OutputTokens);
                histories[i].Add(new ChatMessage { Role = "assistant", Thinking = Reasoning[turn], Content = Answers[turn],
                    RawOutputTokens = requests[i].OutputTokens.ToList() });
                histories[i].Add(new ChatMessage { Role = "user", Content = "continue" });
            }
            previous = prompts;
        }
        Assert.DoesNotContain(logger.Messages, m => m.Contains("could not rewind", StringComparison.Ordinal));
    }

    [Fact]
    public void RetentionMode_APrimaryConversionThatFailedAfterAdopting_IsNotReusedFromItsEmptySlot()
    {
        // Retention caps with a retention budget the native side refuses: the finished turn's
        // conversion adopts the primary, fails to retain it and releases it, which resets the slot. The tree
        // used to register it as the primary anyway, and the next turn - an exact continuation, so nothing
        // asked the model - decoded from position 0 of an empty cache.
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        OracleTraits traits = OracleFakes.N().Traits with { RewindCheckpoint = true, FailedConversionReleasesPrimary = true };
        using var model = new OracleModel(traits);
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        List<int> first = Enumerable.Range(1, 40).ToList();
        model.FailNext("convert");
        var turn1 = session.Generate(first, 20, SamplingConfig.Greedy, cancellationToken: timeout.Token);

        List<int> next = first.Concat(turn1.Tokens).Concat(Enumerable.Range(60, 10)).ToList();
        var turn2 = session.Generate(next, 20, SamplingConfig.Greedy, cancellationToken: timeout.Token);

        Assert.Equal(0, turn2.Completion.PrefixCacheReusedTokens);
        using var coldModel = new OracleModel(traits);
        using var cold = new CliInferenceSession(coldModel, Config, NullLogger.Instance);
        Assert.Equal(cold.Generate(next, 20, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            enablePrefixCache: false).Tokens, turn2.Tokens);
    }

    /// <summary>
    /// The same conversation shape on the real native V4.1 executor: the small generated fixture
    /// (eng/dsv41-fixture.py, then eng/validation/prepare-dsv41-managed-fixture.py) on the CPU
    /// backend, with the retention it always runs with. Every turn generates more than the
    /// raw ring reaches back (ring 256 - window 8 + 1 = 249 positions), so the rewind to the previous
    /// prompt is served by the checkpoint the slot took at that prompt's end, and the admission check
    /// is the native TSGgml_Dsv4SlotCanReuse. Answers are compared with a second model that prefills
    /// every prompt cold.
    /// </summary>
    [ModelFact("TS_TEST_DSV41_FIXTURE_DIR", "deepseek41-fixture", GgmlBackend = BackendType.GgmlCpu)]
    public void NativeFixture_ThinkingTurnsReuseThePreviousPromptThroughTheCheckpoint()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_DSV41_FIXTURE_DIR"), "deepseek41-fixture");
        // Through libc as well as the managed table: the native executor reads the checkpoint, flash
        // attention, Engram and TP switches with getenv, which never sees a managed-only override.
        using var environment = new NativeEnvScope();
        foreach (var (name, value) in new[] { ("MAX_CONTEXT", "2048"),
            ("TS_DSV4_UBATCH", "32"), ("TS_DSV4_THREADS", "2"), ("TS_DSV41_ENGRAM_THREADS", "2"), ("TS_DSV4_FA", "0"),
            ("TS_DSV41_YOCO_KEEP", "0"), ("TS_DSV41_REWIND_CHECKPOINT", "1") })
            environment.Set(name, value);
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromMinutes(5));
            const int Generate = 300;
            var logger = new RecordingLogger();
            using var model = new DeepSeek4Model(path, BackendType.GgmlCpu);
            Assert.True(model.SupportsKVCacheTruncation);
            Assert.Equal(2, model.KVCacheTruncationGranularity);
            using var session = new CliInferenceSession(model, Config, logger);
            using var coldModel = new DeepSeek4Model(path, BackendType.GgmlCpu);
            using var cold = new CliInferenceSession(coldModel, Config, NullLogger.Instance);

            // Token ids stand in for a rendered conversation: each next prompt is the previous one
            // without its last token (the think marker), a token the cache never held (</think>), part
            // of the answer, then a question. The fixture is not a language model and its EOS is id 1,
            // so the question is chosen until a cold run generates all of the turn: a turn that stopped
            // early would rewind within the live ring and never reach the checkpoint.
            List<int>? previous = null;
            List<int> history = new();
            for (int turn = 0; turn < 3; turn++)
            {
                List<int>? prompt = null;
                CliInferenceSession.Result? expected = null;
                for (int variant = 0; variant < 64 && prompt == null; variant++)
                {
                    var candidate = history.Concat(Enumerable.Range(0, turn == 0 ? 41 : 9)
                        .Select(i => 2 + (turn * 61 + variant * 97 + i * 37 + 11) % 250)).ToList();
                    var run = cold.Generate(candidate, Generate, SamplingConfig.Greedy, cancellationToken: timeout.Token,
                        enablePrefixCache: false);
                    if (run.Tokens.Count == Generate) (prompt, expected) = (candidate, run);
                }
                Assert.True(prompt != null, $"no question of turn {turn + 1} ran the whole {Generate} tokens");

                var result = session.Generate(prompt!, Generate, SamplingConfig.Greedy, cancellationToken: timeout.Token);
                if (previous == null)
                    Assert.Equal(0, result.Completion.PrefixCacheReusedTokens);
                else
                    Assert.Equal((previous.Count - 1) / 2 * 2, result.Completion.PrefixCacheReusedTokens);
                Assert.Equal(expected!.Tokens, result.Tokens);

                previous = prompt!;
                history = prompt!.Take(prompt!.Count - 1).Append(253).Concat(result.Tokens.Take(12)).ToList();
            }
            Assert.DoesNotContain(logger.Messages, m => m.Contains("declined at execution time", StringComparison.Ordinal)
                                                        || m.Contains("was revoked before execution", StringComparison.Ordinal));
        }
    }

    /// <summary>Retention is always on: the budget variable takes only a positive size, and anything else
    /// keeps the default instead of turning retention off.</summary>
    [Theory]
    [InlineData(null, 2048UL)]
    [InlineData("512", 512UL)]
    [InlineData("0", 2048UL)]
    [InlineData("-1", 2048UL)]
    [InlineData("lots", 2048UL)]
    public void RetentionBudget_IsAPositiveSize_AndNeverTurnsRetentionOff(string? text, ulong expectedMb)
        => Assert.Equal(expectedMb * 1024 * 1024, DeepSeek4Model.RetentionBudgetFromEnvironment(text));

    /// <summary>Sets a variable in both the managed table and libc's environ (the pattern of
    /// Gemma4VerifyPleTests / Glm5NextNativeTensorParallelTests), restoring both on dispose.</summary>

    [Fact]
    public void ATruncateTheModelRefusesAtExecution_ReprefillsAndStillAnswersLikeAColdRun()
    {
        // The admission check and the execution-time TryTruncateKVCache are separate calls. When the
        // second refuses anyway, the turn must re-prefill from zero rather than decode from a stale head.
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var tokenizer = new CharacterTokenizer();
        var logger = new RecordingLogger();
        using var model = OracleFakes.NLive(vocab: Vocab, tokenizer: tokenizer);
        using var session = new CliInferenceSession(model, Config, logger);

        var history = new List<ChatMessage> { new() { Role = "user", Content = Asks[0] } };
        var turn1 = session.Generate(Render(tokenizer, history), MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        history.Add(new ChatMessage { Role = "assistant", Thinking = Reasoning[0], Content = Answers[0], RawOutputTokens = turn1.Tokens });
        history.Add(new ChatMessage { Role = "user", Content = Asks[1] });
        List<int> prompt = Render(tokenizer, history);

        model.FailNext("truncate");
        var result = session.Generate(prompt, MaxNew, SamplingConfig.Greedy, cancellationToken: timeout.Token);

        Assert.Equal(0, result.Completion.PrefixCacheReusedTokens);
        Assert.Contains(logger.Messages, m => m.Contains("declined at execution time", StringComparison.Ordinal));
        Assert.Equal(Cold(tokenizer, prompt, timeout.Token), result.Tokens);
    }
}
