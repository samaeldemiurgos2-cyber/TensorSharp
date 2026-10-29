// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cli;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Logging;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// No request may fail - let alone take the process down - because its prompt or its reply
/// reaches the model's context window (MAX_CONTEXT).
///
/// <para>
/// Observed on A40s with the CLI under MAX_CONTEXT=4096, ggml_cuda, cuda and --tp 2 alike.
/// Nemotron 3.5: the second skills round's 4,071-token prompt was admitted with a 160-token
/// reply, and the 26th generated token asked the model for sequence length 4,097
/// ("Requested sequence length 4097 exceeds configured max context 4096") - the request
/// failed and the CLI aborted (exit 134). Qwen3.5-0.8B: a 20 KB SKILL.md read in round one
/// made round two's prompt 6,418 tokens, which the engine admitted - its only capacity check
/// is the KV block pool, sized for at least the whole context - and the model refused inside
/// the prefill. The fakes here refuse exactly as the models' EnsureCacheCapacity does.
/// </para>
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class ContextWindowLimitTests : IDisposable
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);
    private const int Context = 64;
    private readonly EnvScope _env = new();
    // The header a ModelBase fake is constructed from (it reads no tensors).
    private readonly string _gguf = Path.Combine(Path.GetTempPath(), $"ts-context-window-{Guid.NewGuid():N}.gguf");

    public ContextWindowLimitTests()
    {
        _env.Set("TS_SCHED_NUM_BLOCKS", null);
        _env.Set("TS_SCHED_DISABLE_BATCHED", null);
        _env.ClearSpeculationVars();
        BackendFailureWarmupTests.WriteProbeGguf(_gguf);
    }

    public void Dispose()
    {
        _env.Dispose();
        File.Delete(_gguf);
    }

    // 64 blocks of 8: the pool holds 512 tokens, eight times the window, as a real pool
    // sized for concurrency does. Its admission check therefore says nothing about the window.
    private static SchedulerConfig Config() => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 16,
        SoloPrefillChunkSize = 64,
        NumBlocks = 64,
        BlockSize = 8,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = 1,
        StopRepetition = false,
    };

    private static SequenceState Sequence(string id, int prompt, int maxNew)
        => new(id, Enumerable.Repeat(1, prompt).ToList(), maxNew, 8, SamplingConfig.Greedy);

    private static ContextBoundModel Model(bool batched) =>
        batched ? new BatchedContextBoundModel(Context) : new ContextBoundModel(Context);

    // ---------------------------------------------------------------- engine: the reply

    [Theory]
    [InlineData(false)]   // per-sequence path (the one the Nemotron trace failed on)
    [InlineData(true)]    // batched paged path
    public async Task ReplyStopsAsALengthCapExactlyWhereTheContextWindowEnds(bool batched)
    {
        using var model = Model(batched);
        using var engine = new InferenceEngine(model, Config());
        var seq = Sequence("near-limit", prompt: 50, maxNew: 100);

        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(Timeout);

        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        // The pipeline's own word for a length stop; every protocol maps it to "length".
        Assert.Equal("max_tokens", completion.FinishReason);
        Assert.Equal(Context - 50, completion.OutputTokenCount);
        Assert.Equal(Context, seq.NumTotalTokens);
        // The last token was forwarded into the window's last row, and nothing past it.
        Assert.Equal(Context, model.MaxSequenceLength);
        Assert.Null(seq.Error);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConcurrentRepliesEachStopAtTheRoomTheirOwnPromptLeaves(bool batched)
    {
        using var model = Model(batched);
        using var engine = new InferenceEngine(model, Config());
        var shorter = Sequence("shorter", prompt: 40, maxNew: 100);
        var longer = Sequence("longer", prompt: 55, maxNew: 100);
        var a = engine.SubmitRequest(shorter);
        var b = engine.SubmitRequest(longer);

        var completions = await Task.WhenAll(a.Completion, b.Completion).WaitAsync(Timeout);

        Assert.All(completions, c => Assert.Equal("max_tokens", c.FinishReason));
        Assert.Equal(Context - 40, completions[0].OutputTokenCount);
        Assert.Equal(Context - 55, completions[1].OutputTokenCount);
        Assert.True(model.MaxSequenceLength <= Context);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ARequestedLimitBelowTheRoomStillWins(bool batched)
    {
        using var model = Model(batched);
        using var engine = new InferenceEngine(model, Config());

        var completion = await engine.SubmitRequest(Sequence("short", prompt: 10, maxNew: 5))
            .Completion.WaitAsync(Timeout);

        Assert.Equal(5, completion.OutputTokenCount);
        Assert.Equal("max_tokens", completion.FinishReason);
    }

    // ---------------------------------------------------------------- engine: the prompt

    [Theory]
    [InlineData(false, Context)]
    [InlineData(true, Context)]
    [InlineData(false, Context + 37)]
    [InlineData(true, Context + 37)]
    public async Task APromptThatLeavesNoRoomForAReplyIsRefusedBeforeAnyForward(bool batched, int promptTokens)
    {
        using var model = Model(batched);
        using var engine = new InferenceEngine(model, Config());
        var tooLong = Sequence("too-long", promptTokens, 8);
        var handle = engine.SubmitRequest(tooLong);

        var error = await Assert.ThrowsAsync<ContextLengthExceededException>(
            () => handle.Completion.WaitAsync(Timeout));

        Assert.Equal(promptTokens, error.PromptTokens);
        Assert.Equal(Context, error.ContextTokens);
        Assert.Contains("MAX_CONTEXT", error.Message);
        // The token stream a host reads first ends with the same refusal.
        await Assert.ThrowsAsync<ContextLengthExceededException>(
            () => handle.Tokens.ReadAllAsync().ToListAsync().AsTask().WaitAsync(Timeout));
        Assert.Equal(SequenceStatus.FinishedError, tooLong.Status);
        Assert.Equal(0, tooLong.BlockTable.NumBlocks);
        Assert.Equal(0, model.TokensForwarded);

        // The longest prompt that fits still gets its one token, and the engine is unharmed.
        var fits = await engine.SubmitRequest(Sequence("fits", Context - 1, 8)).Completion.WaitAsync(Timeout);
        Assert.Equal(1, fits.OutputTokenCount);
        Assert.Equal("max_tokens", fits.FinishReason);
        Assert.Equal(engine.PoolStats.totalBlocks, engine.PoolStats.freeBlocks);
    }

    // ---------------------------------------------------------------- CLI facade

    [Fact]
    public void CliAsksForNoMoreThanTheRoomItsPromptLeaves()
    {
        using var model = new ContextBoundModel(Context);
        using var session = new CliInferenceSession(model, Config(), NullLogger.Instance);
        using var timeout = new CancellationTokenSource(Timeout);

        var result = session.Generate(Enumerable.Repeat(1, 60).ToArray(), 160, SamplingConfig.Greedy,
            cancellationToken: timeout.Token);

        Assert.Equal(4, result.Tokens.Count);
        Assert.Equal(4, result.Sequence.MaxNewTokens);
        Assert.Equal("max_tokens", result.Completion.FinishReason);
    }

    [Fact]
    public void CliSurfacesAnOverlongPromptAsTheRefusalItReportsInOneLine()
    {
        // Program.Main turns this exception type into "Configuration error: ..." and exit
        // code 1 instead of an unhandled exception (a stack trace and SIGABRT).
        using var model = new ContextBoundModel(Context);
        using var session = new CliInferenceSession(model, Config(), NullLogger.Instance);
        using var timeout = new CancellationTokenSource(Timeout);

        var error = Assert.Throws<ContextLengthExceededException>(() => session.Generate(
            Enumerable.Repeat(1, Context + 10).ToArray(), 160, SamplingConfig.Greedy,
            cancellationToken: timeout.Token));

        Assert.Equal(Context + 10, error.PromptTokens);
        Assert.Equal(0, model.TokensForwarded);
        // The session serves the next request.
        Assert.Single(session.Generate(Enumerable.Repeat(1, Context - 1).ToArray(), 160, SamplingConfig.Greedy,
            cancellationToken: timeout.Token).Tokens);
    }

    [Theory]
    [InlineData(null, HostExitCodes.Success)]
    [InlineData("flag", HostExitCodes.ConfigurationError)]
    [InlineData("context", HostExitCodes.ConfigurationError)]
    public void EveryExitTheCliHandlesLeavesTheSameWay(string failure, int exitCode)
    {
        // The way out is where Linux ggml-vulkan skips the C++ static destructors (_exit),
        // which otherwise race the NVIDIA driver's threads and segfault. Only a normal return
        // took it, so a prompt refused for MAX_CONTEXT after a Vulkan run could exit 139, not 1.
        int saved = Environment.ExitCode;
        try
        {
            Environment.ExitCode = HostExitCodes.Success;
            int? leftWith = null;

            TensorSharp.Cli.Program.RunToExit(() =>
            {
                if (failure == "flag")
                    throw new ArgumentException("--max-tokens takes a number");
                if (failure == "context")
                    throw new ContextLengthExceededException(6418, 4096);
            }, () => leftWith = Environment.ExitCode);

            Assert.Equal(exitCode, leftWith);
        }
        finally
        {
            Environment.ExitCode = saved;
        }
    }

    [Fact]
    public void AnUnexpectedCliFailureKeepsItsStackTrace()
    {
        bool left = false;

        Assert.Throws<InvalidOperationException>(() => TensorSharp.Cli.Program.RunToExit(
            () => throw new InvalidOperationException("a bug"), () => left = true));
        Assert.False(left);
    }

    [Fact]
    public void ASkillsRoundWithAPictureLeavesTheReplyItsRoomOnceThePictureExpands()
    {
        // The template writes one placeholder per picture; the injector grows it to the
        // encoder's rows. Fitted to the rendered prompt alone, the results left a 4,096 window
        // 3,996 + 767 tokens of prompt, which the engine refused.
        using var model = new PictureModel(_gguf);
        var images = new List<string> { "photo.png" };
        var priorTurns = new List<ChatMessage> { SkillsReadCall() };

        TensorSharp.Cli.Program.AddSkillResultsWithinContext(model, "What does the skill say about this photo?",
            images, null, false, null, priorTurns, [(SkillFile(20_000), "skills_read")], null,
            enableThinking: false, maxTokens: 100, toolResultsRendered: true, round: 1);

        Assert.Contains("[Truncated by the host", priorTurns[^1].Content);
        int prefill = Prefill(model, "What does the skill say about this photo?", images, priorTurns);
        Assert.True(prefill <= PictureModel.Window - 100, $"the next prefill is {prefill} tokens");
        Assert.True(prefill > PictureModel.Window - 100 - PictureModel.RowsPerPicture,
            $"the picture's rows were taken off twice: {prefill} tokens");
    }

    [Fact]
    public void TheLastSkillsRoundsLimitNoteIsFittedWithItsResults()
    {
        // The note that ends the loop used to be added after the fit: with a small reply
        // reserve the final prompt then passed the window, and the run ended with no answer.
        using var model = new PictureModel(_gguf);
        var priorTurns = new List<ChatMessage> { SkillsReadCall() };
        var limitNote = new ChatMessage
        {
            Role = "tool",
            Content = "Error: the limit on skill lookups for this turn has been reached. Answer now using what you "
                + "have already read, and say which part you could not check.",
        };

        TensorSharp.Cli.Program.AddSkillResultsWithinContext(model, "Summarise the skill.", null, null, false, null,
            priorTurns, [(SkillFile(20_000), "skills_read")], null, enableThinking: false, maxTokens: 16,
            toolResultsRendered: true, round: 4, trailing: limitNote);

        Assert.Same(limitNote, priorTurns[^1]);
        Assert.Contains("[Truncated by the host", priorTurns[^2].Content);
        int prefill = Prefill(model, "Summarise the skill.", null, priorTurns);
        Assert.True(prefill <= PictureModel.Window - 16, $"the final prefill is {prefill} tokens");
    }

    [Fact]
    public void AnInteractiveThinkingBudgetIsSizedForTheRoomTheWindowLeaves()
    {
        // --think --max-tokens 8192 in a 1,024-token window: the reply has room for about a
        // thousand tokens, and a budget of three quarters of 8,192 never fired, so a model still
        // reasoning when the window ended the reply printed no answer. Sized from the room, as
        // the server sizes it, the block closes with room left to answer.
        _env.Set("TS_THINKING_BUDGET", null);
        using var model = new ReasoningModel(_gguf, context: 1024);
        SamplingConfig sampling = SamplingConfig.Greedy;
        sampling.StopRepetition = false;   // the fake reasons in one repeated token; only the budget may close it
        var session = new InteractiveSession(model, _gguf, BackendType.Cpu, null, new ThinkingPromptRenderer(),
            sampling, null, enableThinking: true, maxTokens: 8192, NullLogger.Instance) { PrefixCacheEnabled = false };

        session.RunWithInput(new StringReader("Why?\n/exit\n"), allowSteering: false);

        Assert.True(model.ForwardedThinkingEnd, "the reply reached the window still reasoning");
    }

    private static ChatMessage SkillsReadCall() => new()
    {
        Role = "assistant",
        Content = string.Empty,
        ToolCalls = [new ToolCall { Name = "skills_read", Arguments = new() { ["skill"] = "algorithmic-art" } }],
    };

    /// <summary>The next round's prefill as RunInference builds it: rendered, then expanded.</summary>
    private static int Prefill(ModelBase model, string user, List<string> images, List<ChatMessage> priorTurns)
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = user, ImagePaths = images } };
        messages.AddRange(priorTurns);
        List<int> rendered = new KVCachePromptRenderer(new GgufPromptRenderer()).RenderToTokens(
            model.Tokenizer, model.Config.ChatTemplate, messages, model.Config.Architecture, addGenerationPrompt: true);
        try
        {
            return model.MultimodalInjector.ProcessPromptTokens(messages, rendered, "prefill-probe").Count;
        }
        finally
        {
            model.MultimodalInjector.ClearPreparedPromptState("prefill-probe");
        }
    }

    // ---------------------------------------------------------------- server

    [Fact]
    public async Task TheEnginesRefusalBecomesA400LikeAPromptOverflow()
    {
        var ctx = new DefaultHttpContext();
        ctx.Request.Path = "/v1/chat/completions";
        ctx.Response.Body = new MemoryStream();

        await PromptOverflowMiddleware.InvokeAsync(ctx,
            () => throw new ContextLengthExceededException(6418, 4096));

        Assert.Equal(400, ctx.Response.StatusCode);
        ctx.Response.Body.Seek(0, SeekOrigin.Begin);
        using var doc = JsonDocument.Parse(await new StreamReader(ctx.Response.Body, Encoding.UTF8).ReadToEndAsync());
        var error = doc.RootElement.GetProperty("error");
        Assert.Equal("context_length_exceeded", error.GetProperty("code").GetString());
        Assert.Contains("6418", error.GetProperty("message").GetString());
    }

    // ---------------------------------------------------------------- skills tool results

    // A stand-in renderer: 100 tokens of conversation plus four characters per token of results.
    private static int Count(IReadOnlyList<string> results) => 100 + results.Sum(r => (r.Length + 3) / 4);

    private static string SkillFile(int chars)
    {
        var sb = new StringBuilder();
        for (int i = 0; sb.Length < chars; i++)
            sb.Append("Step ").Append(i).Append(": do the ").Append(i).Append("th thing carefully.\n");
        return sb.ToString(0, chars);
    }

    [Fact]
    public void ToolResultsThatFitAreLeftExactlyAsTheyAre()
    {
        var results = new List<string> { "small result", SkillFile(2_000) };

        var fitted = ToolResultContextBudget.Fit(results, 1_000, 4096, Count, out bool truncated);

        Assert.Same(results, fitted);
        Assert.False(truncated);
    }

    [Fact]
    public void AnOversizedSkillReadKeepsItsHeadAndSaysWhatWasCut()
    {
        string skill = SkillFile(20_342);   // the algorithmic-art SKILL.md from the trace

        var fitted = ToolResultContextBudget.Fit(new[] { skill }, 1_000, 4096, Count, out bool truncated);

        Assert.True(truncated);
        Assert.True(Count(fitted) <= 1_000, $"fitted prompt is {Count(fitted)} tokens");
        string cut = Assert.Single(fitted);
        int note = cut.IndexOf("\n\n[Truncated by the host", StringComparison.Ordinal);
        Assert.True(note > 2_000, "most of the room went to the file itself");
        Assert.StartsWith(cut[..note], skill, StringComparison.Ordinal);
        Assert.Contains("of 20342 characters", cut);
        Assert.Contains("4096-token context window", cut);
        Assert.Contains("offset", cut);
    }

    [Fact]
    public void ShortResultsKeepAllTheirTextWhileTheLongOneShares()
    {
        string listing = SkillFile(300);
        string skill = SkillFile(20_000);

        var fitted = ToolResultContextBudget.Fit(new[] { listing, skill }, 1_000, 4096, Count, out _);

        Assert.Equal(listing, fitted[0]);
        Assert.Contains("[Truncated by the host", fitted[1]);
        Assert.True(Count(fitted) <= 1_000);
    }

    [Fact]
    public void AShortResultIsNeverReplacedByALongerCut()
    {
        // Longer than its omitted note, shorter than a cut plus its ~225-character note.
        string error = ("Cannot read 'references/advanced.md' from skill 'pdf': the path is outside the skill's "
            + "directory or does not exist. Use skills_list to see the files this skill provides. ").PadRight(300, '.');
        Assert.InRange(error.Length, ToolResultContextBudget.Omitted(error.Length, 4096).Length + 1, 440);
        string skill = SkillFile(20_000);

        for (int limit = 150; limit <= 1_200; limit++)
        {
            var fitted = ToolResultContextBudget.Fit(new[] { error, skill }, limit, 4096, Count, out _);
            Assert.True(fitted[0].Length <= error.Length,
                $"limit {limit}: the {error.Length}-character result came back as {fitted[0].Length} characters");
        }
    }

    [Fact]
    public void OnlyTheLastSkillsRoundCarriesTheLimitNote()
    {
        Assert.Null(TensorSharp.Cli.Program.SkillRoundTrailingNote(1, 3, toolResultsRendered: true));
        Assert.Null(TensorSharp.Cli.Program.SkillRoundTrailingNote(2, 3, toolResultsRendered: false));
        var note = TensorSharp.Cli.Program.SkillRoundTrailingNote(3, 3, toolResultsRendered: true);
        Assert.NotNull(note);
        Assert.Contains("limit on skill lookups", note.Content);
    }

    [Fact]
    public void WhenNoTextFitsEachResultIsReplacedByANote()
    {
        var fitted = ToolResultContextBudget.Fit(new[] { SkillFile(9_000), SkillFile(5_000) }, 120, 4096, Count,
            out bool truncated);

        Assert.True(truncated);
        Assert.All(fitted, r => Assert.StartsWith("[Omitted by the host", r));
        Assert.Contains("9000 characters", fitted[0]);
    }

    [Fact]
    public void TheCutConvergesWhenTokensAreNotProportionalToCharacters()
    {
        // Code-like text tokenizes denser than prose, and a note costs a fixed amount.
        static int Dense(IReadOnlyList<string> results) =>
            300 + results.Sum(r => r.Length / 2 + (r.Contains("[Truncated", StringComparison.Ordinal) ? 40 : 0));
        var results = new[] { SkillFile(8_316), SkillFile(20_342), "short" };

        var fitted = ToolResultContextBudget.Fit(results, 2_000, 4096, Dense, out bool truncated);

        Assert.True(truncated);
        Assert.True(Dense(fitted) <= 2_000, $"fitted prompt is {Dense(fitted)} tokens");
        Assert.Equal("short", fitted[2]);
    }

    [Fact]
    public void ACutNeverSplitsASurrogatePair()
    {
        string text = string.Concat(Enumerable.Repeat("\U0001F600", 100));   // 200 UTF-16 units

        for (int keep = 1; keep < text.Length; keep++)
        {
            string cut = ToolResultContextBudget.Cut(text, keep, 4096);
            string head = cut[..cut.IndexOf("\n\n[Truncated", StringComparison.Ordinal)];
            Assert.False(head.Length > 0 && char.IsHighSurrogate(head[^1]), $"keep={keep} ended inside a pair");
        }
    }

    [Fact]
    public void AShortResultIsNeverSwappedForItsLongerNote()
    {
        // A failed call's one-line error beside a 20 KB read. A note is about 210 characters,
        // ten times the error: with both results replaced the prompt was 207 tokens, so Fit gave
        // up at a 170-token limit that keeping the error (159 tokens) meets.
        const string error = "Error: no such file.";
        var results = new[] { error, SkillFile(20_000) };

        var fitted = ToolResultContextBudget.Fit(results, 170, 4096, Count, out bool truncated);

        Assert.True(truncated);
        Assert.True(Count(fitted) <= 170, $"fitted prompt is {Count(fitted)} tokens");
        Assert.Equal(error, fitted[0]);
        Assert.NotEqual(results[1], fitted[1]);

        // When even the shortest form does not fit, the error is still itself, not a note
        // claiming that 20 characters do not fit.
        var floor = ToolResultContextBudget.Fit(results, 120, 4096, Count, out _);
        Assert.Equal(error, floor[0]);
        Assert.StartsWith("[Omitted by the host", floor[1]);
    }

    [Fact]
    public void ResultsNoneOfWhichCanBeShortenedAreLeftAsTheyAre()
    {
        var results = new[] { "Error: no such file.", "Error: no such skill." };

        var fitted = ToolResultContextBudget.Fit(results, 105, 4096, Count, out bool truncated);

        Assert.Same(results, fitted);
        Assert.False(truncated);
    }

    // ---------------------------------------------------------------- fakes

    /// <summary>One linear cache that refuses to grow past the window, the way the models'
    /// EnsureCacheCapacity does. Never emits EOS, so only a length stop ends a reply.</summary>
    private class ContextBoundModel(int context) : IModelArchitecture
    {
        private int _length;
        public int TokensForwarded { get; protected set; }
        public int MaxSequenceLength { get; protected set; }
        public ModelConfig Config { get; } = new() { Architecture = "context-bound-fake", VocabSize = 16 };
        public ITokenizer Tokenizer { get; } = new StubTokenizer();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public int MaxContextLength => context;

        public float[] Forward(int[] tokens)
        {
            Require(_length + tokens.Length);
            _length += tokens.Length;
            TokensForwarded += tokens.Length;
            return Logits();
        }

        protected void Require(int sequenceLength)
        {
            if (sequenceLength > context)
                throw new InvalidOperationException(
                    $"Requested sequence length {sequenceLength} exceeds configured max context {context}.");
            MaxSequenceLength = Math.Max(MaxSequenceLength, sequenceLength);
        }

        protected static float[] Logits()
        {
            var logits = new float[16];
            logits[7] = 100;
            return logits;
        }

        public void ResetKVCache() => _length = 0;
        public void TruncateKVCache(int tokenCount) => _length = tokenCount;
        public void Dispose() { }
    }

    /// <summary>The same limit on the batched paged route: every row is written at the
    /// position the executor hands over.</summary>
    private sealed class BatchedContextBoundModel(int context) : ContextBoundModel(context), IBatchedPagedModel
    {
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            var results = new List<float[]>(ctx.Sequences.Count);
            for (int s = 0; s < ctx.Sequences.Count; s++)
            {
                int start = ctx.QueryStartLoc[s], count = ctx.NumScheduledTokens[s];
                Require(ctx.Positions[start + count - 1] + 1);
                TokensForwarded += count;
                results.Add(Logits());
            }
            return results;
        }
    }

    /// <summary>A Qwen-VL stand-in whose injector grows each picture's one placeholder to
    /// <see cref="RowsPerPicture"/> rows, as its vision encoder's output does (a 1024x768
    /// photo). It is never run.</summary>
    private sealed class PictureModel : ModelBase, IMultimodalPromptExpander
    {
        public const int Window = 4096;
        public const int RowsPerPicture = 768;

        public PictureModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "qwen35", VocabSize = CharTokenizer.Size, NumLayers = 1 };
            Tokenizer = new CharTokenizer();
            _maxContextLength = Window;
        }

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> tokens)
        {
            int pictures = history.Sum(m => m.ImagePaths?.Count ?? 0);
            var expanded = new List<int>(tokens);
            expanded.AddRange(Enumerable.Repeat((int)'.', pictures * (RowsPerPicture - 1)));
            return expanded;
        }

        protected override float[] ForwardCore(int[] tokens) =>
            throw new InvalidOperationException("Only the prompt is measured.");

        protected override void ResetKVCacheCore() { }
    }

    /// <summary>A Qwen 3.5 stand-in that reasons forever: it answers every position with the
    /// same thought token, so only the thinking budget can close its block.</summary>
    private sealed class ReasoningModel : ModelBase
    {
        public ReasoningModel(string path, int context) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "qwen35", VocabSize = CharTokenizer.Size, NumLayers = 1 };
            Tokenizer = new CharTokenizer();
            _maxContextLength = context;
        }

        /// <summary>Whether <c>&lt;/think&gt;</c> reached the model, which happens only after
        /// the budget forced it into the reply.</summary>
        public bool ForwardedThinkingEnd { get; private set; }

        protected override float[] ForwardCore(int[] tokens)
        {
            if (Array.IndexOf(tokens, CharTokenizer.ThinkEnd) >= 0)
                ForwardedThinkingEnd = true;
            var logits = new float[CharTokenizer.Size];
            logits['y'] = 10;
            return logits;
        }

        protected override void ResetKVCacheCore() { }
    }

    /// <summary>The user's text after a fixed header, with the thinking block opened.</summary>
    private sealed class ThinkingPromptRenderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
            => "abcdefgh" + string.Join("\n", messages.Select(m => m.Content)) + (enableThinking ? "<think>\n" : "");
    }

    /// <summary>One token per ASCII character, and <c>&lt;/think&gt;</c> as one trained token.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public const int ThinkEnd = 128;
        public const int Size = ThinkEnd + 1;
        private static readonly string[] Words =
            Enumerable.Range(0, ThinkEnd).Select(i => ((char)i).ToString()).Append("</think>").ToArray();

        public string[] Vocab => Words;
        public int VocabSize => Size;
        public int BosTokenId => -1;
        public int[] EosTokenIds => [1];
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => Array.IndexOf(Words, token);
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => c < ThinkEnd ? c : '?').Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => string.Concat(tokens.Select(id => Words[id]));
        public void AppendTokenBytes(int id, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(Words[id]));
    }

    private sealed class StubTokenizer : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, 16).Select(i => i.ToString()).ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => [];
        public string Decode(List<int> ids) => string.Join(",", ids);
        public void AppendTokenBytes(int id, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(id.ToString()));
        public bool IsEos(int id) => false;
        public int LookupToken(string text) => -1;
    }
}
