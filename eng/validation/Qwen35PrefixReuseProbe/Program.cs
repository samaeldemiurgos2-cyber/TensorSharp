using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

string Option(string name, string fallback)
{
    int index = Array.IndexOf(args, name);
    return index < 0 ? fallback : index + 1 < args.Length ? args[index + 1]
        : throw new ArgumentException($"Missing value for {name}.");
}
string modelPath = Option("--model", Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MODEL")
    ?? @"C:\Works\models\Qwen\Qwen3.8-27B-UD-IQ3_XXS.gguf");
string output = Option("--out", "artifacts/qwen35-agent-prefix-reuse/direct-probe.json");
int steps = int.Parse(Option("--steps", "32"));
int pairs = int.Parse(Option("--pairs", "3"));
if (steps is < 1 or > 256 || pairs is < 1 or > 20)
    throw new ArgumentException("Use 1..256 steps and 1..20 pairs.");
Environment.SetEnvironmentVariable("MAX_CONTEXT", "8192");
KvCacheDtypeConfig.ConfigureFromEnvironment();
var runs = new List<PairRun>();
var seeds = new List<SeedRun>();
PromptFixture[]? fixtures = null;
double loadMilliseconds = 0;
string? kvDtype = null;
int? initialCacheCapacity = null;
string? error = null;
bool passed = false;
var failures = new List<string>();
try
{
    var load = Stopwatch.StartNew();
    using var model = ModelBase.Create(modelPath, BackendType.GgmlCuda);
    loadMilliseconds = load.Elapsed.TotalMilliseconds;
    if (model is not Qwen35Model) throw new InvalidOperationException("Requires a Qwen35 family model.");
    initialCacheCapacity = typeof(Qwen35Model).GetField("_initialKvCacheCapacity", BindingFlags.Instance | BindingFlags.NonPublic)
        ?.GetValue(model) as int?;
    kvDtype = model.KvCacheDtype.ToString();
    fixtures = await CapturePrompts(model);
    Require(fixtures[0].SharedPrefixTokens >= 64, "Fixture public prefix is too short.");
    Require(fixtures[0].SharedPrefixTokens == fixtures[1].SharedPrefixTokens
        && fixtures[0].Tokens.Take(fixtures[0].SharedPrefixTokens)
            .SequenceEqual(fixtures[1].Tokens.Take(fixtures[1].SharedPrefixTokens)),
        "Sibling public prefixes differ.");
    Require(fixtures.All(fixture => fixture.Tokens.Count + steps < 8192), "Fixture exceeds context capacity.");
    Console.WriteLine($"[prefix-probe] shared={fixtures[0].SharedPrefixTokens} prompts={string.Join(',', fixtures.Select(f => f.Tokens.Count))}");

    PairRun coldEnabled = await RunCase(model, fixtures, true, false, "cold-enabled", steps, seeds);
    runs.Add(coldEnabled);
    PairRun coldOff = await RunCase(model, fixtures, false, false, "cold-disabled", steps, seeds);
    runs.Add(coldOff);
    CheckPair(coldEnabled, coldOff, fixtures[0].SharedPrefixTokens, warm: false, steps, failures);

    for (int pair = 0; pair < pairs; pair++)
    {
        PairRun enabled, disabled;
        if (pair % 2 == 0)
        {
            enabled = await RunCase(model, fixtures, true, true, $"pair-{pair + 1}-enabled", steps, seeds);
            runs.Add(enabled);
            disabled = await RunCase(model, fixtures, false, false, $"pair-{pair + 1}-disabled", steps, seeds);
            runs.Add(disabled);
        }
        else
        {
            disabled = await RunCase(model, fixtures, false, false, $"pair-{pair + 1}-disabled", steps, seeds);
            runs.Add(disabled);
            enabled = await RunCase(model, fixtures, true, true, $"pair-{pair + 1}-enabled", steps, seeds);
            runs.Add(enabled);
        }
        CheckPair(enabled, disabled, fixtures[0].SharedPrefixTokens, warm: true, steps, failures);
    }
    passed = failures.Count == 0;
    if (!passed) Environment.ExitCode = 1;
}
catch (Exception exception)
{
    error = exception.ToString();
    Console.Error.WriteLine(error);
    Environment.ExitCode = 1;
}
finally
{
    string nativeName = OperatingSystem.IsWindows() ? "GgmlOps.dll"
        : OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" : "libGgmlOps.so";
    string? nativePath = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
        .FirstOrDefault(module => Path.GetFileName(module.FileName).Equals(nativeName, StringComparison.OrdinalIgnoreCase))?.FileName;
    var enabled = runs.Where(run => run.Warm).ToArray();
    var disabled = runs.Where(run => run.Label.StartsWith("pair-", StringComparison.Ordinal) && !run.CacheEnabled).ToArray();
    bool benchmarkComplete = enabled.Length == pairs && disabled.Length == pairs;
    double? enabledMedian = benchmarkComplete ? Median(enabled.Select(run => run.Milliseconds)) : null;
    double? disabledMedian = benchmarkComplete ? Median(disabled.Select(run => run.Milliseconds)) : null;
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllText(output, JsonSerializer.Serialize(new
    {
        ValidationPassed = passed, PerformanceQualified = passed && benchmarkComplete, Error = error, Failures = failures,
        Model = Path.GetFullPath(modelPath), ModelBytes = new FileInfo(modelPath).Length, Backend = "ggml_cuda",
        KvDtype = kvDtype, Context = 8192, MaxNewTokens = steps, Pairs = pairs, ModelLoadMilliseconds = loadMilliseconds,
        InitialCacheTokensEnvironment = Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS"),
        ActualInitialCacheCapacityTokens = initialCacheCapacity,
        NativePath = nativePath, NativeSha256 = nativePath == null ? null : Hash(nativePath),
        ModelAssemblySha256 = Hash(typeof(Qwen35Model).Assembly.Location),
        RuntimeAssemblySha256 = Hash(typeof(InferenceEngine).Assembly.Location),
        AgentAssemblySha256 = Hash(typeof(MultiAgentSession).Assembly.Location),
        ProbeAssemblySha256 = Hash(typeof(PairRun).Assembly.Location),
        GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
        Device = Environment.GetEnvironmentVariable("TS_VALIDATION_DEVICE"),
        Settings = Config(true), Execution = ExecutionOptions.FromEnvironment(), SeedCosts = seeds, Runs = runs, Fixtures = fixtures,
        WarmCacheMedianMilliseconds = enabledMedian, DisabledMedianMilliseconds = disabledMedian,
        WarmCacheSpeedup = enabledMedian.HasValue ? disabledMedian / enabledMedian : null,
        Limitations = "Direct engine prefix-reuse fixture, not the entire HTTP host preamble. Production MultiAgentSession builds reviewer messages/tools; the real model tokenizer/template renders them. Both requests are submitted with a closed ComputeGate before any completion is awaited. Measured elapsed time starts when that gate opens and includes prefill, cache adoption, decoding and completion. Model loading, engine creation, prompt rendering and seed warmup are excluded. Each pair uses fresh engines and scopes; order alternates. Seed costs are reported separately. This is a maximum-token workload: normal EOS may stop a tool-call or answer turn early. Strict equality requires the paired generated greedy token IDs, actual counts, completion status and finish reasons to match, not logits or completed reviewer reasoning. Generated fragments may stop before a final answer. Model-owned checkpoint copies and private writable caches are used; this does not prove physical KV-page sharing or reduced VRAM. Cached memory samples after completion exclude active peak memory, weights, allocator reserves and primary/private cache bytes. Only the reported model/device/cache dtype is covered. Dependency/device metadata is operator-supplied; no end-to-end HTTP latency claim."
    }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"[prefix-probe] passed={passed} report={Path.GetFullPath(output)}");
}

static async Task<PromptFixture[]> CapturePrompts(ModelBase model)
{
    var captured = new ConcurrentDictionary<string, (List<ChatMessage> Messages, List<ToolFunction> Tools)>();
    var messages = new List<ChatMessage>
    {
        new() { Role = "system", Content = "You are a careful assistant. Follow the assigned task, preserve supplied facts, and check every calculation. Distinguish verified results from assumptions. Do not speculate about the causes of calculation errors." },
        new() { Role = "user", Content = "The parent conversation is private and must not be copied into reviewer context." },
    };
    await using var agents = new MultiAgentSession(messages, SkillTools.BuiltIn(), new SkillToolContext([]),
        id => (history, tools, _) =>
        {
            captured[id] = (new(history), new(tools!));
            return Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "Prompt captured." }));
        }, new MultiAgentOptions { Enabled = true, AllowWorkerTools = true });
    string[] names = ["proposal_a", "proposal_b"];
    string[] tasks =
    [
        "Independently review Proposal A. Sell 120 units at $25 each. Variable cost: $13 per unit. Fixed cost: $300. Claimed profit: $1,300. Calculate revenue, total cost, actual profit, and the exact profit overstatement. Show your arithmetic. Do not speculate about the causes of the error.",
        "Independently review Proposal B. Serve 200 customers paying $18 each. Variable cost: $7 per customer. Fixed cost: $400. Claimed profit: $1,900. Calculate revenue, total cost, actual profit, and the exact profit overstatement. Show your arithmetic. Do not speculate about the causes of the error."
    ];
    for (int i = 0; i < names.Length; i++)
    {
        SkillToolResult result = await agents.ExecuteAsync(new ToolCall
        {
            Name = "spawn_agent", Arguments = new() { ["task_name"] = names[i], ["agent_type"] = "reviewer", ["task"] = tasks[i] }
        });
        Require(result.Ok, result.Content);
    }
    SkillToolResult wait = await agents.ExecuteAsync(new ToolCall
        { Name = "wait_agent", Arguments = new() { ["timeout_ms"] = 10000 } });
    Require(wait.Ok && captured.Count == 2, "Could not capture both child prompts.");
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    return names.Select(name =>
    {
        var child = captured["/root/" + name];
        List<int> tokens = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, child.Messages,
            model.Config.Architecture, true, child.Tools, enableThinking: false);
        var governing = child.Messages.TakeWhile(message => message.Role is "system" or "developer").ToList();
        List<int> prefix = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, governing,
            model.Config.Architecture, false, child.Tools, enableThinking: false);
        int shared = 0;
        int limit = Math.Min(prefix.Count, tokens.Count - 1);
        while (shared < limit && prefix[shared] == tokens[shared]) shared++;
        if (shared < 64) shared = 0; // Same verified boundary and minimum as ChatGenerationPipeline.
        Require(!governing.Any(message => message.Content?.Contains("/root/", StringComparison.Ordinal) == true),
            "Agent identity leaked into the public governing prefix.");
        return new PromptFixture(name, shared, tokens, child.Messages, child.Tools);
    }).ToArray();
}

static SchedulerConfig Config(bool enabled) => new()
{
    MaxNumBatchedTokens = 4096, MaxNumRunningSequences = 4,
    MaxPrefillChunkSize = 256, SoloPrefillChunkSize = 8192,
    NumBlocks = 256, BlockSize = 256, EnablePrefixCaching = enabled,
    StopRepetition = false, DecodeQuantumTokens = 256,
};

static async Task<PairRun> RunCase(ModelBase model, PromptFixture[] fixtures, bool enabled, bool warm,
    string label, int steps, List<SeedRun> seeds)
{
    var gate = new ComputeGate();
    using var engine = new InferenceEngine(model, Config(enabled)) { ComputeGate = gate };
    int shared = fixtures[0].SharedPrefixTokens;
    if (warm)
    {
        var seedClock = Stopwatch.StartNew();
        var seed = new SequenceState(label + "-seed", fixtures[0].Tokens.Take(shared).ToList(), 1, 256,
            SamplingConfig.Greedy, sharedPrefixTokens: shared, cacheScope: Guid.NewGuid().ToString("N"));
        InferenceCompletion completed = await engine.SubmitRequest(seed).Completion.WaitAsync(TimeSpan.FromMinutes(10));
        Require(completed.PrefixCacheReusedTokens == 0 && completed.OutputTokenCount == 1, "Seed was not cold or did not complete.");
        seeds.Add(new SeedRun(label, shared, seedClock.Elapsed.TotalMilliseconds));
    }
    gate.Close();
    long held = engine.StepsHeldByGate;
    var requests = fixtures.Select(fixture => new SequenceState(label + "-" + fixture.Name,
        new List<int>(fixture.Tokens), steps, 256, SamplingConfig.Greedy,
        sharedPrefixTokens: shared, cacheScope: Guid.NewGuid().ToString("N"))).ToArray();
    var handles = requests.Select(request => engine.SubmitRequest(request)).ToArray();
    var park = Stopwatch.StartNew();
    while (engine.StepsHeldByGate <= held && park.Elapsed < TimeSpan.FromSeconds(10)) await Task.Delay(1);
    Require(engine.StepsHeldByGate > held, "Engine never parked before the paired submissions were released.");
    DateTime releasedAt = DateTime.UtcNow;
    TimeSpan before = engine.TotalForwardTime;
    var elapsed = Stopwatch.StartNew();
    gate.Open();
    InferenceCompletion[] completions = await Task.WhenAll(handles.Select(handle => handle.Completion))
        .WaitAsync(TimeSpan.FromMinutes(10));
    elapsed.Stop();
    var rows = completions.Select((completion, index) =>
    {
        return new RequestRun(fixtures[index].Name, completion.PromptTokenCount, completion.PrefixCacheReusedTokens,
            completion.PromptTokenCount - completion.PrefixCacheReusedTokens, completion.OutputTokenCount,
            requests[index].OutputTokens.Count(token => !model.Tokenizer.IsEos(token)),
            requests[index].OutputTokens.ToArray(), model.Tokenizer.Decode(requests[index].OutputTokens.ToList()),
            completion.Status.ToString(), completion.FinishReason,
            completion.FirstTokenAt.HasValue ? (completion.FirstTokenAt.Value - releasedAt).TotalMilliseconds : null);
    }).ToArray();
    Console.WriteLine($"[prefix-probe] {label} ms={elapsed.Elapsed.TotalMilliseconds:F2} reused={string.Join(',', rows.Select(row => row.ReusedTokens))} generated={string.Join(',', rows.Select(row => row.ActualGeneratedTokens))} finish={string.Join(',', rows.Select(row => row.FinishReason))} ttft_ms={string.Join(',', rows.Select(row => row.TimeToFirstTokenMilliseconds?.ToString("F2")))}");
    CacheMemorySample? memory = null;
    lock (model.GpuComputeLock)
    {
        if (model is IPrefixCacheModelDiagnostics diagnostics && model is IPrefixCacheModel cache)
        {
            PayloadFootprint[] payloads = diagnostics.RetainedPayloadKeys.Select(cache.MeasureEndState).ToArray();
            memory = new CacheMemorySample(payloads.Length, diagnostics.PrivateHolderCount,
                diagnostics.PrimaryCacheLength, payloads.Sum(payload => payload.Bytes.HostKv),
                payloads.Sum(payload => payload.Bytes.StateSnapshot), payloads.Sum(payload => payload.Bytes.DeviceKv));
        }
    }
    return new PairRun(label, enabled, warm, elapsed.Elapsed.TotalMilliseconds,
        (engine.TotalForwardTime - before).TotalMilliseconds, true, rows, memory);
}

static void CheckPair(PairRun enabled, PairRun disabled, int shared, bool warm, int maxTokens, List<string> failures)
{
    foreach (PairRun run in new[] { enabled, disabled })
    {
        foreach (RequestRun request in run.Requests)
        {
            bool cleanEnd = request.CompletionStatus == nameof(SequenceStatus.FinishedLengthCapped)
                && request.ActualGeneratedTokens == maxTokens
                || request.CompletionStatus == nameof(SequenceStatus.FinishedStopped) && request.FinishReason == "eos";
            if (!cleanEnd || request.NonEosGeneratedTokens <= 0 || request.ActualGeneratedTokens > maxTokens
                || request.OutputTokens.Length != request.ActualGeneratedTokens)
                failures.Add($"{run.Label}/{request.Name}: unexpected completion {request.CompletionStatus}/{request.FinishReason} with {request.ActualGeneratedTokens} generated tokens (cap {maxTokens}).");
        }
    }
    for (int i = 0; i < 2; i++)
    {
        int expected = warm || i == 1 ? shared : 0;
        if (enabled.Requests[i].ReusedTokens != expected)
            failures.Add($"{enabled.Label}/{i}: expected {expected} reused tokens, got {enabled.Requests[i].ReusedTokens}.");
        if (disabled.Requests[i].ReusedTokens != 0)
            failures.Add($"{disabled.Label}/{i}: cache-disabled request unexpectedly reused tokens.");
        if (!enabled.Requests[i].OutputTokens.SequenceEqual(disabled.Requests[i].OutputTokens))
            failures.Add($"{enabled.Label}/{i}: greedy tokens differ from paired {disabled.Label}.");
        if (enabled.Requests[i].ActualGeneratedTokens != disabled.Requests[i].ActualGeneratedTokens
            || enabled.Requests[i].CompletionStatus != disabled.Requests[i].CompletionStatus
            || enabled.Requests[i].FinishReason != disabled.Requests[i].FinishReason)
            failures.Add($"{enabled.Label}/{i}: generated count, status or finish reason differs from paired {disabled.Label}.");
    }
}
static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}
static double Median(IEnumerable<double> values)
{
    double[] sorted = values.OrderBy(value => value).ToArray();
    return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
}
static string Hash(string path)
{
    using var stream = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(stream));
}
sealed record PromptFixture(string Name, int SharedPrefixTokens, List<int> Tokens, List<ChatMessage> Messages, List<ToolFunction> Tools);
sealed record RequestRun(string Name, int PromptTokens, int ReusedTokens, int PrefillTokens, int ActualGeneratedTokens, int NonEosGeneratedTokens,
    int[] OutputTokens, string OutputText, string CompletionStatus, string? FinishReason, double? TimeToFirstTokenMilliseconds);
sealed record PairRun(string Label, bool CacheEnabled, bool Warm, double Milliseconds, double EngineForwardMilliseconds,
    bool BothSubmittedBeforeGateOpened, RequestRun[] Requests, CacheMemorySample? CachedPayloadMemoryAfterCompletion);
sealed record SeedRun(string Label, int PromptTokens, double Milliseconds);
sealed record CacheMemorySample(int CachedPayloadCount, int PrivateHolderCount, int PrimaryCacheTokens,
    long CachedHostKvBytes, long CachedRecurrentStateBytes, long CachedDeviceMirrorBytes);
