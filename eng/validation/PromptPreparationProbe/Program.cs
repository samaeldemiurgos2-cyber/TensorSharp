using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;

string Option(string name, string fallback = "")
{
    int at = Array.IndexOf(args, name);
    return at < 0 ? fallback : at + 1 < args.Length ? args[at + 1]
        : throw new ArgumentException($"Missing value for {name}.");
}

string model = Option("--model");
string output = Option("--out", "artifacts/prompt-preparation/report.json");
string systemTextFile = Option("--system-text");
int repeats = int.Parse(Option("--repeats", "5"));
int chars = int.Parse(Option("--characters", "65000"));
if (model.Length == 0 || repeats is < 1 or > 100 || chars is < 1000 or > 1000000)
    throw new ArgumentException("Pass --model <BPE GGUF> [--repeats 1..100] [--characters 1000..1000000] [--out report.json].");

// Read vocabulary/template metadata only: the probe never loads model weights or
// initializes a GPU. Both revisions must use this exact probe and GGUF metadata.
using var gguf = GgufFile.OpenWithoutSiblingShards(model);
string[] vocabulary = gguf.GetStringArray("tokenizer.ggml.tokens")
    ?? throw new InvalidDataException("Missing GGUF vocabulary.");
string[] merges = gguf.GetStringArray("tokenizer.ggml.merges")
    ?? throw new InvalidDataException("This probe requires a BPE tokenizer.");
int[] types = gguf.GetInt32Array("tokenizer.ggml.token_type") ?? new int[vocabulary.Length];
int bos = (int)gguf.GetUint32("tokenizer.ggml.bos_token_id", uint.MaxValue);
int eos = (int)gguf.GetUint32("tokenizer.ggml.eos_token_id", uint.MaxValue);
string architecture = gguf.GetString("general.architecture") ?? "qwen35";
string template = gguf.GetString("tokenizer.chat_template") ?? "";
string? preType = gguf.GetString("tokenizer.ggml.model") == "gemma4"
    ? "gemma4" : gguf.GetString("tokenizer.ggml.pre");
BpeTokenizer CreateTokenizer() => new(vocabulary, types, merges, bos,
    eos < 0 ? [] : [eos], gguf.GetBool("tokenizer.ggml.add_bos_token"),
    gguf.GetBool("tokenizer.ggml.add_eos_token"), preType);

const string paragraph = "Read the current page, identify the requested action, and return one tool call. "
    + "Preserve all arguments exactly. Inspect the visible text before acting. "
    + "A screenshot can contain tables, menu entries, browser tabs, and application controls. "
    + "Use the provided function schema and report the observed result.\n";
var system = new StringBuilder(chars + paragraph.Length);
string systemSource = systemTextFile.Length == 0 ? paragraph : File.ReadAllText(systemTextFile);
if (systemSource.Length == 0) throw new ArgumentException("System text must not be empty.");
while (system.Length < chars) system.Append(systemSource);
system.Length = chars;
var tools = Enumerable.Range(0, 34).Select(i => new ToolFunction
{
    Name = $"inspect_surface_{i}",
    Description = "Inspect a selected application surface and return its visible text and available controls.",
    Parameters = new Dictionary<string, ToolParameter>
    {
        ["target"] = new() { Type = "string", Description = "The exact surface identifier from the preceding observation." },
        ["include_text"] = new() { Type = "boolean", Description = "Whether to include visible text." },
    },
    Required = ["target"],
}).ToList();
var histories = Enumerable.Range(0, 4).Select(lane => new List<ChatMessage>
{
    new() { Role = "system", Content = system.ToString() },
    new() { Role = "user", Content = $"Lane {lane}: inspect the table. Which row contains item 472?" },
}).ToArray();
var renderer = new GgufPromptRenderer();
var kvRenderer = new KVCachePromptRenderer(renderer);
var rows = new List<object>();
var tokenizer = CreateTokenizer();

string Render(List<ChatMessage> history) => renderer.Render(template, history,
    addGenerationPrompt: true, architecture: architecture, tools: tools, enableThinking: false);
List<int> Prepare(List<ChatMessage> history) => kvRenderer.RenderToTokens(tokenizer, template,
    history, architecture, addGenerationPrompt: true, tools: tools, enableThinking: false);
string TokenHash(List<int> tokens) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(tokens))));

// Suppress routine template diagnostics in the timed region. Their I/O cost is
// intentionally outside this CPU render/tokenizer probe and remains in HTTP TTFT.
TextWriter diagnostics = Console.Error;
Console.SetError(TextWriter.Null);
try
{
    string rendered = Render(histories[0]);
    List<int> cold = Measure("cold-tokenize", 1, () => tokenizer.Encode(rendered));
    // Let hot methods tier up before comparing individual phases. Otherwise an
    // early tokenization row can include Tier-0 code while the later combined
    // phase already uses optimized code, making the phases incomparable.
    var warmup = Stopwatch.StartNew();
    while (warmup.Elapsed < TimeSpan.FromSeconds(1))
    {
        Render(histories[0]);
        tokenizer.Encode(rendered);
        Prepare(histories[0]);
    }
    for (int i = 0; i < repeats; i++)
    {
        Measure("render", 1, () => Render(histories[0]));
        List<int> warm = Measure("tokenize", 1, () => tokenizer.Encode(rendered));
        if (!cold.SequenceEqual(warm)) throw new InvalidDataException("Warm token ids differ from cold.");
    }
    foreach (int lanes in new[] { 1, 2, 4 })
    {
        for (int i = 0; i < repeats; i++)
        {
            var prepared = Measure("render-and-tokenize", lanes, () =>
            {
                var result = new List<int>[lanes];
                Parallel.For(0, lanes, lane => result[lane] = Prepare(histories[lane]));
                return result;
            });
            if (!cold.SequenceEqual(prepared[0])) throw new InvalidDataException("Parallel render token ids differ.");
        }
    }
    // Real growing histories, as distinct from resubmitting identical strings.
    for (int turn = 0; turn < 4; turn++)
    {
        histories[0].Add(new ChatMessage { Role = "assistant", Content = $"Item 472 is in row {turn + 1}." });
        histories[0].Add(new ChatMessage { Role = "user", Content = string.Concat(Enumerable.Repeat(paragraph, 25)) + $"Next observation {turn}." });
        List<int> ids = Measure($"continuation-{turn + 1}", 1, () => Prepare(histories[0]));
        var reference = new KVCachePromptRenderer(renderer).RenderToTokens(CreateTokenizer(), template,
            histories[0], architecture, true, tools, false);
        if (!ids.SequenceEqual(reference)) throw new InvalidDataException("Continuation ids differ from fresh tokenizer.");
    }

    var report = new
    {
        model = Path.GetFullPath(model), modelBytes = new FileInfo(model).Length,
        architecture, preType, vocabularySize = vocabulary.Length,
        vocabularyHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new { vocabulary, merges, types })))),
        templateHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(template))),
        runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
        processArchitecture = RuntimeInformation.ProcessArchitecture.ToString(), processors = Environment.ProcessorCount,
        warmupSeconds = 1,
        characters = rendered.Length, promptTokens = cold.Count, tokenHash = TokenHash(cold),
        systemTextSource = systemTextFile.Length == 0 ? "repeated synthetic instructions" : Path.GetFullPath(systemTextFile),
        systemTextHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(system.ToString()))),
        limitations = "Synthetic 34-tool text prompts; CPU-only, no GPU/HTTP/vision or end-to-end parity claim. Identical probe and metadata required for revision comparison.",
        rows,
    };
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Saved {rows.Count} measurements: {output}; prompt={cold.Count} tokens.");
}
finally
{
    Console.SetError(diagnostics);
}

T Measure<T>(string name, int lanes, Func<T> action)
{
    long before = GC.GetTotalAllocatedBytes(precise: true);
    var watch = Stopwatch.StartNew();
    T result = action();
    double milliseconds = watch.Elapsed.TotalMilliseconds;
    long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
    string? hash = result is List<int> ids ? TokenHash(ids) : null;
    rows.Add(new { name, lanes, milliseconds, allocatedBytes = allocated, tokenHash = hash });
    return result;
}
