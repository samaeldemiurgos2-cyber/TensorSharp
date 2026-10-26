// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Distributed;
using TensorSharp.Models;
using TensorSharp.Runtime;

// Fixed teacher-forced inputs make every topology execute the same operations,
// including after a near-tie changes a greedy token. This is numerical evidence;
// fixture tokens and timings do not measure trained-model quality or scaling.
var options = new Dictionary<string, string>(StringComparer.Ordinal);
try
{
    if (args.Length % 2 != 0) throw new ArgumentException("Every option requires a value.");
    for (int i = 0; i < args.Length; i += 2)
        if (!options.TryAdd(args[i], args[i + 1])) throw new ArgumentException("Duplicate option: " + args[i]);
    var allowed = new HashSet<string> { "--model", "--tokens", "--text", "--output", "--weights-id", "--tp", "--layer-split",
        "--tp-node-id", "--tp-peers", "--chunk", "--decode-count", "--backend" };
    foreach (string key in options.Keys)
        if (!allowed.Contains(key)) throw new ArgumentException("Unknown option: " + key);
    foreach (string key in new[] { "--model", "--output", "--weights-id" })
        if (!options.ContainsKey(key) || string.IsNullOrWhiteSpace(options[key]))
            throw new ArgumentException("Required option: " + key);
    if (options.ContainsKey("--tokens") == options.ContainsKey("--text"))
        throw new ArgumentException("Specify exactly one of --tokens TOKENS.json or --text UTF8_TEXT_FILE.");
}
catch (ArgumentException error)
{
    Console.Error.WriteLine(error.Message);
    Console.Error.WriteLine("Usage: DeepSeek41DistributedProbe --model MODEL.gguf (--tokens TOKENS.json | --text UTF8_TEXT_FILE) " +
        "--weights-id IMMUTABLE_ID --output FRESH_DIRECTORY [--tp 2 --tp-node-id 0 --tp-peers HOST:PORT,HOST:PORT] " +
        "[--chunk 3 --decode-count 4 --backend ggml_cuda]");
    return 2;
}

string output = Path.GetFullPath(options["--output"]);
if (Directory.Exists(output) || File.Exists(output))
{
    Console.Error.WriteLine("Output must be fresh; preserve previous evidence: " + output);
    return 2;
}
Directory.CreateDirectory(output);
var steps = new List<object>();
var report = new Dictionary<string, object?>
{
    ["schema_version"] = 1, ["status"] = "running", ["steps"] = steps,
    ["weights_id"] = options["--weights-id"], ["arguments"] = args,
    ["limitations"] = new[] { "Fixed-token numerical/control probe; not a model-quality or throughput benchmark.",
        "Model identity is caller-supplied; retain a separate full-shard verification report.",
        "Workers report loop return, not a numerical pass; compare the driver's vectors with an independent/local control." }
};
void Save() => File.WriteAllText(Path.Combine(output, "probe.json"), JsonSerializer.Serialize(report,
    new JsonSerializerOptions { WriteIndented = true }) + "\n");
ModelBase? model = null;
ITensorParallelGroup? unownedGroup = null;
int exitCode = 1;
try
{
    int chunk = int.Parse(options.GetValueOrDefault("--chunk", "3"));
    int decodeCount = int.Parse(options.GetValueOrDefault("--decode-count", "4"));
    int[] tokens;
    byte[] tokenSource;
    if (options.TryGetValue("--text", out string? textPath))
    {
        // Real checkpoint vocabularies often reserve the small IDs used by the
        // synthetic fixture. Tokenize actual text for meaningful real-weight
        // comparisons instead of reusing those fixture IDs across families.
        using var header = GgufFile.OpenWithoutSiblingShards(options["--model"]);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(header);
        byte[] textBytes = File.ReadAllBytes(textPath);
        tokens = tokenizer.Encode(System.Text.Encoding.UTF8.GetString(textBytes), false).ToArray();
        tokenSource = JsonSerializer.SerializeToUtf8Bytes(tokens);
        File.WriteAllBytes(Path.Combine(output, "tokens.json"), tokenSource);
        report["text_sha256"] = Convert.ToHexStringLower(SHA256.HashData(textBytes));
        report["text"] = System.Text.Encoding.UTF8.GetString(textBytes);
    }
    else
    {
        tokenSource = File.ReadAllBytes(options["--tokens"]);
        tokens = JsonSerializer.Deserialize<int[]>(tokenSource)
            ?? throw new ArgumentException("Token file must contain a JSON array.");
    }
    if (chunk < 1 || decodeCount < 1 || tokens.Length < decodeCount + 2 || tokens.Any(x => x < 0))
        throw new ArgumentException("Need positive chunk/decode-count and at least decode-count + 2 nonnegative tokens.");
    report["tokens"] = tokens;
    report["tokens_sha256"] = Convert.ToHexStringLower(SHA256.HashData(tokenSource));
    BackendType backend = options.GetValueOrDefault("--backend", "ggml_cuda") switch
    {
        "ggml_cuda" => BackendType.GgmlCuda,
        "ggml_cpu" => BackendType.GgmlCpu,
        _ => throw new ArgumentException("Probe backend must be ggml_cuda or ggml_cpu.")
    };
    // Match the CLI/server initialization contract: these are managed settings,
    // so exporting the variables alone does not configure a standalone host.
    KvCacheDtypeConfig.ConfigureFromEnvironment();
    MoeCpuOffloadConfig.ConfigureFromEnvironment();
    report["cpu_offload"] = new
    {
        requested_layers = Environment.GetEnvironmentVariable("TS_N_CPU_MOE"),
        requested_all = Environment.GetEnvironmentVariable("TS_CPU_MOE"),
        requested_threads = Environment.GetEnvironmentVariable("TS_CPU_MOE_THREADS"),
        configured_layers = MoeCpuOffloadConfig.CpuMoeLayers,
        configured_all = MoeCpuOffloadConfig.AllLayers,
        explicitly_set = MoeCpuOffloadConfig.IsExplicitlySet,
        configured_threads = MoeCpuOffloadConfig.CpuThreads
    };
    report["requested_kv_cache_dtype"] = Environment.GetEnvironmentVariable("KV_CACHE_DTYPE");
    string[] parallelArgs = options.Where(pair => pair.Key is "--tp" or "--layer-split" or "--tp-node-id" or "--tp-peers")
        .SelectMany(pair => new[] { pair.Key, pair.Value }).ToArray();
    var parallel = ModelParallelismOptions.Parse(parallelArgs);
    if (parallel.Distributed is { } distributed)
    {
        if (parallel.IsLayerSplit)
        {
            ModelBase.ValidateDistributedLayerSplit(options["--model"], backend);
            unownedGroup = new DistributedLayerSplitGroup(
                ModelBase.CreateGgmlLocalLayerSplitGroup(backend, distributed.LocalDegree),
                distributed.NodeId, distributed.PeerEndpoints);
        }
        else
        {
            ModelBase.ValidateDistributedTensorParallelism(options["--model"], backend);
            unownedGroup = new DistributedTensorParallelGroup(
                ModelBase.CreateGgmlLocalTpGroup(backend, distributed.LocalDegree),
                distributed.NodeId, distributed.PeerEndpoints);
        }
    }
    var timer = Stopwatch.StartNew();
    model = ModelBase.Create(options["--model"], backend, parallel.TpDegree, unownedGroup!,
        layerSplitDegree: parallel.LayerSplitDegree);
    ITensorParallelGroup? group = unownedGroup;
    unownedGroup = null; // ModelBase owns the supplied group from this point.
    report["load_seconds"] = timer.Elapsed.TotalSeconds;
    int localDegree = parallel.IsLayerSplit ? parallel.LayerSplitDegree : parallel.TpDegree;
    report["topology"] = new { mode = parallel.IsLayerSplit ? "layer_split" : "tensor_parallel",
        local_degree = localDegree, global_degree = group?.GlobalDegree ?? localDegree,
        rank_offset = group?.GlobalRankOffset ?? 0, node_count = group?.NodeCount ?? 1 };
    report["vocab_size"] = model.Config.VocabSize;
    if (tokens.Any(x => x >= model.Config.VocabSize)) throw new ArgumentException("Token exceeds model vocabulary.");
    group?.Barrier();
    Save();
    if (model.IsDistributedWorker)
    {
        model.RunDistributedWorkerLoop();
        report["status"] = "worker_loop_returned";
    }
    else
    {
        // No hidden warmup: every executed forward is persisted below. Both
        // nodes have completed construction and the barrier before controls start.
        model.BeginDistributedDriver();
        int prefixLength = tokens.Length - decodeCount;
        void Control(string operation, Action action)
        {
            action();
            steps.Add(new { operation, tokens = Array.Empty<int>(), logits = (string?)null });
            Save();
        }
        void Forward(string operation, int[] input, bool refill = false)
        {
            timer.Restart();
            float[] logits = refill ? model.ForwardRefill(input) : model.Forward(input);
            double seconds = timer.Elapsed.TotalSeconds;
            if (logits.Length != model.Config.VocabSize || logits.Any(x => !float.IsFinite(x)))
                throw new InvalidOperationException("Empty, nonfinite or incorrectly sized logits.");
            string file = $"step-{steps.Count:D3}-{operation}.f32";
            using (var writer = new BinaryWriter(File.Create(Path.Combine(output, file))))
                foreach (float value in logits) writer.Write(value);
            steps.Add(new { operation, tokens = input, logits = file, count = logits.Length,
                argmax = Array.IndexOf(logits, logits.Max()), seconds });
            Save();
        }
        Control("reset", model.ResetKVCache);
        Forward("prefill", tokens[..prefixLength]);
        for (int i = prefixLength; i < tokens.Length; i++) Forward("decode", new[] { tokens[i] });
        Control("reset", model.ResetKVCache);
        for (int i = 0; i < prefixLength; i += chunk)
            Forward("chunk", tokens[i..Math.Min(prefixLength, i + chunk)]);
        for (int i = prefixLength; i < tokens.Length; i++) Forward("refill", new[] { tokens[i] }, refill: true);
        Control("reset", model.ResetKVCache);
        Forward("reset-prefill", tokens[..prefixLength]);
        report["cache_truncation"] = "not_requested; the distributed executor serves a single mirrored sequence";
        report["status"] = "completed";
    }
    exitCode = 0;
}
catch (Exception error)
{
    report["status"] = "failed";
    report["error"] = error.ToString();
    Console.Error.WriteLine(error);
}
finally
{
    try
    {
        model?.Dispose();
        unownedGroup?.Dispose();
        report["disposed"] = true;
    }
    catch (Exception error)
    {
        report["status"] = "failed";
        report["disposal_error"] = error.ToString();
        exitCode = 1;
    }
    Save();
}
return exitCode;
