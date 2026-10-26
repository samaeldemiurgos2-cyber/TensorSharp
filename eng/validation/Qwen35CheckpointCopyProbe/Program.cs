// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

string Option(string name, string fallback)
{
    int i = Array.IndexOf(args, name);
    return i < 0 ? fallback : i + 1 < args.Length ? args[i + 1] : throw new ArgumentException(name);
}
string path = Option("--model", Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MODEL") ?? "");
string output = Path.GetFullPath(Option("--out", "artifacts/qwen35-checkpoint-copy/report.json"));
string fixturePath = Option("--fixture", "");
int steps = int.Parse(Option("--steps", "8"));
int chunk = int.Parse(Option("--chunk", "2048"));
int syntheticTokens = int.Parse(Option("--prompt-tokens", "2231"));
int syntheticPrefix = int.Parse(Option("--prefix-tokens", "2072"));
if (steps is < 1 or > 128 || chunk < 1 || syntheticTokens < 2
    || syntheticPrefix < 1 || syntheticPrefix >= syntheticTokens)
    throw new ArgumentOutOfRangeException("steps/chunk/prompt-tokens/prefix-tokens");
KvCacheDtypeConfig.ConfigureFromEnvironment();
var cases = new List<CaseResult>();
string? error = null, kvType = null;
bool complete = false;
try
{
    using var model = ModelBase.Create(path, BackendType.GgmlCuda);
    if (model is not Qwen35Model qwen) throw new InvalidOperationException("Requires Qwen35.");
    kvType = model.KvCacheDtype.ToString();
    var fixtures = new List<(string Name, int[] Tokens, int Prefix)>();
    if (fixturePath.Length > 0)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(fixturePath));
        foreach (var f in doc.RootElement.GetProperty("Fixtures").EnumerateArray())
            fixtures.Add((f.GetProperty("Name").GetString()!,
                f.GetProperty("Tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray(),
                f.GetProperty("SharedPrefixTokens").GetInt32()));
    }
    else
    {
        int[] text = model.Tokenizer.Encode("The river flows through a green valley. Trees grow beside the water. ", addSpecial: false).ToArray();
        fixtures.Add(("repeated-text", Enumerable.Range(0, syntheticTokens).Select(i => text[i % text.Length]).ToArray(), syntheticPrefix));
    }
    foreach (var f in fixtures)
    foreach (int prefix in new[] { f.Prefix, f.Tokens.Length - 1 }.Distinct())
    {
        if (prefix <= 0 || prefix >= f.Tokens.Length || f.Tokens.Length + steps + 1 > model.MaxContextLength)
            throw new ArgumentException("Invalid checkpoint boundary or insufficient context.");
        cases.Add(Probe.Run(qwen, f.Name, f.Tokens, prefix, chunk, steps));
    }
    complete = true;
}
catch (Exception ex)
{
    error = ex.ToString();
    Console.Error.WriteLine(error);
}
bool passed = complete && cases.Count > 0 && cases.All(c => c.Passed);
string? native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .FirstOrDefault(m => Path.GetFileName(m.FileName) is "libGgmlOps.so" or "libGgmlOps.dylib" or "GgmlOps.dll")?.FileName;
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
File.WriteAllText(output, JsonSerializer.Serialize(new
{
    ValidationPassed = passed, Completed = complete, PerformanceQualified = false, Error = error,
    Model = Path.GetFullPath(path), KvDtype = kvType, Fixture = fixturePath, Steps = steps, PrefillChunk = chunk,
    ModelAssemblySha256 = Probe.HashFile(typeof(Qwen35Model).Assembly.Location),
    RuntimeAssemblySha256 = Probe.HashFile(typeof(SequenceState).Assembly.Location),
    NativePath = native, NativeSha256 = native is null ? null : Probe.HashFile(native),
    GgmlRevision = Environment.GetEnvironmentVariable("TS_VALIDATION_GGML_REVISION"),
    Device = Environment.GetEnvironmentVariable("TS_VALIDATION_DEVICE"),
    NativeEnvironment = new
    {
        Pdl = Environment.GetEnvironmentVariable("GGML_CUDA_PDL"),
        DisableGraphs = Environment.GetEnvironmentVariable("GGML_CUDA_DISABLE_GRAPHS"),
        HostHugePages = Environment.GetEnvironmentVariable("TS_GGML_HOST_HUGEPAGES"),
    },
    Execution = ExecutionOptions.FromEnvironment(), Cases = cases,
    Gates = new { MinimumCosine = 0.995, MaximumSoftmaxKl = 0.02, MaximumReferenceMarginForArgmaxFlip = 0.25 },
    Limitations = "One live prefill state is checkpointed, then its unchanged source and fresh/pooled production clones receive identical suffix chunks and teacher-forced greedy-reference tokens. No arena batching is called. Live KV hashes are taken after the original full host synchronization, so checkpoint hash equality checks the optimized partial download against full synchronization. Valid KV rows, GDN ring/index/delta, lengths and rope offset are hashed; checkpoint and clone KV padding must be zero. Host hashes and forced synchronization perturb execution: these are correctness diagnostics, not performance measurements. A successful pooled arm additionally requires object identity of both the holder and its first K storage with the preceding released fresh clone, excluding address reuse after a new mmap. No cold-vs-partitioned-prefill equivalence or task-quality claim. Numerical gates exactly match Qwen35DecodeProbe; all token flips and first-token metrics are retained. Only the named model, dtype, backend and prefixes are covered."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"[checkpoint-copy] passed={passed} report={output}");
Environment.ExitCode = passed ? 0 : 1;

static class Probe
{
    private const BindingFlags InstanceFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    private static readonly Type ModelType = typeof(Qwen35Model);
    private static object Get(object owner, string name) => owner.GetType().GetField(name, InstanceFlags)!.GetValue(owner)!;
    private static object Invoke(Qwen35Model model, string name) => ModelType.GetMethod(name, InstanceFlags)!.Invoke(model, null)!;
    private static object Holder(Qwen35Model model, string dictionary, string key) => ((IDictionary)Get(model, dictionary))[key]!;

    public static CaseResult Run(Qwen35Model model, string name, int[] tokens, int prefix, int chunk, int steps)
    {
        var sequences = (IBatchedPagedModel)model;
        var cache = (IPrefixCacheModel)model;
        string tag = Guid.NewGuid().ToString("N"), source = tag + "-live", key = tag + "-checkpoint";
        string fresh = tag + "-fresh", pooled = tag + "-pooled";
        var ids = new[] { source, fresh, pooled };
        var results = new List<ArmResult>();
        try
        {
            sequences.RestorePrimaryCache();
            model.TrimIdleMemory();
            sequences.BindSequenceCache(source);
            int required = tokens.Length + steps + 1;
            model.PrepareForPrefill(required);
            var segments = new List<int>();
            for (int p = 0; p < prefix;)
            {
                int n = Math.Min(chunk, prefix - p);
                model.Forward(tokens.AsSpan(p, n).ToArray());
                segments.Add(n);
                p += n;
            }
            if (!sequences.TryCheckpointActiveCache(key)) throw new InvalidOperationException("Capture declined.");
            object snapshot = Holder(model, "_retainedFusedHolders", key);
            var checkpoint = HashHolder(snapshot, prefix, checkPadding: true);
            // Independently compare the partial checkpoint sync with the original
            // full sync before advancing the live source. No second copy is used.
            Invoke(model, "EnsureKvCacheHostSynchronized");
            Invoke(model, "EnsureFusedDecodeStateHostSynchronized");
            Invoke(model, "DrainDeviceRecurrentState");
            object liveHolder = Invoke(model, "SnapshotActiveCache");
            var liveHash = HashHolder(liveHolder, prefix, checkPadding: false);
            int reservation = cache.ResolveCloneReservationTokens(key, required);
            if (reservation <= 0) throw new InvalidOperationException("Request-sized clone reservation unavailable.");
            int liveCapacity = (int)Get(liveHolder, "KvCapacity");
            if (reservation != liveCapacity)
                throw new InvalidOperationException($"Unmatched capacities: live={liveCapacity}, clone={reservation}.");
            float[][] reference = Continue(model, tokens, prefix, chunk, steps, null, out int[] referenceTokens);
            results.Add(new("live", liveHash, false, false, liveCapacity, referenceTokens, Array.Empty<LogitResult>()));
            sequences.RestorePrimaryCache();
            nint freshPointer = 0;
            object? releasedFreshHolder = null;
            Storage? releasedFreshStorage = null;
            foreach (var arm in new[] { (Id: fresh, Label: "fresh-clone"), (Id: pooled, Label: "pooled-clone") })
            {
                if (!cache.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, key, arm.Id, prefix, prefix, reservation)))
                    throw new InvalidOperationException("Materialization declined: " + arm.Label);
                object holder = Holder(model, "_fusedHolders", arm.Id);
                var hash = HashHolder(holder, prefix, checkPadding: true);
                Storage storage = ((Tensor[])Get(holder, "K")).First(t => t != null).Storage;
                nint pointer = storage.PtrAtElement(0);
                bool reusedPointer = arm.Label == "pooled-clone" && pointer == freshPointer;
                bool reusedOwner = arm.Label == "pooled-clone" && ReferenceEquals(holder, releasedFreshHolder)
                    && ReferenceEquals(storage, releasedFreshStorage);
                if (arm.Label == "fresh-clone") freshPointer = pointer;
                sequences.BindSequenceCache(arm.Id);
                float[][] observed = Continue(model, tokens, prefix, chunk, steps, referenceTokens, out int[] topTokens);
                var comparisons = reference.Select((row, i) => Compare(row, observed[i], i)).ToArray();
                results.Add(new(arm.Label, hash, reusedPointer, reusedOwner, (int)Get(holder, "KvCapacity"), topTokens, comparisons));
                sequences.RestorePrimaryCache();
                if (arm.Label == "fresh-clone")
                {
                    // Restoring the primary snapshots the checked-out holder into
                    // a new wrapper. Retain the exact wrapper returned to the pool.
                    releasedFreshHolder = Holder(model, "_fusedHolders", arm.Id);
                    releasedFreshStorage = ((Tensor[])Get(releasedFreshHolder, "K")).First(t => t != null).Storage;
                }
                sequences.OnSequenceReleased(arm.Id);
            }
            bool hashes = checkpoint.PaddingIsZero && results.All(r => r.State.Hash == checkpoint.Hash && (r.Name == "live" || r.State.PaddingIsZero));
            bool numerics = results.All(r => r.Logits.All(l => l.Passed));
            ArmResult pooledArm = results.Single(r => r.Name == "pooled-clone");
            bool pool = pooledArm.ReusedPreviousClonePointer && pooledArm.ReusedPreviousCloneOwner;
            bool tokensMatch = results.All(r => r.TopTokenIds.SequenceEqual(referenceTokens));
            Console.WriteLine($"[checkpoint-copy] {name} prefix={prefix} suffix={tokens.Length-prefix} hash={hashes} numerical={numerics} pool={pool} exact_tokens={tokensMatch}");
            return new(name, tokens.Length, prefix, tokens.Length - prefix, segments.ToArray(),
                checkpoint, hashes, numerics, pool, tokensMatch, hashes && numerics && pool, results);
        }
        finally
        {
            sequences.RestorePrimaryCache();
            foreach (string id in ids) sequences.OnSequenceReleased(id);
            sequences.DiscardRetainedCache(key);
        }
    }

    private static float[][] Continue(Qwen35Model model, int[] tokens, int prefix, int chunk, int steps,
        int[]? forced, out int[] topTokens)
    {
        float[]? logits = null;
        for (int p = prefix; p < tokens.Length;)
        {
            int n = Math.Min(chunk, tokens.Length - p);
            logits = model.Forward(tokens.AsSpan(p, n).ToArray());
            p += n;
        }
        var result = new float[steps][];
        topTokens = new int[steps];
        for (int i = 0; i < steps; i++)
        {
            result[i] = (float[])logits!.Clone();
            topTokens[i] = ArgMax(result[i]);
            if (i + 1 < steps) logits = model.Forward([forced?[i] ?? topTokens[i]]);
        }
        return result;
    }

    private static unsafe StateDigest HashHolder(object holder, int rows, bool checkPadding)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        bool zero = true;
        long bytes = 0;
        foreach (string field in new[] { "K", "V" })
        foreach (Tensor? tensor in (Tensor[])Get(holder, field))
        {
            if (tensor == null) continue;
            tensor.Storage.EnsureHostReadable();
            long capacity = tensor.Sizes[1], rowBytes = tensor.Storage.ByteLength / (tensor.Sizes[0] * capacity);
            byte* start = (byte*)tensor.Storage.PtrAtElement(0);
            for (int head = 0; head < tensor.Sizes[0]; head++)
            {
                int valid = checked((int)(rows * rowBytes));
                hash.AppendData(new ReadOnlySpan<byte>(start + head * capacity * rowBytes, valid));
                bytes += valid;
                if (checkPadding)
                    zero &= new ReadOnlySpan<byte>(start + (head * capacity + rows) * rowBytes,
                        checked((int)((capacity - rows) * rowBytes))).IndexOfAnyExcept((byte)0) < 0;
            }
        }
        foreach (float[]? ring in (float[][])Get(holder, "ConvState"))
            if (ring != null) hash.AppendData(MemoryMarshal.AsBytes(ring.AsSpan()));
        hash.AppendData(MemoryMarshal.AsBytes(((int[])Get(holder, "ConvWriteIdx")).AsSpan()));
        foreach (Tensor? state in (Tensor[])Get(holder, "DeltaState"))
        {
            if (state == null) continue;
            state.Storage.EnsureHostReadable();
            var ptr = (IntPtr)ModelType.GetMethod("GdnDeltaStatePointer", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, [state])!;
            long count = (long)ModelType.GetMethod("GdnDeltaStateBytes", BindingFlags.Static | BindingFlags.NonPublic)!.Invoke(null, [state])!;
            hash.AppendData(new ReadOnlySpan<byte>((void*)ptr, checked((int)count)));
        }
        foreach (string field in new[] { "CacheSeqLen", "RopeDelta", "MtpCacheStart" })
            hash.AppendData(BitConverter.GetBytes((int)Get(holder, field)));
        return new(Convert.ToHexString(hash.GetHashAndReset()), bytes, zero, (int)Get(holder, "KvCapacity"));
    }

    private static LogitResult Compare(float[] reference, float[] observed, int step)
    {
        if (reference.Length != observed.Length || reference.Length < 2) throw new InvalidOperationException("Logit shape mismatch.");
        if (reference.Any(x => !float.IsFinite(x)) || observed.Any(x => !float.IsFinite(x)))
            return new(step, false, false, null, null, null, null, null, null, null, null);
        int a = ArgMax(reference), b = ArgMax(observed);
        double dot = 0, aa = 0, bb = 0, squared = 0, sum = 0, absolute = 0, sa = 0, sb = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            dot += (double)reference[i] * observed[i]; aa += (double)reference[i] * reference[i]; bb += (double)observed[i] * observed[i];
            sum += reference[i]; double e = (double)observed[i] - reference[i]; squared += e * e; absolute = Math.Max(absolute, Math.Abs(e));
            sa += Math.Exp((double)reference[i] - reference[a]); sb += Math.Exp((double)observed[i] - observed[b]);
        }
        double la = Math.Log(sa), lb = Math.Log(sb), kl = 0, tv = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double p = (double)reference[i] - reference[a] - la, q = (double)observed[i] - observed[b] - lb;
            kl += Math.Exp(p) * (p - q); tv += Math.Abs(Math.Exp(p) - Math.Exp(q)) / 2;
        }
        kl = Math.Max(0, kl);
        double cosine = dot / Math.Sqrt(Math.Max(1e-20, aa * bb));
        double variance = Math.Max(1e-20, aa / reference.Length - Math.Pow(sum / reference.Length, 2));
        double nrmse = Math.Sqrt(squared / reference.Length / variance);
        double margin = (double)reference[a] - reference.Where((_, i) => i != a).Max();
        bool pass = double.IsFinite(cosine) && cosine >= 0.995 && double.IsFinite(kl) && kl <= 0.02 && (a == b || margin <= 0.25);
        return new(step, true, pass, cosine, nrmse, absolute, kl, tv, a, b, margin);
    }
    private static int ArgMax(float[] logits)
    {
        int top = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[top]) top = i;
        return top;
    }
    public static string HashFile(string file)
    {
        using var stream = File.OpenRead(file);
        return Convert.ToHexString(SHA256.HashData(stream));
    }
}

sealed record StateDigest(string Hash, long ValidKvBytes, bool PaddingIsZero, int Capacity);
sealed record LogitResult(int Step, bool Finite, bool Passed, double? Cosine, double? NormalizedRmse,
    double? MaxAbsoluteError, double? SoftmaxKl, double? TotalVariation, int? ReferenceTopToken, int? ObservedTopToken, double? ReferenceMargin);
sealed record ArmResult(string Name, StateDigest State, bool ReusedPreviousClonePointer, bool ReusedPreviousCloneOwner, int Capacity,
    int[] TopTokenIds, LogitResult[] Logits);
sealed record CaseResult(string Name, int PromptTokens, int PrefixTokens, int SuffixTokens, int[] PrefixSegments,
    StateDigest Checkpoint, bool StateHashesMatch, bool NumericalPassed, bool PoolReuseObserved,
    bool ExactTopTokensMatch, bool Passed, IReadOnlyList<ArmResult> Arms);
