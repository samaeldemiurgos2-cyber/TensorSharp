// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Observe the native cache, rather than treating unchanged logits as proof
/// that a release happened. This test also runs against the previous Models
/// assembly: it uses no new production API or model-internal test hook.
/// </summary>
[Collection("Muse Spark model")]
public sealed class MuseSparkQuantizedResidencyTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-opaque-residency-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly ITestOutputHelper _output;

    public MuseSparkQuantizedResidencyTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        _environment.Set("TS_KV_INITIAL_TOKENS", "2");
        _environment.Set("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
        _environment.Set("TS_MUSE_SPARK_STREAM_EXPERTS", "0");
        _environment.Set("TS_MUSE_SPARK_FUSE_QKVG", "0");
        _environment.Set("TS_MUSE_SPARK_LAYER_SPLIT", null);
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        Directory.Delete(_directory, true);
    }

    [MuseSparkTensorParallelGpuFact(2)]
    public void QuantizedOpaqueKeysReleaseAndReloadAcrossPlacementModes()
    {
        // The latent matrices and output head exceed the native 4096-byte
        // cache threshold, so their actual forward must reload resident copies.
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16,
            experts: 36, used: 2, headDim: 16, latent: 128, ffn: 32);
        string path = MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture,
            Path.Combine(_directory, "opaque-residency.gguf"));
        int[] tokens = [7, 21, 42, 11, 9];
        string[] requiredTrunk = ["output.weight", "blk.0.ffn_latent_down.weight",
            "blk.0.ffn_latent_up.weight", "blk.1.ffn_latent_down.weight", "blk.1.ffn_latent_up.weight"];
        var records = new List<object>();
        var failures = new List<string>();
        var checks = new MuseSparkGpuChecks(_output);

        foreach (var (degree, split) in new[] { (1, 1), (1, 2), (2, 1) })
        {
            using var model = new MuseSparkModel(path, BackendType.GgmlCuda,
                tpDegree: degree, layerSplitDegree: split);
            var weights = GetField<Dictionary<string, QuantizedWeight>>(model, "_quantWeights");
            var sharded = GetField<Dictionary<string, QuantizedWeight[]>>(model, "_tpQuantWeights");
            var keys = new List<WeightKey>();
            foreach (var (name, weight) in weights)
                AddKey(keys, "trunk", name, weight);
            foreach (var (name, shards) in sharded)
                for (int rank = 0; rank < shards.Length; rank++)
                    AddKey(keys, $"shard-{rank}", name, shards[rank]);

            int ranks = Math.Max(degree, split);
            float[] before = model.Forward(tokens);
            var residentBefore = Resident(keys, ranks);
            foreach (string name in requiredTrunk)
            {
                WeightKey key = Assert.Single(keys.Where(key => key.Scope == "trunk" && key.Name == name));
                Assert.True(key.Bytes >= 4096, $"Fixture key {name} must reach the native cache threshold.");
                int expectedRank = split == 2 && (name == "output.weight" || name.StartsWith("blk.1.", StringComparison.Ordinal)) ? 1 : 0;
                Assert.Equal([expectedRank], residentBefore.Where(item => item.Key == key.Key).Select(item => item.Rank));
            }
            if (degree > 1)
                for (int rank = 0; rank < degree; rank++)
                    Assert.Contains(keys, key => key.Scope == $"shard-{rank}" &&
                        residentBefore.Contains((rank, key.Key)));

            model.ReleaseGgmlDeviceResidency();
            var residentAfterRelease = Resident(keys, ranks);
            // Continue collecting all three modes before reporting an old-
            // binary failure; raw before/after/reload sets remain reviewable.
            if (residentAfterRelease.Count != 0)
                failures.Add($"TP={degree}, split={split}: {residentAfterRelease.Count} opaque cache entries survived release.");

            model.ResetKVCache();
            float[] after = model.Forward(tokens);
            var residentAfterReload = Resident(keys, ranks);
            foreach (string name in requiredTrunk)
            {
                WeightKey key = keys.Single(key => key.Scope == "trunk" && key.Name == name);
                int expectedRank = split == 2 && (name == "output.weight" || name.StartsWith("blk.1.", StringComparison.Ordinal)) ? 1 : 0;
                if (!residentAfterReload.Contains((expectedRank, key.Key)) ||
                    residentAfterReload.Any(item => item.Key == key.Key && item.Rank != expectedRank))
                    failures.Add($"TP={degree}, split={split}: used trunk key {name} did not reload only on rank {expectedRank}.");
            }
            // Do not demand eager reloading of every preloaded key. Only used
            // trunk keys above must reload; all enumerated keys must release.
            checks.Compare($"TP={degree}, split={split}, Q8 release/reload", before, after, 5e-5, 5e-5);
            records.Add(new
            {
                degree, split,
                keys = keys.Select(key => new { key.Scope, key.Name, cache_key = key.Key.ToString("x"),
                    host_data = key.Data.ToString("x"), key.Bytes }).ToArray(),
                before = Describe(keys, residentBefore),
                after_release = Describe(keys, residentAfterRelease),
                after_reload = Describe(keys, residentAfterReload),
                required_trunk = requiredTrunk,
            });
            _output.WriteLine($"TP={degree}, split={split}: opaque resident before={residentBefore.Count}, " +
                $"after release={residentAfterRelease.Count}, after lazy reload={residentAfterReload.Count}.");
        }

        string directory = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_TP_CASE_OUTPUT_DIRECTORY");
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
            using var stream = new FileStream(Path.Combine(directory, "quantized-opaque-residency-keys.json"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            JsonSerializer.Serialize(stream, new { records, failures }, new JsonSerializerOptions { WriteIndented = true });
        }
        checks.Complete("quantized-opaque-residency-logits");
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private static T GetField<T>(MuseSparkModel model, string name)
        => (T)typeof(ModelBase).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static void AddKey(List<WeightKey> keys, string scope, string name, QuantizedWeight weight)
    {
        if (weight != null && weight.CacheKey != IntPtr.Zero && weight.CacheKey != weight.Data)
            keys.Add(new WeightKey(scope, name, weight.CacheKey, weight.Data, weight.RawBytes));
    }

    private static HashSet<(int Rank, IntPtr Key)> Resident(List<WeightKey> keys, int ranks)
    {
        var found = new HashSet<(int, IntPtr)>();
        int previousRank = GgmlBasicOps.GetActiveRank();
        bool previousPin = GgmlBasicOps.PinRank(0);
        try
        {
            for (int rank = 0; rank < ranks; rank++)
            {
                GgmlBasicOps.SetActiveRank(rank);
                foreach (WeightKey key in keys)
                    if (GgmlBasicOps.HasResidentDeviceCopy(key.Key, key.Bytes)) found.Add((rank, key.Key));
            }
        }
        finally { GgmlBasicOps.RestoreRankPin(previousPin, previousRank); }
        return found;
    }

    private static object[] Describe(List<WeightKey> keys, HashSet<(int Rank, IntPtr Key)> resident)
        => resident.OrderBy(item => item.Rank).ThenBy(item => item.Key.ToInt64())
            .Select(item => (object)new { rank = item.Rank, cache_key = item.Key.ToString("x"),
                names = keys.Where(key => key.Key == item.Key).Select(key => key.Scope + ":" + key.Name).ToArray() }).ToArray();

    private sealed record WeightKey(string Scope, string Name, IntPtr Key, IntPtr Data, long Bytes);
}
