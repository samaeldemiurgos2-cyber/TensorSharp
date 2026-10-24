// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The published 128-query/16-KV/64-channel attention geometry previously
/// exposed F32 operand narrowing in upstream CUDA flash attention. Spark now
/// requires successful owned F32 execution for ordinary and packed attention.
/// Q8 uses identical native schedules on one GPU and TP. Separate represented-
/// weight F32 controls retain the scalar oracle and ordinary/packed comparison.
/// </summary>
[Collection("Muse Spark model")]
public sealed class MuseSparkTensorParallelFlashAttentionTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-tp-flash-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly ITestOutputHelper _output;

    public MuseSparkTensorParallelFlashAttentionTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        _environment.Set("TS_KV_INITIAL_TOKENS", "2");
        _environment.Set("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
        _environment.Set("TS_MUSE_SPARK_STREAM_EXPERTS", "0");
        _environment.Set("TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES", "1");
        _environment.Set("TS_MUSE_SPARK_FUSE_QKVG", "0");
        _environment.Set("TS_MUSE_SPARK_LAYER_SPLIT", null);
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        Directory.Delete(_directory, true);
    }

    [MuseSparkTensorParallelGpuFact(5)]
    public void FiveRanksPublishedHeadGeometryPackedF32MatchesOrdinaryAndScalar()
        => CheckPackedF32(5);

    [MuseSparkTensorParallelGpuFact(9)]
    public void NineRanksPublishedHeadGeometryPackedF32MatchesOrdinaryAndScalar()
        => CheckPackedF32(9);

    private void CheckPackedF32(int degree)
    {
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 128, kvHeads: 16,
            experts: 36, used: 2, headDim: 64, latent: 32, ffn: 32);
        string path = MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture, Path.Combine(_directory, "head64.gguf"));
        string f32Path = fixture.Write(Path.Combine(_directory, "head64-f32.gguf"));
        int[][] prompts =
        [
            [7, 21, 42, 11, 9, 86, 102, 68, 43],
            [81, 53, 29, 40, 16, 12, 63, 5, 8],
            [17, 2, 199, 7, 106, 43],
        ];
        (int[], int[])[] schedule =
        [
            ([0, 1], [3, 5]), ([1, 0], [2, 3]),
            ([2, 0, 1], [4, 2, 1]), ([0, 2, 1], [1, 2, 1]),
        ];

        // Exact native schedules distinguish TP from ordinary/packed differences
        // in activation quantization and CUDA attention kernel selection.
        var nativePlain = MuseSparkGpuSchedules.Ordinary(path, 1, prompts).ToDictionary(row => (row.Sequence, row.Position));
        var parallelPlain = MuseSparkGpuSchedules.Ordinary(path, degree, prompts).ToDictionary(row => (row.Sequence, row.Position));
        var nativePacked = MuseSparkGpuSchedules.Packed(path, 1, prompts, 4, schedule, requireOwnedAttention: true);
        var parallelPacked = MuseSparkGpuSchedules.Packed(path, degree, prompts, 4, schedule, requireOwnedAttention: true);
        var floatPlain = MuseSparkGpuSchedules.Ordinary(f32Path, degree, prompts).ToDictionary(row => (row.Sequence, row.Position));
        var floatReturned = new List<MuseSparkGpuRow>();
        var floatPacked = MuseSparkGpuSchedules.Packed(f32Path, degree, prompts, 4, schedule, requireOwnedAttention: true,
            captureBatch: (_, batch) => floatReturned.AddRange(MuseSparkGpuSchedules.ReturnedRows(batch)), requireExactHeadRows: false);
        Assert.Equal(24, nativePacked.Length);
        Assert.Equal(nativePacked.Select(row => (row.Sequence, row.Position)), parallelPacked.Select(row => (row.Sequence, row.Position)));
        Assert.Equal(nativePacked.Select(row => (row.Sequence, row.Position)), floatPacked.Select(row => (row.Sequence, row.Position)));
        var checks = new MuseSparkGpuChecks(_output);
        for (int row = 0; row < nativePacked.Length; row++)
        {
            var position = nativePacked[row];
            var key = (position.Sequence, position.Position);
            string label = $"ranks={degree}, seq={key.Sequence}, row={key.Position}";
            float[] scalar = fixture.Reference(prompts[key.Sequence][..(key.Position + 1)]);
            checks.Compare(label + " Q8 ordinary TP/native", nativePlain[key].Values, parallelPlain[key].Values, 1e-4, 1e-4);
            checks.Compare(label + " Q8 packed TP/native", position.Values, parallelPacked[row].Values, 1e-4, 1e-4);
            // These original bounds are recorded, not enlarged or described as
            // exact gates for kernels which quantize their input activations.
            checks.Compare(label + " Q8 packed/scalar diagnostic", scalar, parallelPacked[row].Values, 0.006, 0.002, required: false);
            checks.Compare(label + " Q8 ordinary/scalar diagnostic", scalar, parallelPlain[key].Values, 0.006, 0.002, required: false);
            checks.Compare(label + " Q8 packed/ordinary diagnostic", parallelPlain[key].Values, parallelPacked[row].Values, 0.006, 0.002, required: false);
            checks.Compare(label + " F32 ordinary/scalar", scalar, floatPlain[key].Values, 3e-5, 3e-5);
            checks.Compare(label + " F32 packed/scalar", scalar, floatPacked[row].Values, 3e-5, 3e-5);
            checks.Compare(label + " F32 packed/ordinary", floatPlain[key].Values, floatPacked[row].Values, 3e-5, 3e-5);
        }
        Assert.Equal(schedule.Sum(batch => batch.Item1.Length), floatReturned.Count);
        foreach (var row in floatReturned)
            checks.Compare($"ranks={degree}, returned seq={row.Sequence}, row={row.Position} F32/scalar",
                fixture.Reference(prompts[row.Sequence][..(row.Position + 1)]), row.Values, 3e-5, 3e-5);
        checks.Complete($"head64-ranks-{degree}");
    }
}
