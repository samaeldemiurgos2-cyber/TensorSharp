// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkTensorParallelNumericalControlTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-tp-control-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly ITestOutputHelper _output;

    public MuseSparkTensorParallelNumericalControlTests(ITestOutputHelper output)
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

    [MuseSparkTensorParallelGpuFact(2)]
    public void SameBackendQuantizedAndDequantizedControlsIsolateTwoRankError()
    {
        // Exact geometry, inputs and first Forward([5 tokens]) from the original
        // two-rank failure. Q8 serialization also supplies exact represented
        // weights to the independent scalar oracle and F32 checkpoint.
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16,
            experts: 36, used: 2, headDim: 16, latent: 32, ffn: 32);
        string q8Path = MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture, Path.Combine(_directory, "q8.gguf"));
        string f32Path = fixture.Write(Path.Combine(_directory, "dequantized-f32.gguf"));
        int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
        int[] lengths = [5, 6, 7, 8];
        float[][] scalar = lengths.Select(length => fixture.Reference(tokens[..length])).ToArray();

        float[][] Run(string path, int degree)
        {
            using var model = Assert.IsType<MuseSparkModel>(ModelBase.Create(path, BackendType.GgmlCuda, tpDegree: degree));
            var rows = new List<float[]> { model.Forward(tokens[..5]).ToArray() };
            for (int index = 5; index < tokens.Length; index++)
                rows.Add(model.Forward([tokens[index]]).ToArray());
            return rows.ToArray();
        }

        // Complete every control before evaluating bounds. The single-GPU Q8
        // oracle is actual CUDA inference, not dequantization-only arithmetic.
        float[][] q8Single = Run(q8Path, 1);
        float[][] q8Parallel = Run(q8Path, 2);
        float[][] f32Single = Run(f32Path, 1);
        float[][] f32Parallel = Run(f32Path, 2);
        var comparisons = new List<Comparison>();
        for (int row = 0; row < lengths.Length; row++)
        {
            comparisons.Add(Compare("q8-single/scalar", lengths[row], scalar[row], q8Single[row], 0.006, 0.002));
            comparisons.Add(Compare("q8-tp/scalar", lengths[row], scalar[row], q8Parallel[row], 0.006, 0.002));
            comparisons.Add(Compare("q8-tp/q8-single", lengths[row], q8Single[row], q8Parallel[row], 1e-4, 1e-4));
            comparisons.Add(Compare("f32-single/scalar", lengths[row], scalar[row], f32Single[row], 3e-5, 3e-5));
            comparisons.Add(Compare("f32-tp/scalar", lengths[row], scalar[row], f32Parallel[row], 3e-5, 3e-5));
            comparisons.Add(Compare("f32-tp/f32-single", lengths[row], f32Single[row], f32Parallel[row], 3e-5, 3e-5));
        }
        string output = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_TP_DIAGNOSTIC_OUTPUT");
        if (!string.IsNullOrWhiteSpace(output))
        {
            // Raw diagnostic output is explicitly requested by the test runner,
            // is never a fixture, and cannot replace an earlier failed record.
            using var stream = new FileStream(Path.GetFullPath(output), FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            JsonSerializer.Serialize(stream, new
            {
                status = "numerical_control_capture",
                tokens, lengths,
                q8_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(q8Path))),
                dequantized_f32_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(f32Path))),
                scalar, q8Single, q8Parallel, f32Single, f32Parallel, comparisons,
            }, new JsonSerializerOptions
            {
                WriteIndented = true,
                NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            });
        }
        // Q8 scalar comparisons deliberately retain their original verdicts in
        // the evidence; they cannot be regraded by a same-backend comparison.
        // These additional strict gates test whether TP introduces error beyond
        // the backend baseline and whether the unquantized graph matches math.
        var required = comparisons.Where(value =>
            value.Label == "q8-tp/q8-single" || value.Label.StartsWith("f32-", StringComparison.Ordinal)).ToArray();
        Assert.True(required.All(value => value.Passed),
            string.Join(Environment.NewLine, required.Where(value => !value.Passed)));
    }

    private Comparison Compare(string label, int length, float[] expected, float[] actual,
        double maxBound, double rmsBound)
    {
        Assert.Equal(expected.Length, actual.Length);
        bool finite = expected.All(float.IsFinite) && actual.All(float.IsFinite);
        double maximum = expected.Zip(actual, (a, b) => Math.Abs((double)a - b)).Max();
        double rms = Math.Sqrt(expected.Zip(actual, (a, b) => ((double)a - b) * ((double)a - b)).Average());
        var result = new Comparison(label, length, finite, maximum, rms, maxBound, rmsBound,
            finite && maximum <= maxBound && rms <= rmsBound);
        _output.WriteLine(result.ToString());
        return result;
    }

    private sealed record Comparison(string Label, int PrefixLength, bool Finite,
        double MaximumError, double RmsError, double MaximumBound, double RmsBound, bool Passed);
}
