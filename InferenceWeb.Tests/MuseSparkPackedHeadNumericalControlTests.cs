// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Retains the original packed head identity/scalar verdicts while isolating
/// the two output-head batch shapes with exactly the same captured inputs.
/// This diagnostic does not replace or relax the ordinary packed regression.
/// </summary>
[Collection("Muse Spark model")]
public sealed partial class MuseSparkPackedHeadNumericalControlTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-spark-packed-head-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly ITestOutputHelper _output;

    public MuseSparkPackedHeadNumericalControlTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        _environment.Set("TS_KV_INITIAL_TOKENS", "2");
        _environment.Set("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
        _environment.Set("TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES", "1");
        _environment.Set("TS_MUSE_SPARK_LAYER_SPLIT", null);
        foreach (string name in new[] { "FUSE_QKVG", "FUSE_QK_PREP", "FUSE_RESIDUALS",
            "REUSE_NORM_SCRATCH", "KV_UPLOAD_2D", "STREAM_EXPERTS" })
            _environment.Set("TS_MUSE_SPARK_" + name, "0");
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
    public void PackedFloatHeadBatchShapesRetainRawSingleAndParallelControls()
    {
        var fixture = new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16,
            experts: 36, used: 2, headDim: 16, latent: 32, ffn: 32);
        // Reproduce the failed test's represented-Q8 -> stored-F32 checkpoint.
        string q8 = MuseSparkTensorParallelGpuTests.WriteQ8Fixture(fixture, Path.Combine(_directory, "represented-q8.gguf"));
        string path = fixture.Write(Path.Combine(_directory, "represented-f32.gguf"));
        int[][] prompts = [[7, 21, 42, 11, 9, 86, 102, 68], [81, 53, 29, 40, 16, 12, 63, 5], [17, 2, 199, 7]];
        (int[], int[])[] schedule = [([0, 1], [3, 5]), ([1, 0], [1, 2]), ([2, 0, 1], [4, 1, 1])];
        var controls = new List<HeadControl>();
        foreach (int degree in new[] { 1, 5 })
        {
            MuseSparkGpuSchedules.Packed(path, degree, prompts, 2, schedule, requireOwnedAttention: false,
                captureBatch: (model, captured) =>
                {
                    const int hidden = MuseSparkSyntheticModelBuilder.Hidden;
                    int rows = captured.Positions.Length;
                    int[] lastIndices = captured.Starts.Skip(1).Select(end => end - 1).ToArray();
                    float[] lastHidden = Gather(captured.Hidden, lastIndices, hidden);
                    float[] allRaw = ProjectHead(model, captured.Hidden, rows);
                    float[] lastRaw = ProjectHead(model, lastHidden, lastIndices.Length);
                    float[] oneRowRaw = Enumerable.Range(0, rows)
                        .SelectMany(row => ProjectHead(model, captured.Hidden.AsSpan(row * hidden, hidden).ToArray(), 1)).ToArray();
                    float[] scalarRaw = ScalarHead(captured.Hidden, fixture.Weights["output.weight"].Data);
                    controls.Add(new HeadControl(degree, captured, allRaw, lastRaw, oneRowRaw, scalarRaw));
                }, requireExactHeadRows: false);
        }

        // Preserve every raw value before evaluating numerical assertions,
        // including the original zero-bound identity and 3e-5 scalar failures.
        string directory = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_TP_CASE_OUTPUT_DIRECTORY");
        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
            using var stream = new FileStream(Path.Combine(directory, "packed-head-control-raw.json"),
                FileMode.CreateNew, FileAccess.Write, FileShare.Read);
            JsonSerializer.Serialize(stream, new
            {
                status = "diagnostic_capture_not_regraded_original_result",
                checkpoint_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))),
                represented_q8_sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(q8))),
                outputWeight = fixture.Weights["output.weight"].Data,
                prompts, controls,
            }, new JsonSerializerOptions { WriteIndented = true, NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals });
        }

        var checks = new MuseSparkGpuChecks(_output);
        foreach (HeadControl control in controls)
        {
            var capture = control.Captured;
            int[] lastIndices = capture.Starts.Skip(1).Select(end => end - 1).ToArray();
            string label = $"degree={control.Degree}, batch={capture.Batch}";
            float[] returned = capture.ReturnedLogits.SelectMany(row => row).ToArray();
            checks.Compare(label + " original all/last identity",
                Gather(capture.AllLogits, lastIndices, MuseSparkSyntheticModelBuilder.Vocab), returned, 0, 0, required: false);
            checks.Compare(label + " replay all/saved all", capture.AllLogits, Scale(control.AllRaw), 3e-5, 3e-5);
            checks.Compare(label + " replay last/saved last", returned, Scale(control.LastRaw), 3e-5, 3e-5);
            checks.Compare(label + " all-head/scalar", control.ScalarRaw, control.AllRaw, 3e-5, 3e-5, required: false);
            checks.Compare(label + " last-head/scalar", Gather(control.ScalarRaw, lastIndices, MuseSparkSyntheticModelBuilder.Vocab),
                control.LastRaw, 3e-5, 3e-5, required: false);
            checks.Compare(label + " one-row-head/scalar", control.ScalarRaw, control.OneRowRaw, 3e-5, 3e-5, required: false);
            for (int s = 0; s < capture.SequenceIndices.Length; s++)
            for (int row = capture.Starts[s]; row < capture.Starts[s + 1]; row++)
            {
                float[] scalar = fixture.Reference(prompts[capture.SequenceIndices[s]][..(capture.Positions[row] + 1)]);
                checks.Compare(label + $" row={row} full decoder/scalar", scalar,
                    capture.AllLogits.AsSpan(row * MuseSparkSyntheticModelBuilder.Vocab, MuseSparkSyntheticModelBuilder.Vocab).ToArray(),
                    3e-5, 3e-5, required: false);
            }
        }
        for (int batch = 0; batch < schedule.Length; batch++)
        {
            var single = controls.Single(value => value.Degree == 1 && value.Captured.Batch == batch);
            var parallel = controls.Single(value => value.Degree == 5 && value.Captured.Batch == batch);
            checks.Compare($"batch={batch} TP/single captured hidden", single.Captured.Hidden, parallel.Captured.Hidden, 3e-5, 3e-5);
            checks.Compare($"batch={batch} TP/single all logits", single.Captured.AllLogits, parallel.Captured.AllLogits, 3e-5, 3e-5);
            checks.Compare($"batch={batch} TP/single returned logits", single.Captured.ReturnedLogits.SelectMany(row => row).ToArray(),
                parallel.Captured.ReturnedLogits.SelectMany(row => row).ToArray(), 3e-5, 3e-5);
        }
        checks.Complete("packed-head-control-verdicts");
    }

    private static float[] ProjectHead(MuseSparkModel model, float[] hidden, int rows)
    {
        // Invoke the exact model-owned head without rebuilding its weights or
        // changing backend precision. It uses rank 0 in both single and TP.
        int priorRank = GgmlBasicOps.GetActiveRank();
        bool priorPin = GgmlBasicOps.PinRank(0);
        try
        {
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.NonPublic;
            using var input = (Tensor)typeof(ModelBase).GetMethod("CreateFloatTensor", flags)!
                .Invoke(model, [hidden, new long[] { rows, MuseSparkSyntheticModelBuilder.Hidden }])!;
            using var output = (Tensor)typeof(MuseSparkModel).GetMethod("LinearForward", flags | BindingFlags.DeclaredOnly)!
                .Invoke(model, [input, "output.weight"])!;
            return output.GetElementsAsFloat(checked((int)output.ElementCount()));
        }
        finally { GgmlBasicOps.RestoreRankPin(priorPin, priorRank); }
    }

    private static float[] ScalarHead(float[] hidden, float[] weight)
    {
        const int width = MuseSparkSyntheticModelBuilder.Hidden, vocab = MuseSparkSyntheticModelBuilder.Vocab;
        var result = new float[hidden.Length / width * vocab];
        for (int row = 0; row < hidden.Length / width; row++)
        for (int column = 0; column < vocab; column++)
        {
            double value = 0;
            for (int inner = 0; inner < width; inner++)
                value += (double)hidden[row * width + inner] * weight[column * width + inner];
            result[row * vocab + column] = (float)value;
        }
        return result;
    }

    private static float[] Scale(float[] input) => input.Select(value =>
        MuseSparkSyntheticModelBuilder.Softcap * MathF.Tanh(value * MuseSparkSyntheticModelBuilder.LogitScale /
            MuseSparkSyntheticModelBuilder.Softcap)).ToArray();

    private static float[] Gather(float[] input, int[] indices, int width)
        => indices.SelectMany(index => input.AsSpan(index * width, width).ToArray()).ToArray();

    private sealed record HeadControl(int Degree, MuseSparkPackedCapture Captured,
        float[] AllRaw, float[] LastRaw, float[] OneRowRaw, float[] ScalarRaw);
}
