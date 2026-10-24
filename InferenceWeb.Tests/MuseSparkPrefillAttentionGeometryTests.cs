// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorSharp;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Direct F32 attention controls at actual 32/256-query dispatch sizes. This is
/// an attention test, not a quantized decoder or full-model quality test.
/// </summary>
[Collection("Muse Spark model")]
public sealed class MuseSparkPrefillAttentionGeometryTests(ITestOutputHelper output)
{
    private const int HeadDim = 64, Group = 8, BlockSize = 16, FirstPosition = 17;
    // Existing F32 GgmlAttentionTests and headDim64 PagedAttentionSessionBufferTests
    // use this absolute bound. Retain it for all three comparisons; no bound is
    // chosen from the observed CUDA result. RMS is reported separately.
    private const double MaximumBound = 2e-3;

    [MuseSparkTensorParallelGpuFact(1)]
    public void NativeCudaThirtyTwoQueryPrefillMatchesPagedAndDoubleReference() => Check(32);

    [MuseSparkTensorParallelGpuFact(1)]
    public void NativeCudaTwoHundredFiftySixQueryPrefillMatchesPagedAndDoubleReference() => Check(256);

    private void Check(int rows)
    {
        var context = new GgmlContext([0], GgmlBackendType.Cuda);
        var allocator = new GgmlAllocator(context, 0);
        var records = new List<CaseResult>();
        var failures = new List<string>();
        try
        {
            // One/two local KV groups occur in the nine-rank partition of the
            // published 16 KV groups. The full shape is 128 query / 16 KV heads.
            foreach (int kvHeads in new[] { 1, 2, 16 })
            {
                int heads = kvHeads * Group, length = FirstPosition + rows;
                float scale = 1f / MathF.Sqrt(HeadDim);
                var random = new Random(6301 + rows * 19 + kvHeads);
                float[] q = RandomValues(random, rows * heads * HeadDim);
                float[] k = RandomValues(random, length * kvHeads * HeadDim);
                float[] v = RandomValues(random, k.Length);
                int logicalBlocks = (length + BlockSize - 1) / BlockSize;
                // Reversed odd physical blocks, with unreferenced even blocks
                // and unused tail slots poisoned by large finite values.
                int[] table = Enumerable.Range(0, logicalBlocks)
                    .Select(block => 2 * (logicalBlocks - block) - 1).ToArray();
                int physicalBlocks = logicalBlocks * 2;
                float[] pagedK = Enumerable.Repeat(37f, physicalBlocks * BlockSize * kvHeads * HeadDim).ToArray();
                float[] pagedV = Enumerable.Repeat(-41f, pagedK.Length).ToArray();
                int kvWidth = kvHeads * HeadDim;
                for (int token = 0; token < length; token++)
                {
                    int physical = (table[token / BlockSize] * BlockSize + token % BlockSize) * kvWidth;
                    Array.Copy(k, token * kvWidth, pagedK, physical, kvWidth);
                    Array.Copy(v, token * kvWidth, pagedV, physical, kvWidth);
                }
                int[] positions = Enumerable.Range(FirstPosition, rows).ToArray();
                using var qTensor = new Tensor(allocator, DType.Float32, heads, rows, HeadDim);
                using var kTensor = new Tensor(allocator, DType.Float32, kvHeads, length, HeadDim);
                using var vTensor = new Tensor(allocator, DType.Float32, kvHeads, length, HeadDim);
                qTensor.SetElementsAsFloat(ToHeadFirst(q, rows, heads));
                kTensor.SetElementsAsFloat(ToHeadFirst(k, length, kvHeads));
                vTensor.SetElementsAsFloat(ToHeadFirst(v, length, kvHeads));

                // Window31 deliberately crosses the left cutoff and physical
                // page boundaries within this bounded fixture. It is not the
                // checkpoint's 4096-token window or a long-context test.
                foreach (int window in new[] { 0, 31 })
                {
                    using var ordinaryTensor = new Tensor(allocator, DType.Float32, rows, heads * HeadDim);
                    GgmlBasicOps.FusedPrefillAttention(qTensor, kTensor, vTensor, ordinaryTensor,
                        heads, kvHeads, HeadDim, rows, length, FirstPosition, window, scale);
                    float[] ordinary = ordinaryTensor.GetElementsAsFloat(q.Length);
                    float[] paged = new float[q.Length];
                    long fallbacksBefore = GgmlBasicOps.FlashAttnFallbackCount();
                    GgmlBasicOps.PagedAttentionForward(q, pagedK, pagedV, paged,
                        [0, rows], [length], positions, table, [0],
                        1, rows, heads, kvHeads, HeadDim, BlockSize, scale, window);
                    long fallbacksAfter = GgmlBasicOps.FlashAttnFallbackCount();

                    // Full-array native parity, plus double-precision arithmetic
                    // at fixed causal/window/page boundaries and interior rows.
                    // All 64 channels are checked at each sampled query/head.
                    int[] sampleRows = new[] { 0, 1, 13, 14, 15, 16, 30, 31, 32, rows / 2, rows - 1 }
                        .Where(row => row < rows).Distinct().Order().ToArray();
                    int[] sampleHeads = new[] { 0, Group - 1, Group, heads / 2, heads - Group, heads - 1 }
                        .Where(head => head >= 0 && head < heads).Distinct().Order().ToArray();
                    var indices = new List<int>();
                    var reference = new List<double>();
                    foreach (int row in sampleRows)
                    foreach (int head in sampleHeads)
                    {
                        double[] expected = ReferenceHead(q, k, v, row, head, heads, kvHeads, window, scale);
                        for (int channel = 0; channel < HeadDim; channel++)
                        {
                            indices.Add((row * heads + head) * HeadDim + channel);
                            reference.Add(expected[channel]);
                        }
                    }
                    ErrorStats parity = Compare("paged/ordinary", ordinary.Select(x => (double)x).ToArray(), paged);
                    ErrorStats ordinaryScalar = Compare("ordinary/double-sampled", reference.ToArray(),
                        indices.Select(index => ordinary[index]).ToArray());
                    ErrorStats pagedScalar = Compare("paged/double-sampled", reference.ToArray(),
                        indices.Select(index => paged[index]).ToArray());
                    string label = $"rows={rows}, qHeads={heads}, kvHeads={kvHeads}, window={window}";
                    output.WriteLine($"{label}: fullElements={q.Length}, scalarElements={indices.Count}, " +
                        $"scalarRows=[{string.Join(',', sampleRows)}], scalarHeads=[{string.Join(',', sampleHeads)}], " +
                        $"flashFallbacks={fallbacksBefore}->{fallbacksAfter}");
                    foreach (var stats in new[] { parity, ordinaryScalar, pagedScalar })
                    {
                        output.WriteLine($"{label}: {stats}");
                        if (!stats.Passed) failures.Add($"{label}: {stats}");
                    }
                    if (fallbacksAfter != fallbacksBefore)
                        failures.Add($"{label}: paged attention unexpectedly used explicit fallback.");
                    records.Add(new CaseResult(rows, heads, kvHeads, length, window, table,
                        sampleRows, sampleHeads, indices.Count, q.Length, fallbacksBefore, fallbacksAfter,
                        Hash(q), Hash(k), Hash(v), Hash(ordinary), Hash(paged),
                        parity, ordinaryScalar, pagedScalar));
                }
            }
            string directory = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_TP_CASE_OUTPUT_DIRECTORY");
            if (!string.IsNullOrWhiteSpace(directory))
            {
                Directory.CreateDirectory(directory);
                using var stream = new FileStream(Path.Combine(directory, $"attention-geometry-{rows}.json"),
                    FileMode.CreateNew, FileAccess.Write, FileShare.Read);
                JsonSerializer.Serialize(stream, records, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
                });
            }
            Assert.Equal(6, records.Count);
            Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
            context.ReleasePoolForDisposal();
        }
    }

    private static float[] RandomValues(Random random, int count)
    {
        var values = new float[count];
        for (int index = 0; index < count; index++) values[index] = (float)(random.NextDouble() * 2 - 1);
        return values;
    }

    private static float[] ToHeadFirst(float[] input, int rows, int heads)
    {
        var result = new float[input.Length];
        for (int row = 0; row < rows; row++)
        for (int head = 0; head < heads; head++)
            Array.Copy(input, (row * heads + head) * HeadDim, result, (head * rows + row) * HeadDim, HeadDim);
        return result;
    }

    private static double[] ReferenceHead(float[] q, float[] k, float[] v, int row, int head,
        int heads, int kvHeads, int window, float scale)
    {
        int position = FirstPosition + row, first = window == 0 ? 0 : Math.Max(0, position - window + 1);
        var scores = new double[position - first + 1];
        int queryOffset = (row * heads + head) * HeadDim, kvHead = head / Group;
        double maximum = double.NegativeInfinity;
        for (int token = first; token <= position; token++)
        {
            double dot = 0;
            int keyOffset = (token * kvHeads + kvHead) * HeadDim;
            for (int channel = 0; channel < HeadDim; channel++)
                dot += (double)q[queryOffset + channel] * k[keyOffset + channel];
            scores[token - first] = dot * scale;
            maximum = Math.Max(maximum, scores[token - first]);
        }
        double denominator = 0;
        for (int token = 0; token < scores.Length; token++)
            denominator += scores[token] = Math.Exp(scores[token] - maximum);
        var output = new double[HeadDim];
        for (int token = first; token <= position; token++)
        {
            int valueOffset = (token * kvHeads + kvHead) * HeadDim;
            double probability = scores[token - first] / denominator;
            for (int channel = 0; channel < HeadDim; channel++)
                output[channel] += probability * v[valueOffset + channel];
        }
        return output;
    }

    private static ErrorStats Compare(string label, double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        bool finite = expected.All(double.IsFinite) && actual.All(float.IsFinite);
        double maximum = 0, sum = 0;
        int maximumIndex = 0;
        for (int index = 0; index < expected.Length; index++)
        {
            double error = Math.Abs(expected[index] - actual[index]);
            sum += error * error;
            if (error > maximum) { maximum = error; maximumIndex = index; }
        }
        return new ErrorStats(label, finite, maximum, Math.Sqrt(sum / expected.Length),
            maximumIndex, expected[maximumIndex], actual[maximumIndex], MaximumBound,
            finite && maximum <= MaximumBound);
    }

    private static string Hash(float[] values)
        => Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan())));

    private sealed record ErrorStats(string Label, bool Finite, double Maximum, double Rms,
        int MaximumIndex, double ExpectedAtMaximum, float ActualAtMaximum, double MaximumBound, bool Passed);

    private sealed record CaseResult(int QueryRows, int QueryHeads, int KvHeads, int KvLength,
        int SlidingWindow, int[] PhysicalBlocks, int[] ScalarRows, int[] ScalarHeads,
        int ScalarElements, int FullElements, long FallbacksBefore, long FallbacksAfter,
        string QuerySha256, string KeySha256, string ValueSha256, string OrdinarySha256, string PagedSha256,
        ErrorStats NativeParity, ErrorStats OrdinaryScalar, ErrorStats PagedScalar);
}
