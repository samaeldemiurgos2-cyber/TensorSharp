// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkQKPrepTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuActualGeometryAndPackedStridesMatchIndependentAndUnfusedQK()
        => GraphCheck(GgmlBackendType.Cpu, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaActualGeometryAndPackedStridesMatchIndependentAndUnfusedQK()
        => GraphCheck(GgmlBackendType.Cuda, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaAlternatingRanksAndScratchReusePreserveQK()
        => GraphCheck(GgmlBackendType.Cuda, 2);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuFusedQKPreservesChunkPrefixSpeculativeVisionAndPackedLogits()
        => ModelCheck(BackendType.GgmlCpu, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaFusedQKPreservesChunkPrefixSpeculativeVisionAndPackedLogits()
        => ModelCheck(BackendType.GgmlCuda, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaTwoRankFusedQKPreservesModelLogits()
        => ModelCheck(BackendType.GgmlCuda, 2);

    [Fact]
    public void UnsupportedBackendRetainsOriginalPreparation() => ModelCheck(BackendType.Cpu, 1);

    [Fact]
    public void InvalidQKArgumentsLeaveOutputsUntouched()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var packed = FloatTensor(allocator, new float[2 * 48], 2, 48);
        using var q = FloatTensor(allocator, Enumerable.Repeat(71f, 32).ToArray(), 2, 16);
        using var k = FloatTensor(allocator, Enumerable.Repeat(71f, 16).ToArray(), 2, 8);
        using var norm = FloatTensor(allocator, new float[8], 8);
        using var wrong = FloatTensor(allocator, new float[7], 7);
        void Call(int[] positions, int rd = 8, float eps = 1e-5f, Tensor qn = null)
            => GgmlBasicOps.MuseSparkPrepareQK(q, k, packed, qn ?? norm, norm, positions, 2, 1, 8,
                eps, true, rd, 500000, 1);
        Assert.Throws<ArgumentException>(() => Call(null));
        Assert.Throws<ArgumentException>(() => Call([0]));
        Assert.Throws<ArgumentException>(() => Call([0, 1], 7));
        Assert.Throws<ArgumentException>(() => Call([0, 1], qn: wrong));
        Assert.Throws<ArgumentOutOfRangeException>(() => Call([0, 1], eps: float.NaN));
        Assert.All(q.GetElementsAsFloat(32), value => Assert.Equal(71f, value));
        Assert.All(k.GetElementsAsFloat(16), value => Assert.Equal(71f, value));
    }

    internal static void GraphCheck(GgmlBackendType backend, int devices, int qh = 128, int kh = 16)
    {
        var context = new GgmlContext(Enumerable.Range(0, devices).ToArray(), backend, enableCollectives: false);
        var allocator = new GgmlAllocator(context, 0);
        int oldRank = GgmlBasicOps.GetActiveRank(); bool oldPin = GgmlBasicOps.PinRank(0);
        const int hd = 64;
        int qw = qh * hd, kw = kh * hd, width = 2 * (qw + kw);
        int iteration = 0;
        try
        {
            foreach (int rows in new[] { 1, 2, 9, 128, 256, 1, 256 })
            foreach (int padding in new[] { 0, 7 })
            foreach (int ropeDim in new[] { 0, 32, 64 })
            {
                int rank = iteration++ % devices;
                GgmlBasicOps.SetActiveRank(rank);
                int stride = width + padding;
                float[] source = Enumerable.Range(0, rows * stride).Select(i => MathF.Sin(i * .021f + iteration) * .7f).ToArray();
                float[] qs = Enumerable.Range(0, hd).Select(i => (i % 11 - 5) * .13f + iteration * .001f).ToArray();
                float[] ks = Enumerable.Range(0, hd).Select(i => (i % 7 - 3) * .21f - iteration * .002f).ToArray();
                int[] positions = Enumerable.Range(0, rows).Select(i => i % 3 == 0 ? 65537 + iteration : i % 3 == 1 ? 2048 : i * 7).ToArray();
                using var backing = FloatTensor(allocator, source, rows, stride);
                using var packed = backing.Narrow(1, 0, width);
                using var qWeight = FloatTensor(allocator, qs, hd);
                using var kWeight = FloatTensor(allocator, ks, hd);
                using var query = new Tensor(allocator, DType.Float32, rows, qw);
                using var key = new Tensor(allocator, DType.Float32, rows, kw);
                GgmlBasicOps.MuseSparkPrepareQK(query, key, packed, qWeight, kWeight,
                    ropeDim == 0 ? null : positions, qh, kh, hd, 1e-5f, ropeDim != 0, ropeDim, 500000, .75f);
                Assert.Equal(rank, GgmlBasicOps.GetActiveRank());
                float[] qExpected = Unfused(packed, 0, qh, qWeight);
                float[] kExpected = Unfused(packed, qw, kh, kWeight);
                // Another call replans the shared gallocr before the original
                // outputs are read. A device-only temporary cannot escape.
                if (rows is 1 or 256 && padding == 7 && ropeDim == 64)
                {
                    using var q2 = new Tensor(allocator, DType.Float32, rows, qw);
                    using var k2 = new Tensor(allocator, DType.Float32, rows, kw);
                    qWeight.SetElementsAsFloat(qs.Select(x => -x).ToArray());
                    GgmlBasicOps.MuseSparkPrepareQK(q2, k2, packed, qWeight, kWeight,
                        positions.Reverse().ToArray(), qh, kh, hd, 1e-5f, true, 32, 7777, 1.25f);
                }
                float[] qActual = query.GetElementsAsFloat(rows * qw), kActual = key.GetElementsAsFloat(rows * kw);
                Close(qExpected, qActual, 2e-5f);
                Close(kExpected, kActual, 2e-5f);
                // Independent scalar reference (not ggml normalization/RoPE).
                Close(Scalar(source, stride, rows, qh, hd, 0, qs, positions, ropeDim), qActual, ropeDim == 0 ? 4e-6f : .008f);
                Close(Scalar(source, stride, rows, kh, hd, qw, ks, positions, ropeDim), kActual, ropeDim == 0 ? 4e-6f : .008f);
                Assert.Equal(source, backing.GetElementsAsFloat(source.Length));

                float[] Unfused(Tensor all, int offset, int heads, Tensor weight)
                {
                    using var slice = all.Narrow(1, offset, heads * hd);
                    using var contiguous = Ops.NewContiguous(slice);
                    using var perHead = contiguous.View(rows * heads, hd);
                    using var normalized = Ops.RMSNorm(null, perHead, weight, null, 1e-5f);
                    if (ropeDim == 0) return normalized.GetElementsAsFloat(rows * heads * hd);
                    using var expanded = new Tensor(allocator, DType.Int32, rows * heads);
                    expanded.SetElementsAsInt(positions.SelectMany(pos => Enumerable.Repeat(pos, heads)).ToArray());
                    using var view = normalized.View(1, rows, heads, hd);
                    using var rotated = Ops.RoPEEx(null, view, expanded, ropeDim, 0, 0, 500000, .75f, 0, 1, 0, 0);
                    return rotated.GetElementsAsFloat(rows * heads * hd);
                }
            }
        }
        finally { GgmlBasicOps.RestoreRankPin(oldPin, oldRank); }
    }

    private static float[] Scalar(float[] packed, int stride, int rows, int heads, int hd, int offset,
        float[] weights, int[] positions, int ropeDim)
    {
        var result = new float[rows * heads * hd];
        for (int r = 0; r < rows; r++)
        for (int h = 0; h < heads; h++)
        {
            int start = r * stride + offset + h * hd, dest = (r * heads + h) * hd;
            double sum = 0;
            for (int c = 0; c < hd; c++) sum += (double)packed[start + c] * packed[start + c];
            float inverse = (float)(1 / Math.Sqrt(sum / hd + 1e-5f));
            for (int c = 0; c < hd; c++) result[dest + c] = packed[start + c] * inverse * weights[c];
            for (int c = 0; c < ropeDim; c += 2)
            {
                // Independent F64 frequency calculation; the native recurrence
                // is F32, so large absolute positions have a separate tolerance.
                double angle = positions[r] * .75 * Math.Pow(500000, -(double)c / ropeDim);
                float a = result[dest + c], b = result[dest + c + 1];
                result[dest + c] = (float)(a * Math.Cos(angle) - b * Math.Sin(angle));
                result[dest + c + 1] = (float)(a * Math.Sin(angle) + b * Math.Cos(angle));
            }
        }
        return result;
    }

    private static void ModelCheck(BackendType backend, int split)
    {
        string directory = Path.Combine(Path.GetTempPath(), "spark-qk-prep-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string[] keys = ["TS_MUSE_SPARK_FUSE_QK_PREP", "TS_MUSE_SPARK_FUSE_QKVG", "TS_WEIGHT_FUSION_COPIES",
            "MAX_CONTEXT", "TS_KV_INITIAL_TOKENS", "TS_MUSE_SPARK_PREFILL_CHUNK"];
        string[] old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        var oldDtype = KvCacheDtypeConfig.Current; bool oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        var fixture = new MuseSparkSyntheticModelBuilder();
        try
        {
            Environment.SetEnvironmentVariable(keys[2], "1");
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "64");
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "2");
            Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK", "2");
            MoeCpuOffloadConfig.Reset();
            string path = fixture.Write(Path.Combine(directory, "qk.gguf"), true);
            int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
            foreach (var dtype in new[] { KvCacheDtype.F32, KvCacheDtype.F16 })
            {
                var baseline = new List<float[]>();
                foreach (int mode in new[] { 0, 1, 2 })
                {
                    Environment.SetEnvironmentVariable(keys[0], mode == 0 ? null : "1"); // Default off.
                    Environment.SetEnvironmentVariable(keys[1], mode == 2 ? "0" : "1"); // Unsupported packing fallback.
                    KvCacheDtypeConfig.Set(dtype);
                    using var model = new MuseSparkModel(path, backend, layerSplitDegree: split);
                    var actual = new List<float[]>();
                    void Check(float[] expected, float[] observed)
                    { Close(expected, observed, 3e-3f); actual.Add(observed.ToArray()); }
                    Check(fixture.Reference(tokens[..5], true), model.Forward(tokens[..5]));
                    for (int end = 6; end <= tokens.Length; end++)
                        Check(fixture.Reference(tokens[..end], true), model.Forward([tokens[end - 1]]));
                    var snapshot = new byte[checked((int)model.ComputeKVBlockByteSize(4))];
                    Assert.True(model.TryExtractKVBlock(0, 4, snapshot));
                    model.ResetKVCache(); Assert.True(model.TryInjectKVBlock(0, 4, snapshot));
                    Check(fixture.Reference(tokens, true), model.Forward(tokens[4..]));
                    model.ResetKVCache();
                    float[] logits = new float[tokens.Length * model.Config.VocabSize], hidden = new float[tokens.Length * model.Config.HiddenSize];
                    model.SpecForward(tokens, hidden, logits, true);
                    Assert.All(hidden, x => Assert.True(float.IsFinite(x)));
                    for (int end = 1; end <= tokens.Length; end++)
                        Check(fixture.Reference(tokens[..end], true), logits.AsSpan((end - 1) * model.Config.VocabSize, model.Config.VocabSize).ToArray());
                    model.SpecRewindCache(2);
                    Check(fixture.Reference(tokens[..3], true), model.Forward([tokens[2]]));
                    model.ResetKVCache();
                    float[] image = Enumerable.Range(0, 4 * model.Config.HiddenSize).Select(i => MathF.Sin(i * .37f)).ToArray();
                    model.SetVisionEmbeddings(FloatTensor(new CpuAllocator(BlasEnum.DotNet), image, 4, model.Config.HiddenSize), 1);
                    var replacements = Enumerable.Range(0, 4).ToDictionary(i => i + 1, i => image.AsSpan(i * model.Config.HiddenSize, model.Config.HiddenSize).ToArray());
                    Check(fixture.Reference(tokens, true, replacements), model.Forward(tokens));
                    if (dtype == KvCacheDtype.F32) Packed(model, fixture, Check);
                    Assert.Equal(mode == 1 && backend != BackendType.Cpu, model.FusedQKPrepCallCount > 0);
                    if (mode == 0) baseline = actual;
                    else { Assert.Equal(baseline.Count, actual.Count); for (int i = 0; i < actual.Count; i++) Close(baseline[i], actual[i], mode == 2 ? 3e-3f : 3e-5f); }
                }
            }
        }
        finally
        {
            for (int i = 0; i < keys.Length; i++) Environment.SetEnvironmentVariable(keys[i], old[i]);
            KvCacheDtypeConfig.RestoreForTests(oldDtype, oldExplicit); MoeCpuOffloadConfig.Reset();
            Directory.Delete(directory, true);
        }
    }

    private static void Packed(MuseSparkModel model, MuseSparkSyntheticModelBuilder fixture, Action<float[], float[]> check)
    {
        int[][] prompts = [[7, 21, 42, 11, 9, 86, 102], [81, 53, 29, 40, 16, 12, 63]];
        const int bs = 2;
        var pool = new BlockPool(16, bs, model.ComputeKVBlockByteSize(bs));
        var seqs = prompts.Select((p, i) => new SequenceState($"qk{i}", p, 3, bs, SamplingConfig.Default)).ToArray();
        foreach (var seq in seqs) foreach (var block in pool.AllocateNew(4)) seq.BlockTable.AppendBlock(block);
        foreach (int[] counts in new int[][] { [5, 3], [1, 2] })
        {
            int[] starts = [0, counts[0], counts.Sum()];
            int[][] tables = seqs.Select(s => s.BlockTable.Blocks.Select(b => b.Id).ToArray()).ToArray();
            var positions = new List<int>(); var slots = new List<int>();
            for (int s = 0; s < seqs.Length; s++)
                for (int r = seqs[s].NumComputedTokens; r < seqs[s].NumComputedTokens + counts[s]; r++)
                { positions.Add(r); slots.Add(tables[s][r / bs] * bs + r % bs); }
            var capture = new float[counts.Sum() * model.Config.VocabSize];
            model.ForwardBatch(new BatchedForwardContext
            {
                Sequences = seqs.ToList(), NumScheduledTokens = counts.ToList(), QueryStartLoc = starts.ToList(),
                Positions = positions, SlotMapping = slots, BlockTables = tables, MaxQueryLen = counts.Max(),
                MaxSeqLen = positions.Max() + 1, CaptureLogitsAll = capture,
            });
            for (int s = 0; s < seqs.Length; s++)
            {
                for (int r = 0; r < counts[s]; r++)
                    check(fixture.Reference(prompts[s][..(seqs[s].NumComputedTokens + r + 1)], true),
                        capture.AsSpan((starts[s] + r) * model.Config.VocabSize, model.Config.VocabSize).ToArray());
                seqs[s].AdvanceComputedTokens(counts[s]);
            }
        }
    }

    private static Tensor FloatTensor(IAllocator allocator, float[] data, params long[] sizes)
    { var tensor = new Tensor(allocator, DType.Float32, sizes); tensor.SetElementsAsFloat(data); return tensor; }

    private static void Close(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.All(actual, x => Assert.True(float.IsFinite(x)));
        float error = expected.Zip(actual, (x, y) => MathF.Abs(x - y)).Max();
        Assert.True(error <= tolerance, $"Maximum Q/K or logit error {error} exceeds {tolerance}.");
    }
}
