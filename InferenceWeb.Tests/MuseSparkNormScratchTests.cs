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
public sealed class MuseSparkNormScratchTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuNormGateFreshReuseAndOriginalOpsMatch() => GraphCheck(GgmlBackendType.Cpu, 1);
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaNormGateFreshReuseAndOriginalOpsMatch() => GraphCheck(GgmlBackendType.Cuda, 1);
    [MuseSparkLayerSplitFact]
    public void NativeCudaTwoRankNormGateFreshReuseAndOriginalOpsMatch() => GraphCheck(GgmlBackendType.Cuda, 2);
    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuNormScratchPreservesChunkPrefixSpeculativeVisionAndPackedLogits() => ModelCheck(BackendType.GgmlCpu, 1);
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaNormScratchPreservesChunkPrefixSpeculativeVisionAndPackedLogits() => ModelCheck(BackendType.GgmlCuda, 1);
    [MuseSparkLayerSplitFact]
    public void NativeCudaTwoRankNormScratchPreservesModelLogits() => ModelCheck(BackendType.GgmlCuda, 2);
    [Fact]
    public void UnsupportedBackendRetainsOriginalNorm() => ModelCheck(BackendType.Cpu, 1);

    private static void GraphCheck(GgmlBackendType backend, int devices)
    {
        var context = new GgmlContext(Enumerable.Range(0, devices).ToArray(), backend, enableCollectives: false);
        var allocator = new GgmlAllocator(context, 0);
        int oldRank = GgmlBasicOps.GetActiveRank(); bool oldPin = GgmlBasicOps.PinRank(0);
        int iteration = 0;
        try
        {
            foreach (int rows in new[] { 1, 2, 9, 128, 256, 1, 256 })
            foreach (int width in new[] { 64, 4096, 8192 })
            foreach (int padding in new[] { 0, 7 })
            foreach (bool gated in new[] { false, true })
            {
                int rank = iteration++ % devices; GgmlBasicOps.SetActiveRank(rank);
                int stride = width + padding, hd = gated ? 64 : width;
                float[] source = Enumerable.Range(0, rows * stride).Select(i => MathF.Sin(i * .021f + iteration)).ToArray();
                float[] gs = Enumerable.Range(0, source.Length).Select(i => (i % 31 - 15) * .3f).ToArray();
                float[] gamma = Enumerable.Range(0, hd).Select(i => (i % 11 - 5) * .13f + iteration * .001f).ToArray();
                using var all = FloatTensor(allocator, source, rows, stride);
                using var allGate = FloatTensor(allocator, gs, rows, stride);
                using var input = all.Narrow(1, 0, width);
                using var gate = allGate.Narrow(1, 0, width);
                using var weights = FloatTensor(allocator, gamma, hd);
                using var fresh = new Tensor(allocator, DType.Float32, rows, width);
                using var reuse = new Tensor(allocator, DType.Float32, rows, width);
                Assert.False(GgmlBasicOps.MuseSparkNormGate(fresh, input, weights, gated ? gate : null, 1e-5f, false));
                Assert.True(GgmlBasicOps.MuseSparkNormGate(reuse, input, weights, gated ? gate : null, 1e-5f));
                using var contiguous = Ops.NewContiguous(input);
                using var headView = contiguous.View(rows * (width / hd), hd);
                using var norm = Ops.RMSNorm(null, headView, weights, null, 1e-5f);
                using var flat = norm.View(rows, width);
                if (gated) Ops.SigmoidMul(flat, flat, gate);
                // Repack shared scratch before reading the prior result.
                using var later = new Tensor(allocator, DType.Float32, rows, width);
                weights.SetElementsAsFloat(gamma.Select(x => -x).ToArray());
                Assert.True(GgmlBasicOps.MuseSparkNormGate(later, input, weights, gated ? gate : null, 1e-8f));
                var actual = reuse.GetElementsAsFloat(rows * width);
                Assert.Equal(fresh.GetElementsAsFloat(rows * width), actual);
                Close(flat.GetElementsAsFloat(rows * width), actual, 2e-5f);
                Assert.Equal(source, all.GetElementsAsFloat(source.Length));
                Assert.Equal(gs, allGate.GetElementsAsFloat(gs.Length));
                Assert.Equal(rank, GgmlBasicOps.GetActiveRank());
            }
        }
        finally { GgmlBasicOps.RestoreRankPin(oldPin, oldRank); }
    }

    private static void ModelCheck(BackendType backend, int split)
    {
        string directory = Path.Combine(Path.GetTempPath(), "spark-qk-prep-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string[] keys = ["TS_MUSE_SPARK_REUSE_NORM_SCRATCH", "TS_MUSE_SPARK_FUSE_QK_PREP", "TS_WEIGHT_FUSION_COPIES",
            "MAX_CONTEXT", "TS_KV_INITIAL_TOKENS", "TS_MUSE_SPARK_PREFILL_CHUNK", "TS_MUSE_SPARK_FUSE_RESIDUALS"];
        string[] old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        var oldDtype = KvCacheDtypeConfig.Current; bool oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        var fixture = new MuseSparkSyntheticModelBuilder();
        try
        {
            Environment.SetEnvironmentVariable(keys[2], "1");
            Environment.SetEnvironmentVariable(keys[1], "1");
            Environment.SetEnvironmentVariable(keys[6], "1");
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "64");
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "2");
            Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK", "2");
            MoeCpuOffloadConfig.Reset();
            string path = fixture.Write(Path.Combine(directory, "qk.gguf"), true);
            int[] tokens = [7, 21, 42, 11, 9, 86, 102, 68];
            foreach (var dtype in new[] { KvCacheDtype.F32, KvCacheDtype.F16 })
            {
                var baseline = new List<float[]>();
                foreach (int mode in new[] { 0, 1, 2, 3 })
                {
                    Environment.SetEnvironmentVariable(keys[0], mode switch { 0 => null, 1 => "0", 2 => "true", _ => "1" });
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
                    Assert.Equal(mode == 3 && backend != BackendType.Cpu, model.NormScratchReuseCallCount > 0);
                    Assert.Equal(mode == 3 && backend != BackendType.Cpu, model.HeadNormGateCallCount > 0);
                    Assert.Equal(0, model.NormScratchFallbackCallCount);
                    if (mode == 0) baseline = actual;
                    else { Assert.Equal(baseline.Count, actual.Count); for (int i = 0; i < actual.Count; i++) Close(baseline[i], actual[i], 3e-5f); }
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
