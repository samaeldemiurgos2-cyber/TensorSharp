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
public sealed class MuseSparkResidualTests
{
    [Fact]
    public void ManagedBackendKeepsOriginalResidualPathWhenOptedIn() => ModelCheck(BackendType.Cpu, 1);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuResidualsMatchUnfusedAndIndependentDecoder() => ModelCheck(BackendType.GgmlCpu, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaResidualsMatchUnfusedAndIndependentDecoder() => ModelCheck(BackendType.GgmlCuda, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaTwoRankResidualsMatchUnfusedAndIndependentDecoder() => ModelCheck(BackendType.GgmlCuda, 2);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuResidualGraphMatchesScalarOrderForDecodeAndPrefill() => GraphCheck(GgmlBackendType.Cpu, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaResidualGraphMatchesScalarOrderForDecodeAndPrefill() => GraphCheck(GgmlBackendType.Cuda, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaResidualGraphRespectsPinnedRankAndMatchesScalarOrder() => GraphCheck(GgmlBackendType.Cuda, 2);

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpuResidualScratchReuseKeepsChangingInputsAndRetainedOutputs() => ReuseCheck(GgmlBackendType.Cpu);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCudaResidualScratchReuseKeepsChangingInputsAndRetainedOutputs() => ReuseCheck(GgmlBackendType.Cuda);

    private static void ReuseCheck(GgmlBackendType backend)
    {
        var context = new GgmlContext([0], backend, enableCollectives: false);
        var allocator = new GgmlAllocator(context, 0);
        var retained = new List<(Tensor Output, float[] Expected)>();
        int iteration = 0;
        try
        {
            foreach (int rows in new[] { 256, 1, 256 })
            foreach (bool preScale in new[] { true, false })
            {
                int pass = ++iteration;
                const int width = 8192;
                float[] s = Enumerable.Range(0, rows * width).Select(i => MathF.Sin(i * .019f + pass) * .7f).ToArray();
                float[] b = Enumerable.Range(0, rows * width).Select(i => MathF.Cos(i * .023f - pass) * .4f).ToArray();
                float[] a = Enumerable.Range(0, width).Select(i => (i % 7 - 3) * .4f + pass * .01f).ToArray();
                float[] beta = Enumerable.Range(0, width).Select(i => (i % 5 - 2) * .3f - pass * .02f).ToArray();
                float[] p = Enumerable.Range(0, width).Select(i => (i % 9 - 4) * .6f + pass * .03f).ToArray();
                using var stream = FloatTensor(allocator, s, rows, width);
                using var branch = FloatTensor(allocator, b, rows, width);
                using var alpha = FloatTensor(allocator, a, width);
                using var gate = FloatTensor(allocator, beta, width);
                using var scale = FloatTensor(allocator, p, width);
                var output = new Tensor(allocator, DType.Float32, rows, width);
                retained.Add((output, Scalar(s, b, a, beta, preScale ? p : null, 1e-5f)));
                GgmlBasicOps.MuseSparkPostNormResidual(output, stream, branch, alpha, gate, preScale ? scale : null, 1e-5f);
                Assert.Equal(s, stream.GetElementsAsFloat(s.Length));
                Assert.Equal(b, branch.GetElementsAsFloat(b.Length));
            }
            // Check only after later, differently sized graphs have reused the
            // arena: results must already reside in their caller-owned buffers.
            foreach (var (output, expected) in retained)
                Close(expected, output.GetElementsAsFloat(expected.Length), 3e-5f);
        }
        finally { foreach (var item in retained) item.Output.Dispose(); }
    }

    [Fact]
    public void InvalidResidualArgumentsAreRejectedBeforeNativeDispatch()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var result = new Tensor(allocator, DType.Float32, 2, 3);
        using var stream = new Tensor(allocator, DType.Float32, 2, 3);
        using var branch = new Tensor(allocator, DType.Float32, 2, 3);
        using var weight = new Tensor(allocator, DType.Float32, 3);
        using var wrong = new Tensor(allocator, DType.Float32, 2);
        result.SetElementsAsFloat(Enumerable.Repeat(71f, 6).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => GgmlBasicOps.MuseSparkPostNormResidual(result, stream, branch, weight, weight, null, float.NaN));
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.MuseSparkPostNormResidual(result, stream, branch, wrong, weight, null, 1e-5f));
        Assert.Throws<ArgumentNullException>(() => GgmlBasicOps.MuseSparkPostNormResidual(result, stream, branch, null, weight, null, 1e-5f));
        using var transposed = branch.Transpose();
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.MuseSparkPostNormResidual(result, stream, transposed, weight, weight, null, 1e-5f));
        Assert.All(result.GetElementsAsFloat(6), x => Assert.Equal(71f, x));
    }

    private static void GraphCheck(GgmlBackendType backend, int devices)
    {
        var context = new GgmlContext(Enumerable.Range(0, devices).ToArray(), backend, enableCollectives: false);
        // Even on the second rank the model allocates its host staging tensors
        // through allocator zero. The explicit layer pin must win over that ID.
        var allocator = new GgmlAllocator(context, 0);
        int priorRank = GgmlBasicOps.GetActiveRank();
        bool priorPin = GgmlBasicOps.PinRank(devices - 1);
        try
        {
            foreach (int rows in new[] { 1, 3, 128 })
            foreach (int width in new[] { 32, 8192 })
            foreach (bool prescale in new[] { false, true })
            {
                float[] s = Enumerable.Range(0, rows * width).Select(i => MathF.Sin(i * 0.19f) * 0.7f).ToArray();
                float[] x = Enumerable.Range(0, rows * width).Select(i => i / width == 1 ? 1e-9f : MathF.Cos(i * 0.23f) * 0.4f).ToArray();
                float[] a = Enumerable.Range(0, width).Select(i => (i % 7 - 3) * 0.4f).ToArray();
                float[] b = Enumerable.Range(0, width).Select(i => (i % 5 - 2) * 0.3f).ToArray();
                float[] scale = Enumerable.Range(0, width).Select(i => (i % 9 - 4) * 0.6f).ToArray();
                using var stream = FloatTensor(allocator, s, rows, width);
                using var branch = FloatTensor(allocator, x, rows, width);
                using var alpha = FloatTensor(allocator, a, width);
                using var beta = FloatTensor(allocator, b, width);
                using var pre = FloatTensor(allocator, scale, width);
                using var result = new Tensor(allocator, DType.Float32, rows, width);
                const float eps = 1e-5f;
                float[] expected = Scalar(s, x, a, b, prescale ? scale : null, eps);
                GgmlBasicOps.MuseSparkPostNormResidual(result, stream, branch, alpha, beta, prescale ? pre : null, eps);
                Close(expected, result.GetElementsAsFloat(expected.Length), 3e-5f);
                Assert.Equal(devices - 1, GgmlBasicOps.GetActiveRank());
                Assert.Equal(s, stream.GetElementsAsFloat(s.Length));
                Assert.Equal(x, branch.GetElementsAsFloat(x.Length));
                // Uploads finish before the output is written: aliased output is
                // safe too, though the model deliberately uses a new allocation.
                if (rows == 3 && width == 32)
                {
                    GgmlBasicOps.MuseSparkPostNormResidual(stream, stream, branch, alpha, beta, prescale ? pre : null, eps);
                    Close(expected, stream.GetElementsAsFloat(expected.Length), 3e-5f);
                }
                if (prescale && rows == 3 && width == 32)
                {
                    // Negative control catches an invalid scale-after-norm fold.
                    float[] wrong = Scalar(s, x, a, b.Zip(scale, (v, w) => v * w).ToArray(), null, eps);
                    Assert.True(wrong.Zip(expected, (v, w) => MathF.Abs(v - w)).Max() > 0.1f);
                }
            }
        }
        finally
        {
            GgmlBasicOps.RestoreRankPin(priorPin, priorRank);
            GgmlBasicOps.ClearHostBufferCache();
            context.ReleasePoolForDisposal();
        }
    }

    private static Tensor FloatTensor(IAllocator allocator, float[] values, params long[] shape)
    {
        var tensor = new Tensor(allocator, DType.Float32, shape);
        tensor.SetElementsAsFloat(values);
        return tensor;
    }

    private static float[] Scalar(float[] stream, float[] branch, float[] alpha, float[] beta, float[] scale, float eps)
    {
        int width = alpha.Length;
        var result = new float[stream.Length];
        for (int row = 0; row < stream.Length / width; row++)
        {
            double squares = 0;
            for (int c = 0; c < width; c++)
            {
                float x = branch[row * width + c] * (scale == null ? 1 : scale[c]);
                squares += (double)x * x;
            }
            double inverse = 1 / Math.Sqrt(squares / width + eps);
            for (int c = 0; c < width; c++)
            {
                int i = row * width + c;
                float x = branch[i] * (scale == null ? 1 : scale[c]);
                result[i] = stream[i] * alpha[c] + (float)(x * inverse) * beta[c];
            }
        }
        return result;
    }

    private static void ModelCheck(BackendType backend, int split)
    {
        string directory = Path.Combine(Path.GetTempPath(), "spark-residual-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string[] keys = ["TS_MUSE_SPARK_FUSE_RESIDUALS", "MAX_CONTEXT", "TS_KV_INITIAL_TOKENS", "TS_MUSE_SPARK_PREFILL_CHUNK"];
        string[] old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        var oldDtype = KvCacheDtypeConfig.Current; bool oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        var fixture = new MuseSparkSyntheticModelBuilder();
        try
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "32");
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "2");
            Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK", "3");
            MoeCpuOffloadConfig.Reset();
            string path = fixture.Write(Path.Combine(directory, "residual.gguf"), true);
            int[] tokens = [7, 21, 42, 11, 9, 86];
            var baseline = new List<float[]>();
            foreach (bool fused in new[] { false, true })
            {
                Environment.SetEnvironmentVariable(keys[0], fused ? "1" : null); // Default-off contract.
                KvCacheDtypeConfig.Set(KvCacheDtype.F16);
                using var model = new MuseSparkModel(path, backend, layerSplitDegree: split);
                var actual = new List<float[]> { model.Forward(tokens[..5]).ToArray(), model.Forward(tokens[5..]).ToArray() };
                Close(fixture.Reference(tokens[..5], true), actual[0], 3e-3f);
                Close(fixture.Reference(tokens, true), actual[1], 3e-3f);
                Assert.Equal(fused && backend != BackendType.Cpu ? 12L : 0L, model.FusedResidualCallCount);
                if (!fused) baseline = actual;
                else for (int i = 0; i < actual.Count; i++) Close(baseline[i], actual[i], 3e-5f);
            }
            // Unequal packed rows plus single-token decode exercise the other
            // model entry point, including every captured intermediate logit row.
            baseline.Clear();
            foreach (bool fused in new[] { false, true })
            {
                Environment.SetEnvironmentVariable(keys[0], fused ? "1" : null);
                KvCacheDtypeConfig.Set(KvCacheDtype.F32);
                using var model = new MuseSparkModel(path, backend, layerSplitDegree: split);
                int[][] prompts = [tokens, [81, 53, 29, 40]];
                const int blockSize = 2;
                var pool = new BlockPool(16, blockSize, model.ComputeKVBlockByteSize(blockSize));
                var sequences = prompts.Select((p, i) => new SequenceState($"res{i}", p, 2, blockSize, SamplingConfig.Default)).ToArray();
                foreach (var seq in sequences)
                    foreach (var block in pool.AllocateNew(3)) seq.BlockTable.AppendBlock(block);
                int compare = 0;
                foreach (int[] counts in new int[][] { [5, 3], [1, 1] })
                {
                    int[] starts = [0, counts[0], counts.Sum()];
                    int[][] tables = sequences.Select(s => s.BlockTable.Blocks.Select(b => b.Id).ToArray()).ToArray();
                    var positions = new List<int>(); var slots = new List<int>();
                    for (int s = 0; s < sequences.Length; s++)
                        for (int r = sequences[s].NumComputedTokens; r < sequences[s].NumComputedTokens + counts[s]; r++)
                        { positions.Add(r); slots.Add(tables[s][r / blockSize] * blockSize + r % blockSize); }
                    var capture = new float[counts.Sum() * model.Config.VocabSize];
                    model.ForwardBatch(new BatchedForwardContext
                    {
                        Sequences = sequences.ToList(), NumScheduledTokens = counts.ToList(), QueryStartLoc = starts.ToList(),
                        Positions = positions, SlotMapping = slots, BlockTables = tables,
                        MaxQueryLen = counts.Max(), MaxSeqLen = positions.Max() + 1, CaptureLogitsAll = capture,
                    });
                    for (int s = 0; s < sequences.Length; s++)
                    {
                        for (int r = 0; r < counts[s]; r++)
                        {
                            float[] actual = capture.AsSpan((starts[s] + r) * model.Config.VocabSize, model.Config.VocabSize).ToArray();
                            Close(fixture.Reference(prompts[s][..(sequences[s].NumComputedTokens + r + 1)], true), actual, 3e-3f);
                            if (!fused) baseline.Add(actual); else Close(baseline[compare], actual, 3e-5f);
                            compare++;
                        }
                        sequences[s].AdvanceComputedTokens(counts[s]);
                    }
                }
                Assert.Equal(fused && backend != BackendType.Cpu ? 8L : 0L, model.FusedResidualCallCount);
            }
        }
        finally
        {
            for (int i = 0; i < keys.Length; i++) Environment.SetEnvironmentVariable(keys[i], old[i]);
            KvCacheDtypeConfig.RestoreForTests(oldDtype, oldExplicit); MoeCpuOffloadConfig.Reset();
            Directory.Delete(directory, true);
        }
    }

    private static void Close(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.All(actual, x => Assert.True(float.IsFinite(x)));
        float error = expected.Zip(actual, (x, y) => MathF.Abs(x - y)).Max();
        Assert.True(error <= tolerance, $"Maximum error {error} exceeds {tolerance}.");
    }
}
