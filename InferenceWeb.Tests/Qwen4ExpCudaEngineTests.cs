// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen3.8-Flash-Next (qwen4exp) on the direct-CUDA engine (<c>--backend cuda</c>), held to the native
/// token span <c>--backend ggml_cuda</c> runs, on the tiny checkpoint of
/// <see cref="Qwen4ExpSyntheticModelBuilder"/>: GDN and QSA attention layers, the PLE block, the
/// low-rank hyper-connections and every expert storage type of the shipped file.
/// </summary>
public sealed class Qwen4ExpCudaEngineTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qwen4exp-cuda-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    public Qwen4ExpCudaEngineTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "1024");
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(int indexerTopK = 16, bool q2kxlExperts = false, bool q4kxlExperts = false,
        SyntheticGguf.GgmlType? expertDownType = null, float expertDownGain = 0.25f) => Qwen4ExpSyntheticModelBuilder.Write(
        Path.Combine(_dir, $"qwen4exp-{indexerTopK}{(q2kxlExperts ? "-q2kxl" : "")}{(q4kxlExperts ? "-q4kxl" : "")}" +
            $"{(expertDownType is { } down ? "-down" + down : "")}-g{expertDownGain}.gguf"), indexerTopK,
        q2kxlExperts: q2kxlExperts, q4kxlExperts: q4kxlExperts, expertDownType: expertDownType,
        expertDownGain: expertDownGain);

    private static readonly int[] Prompt = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % 250).ToArray();
    private static readonly int[] Steps = { 17, 203, 99, 4, 150, 61 };

    /// <summary>The prompt's logits, then each decode step's.</summary>
    private static float[][] Run(ModelBase model)
    {
        var rows = new float[Steps.Length + 1][];
        rows[0] = (float[])model.ForwardRefill(Prompt).Clone();
        for (int i = 0; i < Steps.Length; i++)
            rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
        return rows;
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void TheFixture_RunsOnTheNativeTokenSpan_Cpu() => NativeSmoke(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TheFixture_RunsOnTheNativeTokenSpan_Metal() => NativeSmoke(BackendType.GgmlMetal);

    /// <summary>The fixture is the oracle for the direct engine, so first it has to be one the native
    /// executor takes whole: the token span, the in-span PLE block and QSA past its 19-cell width.</summary>
    private void NativeSmoke(BackendType backend)
    {
        string path = Fixture();
        using ModelBase model = ModelBase.Create(path, backend);
        Assert.IsType<Qwen4ExpModel>(model);
        float[][] rows = Run(model);
        for (int i = 0; i < rows.Length; i++)
        {
            Assert.Equal(Qwen4ExpSyntheticModelBuilder.Vocab, rows[i].Length);
            Assert.All(rows[i], v => Assert.True(float.IsFinite(v)));
            double norm = Math.Sqrt(rows[i].Sum(v => (double)v * v));
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: argmax {ArgMax(rows[i])} |logits| {norm:F3}");
            Assert.True(norm > 1e-3, "degenerate logits");
            if (i > 0)
                Assert.True(RelativeError(rows[i], rows[i - 1]) > 1e-3, "the step did not change the logits");
        }
    }

    // ---------------------------------------------------------------- the direct-CUDA engine

    /// <summary>Dense attention (top_k 2048, far above the 46 cells this runs): the prompt's logits and
    /// every decode step's against the native token span. Both engines read the same quantized bytes;
    /// they differ in kernels and in how activations are quantized for them.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDirectCudaEngine_AnswersAsTheNativeSpan()
    {
        string path = Fixture(indexerTopK: 2048);
        float[][] native, cuda;
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            Assert.IsType<Qwen4ExpCudaModel>(model);
            cuda = Run(model);
        }
        double worst = 0;
        for (int i = 0; i < native.Length; i++)
        {
            double err = RelativeError(cuda[i], native[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}, argmax {ArgMax(cuda[i])} vs {ArgMax(native[i])}");
        }
        Assert.True(worst <= 2e-2, $"the direct-CUDA engine strayed {worst:E3} from the native token span");
    }

    /// <summary>The UD-Q2_K_XL expert mix - IQ2_XS gate/up, one IQ3_XXS layer, IQ4_NL down - on the
    /// direct engine's per-token and grouped expert kernels, against ggml-cuda's own kernels for
    /// those formats in the native span. Before the engine learned the two formats it refused the
    /// shipped file at load ("unsupported expert quant type 17").</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDirectCudaEngine_AnswersAsTheNativeSpan_Q2KXLExperts()
    {
        string path = Fixture(indexerTopK: 2048, q2kxlExperts: true);
        float[][] native, cuda;
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            Assert.IsType<Qwen4ExpCudaModel>(model);
            cuda = Run(model);
        }
        double worst = 0;
        for (int i = 0; i < native.Length; i++)
        {
            double err = RelativeError(cuda[i], native[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}, argmax {ArgMax(cuda[i])} vs {ArgMax(native[i])}");
        }
        Assert.True(worst <= 2e-2, $"the direct-CUDA engine strayed {worst:E3} from the native token span");
    }

    /// <summary>The UD-Q4_K_XL expert mix - Q4_K gate/up over Q5_1 down, the layout 43 of the shipped
    /// file's 48 down projections take - on the direct engine's per-token kernels (the decode steps)
    /// and tensor-core kernels (the 40-token prompt), against ggml-cuda's own Q5_1 kernels in the
    /// native span. Before the engine learned Q5_1 it refused the file at load ("unsupported expert
    /// quant type 7"), and with <c>--layer-split 2</c> that refusal aborted the CLI.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TheDirectCudaEngine_AnswersAsTheNativeSpan_Q4KXLExperts()
    {
        // The routed down projection at full gain (the fixture's default writes a quarter-size update):
        // at a quarter, even a decoder that drops every fifth bit stayed under the bound (1.8e-2,
        // against 6.6e-3 here); at full gain it strays 6.3e-2, while this decoder still answers within
        // the 6.6e-3 it does at a quarter.
        string path = Fixture(indexerTopK: 2048, q4kxlExperts: true, expertDownGain: 1f);
        float[][] native, cuda;
        using (ModelBase model = ModelBase.Create(path, BackendType.GgmlCuda))
            native = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
        {
            Assert.IsType<Qwen4ExpCudaModel>(model);
            cuda = Run(model);
        }
        double worst = 0;
        for (int i = 0; i < native.Length; i++)
        {
            double err = RelativeError(cuda[i], native[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}, argmax {ArgMax(cuda[i])} vs {ArgMax(native[i])}");
        }
        Assert.True(worst <= 2e-2, $"the direct-CUDA engine strayed {worst:E3} from the native token span");
    }

    /// <summary>An expert layout no direct-CUDA kernel decodes (Q4_0 down) is refused at load as one
    /// <see cref="NotSupportedException"/> naming the tensor, which the hosts report as a load refusal,
    /// and not from inside the per-device upload.</summary>
    [CudaFact]
    public void AnUndecodableExpertLayout_IsRefusedAtLoadByTensorName()
    {
        string path = Fixture(q4kxlExperts: true, expertDownType: SyntheticGguf.GgmlType.Q4_0);
        var ex = Assert.Throws<NotSupportedException>(() => ModelBase.Create(path, BackendType.Cuda));
        Assert.StartsWith("[q4e-cuda] blk.0.ffn_down_exps.weight is Q4_0 (ggml type 2)", ex.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out string reason));
        Assert.Contains("--backend ggml_cuda", reason);
    }

    /// <summary>Two devices answer as one: the streams cross the boundary through pinned memory, and
    /// each layer keeps its caches, recurrent state and PLE history on its own device.</summary>
    [MultiCudaFact(2)]
    public void ALayerSplit_AnswersAsOneDevice()
    {
        string path = Fixture();
        float[][] single, split;
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda))
            single = Run(model);
        using (ModelBase model = ModelBase.Create(path, BackendType.Cuda, layerSplitDegree: 2))
            split = Run(model);
        for (int i = 0; i < single.Length; i++)
            Assert.True(RelativeError(split[i], single[i]) <= 1e-5, $"forward {i} differs across the split");
    }

    /// <summary>A reset clears the recurrent states, the QSA keys and the PLE history as well as the
    /// attention cache.</summary>
    [CudaFact]
    public void AReset_StartsAFreshSequence()
    {
        using ModelBase model = ModelBase.Create(Fixture(), BackendType.Cuda);
        float[][] first = Run(model);
        model.ResetKVCache();
        float[][] again = Run(model);
        for (int i = 0; i < first.Length; i++)
            Assert.True(RelativeError(again[i], first[i]) <= 1e-6, $"forward {i} differs after a reset");
    }

    // ---------------------------------------------------------------- QSA

    /// <summary>
    /// Every sparse QSA step (top_k 16: past 19 cells each query keeps its best blocks of four) held
    /// to the rules of the native executor's indexer, computed from the engine's own inputs:
    /// <list type="bullet">
    /// <item>a block's key is the mean of its members' cached raw keys, RMS-normed times k_norm and
    /// rotated to the block's first position;</item>
    /// <item>a query scores the complete blocks it has seen as the sum over indexer heads of
    /// ReLU'd dot products;</item>
    /// <item>it attends the cells of its own unfinished block, then the best blocks' cells until
    /// top_k + ratio - 1 cells are listed, ties to the lower block.</item>
    /// </list>
    /// The native top-k leaves the order among equal scores unspecified, so an end-to-end comparison
    /// cannot pin a selection down; the dense comparison above covers everything else.
    /// </summary>
    [CudaFact]
    public void TheSparseSelection_FollowsTheIndexerRules()
    {
        string path = Fixture();
        var probes = new System.Collections.Generic.List<TensorSharp.Cuda.Q4eCudaEngine.SparseLayerProbe>();
        using (var ex = OpenExecutor(path))
        {
            ex.Engine.SparseProbe = probes.Add;
            Step(ex, Prompt);
            foreach (int token in Steps)
                Step(ex, new[] { token });
        }
        // Two QSA layers, each past the width in the prompt and in every step.
        Assert.Equal(2 * (1 + Steps.Length), probes.Count);
        using var gguf = new GgufFile(path);
        foreach (var probe in probes)
            CheckSparseLayer(probe, gguf);
    }

    private static void CheckSparseLayer(TensorSharp.Cuda.Q4eCudaEngine.SparseLayerProbe p, GgufFile gguf)
    {
        int d = p.IdxDim, h = p.IdxHeads, r = p.Ratio;
        string where = $"layer {p.Layer} at {p.P0}";
        float[] kNorm = F32(gguf, $"blk.{p.Layer}.indexer.k_norm.weight");

        // Block keys from the cached raw keys, as stored.
        int blocks = (p.P0 + p.Nt) / r;
        var keys = new float[blocks * d];
        for (int b = 0; b < blocks; b++)
        {
            var mean = new double[d];
            for (int c = 0; c < d; c++)
            {
                double acc = 0;
                for (int j = 0; j < r; j++)
                    acc += (float)p.RawKeys[(long)(b * r + j) * d + c];
                mean[c] = acc / r;
            }
            double ss = mean.Sum(v => v * v);
            double inv = 1.0 / Math.Sqrt(ss / d + p.Eps);
            var x = new double[d];
            for (int c = 0; c < d; c++)
                x[c] = mean[c] * inv * kNorm[c];
            Rope(x, b * r, p.NRot, p.RopeBase, p.RopeFreqScale);
            for (int c = 0; c < d; c++)
                keys[b * d + c] = (float)x[c];
        }
        AssertClose(keys, p.Pooled, 1e-4, $"block keys, {where}");

        int width = p.TopK + r - 1;
        for (int t = 0; t < p.Nt; t++)
        {
            int q = p.P0 + t;
            if (q + 1 <= width)
            {
                Assert.Equal(-1, p.CellCnt[t]);
                continue;
            }
            int nb = (q + 1) / r;
            var want = new float[nb];
            var got = new float[nb];
            for (int b = 0; b < nb; b++)
            {
                double score = 0;
                for (int k = 0; k < h; k++)
                {
                    double dot = 0;
                    for (int c = 0; c < d; c++)
                        dot += (double)p.Queries[((long)t * h + k) * d + c] * p.Pooled[b * d + c];
                    score += Math.Max(dot, 0);
                }
                want[b] = (float)score;
                got[b] = p.Scores[(long)t * p.ScoreStride + b];
            }
            AssertClose(want, got, 1e-4, $"block scores, {where}, query {q}");

            // The rules over the ENGINE's scores (so the check is exact), in the kernel's float order.
            int tail = (q + 1) / r * r, t0 = q + 1 - tail, need = width - t0, full = need / r, part = need % r;
            int[] ranked = Enumerable.Range(0, nb).OrderByDescending(b => OrderKey(got[b])).ThenBy(b => b)
                .Take(full + (part > 0 ? 1 : 0)).ToArray();
            var cells = Enumerable.Range(tail, t0).ToList();
            for (int i = 0; i < ranked.Length; i++)
                cells.AddRange(Enumerable.Range(ranked[i] * r, i < full ? r : part));
            Assert.Equal(width, p.CellCnt[t]);
            Assert.Equal(cells.OrderBy(c => c), p.Cells.Skip(t * p.CellStride).Take(width).OrderBy(c => c));
        }
    }

    /// <summary>NEOX rotary over the first nRot values: pair (i, i + nRot/2) at angle
    /// freqScale * pos * base^(-2i/nRot).</summary>
    private static void Rope(double[] x, int pos, int nRot, float ropeBase, float freqScale)
    {
        int half = nRot / 2;
        for (int i = 0; i < half; i++)
        {
            double theta = freqScale * (pos * Math.Pow(ropeBase, -2.0 * i / nRot));
            double c = Math.Cos(theta), s = Math.Sin(theta);
            double a = x[i], b = x[i + half];
            x[i] = a * c - b * s;
            x[i + half] = a * s + b * c;
        }
    }

    /// <summary>The select kernel's total order on floats (the IEEE bits made unsigned-comparable).</summary>
    private static uint OrderKey(float v)
    {
        uint k = BitConverter.SingleToUInt32Bits(v);
        return (k & 0x80000000u) != 0 ? ~k : k | 0x80000000u;
    }

    private static float[] F32(GgufFile gguf, string name)
    {
        byte[] raw = gguf.ReadTensorData(gguf.Tensors[name]);
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    /// <summary>Worst absolute difference within <paramref name="tol"/> times the expected values'
    /// largest magnitude (at least 1).</summary>
    private static void AssertClose(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1.0, expected.Length == 0 ? 0 : expected.Max(v => Math.Abs((double)v)));
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]) / scale);
        Assert.True(worst <= tol, $"{what}: off by {worst:E3} (tolerance {tol:E1})");
    }

    // ---------------------------------------------------------------- sequence slots

    private static Qwen4ExpCudaExecutor OpenExecutor(string path) => new Qwen4ExpCudaExecutor(path, 1024, 64, 1);

    private static float[] Step(Qwen4ExpCudaExecutor ex, int[] tokens)
    {
        var logits = new float[ex.VocabSize];
        ex.Forward(tokens, logits);
        return logits;
    }

    private static int NewActiveSlot(Qwen4ExpCudaExecutor ex)
    {
        int slot = ex.SlotAlloc();
        Assert.True(slot >= 0);
        Assert.True(ex.SetActiveSlot(slot));
        return slot;
    }

    /// <summary>Two sequences interleaved chunk by chunk answer exactly as each alone: every cache,
    /// recurrent state and PLE history belongs to its slot.</summary>
    [CudaFact]
    public void InterleavedSlots_AnswerExactlyAsEachSequenceAlone()
    {
        using var ex = OpenExecutor(Fixture());
        int[][] a = { Prompt[..24], Steps[..1], Steps[1..2] };
        int[][] b = { Prompt[16..40], Steps[3..4], Steps[4..5] };
        var aloneA = a.Select(c => Step(ex, c)).ToList();
        ex.ResetChecked();
        var aloneB = b.Select(c => Step(ex, c)).ToList();

        int sa = NewActiveSlot(ex);
        int sb = NewActiveSlot(ex);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(ex.SetActiveSlot(sa));
            Assert.True(RelativeError(Step(ex, a[i]), aloneA[i]) <= 1e-6, $"sequence A chunk {i}");
            Assert.True(ex.SetActiveSlot(sb));
            Assert.True(RelativeError(Step(ex, b[i]), aloneB[i]) <= 1e-6, $"sequence B chunk {i}");
        }
    }

    /// <summary>A captured decode step answers bit for bit as the same step issued kernel by kernel:
    /// the token, the position and the PLE rows are read at launch, blocks complete inside the
    /// graph, and crossing the QSA width (19 cells) captures the sparse regime beside the dense one.
    /// A freed sequence takes its graphs with it.</summary>
    [CudaFact]
    public void CapturedDecodeSteps_AnswerAsUncapturedSteps()
    {
        string path = Fixture();
        // Positions 12..27: dense to 18, sparse from 19, four blocks completing on the way.
        int[] tokens = Enumerable.Range(0, 16).Select(i => Steps[i % Steps.Length] + i).ToArray();
        float[][] Decode(Qwen4ExpCudaExecutor ex)
        {
            Step(ex, Prompt[..12]);
            return tokens.Select(t => Step(ex, new[] { t })).ToArray();
        }

        float[][] captured, plain;
        using (var ex = OpenExecutor(path))
        {
            captured = Decode(ex);
            Assert.Equal(2, ex.Engine.CapturedDecodeGraphs);

            int first = ex.ActiveSlot;
            int other = NewActiveSlot(ex);
            Step(ex, Prompt[..4]);
            Step(ex, new[] { 7 });
            Assert.Equal(3, ex.Engine.CapturedDecodeGraphs);
            Assert.True(ex.SetActiveSlot(first));
            Assert.True(ex.SlotFree(other));
            Assert.Equal(2, ex.Engine.CapturedDecodeGraphs);
        }
        _env.Set("TS_Q4E_GRAPHS", "0");
        using (var ex = OpenExecutor(path))
        {
            plain = Decode(ex);
            Assert.Equal(0, ex.Engine.CapturedDecodeGraphs);
        }
        for (int i = 0; i < tokens.Length; i++)
            Assert.Equal(plain[i], captured[i]);
    }

    /// <summary>A batched decode step computes each sequence's row exactly as that sequence's own step
    /// does, over several steps, with the rows at different positions on both sides of the QSA
    /// width.</summary>
    [CudaFact]
    public void BatchedDecode_ComputesEverySequencesOwnStep()
    {
        using var ex = OpenExecutor(Fixture());
        int[] prompts = { 20, 13, 31 };
        int n = prompts.Length, steps = 4;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            batched[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
            serial[i] = NewActiveSlot(ex);
            Step(ex, Prompt[..prompts[i]]);
        }
        for (int step = 0; step < steps; step++)
        {
            int[] tokens = Enumerable.Range(0, n).Select(i => Steps[(i + step) % Steps.Length]).ToArray();
            int[] positions = prompts.Select(p => p + step).ToArray();
            var rows = new float[n * ex.VocabSize];
            Assert.True(ex.ForwardBatchedDecode(batched, tokens, positions, rows));
            for (int i = 0; i < n; i++)
            {
                Assert.True(ex.SetActiveSlot(serial[i]));
                float[] want = Step(ex, new[] { tokens[i] });
                Assert.Equal(want, rows.AsSpan(i * ex.VocabSize, ex.VocabSize).ToArray());
            }
        }
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    private static double RelativeError(float[] actual, float[] expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
