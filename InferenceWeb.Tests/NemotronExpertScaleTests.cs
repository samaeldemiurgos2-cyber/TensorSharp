// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron NVFP4 sidecar scales on every path (PR #282 review item 4). A GGUF with
// ".scale" sidecars must compute what its folded twin computes - on one device, on
// the batched GGML decode kernel, and under tensor parallelism, where the router
// used to drop the per-expert down scale and every scaled file was refused at load.
// See NemotronSyntheticModelBuilder for why the twin is an exact oracle.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NemotronExpertScaleTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-scale-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public NemotronExpertScaleTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        try { Directory.Delete(_directory, true); } catch { /* best effort */ }
    }

    private static readonly int[] LongPrompt = { 3, 17, 42, 9, 100, 7, 64, 11, 5, 88, 23, 51 };   // batched-by-expert prefill
    private static readonly int[] ShortPrompt = { 31, 8, 77, 4, 19 };                             // per-token MoE loop

    private string Write(string name, NemotronSyntheticModelBuilder builder)
        => builder.Write(Path.Combine(_directory, name));

    private static float[] QkvScales(bool distinct) => distinct ? new[] { 0.5f, 2f, 4f } : new[] { 0.5f, 0.5f, 0.5f };

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CpuSingle_ScaledSidecarsMatchFoldedTwin(bool latent, bool distinctQkv)
    {
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Latent = latent, Scaled = true, QkvScales = QkvScales(distinctQkv) });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Latent = latent });

        using var reference = new NemotronModel(twin, BackendType.Cpu);
        using var model = new NemotronModel(scaled, BackendType.Cpu);
        AssertSameTrajectory(reference, model, 1e-3f);
    }

    [Fact]
    public void FuseQkv_PropagatesEqualScale_KeepsDistinctScalesSeparate()
    {
        string equal = Write("equal.gguf", new NemotronSyntheticModelBuilder { Scaled = true, QkvScales = new[] { 0.5f, 0.5f, 0.5f } });
        string distinct = Write("distinct.gguf", new NemotronSyntheticModelBuilder { Scaled = true, QkvScales = new[] { 0.5f, 2f, 4f } });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder());

        using (var model = new NemotronModel(equal, BackendType.Cpu))
        {
            var quant = QuantWeights(model);
            Assert.True(quant.TryGetValue("blk.1.attn_qkv.weight", out var fused));
            Assert.Equal(0.5f, fused.Scale);
            Assert.False(quant.ContainsKey("blk.1.attn_q.weight"));
        }

        using (var model = new NemotronModel(distinct, BackendType.Cpu))
        {
            var quant = QuantWeights(model);
            Assert.False(quant.ContainsKey("blk.1.attn_qkv.weight"));
            Assert.Equal(0.5f, quant["blk.1.attn_q.weight"].Scale);
            Assert.Equal(2f, quant["blk.1.attn_k.weight"].Scale);
            Assert.Equal(4f, quant["blk.1.attn_v.weight"].Scale);
        }

        // An unscaled GGUF fuses exactly as before.
        using (var model = new NemotronModel(twin, BackendType.Cpu))
        {
            var quant = QuantWeights(model);
            Assert.True(quant.TryGetValue("blk.1.attn_qkv.weight", out var fused));
            Assert.Equal(1f, fused.Scale);
            Assert.Equal(NemotronSyntheticModelBuilder.NumHeads * NemotronSyntheticModelBuilder.HeadDim
                + 2 * NemotronSyntheticModelBuilder.NumKvHeads * NemotronSyntheticModelBuilder.HeadDim, fused.Ne1);
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CpuTensorParallel_ScaledSidecarsMatchSingleAndTwin(bool latent, bool distinctQkv)
    {
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Latent = latent, Scaled = true, QkvScales = QkvScales(distinctQkv) });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Latent = latent });

        using var single = new NemotronModel(scaled, BackendType.Cpu);
        var group = new MuseSparkTensorParallelTests.CpuGroup(2);
        using var tp = new NemotronModel(scaled, BackendType.Cpu, 2, group);
        AssertSameTrajectory(single, tp, 2e-3f);
        Assert.True(group.Reductions > 0);

        using var twinTp = new NemotronModel(twin, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2));
        AssertSameTrajectory(twinTp, tp, 2e-3f);
    }

    [Fact]
    public void CpuTensorParallel_DownScaleOnlyIsApplied()
    {
        // Only vector ffn_down_exps.scale sidecars: the load-time sidecar gate (scalar
        // .scale tensors only) never saw such a file, so --tp loaded it and the TP
        // router - which skipped the per-expert down scale - served it unscaled.
        var builder = new NemotronSyntheticModelBuilder
        {
            Scaled = true,
            ExpertUpScales = false,
            QkvScales = new[] { 1f, 1f, 1f },
            AttnOutputScale = 1f,
            SharedUpScale = 1f,
            SharedDownScale = 1f,
        };
        string scaled = Write("down-only.gguf", builder);
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder());

        using var reference = new NemotronModel(twin, BackendType.Cpu);
        using var tp = new NemotronModel(scaled, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2));
        AssertSameTrajectory(reference, tp, 2e-3f);
    }

    [Fact]
    public void TensorParallel_RejectsMisalignedExpertRowSplit()
    {
        // n_ff 96 = 3 Q8_0 blocks per down-projection row: two ranks cannot split it on
        // block boundaries. The slicer used to drop the third block silently.
        string path = Write("misaligned.gguf", new NemotronSyntheticModelBuilder { ExpertFf = 96 });
        var ex = Assert.Throws<NotSupportedException>(() =>
            new NemotronModel(path, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2)));
        Assert.Contains("ffn_down_exps", ex.Message);
        Assert.Contains("blocks of 32", ex.Message);
    }

    [Fact]
    public void ShardSeparateColumnParallel_BakesSourceScaleOnlyWhenNotOne()
    {
        // Unequal q/k/v scales cannot fuse, so the TP load packs each rank's
        // [Q_r | K_r | V_r] from the separate sources (requantized to Q8_0). With the
        // scales baked in, the pack dequantizes to exactly the twin's fused rows.
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Scaled = true, QkvScales = new[] { 0.5f, 2f, 4f } });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder());
        using (var twinTp = new NemotronModel(twin, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2)))
        using (var scaledTp = new NemotronModel(scaled, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2)))
        {
            var expected = TpQuantWeights(twinTp)["blk.1.attn_qkv.weight"];
            var actual = TpQuantWeights(scaledTp)["blk.1.attn_qkv.weight"];
            Assert.False(TpWeightScales(scaledTp).ContainsKey("blk.1.attn_qkv.weight"));
            for (int r = 0; r < 2; r++)
            {
                Assert.Equal(1f, actual[r].Scale);
                Assert.Equal(Dequantize(expected[r]), Dequantize(actual[r]));
            }
        }

        // Scale 1: the pack stays bit-identical to its sources. attn_v in F32 keeps
        // q/k/v apart by type; Q8_0 -> F32 -> Q8_0 of the Q and K rows is the identity.
        string mixed = Write("mixed.gguf", new NemotronSyntheticModelBuilder { F32AttnV = true });
        using var source = new NemotronModel(mixed, BackendType.Cpu);
        using var tp = new NemotronModel(mixed, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2));
        var shards = TpQuantWeights(tp)["blk.1.attn_qkv.weight"];
        var q = QuantWeights(source)["blk.1.attn_q.weight"];
        var k = QuantWeights(source)["blk.1.attn_k.weight"];
        int qRowsPerRank = (int)q.Ne1 / 2, kRowsPerRank = (int)k.Ne1 / 2;
        long rowBytes = q.RawBytes / q.Ne1;
        for (int r = 0; r < 2; r++)
        {
            byte[] shard = Bytes(shards[r]);
            Assert.Equal(Rows(q, r * qRowsPerRank, qRowsPerRank), shard.AsSpan(0, (int)(qRowsPerRank * rowBytes)).ToArray());
            Assert.Equal(Rows(k, r * kRowsPerRank, kRowsPerRank),
                shard.AsSpan((int)(qRowsPerRank * rowBytes), (int)(kRowsPerRank * rowBytes)).ToArray());
        }
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(3)]
    [InlineData(18)]
    public void GgmlCpuSingle_ScaledSidecarsMatchFoldedTwin(int topK)
    {
        // 24 experts at top-18 decode through TSGgml_MoEExpertsForwardF32 in a 16 + 2
        // chunk pair, each chunk carrying its experts' up scales.
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Experts = 24, TopK = topK, Scaled = true, QkvScales = QkvScales(true) });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Experts = 24, TopK = topK });

        using var reference = new NemotronModel(twin, BackendType.GgmlCpu);
        using var model = new NemotronModel(scaled, BackendType.GgmlCpu);
        AssertSameTrajectory(reference, model, 1e-3f);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false)]
    [InlineData(true)]
    public void GgmlCudaSingle_ScaledSidecarsMatchFoldedTwin(bool latent)
    {
        // At the builder's widths every latent projection and every expert of a latent
        // layer is a Q8_0 matrix under 4 KiB ([64 x 32] or [32 x 64], 2176 bytes); ExpertFf
        // 32 makes the experts of a plain layer the same. ggml_cuda releases each weight's
        // host copy once it is preloaded, and the native binders look only larger weights
        // up in the device cache, so these are uploaded through their cache keys on every
        // call: the prefill faulted in TSGgml_AddmmQuantF32 reading the released bytes
        // (ffn_latent_in, or ffn_up_exps in the batched-by-expert prefill), and decode
        // multiplied the experts by the keys' own memory. Managed cpu is the oracle: GGML
        // rounds activations to 8 bits (up to ~0.012 here), a wrong weight moves the logits
        // by whole units or to NaN.
        int expertFf = latent ? 64 : 32;
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Latent = latent, ExpertFf = expertFf, Scaled = true });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Latent = latent, ExpertFf = expertFf });

        List<Step> expected;
        using (var reference = new NemotronModel(twin, BackendType.Cpu))
            expected = Trajectory(reference);
        using var model = new NemotronModel(scaled, BackendType.GgmlCuda);
        AssertFollows(expected, model, 5e-2f);
    }

    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    // Expert parallelism: top-3 of 8 experts, so each rank's 4 whole experts can hold a token's routes.
    [InlineData(false, true, false, true)]
    [InlineData(true, true, false, true)]
    [InlineData(false, false, false, true)]
    [InlineData(false, true, true, true)]
    [InlineData(true, true, true, true)]
    [InlineData(false, false, true, true)]
    // The per-expert split: top-5 of 8 is more routes than a rank's 4 experts, so expert parallelism declines.
    [InlineData(false, true, false, false)]
    [InlineData(true, true, false, false)]
    [InlineData(false, true, true, false)]
    [InlineData(true, true, true, false)]
    public void GgmlCudaTensorParallel_ScaledSidecarsMatchFoldedTwin(bool latent, bool expertUpScales, bool wide, bool expertParallel)
    {
        // The cpu seam above runs only the per-expert host loop. On the GGML backends --tp
        // serves a MoE layer one of two ways, and both must apply the per-expert NVFP4
        // scales: the down scale folded into the route weights, the up scale before ReLU^2.
        // - Expert parallelism, when every rank owns a whole share of each layer's experts
        //   and at least as many as a token routes to. Each rank runs its own experts through
        //   TSGgml_MoEFFNPrefillRouteScaledF32, prefill and decode alike, with every route's
        //   up scale (routeUpScales); a latent layer's experts read the latent projection.
        //   A model with up scales, or a latent MoE, used to be kept off this path.
        // - Otherwise the per-expert split: every rank holds a slice of every expert, decode
        //   runs TSGgml_MoEExpertsForwardF32 on the rank's shards (up_scales) and prefill
        //   the per-expert host loop.
        // One GPU runs this with GGML_CUDA_DEVICES=2. At the builder's widths (ExpertFf
        // 64, LatentWidth 32) every rank's per-expert shard is under 4 KiB, which the native
        // binders upload through its cache key on every call: the batched expert decode
        // read the key's own memory as the weight (NaN from the first decode step, or an
        // access violation), and the one-device reference faulted reading released host
        // bytes (see GgmlCudaSingle_ScaledSidecarsMatchFoldedTwin). Wide (ExpertFf 128,
        // LatentWidth 96) puts every shard over 4 KiB, where the device cache serves it.
        int expertFf = wide ? 128 : 64, latentWidth = wide ? 96 : NemotronSyntheticModelBuilder.LatentDim;
        int topK = expertParallel ? 3 : 5;
        var builder = new NemotronSyntheticModelBuilder { Latent = latent, LatentWidth = latentWidth, ExpertFf = expertFf, TopK = topK };
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder
        {
            Latent = latent, LatentWidth = latentWidth, ExpertFf = expertFf, TopK = topK,
            Scaled = true, ExpertUpScales = expertUpScales,
        });
        string twin = Write("twin.gguf", builder);

        // The TP degree is process-global native state that the last model loaded sets,
        // so the single-device reference runs to completion before the TP model loads.
        List<Step> expected;
        using (var reference = new NemotronModel(twin, BackendType.GgmlCuda))
            expected = Trajectory(reference);

        using var tp = Assert.IsType<NemotronModel>(ModelBase.Create(scaled, BackendType.GgmlCuda, tpDegree: 2));
        int[] moeLayers = Enumerable.Range(0, builder.Layout.Length).Where(l => builder.Layout[l] == 'M').ToArray();

        // The experts carry an up scale on every MoE layer, or (down scale only) on none:
        // an up-scale row cannot match the twin below by having no up scale to apply.
        float[][] upScales = Field<float[][]>(tp, "_layerExpertUpScale");
        Assert.All(moeLayers, l => Assert.Equal(expertUpScales, upScales[l] != null));

        // Which path serves the layers. Expert parallelism builds only the stacked shards;
        // the per-expert split builds the shards the batched decode runs on.
        Assert.True(expertParallel == (Field<object>(tp, "_tpStackedUp") != null),
            expertParallel
                ? "expert parallelism declined this model, so its up-scale path went untested"
                : "expert parallelism took a top-5-of-8 model; these rows need another way to reach the per-expert split");
        Assert.Equal(!expertParallel, Field<bool>(tp, "_tpMoeBatchedDecodeReady"));

        // The per-expert split matches one device exactly here; expert parallelism's
        // ggml_mul_mat_id sums in another order (up to ~0.02). A missing or misplaced
        // expert scale moves the logits by whole units.
        AssertFollows(expected, tp, 5e-2f);

        // On the per-expert split the batched kernel must have run on each MoE layer at
        // every single-token step: a decline to the per-expert loop still matches the
        // twin, and only this count tells the two apart. Under expert parallelism the
        // expert-parallel kernel serves decode too, and the batched path never runs.
        int decodeSteps = expected.Count(step => step.Tokens.Length == 1);
        Assert.Equal(expertParallel ? 0L : (long)moeLayers.Length * decodeSteps, tp.TpMoeBatchedDecodeCallCount);
    }

    /// <summary>
    /// Runs the long prompt (batched-by-expert prefill) plus six greedy decode steps,
    /// then - after a reset - the short prompt (per-token MoE loop) plus six more,
    /// feeding both models the reference's greedy tokens and comparing every step's
    /// logits.
    /// </summary>
    internal static void AssertSameTrajectory(NemotronModel reference, NemotronModel model, float tolerance)
        => AssertFollows(Trajectory(reference), model, tolerance);

    /// <summary>One step of the trajectory: its input tokens and the reference's logits.</summary>
    private sealed record Step(int PromptLength, int Index, int[] Tokens, float[] Logits);

    /// <summary>The reference's side of <see cref="AssertSameTrajectory"/>, recorded so the
    /// reference can be disposed before the model under test loads.</summary>
    private static List<Step> Trajectory(NemotronModel reference)
    {
        var steps = new List<Step>();
        foreach (int[] prompt in new[] { LongPrompt, ShortPrompt })
        {
            reference.ResetKVCache();
            int[] tokens = prompt;
            for (int step = 0; step < 7; step++)
            {
                float[] logits = (float[])reference.Forward(tokens).Clone();
                steps.Add(new Step(prompt.Length, step, tokens, logits));
                tokens = new[] { ArgMax(logits) };
            }
        }
        return steps;
    }

    private static void AssertFollows(List<Step> expected, NemotronModel model, float tolerance)
    {
        foreach (Step step in expected)
        {
            if (step.Index == 0)
                model.ResetKVCache();
            float[] actual = model.Forward(step.Tokens);
            Close(step.Logits, actual, tolerance, $"prompt of {step.PromptLength}, step {step.Index}");
            Assert.Equal(ArgMax(step.Logits), ArgMax(actual));
        }
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static void Close(float[] expected, float[] actual, float tolerance, string where)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{where}: logit {i} is {actual[i]}");
            float d = MathF.Abs(expected[i] - actual[i]);
            if (d > worst) { worst = d; at = i; }
        }
        if (worst > tolerance)
            Assert.Fail($"{where}: max |dlogit| {worst:R} at {at} (expected {expected[at]:R}, actual {actual[at]:R}) exceeds {tolerance:R}");
    }

    private static Dictionary<string, QuantizedWeight> QuantWeights(NemotronModel model)
        => Field<Dictionary<string, QuantizedWeight>>(model, "_quantWeights");

    private static Dictionary<string, QuantizedWeight[]> TpQuantWeights(NemotronModel model)
        => Field<Dictionary<string, QuantizedWeight[]>>(model, "_tpQuantWeights");

    private static Dictionary<string, float> TpWeightScales(NemotronModel model)
        => Field<Dictionary<string, float>>(model, "_tpWeightScales");

    private static T Field<T>(NemotronModel model, string name)
        => (T)(typeof(NemotronModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            ?? typeof(ModelBase).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic))!.GetValue(model)!;

    private static unsafe float[] Dequantize(QuantizedWeight w)
    {
        var values = new float[checked((int)(w.Ne0 * w.Ne1))];
        fixed (float* destination = values)
            ManagedQuantizedOps.DequantizeRowToFloat32(w.GgmlType, w.Data, destination, w.Ne0 * w.Ne1);
        return values;
    }

    private static byte[] Bytes(QuantizedWeight w)
    {
        var bytes = new byte[checked((int)w.RawBytes)];
        System.Runtime.InteropServices.Marshal.Copy(w.Data, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] Rows(QuantizedWeight w, int firstRow, int rows)
    {
        long rowBytes = w.RawBytes / w.Ne1;
        return Bytes(w).AsSpan((int)(firstRow * rowBytes), (int)(rows * rowBytes)).ToArray();
    }
}
