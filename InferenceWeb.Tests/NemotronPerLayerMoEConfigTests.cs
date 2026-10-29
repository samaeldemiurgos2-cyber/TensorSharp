// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron's per-layer MoE geometry (PR #282 review items 6 and 9). Bolt stores
// expert_used_count as a per-layer array whose element [0] - a Mamba2 layer - is 0,
// and expert_count as a scalar larger than any layer's stacked expert tensor. The
// resolver must take each FFN layer's values from the array and the tensors, call a
// layout per-block only when the FFN layers differ from EACH OTHER, refuse a top-k
// no layer can route, and the load banner must report the resolved values.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using LayerType = TensorSharp.Models.NemotronModel.LayerType;

namespace InferenceWeb.Tests;

public sealed class NemotronPerLayerMoEConfigTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-moecfg-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();

    public NemotronPerLayerMoEConfigTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
    }

    public void Dispose()
    {
        _environment.Dispose();
        try { Directory.Delete(_directory, true); } catch { /* best effort */ }
    }

    private static readonly LayerType M = LayerType.Mamba2, A = LayerType.Attention, F = LayerType.FFN;

    [Fact]
    public void UniformScalarConfig_KeepsTheGlobals()
    {
        var types = new[] { A, F, A, F };
        var r = NemotronModel.ResolvePerLayerMoEConfig(types, _ => 8, usedArray: null, globalExperts: 8, globalUsed: 3);

        Assert.False(r.Varied);
        Assert.Equal(new[] { 0, 8, 0, 8 }, r.Experts);
        Assert.Equal(new[] { 0, 3, 0, 3 }, r.Used);
        Assert.Equal(8, r.MaxExperts);
        Assert.Equal(3, r.MaxUsed);
    }

    [Fact]
    public void BoltLikeConfig_UsesPerLayerArraysAndTensors()
    {
        // Bolt: Mamba2 at 0, MoE layers with 240/240/432 experts, top-9 everywhere,
        // a scalar expert_count (496) above every layer and expert_used_count[0] = 0.
        var types = new[] { M, F, M, F, F };
        int[] stacked = { 0, 240, 0, 240, 432 };
        uint[] used = { 0, 9, 0, 9, 9 };
        var r = NemotronModel.ResolvePerLayerMoEConfig(types, l => stacked[l], used, globalExperts: 496, globalUsed: 0);

        Assert.True(r.Varied);
        Assert.Equal(new[] { 0, 240, 0, 240, 432 }, r.Experts);
        Assert.Equal(new[] { 0, 9, 0, 9, 9 }, r.Used);
        Assert.Equal(432, r.MaxExperts);
        Assert.Equal(9, r.MaxUsed);
    }

    [Fact]
    public void ArrayOnlyUniformConfig_IsNotPerBlock()
    {
        // The scalar read of an array-valued expert_used_count is element [0] (a
        // Mamba2 layer's 0). Comparing FFN layers with THAT flagged every such file as
        // per-block - which refuses --tp - although its FFN layers are identical.
        var types = new[] { M, F, M, F };
        uint[] used = { 0, 6, 0, 6 };
        var r = NemotronModel.ResolvePerLayerMoEConfig(types, _ => 128, used, globalExperts: 128, globalUsed: 0);

        Assert.False(r.Varied);
        Assert.Equal(6, r.MaxUsed);
        Assert.Equal(new[] { 0, 6, 0, 6 }, r.Used);
    }

    [Fact]
    public void ArrayWithDifferentTopK_IsPerBlock()
    {
        var types = new[] { M, F, M, F };
        uint[] used = { 0, 6, 0, 4 };
        var r = NemotronModel.ResolvePerLayerMoEConfig(types, _ => 128, used, globalExperts: 128, globalUsed: 0);

        Assert.True(r.Varied);
        Assert.Equal(6, r.MaxUsed);
    }

    [Fact]
    public void ShortArray_FallsBackToTheScalarPastItsEnd()
    {
        var types = new[] { A, F, A, F };
        uint[] used = { 0, 5 };
        var r = NemotronModel.ResolvePerLayerMoEConfig(types, _ => 16, used, globalExperts: 16, globalUsed: 5);

        Assert.Equal(new[] { 0, 5, 0, 5 }, r.Used);
        Assert.False(r.Varied);
    }

    [Fact]
    public void MissingStackedTensor_FallsBackToTheGlobalExpertCount()
    {
        var types = new[] { F, F };
        var r = NemotronModel.ResolvePerLayerMoEConfig(types, _ => 0, null, globalExperts: 32, globalUsed: 4);
        Assert.Equal(new[] { 32, 32 }, r.Experts);
        Assert.False(r.Varied);
    }

    [Theory]
    [InlineData(0u)]
    [InlineData(9u)]
    public void TopKOutsideOneToExperts_IsRefusedNamingTheLayer(uint badUsed)
    {
        var types = new[] { M, F, A, F };
        uint[] used = { 0, 2, 0, badUsed };
        var ex = Assert.Throws<InvalidDataException>(() =>
            NemotronModel.ResolvePerLayerMoEConfig(types, _ => 8, used, globalExperts: 8, globalUsed: 0));
        Assert.Contains("layer 3", ex.Message);
        Assert.Contains($"expert_used_count {badUsed}", ex.Message);
    }

    [Fact]
    public void DenseModel_ResolvesNothing()
    {
        var types = new[] { M, F, A };
        var r = NemotronModel.ResolvePerLayerMoEConfig(types, _ => 0, new uint[] { 0, 0, 0 }, globalExperts: 0, globalUsed: 0);
        Assert.False(r.Varied);
        Assert.All(r.Experts, e => Assert.Equal(0, e));
        Assert.All(r.Used, u => Assert.Equal(0, u));
    }

    [Fact]
    public void DescribeMoELayout_UniformPrintsOneLineWithTheResolvedTopK()
    {
        var types = new[] { M, F, M, F };
        string text = NemotronModel.DescribeMoELayout(types, new[] { 0, 512, 0, 512 }, new[] { 0, 22, 0, 22 }, new[] { 0, 2688, 0, 2688 });
        Assert.Equal("MoE: 2 layers, 512 experts, top-22, n_ff 2688", text);
    }

    [Fact]
    public void DescribeMoELayout_BoltPrintsRangesAndPerLayerDetail()
    {
        var types = new[] { M, F, M, F, M, M, F };
        string text = NemotronModel.DescribeMoELayout(types,
            new[] { 0, 240, 0, 240, 0, 0, 432 },
            new[] { 0, 9, 0, 9, 0, 0, 9 },
            new[] { 0, 2048, 0, 1536, 0, 0, 2560 });

        Assert.Contains("top-9", text);
        Assert.Contains("experts 240-432", text);
        Assert.Contains("n_ff 1536-2560", text);
        Assert.Contains("1:240/9/2048", text);
        Assert.Contains("6:432/9/2560", text);
        Assert.DoesNotContain("top-0", text);
    }

    [Fact]
    public void PerLayerFixture_LoadsWithResolvedArraysAndBanner()
    {
        // Layer 1: 8 experts top-3; layer 3: 4 experts top-2; expert_used_count is an
        // array whose element [0] (an attention layer) is 0.
        string path = new NemotronSyntheticModelBuilder
        {
            Layout = "AMAM",
            LayerExperts = new[] { 8, 4 },
            UsedPerLayer = new uint[] { 0, 3, 0, 2 },
        }.Write(Path.Combine(_directory, "per-layer.gguf"));

        var console = new StringWriter();
        var original = Console.Out;
        NemotronModel model;
        Console.SetOut(console);
        try { model = new NemotronModel(path, BackendType.Cpu); }
        finally { Console.SetOut(original); }

        using (model)
        {
            Assert.Equal(new[] { 0, 8, 0, 4 }, Field<int[]>(model, "_numExpertsPerLayer"));
            Assert.Equal(new[] { 0, 3, 0, 2 }, Field<int[]>(model, "_numExpertsUsedPerLayer"));
            Assert.True(Field<bool>(model, "_hasPerLayerMoEConfig"));
            Assert.Equal(8, model.Config.NumExperts);
            Assert.Equal(3, model.Config.NumExpertsUsed);

            string banner = console.ToString();
            Assert.Contains("MoE (per-layer): 2 layers, experts 4-8, top-2-3", banner);
            Assert.Contains("1:8/3/64 3:4/2/64", banner);
            Assert.DoesNotContain("top-0", banner);

            float[] logits = model.Forward(new[] { 3, 17, 42, 9, 100, 7, 64, 11, 5 });
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
            float[] decode = model.Forward(new[] { 12 });
            Assert.All(decode, v => Assert.True(float.IsFinite(v)));
        }
    }

    [Fact]
    public void PerLayerFixture_UnderTensorParallelism_FollowsTheSingleDeviceTrajectory()
    {
        // Bolt-shaped: each MoE block has its own expert count and top-k, with scalar
        // .scale sidecars and NVFP4 expert scales. TP used to refuse a per-block config;
        // every rank now routes and shards each layer by its own counts.
        string path = new NemotronSyntheticModelBuilder
        {
            Layout = "AMAM",
            LayerExperts = new[] { 8, 4 },
            UsedPerLayer = new uint[] { 0, 3, 0, 2 },
            Scaled = true,
        }.Write(Path.Combine(_directory, "per-layer-tp.gguf"));

        using var single = new NemotronModel(path, BackendType.Cpu);
        var group = new MuseSparkTensorParallelTests.CpuGroup(2);
        using var tp = new NemotronModel(path, BackendType.Cpu, 2, group);
        NemotronExpertScaleTests.AssertSameTrajectory(single, tp, 2e-3f);
        Assert.True(group.Reductions > 0);
    }

    private static T Field<T>(object model, string name)
        => (T)typeof(NemotronModel).GetField(name, System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;
}
