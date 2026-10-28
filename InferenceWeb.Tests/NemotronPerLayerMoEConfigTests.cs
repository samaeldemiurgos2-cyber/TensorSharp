// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron 3.5 Bolt (the "Puzzle" checkpoint) gives each MoE block its own expert count and
// top-k: the count is the routed-expert tensor's last dim, the top-k the per-layer
// expert_used_count array, and the scalar expert_used_count is not set at all (the load banner
// printed "top-0"). These pin the derivation without a 74B checkpoint.
using System;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NemotronPerLayerMoEConfigTests
{
    private static readonly bool[] Pattern = { false, true, false, true, true, false };

    [Fact]
    public void UniformCheckpoint_KeepsTheGlobalCountsOnEveryMoELayer()
    {
        var config = NemotronModel.DerivePerLayerMoEConfig(Pattern, 128, 6, _ => 128, usedCountArray: null);

        Assert.False(config.Varied);
        Assert.Equal(new[] { 0, 128, 0, 128, 128, 0 }, config.Experts);
        Assert.Equal(new[] { 0, 6, 0, 6, 6, 0 }, config.Used);
        Assert.Equal((128, 6), (config.MaxExperts, config.MaxUsed));
        Assert.Equal("3 MoE layers, 128 experts, top-6", config.Describe());
    }

    [Fact]
    public void BoltStyleCheckpoint_TakesCountsFromTheTensorsAndTopKFromTheArray()
    {
        // Bolt: no scalar top-k (0), per-layer array, varied stacked tensors.
        int[] stacked = { 0, 64, 0, 96, 128, 0 };
        uint[] used = { 0, 4, 0, 6, 8, 0 };

        var config = NemotronModel.DerivePerLayerMoEConfig(Pattern, 128, 0, l => stacked[l], used);

        Assert.True(config.Varied);
        Assert.Equal(new[] { 0, 64, 0, 96, 128, 0 }, config.Experts);
        Assert.Equal(new[] { 0, 4, 0, 6, 8, 0 }, config.Used);
        Assert.Equal((128, 8), (config.MaxExperts, config.MaxUsed));
        Assert.Equal("3 MoE layers, 64-128 experts, top-4-8", config.Describe());
    }

    [Fact]
    public void DenseCheckpoint_HasNoMoELayers()
    {
        var config = NemotronModel.DerivePerLayerMoEConfig(Pattern, 0, 0, _ => 0, null);
        Assert.False(config.Varied);
        Assert.All(config.Experts, e => Assert.Equal(0, e));
        Assert.Equal("no MoE layers", config.Describe());
    }

    [Fact]
    public void ATopKAboveTheLayersExpertCount_IsRefused()
    {
        uint[] used = { 0, 70, 0, 6, 8, 0 };
        var ex = Assert.Throws<System.IO.InvalidDataException>(() =>
            NemotronModel.DerivePerLayerMoEConfig(Pattern, 128, 0, l => l == 1 ? 64 : 128, used));
        Assert.Contains("layer 1", ex.Message);
    }
}
