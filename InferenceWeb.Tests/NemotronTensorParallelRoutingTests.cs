// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The tensor-parallel MoE router must route exactly like the single-device one: the
// same experts, IN THE SAME ORDER (the MoE kernels sum the routes in that order, so it
// moves the low bits of every layer's output) and bit-identical weights. --tp once sorted
// its top-k by probability, then switched to a router that normalised by 1/sum where the
// single-device per-token loop divides by the sum; each change moved the --tp greedy text.
// Layer 0 of the synthetic model is a MoE layer fed the token embedding directly, so its
// router logits are bit-identical on one device and under TP and any difference in the
// routes is the router's own. Runs on the managed cpu backend's model-free TP seam.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NemotronTensorParallelRoutingTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-tproute-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public NemotronTensorParallelRoutingTests()
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

    // 12 tokens take the single-device batched-by-expert prefill, 5 its per-token
    // loop; every decode step after either takes the per-token loop.
    private static readonly int[] LongPrompt = { 3, 17, 42, 9, 100, 7, 64, 11, 5, 88, 23, 51 };
    private static readonly int[] ShortPrompt = { 31, 8, 77, 4, 19 };

    private sealed record Route(int Token, int[] Experts, int[] WeightBits);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TensorParallelRouter_MatchesSingleDevice_ExpertsOrderAndWeightBits(bool scaled)
    {
        // 32 experts, top-6: Nemotron 3.5's top-k over enough experts that a sorted
        // top-k and the single-device scan order disagree on many tokens.
        string path = new NemotronSyntheticModelBuilder
        {
            Experts = 32,
            TopK = 6,
            Scaled = scaled,
        }.Write(Path.Combine(_directory, scaled ? "scaled.gguf" : "unscaled.gguf"));

        using var single = new NemotronModel(path, BackendType.Cpu);
        using var tp = new NemotronModel(path, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2));
        var singleRoutes = new List<Route>();
        var tpRoutes = new List<Route>();
        single.MoERouteObserverForTest = Recorder(singleRoutes);
        tp.MoERouteObserverForTest = Recorder(tpRoutes);

        int compared = 0, unsortedTokens = 0;
        foreach (int[] prompt in new[] { LongPrompt, ShortPrompt })
        {
            single.ResetKVCache();
            tp.ResetKVCache();
            int[] tokens = prompt;
            for (int step = 0; step < 6; step++)
            {
                singleRoutes.Clear();
                tpRoutes.Clear();
                float[] logits = (float[])single.Forward(tokens).Clone();
                tp.Forward(tokens);

                Assert.Equal(tokens.Length, singleRoutes.Count);
                Assert.Equal(tokens.Length, tpRoutes.Count);
                for (int i = 0; i < tokens.Length; i++)
                {
                    string where = $"prompt of {prompt.Length}, step {step}, token row {i}";
                    Assert.True(singleRoutes[i].Experts.SequenceEqual(tpRoutes[i].Experts),
                        $"{where}: experts {string.Join(",", tpRoutes[i].Experts)} under --tp, " +
                        $"{string.Join(",", singleRoutes[i].Experts)} on one device");
                    Assert.True(singleRoutes[i].WeightBits.SequenceEqual(tpRoutes[i].WeightBits),
                        $"{where}: route weights differ in their bits");
                    compared++;
                    if (!scaled && !IsInSelectionOrder(singleRoutes[i]))
                        unsortedTokens++;
                }
                tokens = new[] { ArgMax(logits) };
            }
        }

        Assert.Equal(2 * 5 + LongPrompt.Length + ShortPrompt.Length, compared);
        // The router --tp used before this suite existed sorted its top-k by selection
        // probability; it fails the order check on each token counted here.
        if (!scaled)
            Assert.True(unsortedTokens > 0, "no token's single-device route order differs from a sorted top-k");
    }

    private static NemotronModel.MoERouteObserver Recorder(List<Route> routes)
        => (layer, tokenRow, experts, weights) =>
        {
            if (layer != 0)
                return;
            var bits = new int[weights.Length];
            for (int k = 0; k < weights.Length; k++)
                bits[k] = BitConverter.SingleToInt32Bits(weights[k]);
            routes.Add(new Route(tokenRow, experts.ToArray(), bits));
        };

    // The order a top-k sorted by selection probability would emit, for the unscaled
    // model: the bias-pinned expert first, then the rest by falling weight (the weights
    // are then the normalised sigmoid probabilities, which order like the selection
    // scores of the unbiased experts).
    private static bool IsInSelectionOrder(Route route)
    {
        if (route.Experts[0] != NemotronSyntheticModelBuilder.PinnedExpert)
            return false;
        for (int k = 2; k < route.WeightBits.Length; k++)
            if (BitConverter.Int32BitsToSingle(route.WeightBits[k]) > BitConverter.Int32BitsToSingle(route.WeightBits[k - 1]))
                return false;
        return true;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
