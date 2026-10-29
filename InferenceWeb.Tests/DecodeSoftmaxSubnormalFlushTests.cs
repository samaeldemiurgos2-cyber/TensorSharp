// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The host decode attention's exp row (ModelBase.VecExpRow) flushes terms below
// exp(-50) to zero. A sharply peaked attention row otherwise leaves its tail in the
// subnormal range, and every later multiply on a subnormal takes a microcode assist:
// the --tp decode path (host attention on every GGML backend) ran up to 20x slower on
// such rows, so decode speed followed the generated text.
using System;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed unsafe class DecodeSoftmaxSubnormalFlushTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(37)]      // vector body plus a padded tail at every vector width
    [InlineData(640)]
    public void PeakedRow_HasNoSubnormalTerms_AndKeepsEveryTermAboveTheCutoff(int n)
    {
        var rnd = new Random(n);
        var scores = new float[n];
        for (int i = 0; i < n; i++)
            scores[i] = i == 0 ? 0f : -(float)(rnd.NextDouble() * 110.0);   // half the tail is subnormal-bound
        if (n > 2) scores[1] = float.NegativeInfinity;

        float[] row = (float[])scores.Clone();
        float sum;
        fixed (float* p = row)
            sum = ModelBase.VecExpRow(p, 0f, n);

        double expectedSum = 0;
        for (int i = 0; i < n; i++)
        {
            Assert.False(float.IsSubnormal(row[i]), $"term {i} (score {scores[i]}) is subnormal: {row[i]:R}");
            if (scores[i] < ModelBase.DecodeSoftmaxFlushBelow)
            {
                Assert.Equal(0f, row[i]);
            }
            else
            {
                double want = Math.Exp(scores[i]);
                Assert.True(Math.Abs(row[i] - want) <= 1e-6 * want, $"term {i}: {row[i]:R} vs {want:R}");
                expectedSum += want;
            }
        }
        Assert.True(Math.Abs(sum - expectedSum) <= 1e-6 * expectedSum, $"sum {sum:R} vs {expectedSum:R}");
    }

    [Fact]
    public void FlushedTail_IsBelowWhatAnF32SumCanHold()
    {
        // The cutoff only drops what the softmax could not have kept: next to the max
        // term (1), even 2^30 terms at the cutoff sum below half an ulp of 1.
        double cutoff = Math.Exp(ModelBase.DecodeSoftmaxFlushBelow);
        Assert.True(cutoff * (1L << 30) < Math.Pow(2, -24));
        // ... and above it a normalised term times a tiny F16 value stays normal, so the
        // w*V accumulation never meets a subnormal either (2^20 positions, |v| = 2^-24).
        Assert.True(cutoff / (1 << 20) * Math.Pow(2, -24) > float.Epsilon * Math.Pow(2, 23));
    }

    [Fact]
    public void NaNScore_StillPropagates()
    {
        var row = new float[19];
        for (int i = 0; i < row.Length; i++) row[i] = -i;
        row[9] = float.NaN;
        float sum;
        fixed (float* p = row)
            sum = ModelBase.VecExpRow(p, 0f, row.Length);
        Assert.True(float.IsNaN(row[9]));
        Assert.True(float.IsNaN(sum));
    }
}
