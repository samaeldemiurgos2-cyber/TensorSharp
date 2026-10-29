// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The batched Nemotron MoE expert kernel (TSGgml_MoEExpertsForwardF32) and the
// chunk loop around it (PR #282 review items 6 and 10), on ggml-cpu with F32 expert
// weights against a double-precision reference:
//   relu(us_k * (W_up_k x))^2 -> W_down_k -> * route_k, summed over the routes.
// The per-expert up scale has to land BEFORE the squaring: deferring a 2^-66 scale
// past ReLU^2 overflows to +Inf, and a -1 scale shows up as a sign error. Models that
// route more than 16 experts per token (Super VL: 22) run the kernel in chunks, and
// each expert's up scale must travel with it into its chunk.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed unsafe class NemotronMoEExpertsKernelTests
{
    // F32 weights small enough (< 4 KiB each) to stay out of the native weight cache,
    // which is keyed by host pointer and would outlive these buffers.
    private const int InDim = 32;
    private const int Ff = 24;
    private const int OutDim = 16;

    private sealed class Experts : IDisposable
    {
        public readonly float[][] Up, Down;
        public readonly QuantizedWeight[] UpW, DownW;

        public Experts(int count, int seed, Func<int, float> upMultiplier = null)
        {
            var rnd = new Random(seed);
            Up = new float[count][];
            Down = new float[count][];
            UpW = new QuantizedWeight[count];
            DownW = new QuantizedWeight[count];
            for (int e = 0; e < count; e++)
            {
                float m = upMultiplier?.Invoke(e) ?? 1f;
                Up[e] = Enumerable.Range(0, InDim * Ff).Select(_ => (float)(rnd.NextDouble() - 0.5) * m).ToArray();
                Down[e] = Enumerable.Range(0, Ff * OutDim).Select(_ => (float)(rnd.NextDouble() - 0.5)).ToArray();
                UpW[e] = new QuantizedWeight(ToBytes(Up[e]), 0, InDim, Ff);
                DownW[e] = new QuantizedWeight(ToBytes(Down[e]), 0, Ff, OutDim);
            }
        }

        public IntPtr[] UpPtrs(int[] order) => order.Select(e => UpW[e].Data).ToArray();
        public IntPtr[] DownPtrs(int[] order) => order.Select(e => DownW[e].Data).ToArray();

        public void Dispose()
        {
            foreach (var w in UpW) w.Dispose();
            foreach (var w in DownW) w.Dispose();
        }

        private static byte[] ToBytes(float[] values)
        {
            var bytes = new byte[values.Length * sizeof(float)];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return bytes;
        }
    }

    private static float[] Input(int seed)
    {
        var rnd = new Random(seed);
        return Enumerable.Range(0, InDim).Select(_ => (float)(rnd.NextDouble() * 2 - 1)).ToArray();
    }

    private static double[] Reference(Experts experts, int[] order, float[] x, float[] route, float[] upScales)
    {
        var result = new double[OutDim];
        for (int k = 0; k < order.Length; k++)
        {
            int e = order[k];
            var act = new double[Ff];
            for (int j = 0; j < Ff; j++)
            {
                double z = 0;
                for (int i = 0; i < InDim; i++)
                    z += (double)experts.Up[e][j * InDim + i] * x[i];
                z *= upScales?[k] ?? 1.0;
                double relu = Math.Max(z, 0);
                act[j] = relu * relu;
            }
            for (int o = 0; o < OutDim; o++)
            {
                double y = 0;
                for (int j = 0; j < Ff; j++)
                    y += (double)experts.Down[e][o * Ff + j] * act[j];
                result[o] += route[k] * y;
            }
        }
        return result;
    }

    private static GgmlAllocator CpuAllocator() => new(new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu), 0);

    private static float[] RunKernel(IAllocator allocator, Experts experts, int[] order, float[] x, float[] route, float[] upScales)
    {
        using var input = new Tensor(allocator, DType.Float32, 1, InDim);
        input.SetElementsAsFloat(x);
        using var result = new Tensor(allocator, DType.Float32, 1, OutDim);
        GgmlBasicOps.MoEExpertsForward(result, input, order.Length, experts.UpPtrs(order), experts.DownPtrs(order),
            0, InDim, Ff, InDim * Ff * sizeof(float),
            0, Ff, OutDim, Ff * OutDim * sizeof(float),
            route, upScales);
        return result.GetElementsAsFloat(OutDim);
    }

    private static float[] RunChunked(IAllocator allocator, Experts experts, int[] order, float[] x, float[] route,
        float[] upScales, int maxPerCall)
    {
        using var input = new Tensor(allocator, DType.Float32, 1, InDim);
        input.SetElementsAsFloat(x);
        using var scratch = new Tensor(allocator, DType.Float32, 1, OutDim);
        var accum = new float[OutDim];
        var chunkUp = new IntPtr[NemotronModel.MoEExpertsPerKernelCall];
        var chunkDown = new IntPtr[NemotronModel.MoEExpertsPerKernelCall];
        var chunkRoute = new float[NemotronModel.MoEExpertsPerKernelCall];
        var chunkScales = new float[NemotronModel.MoEExpertsPerKernelCall];
        fixed (float* acc = accum)
        {
            NemotronModel.RunMoEExpertsChunked(scratch, input, acc, OutDim, order.Length,
                experts.UpPtrs(order), experts.DownPtrs(order), route, upScales,
                chunkUp, chunkDown, chunkRoute, chunkScales,
                experts.UpW[order[0]], experts.DownW[order[0]], maxPerCall);
        }
        return accum;
    }

    private static void AssertNear(double[] expected, float[] actual, double relTolerance)
    {
        double scale = Math.Max(1e-6, expected.Max(v => Math.Abs(v)));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"output {i} is {actual[i]}");
            Assert.True(Math.Abs(expected[i] - actual[i]) <= relTolerance * scale,
                $"output {i}: expected {expected[i]:R}, actual {actual[i]:R} (scale {scale:R})");
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void UpScaleBeforeReluSquared_NoOverflow()
    {
        // Expert 0's raw up weights are 2^66 too large and its up scale is 2^-66, the
        // NVFP4 situation in miniature: squared before scaling, its activations
        // overflow float. Expert 1's up scale is -1: a scale applied after ReLU^2 gets
        // its sign wrong.
        var allocator = CpuAllocator();
        float big = MathF.Pow(2, 66);
        using var experts = new Experts(4, seed: 11, e => e == 0 ? big : 1f);
        int[] order = { 0, 1, 2, 3 };
        float[] x = Input(5);
        float[] route = { 0.4f, 0.3f, 0.2f, 0.1f };
        float[] upScales = { 1f / big, -1f, 0.5f, 2f };

        float[] actual = RunKernel(allocator, experts, order, x, route, upScales);
        AssertNear(Reference(experts, order, x, route, upScales), actual, 1e-4);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NullUpScalesEqualsUnitScales()
    {
        var allocator = CpuAllocator();
        using var experts = new Experts(5, seed: 3);
        int[] order = { 4, 0, 2, 1, 3 };
        float[] x = Input(9);
        float[] route = { 0.1f, 0.2f, 0.3f, 0.25f, 0.15f };

        float[] none = RunKernel(allocator, experts, order, x, route, upScales: null);
        float[] ones = RunKernel(allocator, experts, order, x, route, Enumerable.Repeat(1f, order.Length).ToArray());
        Assert.Equal(none, ones);
        AssertNear(Reference(experts, order, x, route, null), none, 1e-4);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void RejectsMoreThan16ExpertsPerCall()
    {
        var allocator = CpuAllocator();
        using var experts = new Experts(17, seed: 7);
        int[] order = Enumerable.Range(0, 17).ToArray();
        var ex = Assert.ThrowsAny<Exception>(() =>
            RunKernel(allocator, experts, order, Input(1), Enumerable.Repeat(0.05f, 17).ToArray(), null));
        Assert.Contains("1..16", ex.Message);
        Assert.Equal(16, NemotronModel.MoEExpertsPerKernelCall);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void ChunkedHelper_22ExpertsMatchReference_AndScalesTravelWithTheirExperts()
    {
        // Super VL's top-22 is a 16 + 6 split. Distinct per-expert up scales (one of
        // them negative) and route weights; permuting the route order moves experts
        // across the chunk boundary and must not change the result.
        var allocator = CpuAllocator();
        using var experts = new Experts(30, seed: 21);
        var rnd = new Random(4);
        int[] order = Enumerable.Range(0, 30).OrderBy(_ => rnd.Next()).Take(22).ToArray();
        float[] route = order.Select(e => 0.02f + 0.01f * (e % 7)).ToArray();
        float[] upScales = order.Select(e => e % 5 == 0 ? -0.5f : 0.25f * (1 + e % 4)).ToArray();
        float[] x = Input(13);

        float[] actual = RunChunked(allocator, experts, order, x, route, upScales, NemotronModel.MoEExpertsPerKernelCall);
        AssertNear(Reference(experts, order, x, route, upScales), actual, 1e-4);

        int[] permutation = Enumerable.Range(0, 22).Reverse().ToArray();
        float[] permuted = RunChunked(allocator, experts,
            permutation.Select(i => order[i]).ToArray(), x,
            permutation.Select(i => route[i]).ToArray(),
            permutation.Select(i => upScales[i]).ToArray(), NemotronModel.MoEExpertsPerKernelCall);
        for (int i = 0; i < OutDim; i++)
            Assert.True(MathF.Abs(actual[i] - permuted[i]) <= 1e-4f * Math.Max(1f, MathF.Abs(actual[i])),
                $"output {i}: {actual[i]:R} vs permuted {permuted[i]:R}");
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void ChunkedHelper_SmallChunksEqualOneCall()
    {
        var allocator = CpuAllocator();
        using var experts = new Experts(16, seed: 8);
        int[] order = Enumerable.Range(0, 16).ToArray();
        float[] route = order.Select(e => 1f / 16f + 0.001f * e).ToArray();
        float[] upScales = order.Select(e => 0.5f + 0.125f * (e % 3)).ToArray();
        float[] x = Input(2);

        float[] oneCall = RunChunked(allocator, experts, order, x, route, upScales, maxPerCall: 16);
        float[] fiveAtATime = RunChunked(allocator, experts, order, x, route, upScales, maxPerCall: 5);   // 5+5+5+1
        Assert.Equal(RunKernel(allocator, experts, order, x, route, upScales), oneCall);
        for (int i = 0; i < OutDim; i++)
            Assert.True(MathF.Abs(oneCall[i] - fiveAtATime[i]) <= 1e-5f * Math.Max(1f, MathF.Abs(oneCall[i])),
                $"output {i}: {oneCall[i]:R} vs chunked {fiveAtATime[i]:R}");
    }

    [Fact]
    public void FoldExpertDownScale_MultipliesEachRouteByItsExpertsScale()
    {
        float[] downScale = { 0.5f, 2f, 4f, 1f / 16f };
        int[] selected = { 3, 1, 0, 2, 1 };
        float[] weights = { 1f, 0.25f, 0.5f, 0.125f, 3f };
        NemotronModel.FoldExpertDownScale(downScale, selected, weights, count: 4);
        Assert.Equal(new[] { 1f / 16f, 0.5f, 0.25f, 0.5f, 3f }, weights);   // the fifth route is past count

        float[] untouched = { 1f, 2f };
        NemotronModel.FoldExpertDownScale(null, new[] { 0, 1 }, untouched, 2);
        Assert.Equal(new[] { 1f, 2f }, untouched);
    }
}
