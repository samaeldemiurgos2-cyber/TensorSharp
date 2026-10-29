// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A quantized weight under 4 KiB is never looked up in the native device caches (the
// binders' raw_bytes >= 4096 gates): every call uploads it from the pointer the managed
// side passes, which after a ggml_cuda preload is the weight's opaque cache key, while the
// host copy is released as soon as the preload returns. addmm_quant resolved the key to
// those released bytes (an access violation in Nemotron's prefill on its Q8_0 [64 x 32]
// latent projection), and the MoE expert kernel uploaded from the key itself (NaN from
// the first --tp decode step). These tests overwrite the host bytes after the preload,
// as a release may, so a read of them sees NaN block scales, and a read of the key sees
// the handle's memory, instead of the weight.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class GgmlSmallWeightCacheKeyTests
{
    private const int Q8_0 = 8;
    private const int Hidden = 64, Inner = 32; // [64 x 32] and [32 x 64] Q8_0: 2176 bytes each

    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuAddmmQuantReadsASmallPreloadedWeightAfterItsHostCopyIsReleased() => AddmmQuant(GgmlBackendType.Cpu);
    [GgmlFact(BackendType.GgmlCuda)]
    public void CudaAddmmQuantReadsASmallPreloadedWeightAfterItsHostCopyIsReleased() => AddmmQuant(GgmlBackendType.Cuda);
    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuMoEExpertsForwardResolvesSmallExpertCacheKeys() => MoEExperts(GgmlBackendType.Cpu);
    [GgmlFact(BackendType.GgmlCuda)]
    public void CudaMoEExpertsForwardResolvesSmallExpertCacheKeys() => MoEExperts(GgmlBackendType.Cuda);

    private static void AddmmQuant(GgmlBackendType backend)
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        try
        {
            using var weight = new PreloadedWeight(Hidden, Inner, seed: 1);
            Assert.True(weight.Bytes < 4096);
            weight.ReleaseHostCopy();

            const int rows = 3;
            float[] x = Values(rows * Hidden, seed: 2, magnitude: 1f);
            using var input = new Tensor(allocator, DType.Float32, rows, Hidden);
            using var output = new Tensor(allocator, DType.Float32, rows, Inner);
            input.SetElementsAsFloat(x);
            GgmlBasicOps.AddmmQuant(output, input, weight.Key, Q8_0, Hidden, Inner, weight.Bytes);

            var expected = new float[rows * Inner];
            for (int r = 0; r < rows; r++)
                Array.Copy(weight.Apply(x.AsSpan(r * Hidden, Hidden)), 0, expected, r * Inner, Inner);
            AssertClose(expected, output.GetElementsAsFloat(rows * Inner));
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    private static void MoEExperts(GgmlBackendType backend)
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        try
        {
            // Nemotron's decode: per expert, down(relu(up(x))^2) times its route weight, summed.
            using var up0 = new PreloadedWeight(Hidden, Inner, seed: 3);
            using var up1 = new PreloadedWeight(Hidden, Inner, seed: 4);
            using var down0 = new PreloadedWeight(Inner, Hidden, seed: 5);
            using var down1 = new PreloadedWeight(Inner, Hidden, seed: 6);
            foreach (var w in new[] { up0, up1, down0, down1 })
                w.ReleaseHostCopy();
            float[] route = { 0.75f, -0.5f };

            float[] x = Values(Hidden, seed: 7, magnitude: 1f);
            using var input = new Tensor(allocator, DType.Float32, 1, Hidden);
            using var output = new Tensor(allocator, DType.Float32, 1, Hidden);
            input.SetElementsAsFloat(x);
            GgmlBasicOps.MoEExpertsForward(output, input, 2,
                new[] { up0.Key, up1.Key }, new[] { down0.Key, down1.Key },
                Q8_0, Hidden, Inner, up0.Bytes, Q8_0, Inner, Hidden, down0.Bytes, route);

            var expected = new float[Hidden];
            foreach (var (up, down, weight) in new[] { (up0, down0, route[0]), (up1, down1, route[1]) })
            {
                float[] hidden = up.Apply(x);
                for (int i = 0; i < hidden.Length; i++)
                    hidden[i] = hidden[i] > 0 ? hidden[i] * hidden[i] : 0;
                float[] y = down.Apply(hidden);
                for (int i = 0; i < Hidden; i++)
                    expected[i] += weight * y[i];
            }
            AssertClose(expected, output.GetElementsAsFloat(Hidden));
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    /// <summary>GGML rounds the activations to 8 bits for a Q8_0 matmul (q8_0 on cpu, q8_1 on
    /// cuda), which moves these sums by under 1% of their largest value; a wrong weight byte
    /// moves them by its own size or makes them NaN.</summary>
    private static void AssertClose(float[] expected, float[] actual)
    {
        float scale = 0;
        foreach (float v in expected)
            scale = MathF.Max(scale, MathF.Abs(v));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"output {i} is {actual[i]}");
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 0.02f * scale,
                $"output {i}: expected {expected[i]}, actual {actual[i]} (largest |expected| {scale})");
        }
    }

    private static float[] Values(int count, int seed, float magnitude)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = (float)(random.NextDouble() * 2 - 1) * magnitude;
        return values;
    }

    /// <summary>
    /// A Q8_0 [<c>inDim</c> -&gt; <c>outDim</c>] weight preloaded as a ggml_cuda model load does:
    /// host bytes in a buffer this test owns, the weight's GCHandle cache key, and
    /// TSGgml_PreloadQuantizedWeight under that key.
    /// </summary>
    private sealed class PreloadedWeight : IDisposable
    {
        private readonly int _inDim, _outDim;
        private readonly float[] _values; // the weight as stored, dequantized: [outDim, inDim]
        private readonly IntPtr _host;
        private readonly QuantizedWeight _weight;

        public PreloadedWeight(int inDim, int outDim, int seed)
        {
            _inDim = inDim;
            _outDim = outDim;
            byte[] raw = SyntheticGguf.QuantizeQ8_0(Values(inDim * outDim, seed, magnitude: 0.25f), inDim);
            Bytes = raw.Length;
            _host = Marshal.AllocHGlobal(raw.Length);
            Marshal.Copy(raw, 0, _host, raw.Length);
            _values = new float[inDim * outDim];
            unsafe
            {
                fixed (float* destination = _values)
                    ManagedQuantizedOps.DequantizeRowToFloat32(Q8_0, _host, destination, _values.Length);
            }

            _weight = QuantizedWeight.CreateExternalView(_host, Bytes, Q8_0, inDim, outDim, this);
            Key = _weight.EnsureDeviceCacheKey();
            Assert.NotEqual(_host, Key);
            Assert.True(GgmlBasicOps.PreloadQuantizedWeight(Key, _host, Q8_0, inDim, outDim, Bytes));
        }

        public IntPtr Key { get; }
        public long Bytes { get; }

        /// <summary>What the model load may leave behind once the preload returns: the bytes
        /// are freed or unmapped. 0xFF makes every block scale an F16 NaN.</summary>
        public void ReleaseHostCopy()
        {
            var garbage = new byte[Bytes];
            Array.Fill(garbage, (byte)0xFF);
            Marshal.Copy(garbage, 0, _host, garbage.Length);
        }

        /// <summary>The weight applied to <paramref name="x"/> in float.</summary>
        public float[] Apply(ReadOnlySpan<float> x)
        {
            var y = new float[_outDim];
            for (int o = 0; o < _outDim; o++)
            {
                float sum = 0;
                for (int i = 0; i < _inDim; i++)
                    sum += _values[o * _inDim + i] * x[i];
                y[o] = sum;
            }
            return y;
        }

        public void Dispose()
        {
            _weight.Dispose();
            Marshal.FreeHGlobal(_host);
        }
    }
}
