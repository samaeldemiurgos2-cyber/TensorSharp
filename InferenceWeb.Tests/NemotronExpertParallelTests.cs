using System;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NemotronExpertParallelTests
{
    // ---- Route table ------------------------------------------------------------

    [Fact]
    public void RouteTable_KeepsOwnedRoutes_AndNeutralisesTheRestWithDistinctLocalIds()
    {
        // Two tokens, top-3, 8 experts over 2 ranks: rank 1 owns experts 4..7.
        int[] selected = { 5, 1, 7, /* token 1 */ 0, 2, 3 };
        float[] weights = { 0.5f, 0.3f, 0.2f, 0.6f, 0.3f, 0.1f };

        NemotronModel.BuildExpertParallelRoutes(selected, weights, seqLen: 2, nUsed: 3, first: 4, perRank: 4,
            upScale: null, out int[] ids, out float[] w, out float[] scales);

        Assert.Null(scales);
        // Token 0: experts 5 and 7 are local 1 and 3; expert 1 is someone else's.
        Assert.Equal(1, ids[0]);
        Assert.Equal(3, ids[2]);
        Assert.Equal(new[] { 0.5f, 0f, 0.2f }, w.Take(3).ToArray());
        // Token 1 owns nothing here: every slot is weight 0, and the ids stay distinct.
        Assert.All(w.Skip(3), x => Assert.Equal(0f, x));
        for (int t = 0; t < 2; t++)
        {
            var tokenIds = ids.Skip(t * 3).Take(3).ToArray();
            Assert.Equal(tokenIds.Length, tokenIds.Distinct().Count());
            Assert.All(tokenIds, id => Assert.InRange(id, 0, 3));
        }
    }

    [Fact]
    public void RouteTable_GivesEveryRoute_TheUpScaleOfTheExpertItRuns()
    {
        int[] selected = { 5, 1, 7 };
        float[] weights = { 0.5f, 0.3f, 0.2f };
        float[] upScale = Enumerable.Range(0, 8).Select(e => 1e-5f * (e + 1)).ToArray();

        NemotronModel.BuildExpertParallelRoutes(selected, weights, seqLen: 1, nUsed: 3, first: 4, perRank: 4,
            upScale, out int[] ids, out _, out float[] scales);

        // Owned and neutralised routes alike: an unscaled weight-0 route could overflow, and Inf * 0 is NaN.
        for (int i = 0; i < 3; i++)
            Assert.Equal(upScale[4 + ids[i]], scales[i]);
    }

    // ---- Kernel ----------------------------------------------------------------

    private const int Hidden = 16;
    private const int NFf = 24;
    private const int Experts = 4;
    private const int NUsed = 2;
    private const int SeqLen = 3;

    [GgmlFact(BackendType.GgmlCpu)]
    public unsafe void RouteUpScales_ScaleTheUpProjectionBeforeReluSquared()
    {
        var rng = new Random(1234);
        float[] Rand(int n, float scale) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();

        // Stacked F32 experts in GGUF layout: up [Hidden, NFf, Experts], down [NFf, Hidden, Experts].
        float[] up = Rand(Hidden * NFf * Experts, 0.5f);
        float[] down = Rand(NFf * Hidden * Experts, 0.5f);
        float[] x = Rand(SeqLen * Hidden, 1f);
        int[] ids = { 0, 2, 1, 3, 3, 0 };
        float[] w = { 0.7f, 0.3f, 0.4f, 0.6f, 0.9f, 0.1f };
        // Scales of very different sizes, including the NVFP4 magnitude.
        float[] s = { 1e-5f, 0.5f, 2f, 1f, 3e-3f, 1.5f };

        float[] expected = Reference(x, up, down, ids, w, s);

        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
        void* upPtr = NativeMemory.AlignedAlloc((nuint)(up.Length * 4), 64);
        void* downPtr = NativeMemory.AlignedAlloc((nuint)(down.Length * 4), 64);
        try
        {
            up.AsSpan().CopyTo(new Span<float>(upPtr, up.Length));
            down.AsSpan().CopyTo(new Span<float>(downPtr, down.Length));
            using var input = new Tensor(allocator, DType.Float32, SeqLen, Hidden);
            input.SetElementsAsFloat(x);
            using var output = new Tensor(allocator, DType.Float32, SeqLen, Hidden);

            GgmlBasicOps.MoEFFNPrefill(input, output, SeqLen, Hidden, NFf, Experts, NUsed, ids, w,
                (IntPtr)upPtr, (int)GgmlTensorType.F32, Hidden, NFf, (long)up.Length * 4,
                IntPtr.Zero, 0, 0, 0, 0,
                (IntPtr)downPtr, (int)GgmlTensorType.F32, NFf, Hidden, (long)down.Length * 4,
                gateBias: null, upBias: null, downBias: null,
                activation: GgmlBasicOps.MoEActivation.ReluSquared,
                routeUpScales: s);

            float[] actual = output.GetElementsAsFloat(SeqLen * Hidden);
            float scale = expected.Max(MathF.Abs);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-4f * Math.Max(1f, scale),
                    $"element {i}: expected {expected[i]}, got {actual[i]}");

            // Without the scales the result differs: the hook is not a no-op.
            using var unscaled = new Tensor(allocator, DType.Float32, SeqLen, Hidden);
            GgmlBasicOps.MoEFFNPrefill(input, unscaled, SeqLen, Hidden, NFf, Experts, NUsed, ids, w,
                (IntPtr)upPtr, (int)GgmlTensorType.F32, Hidden, NFf, (long)up.Length * 4,
                IntPtr.Zero, 0, 0, 0, 0,
                (IntPtr)downPtr, (int)GgmlTensorType.F32, NFf, Hidden, (long)down.Length * 4,
                gateBias: null, upBias: null, downBias: null,
                activation: GgmlBasicOps.MoEActivation.ReluSquared);
            float[] plain = unscaled.GetElementsAsFloat(SeqLen * Hidden);
            Assert.Contains(Enumerable.Range(0, plain.Length), i => MathF.Abs(plain[i] - actual[i]) > 1e-2f);
        }
        finally
        {
            GgmlBasicOps.ClearHostBufferCache();
            NativeMemory.AlignedFree(upPtr);
            NativeMemory.AlignedFree(downPtr);
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void RouteUpScales_AreRefusedForAGatedActivation()
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0);
        using var input = new Tensor(allocator, DType.Float32, 1, Hidden);
        using var output = new Tensor(allocator, DType.Float32, 1, Hidden);
        var ex = Assert.Throws<ArgumentException>(() =>
            GgmlBasicOps.MoEFFNPrefill(input, output, 1, Hidden, NFf, Experts, 1, new[] { 0 }, new[] { 1f },
                (IntPtr)1, 0, Hidden, NFf, 4, (IntPtr)1, 0, Hidden, NFf, 4, (IntPtr)1, 0, NFf, Hidden, 4,
                null, null, null, GgmlBasicOps.MoEActivation.SwiGLUSplit, routeUpScales: new[] { 1f }));
        Assert.Equal("routeUpScales", ex.ParamName);
    }

    /// <summary>out[t] = sum_k w[t,k] * Down_e( relu(s[t,k] * Up_e x_t)^2 ), e = ids[t,k].</summary>
    private static float[] Reference(float[] x, float[] up, float[] down, int[] ids, float[] w, float[] s)
    {
        var result = new float[SeqLen * Hidden];
        for (int t = 0; t < SeqLen; t++)
            for (int k = 0; k < NUsed; k++)
            {
                int route = t * NUsed + k, e = ids[route];
                var act = new float[NFf];
                for (int f = 0; f < NFf; f++)
                {
                    // up[e] is [NFf rows][Hidden cols] in memory (ne0 = Hidden fastest).
                    double z = 0;
                    for (int h = 0; h < Hidden; h++)
                        z += up[((long)e * NFf + f) * Hidden + h] * x[t * Hidden + h];
                    double v = Math.Max(0, s[route] * z);
                    act[f] = (float)(v * v);
                }
                for (int o = 0; o < Hidden; o++)
                {
                    double y = 0;
                    for (int f = 0; f < NFf; f++)
                        y += down[((long)e * Hidden + o) * NFf + f] * act[f];
                    result[t * Hidden + o] += (float)(w[route] * y);
                }
            }
        return result;
    }
}
