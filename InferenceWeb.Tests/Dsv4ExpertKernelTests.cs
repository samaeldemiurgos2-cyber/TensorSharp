// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Runtime;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The direct-CUDA engine's tensor-core expert projection (ts_dsv4_moe_mma_f32), held to ggml's
/// own dequantization of the same bytes. The routing is built so that one expert has no member
/// tokens, one has more members than a single pass holds, and gate and up share a launch.
/// The low-bit IQ layouts also run at DeepSeek V4.1 1017's expert widths: 2304 (the down
/// projection's input), whose odd rows start two bytes off a 4-byte boundary, and 5120 (gate/up's),
/// there with the whole matrix moved two bytes off one.
/// </summary>
public sealed class Dsv4ExpertKernelTests
{
    private const int Experts = 5;
    private const int Rows = 160;      // one full 128-row tile and a partial one
    private const int InDim = 512;     // four k-steps, two K-quant super-blocks
    private const int Tokens = 40;
    private const int Used = 2;

    // The kernel's body is guarded by __CUDA_ARCH__ >= 800: the committed compute_75 baseline (a
    // build without nvcc) and a build for an sm_7x GPU compile it to nothing, and the engine
    // never launches it there (HasMoeMma), so the test skips rather than read unwritten output.
    [CudaTheory(TensorCoreModule = "tensorsharp_dsv4_kernels")]
    [InlineData((int)GgmlTensorType.Q2_K)]
    [InlineData((int)GgmlTensorType.Q3_K)]
    [InlineData((int)GgmlTensorType.Q4_K)]
    [InlineData((int)GgmlTensorType.Q5_K)]
    [InlineData((int)GgmlTensorType.Q6_K)]
    [InlineData((int)GgmlTensorType.Q8_0)]
    [InlineData((int)GgmlTensorType.IQ3_S)]
    [InlineData((int)GgmlTensorType.IQ4_NL)]
    [InlineData((int)GgmlTensorType.IQ4_XS)]
    [InlineData((int)GgmlTensorType.Q5_1)]
    [InlineData((int)GgmlTensorType.Q2_K, 2304)]
    [InlineData((int)GgmlTensorType.IQ1_S)]
    [InlineData((int)GgmlTensorType.IQ1_S, 2304)]
    [InlineData((int)GgmlTensorType.IQ1_S, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ2_S)]
    [InlineData((int)GgmlTensorType.IQ2_S, 2304)]
    [InlineData((int)GgmlTensorType.IQ2_S, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ2_XXS)]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 2304)]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ2_XS)]
    [InlineData((int)GgmlTensorType.IQ2_XS, 2304)]
    [InlineData((int)GgmlTensorType.IQ2_XS, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ3_XXS)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 2304)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 5120, 2)]
    public void TheTensorCoreExpertProjection_MatchesTheDequantizedWeights(int ggmlType, int inDim = InDim, int weightOffset = 0)
    {
        long rowBytes = NativeDequant.RowSize(ggmlType, inDim);
        byte[] gate = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 1);
        byte[] up = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 2);

        var act = new float[Tokens, inDim];
        for (int t = 0; t < Tokens; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.013f * (k + 1) * (t + 3)) + 0.25f * MathF.Cos(0.071f * k - t);

        // Every token takes expert 1 (40 members: two passes of 32) and one of 0, 2, 4; expert 3
        // is never chosen.
        var sel = new int[Tokens * Used];
        for (int t = 0; t < Tokens; t++)
        {
            sel[t * Used] = 1;
            sel[t * Used + 1] = new[] { 0, 2, 4 }[t % 3];
        }
        var counts = new int[Experts];
        foreach (int e in sel)
            counts[e]++;
        var offsets = new int[Experts];
        for (int e = 1; e < Experts; e++)
            offsets[e] = offsets[e - 1] + counts[e - 1];
        var cursor = (int[])offsets.Clone();
        var slotToken = new int[sel.Length];
        for (int s = 0; s < sel.Length; s++)
            slotToken[cursor[sel[s]]++] = s / Used;
        Assert.Equal(0, counts[3]);
        Assert.True(counts[1] > 32);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var kernels = Dsv4Kernels.Create();
        Assert.True(kernels.HasMoeMma, "The gate read an sm_80+ module, but the loaded one disables its tensor-core expert kernel.");
        using var gateW = Up(allocator, Shifted(gate, weightOffset));
        using var upW = Up(allocator, Shifted(up, weightOffset));
        using var actT = Up(allocator, act);
        using var countsT = Up(allocator, counts);
        using var offsetsT = Up(allocator, offsets);
        using var slotTokenT = Up(allocator, slotToken);
        using var gateOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);
        using var upOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);

        kernels.MoeMma(Dsv4CudaEngine.Ptr(gateW) + weightOffset, Dsv4CudaEngine.Ptr(upW) + weightOffset, actT, countsT,
            offsetsT, slotTokenT, gateOut, upOut, ggmlType, Rows, inDim, rowBytes, Experts, allocator.Stream.Handle);
        allocator.Stream.Synchronize();

        float[] tokenRow(int s) => Row(act, slotToken[s]);
        AssertMatches("gate", gate, ggmlType, rowBytes, inDim, counts, offsets, tokenRow, Down(gateOut, sel.Length * Rows));
        AssertMatches("up", up, ggmlType, rowBytes, inDim, counts, offsets, tokenRow, Down(upOut, sel.Length * Rows));

        // The down projection's shape: activations already in slot order, one weight set.
        var slotAct = new float[sel.Length, inDim];
        for (int s = 0; s < sel.Length; s++)
            for (int k = 0; k < inDim; k++)
                slotAct[s, k] = 0.5f * MathF.Cos(0.021f * (k + 7) * (s + 1));
        using var slotActT = Up(allocator, slotAct);
        using var downOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);
        kernels.MoeMma(Dsv4CudaEngine.Ptr(gateW) + weightOffset, IntPtr.Zero, slotActT, countsT, offsetsT, null,
            downOut, null, ggmlType, Rows, inDim, rowBytes, Experts, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        AssertMatches("slot-ordered", gate, ggmlType, rowBytes, inDim, counts, offsets, s => Row(slotAct, s),
            Down(downOut, sel.Length * Rows));
    }

    /// <summary>
    /// The decode projection (ts_dsv4_moe_gateup_decode_f32) for several tokens at once, as a
    /// batched decode step runs it: every (token, selected expert) slot reads its own token's q8_1
    /// row, and matches the dequantized weights against that row's q8_1 rounding.
    /// </summary>
    [CudaTheory]
    [InlineData((int)GgmlTensorType.Q2_K)]
    [InlineData((int)GgmlTensorType.Q3_K)]
    [InlineData((int)GgmlTensorType.Q4_K)]
    [InlineData((int)GgmlTensorType.Q5_K)]
    [InlineData((int)GgmlTensorType.Q6_K)]
    [InlineData((int)GgmlTensorType.Q8_0)]
    [InlineData((int)GgmlTensorType.IQ3_S)]
    [InlineData((int)GgmlTensorType.IQ4_NL)]
    [InlineData((int)GgmlTensorType.IQ4_XS)]
    [InlineData((int)GgmlTensorType.Q5_1)]
    [InlineData((int)GgmlTensorType.Q2_K, 2304)]
    [InlineData((int)GgmlTensorType.IQ1_S)]
    [InlineData((int)GgmlTensorType.IQ1_S, 2304)]
    [InlineData((int)GgmlTensorType.IQ1_S, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ2_S)]
    [InlineData((int)GgmlTensorType.IQ2_S, 2304)]
    [InlineData((int)GgmlTensorType.IQ2_S, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ2_XXS)]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 2304)]
    [InlineData((int)GgmlTensorType.IQ2_XXS, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ2_XS)]
    [InlineData((int)GgmlTensorType.IQ2_XS, 2304)]
    [InlineData((int)GgmlTensorType.IQ2_XS, 5120, 2)]
    [InlineData((int)GgmlTensorType.IQ3_XXS)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 2304)]
    [InlineData((int)GgmlTensorType.IQ3_XXS, 5120, 2)]
    public void TheDecodeExpertProjection_ServesEveryTokenOfAStep(int ggmlType, int inDim = InDim, int weightOffset = 0)
    {
        const int nt = 3;
        long rowBytes = NativeDequant.RowSize(ggmlType, inDim);
        byte[] gate = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 3);
        byte[] up = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 4);
        var act = new float[nt, inDim];
        for (int t = 0; t < nt; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(0.017f * (k + 5) * (t + 1)) - 0.3f * MathF.Cos(0.043f * k + t);
        int[] sel = { 4, 0, 2, 4, 1, 3 };   // [nt, Used]; expert 4 serves two tokens

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var kernels = Dsv4Kernels.Create();
        using var gateW = Up(allocator, Shifted(gate, weightOffset));
        using var upW = Up(allocator, Shifted(up, weightOffset));
        using var actT = Up(allocator, act);
        using var actQ = new Tensor(allocator, DType.UInt8, nt * (inDim / 32) * 36L);
        using var selT = Up(allocator, sel);
        using var gateOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);
        using var upOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);
        allocator.Kernels.LaunchQuantizeQ81Rows(Dsv4CudaEngine.Ptr(actT), Dsv4CudaEngine.Ptr(actQ), inDim, nt,
            allocator.Stream.Handle, warpCooperative: true);
        kernels.MoeGateUpDecode(Dsv4CudaEngine.Ptr(gateW) + weightOffset, Dsv4CudaEngine.Ptr(upW) + weightOffset, actQ, selT,
            gateOut, upOut, ggmlType, Rows, inDim, rowBytes, nt, Used, allocator.Stream.Handle);
        allocator.Stream.Synchronize();

        float[,] rounded = QuantizeDequantizeQ8_1(act);
        float[] gotGate = Down(gateOut, sel.Length * Rows);
        float[] gotUp = Down(upOut, sel.Length * Rows);
        var w = new float[inDim];
        for (int s = 0; s < sel.Length; s++)
        {
            float[] a = Row(rounded, s / Used);
            foreach ((byte[] weights, float[] got, string what) in new[] { (gate, gotGate, "gate"), (up, gotUp, "up") })
            {
                for (int r = 0; r < Rows; r++)
                {
                    NativeDequant.DequantizeToFloat32(ggmlType, weights, (int)(((long)sel[s] * Rows + r) * rowBytes), w, 0, inDim);
                    double sum = 0, mag = 0;
                    for (int k = 0; k < inDim; k++)
                    {
                        sum += (double)w[k] * a[k];
                        mag += Math.Abs((double)w[k] * a[k]);
                    }
                    double err = Math.Abs(got[(long)s * Rows + r] - sum);
                    Assert.True(err <= 1e-3 * mag + 1e-6,
                        $"{what}: slot {s} (token {s / Used}, expert {sel[s]}) row {r}: {got[(long)s * Rows + r]} vs {sum}");
                }
            }
        }
    }

    /// <summary>q8_1 as the quantize kernel writes it: per 32 values, d = amax / 127 stored as F16,
    /// round to nearest even.</summary>
    internal static float[,] QuantizeDequantizeQ8_1(float[,] input)
    {
        int rows = input.GetLength(0), cols = input.GetLength(1);
        var result = new float[rows, cols];
        for (int r = 0; r < rows; r++)
        {
            for (int block = 0; block < cols; block += 32)
            {
                float amax = 0;
                for (int i = 0; i < 32; i++)
                    amax = MathF.Max(amax, MathF.Abs(input[r, block + i]));
                float d = amax > 0 ? amax / 127.0f : 0;
                float id = d > 0 ? 1.0f / d : 0;
                float stored = (float)(System.Half)d;
                for (int i = 0; i < 32; i++)
                {
                    int q = Math.Clamp((int)MathF.Round(input[r, block + i] * id, MidpointRounding.ToEven), -127, 127);
                    result[r, block + i] = stored * q;
                }
            }
        }
        return result;
    }

    /// <summary>Each member slot's output against the dequantized expert weights, with both operands
    /// rounded to F16 as the kernel stages them, so what remains is accumulation order.</summary>
    private static void AssertMatches(string what, byte[] weights, int ggmlType, long rowBytes, int inDim,
        int[] counts, int[] offsets, Func<int, float[]> activation, float[] got)
    {
        var w = new float[inDim];
        var wh = new double[inDim];
        for (int e = 0; e < Experts; e++)
        {
            // The members' activations, rounded to F16 once per expert rather than once per row.
            var acts = new double[counts[e]][];
            for (int i = 0; i < counts[e]; i++)
                acts[i] = Array.ConvertAll(activation(offsets[e] + i), v => (double)(float)(System.Half)v);
            for (int r = 0; r < Rows; r++)
            {
                NativeDequant.DequantizeToFloat32(ggmlType, weights, (int)(((long)e * Rows + r) * rowBytes), w, 0, inDim);
                for (int k = 0; k < inDim; k++)
                    wh[k] = (float)(System.Half)w[k];
                for (int i = 0; i < counts[e]; i++)
                {
                    int s = offsets[e] + i;
                    double[] a = acts[i];
                    double sum = 0, mag = 0;
                    for (int k = 0; k < inDim; k++)
                    {
                        double p = wh[k] * a[k];
                        sum += p;
                        mag += Math.Abs(p);
                    }
                    double err = Math.Abs(got[(long)s * Rows + r] - sum);
                    Assert.True(err <= 2e-3 * mag + 1e-6,
                        $"{what}: expert {e} row {r} member {i}: {got[(long)s * Rows + r]} vs {sum} (|w.a| {mag})");
                }
            }
        }
    }

    /// <summary>A tensor whose bytes are on the device: CudaStorage mirrors host writes lazily,
    /// and these kernels take raw device pointers.</summary>
    internal static Tensor Up(CudaAllocator allocator, Array values)
    {
        Tensor t = Tensor.FromArray(allocator, values);
        t.Storage.EnsureDeviceCurrent();
        return t;
    }

    /// <summary><paramref name="bytes"/> behind <paramref name="offset"/> zero bytes: uploaded and read
    /// from the device pointer plus the offset, a matrix whose rows sit off the allocation's alignment
    /// (a weight inside a GGUF mapping is placed by the file).</summary>
    internal static byte[] Shifted(byte[] bytes, int offset)
    {
        if (offset == 0)
            return bytes;
        var shifted = new byte[offset + bytes.Length];
        Buffer.BlockCopy(bytes, 0, shifted, offset, bytes.Length);
        return shifted;
    }

    /// <summary>Read back what a kernel wrote through a raw pointer.</summary>
    internal static float[] Down(Tensor t, int count)
    {
        ((CudaStorage)t.Storage).MarkDeviceModified();
        return t.GetElementsAsFloat(count);
    }

    internal static float[] Row(float[,] m, int r)
    {
        var row = new float[m.GetLength(1)];
        for (int k = 0; k < row.Length; k++)
            row[k] = m[r, k];
        return row;
    }

    /// <summary>Deterministic bytes for <paramref name="rows"/> quantized rows: every bit pattern is a
    /// valid encoding except the block scales (F16, or MXFP4's E8M0), which are set to small finite values.</summary>
    internal static byte[] QuantRows(int ggmlType, int rows, long rowBytes, int seed)
    {
        var bytes = new byte[rows * rowBytes];
        uint x = (uint)(0x9E3779B9 * seed);
        for (int i = 0; i < bytes.Length; i++)
        {
            x ^= x << 13; x ^= x >> 17; x ^= x << 5;
            bytes[i] = (byte)x;
        }
        (int blockBytes, int[] halves) = ggmlType switch
        {
            (int)GgmlTensorType.Q2_K => (84, new[] { 80, 82 }),
            (int)GgmlTensorType.Q3_K => (110, new[] { 108 }),
            (int)GgmlTensorType.Q4_K => (144, new[] { 0, 2 }),
            (int)GgmlTensorType.Q5_K => (176, new[] { 0, 2 }),
            (int)GgmlTensorType.Q6_K => (210, new[] { 208 }),
            (int)GgmlTensorType.Q8_0 => (34, new[] { 0 }),
            (int)GgmlTensorType.IQ3_S => (110, new[] { 0 }),
            (int)GgmlTensorType.IQ4_NL => (18, new[] { 0 }),
            (int)GgmlTensorType.IQ4_XS => (136, new[] { 0 }),
            (int)GgmlTensorType.Q5_1 => (24, new[] { 0 }),
            (int)GgmlTensorType.IQ1_S => (50, new[] { 0 }),
            (int)GgmlTensorType.IQ2_S => (82, new[] { 0 }),
            (int)GgmlTensorType.IQ2_XXS => (66, new[] { 0 }),
            (int)GgmlTensorType.IQ2_XS => (74, new[] { 0 }),
            (int)GgmlTensorType.IQ3_XXS => (98, new[] { 0 }),
            (int)GgmlTensorType.MXFP4 => (17, Array.Empty<int>()),
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType)),
        };
        for (int b = 0; b * blockBytes < bytes.Length; b++)
        {
            for (int h = 0; h < halves.Length; h++)
            {
                ushort bits = BitConverter.HalfToUInt16Bits((System.Half)(0.01f + 0.003f * ((b + h) % 7)));
                bytes[b * blockBytes + halves[h]] = (byte)bits;
                bytes[b * blockBytes + halves[h] + 1] = (byte)(bits >> 8);
            }
            if (ggmlType == (int)GgmlTensorType.Q5_1)
            {
                // The block's offset m: below zero as quantize_row_q5_1 leaves it for centred weights
                // (codes 0..31 around -16 d), and above it in every seventh block.
                float d = (float)BitConverter.UInt16BitsToHalf((ushort)(bytes[b * blockBytes] | bytes[b * blockBytes + 1] << 8));
                float m = b % 7 == 0 ? 3f * d : -(12f + b % 5) * d;
                ushort mBits = BitConverter.HalfToUInt16Bits((System.Half)m);
                bytes[b * blockBytes + 2] = (byte)mBits;
                bytes[b * blockBytes + 3] = (byte)(mBits >> 8);
            }
            // MXFP4's block scale is one E8M0 byte, 2^(e-127): small powers of two, not random exponents.
            if (ggmlType == (int)GgmlTensorType.MXFP4)
                bytes[b * blockBytes] = (byte)(119 + b % 5);
        }
        return bytes;
    }
}
