// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The low-bit IQ routed experts of unsloth's DeepSeek V4.1 1017 UD-IQ2_XXS on the direct-CUDA expert kernels
/// (<see cref="CudaMoe"/>): gate/up in IQ1_S, IQ2_S or IQ2_XXS, down mostly in IQ2_S, IQ3_XXS or IQ2_XS. Before
/// these decoders the kernels read IQ1_S and IQ2_S rows as Q8_0, and ran IQ2_XXS on the warp-per-row kernels only.
/// The managed dequantization is ggml's bit for bit; every route an expert matrix can take (per-token decode,
/// grouped warp, register-staged, tensor-core) is held to ggml's dequantization of the same bytes at the 1017's
/// expert widths, 5120 (gate/up input) and 2304 (down input), with the matrix two bytes off a 4-byte boundary;
/// then the expert block on its routes; and a kernel module that predates the decoders keeps the layouts off them.
/// </summary>
public sealed class CudaMoeLowBitIqExpertTests
{
    private const int IQ1_S = (int)GgmlTensorType.IQ1_S;
    private const int IQ2_S = (int)GgmlTensorType.IQ2_S;
    private const int IQ2_XXS = (int)GgmlTensorType.IQ2_XXS;
    private const int IQ2_XS = (int)GgmlTensorType.IQ2_XS;
    private const int IQ3_XXS = (int)GgmlTensorType.IQ3_XXS;
    private const int Experts = 5;
    private const int Tokens = 40;
    private const int Used = 2;
    private const int Rows = 100;   // ends mid-tile for the staged (16-row) and tensor-core (128-row) kernels
    private const int Shift = 2;    // the weight bytes start two bytes off a 4-byte boundary

    private readonly ITestOutputHelper _output;

    public CudaMoeLowBitIqExpertTests(ITestOutputHelper output) => _output = output;

    // ---------------------------------------------------------------- the reference

    /// <summary>The managed dequantization (the host paths' reference) is ggml's to_float bit for bit, on
    /// random blocks: every grid index, sign pattern, scale and IQ1_S delta sign.</summary>
    [Theory]
    [InlineData(IQ1_S)]
    [InlineData(IQ2_S)]
    [InlineData(IQ2_XXS)]
    [InlineData(IQ2_XS)]
    [InlineData(IQ3_XXS)]
    public void TheManagedDequantization_IsGgmlsBitForBit(int ggmlType)
    {
        const int n = 2304, rows = 16;
        long rowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(ggmlType, n);
        byte[] bytes = QuantRows(ggmlType, rows, rowBytes, seed: 31);
        var managed = new float[n];
        var ggml = new float[n];
        for (int r = 0; r < rows; r++)
        {
            ManagedQuantizedOps.DequantizeToFloat32(ggmlType, bytes, checked((int)(r * rowBytes)), managed, 0, n);
            GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, bytes, checked((int)(r * rowBytes)), ggml, 0, n);
            for (int k = 0; k < n; k++)
            {
                if (BitConverter.SingleToInt32Bits(managed[k]) != BitConverter.SingleToInt32Bits(ggml[k]))
                    Assert.Fail($"{(GgmlTensorType)ggmlType} row {r} value {k}: managed {managed[k]} vs ggml {ggml[k]}");
            }
            Assert.Contains(ggml, v => v < 0);
            Assert.Contains(ggml, v => v > 0);
        }
    }

    /// <summary>
    /// The kernels decode an IQ1_S group as the integers 8 g +/- 1 under the scale d * (2s + 1) / 8, which
    /// needs no activation sum. Their product is ggml's value exactly (not merely to rounding), so the
    /// tensor-core tile's F16 weight is the F16 of ggml's: checked here value by value over every grid entry
    /// a random row reaches, both delta signs and all eight scales.
    /// </summary>
    [Fact]
    public void TheIq1SIntegerForm_IsGgmlsValueExactly()
    {
        const int n = 2304, rows = 32;
        long rowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(IQ1_S, n);
        Assert.Equal(n / 256 * 50, rowBytes);
        byte[] bytes = QuantRows(IQ1_S, rows, rowBytes, seed: 41);
        var ggml = new float[n];
        var ints = new HashSet<int>();
        var scales = new HashSet<int>();
        for (int r = 0; r < rows; r++)
        {
            GgmlGgufTensorDequant.DequantizeToFloat32(IQ1_S, bytes, checked((int)(r * rowBytes)), ggml, 0, n);
            for (int b = 0; b < n / 256; b++)
            {
                int at = checked((int)(r * rowBytes)) + b * 50;
                float d = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(bytes, at));
                for (int g = 0; g < 8; g++)
                {
                    int qh = BitConverter.ToUInt16(bytes, at + 34 + 2 * g);
                    int s = (qh >> 12) & 7;
                    float sc = d * (2 * s + 1) * 0.125f;
                    int bias = (qh & 0x8000) != 0 ? -1 : 1;
                    scales.Add(s);
                    for (int l = 0; l < 4; l++)
                    {
                        ulong grid = IQuantGrids.iq1s_grid[bytes[at + 2 + 4 * g + l] | (((qh >> (3 * l)) & 7) << 8)];
                        for (int j = 0; j < 8; j++)
                        {
                            int w8 = 8 * (sbyte)(byte)(grid >> (8 * j)) + bias;
                            ints.Add(w8);
                            float want = ggml[b * 256 + g * 32 + l * 8 + j];
                            Assert.Equal(BitConverter.SingleToInt32Bits(want), BitConverter.SingleToInt32Bits(sc * w8));
                        }
                    }
                }
            }
        }
        Assert.Equal(new[] { -9, -7, -1, 1, 7, 9 }, ints.OrderBy(v => v));
        Assert.Equal(Enumerable.Range(0, 8), scales.OrderBy(v => v));
    }

    /// <summary>The vendored GPU grid (tensorsharp_iq1s_tables.cuh, ggml's iq1s_grid_gpu) is the transform of
    /// the managed CPU grid every IQ1_S reference here reads: byte j = (v[j] + 1) | ((v[j + 4] + 1) &lt;&lt; 4).</summary>
    [Fact]
    public void TheVendoredGpuGrid_IsTheTransformOfTheManagedGrid()
    {
        string text = File.ReadAllText(RepoFile("TensorSharp.Backends.Cuda", "native", "kernels", "tensorsharp_iq1s_tables.cuh"));
        int open = text.IndexOf("iq1s_grid_gpu[2048] = {", StringComparison.Ordinal);
        Assert.True(open >= 0, "iq1s_grid_gpu[2048] is not declared in tensorsharp_iq1s_tables.cuh");
        int close = text.IndexOf("};", open, StringComparison.Ordinal);
        uint[] gpu = Regex.Matches(text[open..close], @"0x([0-9a-fA-F]{8})\b")
            .Select(m => uint.Parse(m.Groups[1].Value, NumberStyles.HexNumber, CultureInfo.InvariantCulture)).ToArray();
        Assert.Equal(2048, gpu.Length);
        Assert.Equal(2048, IQuantGrids.iq1s_grid.Length);
        for (int i = 0; i < gpu.Length; i++)
        {
            ulong cpu = IQuantGrids.iq1s_grid[i];
            uint want = 0;
            for (int j = 0; j < 4; j++)
            {
                int lo = (sbyte)(byte)(cpu >> (8 * j)) + 1;
                int hi = (sbyte)(byte)(cpu >> (8 * (j + 4))) + 1;
                Assert.InRange(lo, 0, 2);
                Assert.InRange(hi, 0, 2);
                want |= (uint)(lo | (hi << 4)) << (8 * j);
            }
            Assert.True(want == gpu[i], $"iq1s_grid_gpu[{i}] is 0x{gpu[i]:x8}; the managed grid gives 0x{want:x8}");
        }
    }

    // ---------------------------------------------------------------- kernel routes

    /// <summary>
    /// The per-token decode kernels a decode step runs: gate/up for every (token, selected expert) slot from
    /// its token's q8_1 row, and down from each slot's own row. Held to the dequantized weights against each
    /// row's q8_1 rounding, within 1e-3 of the products' absolute sum.
    /// </summary>
    [CudaTheory]
    [InlineData(IQ1_S, 5120)]
    [InlineData(IQ1_S, 2304)]
    [InlineData(IQ2_S, 5120)]
    [InlineData(IQ2_S, 2304)]
    [InlineData(IQ2_XXS, 5120)]
    [InlineData(IQ2_XXS, 2304)]
    [InlineData(IQ2_XS, 2304)]
    [InlineData(IQ3_XXS, 2304)]
    public void TheDecodeRoutes_MatchTheDequantizedWeights(int ggmlType, int inDim)
    {
        int[] sel = { 4, 0, 2, 4, 1, 3 };   // three tokens' two experts; expert 4 serves two
        const int nt = 3;
        long rowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(ggmlType, inDim);
        byte[] gate = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 51);
        byte[] up = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 52);
        float[,] act = Activations(nt, inDim, 0.017f, 5);
        float[,] slotAct = Activations(sel.Length, inDim, 0.023f, 9);

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        IntPtr stream = allocator.Stream.Handle;
        using var gateW = Up(allocator, Shifted(gate, Shift));
        using var upW = Up(allocator, Shifted(up, Shift));
        using var actT = Up(allocator, act);
        using var slotActT = Up(allocator, slotAct);
        using var selT = Up(allocator, sel);
        using var actQ = new Tensor(allocator, DType.UInt8, (long)nt * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var slotQ = new Tensor(allocator, DType.UInt8, (long)sel.Length * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var gateOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);
        using var upOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);
        using var downOut = new Tensor(allocator, DType.Float32, sel.Length, Rows);
        Clear(allocator, gateOut, upOut, downOut);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(actT), Ptr(actQ), inDim, nt, stream, warpCooperative: true);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(slotActT), Ptr(slotQ), inDim, sel.Length, stream, warpCooperative: true);
        dk.MoeGateUpDecode(Ptr(gateW) + Shift, Ptr(upW) + Shift, actQ, selT, gateOut, upOut, ggmlType, Rows, inDim, rowBytes,
            nt, Used, stream);
        dk.MoeDownDecode(Ptr(gateW) + Shift, slotQ, selT, downOut, ggmlType, Rows, inDim, rowBytes, sel.Length, stream);
        allocator.Stream.Synchronize();

        float[][] gateRows = DequantRows(ggmlType, gate, rowBytes, inDim);
        float[][] upRows = DequantRows(ggmlType, up, rowBytes, inDim);
        float[,] rounded = QuantizeDequantizeQ8_1(act);
        float[,] slotRounded = QuantizeDequantizeQ8_1(slotAct);
        float[] gotGate = Down(gateOut, sel.Length * Rows);
        float[] gotUp = Down(upOut, sel.Length * Rows);
        float[] gotDown = Down(downOut, sel.Length * Rows);
        double worst = 0;
        for (int s = 0; s < sel.Length; s++)
        {
            float[] a = Row(rounded, s / Used);
            float[] sa = Row(slotRounded, s);
            for (int r = 0; r < Rows; r++)
            {
                float[] gw = gateRows[sel[s] * Rows + r];
                worst = Math.Max(worst, AssertDot($"gate slot {s} row {r}", gw, a, gotGate[s * Rows + r], 1e-3));
                worst = Math.Max(worst, AssertDot($"up slot {s} row {r}", upRows[sel[s] * Rows + r], a, gotUp[s * Rows + r], 1e-3));
                worst = Math.Max(worst, AssertDot($"down slot {s} row {r}", gw, sa, gotDown[s * Rows + r], 1e-3));
            }
        }
        _output.WriteLine($"{(GgmlTensorType)ggmlType} decode {inDim}x{Rows}: worst error {worst:E2} of |w.a|");
    }

    /// <summary>
    /// The grouped kernels a long ubatch runs, gate/up (each member slot reads its token's row through
    /// slotToken) and down (activations already in slot order): the warp-per-row kernels over interleaved
    /// q8_1 activations and the register-staged kernels over the split q8_1 layout. One expert has no
    /// members and one more than a 32-token pass. Same bound as the decode kernels.
    /// </summary>
    [CudaTheory]
    [InlineData(IQ1_S, 5120)]
    [InlineData(IQ1_S, 2304)]
    [InlineData(IQ2_S, 5120)]
    [InlineData(IQ2_S, 2304)]
    [InlineData(IQ2_XXS, 5120)]
    [InlineData(IQ2_XXS, 2304)]
    [InlineData(IQ2_XS, 2304)]
    [InlineData(IQ3_XXS, 2304)]
    public void TheGroupedWarpAndStagedRoutes_MatchTheDequantizedWeights(int ggmlType, int inDim)
    {
        Assert.True(Dsv4Kernels.StagedSupports(inDim));
        long rowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(ggmlType, inDim);
        byte[] gate = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 61);
        byte[] up = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 62);
        float[,] act = Activations(Tokens, inDim, 0.013f, 3);
        float[,] slotAct = Activations(Tokens * Used, inDim, 0.021f, 7);
        (int[] counts, int[] offsets, int[] slotToken) = Group();
        int slots = Tokens * Used;

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        IntPtr stream = allocator.Stream.Handle;
        using var gateW = Up(allocator, Shifted(gate, Shift));
        using var upW = Up(allocator, Shifted(up, Shift));
        using var actT = Up(allocator, act);
        using var slotActT = Up(allocator, slotAct);
        using var countsT = Up(allocator, counts);
        using var offsetsT = Up(allocator, offsets);
        using var slotTokenT = Up(allocator, slotToken);
        using var actQ = new Tensor(allocator, DType.UInt8, (long)Tokens * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var slotQ = new Tensor(allocator, DType.UInt8, (long)slots * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var actQs = new Tensor(allocator, DType.UInt8, (long)Tokens * inDim);
        using var actD = new Tensor(allocator, DType.Float32, (long)Tokens * (inDim / 32));
        using var slotQs = new Tensor(allocator, DType.UInt8, (long)slots * inDim);
        using var slotD = new Tensor(allocator, DType.Float32, (long)slots * (inDim / 32));
        using var gateOut = new Tensor(allocator, DType.Float32, slots, Rows);
        using var upOut = new Tensor(allocator, DType.Float32, slots, Rows);
        using var downOut = new Tensor(allocator, DType.Float32, slots, Rows);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(actT), Ptr(actQ), inDim, Tokens, stream, warpCooperative: true);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(slotActT), Ptr(slotQ), inDim, slots, stream, warpCooperative: true);
        allocator.Kernels.LaunchQuantizeQ81SplitRows(Ptr(actT), Ptr(actQs), Ptr(actD), inDim, Tokens, stream);
        allocator.Kernels.LaunchQuantizeQ81SplitRows(Ptr(slotActT), Ptr(slotQs), Ptr(slotD), inDim, slots, stream);

        float[][] gateRows = DequantRows(ggmlType, gate, rowBytes, inDim);
        float[][] upRows = DequantRows(ggmlType, up, rowBytes, inDim);
        float[,] rounded = QuantizeDequantizeQ8_1(act);
        float[,] slotRounded = QuantizeDequantizeQ8_1(slotAct);
        float[][] tokenRows = Enumerable.Range(0, slots).Select(s => Row(rounded, slotToken[s])).ToArray();
        float[][] slotRows = Enumerable.Range(0, slots).Select(s => Row(slotRounded, s)).ToArray();

        foreach (string route in new[] { "warp", "staged" })
        {
            Clear(allocator, gateOut, upOut, downOut);
            if (route == "warp")
            {
                dk.MoeGateUp(Ptr(gateW) + Shift, Ptr(upW) + Shift, actQ, countsT, offsetsT, slotTokenT, gateOut, upOut,
                    ggmlType, Rows, inDim, rowBytes, Experts, stream);
                dk.MoeDown(Ptr(gateW) + Shift, slotQ, countsT, offsetsT, downOut, ggmlType, Rows, inDim, rowBytes, Experts, stream);
            }
            else
            {
                dk.MoeGateUpStaged(Ptr(gateW) + Shift, Ptr(upW) + Shift, actQs, actD, countsT, offsetsT, slotTokenT,
                    gateOut, upOut, ggmlType, Rows, inDim, rowBytes, Experts, stream);
                dk.MoeDownStaged(Ptr(gateW) + Shift, slotQs, slotD, countsT, offsetsT, downOut, ggmlType, Rows, inDim,
                    rowBytes, Experts, stream);
            }
            allocator.Stream.Synchronize();
            double worst = 0;
            worst = Math.Max(worst, AssertGrouped($"{route} gate", gateRows, counts, offsets, tokenRows, Down(gateOut, slots * Rows), 1e-3));
            worst = Math.Max(worst, AssertGrouped($"{route} up", upRows, counts, offsets, tokenRows, Down(upOut, slots * Rows), 1e-3));
            worst = Math.Max(worst, AssertGrouped($"{route} down", gateRows, counts, offsets, slotRows, Down(downOut, slots * Rows), 1e-3));
            _output.WriteLine($"{(GgmlTensorType)ggmlType} {route} {inDim}x{Rows}: worst error {worst:E2} of |w.a|");
        }
    }

    /// <summary>
    /// The tensor-core kernel over the same grouping: weights decoded once per 128 x 128 tile into F16 and
    /// F16 activations, so the reference rounds both operands to F16 and what remains is accumulation order
    /// (within 2e-3 of the products' absolute sum, <see cref="Dsv4ExpertKernelTests"/>' bound). It needs the
    /// kernel's sm_80+ body, which a committed PTX baseline lacks: there it is skipped, saying so, and the
    /// staged route serves those builds.
    /// </summary>
    [CudaTheory(TensorCoreModule = "tensorsharp_dsv4_kernels")]
    [InlineData(IQ1_S, 5120)]
    [InlineData(IQ1_S, 2304)]
    [InlineData(IQ2_S, 5120)]
    [InlineData(IQ2_S, 2304)]
    [InlineData(IQ2_XXS, 5120)]
    [InlineData(IQ2_XXS, 2304)]
    public void TheTensorCoreRoute_MatchesTheDequantizedWeights(int ggmlType, int inDim)
    {
        long rowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(ggmlType, inDim);
        byte[] gate = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 71);
        byte[] up = QuantRows(ggmlType, Experts * Rows, rowBytes, seed: 72);
        float[,] act = Activations(Tokens, inDim, 0.013f, 3);
        float[,] slotAct = Activations(Tokens * Used, inDim, 0.021f, 7);
        (int[] counts, int[] offsets, int[] slotToken) = Group();
        int slots = Tokens * Used;

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.HasMoeMma, "The gate read an sm_80+ module, but the loaded one disables its tensor-core expert kernel.");
        IntPtr stream = allocator.Stream.Handle;
        using var gateW = Up(allocator, Shifted(gate, Shift));
        using var upW = Up(allocator, Shifted(up, Shift));
        using var actT = Up(allocator, act);
        using var slotActT = Up(allocator, slotAct);
        using var countsT = Up(allocator, counts);
        using var offsetsT = Up(allocator, offsets);
        using var slotTokenT = Up(allocator, slotToken);
        using var gateOut = new Tensor(allocator, DType.Float32, slots, Rows);
        using var upOut = new Tensor(allocator, DType.Float32, slots, Rows);
        using var downOut = new Tensor(allocator, DType.Float32, slots, Rows);
        Clear(allocator, gateOut, upOut, downOut);
        dk.MoeMma(Ptr(gateW) + Shift, Ptr(upW) + Shift, actT, countsT, offsetsT, slotTokenT, gateOut, upOut,
            ggmlType, Rows, inDim, rowBytes, Experts, stream);
        dk.MoeMma(Ptr(gateW) + Shift, IntPtr.Zero, slotActT, countsT, offsetsT, null, downOut, null,
            ggmlType, Rows, inDim, rowBytes, Experts, stream);
        allocator.Stream.Synchronize();

        float[][] ToF16(float[][] rows) => rows.Select(r => Array.ConvertAll(r, v => (float)(System.Half)v)).ToArray();
        float[][] gateRows = ToF16(DequantRows(ggmlType, gate, rowBytes, inDim));
        float[][] upRows = ToF16(DequantRows(ggmlType, up, rowBytes, inDim));
        float[][] tokenRows = ToF16(Enumerable.Range(0, slots).Select(s => Row(act, slotToken[s])).ToArray());
        float[][] slotRows = ToF16(Enumerable.Range(0, slots).Select(s => Row(slotAct, s)).ToArray());
        double worst = 0;
        worst = Math.Max(worst, AssertGrouped("mma gate", gateRows, counts, offsets, tokenRows, Down(gateOut, slots * Rows), 2e-3));
        worst = Math.Max(worst, AssertGrouped("mma up", upRows, counts, offsets, tokenRows, Down(upOut, slots * Rows), 2e-3));
        worst = Math.Max(worst, AssertGrouped("mma slot-ordered", gateRows, counts, offsets, slotRows, Down(downOut, slots * Rows), 2e-3));
        _output.WriteLine($"{(GgmlTensorType)ggmlType} tensor-core {inDim}x{Rows}: worst error {worst:E2} of |w.a|");
    }

    // ---------------------------------------------------------------- the expert block

    /// <summary>
    /// <see cref="CudaMoe.Experts"/> on the 1017's layout pairs - IQ1_S, IQ2_S and IQ2_XXS gate/up over
    /// IQ2_S, IQ3_XXS and IQ2_XS down - from weights two bytes off a 4-byte boundary: a decode-size step on
    /// the per-slot kernels, then a ubatch past the per-slot bound on the route <see cref="CudaMoe.SelectRoute"/>
    /// picks for the loaded module (tensor cores on an nvcc sm_80+ build). The reference is the block in
    /// double from ggml's dequantized weights with both activations rounded as the route rounds them (q8_1
    /// on the per-slot, staged and warp kernels; F16 on the tensor cores, which stage the weights in F16
    /// too), so what remains is the kernels' own error: ggml's integer rounding of the IQ2 / IQ3 scales in
    /// the q8_1 dots, accumulation order on the tensor cores. Against the unrounded float block these weights stray about 1e-2, nearly all of it the
    /// q8_1 rounding of the heavy-tailed SwiGLU output; that is reported and held to the engine-level 2e-2.
    /// </summary>
    [CudaTheory]
    [InlineData(IQ1_S, IQ2_S)]
    [InlineData(IQ2_S, IQ3_XXS)]
    [InlineData(IQ2_XXS, IQ2_XS)]
    [InlineData(IQ2_XXS, IQ2_XXS)]
    public void TheExpertBlock_OnItsRoutes_MatchesTheBlockAtTheRoutesRounding(int guType, int downType)
    {
        const int e = 512, ff = 256, experts = 6, used = 2, cap = 40, perSlotMaxRows = 16;
        long guRowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(guType, e);
        long downRowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(downType, ff);
        byte[] gate = QuantRows(guType, experts * ff, guRowBytes, seed: 81);
        byte[] up = QuantRows(guType, experts * ff, guRowBytes, seed: 82);
        byte[] down = QuantRows(downType, experts * e, downRowBytes, seed: 83);
        float[] cur = Enumerable.Range(0, cap * e).Select(i => 0.05f * MathF.Sin(i * 0.237f) + 0.02f * MathF.Cos(i * 0.0113f)).ToArray();
        float[] shared = Enumerable.Range(0, cap * e).Select(i => 0.01f * MathF.Cos(i * 0.71f)).ToArray();
        int[] sel = Enumerable.Range(0, cap * used).Select(i => (i * 5 + i / used) % experts).ToArray();
        float[] selW = Enumerable.Range(0, cap * used).Select(i => 0.3f + 0.4f * ((i * 7) % 5) / 4f).ToArray();

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.DecodesLowBitIqExperts);
        using var gateT = Up(allocator, Shifted(gate, Shift));
        using var upT = Up(allocator, Shifted(up, Shift));
        using var downT = Up(allocator, Shifted(down, Shift));
        var g = new DeviceWeight { Ptr = Ptr(gateT) + Shift, Type = guType, Ne0 = e, Ne1 = ff, RowBytes = guRowBytes };
        var u = new DeviceWeight { Ptr = Ptr(upT) + Shift, Type = guType, Ne0 = e, Ne1 = ff, RowBytes = guRowBytes };
        var d = new DeviceWeight { Ptr = Ptr(downT) + Shift, Type = downType, Ne0 = ff, Ne1 = e, RowBytes = downRowBytes };
        using var curT = Up(allocator, cur);
        using var sharedT = Up(allocator, shared);
        using var output = new Tensor(allocator, DType.Float32, cap, e);
        var owned = new List<Tensor>();
        Tensor Alloc(DType type, long[] sizes) { var t = new Tensor(allocator, type, sizes); owned.Add(t); return t; }
        try
        {
            var scratch = new CudaMoeScratch(Alloc, cap, experts, used, e, ff);
            scratch.Sel.SetElementsAsInt(sel);
            scratch.SelW.SetElementsAsFloat(selW);
            scratch.Sel.Storage.EnsureDeviceCurrent();
            scratch.SelW.Storage.EnsureDeviceCurrent();
            float[][] gateRows = DequantRows(guType, gate, guRowBytes, e), upRows = DequantRows(guType, up, guRowBytes, e);
            float[][] downRows = DequantRows(downType, down, downRowBytes, ff);
            Func<float[], float[]> exact = x => x;
            Func<float[], float[]> q8_1 = x =>
            {
                var m = new float[1, x.Length];
                for (int k = 0; k < x.Length; k++)
                    m[0, k] = x[k];
                return Row(QuantizeDequantizeQ8_1(m), 0);
            };
            Func<float[], float[]> f16 = x => Array.ConvertAll(x, v => (float)(System.Half)v);
            float[][] F16Rows(float[][] rows) => rows.Select(f16).ToArray();
            float[][] gateF16 = F16Rows(gateRows), upF16 = F16Rows(upRows), downF16 = F16Rows(downRows);
            double[,] floatBlock = Block(gateRows, upRows, downRows, cur, shared, sel, selW, cap, used, e, ff, exact);
            foreach (int nt in new[] { 3, cap })
            {
                var route = CudaMoe.SelectRoute(nt, perSlotMaxRows, dk.HasMoeMma, dk.DecodesLowBitIqExperts, guType, downType, e, ff);
                Assert.Equal(nt <= perSlotMaxRows ? CudaMoe.ExpertRoute.PerSlot
                    : dk.HasMoeMma ? CudaMoe.ExpertRoute.TensorCore : CudaMoe.ExpertRoute.Staged, route);
                double[,] want = route == CudaMoe.ExpertRoute.TensorCore
                    ? Block(gateF16, upF16, downF16, cur, shared, sel, selW, nt, used, e, ff, f16)
                    : Block(gateRows, upRows, downRows, cur, shared, sel, selW, nt, used, e, ff, q8_1);
                Clear(allocator, output);
                CudaMoe.Experts(dk, allocator.Kernels, scratch, g, u, d, curT, sharedT, output,
                    nt, used, experts, e, ff, 0f, perSlotMaxRows, allocator.Stream.Handle);
                allocator.Stream.Synchronize();
                float[] got = Down(output, nt * e);
                Assert.All(got, v => Assert.True(float.IsFinite(v)));
                double rel = RelativeL2(got, want, nt, e), relFloat = RelativeL2(got, floatBlock, nt, e);
                _output.WriteLine($"{(GgmlTensorType)guType} gate/up, {(GgmlTensorType)downType} down, {nt} tokens ({route}): " +
                    $"relative error {rel:E2} at the route's rounding, {relFloat:E2} against the float block");
                Assert.True(rel <= 2e-3, $"{route}: the expert block strayed {rel:E2} from the block at the route's rounding");
                Assert.True(relFloat <= 2e-2, $"{route}: the expert block strayed {relFloat:E2} from the float block");
            }
        }
        finally
        {
            allocator.Synchronize();
            foreach (Tensor t in owned)
                t.Dispose();
        }
    }

    private static double RelativeL2(float[] got, double[,] want, int nt, int e)
    {
        double num = 0, den = 0;
        for (int t = 0; t < nt; t++)
        {
            for (int c = 0; c < e; c++)
            {
                double diff = got[t * e + c] - want[t, c];
                num += diff * diff;
                den += want[t, c] * want[t, c];
            }
        }
        return Math.Sqrt(num / den);
    }

    /// <summary>out[t] = shared[t] + sum_j w[t, j] * down_j(silu(gate_j x) * up_j x) in double from dequantized
    /// rows (gate/up indexed expert * ff + row, down expert * e + row), with x and the SwiGLU output each
    /// passed through <paramref name="round"/> first, as a route rounds its two activations.</summary>
    private static double[,] Block(float[][] gate, float[][] up, float[][] down, float[] cur, float[] shared,
        int[] sel, float[] selW, int nt, int used, int e, int ff, Func<float[], float[]> round)
    {
        var result = new double[nt, e];
        var h = new float[ff];
        for (int t = 0; t < nt; t++)
        {
            float[] x = round(cur.AsSpan(t * e, e).ToArray());
            for (int c = 0; c < e; c++)
                result[t, c] = shared[t * e + c];
            for (int j = 0; j < used; j++)
            {
                int xp = sel[t * used + j];
                for (int r = 0; r < ff; r++)
                {
                    double gv = 0, uv = 0;
                    float[] gw = gate[xp * ff + r], uw = up[xp * ff + r];
                    for (int k = 0; k < e; k++)
                    {
                        gv += (double)gw[k] * x[k];
                        uv += (double)uw[k] * x[k];
                    }
                    h[r] = (float)(gv / (1 + Math.Exp(-gv)) * uv);
                }
                float[] hr = round(h);
                for (int c = 0; c < e; c++)
                {
                    float[] dw = down[xp * e + c];
                    double y = 0;
                    for (int k = 0; k < ff; k++)
                        y += (double)dw[k] * hr[k];
                    result[t, c] += selW[t * used + j] * y;
                }
            }
        }
        return result;
    }

    /// <summary>
    /// A layout no decoder reads (IQ1_M here, which no gate admits) answers NaN on every route - the
    /// per-token decode, grouped warp and register-staged kernels and the dense q8_1 rows, which every
    /// module carries - where the decoder chains used to end in a bare Q8_0 branch that read any such row
    /// as Q8_0: wrong output and no error once a gate admitted a type before its decoder existed.
    /// </summary>
    [CudaFact]
    public void ALayoutNoDecoderReads_AnswersNaN_OnEveryRoute() => NoDecoderAnswersNaN(tensorCore: false);

    /// <summary>The same on the tensor-core kernel, which needs the module's sm_80+ body (skipped, saying
    /// so, on a module without one).</summary>
    [CudaFact(TensorCoreModule = "tensorsharp_dsv4_kernels")]
    public void ALayoutNoDecoderReads_AnswersNaN_OnTheTensorCores() => NoDecoderAnswersNaN(tensorCore: true);

    private static void NoDecoderAnswersNaN(bool tensorCore)
    {
        const int type = (int)GgmlTensorType.IQ1_M, inDim = 512, slots = Tokens * Used;
        long rowBytes = GgmlGgufTensorDequant.GetRowSizeBytes(type, inDim);
        // Every byte 0x11: read as anything (a Q8_0 scale is 0x1111, about 5e-4) the bytes are finite, so only
        // the NaN default answers NaN; the slack keeps a Q8_0 read of the last rows inside the buffer.
        byte[] weights = Enumerable.Repeat((byte)0x11, checked((int)(Experts * Rows * rowBytes)) + 4096).ToArray();
        float[,] act = Activations(Tokens, inDim, 0.013f, 3);
        (int[] counts, int[] offsets, int[] slotToken) = Group();
        int[] sel = { 4, 0, 2, 4, 1, 3 };

        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        if (tensorCore)
            Assert.True(dk.HasMoeMma, "The gate read an sm_80+ module, but the loaded one disables its tensor-core expert kernel.");
        IntPtr stream = allocator.Stream.Handle;
        using var w = Up(allocator, weights);
        using var actT = Up(allocator, act);
        using var countsT = Up(allocator, counts);
        using var offsetsT = Up(allocator, offsets);
        using var slotTokenT = Up(allocator, slotToken);
        using var selT = Up(allocator, sel);
        using var actQ = new Tensor(allocator, DType.UInt8, (long)Tokens * (inDim / 32) * CudaKernels.Q81BlockBytes);
        using var actQs = new Tensor(allocator, DType.UInt8, (long)Tokens * inDim);
        using var actD = new Tensor(allocator, DType.Float32, (long)Tokens * (inDim / 32));
        using var out0 = new Tensor(allocator, DType.Float32, slots, Rows);
        using var out1 = new Tensor(allocator, DType.Float32, slots, Rows);
        allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(actT), Ptr(actQ), inDim, Tokens, stream, warpCooperative: true);
        allocator.Kernels.LaunchQuantizeQ81SplitRows(Ptr(actT), Ptr(actQs), Ptr(actD), inDim, Tokens, stream);

        void Expect(string route, int count, Action launch)
        {
            allocator.Kernels.LaunchFillF32(Ptr(out0), slots * Rows, 0f, stream);
            allocator.Kernels.LaunchFillF32(Ptr(out1), slots * Rows, 0f, stream);
            launch();
            allocator.Stream.Synchronize();
            foreach (Tensor o in new[] { out0, out1 })
                Assert.All(Down(o, count), v => Assert.True(float.IsNaN(v), $"{route}: {v} where NaN was due"));
        }
        if (tensorCore)
        {
            Expect("tensor-core", slots * Rows, () => dk.MoeMma(Ptr(w), Ptr(w), actT, countsT, offsetsT, slotTokenT, out0, out1,
                type, Rows, inDim, rowBytes, Experts, stream));
            return;
        }
        Expect("decode", sel.Length * Rows, () => dk.MoeGateUpDecode(Ptr(w), Ptr(w), actQ, selT, out0, out1, type, Rows, inDim,
            rowBytes, 3, Used, stream));
        Expect("warp", slots * Rows, () => dk.MoeGateUp(Ptr(w), Ptr(w), actQ, countsT, offsetsT, slotTokenT, out0, out1, type,
            Rows, inDim, rowBytes, Experts, stream));
        Expect("staged", slots * Rows, () => dk.MoeGateUpStaged(Ptr(w), Ptr(w), actQs, actD, countsT, offsetsT, slotTokenT,
            out0, out1, type, Rows, inDim, rowBytes, Experts, stream));
        Expect("dense", Tokens * Rows, () =>
        {
            dk.DenseQ81Rows(Ptr(w), Ptr(actQ), Ptr(out0), type, Rows, inDim, rowBytes, Tokens, stream);
            dk.DenseQ81Rows(Ptr(w), Ptr(actQ), Ptr(out1), type, Rows, inDim, rowBytes, Tokens, stream);
        });
    }

    // ---------------------------------------------------------------- gates

    /// <summary>The module this build loads carries the low-bit IQ decoders' marker, so the engines take
    /// these experts on it.</summary>
    [CudaFact]
    public void TheLoadedKernelModule_DecodesLowBitIqExperts()
    {
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(dk.DecodesLowBitIqExperts);
        Assert.True(dk.DecodesQ5_1Experts);
        _output.WriteLine($"HasMoeMma {dk.HasMoeMma}");
    }

    /// <summary>
    /// A kernel module built from source older than these decoders (a committed PTX baseline not yet
    /// regenerated, which a machine without nvcc loads) reads IQ1_S and IQ2_S rows as Q8_0: it refuses them,
    /// by tensor name, as does a caller that does not pass the marker at all. IQ2_XXS passes either way, and
    /// Q5_1 still answers to its own marker. The current kernel source lists all of them.
    /// </summary>
    [Fact]
    public void LowBitIqExperts_AreRefusedOnAModuleThatPredatesTheirDecoders()
    {
        var gate = Desc(IQ1_S, "blk.3.ffn_gate_exps.weight");
        var up = Desc(IQ1_S, "blk.3.ffn_up_exps.weight");
        var down = Desc(IQ2_S, "blk.3.ffn_down_exps.weight");
        CudaMoe.RequireExpertWeights("dsv4-cuda", 3, gate, up, down);
        CudaMoe.RequireExpertDecoders("dsv4-cuda", true, 3, gate, up, down, decodesLowBitIq: true);

        var ex = Assert.Throws<NotSupportedException>(() =>
            CudaMoe.RequireExpertDecoders("dsv4-cuda", true, 3, gate, up, down, decodesLowBitIq: false));
        Assert.StartsWith("[dsv4-cuda] blk.3.ffn_gate_exps.weight is IQ1_S, and the loaded tensorsharp_dsv4_kernels module", ex.Message);
        Assert.Contains("older than its IQ1_S expert decoder", ex.Message);
        Assert.Contains("nvcc", ex.Message);
        Assert.Contains("--backend ggml_cuda", ex.Message);
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out _));

        // A caller that passes only the Q5_1 marker refuses them the same way; a nameless descriptor
        // still says which tensor.
        var q4k = Desc((int)GgmlTensorType.Q4_K, null);
        ex = Assert.Throws<NotSupportedException>(() =>
            CudaMoe.RequireExpertDecoders("glm-cuda", true, 7, q4k, q4k, Desc(IQ2_S, null)));
        Assert.StartsWith("[glm-cuda] blk.7.ffn_down_exps.weight is IQ2_S, and", ex.Message);

        // IQ2_XXS predates the marker on the warp-per-row kernels; Q5_1 keys on its own.
        var iq2xxs = Desc(IQ2_XXS, "blk.4.ffn_gate_exps.weight");
        CudaMoe.RequireExpertDecoders("dsv4-cuda", true, 4, iq2xxs, iq2xxs, Desc(IQ3_XXS, null), decodesLowBitIq: false);
        ex = Assert.Throws<NotSupportedException>(() => CudaMoe.RequireExpertDecoders("q4e-cuda", false, 4, iq2xxs, iq2xxs,
            Desc((int)GgmlTensorType.Q5_1, "blk.4.ffn_down_exps.weight"), decodesLowBitIq: true));
        Assert.StartsWith("[q4e-cuda] blk.4.ffn_down_exps.weight is Q5_1, and", ex.Message);

        foreach (int type in new[] { IQ1_S, IQ2_S, IQ2_XXS, IQ2_XS, IQ3_XXS })
            Assert.True(CudaMoe.SupportsExpertType(type), CudaMoe.GgmlTypeName(type));
        Assert.False(CudaMoe.SupportsExpertType((int)GgmlTensorType.IQ1_M));
        Assert.Contains("IQ1_S", CudaMoe.SupportedExpertTypes.Split(", "));
        Assert.Contains("IQ2_S", CudaMoe.SupportedExpertTypes.Split(", "));
    }

    /// <summary>
    /// The marker gates exactly the three layouts on the grouped kernels: without it IQ2_XXS (which the
    /// warp-per-row kernels decode on any module) never reaches a staged or tensor-core kernel, where an old
    /// module would read it as Q8_0 - in either projection, whatever the other layout - and the other
    /// layouts route as before. Shapes are the 1017's: 5120-wide gate/up, 2304-wide down.
    /// </summary>
    [Fact]
    public void WithoutTheMarker_Iq2XxsNeverReachesTheGroupedKernels()
    {
        CudaMoe.ExpertRoute Route(int nt, bool mma, bool caps, int gu, int dn)
            => CudaMoe.SelectRoute(nt, 16, mma, caps, gu, dn, 5120, 2304);

        Assert.Equal(CudaMoe.ExpertRoute.Warp, Route(40, true, false, IQ2_XXS, IQ3_XXS));
        Assert.Equal(CudaMoe.ExpertRoute.Warp, Route(40, false, false, IQ2_XXS, IQ3_XXS));
        Assert.Equal(CudaMoe.ExpertRoute.Warp, Route(40, true, false, IQ2_XS, IQ2_XXS));
        Assert.Equal(CudaMoe.ExpertRoute.Warp, Route(40, false, false, IQ2_XS, IQ2_XXS));
        Assert.Equal(CudaMoe.ExpertRoute.PerSlot, Route(16, true, false, IQ2_XXS, IQ3_XXS));

        Assert.Equal(CudaMoe.ExpertRoute.TensorCore, Route(40, true, true, IQ2_XXS, IQ3_XXS));
        Assert.Equal(CudaMoe.ExpertRoute.Staged, Route(40, false, true, IQ2_XXS, IQ3_XXS));
        Assert.Equal(CudaMoe.ExpertRoute.TensorCore, Route(40, true, true, IQ1_S, IQ2_S));
        Assert.Equal(CudaMoe.ExpertRoute.Staged, Route(40, false, true, IQ1_S, IQ2_S));
        Assert.Equal(CudaMoe.ExpertRoute.TensorCore, Route(40, true, false, IQ2_XS, IQ3_XXS));
        Assert.Equal(CudaMoe.ExpertRoute.Staged, Route(40, false, false, IQ2_XS, IQ3_XXS));

        Assert.False(CudaMoe.GroupedSupportsType(IQ2_XXS));
        Assert.True(CudaMoe.GroupedSupportsType(IQ2_XXS, decodesLowBitIq: true));
        var gated = Enumerable.Range(0, 64)
            .Where(t => CudaMoe.GroupedSupportsType(t, decodesLowBitIq: true) != CudaMoe.GroupedSupportsType(t))
            .ToArray();
        Assert.Equal(new[] { IQ2_XXS, IQ1_S, IQ2_S }, gated);
        Assert.All(Enumerable.Range(0, 64).Where(t => CudaMoe.GroupedSupportsType(t)),
            t => Assert.True(CudaMoe.GroupedSupportsType(t, decodesLowBitIq: true)));
    }

    // ---------------------------------------------------------------- helpers

    private static CudaWeightDesc Desc(int type, string name) => new CudaWeightDesc { GgmlType = type, Name = name };

    private static float[,] Activations(int rows, int inDim, float freq, int phase)
    {
        var act = new float[rows, inDim];
        for (int t = 0; t < rows; t++)
            for (int k = 0; k < inDim; k++)
                act[t, k] = MathF.Sin(freq * (k + 1) * (t + phase)) + 0.25f * MathF.Cos(0.071f * k - t);
        return act;
    }

    /// <summary>Every row of a quantized matrix through ggml's own to_float.</summary>
    private static float[][] DequantRows(int ggmlType, byte[] bytes, long rowBytes, int n)
    {
        var rows = new float[bytes.Length / rowBytes][];
        for (int r = 0; r < rows.Length; r++)
        {
            rows[r] = new float[n];
            GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, bytes, checked((int)(r * rowBytes)), rows[r], 0, n);
        }
        return rows;
    }

    /// <summary>Every token takes expert 1 (40 members: two passes of 32) and one of 0, 2, 4; expert 3 is
    /// never chosen. Returns the per-expert counts, their exclusive scan and each slot row's token.</summary>
    private static (int[] Counts, int[] Offsets, int[] SlotToken) Group()
    {
        var sel = new int[Tokens * Used];
        for (int t = 0; t < Tokens; t++)
        {
            sel[t * Used] = 1;
            sel[t * Used + 1] = new[] { 0, 2, 4 }[t % 3];
        }
        var counts = new int[Experts];
        foreach (int x in sel)
            counts[x]++;
        var offsets = new int[Experts];
        for (int x = 1; x < Experts; x++)
            offsets[x] = offsets[x - 1] + counts[x - 1];
        var cursor = (int[])offsets.Clone();
        var slotToken = new int[sel.Length];
        for (int s = 0; s < sel.Length; s++)
            slotToken[cursor[sel[s]]++] = s / Used;
        Assert.Equal(0, counts[3]);
        Assert.True(counts[1] > 32);
        return (counts, offsets, slotToken);
    }

    /// <summary>Each member slot's output of a grouped launch against its expert's dequantized row;
    /// returns the worst error as a fraction of |w.a|.</summary>
    private static double AssertGrouped(string what, float[][] rows, int[] counts, int[] offsets, float[][] activation,
        float[] got, double bound)
    {
        double worst = 0;
        for (int x = 0; x < Experts; x++)
        {
            for (int r = 0; r < Rows; r++)
            {
                for (int i = 0; i < counts[x]; i++)
                {
                    int s = offsets[x] + i;
                    worst = Math.Max(worst, AssertDot($"{what}: expert {x} row {r} member {i}", rows[x * Rows + r],
                        activation[s], got[(long)s * Rows + r], bound));
                }
            }
        }
        return worst;
    }

    /// <summary>Within <paramref name="bound"/> of the products' absolute sum (plus 1e-6).</summary>
    private static double AssertDot(string what, float[] w, float[] a, float got, double bound)
    {
        double sum = 0, mag = 0;
        for (int k = 0; k < a.Length; k++)
        {
            double p = (double)w[k] * a[k];
            sum += p;
            mag += Math.Abs(p);
        }
        double err = Math.Abs(got - sum);
        if (!(err <= bound * mag + 1e-6))
            Assert.Fail($"{what}: {got} vs {sum} (|w.a| {mag})");
        return err / Math.Max(mag, 1e-30);
    }

    private static void Clear(CudaAllocator allocator, params Tensor[] tensors)
    {
        foreach (Tensor t in tensors)
            allocator.Kernels.LaunchFillF32(Ptr(t), checked((int)t.ElementCount()), float.NaN, allocator.Stream.Handle);
    }

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    private static string RepoFile(params string[] parts)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(new[] { dir.FullName }.Concat(parts).ToArray());
            if (File.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }
        throw new FileNotFoundException(string.Join("/", parts));
    }
}
