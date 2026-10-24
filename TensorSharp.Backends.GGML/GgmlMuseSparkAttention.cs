// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public partial class GgmlBasicOps
{
    /// <summary>
    /// Spark CUDA attention preserving F32 Q/K/V operands. Input format zero is
    /// contiguous head-first Q/K/V; format one is token-first. Output is always
    /// token-first, irrespective of its Tensor shape. This is an owned F32 path,
    /// not upstream flash attention.
    /// </summary>
    public static void MuseSparkPrefillAttentionF32(Tensor query, Tensor key, Tensor value, Tensor output,
        int numHeads, int numKvHeads, int headDim, int seqLen, int kvLen,
        int maskStartPos, int slidingWindow, float scale, int inputFormat = 0)
    {
        ValidateSparkAttentionGeometry(numHeads, numKvHeads, headDim, seqLen, slidingWindow, scale);
        if (kvLen <= 0 || maskStartPos < 0 || (long)maskStartPos + seqLen > kvLen ||
            inputFormat is not (0 or 1))
            throw new ArgumentException("Invalid Spark F32 attention sequence geometry.");
        int qRows = inputFormat == 0 ? numHeads : seqLen;
        int qWidth = checked((inputFormat == 0 ? seqLen : numHeads) * headDim);
        int kvRows = inputFormat == 0 ? numKvHeads : kvLen;
        int kvWidth = checked((inputFormat == 0 ? kvLen : numKvHeads) * headDim);
        var q = SparkAttentionView(query, qRows, qWidth);
        var k = SparkAttentionView(key, kvRows, kvWidth);
        var v = SparkAttentionView(value, kvRows, kvWidth);
        var result = SparkAttentionView(output, seqLen, checked(numHeads * headDim));
        if (ReferenceEquals(output.Storage, query.Storage) || ReferenceEquals(output.Storage, key.Storage) ||
            ReferenceEquals(output.Storage, value.Storage))
            throw new ArgumentException("Spark F32 attention output must have distinct storage.");
        if (output.Storage is not GgmlStorage storage || storage.Allocator is not GgmlAllocator allocator ||
            allocator.Context.BackendType != GgmlBackendType.Cuda)
            throw new NotSupportedException("Spark owned F32 attention requires GGML CUDA.");
        GgmlNative.SetActiveRankIfUnpinned(storage.DeviceId);
        GgmlNative.MuseSparkPrefillAttentionF32(q, k, v, result, numHeads, numKvHeads, headDim,
            seqLen, kvLen, maskStartPos, slidingWindow, scale, inputFormat);
    }

    /// <summary>
    /// Spark CUDA paged F32 attention. Q/output are token-first, pages are
    /// [physical token capacity, KV heads * head dimension]. Metadata contents
    /// and physical page bounds are checked by the native entry point.
    /// </summary>
    public static unsafe void MuseSparkPagedAttentionF32(float[] query, float[] keyPages, float[] valuePages,
        float[] output, int[] queryStartLoc, int[] seqLens, int[] positions, int[] blockTableFlat,
        int[] blockTableOffsets, int numSeqs, int numTokens, int numHeads, int numKvHeads, int headDim,
        int blockSize, float scale, int slidingWindow = 0)
    {
        ValidateSparkAttentionGeometry(numHeads, numKvHeads, headDim, numTokens, slidingWindow, scale);
        int width = checked(numHeads * headDim), kvWidth = checked(numKvHeads * headDim);
        if (numSeqs <= 0 || numSeqs > numTokens || blockSize <= 0 ||
            query == null || output == null || keyPages == null || valuePages == null ||
            query.LongLength != (long)numTokens * width || output.Length != query.Length ||
            keyPages.Length == 0 || keyPages.Length != valuePages.Length ||
            keyPages.Length % kvWidth != 0 || keyPages.Length / kvWidth % blockSize != 0 ||
            queryStartLoc == null || queryStartLoc.LongLength < (long)numSeqs + 1 ||
            seqLens == null || seqLens.Length < numSeqs || positions == null || positions.Length != numTokens ||
            blockTableOffsets == null || blockTableOffsets.Length < numSeqs ||
            blockTableFlat == null || blockTableFlat.Length == 0)
            throw new ArgumentException("Invalid Spark paged F32 attention buffers or metadata lengths.");
        if (ReferenceEquals(output, query) || ReferenceEquals(output, keyPages) || ReferenceEquals(output, valuePages))
            throw new ArgumentException("Spark paged F32 attention output must have distinct storage.");
        fixed (float* q = query, k = keyPages, v = valuePages, o = output)
        fixed (int* starts = queryStartLoc, lengths = seqLens, pos = positions,
            blocks = blockTableFlat, offsets = blockTableOffsets)
            GgmlNative.MuseSparkPagedAttentionF32(
                SparkAttentionArrayView(q, numTokens, width), SparkAttentionArrayView(k, keyPages.Length / kvWidth, kvWidth),
                SparkAttentionArrayView(v, valuePages.Length / kvWidth, kvWidth), SparkAttentionArrayView(o, numTokens, width),
                (IntPtr)starts, (IntPtr)lengths, (IntPtr)pos, (IntPtr)blocks, (IntPtr)offsets,
                numSeqs, blockTableFlat.Length, numHeads, numKvHeads, headDim, blockSize, slidingWindow, scale);
    }

    private static void ValidateSparkAttentionGeometry(int heads, int kvHeads, int dim, int rows, int window, float scale)
    {
        if (heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || dim <= 0 || dim > 2048 || rows <= 0 ||
            window < 0 || !float.IsFinite(scale) || scale <= 0)
            throw new ArgumentException("Invalid Spark F32 attention geometry or scale.");
    }

    private static GgmlTensorView2D SparkAttentionView(Tensor tensor, int rows, int width)
    {
        if (tensor == null || tensor.ElementType != DType.Float32 || !tensor.IsContiguous() ||
            tensor.ElementCount() != (long)rows * width || tensor.StorageOffset < 0 ||
            tensor.StorageOffset > tensor.Storage.ElementCount ||
            tensor.ElementCount() > tensor.Storage.ElementCount - tensor.StorageOffset)
            throw new ArgumentException("Spark F32 attention requires exact contiguous Float32 tensor spans.");
        return new GgmlTensorView2D(GetBufferStart(tensor), rows, width, width, 1,
            checked((long)rows * width * sizeof(float)));
    }

    private static unsafe GgmlTensorView2D SparkAttentionArrayView(float* data, int rows, int width)
        => new((IntPtr)data, rows, width, width, 1, checked((long)rows * width * sizeof(float)));
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_MuseSparkPrefillAttentionF32(
        GgmlTensorView2D query, GgmlTensorView2D key, GgmlTensorView2D value, GgmlTensorView2D output,
        int numHeads, int numKvHeads, int headDim, int seqLen, int kvLen,
        int maskStartPos, int slidingWindow, float scale, int inputFormat);

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_MuseSparkPagedAttentionF32(
        GgmlTensorView2D query, GgmlTensorView2D keyPages, GgmlTensorView2D valuePages, GgmlTensorView2D output,
        IntPtr queryStartLoc, IntPtr seqLens, IntPtr positions, IntPtr blockTableFlat, IntPtr blockTableOffsets,
        int numSeqs, int blockTableEntries, int numHeads, int numKvHeads, int headDim,
        int blockSize, int slidingWindow, float scale);

    internal static void MuseSparkPrefillAttentionF32(
        GgmlTensorView2D query, GgmlTensorView2D key, GgmlTensorView2D value, GgmlTensorView2D output,
        int numHeads, int numKvHeads, int headDim, int seqLen, int kvLen,
        int maskStartPos, int slidingWindow, float scale, int inputFormat)
        => RequireSparkOwnedAttention(TSGgml_MuseSparkPrefillAttentionF32(query, key, value, output,
            numHeads, numKvHeads, headDim, seqLen, kvLen, maskStartPos, slidingWindow, scale, inputFormat));

    internal static void MuseSparkPagedAttentionF32(
        GgmlTensorView2D query, GgmlTensorView2D keyPages, GgmlTensorView2D valuePages, GgmlTensorView2D output,
        IntPtr queryStartLoc, IntPtr seqLens, IntPtr positions, IntPtr blockTableFlat, IntPtr blockTableOffsets,
        int numSeqs, int blockTableEntries, int numHeads, int numKvHeads, int headDim,
        int blockSize, int slidingWindow, float scale)
        => RequireSparkOwnedAttention(TSGgml_MuseSparkPagedAttentionF32(query, keyPages, valuePages, output,
            queryStartLoc, seqLens, positions, blockTableFlat, blockTableOffsets,
            numSeqs, blockTableEntries, numHeads, numKvHeads, headDim, blockSize, slidingWindow, scale));

    private static void RequireSparkOwnedAttention(int result)
    {
        CheckResult(result, "muse_spark_owned_f32_attention");
        if (result != 2)
            throw new InvalidOperationException("Spark F32 attention did not execute its owned CUDA path.");
    }
}
