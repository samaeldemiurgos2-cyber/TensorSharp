// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime;

namespace TensorSharp.MLX;

/// <summary>F32 device operations shared by the direct diffusion models.</summary>
public static class MlxDirectOps
{
    /// <summary>Apply pair-interleaved rotary embeddings to contiguous token-major heads.</summary>
    public static bool TryRopeInterleaved(Tensor data, Tensor cos, Tensor sin, int heads, int tokens, int headDim)
    {
        if (data?.Storage is not MlxStorage ds || cos?.Storage is not MlxStorage cs || sin?.Storage is not MlxStorage ss ||
            data.ElementType != DType.Float32 || cos.ElementType != DType.Float32 || sin.ElementType != DType.Float32 ||
            data.DimensionCount != 2 || cos.DimensionCount != 2 || sin.DimensionCount != 2 ||
            !data.IsContiguous() || !cos.IsContiguous() || !sin.IsContiguous() ||
            data.StorageOffset != 0 || data.Storage.ElementCount != data.ElementCount() ||
            heads <= 0 || tokens <= 0 || headDim <= 0 || (headDim & 1) != 0 ||
            data.Sizes[0] != tokens || data.Sizes[1] != (long)heads * headDim ||
            cos.Sizes[0] != tokens || cos.Sizes[1] != headDim || sin.Sizes[0] != tokens || sin.Sizes[1] != headDim ||
            !ReferenceEquals(ds.AllocatorImpl, cs.AllocatorImpl) || !ReferenceEquals(ds.AllocatorImpl, ss.AllocatorImpl)) return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray x = default, c = default, s = default, output = default;
            try
            {
                x = ds.CreateArrayView(data); c = cs.CreateArrayView(cos); s = ss.CreateArrayView(sin);
                output = MlxNative.DirectRopeInterleaved(x, c, s, heads, tokens, headDim);
                ds.ReplaceDeviceArray(output); output = default; return true;
            }
            finally { MlxNative.FreeArray(x); MlxNative.FreeArray(c); MlxNative.FreeArray(s); MlxNative.FreeArray(output); }
        });
    }
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel directInterleavedRopeKernel;
    private static MlxFastMetalKernel directQ6KFloatDequantKernel;
    private static MlxFastMetalKernel directQ2KFloatDequantKernel;
    private static MlxFastMetalKernel directQ3KFloatDequantKernel;
    private static MlxFastMetalKernel directBf16FloatDequantKernel;

    /// <summary>
    /// GGUF types the direct F32 projections keep in their file bytes and dequantize per call
    /// (<see cref="DirectRawFloatMatmul"/>) because MLX has no repacked form for them: K-quants
    /// with 16-value sub-blocks (MLX's affine groups are 32, 64 or 128) and BF16. Expanded to
    /// dense F32 at load instead, Unsloth's mixed LTX 2.5 UD-Q4_K_XL grew from 14 GB on disk to a
    /// 40 GB process on a 48 GB Mac and swapped.
    /// </summary>
    internal static bool IsDirectRawFloatType(int ggmlType) =>
        ggmlType is (int)GgmlTensorType.Q2_K or (int)GgmlTensorType.Q3_K or (int)GgmlTensorType.BF16;

    // Element `e` (0..255) of one block, as ggml's dequantize_row_q2_K / dequantize_row_q3_K
    // compute it. Every product is exact in F32 (an F16 scale times small integers), and Q2_K's
    // one subtraction rounds once with or without contraction, so these equal the CPU dequant
    // bit for bit.
    private const string DirectKQuantHelpers = """
        inline float tensorsharp_direct_q2k(const device uchar * block, uint e) {
            const float d = float(*reinterpret_cast<const device half *>(block + 80));
            const float dmin = float(*reinterpret_cast<const device half *>(block + 82));
            const uint n = e >> 7, r = e & 127u, j = r >> 5, k = r & 31u;
            const uchar sc = block[n * 8u + j * 2u + (k >> 4)];
            const int q = (block[16u + n * 32u + k] >> (2u * j)) & 3;
            const float dl = d * float(sc & 0xF);
            const float ml = dmin * float(sc >> 4);
            return dl * float(q) - ml;
        }
        inline float tensorsharp_direct_q3k(const device uchar * block, uint e) {
            const device uchar * scales = block + 96;
            const float d = float(*reinterpret_cast<const device half *>(block + 108));
            const uint n = e >> 7, r = e & 127u, j = r >> 5, k = r & 31u;
            const uint is = n * 8u + j * 2u + (k >> 4);
            const int low = (scales[is & 7u] >> (4u * (is >> 3))) & 0xF;
            const int high = (scales[8u + (is & 3u)] >> (2u * (is >> 2))) & 3;
            const int q = int((block[32u + n * 32u + k] >> (2u * j)) & 3) - ((block[k] & (1u << (n * 4u + j))) ? 0 : 4);
            const float dl = d * float((low | (high << 4)) - 32);
            return dl * float(q);
        }
        """;

    private static MlxFastMetalKernel EnsureDirectRawDequantKernel(int ggmlType)
    {
        lock (fastKernelSync)
        {
            switch (ggmlType)
            {
                case (int)GgmlTensorType.Q2_K:
                    if (!directQ2KFloatDequantKernel.IsValid)
                        directQ2KFloatDequantKernel = CreateFastMetalKernel("tensorsharp_direct_q2k_dequant_f32",
                            ["w", "in_dim", "row0"], ["y"], DirectKQuantDequantSource(84, "tensorsharp_direct_q2k"), DirectKQuantHelpers);
                    return directQ2KFloatDequantKernel;
                case (int)GgmlTensorType.Q3_K:
                    if (!directQ3KFloatDequantKernel.IsValid)
                        directQ3KFloatDequantKernel = CreateFastMetalKernel("tensorsharp_direct_q3k_dequant_f32",
                            ["w", "in_dim", "row0"], ["y"], DirectKQuantDequantSource(110, "tensorsharp_direct_q3k"), DirectKQuantHelpers);
                    return directQ3KFloatDequantKernel;
                case (int)GgmlTensorType.BF16:
                    if (!directBf16FloatDequantKernel.IsValid)
                        directBf16FloatDequantKernel = CreateFastMetalKernel("tensorsharp_direct_bf16_dequant_f32",
                            ["w", "in_dim", "row0"], ["y"], """
                            const uint column = thread_position_in_grid.x;
                            const uint row = thread_position_in_grid.y;
                            const uint inner = uint(in_dim);
                            if (column < inner) {
                                device const ushort * source = reinterpret_cast<device const ushort *>(w);
                                const ushort bits = source[(uint64_t(uint(row0) + row)) * uint64_t(inner) + column];
                                y[uint64_t(row) * uint64_t(inner) + column] = as_type<float>(uint(bits) << 16);
                            }
                            """, string.Empty);
                    return directBf16FloatDequantKernel;
                default:
                    throw new NotSupportedException($"No direct F32 dequantization for GGML type {(GgmlTensorType)ggmlType}.");
            }
        }
    }

    private static string DirectKQuantDequantSource(int blockBytes, string helper) => $$"""
        const uint column = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        const uint inner = uint(in_dim);
        if (column < inner) {
            device const uchar * block = w + ((uint64_t(uint(row0) + row)) * uint64_t(inner / 256u) + uint64_t(column / 256u)) * {{blockBytes}}u;
            y[uint64_t(row) * uint64_t(inner) + column] = {{helper}}(block, column % 256u);
        }
        """;

    /// <summary>
    /// Rows [row0, row0 + rows) of a raw Q2_K, Q3_K or BF16 weight as exact F32 [rows, inDim]:
    /// the values the CPU dequantizer produces, bit for bit. <paramref name="inDimArray"/> may
    /// pass in a 0-d int array of inDim to share across slices.
    /// </summary>
    internal static MlxArray DirectRawDequantize(MlxArray rawWeight, int ggmlType, int row0, int rows, int inDim, MlxArray inDimArray = default)
    {
        if (!rawWeight.IsValid)
            throw new ArgumentException("Direct F32 dequantization requires a valid raw weight array.", nameof(rawWeight));
        if (rows <= 0 || inDim <= 0 || row0 < 0 || (ggmlType != (int)GgmlTensorType.BF16 && inDim % 256 != 0))
            throw new ArgumentOutOfRangeException(nameof(inDim), $"Direct F32 {(GgmlTensorType)ggmlType} dequantization requires positive dimensions" +
                (ggmlType == (int)GgmlTensorType.BF16 ? "." : " and an input dim aligned to 256."));
        MlxFastMetalKernel kernel = EnsureDirectRawDequantKernel(ggmlType);
        string label = ((GgmlTensorType)ggmlType).ToString();
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        MlxArray ownInDim = default, row0Array = default;
        try
        {
            if (!inDimArray.IsValid) inDimArray = ownInDim = mlx_array_new_int(inDim);
            int[] shape = [rows, inDim];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), $"configuring direct F32 {label} weight slice");
            Check(mlx_fast_metal_kernel_config_set_grid(config, inDim, rows, 1), $"configuring direct F32 {label} dequantization grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), $"configuring direct F32 {label} dequantization threadgroup");
            row0Array = mlx_array_new_int(row0);
            inputs = CreateVectorArray(rawWeight, inDimArray, row0Array); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, kernel, inputs, config, DefaultStream()), $"dequantizing direct F32 {label} weight");
            Check(mlx_vector_array_get(out MlxArray weight, outputs, 0), $"reading direct F32 {label} weight");
            return weight;
        }
        finally
        {
            FreeArray(row0Array); FreeArray(ownInDim);
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }

    /// <summary>
    /// [rows, inDim] F32 x [outDim, inDim]^T for a Q2_K, Q3_K or BF16 weight kept as its raw GGUF
    /// bytes: each slice of output rows is dequantized to exact F32 (<see cref="DirectRawDequantize"/>)
    /// and multiplied in F32, the precision <see cref="DirectQ6KFloatMatmul"/> gives Q6_K. A slice's
    /// dense copy is bounded at 64 MiB and finished before the next is made, so a 16k-wide layer
    /// never holds more. Lengths and offsets are kernel inputs, not template values, so every shape
    /// shares one compiled kernel per type.
    /// </summary>
    internal static MlxArray DirectRawFloatMatmul(MlxArray input, MlxArray rawWeight, int ggmlType, int rows, int inDim, int outDim)
    {
        if (!input.IsValid || !rawWeight.IsValid)
            throw new ArgumentException("Direct F32 matmul requires valid input and raw weight arrays.");
        if (rows <= 0 || outDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Direct F32 matmul requires positive dimensions.");
        string label = ((GgmlTensorType)ggmlType).ToString();
        int sliceRows = (int)Math.Clamp((64L * 1024 * 1024) / (sizeof(float) * (long)inDim), 1, outDim);
        var parts = new System.Collections.Generic.List<MlxArray>();
        MlxArray joined = default, inDimArray = default;
        try
        {
            inDimArray = mlx_array_new_int(inDim);
            for (int start = 0; start < outDim; start += sliceRows)
            {
                int count = Math.Min(sliceRows, outDim - start);
                MlxArray weight = default, transposed = default, part = default;
                try
                {
                    weight = DirectRawDequantize(rawWeight, ggmlType, start, count, inDim, inDimArray);
                    transposed = Transpose(weight, [1, 0]);
                    Check(mlx_matmul(out part, input, transposed, DefaultStream()), $"running direct F32 {label} projection");
                    Eval(part); parts.Add(part); part = default;
                }
                finally { FreeArray(part); FreeArray(transposed); FreeArray(weight); }
            }
            if (parts.Count == 1) { var result = parts[0]; parts.Clear(); return result; }
            var arrays = CreateVectorArray(parts.ToArray());
            try { Check(mlx_concatenate_axis(out joined, arrays, 1, DefaultStream()), $"joining direct F32 {label} projections"); }
            finally { if (arrays.IsValid) _ = mlx_vector_array_free(arrays); }
            var output = joined; joined = default; return output;
        }
        finally { foreach (var part in parts) FreeArray(part); FreeArray(joined); FreeArray(inDimArray); }
    }

    internal static MlxArray DirectQ6KFloatMatmul(MlxArray input, MlxArray rawWeight, int rows, int inDim, int outDim)
    {
        if (rows <= Q6KMatvecMaxRows) return Q6KMatvec(input, rawWeight, rows, inDim, outDim);
        lock (fastKernelSync)
        {
            if (!directQ6KFloatDequantKernel.IsValid)
                directQ6KFloatDequantKernel = CreateFastMetalKernel("tensorsharp_direct_q6k_dequant_f32", ["w"], ["y"], """
                    const uint column = thread_position_in_grid.x;
                    const uint row = thread_position_in_grid.y;
                    if (column < Inner && row < Rows) {
                        const uint block = column / 256;
                        device const uchar * source = w + (ulong(row + Start) * (Inner / 256) + block) * 210;
                        y[ulong(row) * Inner + column] = tensorsharp_dequant_q6k(source, column % 256);
                    }
                    """, KQuantHelpersHeader);
        }
        int sliceRows = (int)Math.Clamp((64L * 1024 * 1024) / (sizeof(float) * (long)inDim), 1, outDim);
        var parts = new System.Collections.Generic.List<MlxArray>();
        MlxArray joined = default;
        try
        {
            for (int start = 0; start < outDim; start += sliceRows)
            {
                int count = Math.Min(sliceRows, outDim - start);
                var config = mlx_fast_metal_kernel_config_new();
                MlxVectorArray inputs = default, outputs = default;
                MlxArray weight = default, transposed = default, part = default;
                try
                {
                    AddTemplateInt(config, "Start", start); AddTemplateInt(config, "Rows", count); AddTemplateInt(config, "Inner", inDim);
                    int[] shape = [count, inDim];
                    Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring direct F32 Q6_K weight slice");
                    Check(mlx_fast_metal_kernel_config_set_grid(config, inDim, count, 1), "configuring direct F32 Q6_K dequantization grid");
                    Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring direct F32 Q6_K dequantization threadgroup");
                    inputs = CreateVectorArray(rawWeight); outputs = mlx_vector_array_new();
                    Check(mlx_fast_metal_kernel_apply(ref outputs, directQ6KFloatDequantKernel, inputs, config, DefaultStream()), "dequantizing direct F32 Q6_K weight");
                    Check(mlx_vector_array_get(out weight, outputs, 0), "reading direct F32 Q6_K weight");
                    transposed = Transpose(weight, [1, 0]);
                    Check(mlx_matmul(out part, input, transposed, DefaultStream()), "running direct F32 Q6_K projection");
                    // Finish each weight slice before releasing it. The exact
                    // dense temporary remains bounded at 64 MiB on large layers.
                    Eval(part); parts.Add(part); part = default;
                }
                finally
                {
                    FreeArray(part); FreeArray(transposed); FreeArray(weight);
                    if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
                    if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
                    if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
                }
            }
            if (parts.Count == 1) { var result = parts[0]; parts.Clear(); return result; }
            var arrays = CreateVectorArray(parts.ToArray());
            try { Check(mlx_concatenate_axis(out joined, arrays, 1, DefaultStream()), "joining direct F32 Q6_K projections"); }
            finally { if (arrays.IsValid) _ = mlx_vector_array_free(arrays); }
            var output = joined; joined = default; return output;
        }
        finally { foreach (var part in parts) FreeArray(part); FreeArray(joined); }
    }

    internal static MlxArray DirectRopeInterleaved(MlxArray input, MlxArray cos, MlxArray sin,
        int heads, int tokens, int headDim)
    {
        int pairs = checked(tokens * heads * (headDim / 2));
        lock (fastKernelSync)
        {
            if (!directInterleavedRopeKernel.IsValid)
                directInterleavedRopeKernel = CreateFastMetalKernel("tensorsharp_direct_interleaved_rope",
                    ["x", "c", "s"], ["y"], """
                    const uint pair = thread_position_in_grid.x;
                    if (pair < Pairs) {
                        const uint halfPairs = HeadDim / 2;
                        const uint token = pair / (Heads * halfPairs);
                        const uint position = pair % halfPairs;
                        const uint offset = pair * 2;
                        const uint table = token * HeadDim + position * 2;
                        const float even = x[offset], odd = x[offset + 1];
                        const float cosine = c[table], sine = s[table];
                        y[offset] = even * cosine - odd * sine;
                        y[offset + 1] = odd * cosine + even * sine;
                    }
                    """, string.Empty);
        }
        var config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        try
        {
            AddTemplateInt(config, "Pairs", pairs); AddTemplateInt(config, "Heads", heads); AddTemplateInt(config, "HeadDim", headDim);
            int[] shape = [tokens, checked(heads * headDim)];
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring direct interleaved RoPE output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, pairs, 1, 1), "configuring direct interleaved RoPE grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring direct interleaved RoPE threadgroup");
            inputs = CreateVectorArray(input, cos, sin); outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, directInterleavedRopeKernel, inputs, config, DefaultStream()), "running direct interleaved RoPE");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading direct interleaved RoPE output");
            return result;
        }
        finally
        {
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }
}
