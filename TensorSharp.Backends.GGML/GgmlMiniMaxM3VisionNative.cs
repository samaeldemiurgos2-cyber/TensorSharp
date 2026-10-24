// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Binding for TSGgml_MiniMaxM3VisionHeadF32 (ggml_ops_minimax_m3_vision.cpp): the
// MiniMax-M3.1 per-patch projector, 2x2 merge and patch-merge MLP as one GGML graph.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public static partial class GgmlMiniMaxM3VisionNative
{
    private const string DllName = "GgmlOps";
    static GgmlMiniMaxM3VisionNative() => GgmlNative.EnsureImportResolverRegistered();

    /// <summary>Mirrors the native MiniMaxM3VisionHeadDesc; StructBytes guards drift.</summary>
    [StructLayout(LayoutKind.Sequential)]
    private struct HeadDesc
    {
        public int StructBytes;
        public int NumPatches;
        public int Cells;
        public int VisionHidden;
        public int ProjHidden;
        public int TextHidden;
        public int MergeHidden;
        public int Reserved;
        public IntPtr Hidden;
        public IntPtr Output;
        public IntPtr W1, B1, W2, B2, Wf1, Bf1, Wf2, Bf2;
    }

    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_MiniMaxM3VisionHeadF32(in HeadDesc desc);

    /// <summary>
    /// output[tokens, textHidden] = fc2(gelu(fc1(concat4(mm2(gelu(mm1(hidden))))))) with all
    /// weights F32 [out, in] and the biases F32. Every tensor must be a contiguous F32 tensor
    /// on a GGML allocator; the weights are bound device-resident by host pointer, so the
    /// caller invalidates them (<see cref="GgmlBasicOps.InvalidateTensorHostBuffer"/>)
    /// before freeing them. Throws <see cref="EntryPointNotFoundException"/> on a native
    /// build that predates the entry point; returns false with
    /// <see cref="GgmlBasicOps.LastNativeError"/> set when the graph fails.
    /// </summary>
    public static bool TryHead(Tensor hidden, Tensor output, int cells,
        Tensor w1, Tensor b1, Tensor w2, Tensor b2, Tensor wf1, Tensor bf1, Tensor wf2, Tensor bf2)
    {
        foreach (Tensor t in new[] { hidden, output, w1, b1, w2, b2, wf1, bf1, wf2, bf2 })
        {
            if (t == null || t.Storage is not GgmlStorage || t.ElementType != DType.Float32 || !t.IsContiguous())
                throw new ArgumentException("MiniMax-M3 vision head tensors must be contiguous F32 GGML tensors.");
        }
        int patches = checked((int)hidden.Sizes[0]);
        var desc = new HeadDesc
        {
            StructBytes = Marshal.SizeOf<HeadDesc>(),
            NumPatches = patches,
            Cells = cells,
            VisionHidden = checked((int)hidden.Sizes[1]),
            ProjHidden = checked((int)w1.Sizes[0]),
            TextHidden = checked((int)w2.Sizes[0]),
            MergeHidden = checked((int)wf1.Sizes[0]),
            Hidden = TensorComputePrimitives.GetStoragePointer(hidden),
            Output = TensorComputePrimitives.GetStoragePointer(output),
            W1 = TensorComputePrimitives.GetStoragePointer(w1),
            B1 = TensorComputePrimitives.GetStoragePointer(b1),
            W2 = TensorComputePrimitives.GetStoragePointer(w2),
            B2 = TensorComputePrimitives.GetStoragePointer(b2),
            Wf1 = TensorComputePrimitives.GetStoragePointer(wf1),
            Bf1 = TensorComputePrimitives.GetStoragePointer(bf1),
            Wf2 = TensorComputePrimitives.GetStoragePointer(wf2),
            Bf2 = TensorComputePrimitives.GetStoragePointer(bf2),
        };
        if (w1.Sizes[1] != desc.VisionHidden || w2.Sizes[1] != desc.ProjHidden
            || wf1.Sizes[1] != (long)cells * desc.TextHidden || wf2.Sizes[0] != desc.TextHidden
            || wf2.Sizes[1] != desc.MergeHidden || cells <= 0 || patches % cells != 0
            || output.Sizes[0] != patches / cells || output.Sizes[1] != desc.TextHidden)
            throw new ArgumentException("MiniMax-M3 vision head tensor shapes do not chain.");

        // A direct entry point never passes through OpRegistry's rank hook; route it to
        // the device that owns the output, as GgmlBasicOps' direct calls do.
        GgmlNative.SetActiveRankIfUnpinned(((GgmlStorage)output.Storage).DeviceId);
        return TSGgml_MiniMaxM3VisionHeadF32(in desc) != 0;
    }
}
