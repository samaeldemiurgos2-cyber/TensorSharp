// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    internal static partial class GgmlNative
    {
        [LibraryImport(DllName)]
        internal static partial int TSGgml_Qwen4ExpTpWeightSupported(
            int type, int input, int rows, int experts, int degree, int expandTiles);
        [LibraryImport(DllName)]
        internal static partial int TSGgml_Qwen4ExpTpWeightSupportedGlobal(
            int type, int input, int rows, int experts, int degree, int expandTiles, int rankOffset);
        [LibraryImport(DllName)]
        internal static unsafe partial void TSGgml_Qwen4ExpTestDumpTpPlans(IntPtr* plans, int ranks,
            int begin, int end, int tokens, int position, int logitsRows, int nKv);
        [LibraryImport(DllName)]
        internal static partial int TSGgml_Qwen4ExpTokenSpanTp(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr kinds,
            int layerBegin, int layerEnd, IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            int nExpert, int nExpertUsed, int nFf, int nFfSh,
            float eps, int cacheSlot, IntPtr head, IntPtr logitsOut,
            IntPtr ple, int pleLayer, IntPtr pleEmb,
            IntPtr mropePos, IntPtr mropeSections, int ropePosition, int device,
            IntPtr hiddenOut, int logitsRows, IntPtr qsa, IntPtr qsaPositions, int qsaPositionCount, out IntPtr plan);
        [LibraryImport(DllName)]
        internal static partial int TSGgml_Qwen4ExpTokenSpanTpGlobal(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr kinds,
            int layerBegin, int layerEnd, IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            int nExpert, int nExpertUsed, int nFf, int nFfSh,
            float eps, int cacheSlot, IntPtr head, IntPtr logitsOut,
            IntPtr ple, int pleLayer, IntPtr pleEmb,
            IntPtr mropePos, IntPtr mropeSections, int ropePosition, int device,
            IntPtr hiddenOut, int logitsRows, IntPtr qsa, IntPtr qsaPositions, int qsaPositionCount, int globalDegree, int rankOffset, out IntPtr plan);
    }

    public partial class GgmlBasicOps
    {
        public static void RequireQwen4ExpTensorParallelWeight(
            int type, int input, int rows, int experts, int degree, bool expandTiles)
        {
            if (GgmlNative.TSGgml_Qwen4ExpTpWeightSupported(type, input, rows, experts, degree, expandTiles ? 1 : 0) == 0)
                throw new NotSupportedException(LastNativeError("Qwen4Exp TP weight is unsupported."));
        }

        public static void RequireQwen4ExpTensorParallelWeightGlobal(
            int type, int input, int rows, int experts, int degree, bool expandTiles, int rankOffset)
        {
            if (GgmlNative.TSGgml_Qwen4ExpTpWeightSupportedGlobal(type, input, rows, experts, degree, expandTiles ? 1 : 0, rankOffset) == 0)
                throw new NotSupportedException(LastNativeError("Qwen4Exp global TP weight is unsupported."));
        }

        // Explicit diagnostic use only; requires a native test-hook build.
        public static unsafe void Qwen4ExpTestDumpTpPlans(IntPtr[] plans,
            int begin, int end, int tokens, int position, int logitsRows, int nKv)
        {
            fixed (IntPtr* pointer = plans)
                GgmlNative.TSGgml_Qwen4ExpTestDumpTpPlans(pointer, plans.Length,
                    begin, end, tokens, position, logitsRows, nKv);
        }

        public static IntPtr Qwen4ExpTokenSpanTp(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr kinds,
            int layerBegin, int layerEnd,
            IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            int nExpert, int nExpertUsed, int nFf, int nFfSh,
            float eps, int cacheSlot,
            IntPtr head = default, IntPtr logitsOut = default,
            IntPtr ple = default, int pleLayer = -1, IntPtr pleEmb = default,
            IntPtr mropePos = default, IntPtr mropeSections = default,
            int ropePosition = -1,
            // GPU this span's layers live on (layer split). -1 / 0 = the current
            // rank, which is the only rank on a single-GPU run.
            int device = 0, IntPtr hiddenOut = default, int logitsRows = 1,
            IntPtr qsa = default, IntPtr qsaPositions = default, int qsaPositionCount = 0)
        {
            if (GgmlNative.TSGgml_Qwen4ExpTokenSpanTp(ffn, gdn, attn, kinds, layerBegin, layerEnd,
                resData, maskData, nEmbd, hc, hcLowRank, nTokens,
                headKDim, headVDim, nKHeads, nVHeads, dConv,
                headDim, nHead, nHeadKv, kvCapacity, nKv, position,
                nRot, ropeBase, ropeFreqScale, attnScale,
                nExpert, nExpertUsed, nFf, nFfSh, eps, cacheSlot,
                head, logitsOut, ple, pleLayer, pleEmb, mropePos, mropeSections, ropePosition,
                device, hiddenOut, logitsRows, qsa, qsaPositions, qsaPositionCount, out IntPtr plan) == 0)
                throw new InvalidOperationException(LastNativeError("Qwen4Exp TP graph build failed."));
            return plan;
        }
        public static IntPtr Qwen4ExpTokenSpanTpGlobal(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr kinds,
            int layerBegin, int layerEnd,
            IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            int nExpert, int nExpertUsed, int nFf, int nFfSh,
            float eps, int cacheSlot,
            IntPtr head = default, IntPtr logitsOut = default,
            IntPtr ple = default, int pleLayer = -1, IntPtr pleEmb = default,
            IntPtr mropePos = default, IntPtr mropeSections = default,
            int ropePosition = -1,
            // GPU this span's layers live on (layer split). -1 / 0 = the current
            // rank, which is the only rank on a single-GPU run.
            int device = 0, IntPtr hiddenOut = default, int logitsRows = 1,
            IntPtr qsa = default, IntPtr qsaPositions = default, int qsaPositionCount = 0, int globalDegree = 0, int rankOffset = 0)
        {
            if (GgmlNative.TSGgml_Qwen4ExpTokenSpanTpGlobal(ffn, gdn, attn, kinds, layerBegin, layerEnd,
                resData, maskData, nEmbd, hc, hcLowRank, nTokens,
                headKDim, headVDim, nKHeads, nVHeads, dConv,
                headDim, nHead, nHeadKv, kvCapacity, nKv, position,
                nRot, ropeBase, ropeFreqScale, attnScale,
                nExpert, nExpertUsed, nFf, nFfSh, eps, cacheSlot,
                head, logitsOut, ple, pleLayer, pleEmb, mropePos, mropeSections, ropePosition,
                device, hiddenOut, logitsRows, qsa, qsaPositions, qsaPositionCount, globalDegree, rankOffset, out IntPtr plan) == 0)
                throw new InvalidOperationException(LastNativeError("Qwen4Exp TP graph build failed."));
            return plan;
        }
    }
}
