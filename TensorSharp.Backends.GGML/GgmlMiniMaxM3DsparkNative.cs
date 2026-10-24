// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the MiniMax-M3.1 executor's DSpark drafter
// (TensorSharp.GGML.Native/ggml_ops_minimax_m3_dspark.inc). The drafter is loaded with
// the target (it is placed beside the target's output head, which it reads), commits
// its sliding-window context inside every trunk graph, and drafts one whole block per
// call for the active slot.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static class GgmlMiniMaxM3DsparkNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Conv = CallingConvention.Cdecl;

        static GgmlMiniMaxM3DsparkNative()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern IntPtr TSGgml_MiniMaxM3LoadModelDraft([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName, int ctxIsHardLimit,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string layerSplit, int tp,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string draftPath);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3DsparkBlockSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_MiniMaxM3DsparkDraft(IntPtr handle, int anchorToken, int* toksOut, float* confOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3DsparkStats(IntPtr handle, long[] output, int capacity);

        /// <summary>Indices into <see cref="Stats"/>.</summary>
        public const int StatBlock = 0, StatRingRows = 1, StatDrafts = 2, StatDraftMicroseconds = 3, StatRings = 4,
            StatDevice = 5, StatGraphBuilds = 6, StatLayers = 7, StatTaps = 8, StatsCount = 9;

        /// <summary><see cref="GgmlMiniMaxM3Native.LoadModel"/> with a DSpark drafter GGUF loaded beside the
        /// output head. A drafter that does not match the target refuses the load (zero handle, the reason
        /// in the native last error).</summary>
        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            string backendName, bool ctxIsHardLimit, string layerSplit, int tp, string draftPath)
            => TSGgml_MiniMaxM3LoadModelDraft(ggufPath, nGpu, nCtx, nUbatch, nThreads, nCpuMoe, backendName ?? string.Empty,
                ctxIsHardLimit ? 1 : 0, layerSplit ?? string.Empty, tp, draftPath ?? string.Empty);

        /// <summary>Rows one draft proposes (the drafter's block size); 0 without a drafter.</summary>
        public static int BlockSize(IntPtr handle) => TSGgml_MiniMaxM3DsparkBlockSize(handle);

        /// <summary>
        /// One draft for the active slot from <paramref name="anchorToken"/> (the sampled token at the
        /// slot's head, not yet forwarded): fills <paramref name="toksOut"/> and <paramref name="confOut"/>
        /// (at least <see cref="BlockSize"/> long) and returns the count, 0 when the slot cannot draft now.
        /// </summary>
        public static unsafe int Draft(IntPtr handle, int anchorToken, int[] toksOut, float[] confOut)
        {
            int block = TSGgml_MiniMaxM3DsparkBlockSize(handle);
            if (block <= 0 || toksOut == null || confOut == null || toksOut.Length < block || confOut.Length < block)
                return 0;
            fixed (int* t = toksOut)
            fixed (float* c = confOut)
                return TSGgml_MiniMaxM3DsparkDraft(handle, anchorToken, t, c);
        }

        /// <summary>[block, ring rows, drafts, draft microseconds, sequences with a ring, device, draft graphs
        /// built, draft layers, tapped target layers]; null when the executor reports fewer.</summary>
        public static long[] Stats(IntPtr handle)
        {
            var stats = new long[StatsCount];
            return TSGgml_MiniMaxM3DsparkStats(handle, stats, stats.Length) == StatsCount ? stats : null;
        }
    }
}
