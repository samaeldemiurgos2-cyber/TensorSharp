// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the native MiniMax-M3.1 whole-model executor
// (TensorSharp.GGML.Native/ggml_ops_minimax_m3_exec.cpp). The ABI is Muse Spark's
// (GgmlMuseSparkNative) under the TSGgml_MiniMaxM3 prefix: the native side loads
// the split GGUF itself, places whole layers across the visible GPUs, serves
// offloaded routed experts from the GGUF mapping, owns every sequence slot's
// K, V and MSA index-key rows, and runs prefill / decode ubatches through
// ggml_backend_sched.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static class GgmlMiniMaxM3Native
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Conv = CallingConvention.Cdecl;

        static GgmlMiniMaxM3Native()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern IntPtr TSGgml_MiniMaxM3LoadModelTp([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName, int ctxIsHardLimit,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string layerSplit, int tp);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_MiniMaxM3SetOption([MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string value);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_MiniMaxM3Free(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3VocabSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3CtxSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3HiddenSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3NPast(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_MiniMaxM3Forward(IntPtr handle, int* tokens, int nTokens, float* logitsOut, int allRows);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3ForwardBatchedDecode(IntPtr handle, int n, int[] slotIds,
            int[] tokens, int[] positions, float[] logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3ForwardSegments(IntPtr handle, int n, int[] slotIds, int[] lens,
            int[] tokens, int[] positions, float[] logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3Ubatch(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_MiniMaxM3QueueVisionRows(IntPtr handle, float* rows, int nRows, int index);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_MiniMaxM3ClearVisionRows(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3ResetChecked(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3Rewind(IntPtr handle, int nPast);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3SlotAlloc(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3SetActiveSlot(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3SlotFree(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3SlotNPast(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3SlotCopy(IntPtr handle, int srcSlot, int dstSlot);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3SetReclaimableSlots(IntPtr handle, int[] slotIds, int n);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3TakeReclaimedSlots(IntPtr handle, int[] slotIds, int cap);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3GetStats(IntPtr handle, long[] stats, int cap);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MiniMaxM3PipelinePlan(IntPtr handle, long p0, long n);

        /// <summary>Muse Spark's twelve counters (see <see cref="GetStats"/>); the executor may append more.</summary>
        public const int BaseStatsCount = 12;

        /// <summary>Counters <see cref="GetStats"/> asks for: the base twelve plus room for appended ones.</summary>
        private const int StatsCapacity = 32;

        /// <summary>Index of the attention-mode counter (0 auto, 1 dense, 2 ggml, 3 kernel).</summary>
        public const int StatAttentionMode = 12;

        /// <summary>Index of the count of layers the executor runs with sparse (MSA) attention.</summary>
        public const int StatSparseLayers = 15;

        /// <param name="nGpu">GPUs to spread the layers over; 0 = every visible device.</param>
        /// <param name="nCpuMoe">Leading layers whose routed experts stay in system RAM; -1 offloads the fewest that fit.</param>
        /// <param name="backendName">ggml backend registry name ("CUDA", "CPU").</param>
        /// <param name="layerSplit">Optional comma-separated layer count per GPU.</param>
        /// <param name="tp">GPUs to split every layer over; 0 or 1 for a layer split over <paramref name="nGpu"/> GPUs.</param>
        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            string backendName, bool ctxIsHardLimit, string layerSplit, int tp = 0)
            => TSGgml_MiniMaxM3LoadModelTp(ggufPath, nGpu, nCtx, nUbatch, nThreads, nCpuMoe, backendName ?? string.Empty,
                ctxIsHardLimit ? 1 : 0, layerSplit ?? string.Empty, tp);

        /// <summary>Override (or with a null value, clear) a native option. Takes precedence over the
        /// process environment, which .NET cannot update for native code on Linux.</summary>
        public static void SetOption(string name, string value) => TSGgml_MiniMaxM3SetOption(name, value);

        public static void Free(IntPtr handle) => TSGgml_MiniMaxM3Free(handle);
        public static int VocabSize(IntPtr handle) => TSGgml_MiniMaxM3VocabSize(handle);
        public static int CtxSize(IntPtr handle) => TSGgml_MiniMaxM3CtxSize(handle);
        public static int HiddenSize(IntPtr handle) => TSGgml_MiniMaxM3HiddenSize(handle);
        public static int NPast(IntPtr handle) => TSGgml_MiniMaxM3NPast(handle);

        /// <summary>Evaluate tokens on the active slot. <paramref name="logitsOut"/> receives the last row's
        /// logits, or every row's when <paramref name="allRows"/>.</summary>
        public static unsafe bool Forward(IntPtr handle, ReadOnlySpan<int> tokens, Span<float> logitsOut, bool allRows)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
                return TSGgml_MiniMaxM3Forward(handle, t, tokens.Length, l, allRows ? 1 : 0) != 0;
        }

        public static bool ForwardBatchedDecode(IntPtr handle, int[] slots, int[] tokens, int[] positions, float[] logits)
            => TSGgml_MiniMaxM3ForwardBatchedDecode(handle, slots.Length, slots, tokens, positions, logits) != 0;

        /// <summary>Several slots' consecutive-token segments in one graph. <paramref name="tokens"/> holds the
        /// segments back to back; <paramref name="logits"/> receives one row per segment (its last token).</summary>
        public static bool ForwardSegments(IntPtr handle, int[] slots, int[] lens, int[] tokens, int[] positions, float[] logits)
            => TSGgml_MiniMaxM3ForwardSegments(handle, slots.Length, slots, lens, tokens, positions, logits) != 0;

        /// <summary>Most tokens one native graph evaluates.</summary>
        public static int Ubatch(IntPtr handle) => TSGgml_MiniMaxM3Ubatch(handle);

        /// <summary>Queue <paramref name="nRows"/> hidden-size rows that replace the token embeddings at absolute
        /// positions <paramref name="index"/> .. <paramref name="index"/> + <paramref name="nRows"/> - 1 of the active
        /// slot's sequence in the next <see cref="Forward"/> call (unlike Muse Spark's call-relative index). Rows
        /// outside that call are dropped and the queue empties after every Forward.</summary>
        public static unsafe bool QueueVisionRows(IntPtr handle, ReadOnlySpan<float> rows, int nRows, int index)
        {
            fixed (float* r = rows)
                return TSGgml_MiniMaxM3QueueVisionRows(handle, r, nRows, index) != 0;
        }

        public static void ClearVisionRows(IntPtr handle) => TSGgml_MiniMaxM3ClearVisionRows(handle);
        public static bool ResetChecked(IntPtr handle) => TSGgml_MiniMaxM3ResetChecked(handle) != 0;
        public static bool Rewind(IntPtr handle, int nPast) => TSGgml_MiniMaxM3Rewind(handle, nPast) != 0;
        public static int SlotAlloc(IntPtr handle) => TSGgml_MiniMaxM3SlotAlloc(handle);
        public static bool SetActiveSlot(IntPtr handle, int slot) => TSGgml_MiniMaxM3SetActiveSlot(handle, slot) != 0;
        public static bool SlotFree(IntPtr handle, int slot) => TSGgml_MiniMaxM3SlotFree(handle, slot) != 0;
        public static int SlotNPast(IntPtr handle, int slot) => TSGgml_MiniMaxM3SlotNPast(handle, slot);

        /// <summary>Make <paramref name="dstSlot"/> an independent device copy of <paramref name="srcSlot"/>
        /// (its K, V and index-key rows and its head).</summary>
        public static bool SlotCopy(IntPtr handle, int srcSlot, int dstSlot) => TSGgml_MiniMaxM3SlotCopy(handle, srcSlot, dstSlot) != 0;
        public static bool SetReclaimableSlots(IntPtr handle, int[] slots)
            => TSGgml_MiniMaxM3SetReclaimableSlots(handle, slots ?? Array.Empty<int>(), slots?.Length ?? 0) != 0;
        public static int TakeReclaimedSlots(IntPtr handle, int[] buffer)
            => TSGgml_MiniMaxM3TakeReclaimedSlots(handle, buffer, buffer.Length);

        /// <summary>Micro-batch rows a prompt of <paramref name="n"/> tokens starting at position
        /// <paramref name="p0"/> would prefill pipelined with; 0 when it would run one ubatch at a time.</summary>
        public static int PipelinePlan(IntPtr handle, long p0, long n) => TSGgml_MiniMaxM3PipelinePlan(handle, p0, n);

        /// <summary>[graphs built, graphs reused, host weight bytes, CPU-MoE layers, GPUs, slots,
        /// pipelined prefill calls, their micro-batches, pipelined prefills declined,
        /// pipelined prefill micro-batch rows (0: off), GPU stages a pipelined prefill spans (0: off),
        /// tensor-parallel degree (0: layer split)], then MiniMax-M3's own: [<see cref="StatAttentionMode"/>]
        /// attention mode (0 auto, 1 dense, 2 ggml, 3 kernel), [13] segments computed with sparse attention,
        /// [14] segments computed with dense attention, [<see cref="StatSparseLayers"/>] sparse (MSA) layers,
        /// [16] the sparse layers that run TensorSharp's MSA kernels.
        /// Null when the executor reports fewer than <see cref="BaseStatsCount"/>.</summary>
        public static long[] GetStats(IntPtr handle)
        {
            var stats = new long[StatsCapacity];
            int n = TSGgml_MiniMaxM3GetStats(handle, stats, stats.Length);
            return n >= BaseStatsCount && n <= stats.Length ? stats.AsSpan(0, n).ToArray() : null;
        }
    }
}
