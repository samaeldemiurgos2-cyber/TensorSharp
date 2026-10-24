// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// P/Invoke surface for the native Muse Spark whole-model executor
// (TensorSharp.GGML.Native/ggml_ops_muse_spark_exec.cpp). The native side loads
// the split GGUF itself, places whole layers across the visible GPUs, serves
// offloaded routed experts from the GGUF mapping, owns every sequence slot's
// K/V rows, and runs prefill / decode ubatches through ggml_backend_sched.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML
{
    public static class GgmlMuseSparkNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention Conv = CallingConvention.Cdecl;

        static GgmlMuseSparkNative()
        {
            GgmlNative.EnsureImportResolverRegistered();
        }

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern IntPtr TSGgml_MuseSparkLoadModelTp([MarshalAs(UnmanagedType.LPUTF8Str)] string ggufPath,
            int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string backendName, int ctxIsHardLimit,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string layerSplit, int tp);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_MuseSparkSetOption([MarshalAs(UnmanagedType.LPUTF8Str)] string name,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string value);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_MuseSparkFree(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkVocabSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkCtxSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkHiddenSize(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkNPast(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_MuseSparkForward(IntPtr handle, int* tokens, int nTokens, float* logitsOut, int allRows);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkForwardBatchedDecode(IntPtr handle, int n, int[] slotIds,
            int[] tokens, int[] positions, float[] logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkForwardSegments(IntPtr handle, int n, int[] slotIds, int[] lens,
            int[] tokens, int[] positions, float[] logitsOut);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkUbatch(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern unsafe int TSGgml_MuseSparkQueueVisionRows(IntPtr handle, float* rows, int nRows, int index);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern void TSGgml_MuseSparkClearVisionRows(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkResetChecked(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkRewind(IntPtr handle, int nPast);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkSlotAlloc(IntPtr handle);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkSetActiveSlot(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkSlotFree(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkSlotNPast(IntPtr handle, int slotId);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkSlotCopy(IntPtr handle, int srcSlot, int dstSlot);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkSetReclaimableSlots(IntPtr handle, int[] slotIds, int n);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkTakeReclaimedSlots(IntPtr handle, int[] slotIds, int cap);

        [DllImport(DllName, CallingConvention = Conv)]
        private static extern int TSGgml_MuseSparkGetStats(IntPtr handle, long[] stats, int cap);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int TSGgml_MuseSparkPipelinePlan(IntPtr handle, long p0, long n);

        /// <param name="nGpu">GPUs to spread the layers over; 0 = every visible device.</param>
        /// <param name="nCpuMoe">Leading layers whose routed experts stay in system RAM; -1 offloads the fewest that fit.</param>
        /// <param name="backendName">ggml backend registry name ("CUDA", "Vulkan", "CPU").</param>
        /// <param name="layerSplit">Optional comma-separated layer count per GPU.</param>
        /// <param name="tp">GPUs to split every layer over (tensor parallelism through ggml's meta
        /// backend, CUDA only); 0 or 1 for a layer split over <paramref name="nGpu"/> GPUs.</param>
        public static IntPtr LoadModel(string ggufPath, int nGpu, int nCtx, int nUbatch, int nThreads, int nCpuMoe,
            string backendName, bool ctxIsHardLimit, string layerSplit, int tp = 0)
            => TSGgml_MuseSparkLoadModelTp(ggufPath, nGpu, nCtx, nUbatch, nThreads, nCpuMoe, backendName ?? string.Empty,
                ctxIsHardLimit ? 1 : 0, layerSplit ?? string.Empty, tp);

        /// <summary>Override (or with a null value, clear) a native option. Takes precedence over the
        /// process environment, which .NET cannot update for native code on Linux.</summary>
        public static void SetOption(string name, string value) => TSGgml_MuseSparkSetOption(name, value);

        public static void Free(IntPtr handle) => TSGgml_MuseSparkFree(handle);
        public static int VocabSize(IntPtr handle) => TSGgml_MuseSparkVocabSize(handle);
        public static int CtxSize(IntPtr handle) => TSGgml_MuseSparkCtxSize(handle);
        public static int HiddenSize(IntPtr handle) => TSGgml_MuseSparkHiddenSize(handle);
        public static int NPast(IntPtr handle) => TSGgml_MuseSparkNPast(handle);

        /// <summary>Evaluate tokens on the active slot. <paramref name="logitsOut"/> receives the last row's
        /// logits, or every row's when <paramref name="allRows"/>.</summary>
        public static unsafe bool Forward(IntPtr handle, ReadOnlySpan<int> tokens, Span<float> logitsOut, bool allRows)
        {
            fixed (int* t = tokens)
            fixed (float* l = logitsOut)
                return TSGgml_MuseSparkForward(handle, t, tokens.Length, l, allRows ? 1 : 0) != 0;
        }

        public static bool ForwardBatchedDecode(IntPtr handle, int[] slots, int[] tokens, int[] positions, float[] logits)
            => TSGgml_MuseSparkForwardBatchedDecode(handle, slots.Length, slots, tokens, positions, logits) != 0;

        /// <summary>Several slots' consecutive-token segments in one graph. <paramref name="tokens"/> holds the
        /// segments back to back; <paramref name="logits"/> receives one row per segment (its last token).</summary>
        public static bool ForwardSegments(IntPtr handle, int[] slots, int[] lens, int[] tokens, int[] positions, float[] logits)
            => TSGgml_MuseSparkForwardSegments(handle, slots.Length, slots, lens, tokens, positions, logits) != 0;

        /// <summary>Most tokens one native graph evaluates.</summary>
        public static int Ubatch(IntPtr handle) => TSGgml_MuseSparkUbatch(handle);

        public static unsafe bool QueueVisionRows(IntPtr handle, ReadOnlySpan<float> rows, int nRows, int index)
        {
            fixed (float* r = rows)
                return TSGgml_MuseSparkQueueVisionRows(handle, r, nRows, index) != 0;
        }

        public static void ClearVisionRows(IntPtr handle) => TSGgml_MuseSparkClearVisionRows(handle);
        public static bool ResetChecked(IntPtr handle) => TSGgml_MuseSparkResetChecked(handle) != 0;
        public static bool Rewind(IntPtr handle, int nPast) => TSGgml_MuseSparkRewind(handle, nPast) != 0;
        public static int SlotAlloc(IntPtr handle) => TSGgml_MuseSparkSlotAlloc(handle);
        public static bool SetActiveSlot(IntPtr handle, int slot) => TSGgml_MuseSparkSetActiveSlot(handle, slot) != 0;
        public static bool SlotFree(IntPtr handle, int slot) => TSGgml_MuseSparkSlotFree(handle, slot) != 0;
        public static int SlotNPast(IntPtr handle, int slot) => TSGgml_MuseSparkSlotNPast(handle, slot);

        /// <summary>Make <paramref name="dstSlot"/> an independent device copy of <paramref name="srcSlot"/>.</summary>
        public static bool SlotCopy(IntPtr handle, int srcSlot, int dstSlot) => TSGgml_MuseSparkSlotCopy(handle, srcSlot, dstSlot) != 0;
        public static bool SetReclaimableSlots(IntPtr handle, int[] slots)
            => TSGgml_MuseSparkSetReclaimableSlots(handle, slots ?? Array.Empty<int>(), slots?.Length ?? 0) != 0;
        public static int TakeReclaimedSlots(IntPtr handle, int[] buffer)
            => TSGgml_MuseSparkTakeReclaimedSlots(handle, buffer, buffer.Length);

        /// <summary>[graphs built, graphs reused, host weight bytes, CPU-MoE layers, GPUs, slots,
        /// pipelined prefill calls, their micro-batches, pipelined prefills declined,
        /// pipelined prefill micro-batch rows (0: off), GPU stages a pipelined prefill spans (0: off),
        /// tensor-parallel degree (0: layer split)].</summary>
        /// <summary>Micro-batch rows a prompt of <paramref name="n"/> tokens starting at position
        /// <paramref name="p0"/> would prefill pipelined with; 0 when it would run one ubatch at a time.</summary>
        public static int PipelinePlan(IntPtr handle, long p0, long n) => TSGgml_MuseSparkPipelinePlan(handle, p0, n);

        public static long[] GetStats(IntPtr handle)
        {
            var stats = new long[12];
            return TSGgml_MuseSparkGetStats(handle, stats, stats.Length) == 12 ? stats : null;
        }
    }
}
