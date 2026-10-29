// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Every multi-GPU GgmlContext re-runs the native device-group init, which frees each
// rank's ggml backend and creates a new one (rank 0's too when its GPU changes). The
// tensor-parallel communicator - ggml's NCCL / pinned-host collective and TensorSharp's
// exact F32 NCCL transport - was resolved once per process and kept the backends it had
// been built over. So the second group in a process reduced through freed backends: on a
// real 2-GPU host the ggml_cuda test lane died with SIGSEGV in
// ggml_backend_cuda_comm_allreduce_nccl (it read a freed backend's context), whichever
// test happened to issue the first large device AllReduce of a later group; and the exact
// F32 transport refused every collective of a regrouped device set as "a different device
// group". Device emulation on one GPU (GGML_CUDA_DEVICES=2) never runs NCCL, so only real
// hardware showed it. A device-group init now releases the communicator before it
// replaces any backend, and the next collective is built over the new ones. The verdict
// behind GgmlContext.HasDeviceAllReduce, a probe of that communicator, goes with it.
using System;
using System.Reflection;
using TensorSharp.GGML;
using Xunit;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class GgmlTensorParallelRegroupTests
{
    [MuseSparkTensorParallelGpuFact(2)]
    public void EachNewDeviceGroupReducesThroughACollectiveBuiltOverItsOwnBackends()
    {
        // Past the device threshold and past NCCL's FP32 cutoff, so the reduction goes
        // through the collective's wire path, which touches every rank's backend stream.
        const int count = 1 << 20;
        Assert.InRange(GgmlTensorParallelGroup.DeviceAllReduceThreshold, 0, count);

        // The same two GPUs three times. The middle group swaps the ranks, so every rank's
        // backend is replaced by one on the other GPU; the last group puts them back.
        foreach (int[] devices in new[] { new[] { 0, 1 }, new[] { 1, 0 }, new[] { 0, 1 } })
        {
            var context = new GgmlContext(devices, GgmlBackendType.Cuda);
            string group = $"[{string.Join(", ", devices)}]";
            try
            {
                Assert.True(context.HasDeviceAllReduce, $"group {group} reports no device collective");
                // Exact first: with a stale communicator it is refused, where the ordinary
                // collective would dereference the freed backends and take the process down.
                foreach (bool exact in new[] { true, false })
                {
                    float[][] partials = new float[2][];
                    for (int rank = 0; rank < 2; rank++)
                    {
                        partials[rank] = new float[count];
                        // Small integers: exact through the ordinary collective's BF16 wire
                        // compression, and different per element so a misplaced chunk shows.
                        for (int i = 0; i < count; i++)
                            partials[rank][i] = i % 7 + 8 * rank;
                    }

                    bool reduced = ReduceOnDevice(partials, count, exact);
                    Assert.True(reduced,
                        $"group {group}: the {(exact ? "exact F32" : "ordinary")} device AllReduce was refused " +
                        $"({Native("LastNativeError").Invoke(null, ["(no native error)"])})");
                    for (int rank = 0; rank < 2; rank++)
                        for (int i = 0; i < count; i++)
                        {
                            float expected = 2 * (i % 7) + 8;
                            if (partials[rank][i] != expected)
                                Assert.Fail($"group {group}, {(exact ? "exact" : "ordinary")}: rank {rank} element {i} " +
                                            $"is {partials[rank][i]}, expected {expected}");
                        }
                }
            }
            finally
            {
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
                context.ReleasePoolForDisposal();
            }
        }
    }

    /// <summary>
    /// HasDeviceAllReduce comes from one real AllReduce on the communicator, and that answer
    /// was kept for the life of the process: every later device group got the first one's,
    /// whether or not its own collective could reduce.
    /// </summary>
    [MuseSparkTensorParallelGpuFact(2)]
    public void TheDeviceCollectiveVerdictBelongsToTheCurrentDeviceGroup()
    {
        // One rank has nothing to reduce across.
        Native("TensorParallelInit").Invoke(null, [GgmlBackendType.Cuda, new[] { 0 }, true]);
        Assert.False((bool)Native("TensorParallelHasDeviceAllReduce").Invoke(null, null)!,
            "a one-rank device group reports a device collective (a verdict left by an earlier group)");

        var context = new GgmlContext([0, 1], GgmlBackendType.Cuda);
        try
        {
            Assert.True(context.HasDeviceAllReduce,
                "a two-GPU group reports no device collective (the one-rank group's verdict)");
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
            context.ReleasePoolForDisposal();
        }
    }

    // The managed group falls back to the host sum whenever the device route declines, so
    // the route is called directly: a refusal has to fail the test, not hide behind a
    // correct host result.
    private static unsafe bool ReduceOnDevice(float[][] partials, long count, bool exact)
    {
        MethodInfo reduce = Native(exact ? "TensorParallelAllReduceDeviceF32" : "TensorParallelAllReduceDevice");
        fixed (float* rank0 = partials[0])
        fixed (float* rank1 = partials[1])
        {
            float** buffers = stackalloc float*[2];
            buffers[0] = rank0;
            buffers[1] = rank1;
            return (bool)reduce.Invoke(null, [Pointer.Box(buffers, typeof(float**)), 2, count])!;
        }
    }

    private static MethodInfo Native(string name) =>
        typeof(GgmlBasicOps).Assembly.GetType("TensorSharp.GGML.GgmlNative", throwOnError: true)!
            .GetMethod(name, BindingFlags.Public | BindingFlags.Static)
        ?? throw new MissingMethodException("TensorSharp.GGML.GgmlNative", name);
}
