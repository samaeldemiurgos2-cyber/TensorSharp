// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class GgmlTensorParallelViewTests
{
    [MuseSparkTensorParallelGpuFact(2)]
    public void OffsetViewsPreserveSentinelsWithDifferentBackingStorageSizes()
    {
        const int largeCount = 1_048_580;
        Assert.InRange(GgmlTensorParallelGroup.DeviceAllReduceThreshold, 0, largeCount);
        WithGroup(group =>
        {
            // Exercise the host and large-payload routes. The logical views
            // have equal lengths, while their offsets and backing sizes differ.
            foreach (int count in new[] { 16, largeCount })
            foreach (bool exact in new[] { false, true })
            {
                var buffers = new Tensor[2];
                var views = new Tensor[2];
                var originals = new float[2][];
                int[] offsets = [3, 5];
                try
                {
                    for (int rank = 0; rank < 2; rank++)
                    {
                        originals[rank] = Enumerable.Repeat(-1000f - rank, count + 9 + rank * 4).ToArray();
                        // Exactly represented integers keep the ordinary
                        // collective's optional wire compression irrelevant.
                        for (int i = 0; i < count; i++)
                            originals[rank][offsets[rank] + i] = rank + 1;
                        buffers[rank] = new Tensor(group.GetAllocator(rank), DType.Float32, originals[rank].Length);
                        buffers[rank].SetElementsAsFloat(originals[rank]);
                        views[rank] = buffers[rank].Narrow(0, offsets[rank], count);
                    }
                    if (exact) group.AllReduceFloat32(views); else group.AllReduce(views);
                    for (int rank = 0; rank < 2; rank++)
                    {
                        float[] actual = new float[originals[rank].Length];
                        buffers[rank].CopyToArray(actual);
                        for (int i = 0; i < actual.Length; i++)
                            Assert.Equal(i >= offsets[rank] && i < offsets[rank] + count
                                ? 3f : originals[rank][i], actual[i]);
                    }
                }
                finally
                {
                    foreach (var view in views) view?.Dispose();
                    foreach (var buffer in buffers) buffer?.Dispose();
                }
            }
        });
    }

    [MuseSparkTensorParallelGpuFact(2)]
    public void UnequalLogicalCountsAndStridedViewsAreRejectedBeforeWriting()
    {
        WithGroup(group =>
        {
            using var left = new Tensor(group.GetAllocator(0), DType.Float32, 8);
            using var right = new Tensor(group.GetAllocator(1), DType.Float32, 8);
            left.SetElementsAsFloat(Enumerable.Repeat(11f, 8).ToArray());
            right.SetElementsAsFloat(Enumerable.Repeat(29f, 8).ToArray());
            using var shorter = left.Narrow(0, 1, 3);
            using var longer = right.Narrow(0, 2, 4);
            using var matrix = left.View(2, 4);
            using var strided = matrix.Transpose();
            using var outside = new Tensor([4L], [1L], left.Storage, 6);
            using var negativeOffset = new Tensor([4L], [1L], left.Storage, -1);
            foreach (bool exact in new[] { false, true })
            {
                void Reduce(Tensor[] values)
                {
                    if (exact) group.AllReduceFloat32(values); else group.AllReduce(values);
                }
                Assert.Throws<ArgumentException>(() => Reduce([shorter, longer]));
                Assert.Throws<ArgumentException>(() => Reduce([strided, right]));
                Assert.Throws<ArgumentException>(() => Reduce([outside, longer]));
                Assert.Throws<ArgumentException>(() => Reduce([negativeOffset, longer]));
                float[] first = new float[8], second = new float[8];
                left.CopyToArray(first);
                right.CopyToArray(second);
                Assert.All(first, value => Assert.Equal(11f, value));
                Assert.All(second, value => Assert.Equal(29f, value));
            }
        });
    }

    private static void WithGroup(Action<GgmlTensorParallelGroup> action)
    {
        var context = new GgmlContext([0, 1], GgmlBackendType.Cuda);
        using var group = new GgmlTensorParallelGroup(context);
        try { action(group); }
        finally
        {
            group.Dispose();
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
            context.ReleasePoolForDisposal();
        }
    }
}
