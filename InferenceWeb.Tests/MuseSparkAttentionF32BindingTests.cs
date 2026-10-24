// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class MuseSparkAttentionF32BindingTests
{
    [Fact]
    public void OrdinaryRejectsNarrowedStridedAliasedAndWrongBackendBuffers()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var query = new Tensor(allocator, DType.Float32, 2, 2, 4);
        using var key = new Tensor(allocator, DType.Float32, 1, 3, 4);
        using var value = new Tensor(allocator, DType.Float32, 1, 3, 4);
        using var output = new Tensor(allocator, DType.Float32, 2, 8);
        using var half = new Tensor(allocator, DType.Float16, 1, 3, 4);
        using var padded = new Tensor(allocator, DType.Float32, 2, 9);
        using var strided = padded.Narrow(1, 0, 8);
        using var shortQuery = new Tensor(allocator, DType.Float32, 15);
        using var negativeOffset = new Tensor([2, 8], [8, 1], query.Storage, -1);
        using var oversizedSpan = new Tensor([2, 8], [8, 1], query.Storage, query.Storage.ElementCount - 15);
        output.SetElementsAsFloat(Enumerable.Repeat(37f, 16).ToArray());
        void Run(Tensor q, Tensor k, Tensor result, int format = 0) =>
            GgmlBasicOps.MuseSparkPrefillAttentionF32(q, k, value, result, 2, 1, 4, 2, 3, 1, 0, 0.5f, format);
        Assert.Throws<ArgumentException>(() => Run(query, half, output));
        Assert.Throws<ArgumentException>(() => Run(strided, key, output));
        Assert.Throws<ArgumentException>(() => Run(shortQuery, key, output));
        Assert.Throws<ArgumentException>(() => Run(negativeOffset, key, output));
        Assert.Throws<ArgumentException>(() => Run(oversizedSpan, key, output));
        Assert.Throws<ArgumentException>(() => Run(query, key, query));
        Assert.Throws<ArgumentException>(() => Run(query, key, output, 2));
        Assert.Throws<ArgumentException>(() => GgmlBasicOps.MuseSparkPrefillAttentionF32(query, key, value, output,
            2, 1, 2049, 2, 3, 1, 0, 0.5f));
        // Valid host spans still do not authorize pretending a CPU allocation
        // executed the explicit CUDA-only operation.
        Assert.Throws<NotSupportedException>(() => Run(query, key, output));
        Assert.All(output.GetElementsAsFloat(16), element => Assert.Equal(37f, element));
    }

    [Fact]
    public void PagedRejectsUnsafeLengthsAndAliasingBeforeNativeAccess()
    {
        float[] q = new float[16], k = new float[16], v = new float[16];
        float[] output = Enumerable.Repeat(41f, 16).ToArray();
        void Run(float[] query, float[] keys, float[] result, int[] starts, int[] positions, int[] blocks, int sequences = 1) =>
            GgmlBasicOps.MuseSparkPagedAttentionF32(query, keys, v, result,
                starts, [3], positions, blocks, [0], sequences, 2, 2, 1, 4, 2, 0.5f);
        Assert.Throws<ArgumentException>(() => Run(q[..15], k, output, [0, 2], [1, 2], [0, 1]));
        Assert.Throws<ArgumentException>(() => Run(q, k[..15], output, [0, 2], [1, 2], [0, 1]));
        Assert.Throws<ArgumentException>(() => Run(q, k, output, [0], [1, 2], [0, 1]));
        Assert.Throws<ArgumentException>(() => Run(q, k, output, [0, 2], [1], [0, 1]));
        Assert.Throws<ArgumentException>(() => Run(q, k, output, [0, 2], [1, 2], []));
        Assert.Throws<ArgumentException>(() => Run(q, k, output, [0, 2], [1, 2], [0, 1], 2));
        Assert.Throws<ArgumentException>(() => Run(q, k, q, [0, 2], [1, 2], [0, 1]));
        Assert.Throws<ArgumentException>(() => Run(q, k, k, [0, 2], [1, 2], [0, 1]));
        Assert.All(output, element => Assert.Equal(41f, element));
    }
}
