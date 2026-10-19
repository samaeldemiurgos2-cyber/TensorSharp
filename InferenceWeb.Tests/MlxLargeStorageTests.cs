using TensorSharp;
using TensorSharp.MLX;

namespace InferenceWeb.Tests;

public sealed class MlxLargeStorageTests
{
    [Fact]
    public void SliceBoxesCoverEverySmallFlatIntervalInOrder()
    {
        int[] shape = [2, 3, 4];
        for (long start = 0; start <= 24; ++start)
            for (long length = 0; length <= 24 - start; ++length)
                AssertBoxes(shape, start, length);
    }

    [Theory]
    [InlineData(0L, 17_246_978_048L)]
    [InlineData(2_147_483_640L, 4_294_967_311L)]
    [InlineData(17_246_978_040L, 8L)]
    [InlineData(17_246_978_048L, 0L)]
    public void SliceBoxesPreserveLongOffsetsAndLengths(long start, long length)
    {
        AssertBoxes([257, 1024, 1024, 64], start, length);
    }

    [Fact]
    public void SliceBoxesRejectOutOfRangeIntervalsWithoutIntegerOverflow()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => MlxStorage.GetSliceBoxes([2, 3], -1, 1).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => MlxStorage.GetSliceBoxes([2, 3], 4, long.MaxValue).ToArray());
        Assert.Throws<ArgumentOutOfRangeException>(() => MlxStorage.GetSliceBoxes([2, 3], long.MaxValue, 1).ToArray());
    }

    [MlxFact]
    public void LargeVideoStorageRetainsItsShapeThroughCopyAndSliceUpdates()
    {
        // All arrays remain lazy. Metadata alone reproduces the old flattening
        // failure without allocating a multi-gigabyte device or managed buffer.
        using var allocator = new MlxAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 257, 1024, 1024, 16);
        long count = 257L * 1024 * 1024 * 16;
        Ops.Fill(tensor, 1f);
        AssertNativeShape(tensor, [257, 1024, 1024, 16], count);

        using var copy = Ops.NewContiguous(tensor);
        AssertNativeShape(copy, [257, 1024, 1024, 16], count);

        using var flatSlice = new Tensor([8L], [1L], copy.Storage, (long)int.MaxValue + 1);
        Ops.Fill(flatSlice, 2f);
        AssertNativeShape(copy, [257, 1024, 1024, 16], count);

        using var box = copy.Narrow(2, 1, 2);
        Ops.Fill(box, 3f);
        AssertNativeShape(copy, [257, 1024, 1024, 16], count);
    }

    [MlxFact]
    public void LargeUninitializedStorageUsesTheTensorShapeForItsFirstDeviceView()
    {
        using var allocator = new MlxAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 257, 1024, 1024, 16);
        tensor.EnsureDeviceCurrent();
        AssertNativeShape(tensor, [257, 1024, 1024, 16], tensor.ElementCount());
    }

    [MlxFact]
    public void LargeDeviceReplacementRejectsAnArrayWithTheWrongElementCount()
    {
        using var allocator = new MlxAllocator();
        using var tensor = new Tensor(allocator, DType.Float32, 257, 1024, 1024, 16);
        var wrong = MlxNative.Full([2, 3], 1f, DType.Float32);
        try
        {
            Assert.Throws<ArgumentException>(() => ((MlxStorage)tensor.Storage).ReplaceDeviceArray(wrong));
        }
        finally { MlxNative.FreeArray(wrong); }
    }

    [MlxFact]
    public unsafe void OwnedStridedViewEvaluatesTransposedNegativeAndBroadcastStrides()
    {
        using var allocator = new MlxAllocator();
        float[] values = Enumerable.Range(0, 12).Select(i => (float)i).ToArray();
        MlxNative.MlxArray source;
        fixed (float* pointer = values)
            source = MlxNative.NewArrayFromHost((IntPtr)pointer, [3, 4], DType.Float32);
        try
        {
            AssertView([2, 3], [1L, 4L], 1, [1f, 5f, 9f, 2f, 6f, 10f]);
            AssertView([4], [-1L], 7, [7f, 6f, 5f, 4f]);
            AssertView([2, 3], [0L, 1L], 4, [4f, 5f, 6f, 4f, 5f, 6f]);
        }
        finally { MlxNative.FreeArray(source); }

        void AssertView(int[] shape, long[] strides, long offset, float[] expected)
        {
            var view = MlxNative.AsStrided(source, shape, strides, offset, largeStorage: true);
            MlxNative.MlxArray contiguous = default;
            try
            {
                contiguous = MlxNative.Contiguous(view);
                float[] actual = new float[expected.Length];
                fixed (float* destination = actual)
                    MlxNative.CopyArrayToHost(contiguous, DType.Float32, (IntPtr)destination, actual.LongLength * sizeof(float));
                Assert.Equal(expected, actual);
            }
            finally
            {
                MlxNative.FreeArray(view);
                MlxNative.FreeArray(contiguous);
            }
        }
    }

    private static void AssertNativeShape(Tensor tensor, int[] expectedShape, long expectedCount)
    {
        var view = ((MlxStorage)tensor.Storage).CreateArrayView(tensor);
        try
        {
            Assert.Equal(expectedCount, MlxNative.GetArraySize(view));
            Assert.Equal(expectedShape, MlxNative.GetArrayShape(view));
        }
        finally { MlxNative.FreeArray(view); }
    }

    private static void AssertBoxes(int[] shape, long start, long length)
    {
        var boxes = MlxStorage.GetSliceBoxes(shape, start, length).ToArray();
        Assert.True(boxes.Length <= 2 * shape.Length + 1);
        long consumed = 0;
        foreach (var box in boxes)
        {
            Assert.Equal(consumed, box.SourceOffset);
            long first = 0, last = 0, count = 1, contiguousStride = 1;
            for (int axis = shape.Length - 1; axis >= 0; --axis)
            {
                Assert.InRange(box.Starts[axis], 0, shape[axis] - 1);
                Assert.InRange(box.Stops[axis], box.Starts[axis] + 1, shape[axis]);
                Assert.Equal(box.Stops[axis] - box.Starts[axis], box.Shape[axis]);
                Assert.Equal(contiguousStride, box.Strides[axis]);
                contiguousStride *= box.Shape[axis];
                count *= box.Shape[axis];
            }
            for (int axis = 0; axis < shape.Length; ++axis)
            {
                first = first * shape[axis] + box.Starts[axis];
                last = last * shape[axis] + box.Stops[axis] - 1;
            }
            Assert.Equal(start + consumed, first);
            Assert.Equal(first + count - 1, last);
            consumed += count;
        }
        Assert.Equal(length, consumed);
    }
}
