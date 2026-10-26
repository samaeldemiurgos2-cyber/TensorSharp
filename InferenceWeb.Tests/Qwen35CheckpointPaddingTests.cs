// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public class Qwen35CheckpointPaddingTests
{
    [Theory]
    [InlineData(DType.Float32)]
    [InlineData(DType.Float16)]
    [InlineData(DType.Q8_0)]
    [InlineData(DType.Q4_0)]
    public unsafe void CopyPadding_DoesNotOverwriteLiveRows_AndClearsEveryUnwrittenHeadRow(DType dtype)
    {
        // 0xFF produces non-finite F32/F16 values and invalid quant scales.
        // Recycled allocation bytes must never reach padded attention reads.
        using var tensor = new Tensor(new CpuAllocator(BlasEnum.DotNet), dtype, 3, 17, 64);
        byte* bytes = (byte*)tensor.Storage.PtrAtElement(0);
        NativeMemory.Fill(bytes, checked((nuint)tensor.Storage.ByteLength), 0xff);
        long rowBytes = tensor.Storage.ByteLength / (3 * 17);
        Qwen35Model.ClearCheckpointPadding(tensor, 13);
        for (int head = 0; head < 3; head++)
            for (int row = 0; row < 17; row++)
                for (int b = 0; b < rowBytes; b++)
                    Assert.Equal(row < 13 ? (byte)0xff : (byte)0,
                        bytes[(head * 17 + row) * rowBytes + b]);
    }

    [Fact]
    public unsafe void EmptyAndFullPrefixes_KeepThePaddingBoundaryExact()
    {
        using var tensor = new Tensor(new CpuAllocator(BlasEnum.DotNet), DType.Float16, 2, 3, 32);
        byte* bytes = (byte*)tensor.Storage.PtrAtElement(0);
        NativeMemory.Fill(bytes, checked((nuint)tensor.Storage.ByteLength), 0x45);
        Qwen35Model.ClearCheckpointPadding(tensor, 3);
        for (int i = 0; i < tensor.Storage.ByteLength; i++) Assert.Equal((byte)0x45, bytes[i]);
        Qwen35Model.ClearCheckpointPadding(tensor, 0);
        for (int i = 0; i < tensor.Storage.ByteLength; i++) Assert.Equal((byte)0, bytes[i]);
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35Model.ClearCheckpointPadding(tensor, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35Model.ClearCheckpointPadding(tensor, 4));
    }
}
