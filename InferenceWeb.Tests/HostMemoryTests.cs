// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp;

namespace InferenceWeb.Tests;

public sealed unsafe class HostMemoryTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(17)]
    [InlineData((1 << 20) - 3)]          // just below the parallel threshold
    [InlineData((3 << 20) + 12345)]      // several chunks with a ragged tail
    public void CopyMatchesTheSourceExactly(int bytes)
    {
        byte* src = (byte*)NativeMemory.Alloc((nuint)Math.Max(bytes, 1));
        byte* dst = (byte*)NativeMemory.Alloc((nuint)bytes + 2);
        try
        {
            for (int i = 0; i < bytes; i++) src[i] = (byte)(i * 31 + 7);
            dst[bytes] = 0xAB;
            dst[bytes + 1] = 0xCD;
            HostMemory.Copy(src, dst, bytes);
            Assert.True(new ReadOnlySpan<byte>(src, bytes).SequenceEqual(new ReadOnlySpan<byte>(dst, bytes)));
            Assert.Equal(0xAB, dst[bytes]);       // nothing past the range is touched
            Assert.Equal(0xCD, dst[bytes + 1]);
        }
        finally
        {
            NativeMemory.Free(src);
            NativeMemory.Free(dst);
        }
    }

    [Theory]
    [InlineData(5)]
    [InlineData((2 << 20) + 777)]
    public void ClearZeroesOnlyTheRange(int bytes)
    {
        byte* buffer = (byte*)NativeMemory.Alloc((nuint)bytes + 2);
        try
        {
            new Span<byte>(buffer, bytes + 2).Fill(0x5A);
            HostMemory.Clear(buffer + 1, bytes);
            Assert.Equal(0x5A, buffer[0]);
            Assert.True(new ReadOnlySpan<byte>(buffer + 1, bytes).IndexOfAnyExcept((byte)0) < 0);
            Assert.Equal(0x5A, buffer[bytes + 1]);
        }
        finally
        {
            NativeMemory.Free(buffer);
        }
    }

    [Fact]
    public void ClearRangesZeroesEveryRangeAndLeavesTheGaps()
    {
        const int plane = (1 << 20) + 4096, gap = 64;
        byte* buffer = (byte*)NativeMemory.Alloc(3 * plane + 2 * gap);
        try
        {
            int total = 3 * plane + 2 * gap;
            new Span<byte>(buffer, total).Fill(0x33);
            var ranges = new (IntPtr, long)[3];
            for (int i = 0; i < 3; i++)
                ranges[i] = ((IntPtr)(buffer + i * (plane + gap)), plane);
            HostMemory.Clear(ranges);
            for (int i = 0; i < 3; i++)
            {
                Assert.True(new ReadOnlySpan<byte>(buffer + i * (plane + gap), plane).IndexOfAnyExcept((byte)0) < 0);
                if (i < 2)
                    Assert.True(new ReadOnlySpan<byte>(buffer + i * (plane + gap) + plane, gap).IndexOfAnyExcept((byte)0x33) < 0);
            }
        }
        finally
        {
            NativeMemory.Free(buffer);
        }
    }
}
