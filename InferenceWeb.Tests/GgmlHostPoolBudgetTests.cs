// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

// Byte-budgeted host block retention for CUDA per-request caches. Host memory
// only: no native GGML library or device is required.
public sealed class GgmlHostPoolBudgetTests
{
    private const long MiB = 1024 * 1024;

    [Fact]
    public void LargeFreedBlocksAreReusedWithTheirContents()
    {
        using var pool = new MuseSparkHostPoolTests.Pool(GgmlBackendType.Cuda, retainBudgetBytes: 256 * MiB);
        IntPtr first = pool.Allocate(48 * MiB);
        Marshal.WriteInt64(first, (int)(48 * MiB - 8), 0x1234_5678_9abc);
        pool.Free(first, 48 * MiB);
        Assert.Equal(48 * MiB, pool.Stats.RetainedBytes);

        IntPtr second = pool.Allocate(48 * MiB);
        Assert.Equal(first, second);
        Assert.Equal(0x1234_5678_9abc, Marshal.ReadInt64(second, (int)(48 * MiB - 8)));
        Assert.Equal(1, pool.Stats.PoolHits);
        Assert.Equal(0, pool.Stats.RetainedBytes);
        pool.Free(second, 48 * MiB);
    }

    [Fact]
    public void OnlyFreshMappingsReportZeroContents()
    {
        // Holders skip their allocation-time zero fill on a fresh mapping (it is zero already);
        // a reused block keeps the previous owner's bytes and must still be filled.
        using var pool = new MuseSparkHostPoolTests.Pool(GgmlBackendType.Cuda, retainBudgetBytes: 256 * MiB);
        IntPtr large = pool.AllocateReportingZero(48 * MiB, out bool largeZero);
        IntPtr small = pool.AllocateReportingZero(1 * MiB, out bool smallZero);
        Assert.True(largeZero);
        Assert.True(smallZero);
        Assert.Equal(0, Marshal.ReadInt64(large, (int)(48 * MiB - 8)));
        Marshal.WriteInt64(large, 0, 42);
        Marshal.WriteInt64(small, 0, 42);
        pool.Free(large, 48 * MiB);   // byte-budgeted retention
        pool.Free(small, 1 * MiB);    // count-bounded pool

        IntPtr largeAgain = pool.AllocateReportingZero(48 * MiB, out bool largeAgainZero);
        IntPtr smallAgain = pool.AllocateReportingZero(1 * MiB, out bool smallAgainZero);
        Assert.Equal(large, largeAgain);
        Assert.Equal(small, smallAgain);
        Assert.False(largeAgainZero);
        Assert.False(smallAgainZero);
        pool.Free(largeAgain, 48 * MiB);
        pool.Free(smallAgain, 1 * MiB);
    }

    [Fact]
    public void OverflowOfTheCountBoundedPoolIsRetainedByBytes()
    {
        // Recurrent-state tensors are a few MiB each and a holder has dozens: the
        // eight-block small pool alone would unmap most of them on every release.
        using var pool = new MuseSparkHostPoolTests.Pool(GgmlBackendType.Cuda, retainBudgetBytes: 512 * MiB);
        var pointers = Enumerable.Range(0, 48).Select(_ => pool.Allocate(3 * MiB)).ToArray();
        foreach (IntPtr p in pointers) pool.Free(p, 3 * MiB);
        Assert.Equal(48, pool.Stats.RetainedBlocks);
        Assert.Equal(0, pool.Stats.SystemFrees);

        var again = Enumerable.Range(0, 48).Select(_ => pool.Allocate(3 * MiB)).ToArray();
        Assert.Equal(pointers.OrderBy(p => p), again.OrderBy(p => p));
        Assert.Equal(48, pool.Stats.PoolHits);
        foreach (IntPtr p in again) pool.Free(p, 3 * MiB);
    }

    [Fact]
    public void BudgetEvictsOldestFirst()
    {
        using var pool = new MuseSparkHostPoolTests.Pool(GgmlBackendType.Cuda, retainBudgetBytes: 100 * MiB);
        IntPtr a = pool.Allocate(40 * MiB);
        IntPtr b = pool.Allocate(40 * MiB);
        IntPtr c = pool.Allocate(40 * MiB);
        pool.Free(a, 40 * MiB);
        pool.Free(b, 40 * MiB);
        pool.Free(c, 40 * MiB);
        Assert.Equal(80 * MiB, pool.Stats.RetainedBytes);
        Assert.Equal(1, pool.Stats.SystemFrees);

        // a was the oldest and went back to the system; b and c remain reusable.
        IntPtr x = pool.Allocate(40 * MiB);
        IntPtr y = pool.Allocate(40 * MiB);
        Assert.Equal(new[] { b, c }.OrderBy(p => p), new[] { x, y }.OrderBy(p => p));
        pool.Free(x, 40 * MiB);
        pool.Free(y, 40 * MiB);
    }

    [Fact]
    public void SmallRequestsDoNotTieUpMuchLargerBlocks()
    {
        using var pool = new MuseSparkHostPoolTests.Pool(GgmlBackendType.Cuda, retainBudgetBytes: 256 * MiB);
        IntPtr large = pool.Allocate(64 * MiB);
        pool.Free(large, 64 * MiB);
        IntPtr small = pool.Allocate(16 * MiB);
        Assert.NotEqual(large, small);
        Assert.Equal(64 * MiB, pool.Stats.RetainedBytes);
        pool.Free(small, 16 * MiB);
        Assert.Equal(80 * MiB, pool.Stats.RetainedBytes);
        Assert.Equal(80 * MiB, pool.Trim());
        Assert.Equal(0, pool.Stats.RetainedBlocks);
    }

    [Fact]
    public void ClosedPoolAndZeroBudgetRetainNothingLarge()
    {
        using (var off = new MuseSparkHostPoolTests.Pool(GgmlBackendType.Cuda, retainBudgetBytes: 0))
        {
            IntPtr p = off.Allocate(32 * MiB);
            off.Free(p, 32 * MiB);
            Assert.Equal(0, off.Stats.RetainedBytes);
            Assert.Equal(1, off.Stats.SystemFrees);
        }
        using var pool = new MuseSparkHostPoolTests.Pool(GgmlBackendType.Cuda, retainBudgetBytes: 256 * MiB);
        IntPtr q = pool.Allocate(32 * MiB);
        Assert.Equal(0, pool.Close());
        pool.Free(q, 32 * MiB);
        Assert.Equal(0, pool.Stats.RetainedBytes);
    }

    [Theory]
    [InlineData("0", 64L << 30, 0L)]
    [InlineData("512", 64L << 30, 512L << 20)]
    [InlineData(null, 32L << 30, 2L << 30)]
    [InlineData(null, 512L << 30, 8L << 30)]
    [InlineData("bogus", 16L << 30, 1L << 30)]
    public void BudgetResolvesFromSettingOrAvailableMemory(string setting, long available, long expected)
    {
        var method = typeof(GgmlContext).Assembly.GetType("TensorSharp.GGML.GgmlMemoryPool", true)!
            .GetMethod("ResolveRetainBudgetBytes", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic)!;
        Assert.Equal(expected, (long)method.Invoke(null, new object[] { setting!, available })!);
    }
}
