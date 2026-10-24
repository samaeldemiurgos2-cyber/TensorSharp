// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

// Instantiate the host allocator directly: these tests require neither a
// native GGML library nor a CUDA device. The CUDA policy is host memory only.
public sealed class MuseSparkHostPoolTests
{
    private static long HostPageSize => Math.Max(Environment.SystemPageSize,
        OperatingSystem.IsMacOS() || OperatingSystem.IsIOS() || OperatingSystem.IsMacCatalyst() ||
        OperatingSystem.IsTvOS() ? 16 * 1024 : 32);
    private static long SmallBlockExtent => (4096 + HostPageSize - 1) / HostPageSize * HostPageSize;

    [Theory]
    [InlineData(GgmlBackendType.Cuda, 72)]
    [InlineData(GgmlBackendType.Vulkan, 8)]
    [InlineData(GgmlBackendType.Cpu, 32)]
    [InlineData(GgmlBackendType.Metal, 32)]
    public void OnlyExplicitCudaPoolOptInChangesRetention(GgmlBackendType backend, int expected)
    {
        using var pool = new Pool(backend);
        var original = pool.Stats;
        Assert.False(pool.Configure(1));
        Assert.Equal(original, pool.Stats);
        Assert.Equal(backend == GgmlBackendType.Cuda, pool.Configure(9));
        Assert.Equal(expected, pool.Stats.MaxPooledBlocks);
        Assert.Equal(0, pool.Stats.FreshAllocations); // Configuration never preallocates.
        Assert.Equal(0, pool.Stats.RetainedBytes);
    }

    [Fact]
    public void NineRankBurstsReuseBoundedBlocksWhileBaselineReturnsOverflow()
    {
        foreach (bool scaled in new[] { false, true })
        {
            using var pool = new Pool(GgmlBackendType.Cuda);
            if (scaled) Assert.True(pool.Configure(9));
            for (int round = 0; round < 3; round++)
            {
                var pointers = Enumerable.Range(0, 72).Select(_ => pool.Allocate(4096)).ToArray();
                Assert.Equal(72, pointers.Distinct().Count());
                // Nine workers own disjoint live blocks. Values change on every
                // round so recycling cannot conceal an alias or stale write.
                Parallel.For(0, 9, rank =>
                {
                    for (int block = 0; block < 8; block++)
                    {
                        int index = rank * 8 + block;
                        Marshal.WriteInt32(pointers[index], round * 1000 + index);
                        Marshal.WriteInt32(pointers[index], 4092, -round * 1000 - index);
                    }
                });
                for (int index = 0; index < 72; index++)
                {
                    Assert.Equal(round * 1000 + index, Marshal.ReadInt32(pointers[index]));
                    Assert.Equal(-round * 1000 - index, Marshal.ReadInt32(pointers[index], 4092));
                }
                Parallel.ForEach(pointers, pointer => pool.Free(pointer, 4096));
                var stats = pool.Stats;
                int retained = scaled ? 72 : 8;
                Assert.Equal(retained, stats.RetainedBlocks);
                Assert.Equal(retained * SmallBlockExtent, stats.RetainedBytes);
                Assert.Equal(72 + (scaled ? 0 : round * 64), stats.FreshAllocations);
                Assert.Equal(scaled ? 0 : (round + 1) * 64, stats.SystemFrees);
                Assert.Equal(round * retained, stats.PoolHits);
                Assert.Equal(0, stats.OutstandingBlocks);
                Assert.Equal(0, stats.SystemFreeFailures);
            }
            long expectedBytes = (scaled ? 72 : 8) * SmallBlockExtent;
            var snapshot = pool.Stats;
            Assert.Equal(expectedBytes, pool.Close());
            Assert.Equal(expectedBytes, snapshot.RetainedBytes); // Immutable snapshot.
            Assert.Equal(0, pool.Stats.RetainedBytes);
            Assert.Equal(pool.Stats.FreshAllocations, pool.Stats.SystemFrees);
        }
    }

    [Fact]
    public void CapAndPerBlockLimitRemainBounded()
    {
        using var pool = new Pool(GgmlBackendType.Cuda);
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.Configure(0));
        Assert.True(pool.Configure(int.MaxValue));
        Assert.Equal(128, pool.Stats.MaxPooledBlocks);
        Assert.Equal(8L << 20, pool.Stats.MaxRetainedBlockBytes);
        Assert.Equal(1L << 30, pool.Stats.MaxPooledBlocks * pool.Stats.MaxRetainedBlockBytes);
        var pointers = Enumerable.Range(0, 140).Select(_ => pool.Allocate(4096)).ToArray();
        Parallel.ForEach(pointers, pointer => pool.Free(pointer, 4096));
        Assert.Equal(128, pool.Stats.RetainedBlocks);
        Assert.Equal(12, pool.Stats.SystemFrees);
        IntPtr large = pool.Allocate((8L << 20) + 4096);
        pool.Free(large, (8L << 20) + 4096);
        Assert.Equal(128, pool.Stats.RetainedBlocks);
        Assert.Equal(13, pool.Stats.SystemFrees); // Oversized blocks remain unpooled.
    }

    [Fact]
    public void BestFitReturnsFullOriginalExtentAndCloseRejectsLateRetention()
    {
        using var pool = new Pool(GgmlBackendType.Cuda);
        pool.Configure(9);
        IntPtr large = pool.Allocate(8L << 20);
        Marshal.WriteInt32(large, (8 << 20) - 4, 12345);
        pool.Free(large, 8L << 20);
        IntPtr small = pool.Allocate(4096);
        Assert.Equal(large, small);
        Assert.Equal(12345, Marshal.ReadInt32(small, (8 << 20) - 4));
        pool.Free(small, 4096);
        Assert.Equal(8L << 20, pool.Stats.RetainedBytes);
        Assert.Equal(8L << 20, pool.Trim()); // Caller length must not truncate munmap.
        Assert.Equal(1, pool.Stats.SystemFrees);
        IntPtr late = pool.Allocate(4096);
        Assert.Equal(0, pool.Close());
        Assert.False(pool.Configure(9));
        pool.Free(late, 4096);
        Assert.True(pool.Stats.IsClosed);
        Assert.Equal(0, pool.Stats.RetainedBytes);
        Assert.Equal(0, pool.Trim());
        Assert.Equal(2, pool.Stats.SystemFrees);
    }

    [Fact]
    public async Task ConcurrentReturnsAfterCloseLeaveNoRetainedOrOutstandingBlocks()
    {
        using var pool = new Pool(GgmlBackendType.Cuda);
        pool.Configure(9);
        var pointers = Enumerable.Range(0, 72).Select(_ => pool.Allocate(4096)).ToArray();
        using var started = new CountdownEvent(9);
        using var release = new ManualResetEventSlim();
        var workers = Enumerable.Range(0, 9).Select(rank => Task.Factory.StartNew(() =>
        {
            started.Signal();
            Assert.True(release.Wait(TimeSpan.FromSeconds(10)));
            for (int index = rank * 8; index < (rank + 1) * 8; index++) pool.Free(pointers[index], 4096);
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();
        try
        {
            Assert.True(started.Wait(TimeSpan.FromSeconds(10)));
            Assert.Equal(0, pool.Close());
        }
        finally { release.Set(); await Task.WhenAll(workers); }
        Assert.Equal(0, pool.Stats.OutstandingBlocks);
        Assert.Equal(0, pool.Stats.RetainedBlocks);
        Assert.Equal(72, pool.Stats.SystemFrees);
        Assert.Equal(0, pool.Stats.SystemFreeFailures);
    }

    private sealed class Pool : IDisposable
    {
        private readonly Type _type = typeof(GgmlContext).Assembly.GetType("TensorSharp.GGML.GgmlMemoryPool", true)!;
        private readonly object _pool;
        private readonly ConcurrentDictionary<IntPtr, long> _owned = new();
        internal Pool(GgmlBackendType backend) => _pool = Activator.CreateInstance(_type, backend)!;
        private object Call(string name, params object[] arguments)
        {
            try { return _type.GetMethod(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(_pool, arguments)!; }
            catch (TargetInvocationException exception) when (exception.InnerException != null)
            { ExceptionDispatchInfo.Capture(exception.InnerException).Throw(); throw; }
        }
        internal GgmlMemoryPoolStatistics Stats => (GgmlMemoryPoolStatistics)Call("GetStatistics");
        internal bool Configure(int degree) => (bool)Call("ConfigureCudaTensorParallel", degree);
        internal IntPtr Allocate(long bytes)
        {
            IntPtr pointer = (IntPtr)Call("Allocate", bytes);
            Assert.True(_owned.TryAdd(pointer, bytes));
            return pointer;
        }
        internal void Free(IntPtr pointer, long callerBytes)
        {
            Assert.True(_owned.TryRemove(pointer, out _));
            Call("Free", pointer, callerBytes);
        }
        internal long Close() => (long)Call("Close");
        internal long Trim() => (long)Call("Trim");
        public void Dispose()
        {
            Close();
            foreach (var (pointer, bytes) in _owned) Free(pointer, bytes);
        }
    }
}
