using System.Reflection;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GgmlHostHugePageTests
{
    private const long Threshold = 32L << 20;
    private static readonly Type PoolType = typeof(GgmlContext).Assembly.GetType("TensorSharp.GGML.GgmlMemoryPool", true)!;

    [Theory]
    [InlineData(true, GgmlBackendType.Cuda, "1", Threshold, true)]
    [InlineData(true, GgmlBackendType.Cuda, "1", Threshold + 4096, true)]
    [InlineData(true, GgmlBackendType.Cuda, "1", Threshold - 4096, false)]
    [InlineData(true, GgmlBackendType.Cuda, "1", 0L, false)]
    [InlineData(true, GgmlBackendType.Cuda, null, Threshold, false)]
    [InlineData(true, GgmlBackendType.Cuda, "0", Threshold, false)]
    [InlineData(true, GgmlBackendType.Cuda, "true", Threshold, false)]
    [InlineData(true, GgmlBackendType.Cuda, " 1", Threshold, false)]
    [InlineData(false, GgmlBackendType.Cuda, "1", Threshold, false)]
    [InlineData(true, GgmlBackendType.Cpu, "1", Threshold, false)]
    [InlineData(true, GgmlBackendType.Vulkan, "1", Threshold, false)]
    [InlineData(true, GgmlBackendType.Metal, "1", Threshold, false)]
    public void AdviceRequiresExplicitLinuxCudaOptInAndLargeMapping(bool linux, GgmlBackendType backend,
        string? setting, long mappedBytes, bool expected)
    {
        var policy = PoolType.GetMethod("ShouldAdviseHostHugePages", BindingFlags.Static | BindingFlags.NonPublic)!;
        Assert.Equal(expected, (bool)policy.Invoke(null, [linux, backend, setting, (nuint)mappedBytes])!);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("0")]
    public void DisabledLargeMappingsKeepOrdinaryOwnershipAndZeroAdviceCounters(string? setting)
    {
        using var pool = new Pool(setting);
        IntPtr pointer = pool.Allocate(Threshold);
        System.Runtime.InteropServices.Marshal.WriteInt32(pointer, 123);
        System.Runtime.InteropServices.Marshal.WriteInt32(pointer, (int)Threshold - 4, 456);
        Assert.False(pool.Stats.HugePageAdviceEnabled);
        Assert.Equal(0, pool.Stats.HugePageAdviceRequests);
        Assert.Equal(123, System.Runtime.InteropServices.Marshal.ReadInt32(pointer));
        Assert.Equal(456, System.Runtime.InteropServices.Marshal.ReadInt32(pointer, (int)Threshold - 4));
        pool.Free(pointer, Threshold);
        Assert.Equal(1, pool.Stats.SystemFrees);
        Assert.Equal(0, pool.Stats.SystemFreeFailures);
        Assert.Equal(0, pool.Stats.OutstandingBlocks);
    }

    [Fact]
    public void SmallMappingNeverRequestsAdviceEvenWithOptIn()
    {
        using var pool = new Pool("1");
        IntPtr pointer = pool.Allocate(Threshold - Environment.SystemPageSize);
        Assert.Equal(0, pool.Stats.HugePageAdviceRequests);
        pool.Free(pointer, Threshold - Environment.SystemPageSize);
        Assert.Equal(0, pool.Stats.SystemFreeFailures);
    }

    [LinuxHostMappingFact]
    public unsafe void LinuxAcceptedOrRejectedHintPreservesCopiedBytesAndReleaseExtent()
    {
        using var pool = new Pool("1");
        IntPtr pointer = pool.Allocate(Threshold);
        var bytes = new Span<byte>((void*)pointer, (int)Threshold);
        bytes.Fill(0x5a);
        Assert.Equal(-1, bytes.IndexOfAnyExcept((byte)0x5a));
        GgmlMemoryPoolStatistics initial = pool.Stats;
        Assert.True(initial.HugePageAdviceEnabled);
        Assert.Equal(1, initial.HugePageAdviceRequests);
        Assert.Equal(1, initial.HugePageAdviceAccepted + initial.HugePageAdviceFailures);
        Assert.Equal(initial.HugePageAdviceAccepted * Threshold, initial.HugePageAdvisedBytes);

        // An unaligned madvise must fail with EINVAL. Exercise that fallthrough
        // without invalidating the real owned mapping or changing any OS setting.
        PoolType.GetMethod("AdviseHostHugePages", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(pool.Instance, [IntPtr.Add(pointer, 1), (nuint)(Threshold - 1)]);
        Assert.Equal(initial.HugePageAdviceFailures + 1, pool.Stats.HugePageAdviceFailures);
        Assert.Equal(initial.HugePageAdviceAccepted, pool.Stats.HugePageAdviceAccepted);
        Assert.Equal(-1, bytes.IndexOfAnyExcept((byte)0x5a));
        pool.Free(pointer, Threshold);
        Assert.Equal(1, pool.Stats.SystemFrees);
        Assert.Equal(0, pool.Stats.SystemFreeFailures);
        Assert.Equal(0, pool.Stats.OutstandingBlocks);
    }

    private sealed class Pool : IDisposable
    {
        private readonly Dictionary<IntPtr, long> _owned = new();
        internal object Instance { get; }
        // Zero retain budget: these tests observe each mapping's release.
        internal Pool(string? setting) => Instance = PoolType.GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic, binder: null,
            [typeof(GgmlBackendType), typeof(string), typeof(long)], modifiers: null)!.Invoke([GgmlBackendType.Cuda, setting, 0L]);
        private object Call(string name, params object[] arguments) => PoolType.GetMethod(name,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(Instance, arguments)!;
        internal GgmlMemoryPoolStatistics Stats => (GgmlMemoryPoolStatistics)Call("GetStatistics");
        internal IntPtr Allocate(long bytes)
        {
            var ptr = (IntPtr)Call("Allocate", bytes);
            _owned.Add(ptr, bytes);
            return ptr;
        }
        internal void Free(IntPtr ptr, long bytes)
        {
            Call("Free", ptr, bytes);
            _owned.Remove(ptr);
        }
        public void Dispose()
        {
            Call("Close");
            foreach (var entry in _owned) Call("Free", entry.Key, entry.Value);
        }
    }
}

public sealed class LinuxHostMappingFactAttribute : FactAttribute
{
    public LinuxHostMappingFactAttribute()
    {
        if (!OperatingSystem.IsLinux())
            Skip = "Linux anonymous mapping/madvise validation requires Linux; no GPU is required.";
    }
}
