// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkSourcePageTests
{
    [Theory]
    [InlineData(false, BackendType.GgmlCuda, false, false)]
    [InlineData(true, BackendType.GgmlCuda, false, true)]
    [InlineData(true, BackendType.GgmlCuda, true, false)]
    [InlineData(true, BackendType.GgmlCpu, false, false)]
    [InlineData(true, BackendType.GgmlMetal, false, false)]
    [InlineData(true, BackendType.GgmlVulkan, false, false)]
    [InlineData(true, BackendType.Cpu, false, false)]
    public void AdviceIsOptInAndExcludesCpuOffloadAndSharedMemoryBackends(
        bool enabled, BackendType backend, bool offloaded, bool expected)
        => Assert.Equal(expected, MuseSparkModel.CanReleaseGpuSourcePages(enabled, backend, offloaded));

    [Theory]
    [InlineData(32, 12288, 4096, 4096, 8192)]
    [InlineData(4096, 8192, 4096, 4096, 8192)]
    [InlineData(32, 32768, 16384, 16384, 16384)]
    public void PageAdviceNeverIncludesNeighboringPartialPages(long offset, long bytes, int page,
        long expectedStart, long expectedLength)
    {
        Assert.True(GgufFile.TryGetCompletePageRange(offset, bytes, page, out long start, out long length));
        Assert.Equal(expectedStart, start);
        Assert.Equal(expectedLength, length);
        Assert.InRange(start, offset, offset + bytes);
        Assert.InRange(start + length, start, offset + bytes);
    }

    [Theory]
    [InlineData(1, 4095)]
    [InlineData(4095, 2)]
    [InlineData(-1, 8192)]
    [InlineData(long.MaxValue - 1, 8192)]
    public void EmptyOrInvalidPageRangesAreRefused(long offset, long bytes)
        => Assert.False(GgufFile.TryGetCompletePageRange(offset, bytes, 4096, out _, out _));

    [Fact]
    public void AdviceRetainsSplitShardPointersAndRejectsCopiedBuffersOrWrongRanges()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-source-pages-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string first = Write(directory, 1, "cpu");
            string second = Write(directory, 2, "gpu");
            using var file = new GgufFile(first);
            using (var shard = GgufFile.OpenWithoutSiblingShards(second))
            {
                Assert.NotEqual(file.DataOffset, shard.DataOffset);
                Assert.Equal(32ul, shard.Tensors["gpu"].Offset);
                Assert.Equal(3, shard.Tensors["gpu"].Shape.Length);
            }
            var info = file.Tensors["gpu"];
            Assert.True(file.TryGetTensorDataPointer(info, out IntPtr pointer));
            int bytes = checked((int)file.GetTensorByteCount(info));
            byte[] before = new byte[bytes];
            Marshal.Copy(pointer, before, 0, bytes);
            IntPtr copied = Marshal.AllocHGlobal(bytes);
            try
            {
                Marshal.Copy(before, 0, copied, bytes);
                using var firstCopy = new QuantizedWeight(before[..(bytes / 2)], 0, 128, 64);
                using var secondCopy = new QuantizedWeight(before[(bytes / 2)..], 0, 128, 64);
                using var fusedCopy = QuantizedWeight.ConcatOrCreateCopy(firstCopy, secondCopy);
                Assert.False(fusedCopy.IsFileBacked);
                Assert.False(file.TryGetMappedTensorSourcePages(info, fusedCopy.Data, fusedCopy.RawBytes, out _, out _, out _));
                Assert.False(file.TryReleaseMappedTensorPages(info, fusedCopy.Data, fusedCopy.RawBytes));
                Assert.False(file.TryGetMappedTensorSourcePages(info, copied, bytes, out _, out _, out _));
                Assert.False(file.TryGetMappedTensorSourcePages(info, pointer + 1, bytes, out _, out _, out _));
                Assert.False(file.TryGetMappedTensorSourcePages(info, pointer, bytes - 1, out _, out _, out _));
                Assert.False(file.TryGetMappedTensorSourcePages(file.Tensors["cpu"], pointer, bytes, out _, out _, out _));
                Assert.True(file.TryGetMappedTensorSourcePages(info, pointer, bytes, out var owner, out long start, out long length));
                Assert.Equal(second, owner.FilePaths[0]);
                Assert.NotSame(file, owner);
                Assert.True(GgufFile.TryGetCompletePageRange(owner.DataOffset + 32, bytes,
                    Environment.SystemPageSize, out long expectedStart, out long expectedLength));
                Assert.Equal(expectedStart, start); Assert.Equal(expectedLength, length);
                Assert.False(file.TryReleaseMappedTensorPages(info, copied, bytes));
                Assert.False(file.TryReleaseMappedTensorPages(info, pointer + 1, bytes));
                Assert.False(file.TryReleaseMappedTensorPages(info, pointer, bytes - 1));
                Assert.False(file.TryReleaseMappedTensorPages(file.Tensors["cpu"], pointer, bytes));
                bool advised = file.TryReleaseMappedTensorPages(info, pointer, bytes);
                if (OperatingSystem.IsLinux()) Assert.True(advised);
                else Assert.False(advised);
                Assert.True(file.TryGetTensorDataPointer(info, out IntPtr after));
                Assert.Equal(pointer, after);
                var reread = new byte[bytes];
                Marshal.Copy(after, reread, 0, bytes);
                Assert.Equal(before, reread);
                Assert.Equal(before, file.ReadTensorData(info));
                Assert.All(file.ReadTensorData(file.Tensors["cpu"]), value => Assert.Equal((byte)17, value));
                Marshal.Copy(copied, reread, 0, bytes);
                Assert.Equal(before, reread);
            }
            finally { Marshal.FreeHGlobal(copied); }
        }
        finally { Directory.Delete(directory, true); }
    }

    [MuseSparkSourcePagesCudaFact]
    public void NativeCudaAdvicePreservesCacheKeyAndReloadsFromTheRetainedShardMapping()
        => NativeAdviceReloadCheck(1);

    [MuseSparkSourcePagesCudaFact(2)]
    public void NativeCudaResidentProofIsSpecificToTheUploadingRank()
        => NativeAdviceReloadCheck(2);

    private static void NativeAdviceReloadCheck(int devices)
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-source-reload-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var context = new GgmlContext(Enumerable.Range(0, devices).ToArray(), GgmlBackendType.Cuda, enableCollectives: false);
        var allocator = new GgmlAllocator(context, 0);
        try
        {
            string first = Write(directory, 1, "cpu");
            Write(directory, 2, "gpu", unitWeights: true);
            using var file = new GgufFile(first);
            var info = file.Tensors["gpu"];
            Assert.True(file.TryGetTensorDataPointer(info, out IntPtr pointer));
            long bytes = file.GetTensorByteCount(info);
            using var weight = QuantizedWeight.CreateExternalView(pointer, bytes, 0, 128, 128, file);
            IntPtr key = weight.EnsureDeviceCacheKey();
            try
            {
                Assert.True(GgmlBasicOps.PreloadQuantizedWeight(key, pointer, 0, 128, 128, bytes));
                Assert.True(GgmlBasicOps.HasResidentDeviceCopy(key, bytes));
                Assert.False(GgmlBasicOps.HasResidentDeviceCopy(key, bytes - 1));
                if (devices > 1)
                {
                    try
                    {
                        GgmlBasicOps.SetActiveRank(1);
                        Assert.False(GgmlBasicOps.HasResidentDeviceCopy(key, bytes));
                    }
                    finally { GgmlBasicOps.SetActiveRank(0); }
                }
                Assert.True(file.TryReleaseMappedTensorPages(info, pointer, bytes, out long advised));
                Assert.InRange(advised, bytes - 2 * Environment.SystemPageSize, bytes);
                Assert.Equal(pointer, weight.Data); Assert.Equal(key, weight.CacheKey);
                Assert.True(weight.HasHostData); Assert.True(weight.IsFileBacked);
                using var input = new Tensor(allocator, DType.Float32, 1, 128);
                using var output = new Tensor(allocator, DType.Float32, 1, 128);
                input.SetElementsAsFloat(Enumerable.Repeat(1f, 128).ToArray());
                GgmlBasicOps.AddmmQuant(output, input, key, 0, 128, 128, bytes);
                Assert.All(output.GetElementsAsFloat(128), value => Assert.Equal(128f, value));
                GgmlBasicOps.InvalidateHostBuffer(key);
                Assert.False(GgmlBasicOps.HasResidentDeviceCopy(key, bytes));
                // This miss resolves the unchanged opaque key back to the GGUF
                // host pointer and faults in the original source bytes again.
                GgmlBasicOps.AddmmQuant(output, input, key, 0, 128, 128, bytes);
                Assert.All(output.GetElementsAsFloat(128), value => Assert.Equal(128f, value));
                Assert.True(GgmlBasicOps.HasResidentDeviceCopy(key, bytes));
                Assert.Equal(pointer, weight.Data); Assert.Equal(key, weight.CacheKey);
                var host = new float[128 * 128]; Marshal.Copy(pointer, host, 0, host.Length);
                Assert.All(host, value => Assert.Equal(1f, value));
            }
            finally { GgmlBasicOps.ClearHostBufferCache(); }
        }
        finally { context.ReleasePoolForDisposal(); Directory.Delete(directory, true); }
    }

    private static string Write(string directory, int shard, string name, bool unitWeights = false)
    {
        string path = Path.Combine(directory, $"source-{shard:D5}-of-00002.gguf");
        using var writer = new BinaryWriter(File.Create(path), Encoding.UTF8);
        writer.Write(0x46554747u); writer.Write(3u); writer.Write(1ul); writer.Write(1ul);
        Text(writer, "split.count"); writer.Write((uint)GgufValueType.Uint32); writer.Write(2u);
        Text(writer, name);
        if (shard == 2) { writer.Write(3u); writer.Write(128ul); writer.Write(64ul); writer.Write(2ul); }
        else { writer.Write(1u); writer.Write(16384ul); }
        writer.Write((uint)GgmlTensorType.F32); writer.Write(shard == 2 ? 32ul : 0ul);
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        if (shard == 2) writer.Write(new byte[32]);
        if (unitWeights) for (int i = 0; i < 16384; i++) writer.Write(1f);
        else writer.Write(Enumerable.Repeat((byte)(16 + shard), 65536).ToArray());
        return path;
    }

    private static void Text(BinaryWriter writer, string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text); writer.Write((ulong)bytes.Length); writer.Write(bytes);
    }
}

internal sealed class MuseSparkSourcePagesCudaFactAttribute : FactAttribute
{
    public MuseSparkSourcePagesCudaFactAttribute(int devices = 1)
        => Skip = !OperatingSystem.IsLinux() ? "Source-page advice requires Linux."
            : TestGates.CudaDevicesSkip(devices) ?? TestGates.GgmlPinSkip(BackendType.GgmlCuda);
}
