// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class NativeDistributedExecutionTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(15)]
    [InlineData(4097)]
    public void OpaqueStagePayloadPreservesEveryBitAndNeverOverwritesTheTail(int bytes)
    {
        var group = new Group();
        var bridge = new NativeDistributedExecution(group);
        // Includes a signalling NaN and arbitrary non-float words. No floating
        // arithmetic or Marshal.Copy(float[]) may touch the stage payload.
        byte[] pattern = { 0x01, 0x00, 0x80, 0x7f, 0xff, 0x80, 0x43, 0x21 };
        byte[] input = Enumerable.Range(0, bytes).Select(i => pattern[i % pattern.Length]).ToArray();
        IntPtr memory = Marshal.AllocHGlobal(bytes + 8);
        try
        {
            Marshal.Copy(input, 0, memory, bytes);
            Assert.True(bridge.Transfer(IntPtr.Zero, 1, 1, memory, bytes));
            Assert.Equal((bytes + 3) / 4 * 4, group.Payload.Length);
            Assert.Equal(input, group.Payload[..bytes]);
            Assert.All(group.Payload[bytes..], value => Assert.Equal(0, value));
            byte[] sentinel = Enumerable.Repeat((byte)0xA5, bytes + 8).ToArray();
            Marshal.Copy(sentinel, 0, memory, sentinel.Length);
            Assert.True(bridge.Transfer(IntPtr.Zero, 1, 0, memory, bytes));
            var result = new byte[bytes + 8];
            Marshal.Copy(memory, result, 0, result.Length);
            Assert.Equal(input, result[..bytes]);
            Assert.All(result[bytes..], value => Assert.Equal(0xA5, value));
            Assert.Equal(0, group.AbortCount);
            Assert.Equal(0, group.DisposeCount);
        }
        finally { Marshal.FreeHGlobal(memory); }
    }

    [Fact]
    public void ReusedStageScratchDoesNotLeakPreviousTailBytes()
    {
        var group = new Group();
        var bridge = new NativeDistributedExecution(group);
        IntPtr memory = Marshal.AllocHGlobal(32);
        try
        {
            Marshal.Copy(Enumerable.Repeat((byte)0xFF, 32).ToArray(), 0, memory, 32);
            Assert.True(bridge.Transfer(IntPtr.Zero, 1, 1, memory, 32));
            foreach (int length in new[] { 1, 6, 15, 5 })
            {
                Assert.True(bridge.Transfer(IntPtr.Zero, 1, 1, memory, length));
                Assert.All(group.Payload[..length], value => Assert.Equal(0xFF, value));
                Assert.All(group.Payload[length..], value => Assert.Equal(0, value));
            }
        }
        finally { Marshal.FreeHGlobal(memory); }
    }

    [Fact]
    public void LocalReducedValuesReceiveExactlyOneCrossNodeSumIncludingWhenScratchShrinks()
    {
        int calls = 0;
        var group = new Group { Reduce = (values, count) =>
        {
            calls++;
            for (int i = 0; i < count; i++) values[i] += i + 0.5f;
        } };
        var bridge = new NativeDistributedExecution(group);
        IntPtr memory = Marshal.AllocHGlobal(64);
        try
        {
            foreach (int count in new[] { 16, 3, 1, 12 })
            {
                float[] input = Enumerable.Range(0, count).Select(i => (float)(i * 2 - 7)).ToArray();
                Marshal.Copy(input, 0, memory, count);
                Assert.True(bridge.AllReduce(IntPtr.Zero, memory, count));
                var result = new float[count];
                Marshal.Copy(memory, result, 0, count);
                Assert.Equal(input.Select((v, i) => v + i + 0.5f), result);
            }
            Assert.True(bridge.AllReduce(IntPtr.Zero, IntPtr.Zero, 0));
            Assert.Equal(4, calls);
        }
        finally { Marshal.FreeHGlobal(memory); }
    }

    [Theory]
    [InlineData(-1, 1, 4)]
    [InlineData(0, 1, 4)]
    [InlineData(2, 1, 4)]
    [InlineData(1, -1, 0)]
    [InlineData(1, 1, -1)]
    [InlineData(1, 1, 4)]
    public void InvalidNativeTransferAbortsBeforeDereferencingBuffers(int peer, int send, int bytes)
    {
        var group = new Group();
        var bridge = new NativeDistributedExecution(group);
        Assert.False(bridge.Transfer(IntPtr.Zero, peer, send, IntPtr.Zero, bytes));
        Assert.True(bridge.IsAborted);
        Assert.Equal(1, group.AbortCount);
        Assert.Equal(0, group.DisposeCount);
        Assert.False(bridge.Transfer(IntPtr.Zero, 1, 1, IntPtr.Zero, 0));
        Assert.False(bridge.AllReduce(IntPtr.Zero, IntPtr.Zero, 0));
        Assert.Equal(1, group.AbortCount);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    public void NativeReductionAbortAndNullBufferFailClosed(int count)
    {
        var group = new Group();
        var bridge = new NativeDistributedExecution(group);
        Assert.False(bridge.AllReduce(IntPtr.Zero, IntPtr.Zero, count));
        Assert.True(bridge.IsAborted);
        Assert.Equal(1, group.AbortCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TransportFailureCannotEscapeNativeAbiOrReleaseDeviceAllocations(bool receiving)
    {
        var original = new IOException("injected link failure");
        var group = new Group { TransferFailure = original, AbortThrows = true };
        var bridge = new NativeDistributedExecution(group);
        Assert.False(bridge.Transfer(IntPtr.Zero, 1, receiving ? 0 : 1, IntPtr.Zero, 0));
        Assert.Same(original, bridge.Failure);
        Assert.Equal(0, group.DisposeCount);
        bridge.Abort(new Exception("secondary failure"));
        Assert.Equal(1, group.AbortCount);
        var error = Assert.Throws<InvalidOperationException>(bridge.ThrowIfAborted);
        Assert.Same(original, error.InnerException);
    }

    [Fact]
    public void ConfigurationAgreementUsesExactBitVotes()
    {
        int calls = 0;
        var group = new Group { Reduce = (values, count) =>
        {
            Assert.Equal(256, count);
            Assert.All(values, value => Assert.True(value is 0 or 1));
            for (int i = 0; i < count; i++) values[i] *= 2;
            calls++;
        } };
        new NativeDistributedExecution(group).ValidateConfiguration("model|resolved topology and numerical settings");
        Assert.Equal(1, calls);
        Assert.Equal(0, group.AbortCount);
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void ConfigurationMismatchAbortsPeersBeforeNativeLoad(float badVote)
    {
        var group = new Group { Reduce = (values, count) =>
        {
            for (int i = 0; i < count; i++) values[i] *= 2;
            values[7] = badVote;
        } };
        var bridge = new NativeDistributedExecution(group);
        var error = Assert.Throws<InvalidOperationException>(() => bridge.ValidateConfiguration("configuration"));
        Assert.Same(error, bridge.Failure);
        Assert.Equal(1, group.AbortCount);
        Assert.Equal(0, group.DisposeCount);
    }

    [Fact]
    public void ConfigurationTransportFailureRemainsTheOriginalError()
    {
        var original = new IOException("injected reduction failure");
        var group = new Group { Reduce = (_, _) => throw original };
        var bridge = new NativeDistributedExecution(group);
        Assert.Same(original, Assert.Throws<IOException>(() => bridge.ValidateConfiguration("settings")));
        Assert.Same(original, bridge.Failure);
        Assert.Equal(1, group.AbortCount);
    }

    [Fact]
    public void MetadataIdentityCoversLateTokenizerAndTensorLayoutWithoutDependingOnDictionaryOrder()
    {
        string path = Path.Combine(Path.GetTempPath(), "native-distributed-identity-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            GlmDsaSyntheticModelBuilder.Write(path);
            using var file = new GgufFile(path);
            string header = NativeDistributedExecution.HeaderIdentity(path);
            string original = NativeDistributedExecution.MetadataIdentity(file);
            var metadata = file.Metadata.Reverse().ToArray();
            var tensors = file.Tensors.Reverse().ToArray();
            file.Metadata.Clear();
            file.Tensors.Clear();
            foreach (var entry in metadata) file.Metadata.Add(entry.Key, entry.Value);
            foreach (var entry in tensors) file.Tensors.Add(entry.Key, entry.Value);
            Assert.Equal(original, NativeDistributedExecution.MetadataIdentity(file));

            file.Metadata["test.large.tokenizer"] = new string('x', 128 * 1024) + "A";
            string large = NativeDistributedExecution.MetadataIdentity(file);
            file.Metadata["test.large.tokenizer"] = new string('x', 128 * 1024) + "B";
            Assert.NotEqual(large, NativeDistributedExecution.MetadataIdentity(file));
            file.Metadata.Remove("test.large.tokenizer");
            var tensor = file.Tensors.Values.First();
            var type = tensor.Type;
            tensor.Type = type == GgmlTensorType.F32 ? GgmlTensorType.F16 : GgmlTensorType.F32;
            Assert.NotEqual(original, NativeDistributedExecution.MetadataIdentity(file));
            tensor.Type = type;
            tensor.Shape[0]++;
            Assert.NotEqual(original, NativeDistributedExecution.MetadataIdentity(file));
            tensor.Shape[0]--;
            Assert.Equal(original, NativeDistributedExecution.MetadataIdentity(file));
            Assert.Equal(header, NativeDistributedExecution.HeaderIdentity(path));
        }
        finally { File.Delete(path); }
    }

    private sealed class Group : ILayerSplitGroup, INestedTensorParallelGroup, IDistributedExecutionGroup
    {
        public int NodeId => 0;
        public int NodeCount => 2;
        public int Degree => 2;
        public int GlobalDegree => 4;
        public int GlobalRankOffset => 0;
        public bool IsActive => false;
        public int AbortCount { get; private set; }
        public int DisposeCount { get; private set; }
        public bool AbortThrows { get; init; }
        public Exception TransferFailure { get; init; }
        public byte[] Payload { get; private set; } = Array.Empty<byte>();
        public Action<float[], int> Reduce { get; init; }
        public ITensorParallelGroup LocalGroup => throw new NotSupportedException();
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void CrossNodeAllReduce(float[] buffer, int count) => Reduce(buffer, count);
        public void Synchronize() => throw new NotSupportedException();
        public void Barrier() => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() => DisposeCount++;
        public void Abort(Exception failure)
        {
            AbortCount++;
            if (AbortThrows) throw new IOException("injected cleanup failure");
        }
        public void Send(float[] buffer, int count, int destination)
        {
            if (TransferFailure != null) throw TransferFailure;
            Payload = new byte[count * 4];
            Buffer.BlockCopy(buffer, 0, Payload, 0, Payload.Length);
        }
        public void Receive(float[] buffer, int count, int source)
        {
            if (TransferFailure != null) throw TransferFailure;
            Assert.Equal(count * 4, Payload.Length);
            Buffer.BlockCopy(Payload, 0, buffer, 0, Payload.Length);
        }
    }
}
