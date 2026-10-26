// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Net.Sockets;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Distributed;

namespace InferenceWeb.Tests;

/// <summary>Exercises the real TCP transport with CPU-backed local ranks.
/// This is collective/control coverage, not a GPU or model quality gate.</summary>
public sealed class DistributedTensorParallelGroupTests
{
    [Fact]
    public async Task HierarchicalReductionUsesLogicalViewsAndPreservesBackingSentinels()
    {
        var endpoints = Endpoints();
        var connect0 = Task.Run(() => new DistributedTensorParallelGroup(new LocalCpuGroup(2), 0, endpoints));
        var connect1 = Task.Run(() => new DistributedTensorParallelGroup(new LocalCpuGroup(2), 1, endpoints));
        await Task.WhenAll(connect0, connect1).WaitAsync(TimeSpan.FromSeconds(15));
        using var node0 = await connect0;
        using var node1 = await connect1;
        Assert.Equal(4, node0.GlobalDegree);
        Assert.Equal(2, node1.GlobalRankOffset);
        Assert.Equal(2, node1.NodeCount);
        Assert.IsType<LocalCpuGroup>(node0.LocalGroup);

        var backing = new Tensor[4];
        var views = new Tensor[4];
        try
        {
            for (int rank = 0; rank < 4; rank++)
            {
                backing[rank] = new Tensor(node0.GetAllocator(0), DType.Float32, 9 + rank);
                var values = Enumerable.Repeat(-100f - rank, 9 + rank).ToArray();
                for (int i = 0; i < 3; i++) values[rank + 1 + i] = rank + 1;
                backing[rank].SetElementsAsFloat(values);
                views[rank] = backing[rank].Narrow(0, rank + 1, 3);
            }
            await Task.WhenAll(
                Task.Run(() => node0.AllReduce([views[0], views[1]])),
                Task.Run(() => node1.AllReduce([views[2], views[3]])))
                .WaitAsync(TimeSpan.FromSeconds(15));
            for (int rank = 0; rank < 4; rank++)
            {
                float[] actual = backing[rank].GetElementsAsFloat(9 + rank);
                for (int i = 0; i < actual.Length; i++)
                    Assert.Equal(i >= rank + 1 && i < rank + 4 ? 10f : -100f - rank, actual[i]);
            }
        }
        finally
        {
            foreach (var view in views) view?.Dispose();
            foreach (var tensor in backing) tensor?.Dispose();
        }
    }

    [Fact]
    public async Task DriverControlsInterleaveWithPrefillAndRepeatedDecodeCollectives()
    {
        var (driver, worker) = await Connect();
        using (driver)
        using (worker)
        {
            int[] sizes = [8192, 1, 17, 1, 4096, 1];
            var workerTask = Task.Run(() =>
            {
                foreach (int size in sizes)
                {
                    var (op, payload) = worker.ReceiveControl();
                    Assert.Equal(1, op);
                    Assert.Equal(new[] { size, 1017 }, payload);
                    var buffer = Enumerable.Repeat(2f, size).ToArray();
                    worker.AllReduce(buffer, size);
                    Assert.All(buffer, value => Assert.Equal(3f, value));
                }
                var (shutdown, empty) = worker.ReceiveControl();
                Assert.Equal(4, shutdown);
                Assert.Empty(empty);
            });
            foreach (int size in sizes)
            {
                driver.BroadcastControl(1, [size, 1017]);
                var buffer = Enumerable.Repeat(1f, size).ToArray();
                await Task.Run(() => driver.AllReduce(buffer, size)).WaitAsync(TimeSpan.FromSeconds(15));
                Assert.All(buffer, value => Assert.Equal(3f, value));
            }
            driver.BroadcastControl(4, []);
            await workerTask.WaitAsync(TimeSpan.FromSeconds(15));
        }
    }

    [Theory]
    [InlineData(1, 4)]
    [InlineData(4, 1)]
    public async Task MismatchedCollectiveShapesFailBeforeUnsafePayloadRead(int driverSize, int workerSize)
    {
        var (driver, worker) = await Connect();
        using (driver)
        using (worker)
        {
            var workerTask = Task.Run(() => Record.Exception(() => worker.AllReduce(new float[workerSize], workerSize)));
            var error = await Task.Run(() => Record.Exception(() => driver.AllReduce(new float[driverSize], driverSize)))
                .WaitAsync(TimeSpan.FromSeconds(15));
            Assert.IsType<InvalidOperationException>(error);
            Assert.Contains("Collective shape mismatch", error.Message);
            // The mismatched operation is terminal; releasing the driver also
            // releases the worker waiting for a result that cannot be produced.
            driver.Dispose();
            Assert.NotNull(await workerTask.WaitAsync(TimeSpan.FromSeconds(15)));
        }
    }

    [Fact]
    public async Task InvalidCollectiveLengthsAreRejectedWithoutEnteringTransport()
    {
        var (driver, worker) = await Connect();
        using (driver)
        using (worker)
        {
            Assert.Throws<ArgumentNullException>(() => driver.AllReduce(null, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => driver.AllReduce(new float[3], -1));
            Assert.Throws<ArgumentOutOfRangeException>(() => driver.AllReduce(new float[3], 4));
        }
    }

    private static async Task<(TcpCommunicator driver, TcpCommunicator worker)> Connect()
    {
        var endpoints = Endpoints();
        var driver = Task.Run(() => new TcpCommunicator(0, endpoints));
        var worker = Task.Run(() => new TcpCommunicator(1, endpoints));
        await Task.WhenAll(driver, worker).WaitAsync(TimeSpan.FromSeconds(15));
        return (await driver, await worker);
    }

    private static IPEndPoint[] Endpoints()
    {
        var listeners = new[] { new TcpListener(IPAddress.Loopback, 0), new TcpListener(IPAddress.Loopback, 0) };
        try
        {
            foreach (var listener in listeners) listener.Start();
            return listeners.Select(listener => (IPEndPoint)listener.LocalEndpoint).ToArray();
        }
        finally { foreach (var listener in listeners) listener.Stop(); }
    }

    private sealed class LocalCpuGroup(int degree) : ITensorParallelGroup
    {
        private readonly CpuAllocator _allocator = new(BlasEnum.DotNet);
        public int Degree => degree;
        public bool IsActive => degree > 1;
        public int GlobalDegree => degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => _allocator;
        public void AllReduce(Tensor[] tensors)
        {
            int count = checked((int)tensors[0].ElementCount());
            var sum = new float[count];
            foreach (var tensor in tensors)
            {
                var values = tensor.GetElementsAsFloat(count);
                for (int i = 0; i < count; i++) sum[i] += values[i];
            }
            foreach (var tensor in tensors) tensor.SetElementsAsFloat(sum);
        }
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
