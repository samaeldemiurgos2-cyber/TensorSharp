// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using TensorSharp.Distributed;

namespace InferenceWeb.Tests;

/// <summary>
/// Symmetric all-reduce must give every node the star's exact rank-ordered sum,
/// stay deadlock-free above the inline limit, and refuse mismatched settings.
/// </summary>
public class TcpSymmetricAllReduceTests
{
    private static IPEndPoint[] Endpoints(int count)
    {
        var listeners = Enumerable.Range(0, count).Select(_ => new TcpListener(IPAddress.Loopback, 0)).ToArray();
        try
        {
            foreach (var listener in listeners) listener.Start();
            return listeners.Select(listener => (IPEndPoint)listener.LocalEndpoint).ToArray();
        }
        finally { foreach (var listener in listeners) listener.Stop(); }
    }

    private static TcpTransportOptions Options(TcpAllReduceAlgorithm algorithm, int inlineBytes = 32 * 1024) =>
        new() { AllReduce = algorithm, SymmetricInlineBytes = inlineBytes };

    private static TcpCommunicator[] Connect(int count, TcpTransportOptions options) =>
        Connect(Enumerable.Repeat(options, count).ToArray());

    private static TcpCommunicator[] Connect(TcpTransportOptions[] options)
    {
        var endpoints = Endpoints(options.Length);
        var tasks = options.Select((o, rank) => Task.Run(() => new TcpCommunicator(rank, endpoints,
            TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(30), null, o))).ToArray();
        Assert.True(Task.WaitAll(tasks, TimeSpan.FromSeconds(20)));
        return tasks.Select(t => t.Result).ToArray();
    }

    private static float[][] Contributions(int nodes, int length, int seed)
    {
        var random = new Random(seed);
        // Magnitudes spread over many exponents so grouping changes rounding,
        // plus signed zeros, infinities and subnormals in fixed lanes.
        var values = new float[nodes][];
        for (int r = 0; r < nodes; r++)
        {
            values[r] = new float[length];
            for (int i = 0; i < length; i++)
                values[r][i] = (float)((random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-6, 9)));
        }
        if (length >= 8 && nodes >= 3)
        {
            values[0][0] = 1e8f; values[1][0] = 1f; values[2][0] = -1e8f;
            values[0][1] = -0f; values[1][1] = -0f; values[2][1] = -0f;
            values[0][2] = float.PositiveInfinity; values[1][2] = 1f; values[2][2] = 2f;
            values[0][3] = float.Epsilon; values[1][3] = float.Epsilon; values[2][3] = -float.Epsilon;
        }
        return values;
    }

    private static float[] RankOrderedSum(float[][] values)
    {
        var sum = (float[])values[0].Clone();
        for (int r = 1; r < values.Length; r++)
            for (int i = 0; i < sum.Length; i++) sum[i] += values[r][i];
        return sum;
    }

    private static float[][] Run(TcpCommunicator[] nodes, float[][] contributions)
    {
        var buffers = contributions.Select(v => (float[])v.Clone()).ToArray();
        var tasks = nodes.Select((n, r) => Task.Run(() => n.AllReduce(buffers[r], buffers[r].Length))).ToArray();
        Assert.True(Task.WaitAll(tasks, TimeSpan.FromSeconds(60)));
        return buffers;
    }

    private static void AssertBitIdentical(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"lane {i}: expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Theory]
    [InlineData(2, 7)]
    [InlineData(2, 2560)]
    [InlineData(3, 7040)]
    [InlineData(3, 100_003)]
    public void SymmetricMatchesStarAndTheRankOrderedSumOnEveryNode(int nodeCount, int length)
    {
        var contributions = Contributions(nodeCount, length, seed: nodeCount * 1000 + length);
        float[] expected = RankOrderedSum(contributions);
        foreach (var algorithm in new[] { TcpAllReduceAlgorithm.Star, TcpAllReduceAlgorithm.Symmetric })
        {
            var nodes = Connect(nodeCount, Options(algorithm));
            try
            {
                foreach (var result in Run(nodes, contributions)) AssertBitIdentical(expected, result);
            }
            finally { foreach (var node in nodes) node.Dispose(); }
        }
    }

    [Fact]
    public void SizesAroundTheInlineLimitInterleaveWithOtherOperations()
    {
        const int limit = 4096;
        var nodes = Connect(2, Options(TcpAllReduceAlgorithm.Symmetric, limit));
        try
        {
            int[] lengths = [limit / 4 - 1, limit / 4, limit / 4 + 1, 1, 0, 300_000, limit / 4];
            var work = Task.Run(() =>
            {
                foreach (int length in lengths)
                {
                    var contributions = Contributions(2, length, seed: length);
                    float[] expected = RankOrderedSum(contributions);
                    var buffers = Run(nodes, contributions);
                    AssertBitIdentical(expected, buffers[0]);
                    AssertBitIdentical(expected, buffers[1]);

                    var tensor = contributions[0];
                    var received = new float[length];
                    var send = Task.Run(() => nodes[0].SendTensor(tensor, 1));
                    nodes[1].ReceiveTensor(received, 0);
                    send.Wait();
                    AssertBitIdentical(tensor, received);

                    var control = Task.Run(nodes[1].ReceiveControl);
                    nodes[0].BroadcastControl(length, [length, -length]);
                    Assert.Equal(length, control.Result.op);
                    Assert.Equal(new[] { length, -length }, control.Result.payload);
                }
            });
            Assert.True(work.Wait(TimeSpan.FromSeconds(60)));
        }
        finally { foreach (var node in nodes) node.Dispose(); }
    }

    [Fact]
    public void LargeSymmetricPayloadsDoNotDeadlockWhenBothNodesWriteFirst()
    {
        // 32 MiB per direction far exceeds the loopback socket buffers, so inline
        // writes on both nodes would block forever; the writer threads must not.
        var nodes = Connect(2, Options(TcpAllReduceAlgorithm.Symmetric));
        try
        {
            var contributions = Contributions(2, 8 * 1024 * 1024, seed: 5);
            float[] expected = RankOrderedSum(contributions);
            foreach (var result in Run(nodes, contributions)) AssertBitIdentical(expected, result);
        }
        finally { foreach (var node in nodes) node.Dispose(); }
    }

    [Theory]
    [InlineData(TcpAllReduceAlgorithm.Star, 32768, TcpAllReduceAlgorithm.Symmetric, 32768)]
    [InlineData(TcpAllReduceAlgorithm.Symmetric, 32768, TcpAllReduceAlgorithm.Symmetric, 16384)]
    public void MismatchedTransportSettingsFailTheHandshake(
        TcpAllReduceAlgorithm algorithm0, int inline0, TcpAllReduceAlgorithm algorithm1, int inline1)
    {
        var endpoints = Endpoints(2);
        var options = new[] { Options(algorithm0, inline0), Options(algorithm1, inline1) };
        // A long connect deadline: both nodes must fail on the settings themselves,
        // not by the dialer retrying a refused connection until the deadline.
        var tasks = options.Select((o, rank) => Task.Run(() => Record.Exception(() =>
        {
            using var node = new TcpCommunicator(rank, endpoints, TimeSpan.FromSeconds(60),
                TimeSpan.FromSeconds(5), null, o);
        }))).ToArray();
        Assert.True(Task.WaitAll(tasks, TimeSpan.FromSeconds(20)));
        Assert.All(tasks, t => Assert.Contains("TENSORSHARP_TP_ALLREDUCE",
            Assert.IsType<InvalidOperationException>(t.Result).Message));
    }

    [Fact]
    public async Task CoalescedFramesKeepTheWireFormat()
    {
        var endpoints = Endpoints(2);
        var options = Options(TcpAllReduceAlgorithm.Symmetric);
        var connecting = Task.Run(() => new TcpCommunicator(1, endpoints, TimeSpan.FromSeconds(5),
            TimeSpan.FromSeconds(5), null, options));
        using var raw = new TcpClient();
        for (int attempt = 0; ; attempt++)
        {
            try { await raw.ConnectAsync(endpoints[1].Address, endpoints[1].Port); break; }
            catch (SocketException) when (attempt < 50) { await Task.Delay(20); }
        }
        var stream = raw.GetStream();
        byte[] handshake = TcpCommunicator.CreateHandshake(0, 2, options);
        await stream.WriteAsync(handshake);
        await stream.ReadExactlyAsync(new byte[TcpCommunicator.HandshakeBytes]);
        using var node = await connecting.WaitAsync(TimeSpan.FromSeconds(5));

        foreach (int count in new[] { 3, 2560, 7040, 16384, 16385 })
        {
            var payload = Enumerable.Range(0, count).Select(i => i * 0.5f - 7f).ToArray();
            await Task.Run(() => node.SendTensor(payload, 0));
            byte[] header = new byte[5];
            await stream.ReadExactlyAsync(header);
            Assert.Equal(count * sizeof(float) + 1, BinaryPrimitives.ReadInt32LittleEndian(header));
            Assert.Equal(6, header[4]);
            byte[] body = new byte[count * sizeof(float)];
            await stream.ReadExactlyAsync(body);
            var received = new float[count];
            Buffer.BlockCopy(body, 0, received, 0, body.Length);
            AssertBitIdentical(payload, received);
        }
    }
}
