using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using TensorSharp.Distributed;

namespace InferenceWeb.Tests;

/// <summary>Real loopback framing, pipeline transfer and failure propagation.
/// These tests do not qualify model quality or a physical network route.</summary>
public sealed class TcpTransportSafetyTests
{
    [Fact]
    public void OriginalTwoArgumentConstructorRemainsAvailableToCompiledConsumers()
        => Assert.NotNull(typeof(TcpCommunicator).GetConstructor([typeof(int), typeof(IPEndPoint[])]));

    [Theory]
    [InlineData("tensor-parallel:2", "layer-split:2")]
    [InlineData("layer-split:1", "layer-split:2")]
    public async Task ConfigurationPreflightRejectsModeOrDegreeMismatchOnBothNodes(string first, string second)
    {
        var nodes = await Connect(2);
        try
        {
            // Matching preflight may precede further model-specific preflights.
            await Task.WhenAll(nodes.Select(node => Task.Run(() => node.ValidateConfiguration("shared-model"))))
                .WaitAsync(TimeSpan.FromSeconds(5));
            var errors = await Task.WhenAll(
                Task.Run(() => Record.Exception(() => nodes[0].ValidateConfiguration(first))),
                Task.Run(() => Record.Exception(() => nodes[1].ValidateConfiguration(second))))
                .WaitAsync(TimeSpan.FromSeconds(5));
            Assert.All(errors, error => Assert.NotNull(error));
            Assert.Contains(errors, error => error is InvalidOperationException && error.Message.Contains("configuration mismatch"));
        }
        finally { foreach (var node in nodes) node.Dispose(); }
    }

    [Fact]
    public async Task ThreeStagePipelinePreservesFloatBitsAndInterleavesControls()
    {
        var nodes = await Connect(3);
        try
        {
            int[] sizes = [0, 1, 1023, 1024, 1025, 1024 * 1024, 7];
            var stages = Enumerable.Range(1, 2).Select(rank => Task.Run(() =>
            {
                foreach (int size in sizes)
                {
                    var (op, payload) = nodes[rank].ReceiveControl();
                    Assert.Equal(17, op);
                    Assert.Equal([size], payload);
                    var data = new float[size];
                    nodes[rank].ReceiveTensor(data, rank - 1);
                    nodes[rank].SendTensor(data, rank == 2 ? 0 : 2);
                }
                nodes[rank].Barrier();
            })).ToArray();
            foreach (int size in sizes)
            {
                nodes[0].BroadcastControl(17, [size]);
                // Include signed zero, infinities, NaNs and denormal encodings:
                // transport must preserve bytes, not quantize or sanitize values.
                var sent = Enumerable.Range(0, size).Select(i => BitConverter.Int32BitsToSingle(
                    i % 5 == 0 ? unchecked((int)0x80000000) : i % 5 == 1 ? 0x7FC12345 : i * 12347)).ToArray();
                var received = new float[size];
                nodes[0].SendTensor(sent, 1);
                nodes[0].ReceiveTensor(received, 2);
                Assert.Equal(sent.Select(BitConverter.SingleToInt32Bits), received.Select(BitConverter.SingleToInt32Bits));
            }
            nodes[0].Barrier();
            await Task.WhenAll(stages).WaitAsync(TimeSpan.FromSeconds(15));
        }
        finally { foreach (var node in nodes) node.Dispose(); }
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(4, 1)]
    [InlineData(1, 4)]
    public async Task TensorShapeMismatchDoesNotWriteDestinationAndFaultsPeer(int sentCount, int receivedCount)
    {
        var nodes = await Connect(2);
        try
        {
            var destination = Enumerable.Repeat(123f, receivedCount).ToArray();
            var sender = Task.Run(() => Record.Exception(() =>
            {
                nodes[0].SendTensor(new float[sentCount], 1);
                nodes[0].Barrier();
            }));
            var error = Assert.Throws<InvalidOperationException>(() => nodes[1].ReceiveTensor(destination, 0));
            Assert.Contains("shape mismatch", error.Message);
            Assert.All(destination, value => Assert.Equal(123f, value));
            Assert.NotNull(await sender.WaitAsync(TimeSpan.FromSeconds(5)));
            Assert.Throws<IOException>(() => nodes[1].ReceiveControl());
        }
        finally { foreach (var node in nodes) node.Dispose(); }
    }

    [Fact]
    public async Task BroadcastFromNonzeroRootChecksShapeAndFailsAllParticipants()
    {
        var nodes = await Connect(2);
        try
        {
            var root = Task.Run(() => Record.Exception(() => nodes[1].Broadcast(new float[3], 1)));
            Assert.Throws<InvalidOperationException>(() => nodes[0].Broadcast(new float[1], 1));
            Assert.NotNull(await root.WaitAsync(TimeSpan.FromSeconds(5)));
        }
        finally { foreach (var node in nodes) node.Dispose(); }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public async Task MissingPeerHasBoundedConnectOrAcceptDeadline(int rank)
    {
        var endpoints = Endpoints(2);
        var error = await Task.Run(() => Record.Exception(() =>
        {
            using var node = new TcpCommunicator(rank, endpoints, TimeSpan.FromMilliseconds(250));
        })).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.IsType<TimeoutException>(error);
    }

    [Fact]
    public async Task DisposeInterruptsBlockedReceiveWithoutWaitingForSocketTimeout()
    {
        var nodes = await Connect(2);
        try
        {
            var receiver = Task.Run(() => Record.Exception(() => nodes[1].ReceiveControl()));
            nodes[1].Dispose();
            Assert.NotNull(await receiver.WaitAsync(TimeSpan.FromSeconds(5)));
        }
        finally { foreach (var node in nodes) node.Dispose(); }
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-1, 0)]
    [InlineData(4, 0)]
    [InlineData(9, -1)]
    [InlineData(9, int.MaxValue)]
    [InlineData(int.MaxValue, 0)]
    public async Task MalformedControlFramesFailBeforeAllocatingPayload(int wireLength, int declaredCount)
    {
        var endpoints = Endpoints(2);
        var connecting = Task.Run(() => new TcpCommunicator(1, endpoints, TimeSpan.FromSeconds(5)));
        using var raw = new TcpClient();
        await ConnectRaw(raw, endpoints[1]);
        using var stream = raw.GetStream();
        byte[] handshake = new byte[16];
        BinaryPrimitives.WriteInt32LittleEndian(handshake, 0x54535032);
        BinaryPrimitives.WriteInt32LittleEndian(handshake.AsSpan(4), 3);
        BinaryPrimitives.WriteInt32LittleEndian(handshake.AsSpan(8), 0);
        BinaryPrimitives.WriteInt32LittleEndian(handshake.AsSpan(12), 2);
        await stream.WriteAsync(handshake);
        await stream.ReadExactlyAsync(handshake);
        using var node = await connecting.WaitAsync(TimeSpan.FromSeconds(5));
        byte[] message = new byte[13];
        BinaryPrimitives.WriteInt32LittleEndian(message, wireLength);
        message[4] = 4;
        BinaryPrimitives.WriteInt32LittleEndian(message.AsSpan(9), declaredCount);
        await stream.WriteAsync(message);
        var error = await Task.Run(() => Record.Exception(() => node.ReceiveControl())).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.IsType<InvalidOperationException>(error);
    }

    [Fact]
    public async Task WrongProtocolFailsHandshakeInsteadOfEnteringInference()
    {
        var endpoints = Endpoints(2);
        var connecting = Task.Run(() => Record.Exception(() =>
        {
            using var node = new TcpCommunicator(1, endpoints, TimeSpan.FromSeconds(5));
        }));
        using var raw = new TcpClient();
        await ConnectRaw(raw, endpoints[1]);
        await raw.GetStream().WriteAsync(new byte[4]); // legacy rank-only handshake
        var error = await connecting.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.IsType<InvalidOperationException>(error);
        Assert.Contains("Incompatible TCP protocol", error.Message);
    }

    [Fact]
    public async Task ForwardThatAcceptsBeforeDestinationStartsRetriesHandshakeEof()
    {
        var endpoints = Endpoints(2);
        var forwarding = new TcpListener(endpoints[1]);
        forwarding.Start();
        TcpCommunicator driver = null;
        try
        {
            var connecting = Task.Run(() => new TcpCommunicator(0, endpoints, TimeSpan.FromSeconds(5)));
            using (var notReady = await forwarding.AcceptTcpClientAsync())
            {
                // Match SSH forwarding when the target endpoint is not yet listening.
                notReady.Close();
            }
            using var ready = await forwarding.AcceptTcpClientAsync().WaitAsync(TimeSpan.FromSeconds(5));
            var stream = ready.GetStream();
            byte[] handshake = new byte[16];
            await stream.ReadExactlyAsync(handshake);
            BinaryPrimitives.WriteInt32LittleEndian(handshake.AsSpan(8), 1);
            await stream.WriteAsync(handshake);
            driver = await connecting.WaitAsync(TimeSpan.FromSeconds(5));
            Assert.Equal(0, driver.Rank);
        }
        finally { driver?.Dispose(); forwarding.Stop(); }
    }

    [Theory]
    [InlineData("heartbeats")]
    [InlineData("partial-header")]
    [InlineData("partial-payload")]
    public async Task DataFrameDeadlineIncludesHeartbeatsAndEveryFragment(string traffic)
    {
        var (node, raw) = await ConnectRawPeer(TimeSpan.FromSeconds(2), TimeSpan.FromMilliseconds(250));
        using (node)
        using (raw)
        using (var stop = new CancellationTokenSource())
        {
            var stream = raw.GetStream();
            var sending = Task.Run(async () =>
            {
                try
                {
                    if (traffic == "heartbeats")
                    {
                        byte[] heartbeat = [1, 0, 0, 0, 5];
                        while (true)
                        {
                            await stream.WriteAsync(heartbeat, stop.Token);
                            await Task.Delay(25, stop.Token);
                        }
                    }
                    else
                    {
                        byte[] frame = new byte[5 + 4 * sizeof(float)];
                        BinaryPrimitives.WriteInt32LittleEndian(frame, frame.Length - 4);
                        frame[4] = 6;
                        int offset = traffic == "partial-header" ? 0 : 5;
                        if (offset != 0) await stream.WriteAsync(frame.AsMemory(0, offset), stop.Token);
                        for (; offset < frame.Length; offset++)
                        {
                            await stream.WriteAsync(frame.AsMemory(offset, 1), stop.Token);
                            await Task.Delay(75, stop.Token);
                        }
                    }
                }
                catch (Exception ex) when (ex is IOException or SocketException or OperationCanceledException) { }
            });
            try
            {
                var error = await Task.Run(() => Record.Exception(() => node.ReceiveTensor(new float[4], 0)))
                    .WaitAsync(TimeSpan.FromSeconds(5));
                Assert.IsType<TimeoutException>(error);
                Assert.Contains("operation deadline", error.Message);
                Assert.Throws<IOException>(() => node.ReceiveControl());
            }
            finally
            {
                stop.Cancel();
                await sending.WaitAsync(TimeSpan.FromSeconds(5));
            }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task IdleControlAndLoadBarrierTolerateHeartbeatsBeyondDataDeadline(bool barrier)
    {
        var (node, raw) = await ConnectRawPeer(TimeSpan.FromSeconds(2), TimeSpan.FromMilliseconds(150));
        using (node)
        using (raw)
        {
            (int op, int[] payload) control = default;
            var receiving = Task.Run(() =>
            {
                if (barrier) node.Barrier();
                else control = node.ReceiveControl();
            });
            var stream = raw.GetStream();
            byte[] heartbeat = [1, 0, 0, 0, 5];
            for (int i = 0; i < 8; i++)
            {
                await stream.WriteAsync(heartbeat);
                await Task.Delay(60);
            }
            Assert.False(receiving.IsCompleted);
            if (barrier)
                await stream.WriteAsync(new byte[] { 1, 0, 0, 0, 3 });
            else
            {
                byte[] frame = new byte[13];
                BinaryPrimitives.WriteInt32LittleEndian(frame, 9);
                frame[4] = 4;
                BinaryPrimitives.WriteInt32LittleEndian(frame.AsSpan(5), 27);
                await stream.WriteAsync(frame);
            }
            await receiving.WaitAsync(TimeSpan.FromSeconds(5));
            if (!barrier)
            {
                Assert.Equal(27, control.op);
                Assert.Empty(control.payload);
            }
        }
    }

    private static async Task<(TcpCommunicator, TcpClient)> ConnectRawPeer(TimeSpan ioTimeout, TimeSpan operationTimeout)
    {
        var endpoints = Endpoints(2);
        var connecting = Task.Run(() => new TcpCommunicator(1, endpoints,
            TimeSpan.FromSeconds(5), ioTimeout, operationTimeout));
        var raw = new TcpClient();
        try
        {
            await ConnectRaw(raw, endpoints[1]);
            var stream = raw.GetStream();
            byte[] handshake = new byte[16];
            BinaryPrimitives.WriteInt32LittleEndian(handshake, 0x54535032);
            BinaryPrimitives.WriteInt32LittleEndian(handshake.AsSpan(4), 3);
            BinaryPrimitives.WriteInt32LittleEndian(handshake.AsSpan(12), 2);
            await stream.WriteAsync(handshake);
            await stream.ReadExactlyAsync(handshake);
            return (await connecting.WaitAsync(TimeSpan.FromSeconds(5)), raw);
        }
        catch { raw.Dispose(); throw; }
    }

    private static async Task ConnectRaw(TcpClient client, IPEndPoint endpoint)
    {
        for (int attempt = 0; ; attempt++)
        {
            try { await client.ConnectAsync(endpoint.Address, endpoint.Port); return; }
            catch (SocketException) when (attempt < 20) { await Task.Delay(20); }
        }
    }

    private static async Task<TcpCommunicator[]> Connect(int count)
    {
        var endpoints = Endpoints(count);
        var tasks = Enumerable.Range(0, count).Select(rank => Task.Run(() =>
            new TcpCommunicator(rank, endpoints, TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(10)))).ToArray();
        return await Task.WhenAll(tasks).WaitAsync(TimeSpan.FromSeconds(15));
    }

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
}
