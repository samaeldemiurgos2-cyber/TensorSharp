// https://github.com/zhongkaifu/TensorSharp
using System.Net;
using TensorSharp.Distributed;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for the TCP communicator and distributed TP configuration.
/// These run in-process using localhost loopback connections — no
/// CUDA hardware or network peers are required.
/// </summary>
public class TcpCommunicatorTests
{
    private static int _nextPort = 19500;
    private static int GetFreePort() => Interlocked.Increment(ref _nextPort);

    private static (TcpCommunicator, TcpCommunicator) CreatePair()
    {
        int port0 = GetFreePort();
        int port1 = GetFreePort();
        var endpoints = new[]
        {
            new IPEndPoint(IPAddress.Loopback, port0),
            new IPEndPoint(IPAddress.Loopback, port1),
        };

        TcpCommunicator c0 = null, c1 = null;
        var t0 = new Thread(() => c0 = new TcpCommunicator(0, endpoints));
        var t1 = new Thread(() => c1 = new TcpCommunicator(1, endpoints));
        t0.Start(); t1.Start();
        t0.Join(10_000); t1.Join(10_000);

        Assert.NotNull(c0);
        Assert.NotNull(c1);
        return (c0, c1);
    }

    [Fact]
    public void AllReduce_TwoNodes_SumsCorrectly()
    {
        var (c0, c1) = CreatePair();
        try
        {
            var buf0 = new float[] { 1f, 2f, 3f, 4f };
            var buf1 = new float[] { 10f, 20f, 30f, 40f };

            var t0 = new Thread(() => c0.AllReduce(buf0, buf0.Length));
            var t1 = new Thread(() => c1.AllReduce(buf1, buf1.Length));
            t0.Start(); t1.Start();
            t0.Join(10_000); t1.Join(10_000);

            Assert.Equal(new float[] { 11f, 22f, 33f, 44f }, buf0);
            Assert.Equal(new float[] { 11f, 22f, 33f, 44f }, buf1);
        }
        finally
        {
            c0.Dispose(); c1.Dispose();
        }
    }

    [Fact]
    public void AllReduce_LargeBuffer_SumsCorrectly()
    {
        var (c0, c1) = CreatePair();
        try
        {
            int n = 1024 * 1024; // 1M elements = 4 MB
            var buf0 = new float[n];
            var buf1 = new float[n];
            for (int i = 0; i < n; i++)
            {
                buf0[i] = i * 0.001f;
                buf1[i] = i * 0.002f;
            }

            var t0 = new Thread(() => c0.AllReduce(buf0, buf0.Length));
            var t1 = new Thread(() => c1.AllReduce(buf1, buf1.Length));
            t0.Start(); t1.Start();
            t0.Join(30_000); t1.Join(30_000);

            // AllReduce computes i*0.001f + i*0.002f, which is NOT bit-identical
            // to i*0.003f: each product rounds independently and the sum rounds
            // again. At i near 1M the value is ~3000, where one fp32 ulp is
            // ~2.4e-4 — far coarser than the 5-decimal-place (1e-5) bound this
            // used to assert, so the test failed on the first index whose sum
            // happened to round the other way (i=12196: 36.5880051 vs
            // 36.5880013). Assert a relative fp32 tolerance instead: a real
            // reduce bug (a dropped chunk, a wrong offset, a partial buffer)
            // moves values by whole percent, not by ulps.
            const float relTol = 1e-5f;
            for (int i = 0; i < n; i++)
            {
                float expected = i * 0.003f;
                float tol = Math.Max(1e-6f, Math.Abs(expected) * relTol);
                Assert.True(Math.Abs(expected - buf0[i]) <= tol,
                    $"rank0[{i}]: expected {expected}, actual {buf0[i]}");
                Assert.True(Math.Abs(expected - buf1[i]) <= tol,
                    $"rank1[{i}]: expected {expected}, actual {buf1[i]}");
            }
        }
        finally
        {
            c0.Dispose(); c1.Dispose();
        }
    }

    [Fact]
    public void Barrier_TwoNodes_DoesNotDeadlock()
    {
        var (c0, c1) = CreatePair();
        try
        {
            var t0 = new Thread(() => c0.Barrier());
            var t1 = new Thread(() => c1.Barrier());
            t0.Start(); t1.Start();
            bool ok0 = t0.Join(10_000);
            bool ok1 = t1.Join(10_000);
            Assert.True(ok0, "Barrier timed out on rank 0");
            Assert.True(ok1, "Barrier timed out on rank 1");
        }
        finally
        {
            c0.Dispose(); c1.Dispose();
        }
    }

    [Fact]
    public void Broadcast_FromRank0_ReachesRank1()
    {
        var (c0, c1) = CreatePair();
        try
        {
            var buf0 = new float[] { 42f, 43f, 44f };
            var buf1 = new float[] { 0f, 0f, 0f };

            var t0 = new Thread(() => c0.Broadcast(buf0, rootRank: 0));
            var t1 = new Thread(() => c1.Broadcast(buf1, rootRank: 0));
            t0.Start(); t1.Start();
            t0.Join(10_000); t1.Join(10_000);

            Assert.Equal(new float[] { 42f, 43f, 44f }, buf0);
            Assert.Equal(new float[] { 42f, 43f, 44f }, buf1);
        }
        finally
        {
            c0.Dispose(); c1.Dispose();
        }
    }

    [Fact]
    public void AllReduce_ThreeNodes_SumsCorrectly()
    {
        int port0 = GetFreePort();
        int port1 = GetFreePort();
        int port2 = GetFreePort();
        var endpoints = new[]
        {
            new IPEndPoint(IPAddress.Loopback, port0),
            new IPEndPoint(IPAddress.Loopback, port1),
            new IPEndPoint(IPAddress.Loopback, port2),
        };

        TcpCommunicator c0 = null, c1 = null, c2 = null;
        var t0 = new Thread(() => c0 = new TcpCommunicator(0, endpoints));
        var t1 = new Thread(() => c1 = new TcpCommunicator(1, endpoints));
        var t2 = new Thread(() => c2 = new TcpCommunicator(2, endpoints));
        t0.Start(); t1.Start(); t2.Start();
        t0.Join(10_000); t1.Join(10_000); t2.Join(10_000);

        try
        {
            var buf0 = new float[] { 1f, 2f };
            var buf1 = new float[] { 10f, 20f };
            var buf2 = new float[] { 100f, 200f };

            var ta0 = new Thread(() => c0.AllReduce(buf0, buf0.Length));
            var ta1 = new Thread(() => c1.AllReduce(buf1, buf1.Length));
            var ta2 = new Thread(() => c2.AllReduce(buf2, buf2.Length));
            ta0.Start(); ta1.Start(); ta2.Start();
            ta0.Join(10_000); ta1.Join(10_000); ta2.Join(10_000);

            Assert.Equal(new float[] { 111f, 222f }, buf0);
            Assert.Equal(new float[] { 111f, 222f }, buf1);
            Assert.Equal(new float[] { 111f, 222f }, buf2);
        }
        finally
        {
            c0.Dispose(); c1.Dispose(); c2.Dispose();
        }
    }

    [Fact]
    public void MultipleSequentialAllReduces_Work()
    {
        var (c0, c1) = CreatePair();
        try
        {
            for (int iter = 0; iter < 5; iter++)
            {
                float v = iter + 1;
                var buf0 = new float[] { v, v * 2 };
                var buf1 = new float[] { v * 10, v * 20 };

                var t0 = new Thread(() => c0.AllReduce(buf0, buf0.Length));
                var t1 = new Thread(() => c1.AllReduce(buf1, buf1.Length));
                t0.Start(); t1.Start();
                t0.Join(10_000); t1.Join(10_000);

                Assert.Equal(v * 11, buf0[0], 3);
                Assert.Equal(v * 22, buf0[1], 3);
            }
        }
        finally
        {
            c0.Dispose(); c1.Dispose();
        }
    }

    [Fact]
    public void BroadcastBytes_FromRank0_ArrivesVerbatim_AndKeepsLaterOperationsInOrder()
    {
        var (c0, c1) = CreatePair();
        try
        {
            // KV snapshot sized, with bytes that read as NaN when taken as floats.
            var sent = new byte[3 * 1024 * 1024 + 7];
            new Random(5).NextBytes(sent);
            sent[0] = 0xFF; sent[1] = 0xFF; sent[2] = 0xFF; sent[3] = 0x7F;
            var buf0 = (byte[])sent.Clone();
            var buf1 = new byte[sent.Length];
            var sum0 = new float[] { 1f };
            var sum1 = new float[] { 2f };

            var t0 = new Thread(() => { c0.BroadcastBytes(buf0); c0.AllReduce(sum0, 1); });
            var t1 = new Thread(() => { c1.BroadcastBytes(buf1); c1.AllReduce(sum1, 1); });
            t0.Start(); t1.Start();
            Assert.True(t0.Join(30_000) && t1.Join(30_000));

            Assert.Equal(sent, buf0);
            Assert.Equal(sent, buf1);
            Assert.Equal(3f, sum0[0]);
            Assert.Equal(3f, sum1[0]);
        }
        finally
        {
            c0.Dispose(); c1.Dispose();
        }
    }

    [Fact]
    public void GatherBytesToRoot_ThreeNodes_UnitesDisjointRangesOnRank0Only()
    {
        var endpoints = new[]
        {
            new IPEndPoint(IPAddress.Loopback, GetFreePort()),
            new IPEndPoint(IPAddress.Loopback, GetFreePort()),
            new IPEndPoint(IPAddress.Loopback, GetFreePort()),
        };
        var nodes = new TcpCommunicator[3];
        var connect = Enumerable.Range(0, 3).Select(r => new Thread(() => nodes[r] = new TcpCommunicator(r, endpoints))).ToArray();
        foreach (var t in connect) t.Start();
        foreach (var t in connect) t.Join(10_000);
        try
        {
            // Uneven ranges, not a multiple of the vector width.
            var whole = new byte[1001];
            new Random(9).NextBytes(whole);
            int[] cuts = { 0, 333, 334, whole.Length };
            var buffers = new byte[3][];
            for (int r = 0; r < 3; r++)
            {
                buffers[r] = new byte[whole.Length];
                Array.Copy(whole, cuts[r], buffers[r], cuts[r], cuts[r + 1] - cuts[r]);
            }
            var before = buffers.Select(b => (byte[])b.Clone()).ToArray();

            var run = Enumerable.Range(0, 3).Select(r => new Thread(() => nodes[r].GatherBytesToRoot(buffers[r]))).ToArray();
            foreach (var t in run) t.Start();
            Assert.All(run, t => Assert.True(t.Join(10_000)));

            Assert.Equal(whole, buffers[0]);
            Assert.Equal(before[1], buffers[1]);
            Assert.Equal(before[2], buffers[2]);
        }
        finally
        {
            foreach (var n in nodes) n?.Dispose();
        }
    }
}

public class DistributedTpConfigTests
{
    [Fact]
    public void ParsePeers_ValidInput_ReturnsEndpoints()
    {
        var peers = DistributedTpConfig.ParsePeers("192.168.1.10:9500,192.168.1.11:9501");
        Assert.Equal(2, peers.Length);
        Assert.Equal(IPAddress.Parse("192.168.1.10"), peers[0].Address);
        Assert.Equal(9500, peers[0].Port);
        Assert.Equal(IPAddress.Parse("192.168.1.11"), peers[1].Address);
        Assert.Equal(9501, peers[1].Port);
    }

    [Fact]
    public void ParsePeers_SinglePeer_ReturnsOneEndpoint()
    {
        var peers = DistributedTpConfig.ParsePeers("10.0.0.1:8000");
        Assert.Single(peers);
        Assert.Equal(8000, peers[0].Port);
    }

    [Fact]
    public void ParsePeers_InvalidFormat_Throws()
    {
        Assert.Throws<FormatException>(() => DistributedTpConfig.ParsePeers("nohostport"));
    }

    [Fact]
    public void ParsePeers_InvalidPort_Throws()
    {
        Assert.Throws<FormatException>(() => DistributedTpConfig.ParsePeers("host:notaport"));
    }

    [Fact]
    public void ParsePeers_EmptyString_Throws()
    {
        Assert.Throws<ArgumentException>(() => DistributedTpConfig.ParsePeers(""));
    }

    [Fact]
    public void Constructor_NodeIdOutOfRange_Throws()
    {
        var peers = new[] { new IPEndPoint(IPAddress.Loopback, 9500) };
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new DistributedTpConfig(5, 1, peers));
    }

    [Fact]
    public void GlobalDegree_ComputedCorrectly()
    {
        var peers = new[]
        {
            new IPEndPoint(IPAddress.Loopback, 9500),
            new IPEndPoint(IPAddress.Loopback, 9501),
        };
        var config = new DistributedTpConfig(0, 2, peers);
        Assert.Equal(4, config.GlobalDegree);
    }

    [Fact]
    public void TryFromEnvironment_NoVars_ReturnsNull()
    {
        // Ensure the env vars are not set.
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_NODE_ID", null);
        Environment.SetEnvironmentVariable("TENSORSHARP_TP_PEERS", null);

        var config = DistributedTpConfig.TryFromEnvironment(2);
        Assert.Null(config);
    }
}
