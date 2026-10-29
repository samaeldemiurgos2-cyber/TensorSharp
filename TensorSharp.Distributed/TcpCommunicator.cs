using System;
using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Distributed
{
    /// <summary>All-reduce exchange pattern. Both produce the same rank-ordered sum.</summary>
    public enum TcpAllReduceAlgorithm
    {
        /// <summary>Workers send to rank 0, which sums and returns the result: one round trip.</summary>
        Star = 0,
        /// <summary>Every node sends to every peer and sums locally: one one-way latency.</summary>
        Symmetric = 1,
    }

    /// <summary>Wire settings every node must share; they are checked in the handshake.</summary>
    public sealed record TcpTransportOptions
    {
        public TcpAllReduceAlgorithm AllReduce { get; init; } = TcpAllReduceAlgorithm.Symmetric;

        /// <summary>Symmetric payloads up to this size are sent inline before receiving.
        /// Larger ones are written on a separate thread so two large concurrent
        /// writes cannot fill both directions' buffers and deadlock.</summary>
        public int SymmetricInlineBytes { get; init; } = 32 * 1024;

        public static TcpTransportOptions FromEnvironment()
        {
            var options = new TcpTransportOptions();
            string algorithm = Environment.GetEnvironmentVariable("TENSORSHARP_TP_ALLREDUCE");
            if (!string.IsNullOrWhiteSpace(algorithm))
            {
                options = options with
                {
                    AllReduce = algorithm.Trim().ToLowerInvariant() switch
                    {
                        "star" => TcpAllReduceAlgorithm.Star,
                        "symmetric" => TcpAllReduceAlgorithm.Symmetric,
                        _ => throw new ArgumentException("TENSORSHARP_TP_ALLREDUCE must be 'symmetric' or 'star'."),
                    },
                };
            }
            string inline = Environment.GetEnvironmentVariable("TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES");
            if (!string.IsNullOrWhiteSpace(inline))
            {
                if (!int.TryParse(inline, out int bytes) || bytes < 0)
                    throw new ArgumentException("TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES must be a non-negative integer.");
                options = options with { SymmetricInlineBytes = bytes };
            }
            return options;
        }
    }

    /// <summary>
    /// Persistent TCP mesh shared by tensor parallelism and layer pipelines.
    /// Frames contain an int32 little-endian length (including the type byte),
    /// a type byte, then the payload. Float payloads are lossless little-endian F32.
    /// Callers must issue operations in matching order. A failed operation closes
    /// the mesh so peers cannot continue with a partially consumed frame.
    /// </summary>
    public sealed class TcpCommunicator : IDisposable
    {
        private const byte MsgAllReduceContribute = 1, MsgAllReduceResult = 2,
            MsgBarrierAck = 3, MsgControl = 4, MsgHeartbeat = 5, MsgTensor = 6, MsgBroadcast = 7,
            MsgAllReduceExchange = 8, MsgBytesBroadcast = 9, MsgBytesGather = 10;
        private const int ProtocolMagic = 0x54535032;
        /// <summary>Version 3 adds the all-reduce settings to the handshake; version 4 the raw
        /// byte broadcast and gather (KV snapshots).</summary>
        public const int ProtocolVersion = 4;
        /// <summary>Base handshake (magic, version, rank, world size) plus the all-reduce settings.</summary>
        public const int HandshakeBytes = 24;
        // Frames up to this size are copied behind their header and written once,
        // so a decode collective is one segment (and one SSH packet on a tunnel).
        private const int CoalesceBytes = 64 * 1024, MaxControlBytes = 16 * 1024 * 1024;
        private readonly int _rank, _worldSize, _ioTimeoutMs, _operationTimeoutMs;
        private readonly TcpTransportOptions _options;
        private readonly TcpListener _listener;
        private readonly TcpClient[] _peers;
        private readonly NetworkStream[] _streams;
        private readonly object[] _sendLocks;
        private readonly byte[][] _frames;
        private readonly object _operationLock = new();
        private Timer _heartbeatTimer;
        private float[] _reduceBuffer = Array.Empty<float>();
        private float[][] _peerBuffers;
        private byte[] _gatherBuffer = Array.Empty<byte>();
        private Exception _fault;
        private int _closed, _disposed;

        // Keep the original CLR signature for consumers compiled before timeout
        // overrides were added; optional parameters alone preserve only source compatibility.
        public TcpCommunicator(int rank, IPEndPoint[] endpoints)
            : this(rank, endpoints, null, null)
        {
        }

        public TcpCommunicator(int rank, IPEndPoint[] endpoints,
            TimeSpan? connectTimeout = null, TimeSpan? ioTimeout = null)
            : this(rank, endpoints, connectTimeout, ioTimeout, null)
        {
        }

        /// <param name="rank">This node's rank (0..worldSize-1).</param>
        /// <param name="endpoints">Listen/connect endpoints indexed by node rank.</param>
        /// <param name="connectTimeout">Deadline for mesh setup, including inbound
        /// accepts and handshakes. Defaults to the environment setting.</param>
        /// <param name="ioTimeout">Read/write timeout; defaults to the environment setting.</param>
        /// <param name="operationTimeout">Deadline for a complete data frame, including
        /// heartbeat-only waiting. Defaults to TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS,
        /// or the I/O timeout when unset. Idle controls and load barriers are exempt.</param>
        public TcpCommunicator(int rank, IPEndPoint[] endpoints,
            TimeSpan? connectTimeout, TimeSpan? ioTimeout, TimeSpan? operationTimeout)
            : this(rank, endpoints, connectTimeout, ioTimeout, operationTimeout, null)
        {
        }

        /// <param name="options">Wire settings; defaults to <see cref="TcpTransportOptions.FromEnvironment"/>.
        /// Every node must use the same settings, which the handshake verifies.</param>
        public TcpCommunicator(int rank, IPEndPoint[] endpoints,
            TimeSpan? connectTimeout, TimeSpan? ioTimeout, TimeSpan? operationTimeout,
            TcpTransportOptions options)
        {
            ArgumentNullException.ThrowIfNull(endpoints);
            _options = options ?? TcpTransportOptions.FromEnvironment();
            if (_options.SymmetricInlineBytes < 0)
                throw new ArgumentOutOfRangeException(nameof(options), "SymmetricInlineBytes must not be negative.");
            // Both nodes write up to two inline frames per direction before reading.
            // A socket buffer override smaller than that would block both writers.
            string socketBuffer = Environment.GetEnvironmentVariable("TENSORSHARP_TP_SOCKET_BUFFER_BYTES");
            if (_options.AllReduce == TcpAllReduceAlgorithm.Symmetric && int.TryParse(socketBuffer, out int socketBytes)
                && 2L * (_options.SymmetricInlineBytes + 5) > socketBytes)
                throw new ArgumentException($"TENSORSHARP_TP_SOCKET_BUFFER_BYTES ({socketBytes}) must hold two inline frames; " +
                    $"lower TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES to at most {Math.Max(0, socketBytes / 2 - 5)}.");
            if (endpoints.Length < 2)
                throw new ArgumentException("TCP communicator requires at least 2 nodes.", nameof(endpoints));
            if ((uint)rank >= (uint)endpoints.Length) throw new ArgumentOutOfRangeException(nameof(rank));
            foreach (var endpoint in endpoints) ArgumentNullException.ThrowIfNull(endpoint);
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("Distributed F32 transport requires a little-endian host.");
            _rank = rank;
            _worldSize = endpoints.Length;
            _ioTimeoutMs = TimeoutMilliseconds(ioTimeout ?? ResolveTimeout("TENSORSHARP_TP_RECV_TIMEOUT_SECONDS", 300));
            _operationTimeoutMs = TimeoutMilliseconds(operationTimeout ??
                (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS"))
                    ? TimeSpan.FromMilliseconds(_ioTimeoutMs)
                    : ResolveTimeout("TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS", 300)));
            int connectMs = TimeoutMilliseconds(connectTimeout ?? ResolveTimeout("TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS", 120));
            _peers = new TcpClient[_worldSize];
            _streams = new NetworkStream[_worldSize];
            _sendLocks = new object[_worldSize];
            _frames = new byte[_worldSize][];
            _peerBuffers = new float[_worldSize][];
            for (int r = 0; r < _worldSize; r++) _sendLocks[r] = new object();
            _listener = new TcpListener(endpoints[rank]);
            try
            {
                _listener.Start();
                using var deadline = new CancellationTokenSource(connectMs);
                var expires = DateTime.UtcNow.AddMilliseconds(connectMs);
                // Exactly one connection per pair, with a bidirectional handshake.
                for (int r = rank + 1; r < _worldSize; r++)
                {
                    var (client, stream) = ConnectWithRetry(endpoints[r], r, expires, deadline.Token);
                    _peers[r] = client;
                    _streams[r] = stream;
                }
                for (int accepted = 0; accepted < rank; accepted++)
                {
                    var client = _listener.AcceptTcpClientAsync(deadline.Token).AsTask().GetAwaiter().GetResult();
                    try
                    {
                        var stream = Configure(client);
                        SetHandshakeTimeout(stream, expires);
                        int remoteRank = ReadHandshake(stream, out bool sameSettings);
                        if (remoteRank < 0 || remoteRank >= rank || _peers[remoteRank] != null)
                            throw new InvalidOperationException($"Invalid or duplicate inbound rank {remoteRank} at rank {rank}.");
                        // Reply before refusing different settings: the dialer then sees the
                        // mismatch itself instead of retrying a closed connection until its deadline.
                        WriteHandshake(stream);
                        if (!sameSettings) throw TransportSettingsMismatch();
                        _peers[remoteRank] = client;
                        _streams[remoteRank] = stream;
                        stream.ReadTimeout = stream.WriteTimeout = _ioTimeoutMs;
                    }
                    catch { client.Dispose(); throw; }
                }
                _listener.Stop();
            }
            catch (OperationCanceledException ex)
            {
                Dispose();
                throw new TimeoutException($"Rank {rank} timed out establishing the TCP mesh. Check peer endpoints and TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS.", ex);
            }
            catch { Dispose(); throw; }
            _heartbeatTimer = new Timer(SendHeartbeats, null, 15_000, 15_000);
            Console.WriteLine($"[TcpCommunicator] Rank {rank}/{_worldSize} connected to all peers (protocol {ProtocolVersion}, " +
                $"all-reduce {_options.AllReduce.ToString().ToLowerInvariant()}, inline {_options.SymmetricInlineBytes} B).");
        }

        public int Rank => _rank;
        public int WorldSize => _worldSize;
        public TcpTransportOptions Options => _options;

        /// <summary>Verify that every node agrees on a mode, placement or model
        /// identity before doing dependent work. This is a consistency check,
        /// not authentication. Every node must call it at the same point.</summary>
        public void ValidateConfiguration(string identity)
        {
            ArgumentNullException.ThrowIfNull(identity);
            byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(identity));
            float[] votes = new float[digest.Length * 8];
            for (int i = 0; i < votes.Length; i++) votes[i] = (digest[i / 8] >> (i % 8)) & 1;
            lock (_operationLock)
            {
                AllReduce(votes, votes.Length);
                foreach (float count in votes)
                    if (count != 0 && count != _worldSize)
                    {
                        var error = new InvalidOperationException("Distributed configuration mismatch: all nodes must use the same parallel mode, local degree and model configuration.");
                        Fail(error);
                        throw error;
                    }
            }
        }

        private static TimeSpan ResolveTimeout(string variable, int fallbackSeconds)
        {
            string value = Environment.GetEnvironmentVariable(variable);
            if (string.IsNullOrWhiteSpace(value)) return TimeSpan.FromSeconds(fallbackSeconds);
            if (!int.TryParse(value, out int seconds) || seconds < 1 || seconds > int.MaxValue / 1000)
                throw new ArgumentException($"{variable} must be between 1 and {int.MaxValue / 1000} seconds.");
            return TimeSpan.FromSeconds(seconds);
        }

        private static int TimeoutMilliseconds(TimeSpan timeout)
        {
            if (timeout.TotalMilliseconds < 1 || timeout.TotalMilliseconds > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(timeout));
            return (int)Math.Ceiling(timeout.TotalMilliseconds);
        }

        private void SetHandshakeTimeout(NetworkStream stream, DateTime expires)
        {
            int remaining = (int)Math.Ceiling((expires - DateTime.UtcNow).TotalMilliseconds);
            if (remaining < 1) throw new TimeoutException("TCP mesh handshake deadline expired.");
            stream.ReadTimeout = stream.WriteTimeout = Math.Min(remaining, _ioTimeoutMs);
        }

        private NetworkStream Configure(TcpClient client)
        {
            client.NoDelay = true;
            client.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
            // Keep OS autotuning by default; allow an explicit budget for measured routes.
            string value = Environment.GetEnvironmentVariable("TENSORSHARP_TP_SOCKET_BUFFER_BYTES");
            if (!string.IsNullOrWhiteSpace(value))
            {
                if (!int.TryParse(value, out int bytes) || bytes < 4096)
                    throw new ArgumentException("TENSORSHARP_TP_SOCKET_BUFFER_BYTES must be an integer >= 4096.");
                client.SendBufferSize = client.ReceiveBufferSize = bytes;
            }
            var stream = client.GetStream();
            stream.ReadTimeout = stream.WriteTimeout = _ioTimeoutMs;
            return stream;
        }

        private (TcpClient, NetworkStream) ConnectWithRetry(IPEndPoint endpoint, int remoteRank,
            DateTime expires, CancellationToken cancellation)
        {
            bool announced = false;
            while (true)
            {
                cancellation.ThrowIfCancellationRequested();
                var client = new TcpClient(endpoint.AddressFamily);
                try
                {
                    if (OperatingSystem.IsWindows())
                        client.ConnectAsync(endpoint.Address, endpoint.Port, cancellation).AsTask().GetAwaiter().GetResult();
                    else
                    {
                        // An async connect permanently switches a Unix socket to .NET's
                        // emulated non-blocking mode, which routes every later blocking
                        // read through the epoll thread. Rank 0 receives every star
                        // contribution and every stage's logits, so keep its sockets
                        // on direct blocking reads. SO_SNDTIMEO bounds connect() here.
                        int remaining = (int)Math.Ceiling((expires - DateTime.UtcNow).TotalMilliseconds);
                        if (remaining < 1) cancellation.ThrowIfCancellationRequested();
                        client.Client.SendTimeout = Math.Max(1, remaining);
                        client.Connect(endpoint.Address, endpoint.Port);
                    }
                    var stream = Configure(client);
                    SetHandshakeTimeout(stream, expires);
                    WriteHandshake(stream);
                    int receivedRank = ReadHandshake(stream, out bool sameSettings);
                    if (!sameSettings) throw TransportSettingsMismatch();
                    if (receivedRank != remoteRank)
                        throw new InvalidOperationException($"Expected rank {remoteRank} at {endpoint}, received rank {receivedRank}.");
                    stream.ReadTimeout = stream.WriteTimeout = _ioTimeoutMs;
                    return (client, stream);
                }
                // A TCP proxy/SSH forward can accept before its destination is
                // listening, then immediately close. The session is not connected
                // until its handshake succeeds. Protocol disagreements still fail
                // immediately; only pre-handshake transport failures are retried.
                catch (Exception ex) when ((ex is SocketException or IOException) && !cancellation.IsCancellationRequested)
                {
                    client.Dispose();
                    if (!announced)
                    {
                        Console.WriteLine($"[TcpCommunicator] Rank {_rank}: waiting for rank {remoteRank} at {endpoint}...");
                        announced = true;
                    }
                    cancellation.WaitHandle.WaitOne(250);
                }
                catch (Exception ex) when (cancellation.IsCancellationRequested)
                {
                    client.Dispose();
                    throw new OperationCanceledException("TCP mesh setup deadline expired.", ex, cancellation);
                }
                catch { client.Dispose(); throw; }
            }
        }

        /// <summary>Handshake bytes a node with these settings sends: magic, version,
        /// rank, world size, all-reduce algorithm and symmetric inline limit.</summary>
        public static byte[] CreateHandshake(int rank, int worldSize, TcpTransportOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            var data = new byte[HandshakeBytes];
            BinaryPrimitives.WriteInt32LittleEndian(data, ProtocolMagic);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(4), ProtocolVersion);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(8), rank);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(12), worldSize);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(16), (int)options.AllReduce);
            BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(20), options.SymmetricInlineBytes);
            return data;
        }

        private void WriteHandshake(NetworkStream stream) => stream.Write(CreateHandshake(_rank, _worldSize, _options));

        private static InvalidOperationException TransportSettingsMismatch() => new(
            "Distributed transport settings differ between peers. Use the same " +
            "TENSORSHARP_TP_ALLREDUCE and TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES on every node.");

        private int ReadHandshake(NetworkStream stream, out bool sameSettings)
        {
            Span<byte> data = stackalloc byte[HandshakeBytes];
            ReadExact(stream, data[..4]);
            if (BinaryPrimitives.ReadInt32LittleEndian(data) != ProtocolMagic)
                throw new InvalidOperationException("Incompatible TCP protocol. Run the same TensorSharp build on every node.");
            // Check the version before reading the settings: an older peer sends no
            // settings, and waiting for them would stall until the handshake deadline.
            ReadExact(stream, data[4..16]);
            if (BinaryPrimitives.ReadInt32LittleEndian(data[4..]) != ProtocolVersion ||
                BinaryPrimitives.ReadInt32LittleEndian(data[12..]) != _worldSize)
                throw new InvalidOperationException("TCP protocol version or node count differs between peers. Run the same TensorSharp build on every node.");
            ReadExact(stream, data[16..]);
            sameSettings = BinaryPrimitives.ReadInt32LittleEndian(data[16..]) == (int)_options.AllReduce &&
                BinaryPrimitives.ReadInt32LittleEndian(data[20..]) == _options.SymmetricInlineBytes;
            return BinaryPrimitives.ReadInt32LittleEndian(data[8..]);
        }

        /// <summary>Lossless rank-ordered sum ((v0 + v1) + v2) + ..., bit-identical on
        /// every node and for both algorithms. Direct synchronous I/O avoids a thread
        /// wakeup for every decode collective.</summary>
        public void AllReduce(float[] buffer, int length)
        {
            ArgumentNullException.ThrowIfNull(buffer);
            if ((uint)length > (uint)buffer.Length || length > (int.MaxValue - 1) / sizeof(float))
                throw new ArgumentOutOfRangeException(nameof(length));
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try
                {
                    if (_options.AllReduce == TcpAllReduceAlgorithm.Symmetric) AllReduceSymmetric(buffer, length);
                    else AllReduceStar(buffer, length);
                    // TCP orders the first and every subsequent operation; no extra barrier.
                }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        private void AllReduceStar(float[] buffer, int length)
        {
            var values = buffer.AsSpan(0, length);
            var bytes = MemoryMarshal.AsBytes(values);
            if (_rank == 0)
            {
                if (_reduceBuffer.Length < length) _reduceBuffer = new float[length];
                var incoming = _reduceBuffer.AsSpan(0, length);
                for (int r = 1; r < _worldSize; r++)
                {
                    ReceiveInto(r, MsgAllReduceContribute, MemoryMarshal.AsBytes(incoming));
                    AddInto(values, incoming);
                }
                for (int r = 1; r < _worldSize; r++) SendMessage(r, MsgAllReduceResult, bytes);
            }
            else
            {
                SendMessage(0, MsgAllReduceContribute, bytes);
                ReceiveInto(0, MsgAllReduceResult, bytes);
            }
        }

        /// <summary>
        /// Every node sends its contribution to every peer, receives theirs, then
        /// sums in rank order. A node waits for one one-way latency instead of the
        /// star's round trip through rank 0. Small payloads are sent inline: two
        /// frames per direction fit in the socket buffers, so both nodes writing
        /// first cannot deadlock. Larger payloads use one writer per peer while
        /// this thread receives; a writer only waits for its own destination to read.
        /// </summary>
        private void AllReduceSymmetric(float[] buffer, int length)
        {
            for (int r = 0; r < _worldSize; r++)
                if (r != _rank && (_peerBuffers[r] == null || _peerBuffers[r].Length < length))
                    _peerBuffers[r] = new float[length];
            int bytes = length * sizeof(float);
            Task[] writers = null;
            if (bytes <= _options.SymmetricInlineBytes)
            {
                var own = MemoryMarshal.AsBytes(buffer.AsSpan(0, length));
                for (int r = 0; r < _worldSize; r++)
                    if (r != _rank) SendMessage(r, MsgAllReduceExchange, own);
            }
            else
            {
                // Writers only read the buffer; it is not summed into until they finish.
                writers = new Task[_worldSize - 1];
                for (int r = 0, w = 0; r < _worldSize; r++)
                {
                    if (r == _rank) continue;
                    int peer = r;
                    writers[w++] = Task.Factory.StartNew(
                        () => SendMessage(peer, MsgAllReduceExchange, MemoryMarshal.AsBytes(buffer.AsSpan(0, length))),
                        CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
                }
            }

            Exception receiveError = null;
            try
            {
                for (int r = 0; r < _worldSize; r++)
                    if (r != _rank)
                        ReceiveInto(r, MsgAllReduceExchange, MemoryMarshal.AsBytes(_peerBuffers[r].AsSpan(0, length)));
            }
            catch (Exception ex)
            {
                // Close the mesh first so a writer blocked on a stalled peer fails now
                // instead of waiting for its socket write timeout.
                receiveError = ex;
                Fail(ex);
            }
            if (writers != null)
            {
                try { Task.WaitAll(writers); }
                catch (AggregateException ex) when (receiveError == null) { ExceptionDispatchInfo.Throw(ex.InnerException ?? ex); }
                catch (AggregateException) { /* The receive failure is the primary error. */ }
            }
            if (receiveError != null) ExceptionDispatchInfo.Throw(receiveError);

            // Reproduce the star's order exactly. IEEE addition is commutative, so
            // own + partial equals partial + own; only the grouping must match.
            // (A lane holding two NaNs may keep either payload; every other value matches.)
            var values = buffer.AsSpan(0, length);
            if (_rank == 0)
            {
                for (int r = 1; r < _worldSize; r++) AddInto(values, _peerBuffers[r].AsSpan(0, length));
                return;
            }
            var lower = _peerBuffers[0].AsSpan(0, length);
            for (int r = 1; r < _rank; r++) AddInto(lower, _peerBuffers[r].AsSpan(0, length));
            AddInto(values, lower);
            for (int r = _rank + 1; r < _worldSize; r++) AddInto(values, _peerBuffers[r].AsSpan(0, length));
        }

        private static void AddInto(Span<float> destination, ReadOnlySpan<float> source)
        {
            int i = 0;
            if (Vector.IsHardwareAccelerated)
                for (; i <= destination.Length - Vector<float>.Count; i += Vector<float>.Count)
                    (new Vector<float>(destination.Slice(i)) + new Vector<float>(source.Slice(i))).CopyTo(destination.Slice(i));
            for (; i < destination.Length; i++) destination[i] += source[i];
        }

        /// <summary>Send a stage activation directly from its F32 source buffer.</summary>
        public void SendTensor(ReadOnlySpan<float> buffer, int destinationRank)
        {
            ValidateRemoteRank(destinationRank);
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try { SendMessage(destinationRank, MsgTensor, MemoryMarshal.AsBytes(buffer)); }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        /// <summary>Receive into an F32 destination, checking its expected size before writing.</summary>
        public void ReceiveTensor(Span<float> buffer, int sourceRank)
        {
            ValidateRemoteRank(sourceRank);
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try { ReceiveInto(sourceRank, MsgTensor, MemoryMarshal.AsBytes(buffer)); }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        public void BroadcastControl(int op, int[] payload)
        {
            if (_rank != 0) throw new InvalidOperationException("Only rank 0 may broadcast driver controls.");
            int count = payload?.Length ?? 0;
            if (count > (MaxControlBytes - 8) / 4) throw new ArgumentOutOfRangeException(nameof(payload));
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try
                {
                    int size = 8 + count * 4;
                    Span<byte> data = size <= 4096 ? stackalloc byte[size] : new byte[size];
                    BinaryPrimitives.WriteInt32LittleEndian(data, op);
                    BinaryPrimitives.WriteInt32LittleEndian(data[4..], count);
                    if (count > 0) MemoryMarshal.AsBytes(payload.AsSpan()).CopyTo(data[8..]);
                    for (int r = 1; r < _worldSize; r++) SendMessage(r, MsgControl, data);
                }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        public (int op, int[] payload) ReceiveControl()
        {
            if (_rank == 0) throw new InvalidOperationException("Rank 0 cannot receive its own driver controls.");
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try
                {
                    int bytes = ReceiveHeader(0, MsgControl);
                    if (bytes < 8 || bytes > MaxControlBytes || bytes % 4 != 0)
                        throw new InvalidOperationException($"Malformed control frame length {bytes}.");
                    Span<byte> header = stackalloc byte[8];
                    ReadExact(_streams[0], header);
                    int op = BinaryPrimitives.ReadInt32LittleEndian(header);
                    int count = BinaryPrimitives.ReadInt32LittleEndian(header[4..]);
                    if (count != (bytes - 8) / 4)
                        throw new InvalidOperationException($"Malformed control count {count} for {bytes} bytes.");
                    int[] payload = new int[count];
                    ReadExact(_streams[0], MemoryMarshal.AsBytes(payload.AsSpan()));
                    return (op, payload);
                }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        public void Broadcast(Span<float> buffer, int rootRank)
        {
            if ((uint)rootRank >= (uint)_worldSize) throw new ArgumentOutOfRangeException(nameof(rootRank));
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try
                {
                    var bytes = MemoryMarshal.AsBytes(buffer);
                    if (_rank == rootRank)
                    {
                        for (int r = 0; r < _worldSize; r++)
                            if (r != rootRank) SendMessage(r, MsgBroadcast, bytes);
                    }
                    else ReceiveInto(rootRank, MsgBroadcast, bytes);
                    BarrierCore();
                }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        /// <summary>Rank 0's bytes replace every other rank's, verbatim. TCP orders it
        /// against later operations, so no barrier follows.</summary>
        public void BroadcastBytes(Span<byte> buffer)
        {
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try
                {
                    if (_rank == 0)
                        for (int r = 1; r < _worldSize; r++) SendMessage(r, MsgBytesBroadcast, buffer);
                    else
                        ReceiveInto(0, MsgBytesBroadcast, buffer);
                }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        /// <summary>Every rank passes the bytes it owns and zeros elsewhere; rank 0 ends
        /// with the bitwise OR of them all, which is the union when the owned ranges are
        /// disjoint. Other ranks' buffers are left as they were.</summary>
        public void GatherBytesToRoot(Span<byte> buffer)
        {
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try
                {
                    if (_rank != 0)
                    {
                        SendMessage(0, MsgBytesGather, buffer);
                        return;
                    }
                    if (_gatherBuffer.Length < buffer.Length) _gatherBuffer = new byte[buffer.Length];
                    var incoming = _gatherBuffer.AsSpan(0, buffer.Length);
                    for (int r = 1; r < _worldSize; r++)
                    {
                        ReceiveInto(r, MsgBytesGather, incoming);
                        int i = 0;
                        if (Vector.IsHardwareAccelerated)
                            for (; i <= buffer.Length - Vector<byte>.Count; i += Vector<byte>.Count)
                                (new Vector<byte>(buffer.Slice(i)) | new Vector<byte>(incoming.Slice(i))).CopyTo(buffer.Slice(i));
                        for (; i < buffer.Length; i++) buffer[i] |= incoming[i];
                    }
                }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        public void Barrier()
        {
            lock (_operationLock)
            {
                ThrowIfUnavailable();
                try { BarrierCore(); }
                catch (Exception ex) { Fail(ex); throw; }
            }
        }

        private void BarrierCore()
        {
            if (_rank == 0)
            {
                for (int r = 1; r < _worldSize; r++) ReceiveInto(r, MsgBarrierAck, Span<byte>.Empty);
                for (int r = 1; r < _worldSize; r++) SendMessage(r, MsgBarrierAck, ReadOnlySpan<byte>.Empty);
            }
            else
            {
                SendMessage(0, MsgBarrierAck, ReadOnlySpan<byte>.Empty);
                ReceiveInto(0, MsgBarrierAck, Span<byte>.Empty);
            }
        }

        private void SendHeartbeats(object state)
        {
            if (Volatile.Read(ref _closed) != 0) return;
            for (int r = 0; r < _worldSize; r++)
            {
                // Busy writers already prove liveness; do not queue behind a large transfer.
                if (r == _rank || !Monitor.TryEnter(_sendLocks[r])) continue;
                try { SendMessage(r, MsgHeartbeat, ReadOnlySpan<byte>.Empty); }
                catch (Exception ex) { Fail(ex); return; }
                finally { Monitor.Exit(_sendLocks[r]); }
            }
        }

        private void SendMessage(int destination, byte type, ReadOnlySpan<byte> payload)
        {
            if (payload.Length == int.MaxValue) throw new ArgumentOutOfRangeException(nameof(payload));
            lock (_sendLocks[destination])
            {
                var stream = _streams[destination];
                if (payload.Length <= CoalesceBytes)
                {
                    // One write per decode/control frame: a separate header write is
                    // an extra segment, and an extra packet through an SSH tunnel.
                    // The buffer is per destination and guarded by its send lock.
                    int length = 5 + payload.Length;
                    byte[] frame = _frames[destination];
                    if (frame == null || frame.Length < length)
                        _frames[destination] = frame = new byte[Math.Min(CoalesceBytes + 5, Math.Max(length, 2 * (frame?.Length ?? 2048)))];
                    BinaryPrimitives.WriteInt32LittleEndian(frame, payload.Length + 1);
                    frame[4] = type;
                    payload.CopyTo(frame.AsSpan(5));
                    stream.Write(frame, 0, length);
                }
                else
                {
                    Span<byte> header = stackalloc byte[5];
                    BinaryPrimitives.WriteInt32LittleEndian(header, payload.Length + 1);
                    header[4] = type;
                    stream.Write(header);
                    stream.Write(payload);
                }
            }
        }

        private int ReceiveHeader(int source, byte expectedType, long operationDeadline = 0)
        {
            Span<byte> header = stackalloc byte[5];
            while (true)
            {
                ReadExact(_streams[source], header, operationDeadline);
                int length = BinaryPrimitives.ReadInt32LittleEndian(header);
                if (length < 1) throw new InvalidOperationException($"Invalid frame length {length} from rank {source}.");
                byte type = header[4];
                if (type == MsgHeartbeat)
                {
                    if (length != 1) throw new InvalidOperationException("A heartbeat must have an empty payload.");
                    continue;
                }
                if (type != expectedType)
                    throw new InvalidOperationException($"Expected message type 0x{expectedType:X2} from rank {source}, got 0x{type:X2}.");
                return length - 1;
            }
        }

        private void ReceiveInto(int source, byte type, Span<byte> destination)
        {
            // One monotonic deadline spans every heartbeat, header fragment and
            // payload fragment. Loading barriers may legitimately outlast it.
            long deadline = type == MsgBarrierAck ? 0 : Stopwatch.GetTimestamp()
                + (long)Math.Ceiling(_operationTimeoutMs * (double)Stopwatch.Frequency / 1000);
            int length = ReceiveHeader(source, type, deadline);
            if (length != destination.Length)
                throw new InvalidOperationException($"Collective shape mismatch from rank {source}: expected {destination.Length} bytes, received {length}. All nodes must execute the same operation with the same tensor shape.");
            ReadExact(_streams[source], destination, deadline);
        }

        private static TimeoutException OperationTimeout(Exception inner = null) => new(
            "Timed out waiting for a complete peer data frame: the operation deadline expired. " +
            "Heartbeats do not count as data progress. Check matching operations and TENSORSHARP_TP_OPERATION_TIMEOUT_SECONDS.", inner);

        private void ReadExact(NetworkStream stream, Span<byte> buffer, long operationDeadline = 0)
        {
            int offset = 0;
            int currentTimeout = _ioTimeoutMs;
            try
            {
                while (offset < buffer.Length)
                {
                    if (operationDeadline != 0)
                    {
                        long remainingTicks = operationDeadline - Stopwatch.GetTimestamp();
                        if (remainingTicks <= 0) throw OperationTimeout();
                        int remainingMs = (int)Math.Min(int.MaxValue,
                            Math.Ceiling(remainingTicks * 1000d / Stopwatch.Frequency));
                        // Shorten the socket timeout only when the deadline is at least a
                        // second sooner. Otherwise every read paid two setsockopt calls
                        // whenever the operation and I/O timeouts were equal (the default).
                        int timeout = remainingMs + 1000L < _ioTimeoutMs ? Math.Max(1, remainingMs) : _ioTimeoutMs;
                        if (timeout != currentTimeout)
                        {
                            stream.ReadTimeout = timeout;
                            currentTimeout = timeout;
                        }
                    }
                    int count;
                    try { count = stream.Read(buffer[offset..]); }
                    catch (IOException ex) when (ex.InnerException is SocketException se &&
                        (se.SocketErrorCode == SocketError.TimedOut || se.SocketErrorCode == SocketError.WouldBlock))
                    {
                        if (operationDeadline != 0 &&
                            (currentTimeout < _ioTimeoutMs || Stopwatch.GetTimestamp() >= operationDeadline))
                            throw OperationTimeout(ex);
                        throw new TimeoutException($"Timed out waiting for peer data ({buffer.Length} bytes expected, {offset} received). Adjust TENSORSHARP_TP_RECV_TIMEOUT_SECONDS.", ex);
                    }
                    if (count == 0) throw new EndOfStreamException("Peer closed the connection.");
                    offset += count;
                }
                if (operationDeadline != 0 && Stopwatch.GetTimestamp() >= operationDeadline)
                    throw OperationTimeout();
            }
            finally
            {
                if (operationDeadline != 0 && currentTimeout != _ioTimeoutMs)
                    try { stream.ReadTimeout = _ioTimeoutMs; } catch { /* abort may have closed this socket */ }
            }
        }

        private void ValidateRemoteRank(int rank)
        {
            if ((uint)rank >= (uint)_worldSize || rank == _rank) throw new ArgumentOutOfRangeException(nameof(rank));
        }

        private void ThrowIfUnavailable()
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            var fault = Volatile.Read(ref _fault);
            if (fault != null) throw new IOException("The TCP communicator is faulted; create a new group before retrying.", fault);
        }

        private void Fail(Exception error)
        {
            Interlocked.CompareExchange(ref _fault, error, null);
            CloseConnections();
        }

        /// <summary>Close the network after a stage/compute failure. No local
        /// backend resources are owned or released by this communicator.</summary>
        public void Abort(Exception failure)
        {
            ArgumentNullException.ThrowIfNull(failure);
            Fail(failure);
        }

        private void CloseConnections()
        {
            if (Interlocked.Exchange(ref _closed, 1) != 0) return;
            _heartbeatTimer?.Dispose();
            // Close before waiting: this wakes blocked readers/writers immediately.
            for (int r = 0; r < _worldSize; r++)
            {
                try { _peers[r]?.Client.Shutdown(SocketShutdown.Both); } catch { }
                try { _streams[r]?.Dispose(); } catch { }
                try { _peers[r]?.Dispose(); } catch { }
            }
            try { _listener.Stop(); } catch { }
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _disposed, 1);
            CloseConnections();
        }
    }
}
