using System;
using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace TensorSharp.Distributed
{
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
            MsgBarrierAck = 3, MsgControl = 4, MsgHeartbeat = 5, MsgTensor = 6, MsgBroadcast = 7;
        private const int ProtocolMagic = 0x54535032, ProtocolVersion = 2;
        private const int SmallFrameBytes = 4096, MaxControlBytes = 16 * 1024 * 1024;
        private readonly int _rank, _worldSize, _ioTimeoutMs, _operationTimeoutMs;
        private readonly TcpListener _listener;
        private readonly TcpClient[] _peers;
        private readonly NetworkStream[] _streams;
        private readonly object[] _sendLocks;
        private readonly object _operationLock = new();
        private Timer _heartbeatTimer;
        private float[] _reduceBuffer = Array.Empty<float>();
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
        {
            ArgumentNullException.ThrowIfNull(endpoints);
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
                        int remoteRank = ReadHandshake(stream);
                        if (remoteRank < 0 || remoteRank >= rank || _peers[remoteRank] != null)
                            throw new InvalidOperationException($"Invalid or duplicate inbound rank {remoteRank} at rank {rank}.");
                        _peers[remoteRank] = client;
                        _streams[remoteRank] = stream;
                        WriteHandshake(stream);
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
            Console.WriteLine($"[TcpCommunicator] Rank {rank}/{_worldSize} connected to all peers (protocol {ProtocolVersion}).");
        }

        public int Rank => _rank;
        public int WorldSize => _worldSize;

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
                    client.ConnectAsync(endpoint.Address, endpoint.Port, cancellation).AsTask().GetAwaiter().GetResult();
                    var stream = Configure(client);
                    SetHandshakeTimeout(stream, expires);
                    WriteHandshake(stream);
                    int receivedRank = ReadHandshake(stream);
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

        private void WriteHandshake(NetworkStream stream)
        {
            Span<byte> data = stackalloc byte[16];
            BinaryPrimitives.WriteInt32LittleEndian(data, ProtocolMagic);
            BinaryPrimitives.WriteInt32LittleEndian(data[4..], ProtocolVersion);
            BinaryPrimitives.WriteInt32LittleEndian(data[8..], _rank);
            BinaryPrimitives.WriteInt32LittleEndian(data[12..], _worldSize);
            stream.Write(data);
        }

        private int ReadHandshake(NetworkStream stream)
        {
            Span<byte> data = stackalloc byte[16];
            ReadExact(stream, data[..4]);
            if (BinaryPrimitives.ReadInt32LittleEndian(data) != ProtocolMagic)
                throw new InvalidOperationException("Incompatible TCP protocol. Run the same TensorSharp build on every node.");
            ReadExact(stream, data[4..]);
            if (BinaryPrimitives.ReadInt32LittleEndian(data[4..]) != ProtocolVersion ||
                BinaryPrimitives.ReadInt32LittleEndian(data[12..]) != _worldSize)
                throw new InvalidOperationException("TCP protocol version or node count differs between peers.");
            return BinaryPrimitives.ReadInt32LittleEndian(data[8..]);
        }

        /// <summary>Lossless rank-ordered sum. Only root needs a reusable receive buffer.
        /// Direct synchronous I/O avoids a thread wakeup for every decode collective.</summary>
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
                    var values = buffer.AsSpan(0, length);
                    var bytes = MemoryMarshal.AsBytes(values);
                    if (_rank == 0)
                    {
                        if (_reduceBuffer.Length < length) _reduceBuffer = new float[length];
                        var incoming = _reduceBuffer.AsSpan(0, length);
                        for (int r = 1; r < _worldSize; r++)
                        {
                            ReceiveInto(r, MsgAllReduceContribute, MemoryMarshal.AsBytes(incoming));
                            int i = 0;
                            if (Vector.IsHardwareAccelerated)
                                for (; i <= length - Vector<float>.Count; i += Vector<float>.Count)
                                    (new Vector<float>(values.Slice(i)) + new Vector<float>(incoming.Slice(i))).CopyTo(values.Slice(i));
                            for (; i < length; i++) values[i] += incoming[i];
                        }
                        for (int r = 1; r < _worldSize; r++) SendMessage(r, MsgAllReduceResult, bytes);
                    }
                    else
                    {
                        SendMessage(0, MsgAllReduceContribute, bytes);
                        ReceiveInto(0, MsgAllReduceResult, bytes);
                    }
                    // TCP orders the first and every subsequent operation; no extra barrier.
                }
                catch (Exception ex) { Fail(ex); throw; }
            }
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
                    Span<byte> data = size <= SmallFrameBytes ? stackalloc byte[size] : new byte[size];
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
                // Coalesce small decode/control messages into one syscall/TCP segment.
                Span<byte> frame = stackalloc byte[SmallFrameBytes];
                BinaryPrimitives.WriteInt32LittleEndian(frame, payload.Length + 1);
                frame[4] = type;
                var stream = _streams[destination];
                if (payload.Length <= SmallFrameBytes - 5)
                {
                    payload.CopyTo(frame[5..]);
                    stream.Write(frame[..(5 + payload.Length)]);
                }
                else
                {
                    stream.Write(frame[..5]);
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
                        int timeout = Math.Min(_ioTimeoutMs, Math.Max(1, remainingMs));
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
