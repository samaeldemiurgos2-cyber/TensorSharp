// ============================================================================
// NemotronModel.TensorParallelSnapshot.cs
//
// KV-state snapshots under tensor parallelism, which is what lets the server's
// inference engine host a tensor-parallel Nemotron: it swaps requests on the one
// live cache by extracting and injecting fixed-size blocks.
//
// The block layout is the single-device one (see TryExtractKVBlock). It needs no TP
// variant: every attention layer's K/V heads are sharded CONTIGUOUSLY by global rank
// (ShardConcatenatedColumnParallel splits Q, K and V each by rank), so rank g's
// [kvHeads/G, tokens, row] shard is exactly heads g*kvHeads/G.. of the single-device
// [kvHeads, tokens, row] section. Concatenating the shards in rank order IS the
// single-device section. Mamba2 state is replicated, so it comes straight from this
// node's host arrays.
//
// Across nodes each node holds only its own ranks' heads, while the engine runs on the
// driver. Extract and inject are therefore mirrored operations: the driver broadcasts
// the op, then every node runs the same body and the bytes move between nodes as they
// are - an extract gathers each node's shards on the driver (TpGatherBytesToDriver), an
// inject sends the driver's whole block to every node (TpBroadcastBytesFromDriver).
// ============================================================================
using System;
using System.Diagnostics;

namespace TensorSharp.Models
{
    public partial class NemotronModel
    {
        private const int TpControlSnapshotExtract = TpControlModelFirst + 1; // [start, count, withState]
        private const int TpControlSnapshotInject = TpControlModelFirst + 2;  // [dest, count, withState, bytes]

        // Diagnostics for the snapshot path: TS_TP_SNAPSHOT_TRACE=1 logs every extract and
        // inject; without it a call over a second, and a running total every 64 calls, are logged.
        private static readonly bool TpSnapshotTrace = Environment.GetEnvironmentVariable("TS_TP_SNAPSHOT_TRACE") == "1";
        private long _tpSnapshotCalls, _tpSnapshotTicks, _tpSnapshotTransferTicks, _tpSnapshotBytes;

        private void TraceTpSnapshot(string op, int start, int count, bool withState, long bytes, long ticks, long transferTicks)
        {
            _tpSnapshotCalls++;
            _tpSnapshotTicks += ticks;
            _tpSnapshotTransferTicks += transferTicks;
            _tpSnapshotBytes += bytes;
            double ms = ticks * 1000.0 / Stopwatch.Frequency;
            if (TpSnapshotTrace || ms > 1000)
                Console.WriteLine($"[tp-snapshot] {op} rows {start}..{start + count} state={withState} bytes={bytes} " +
                    $"{ms:F1} ms (cross-node {transferTicks * 1000.0 / Stopwatch.Frequency:F1} ms) node={TpRankOffset}");
            if (_tpSnapshotCalls % 64 == 0)
                Console.WriteLine($"[tp-snapshot] {_tpSnapshotCalls} calls, {_tpSnapshotBytes / (1 << 20)} MiB, " +
                    $"{_tpSnapshotTicks * 1000.0 / Stopwatch.Frequency:F0} ms total " +
                    $"({_tpSnapshotTransferTicks * 1000.0 / Stopwatch.Frequency:F0} ms moving bytes between nodes)");
        }

        /// <summary>One attention layer's K (or V) section across every rank of the cluster.</summary>
        private long TpAttentionSectionBytes(int layer, int tokenCount)
            => NemotronLayerBlockBytes(_tpKvCacheK[layer][0], tokenCount) * GlobalTpDegree;

        /// <summary>The attention sections of a block, without the Mamba2 state between them.</summary>
        private long TpAttentionBlockBytes(int tokenCount)
        {
            long total = 0;
            for (int l = 0; l < Config.NumLayers; l++)
                if (_layerTypes[l] == LayerType.Attention)
                    total += 2 * TpAttentionSectionBytes(l, tokenCount);
            return total;
        }

        private bool TryExtractKVBlockTP(int startToken, int tokenCount, Span<byte> destination)
        {
            // Every check that can refuse runs BEFORE the broadcast: once the workers have
            // been told, they join the collective whatever the driver does next.
            if (NemotronBlockFormOf(tokenCount, destination.Length) is not bool withState) return false;
            if (startToken < 0 || tokenCount <= 0 || startToken + tokenCount > _cacheSeqLen) return false;
            if (TpAttentionBlockBytes(tokenCount) > int.MaxValue) return false;

            BroadcastModelControl(TpControlSnapshotExtract, new[] { startToken, tokenCount, withState ? 1 : 0 });
            try
            {
                RunTpSnapshotExtract(startToken, tokenCount, withState, destination, isDriver: true);
            }
            catch (Exception ex)
            {
                if (TpNodeCount > 1) AbortDistributedModelControl(ex);
                throw;
            }
            return true;
        }

        private bool TryInjectKVBlockTP(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _cacheSeqLen || tokenCount <= 0) return false;
            if (NemotronBlockFormOf(tokenCount, source.Length) is not bool withState) return false;
            if (destToken + tokenCount > _maxContextLength) return false;

            BroadcastModelControl(TpControlSnapshotInject,
                new[] { destToken, tokenCount, withState ? 1 : 0, source.Length });
            try
            {
                RunTpSnapshotInject(destToken, tokenCount, withState, source, source.Length, isDriver: true);
            }
            catch (Exception ex)
            {
                if (TpNodeCount > 1) AbortDistributedModelControl(ex);
                throw;
            }
            return true;
        }

        protected override bool HandleModelControl(int op, int[] payload)
        {
            switch (op)
            {
                case TpControlSnapshotExtract:
                    RunTpSnapshotExtract(payload[0], payload[1], payload[2] != 0, Span<byte>.Empty, isDriver: false);
                    return true;
                case TpControlSnapshotInject:
                    RunTpSnapshotInject(payload[0], payload[1], payload[2] != 0, ReadOnlySpan<byte>.Empty, payload[3], isDriver: false);
                    return true;
                default:
                    return base.HandleModelControl(op, payload);
            }
        }

        /// <summary>
        /// Gather this node's shards of rows [start, start + count) into the attention
        /// sections, complete them across nodes, and (on the driver) lay the block out
        /// with the Mamba2 state between them. A worker contributes its shards and keeps
        /// nothing.
        /// </summary>
        private void RunTpSnapshotExtract(int startToken, int tokenCount, bool withState, Span<byte> destination, bool isDriver)
        {
            long t0 = Stopwatch.GetTimestamp();
            long transferTicks = 0;
            var attention = new byte[TpAttentionBlockBytes(tokenCount)];
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Attention) continue;
                foreach (Tensor[] shards in new[] { _tpKvCacheK[l], _tpKvCacheV[l] })
                {
                    int perRank = checked((int)NemotronLayerBlockBytes(shards[0], tokenCount));
                    for (int r = 0; r < TpDegree; r++)
                    {
                        int at = offset + (TpRankOffset + r) * perRank;
                        if (!CopyAttentionOut(shards[r], startToken, tokenCount, attention.AsSpan(at, perRank), out _))
                            throw new InvalidOperationException($"Nemotron TP snapshot: layer {l} rank {r} has no rows {startToken}..{startToken + tokenCount}.");
                    }
                    offset += perRank * GlobalTpDegree;
                }
            }

            if (TpNodeCount > 1)
            {
                long r0 = Stopwatch.GetTimestamp();
                TpGatherBytesToDriver(attention);
                transferTicks = Stopwatch.GetTimestamp() - r0;
            }
            if (!isDriver)
            {
                TraceTpSnapshot("extract", startToken, tokenCount, withState, attention.Length,
                    Stopwatch.GetTimestamp() - t0, transferTicks);
                return;
            }

            int a = 0, o = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        int section = checked((int)(2 * TpAttentionSectionBytes(l, tokenCount)));
                        attention.AsSpan(a, section).CopyTo(destination.Slice(o, section));
                        a += section;
                        o += section;
                        break;
                    case LayerType.Mamba2:
                        if (!withState) break;
                        if (!CopyMamba2StateOut(l, destination[o..], out int written))
                            throw new InvalidOperationException($"Nemotron TP snapshot: no room for layer {l}'s Mamba2 state.");
                        o += written;
                        break;
                }
            }
            if (o != destination.Length)
                throw new InvalidOperationException($"Nemotron TP snapshot wrote {o} of {destination.Length} bytes.");
            TraceTpSnapshot("extract", startToken, tokenCount, withState, destination.Length,
                Stopwatch.GetTimestamp() - t0, transferTicks);
        }

        /// <summary>
        /// Write a block at <paramref name="destToken"/>: across nodes the driver's block
        /// reaches every node first, then each node keeps its own ranks' heads and the
        /// (replicated) Mamba2 state.
        /// </summary>
        private void RunTpSnapshotInject(int destToken, int tokenCount, bool withState, ReadOnlySpan<byte> source,
            int length, bool isDriver)
        {
            long t0 = Stopwatch.GetTimestamp();
            long transferTicks = 0;
            byte[] shared = null;
            if (TpNodeCount > 1)
            {
                shared = new byte[length];
                if (isDriver) source.CopyTo(shared);
                long r0 = Stopwatch.GetTimestamp();
                TpBroadcastBytesFromDriver(shared);
                transferTicks = Stopwatch.GetTimestamp() - r0;
            }
            ReadOnlySpan<byte> block = shared ?? source;

            EnsureNemotronTpCacheCapacity(destToken + tokenCount);

            int o = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        foreach (Tensor[] shards in new[] { _tpKvCacheK[l], _tpKvCacheV[l] })
                        {
                            int perRank = checked((int)NemotronLayerBlockBytes(shards[0], tokenCount));
                            for (int r = 0; r < TpDegree; r++)
                            {
                                int at = o + (TpRankOffset + r) * perRank;
                                if (!CopyAttentionIn(shards[r], destToken, tokenCount, block.Slice(at, perRank), out _))
                                    throw new InvalidOperationException($"Nemotron TP snapshot: layer {l} rank {r} cannot hold rows {destToken}..{destToken + tokenCount}.");
                                // The rows were written through the host copy.
                                InvalidateTensorDeviceCache(shards[r]);
                            }
                            o += perRank * GlobalTpDegree;
                        }
                        break;
                    case LayerType.Mamba2:
                        if (!withState) break;
                        if (!CopyMamba2StateIn(l, block[o..], out int read))
                            throw new InvalidOperationException($"Nemotron TP snapshot: block too short for layer {l}'s Mamba2 state.");
                        o += read;
                        break;
                }
            }
            if (o != length)
                throw new InvalidOperationException($"Nemotron TP snapshot read {o} of {length} bytes.");
            _cacheSeqLen = destToken + tokenCount;
            TraceTpSnapshot("inject", destToken, tokenCount, withState, length, Stopwatch.GetTimestamp() - t0, transferTicks);
        }
    }
}
