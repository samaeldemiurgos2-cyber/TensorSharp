// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Cross-node layer split: this engine as one pipeline stage of a distributed
// run (see IDsv4StageLink.cs for the contract and the wire format).
//
// A stage loads only its own layers (the embedding when it owns layer 0, the
// head when it owns the last layer) and keeps the others' topology so it can
// name what crosses. Its first local device receives the boundary payload of
// each ubatch before any of its layers run; its last local device sends one
// after its last layer. Inside the node, local devices hand over exactly as a
// single-process split does. The payload goes through pinned staging: the
// sender copies every section device-to-host on its stream, waits for the
// copies and sends; the receiver validates the header and the committed ranges
// first, then queues the host-to-device copies ahead of its layers on the same
// stream. Logits never cross: the driver runs the last stage.
// ---------------------------------------------------------------------------
using System;
using System.Diagnostics;
using System.Linq;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Dsv4CudaEngine
    {
        // Null on a single-process engine, which owns every layer.
        private readonly Dsv4StagePlan _stage;
        // Every layer's global device (a single-process engine's own assignment), this node's first
        // global device, and the owned layer range [_firstLayer, _endLayer).
        private int[] _globalDev;
        private int _firstGlobal;
        private int _firstLayer, _endLayer;
        private bool _ownsEngram;

        // Network staging: one pinned body per direction and the header words. A receiving stage
        // records _netRecvCopied after the copies out of its staging and waits for it before the
        // next ubatch's payload overwrites the staging.
        private IntPtr _netSendPinned, _netRecvPinned, _netRecvCopied;
        private long _netSendBytes, _netRecvBytes;
        private readonly int[] _netHeaderSend = new int[Dsv4BoundaryLayout.HeaderWords];
        private readonly int[] _netHeaderRecv = new int[Dsv4BoundaryLayout.HeaderWords];
        private long _sendSequence, _recvSequence;
        private Dsv4CommittedRange[] _sendRanges;

        /// <summary>Whether this engine is one stage of a cross-node layer split.</summary>
        public bool IsDistributedStage => _stage != null;

        /// <summary>Owned layers [first, end): every layer on a single-process engine.</summary>
        internal (int First, int End) OwnedLayers => (_firstLayer, _endLayer);

        /// <summary>Each layer's local device (-1 for another stage's layer), for tests and messages.</summary>
        internal int[] LayerDevices => _layers == null ? Array.Empty<int>() : Array.ConvertAll(_layers, l => l.Device);

        private bool OwnsEmbedding => _firstLayer == 0;

        private bool OwnsHead => _endLayer == _m.NLayer;

        /// <summary>What this engine's network boundaries carried.</summary>
        internal Dsv4BoundaryStats BoundaryStats { get; } = new Dsv4BoundaryStats();

        /// <summary>How many ubatches ran each layer, and each Engram table.</summary>
        internal long[] LayerRuns { get; private set; } = Array.Empty<long>();

        internal long[] EngramRuns { get; private set; } = Array.Empty<long>();

        /// <summary>Tests only: called on a sending stage, with its stage index and boundary sequence number,
        /// just before the payload goes out (a test throws here to fail the stage mid-forward).</summary>
        internal static Action<int, long> BeforeBoundarySend { get; set; }

        /// <summary>Tests only: may rewrite the header words or the body of an outgoing payload (stage,
        /// header, body, body bytes), to prove a receiver refuses a payload that does not match.</summary>
        internal static BoundaryTamper TamperBoundary { get; set; }

        internal delegate void BoundaryTamper(int stage, int[] header, IntPtr body, int bodyBytes);

        /// <summary>Tests only: called on a receiving stage with its stage index, the boundary sequence number and
        /// the receiving device's stream once a payload sits in the pinned staging, before the copies out of it are
        /// queued (a test stalls the stream here so the device falls behind the host).</summary>
        internal static Action<int, long, IntPtr> BeforeBoundaryCopies { get; set; }

        /// <summary>Tests only: the free VRAM a stage's local device reports to the fit check, (stage, local device)
        /// to bytes; a negative value reads the device.</summary>
        internal static Func<int, int, long> FreeVramForTests { get; set; }

        /// <summary>What the fit check priced on each local device (a stage only; null on a single-process engine).</summary>
        internal Dsv4DeviceNeed[] StageNeeds { get; private set; }

        /// <summary>The routed-expert type check of the layers this engine owns (all of them, unless it is a stage).</summary>
        private static Func<int, bool> OwnedBy(Dsv4StagePlan stage)
        {
            if (stage == null)
                return null;
            int first = stage.FirstLocalGlobalDevice, end = first + stage.LocalDevices;
            int[] g = stage.GlobalDeviceOfLayer;
            return il => g[il] >= first && g[il] < end;
        }

        /// <summary>What a stage cannot be, refused before any device is touched.</summary>
        private static void ValidateStage(ModelDesc m, Dsv4StagePlan stage)
        {
            if (!m.V41)
                throw new NotSupportedException("[dsv4-cuda] cross-node layer split is implemented for DeepSeek V4.1 only.");
            if (m.Dspark != null)
                throw new NotSupportedException("[dsv4-cuda] cross-node layer split does not mirror DSpark operations; omit --draft-model.");
            if (m.HeadDim != 512)
                throw new NotSupportedException($"DSV4 CUDA engine requires head_dim 512, got {m.HeadDim}.");
            stage.Validate(m.NLayer);
            if (stage.GlobalDeviceOfLayer[m.NLayer - 1] != stage.StageCount * stage.LocalDevices - 1)
                throw new ArgumentException("[dsv4-cuda] the last stage's last device must own the last layer and the head.");
            Dsv4StagePlacement.ValidateWireLimits(Dsv4Topology.From(m), stage.GlobalDeviceOfLayer, stage.LocalDevices);
        }

        /// <summary>
        /// A stage's placement: the global assignment it was handed, its own layers on its local devices
        /// (-1 elsewhere in <paramref name="assignment"/>), and no routed-expert offload. Engram tables go
        /// to VRAM only when TS_DSV41_ENGRAM_DEVICE=1 asked for it, so the choice is the same on every node.
        /// </summary>
        private long[] PlaceStage(ModelDesc m, int useDevs, int[] assignment)
        {
            Dsv4StagePlan st = _stage;
            _globalDev = (int[])st.GlobalDeviceOfLayer.Clone();
            _firstGlobal = st.FirstLocalGlobalDevice;
            _firstLayer = -1;
            _endLayer = -1;
            for (int il = 0; il < m.NLayer; il++)
            {
                int d = _globalDev[il] - _firstGlobal;
                bool owned = d >= 0 && d < useDevs;
                assignment[il] = owned ? d : -1;
                if (owned)
                {
                    if (_firstLayer < 0)
                        _firstLayer = il;
                    _endLayer = il + 1;
                }
            }
            if (_firstLayer < 0)
                throw new InvalidOperationException($"[dsv4-cuda] stage {st.Stage} owns no layer.");
            _nCpuMoe = 0;
            bool hasTables = false;
            for (int il = _firstLayer; il < _endLayer; il++)
            {
                hasTables |= m.Layers[il].EngramTable.IsValid;
                _ownsEngram |= m.Layers[il].EngramIndex >= 0;
            }
            EngramResident = hasTables && m.EngramOnDevice == true;
            var bytes = new long[m.NLayer];
            for (int il = _firstLayer; il < _endLayer; il++)
                foreach (var qw in EnumerateQuantWeights(m.Layers[il]))
                    bytes[il] += qw.TotalBytes;
            return bytes;
        }

        /// <summary>
        /// The fit check that replaces the free-VRAM split on a stage: each local device must hold its
        /// layers' weights and caches, the replicas of the groups whose sources lie before it, the embedding
        /// or head it owns and the working set, with the reserve left over. The split itself never moves to
        /// make room (it would desynchronize the nodes), so a device that cannot hold its part is refused,
        /// naming what it needs. Every node then learns whether every stage fits, before any weight is
        /// uploaded: the stage that does not fit refuses with what it needs, and its peers refuse with it
        /// instead of streaming their whole share of the weights and failing later at the first barrier.
        /// </summary>
        private void CheckStageFits(ModelDesc m, int useDevs, int[] assignment)
        {
            long reserve = EnvInt("TS_DSV4_VRAM_RESERVE_MB", 2048) * 1024L * 1024L;
            var topology = Dsv4Topology.From(m);
            var needs = new Dsv4DeviceNeed[useDevs];
            string shortfall = null;
            for (int d = 0; d < useDevs; d++)
            {
                long weights = 0, caches = 0;
                int first = -1, last = -1;
                for (int il = 0; il < m.NLayer; il++)
                {
                    if (assignment[il] != d)
                        continue;
                    if (first < 0)
                        first = il;
                    last = il;
                    foreach (var qw in EnumerateQuantWeights(m.Layers[il]))
                        weights += Align(qw.TotalBytes);
                    caches += LayerCacheBytes(m, il);
                }
                int g = _firstGlobal + d;
                long replicas = Dsv4StagePlacement.DeviceReplicaBytes(topology, _globalDev, g);
                long ends = 0;
                if (d == 0 && OwnsEmbedding)
                    ends += Align(m.TokEmbd.TotalBytes);
                if (d == useDevs - 1 && OwnsHead)
                    ends += Align(m.Output.TotalBytes);
                var need = new Dsv4DeviceNeed(weights, caches, replicas, ends, PerDeviceFixedBytes(m));
                needs[d] = need;
                long freeBytes = FreeVramForTests?.Invoke(_stage.Stage, d) ?? -1;
                if (freeBytes < 0)
                {
                    _devs[d].MakeCurrent();
                    CudaDriverApi.cuMemGetInfo(out UIntPtr free, out UIntPtr _).ThrowOnError();
                    freeBytes = (long)free.ToUInt64();
                }
                if (shortfall == null && need.Total + reserve > freeBytes)
                    shortfall =
                        $"[dsv4-cuda] stage {_stage.Stage}, local device {d} (global device {g}, layers {first}..{last}) needs " +
                        $"{need.Total / (double)(1L << 30):F2} GiB ({replicas / (double)(1L << 20):F0} MiB of it cache replicas) plus the " +
                        $"{reserve / (double)(1L << 30):F1} GiB reserve, but has {freeBytes / (double)(1L << 30):F2} GiB free. The " +
                        "cross-node split comes from the checkpoint's bytes so that every node agrees, and never moves layers to " +
                        "fit one node: free VRAM, shorten the context (MAX_CONTEXT), lower TS_DSV4_VRAM_RESERVE_MB, or add nodes or GPUs.";
            }
            StageNeeds = needs;

            // A vote, not a message: the fitting nodes cannot know which peer is short or by how much, only that
            // one is, and that node names it. Every earlier identity was agreed, so a disagreement here is the
            // fit verdict and nothing else.
            InvalidOperationException vote = null;
            try { _stage.Agree?.Invoke("dsv4-cuda-fit|" + (shortfall == null ? "ok" : "short")); }
            catch (InvalidOperationException e) { vote = e; }
            if (shortfall != null)
                throw new InvalidOperationException(shortfall, vote);
            if (vote != null)
                throw new InvalidOperationException(
                    $"[dsv4-cuda] stage {_stage.Stage} fits its GPUs, but another node's stage does not (that node names what it " +
                    "needs), so every node refuses before uploading any weight. Restart all nodes once it has room.", vote);
        }

        /// <summary>The pinned staging and the event of the network boundaries this stage takes part in.</summary>
        private void AllocateNetworkStaging(int useDevs)
        {
            Dev dev0 = _devs[0];
            dev0.MakeCurrent();
            // ValidateStage refused a placement whose payloads exceed a transport frame before any device was
            // touched; the staging and the header's 32-bit body count rely on it.
            if (_stage.Stage > 0)
            {
                _netRecvBytes = Math.Max(4, _layouts[_firstGlobal].MaxBodyBytes());
                if (_netRecvBytes > Dsv4BoundaryLayout.MaxWireBodyBytes)
                    throw new InvalidOperationException($"[dsv4-cuda] a {_netRecvBytes}-byte boundary payload exceeds one transport frame");
                CudaDriverApi.cuMemHostAlloc(out _netRecvPinned, new UIntPtr((ulong)_netRecvBytes), 0x1 /*PORTABLE*/).ThrowOnError();
                CudaDriverApi.cuEventCreate(out _netRecvCopied, 0x02 /*DISABLE_TIMING*/).ThrowOnError();
            }
            if (_stage.Stage < _stage.StageCount - 1)
            {
                Dsv4BoundaryLayout send = _layouts[_firstGlobal + useDevs];
                _netSendBytes = Math.Max(4, send.MaxBodyBytes());
                if (_netSendBytes > Dsv4BoundaryLayout.MaxWireBodyBytes)
                    throw new InvalidOperationException($"[dsv4-cuda] a {_netSendBytes}-byte boundary payload exceeds one transport frame");
                CudaDriverApi.cuMemHostAlloc(out _netSendPinned, new UIntPtr((ulong)_netSendBytes), 0x1 /*PORTABLE*/).ThrowOnError();
                _sendRanges = new Dsv4CommittedRange[send.KvGroups.Length];
            }
        }

        private void FreeNetworkStaging()
        {
            if (_netRecvCopied != IntPtr.Zero)
                CudaDriverApi.cuEventDestroy(_netRecvCopied);
            if (_netRecvPinned != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_netRecvPinned);
            if (_netSendPinned != IntPtr.Zero)
                CudaDriverApi.cuMemFreeHost(_netSendPinned);
            _netRecvCopied = _netRecvPinned = _netSendPinned = IntPtr.Zero;
        }

        /// <summary>
        /// The previous stage's boundary state for the ubatch at [p0, p0 + nt), onto <paramref name="dst"/> (the
        /// first local device): header, then body, each validated before anything is queued on the device.
        /// </summary>
        private void ReceiveBoundary(Dev dst, int p0, int nt)
        {
            long start = Stopwatch.GetTimestamp();
            Dsv4BoundaryLayout layout = _layouts[_firstGlobal];
            dst.MakeCurrent();
            // The device may still be copying the previous payload out of the staging (its stream runs behind this
            // host whenever nothing in between synchronized it): wait for those copies before this one lands.
            CudaDriverApi.cuEventSynchronize(_netRecvCopied).ThrowOnError();
            long sequence = _recvSequence++;
            fixed (int* h = _netHeaderRecv)
                _stage.Link.ReceivePrevious((IntPtr)h, Dsv4BoundaryLayout.HeaderBytes);
            var (active, bodyBytes) = layout.ValidateHeader(_netHeaderRecv, sequence, p0, nt);
            if (bodyBytes > _netRecvBytes)
                throw new InvalidOperationException($"[dsv4-cuda] boundary body of {bodyBytes} bytes exceeds the {_netRecvBytes}-byte staging");
            _stage.Link.ReceivePrevious(_netRecvPinned, bodyBytes);
            int expected = layout.ExpectedRanges(p0, nt).Length;
            Dsv4CommittedRange[] ranges = layout.ValidateRanges(new ReadOnlySpan<int>((void*)_netRecvPinned, 3 * expected), p0, nt);
            int width = layout.CarriesMask(active) ? layout.MaskWidth(p0, nt) : 0;
            Dsv4Section[] sections = layout.Sections(nt, ranges, active, width);
            BeforeBoundaryCopies?.Invoke(_stage.Stage, sequence, dst.Stream);

            int hd = _m.HeadDim, id = _m.IdxHeadSize;
            foreach (Dsv4Section s in sections)
            {
                IntPtr from = (IntPtr)((long)_netRecvPinned + s.Offset);
                IntPtr to;
                switch (s.Kind)
                {
                    case Dsv4SectionKind.Ranges:
                        continue;
                    case Dsv4SectionKind.Xs:
                        to = Ptr(dst.Xs);
                        break;
                    case Dsv4SectionKind.Gates:
                        to = Ptr(dst.PreFfn);
                        break;
                    case Dsv4SectionKind.CompK:
                        to = At(CompKOn(_active, ranges[s.Range].Source, dst), (long)ranges[s.Range].FirstBlock * hd * 2);
                        break;
                    case Dsv4SectionKind.LidK:
                        to = At(LidKOn(_active, ranges[s.Range].Source, dst), (long)ranges[s.Range].FirstBlock * id * 2);
                        break;
                    case Dsv4SectionKind.TopkIdx:
                        to = Ptr(dst.TopkIdx);
                        break;
                    case Dsv4SectionKind.TopkCnt:
                        to = Ptr(dst.TopkCnt);
                        break;
                    case Dsv4SectionKind.Mask:
                    {
                        var copy = new CudaDriverApi.CudaMemcpy2D
                        {
                            SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_HOST,
                            SrcHost = from,
                            SrcPitch = (UIntPtr)(ulong)s.Width,
                            DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE,
                            DstDevice = Ptr(dst.CandMask),
                            DstPitch = (UIntPtr)(ulong)dst.CandMask.Sizes[1],
                            WidthInBytes = (UIntPtr)(ulong)s.Width,
                            Height = (UIntPtr)(ulong)s.Rows,
                        };
                        CudaDriverApi.cuMemcpy2DAsync(copy, dst.Stream).ThrowOnError();
                        continue;
                    }
                    default:
                        throw new InvalidOperationException($"[dsv4-cuda] unknown boundary section {s.Kind}");
                }
                CudaDriverApi.cuMemcpyHtoDAsync(to, from, new UIntPtr((ulong)s.Bytes), dst.Stream).ThrowOnError();
            }
            CudaDriverApi.cuEventRecord(_netRecvCopied, dst.Stream).ThrowOnError();

            // The rows now sit in this node's replicas: a local boundary or the next node's payload hands
            // them on as if this node's own source had committed them.
            foreach (Dsv4CommittedRange r in ranges)
            {
                if (r.Count > 0 && Replicated(r.Source))
                    _committed.Add((r.Source, _active, r.FirstBlock, r.Count));
            }
            _v41CandActive = active;

            Dsv4BoundaryStats st = BoundaryStats;
            st.Received++;
            st.BytesReceived += Dsv4BoundaryLayout.HeaderBytes + bodyBytes;
            foreach (Dsv4CommittedRange r in ranges)
            {
                st.CommittedRowsReceived += r.Count;
                st.SourcesReceived.Add(r.Source);
            }
            if (layout.Selection)
                st.SelectionsReceived++;
            if (width > 0)
            {
                st.MasksReceived++;
                st.MaskBytesReceived += (long)nt * width;
            }
            st.ReceiveTicks += Stopwatch.GetTimestamp() - start;
        }

        /// <summary>
        /// This stage's boundary state for the ubatch at [p0, p0 + nt), from <paramref name="src"/> (the last local
        /// device) to the next stage. The device-to-host copies run on the source stream, which the send then waits for.
        /// </summary>
        private void SendBoundary(Dev src, int p0, int nt)
        {
            long start = Stopwatch.GetTimestamp();
            Dsv4BoundaryLayout layout = _layouts[_firstGlobal + _devs.Length];
            int n = 0;
            foreach (int source in layout.KvGroups)
            {
                foreach (var entry in _committed)
                {
                    if (entry.Source != source || entry.Slot != _active)
                        continue;
                    if (n == _sendRanges.Length)
                        throw new InvalidOperationException($"[dsv4-cuda] more committed ranges than boundary groups at source {source}");
                    _sendRanges[n++] = new Dsv4CommittedRange(entry.Source, entry.FirstBlock, entry.Count);
                }
            }
            var ranges = new ReadOnlySpan<Dsv4CommittedRange>(_sendRanges, 0, n);
            bool active = _v41CandActive;
            int width = layout.CarriesMask(active) ? layout.MaskWidth(p0, nt) : 0;
            Dsv4Section[] sections = layout.Sections(nt, ranges, active, width);
            long body = Dsv4BoundaryLayout.BodyBytes(sections);
            if (body > _netSendBytes)
                throw new InvalidOperationException($"[dsv4-cuda] boundary body of {body} bytes exceeds the {_netSendBytes}-byte staging");

            int hd = _m.HeadDim, id = _m.IdxHeadSize;
            src.MakeCurrent();
            foreach (Dsv4Section s in sections)
            {
                IntPtr to = (IntPtr)((long)_netSendPinned + s.Offset);
                IntPtr from;
                switch (s.Kind)
                {
                    case Dsv4SectionKind.Ranges:
                        Dsv4BoundaryLayout.WriteRanges(new Span<int>((void*)to, 3 * n), ranges);
                        continue;
                    case Dsv4SectionKind.Xs:
                        from = Ptr(src.Xs);
                        break;
                    case Dsv4SectionKind.Gates:
                        from = Ptr(src.PreFfn);
                        break;
                    case Dsv4SectionKind.CompK:
                        from = At(CompKOn(_active, ranges[s.Range].Source, src), (long)ranges[s.Range].FirstBlock * hd * 2);
                        break;
                    case Dsv4SectionKind.LidK:
                        from = At(LidKOn(_active, ranges[s.Range].Source, src), (long)ranges[s.Range].FirstBlock * id * 2);
                        break;
                    case Dsv4SectionKind.TopkIdx:
                        from = Ptr(src.TopkIdx);
                        break;
                    case Dsv4SectionKind.TopkCnt:
                        from = Ptr(src.TopkCnt);
                        break;
                    case Dsv4SectionKind.Mask:
                    {
                        var copy = new CudaDriverApi.CudaMemcpy2D
                        {
                            SrcMemoryType = CudaDriverApi.CU_MEMORYTYPE_DEVICE,
                            SrcDevice = Ptr(src.CandMask),
                            SrcPitch = (UIntPtr)(ulong)src.CandMask.Sizes[1],
                            DstMemoryType = CudaDriverApi.CU_MEMORYTYPE_HOST,
                            DstHost = to,
                            DstPitch = (UIntPtr)(ulong)s.Width,
                            WidthInBytes = (UIntPtr)(ulong)s.Width,
                            Height = (UIntPtr)(ulong)s.Rows,
                        };
                        CudaDriverApi.cuMemcpy2DAsync(copy, src.Stream).ThrowOnError();
                        continue;
                    }
                    default:
                        throw new InvalidOperationException($"[dsv4-cuda] unknown boundary section {s.Kind}");
                }
                CudaDriverApi.cuMemcpyDtoHAsync(to, from, new UIntPtr((ulong)s.Bytes), src.Stream).ThrowOnError();
            }
            StageEnd(src, 11);
            CudaDriverApi.cuStreamSynchronize(src.Stream).ThrowOnError();

            long sequence = _sendSequence++;
            layout.WriteHeader(_netHeaderSend, sequence, p0, nt, active, n, width, body);
            TamperBoundary?.Invoke(_stage.Stage, _netHeaderSend, _netSendPinned, (int)body);
            BeforeBoundarySend?.Invoke(_stage.Stage, sequence);
            fixed (int* h = _netHeaderSend)
                _stage.Link.SendNext((IntPtr)h, Dsv4BoundaryLayout.HeaderBytes);
            _stage.Link.SendNext(_netSendPinned, (int)body);

            Dsv4BoundaryStats st = BoundaryStats;
            st.Sent++;
            st.BytesSent += Dsv4BoundaryLayout.HeaderBytes + body;
            for (int i = 0; i < n; i++)
            {
                st.CommittedRowsSent += ranges[i].Count;
                st.SourcesSent.Add(ranges[i].Source);
            }
            if (layout.Selection)
                st.SelectionsSent++;
            if (width > 0)
            {
                st.MasksSent++;
                st.MaskBytesSent += (long)nt * width;
            }
            st.SendTicks += Stopwatch.GetTimestamp() - start;
        }

        /// <summary>TS_DSV4_PERF: what the network boundaries of the last forward cost.</summary>
        private string DescribeBoundaries()
        {
            Dsv4BoundaryStats st = BoundaryStats;
            double f = 1000.0 / Stopwatch.Frequency;
            return $"stage {_stage.Stage}/{_stage.StageCount}: sent {st.Sent} ({st.BytesSent} B, {st.SendTicks * f:F1} ms), " +
                   $"received {st.Received} ({st.BytesReceived} B, {st.ReceiveTicks * f:F1} ms) in total";
        }

        /// <summary>Owned layer ids, for messages.</summary>
        internal string DescribeOwnedLayers()
            => _stage == null ? $"0..{_m.NLayer - 1}"
                : $"{_firstLayer}..{_endLayer - 1} (stage {_stage.Stage}/{_stage.StageCount}, global devices " +
                  $"{_firstGlobal}..{_firstGlobal + _devs.Length - 1}{(OwnsEmbedding ? ", embedding" : "")}{(OwnsHead ? ", head" : "")})";

        /// <summary>The Engram table count the run counters are sized for.</summary>
        private static int EngramTableCount(ModelDesc m)
            => m.Layers.Length == 0 ? 0 : m.Layers.Max(l => l.EngramIndex) + 1;
    }

    /// <summary>One local device's share of a distributed stage as the fit check prices it: its layers' weights
    /// (256-byte aligned) and caches, the replicas of earlier groups it reads, the embedding or head it owns, and
    /// the per-device working set.</summary>
    internal readonly record struct Dsv4DeviceNeed(long Weights, long Caches, long Replicas, long EmbeddingAndHead, long Fixed)
    {
        public long Total => Weights + Caches + Replicas + EmbeddingAndHead + Fixed;
    }
}
