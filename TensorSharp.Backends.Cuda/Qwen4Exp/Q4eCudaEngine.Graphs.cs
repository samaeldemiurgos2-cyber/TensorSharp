// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Captured single-sequence decode steps. One decode step issues about 30 launches per layer; on
// the A40 the host needs longer to queue them than the GPU needs to run them, so the step ran at
// the launch rate. A captured step is one graph launch per device.
//
// Each device captures its run of layers once per (sequence, regime) and replays it. What changes
// from step to step is read at launch time, never baked in:
//   - the token and the position come from pinned host memory through the graph's first copies,
//     and every kernel that depends on the position reads the device copy (the KV and indexer
//     cache writes, RoPE, the block pooling, the scores and the selection, the attention span);
//   - the PLE rows are gathered on the host into the pinned staging the graph copies from;
//   - the streams cross a layer-split boundary through the pinned boundary buffer, ordered by an
//     event wait queued between the graph launches.
// The regime is the set of QSA layers whose selection runs at this position (the sparse path is
// only issued past the indexer's width): the sets are nested as the position grows, so their size
// names them.
//
// Every buffer a captured step touches belongs to the engine or the sequence, so graphs stay valid
// until their sequence is freed (CudaDecodeCapture refuses a step that allocates or reads shared
// scratch). A capture that fails for any reason turns graphs off for the engine with a note and
// the step runs uncaptured.
using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly bool _graphsEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_GRAPHS"), "0", StringComparison.Ordinal);
        private bool _graphsFailed;
        /// <summary>Per (sequence, regime): one executable graph per device holding layers.</summary>
        private readonly Dictionary<(int Slot, int Regime), IntPtr[]> _graphs = new Dictionary<(int, int), IntPtr[]>();
        /// <summary>The step's position, read by every device's graph (PORTABLE pinned memory).</summary>
        private IntPtr _posPinned;

        /// <summary>Decode steps may be captured: not while stages are timed or the sparse selection or
        /// the router is probed (they read the device mid-step).</summary>
        private bool CanCaptureDecode => _graphsEnabled && !_graphsFailed && _perf < 2 && SparseProbe == null
            && RouterProbe == null;

        /// <summary>Captured decode steps this engine holds (tests, diagnostics).</summary>
        internal int CapturedDecodeGraphs => _graphs.Count;

        /// <summary>QSA layers whose selection runs for a row at <paramref name="p0"/>.</summary>
        private int DecodeRegime(int p0)
        {
            int n = 0;
            foreach (var L in _layers)
                if (L.Ratio > 0 && p0 + 1 > _m.IdxTopK + L.Ratio - 1)
                    n++;
            return n;
        }

        /// <summary>One decode step of <paramref name="slot"/> through its captured graphs, capturing
        /// them on first use. False when graphs are off, having just failed: the caller runs the step
        /// uncaptured.</summary>
        private bool ForwardDecodeGraphed(Slot slot, int token, int p0, float[] logitsOut)
        {
            // The pinned inputs are rewritten below: every device's previous work must have run.
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            *(int*)_pinnedTokens = token;
            *(int*)_posPinned = p0;
            if (_pleLayer >= 0)
                GatherPleRows(slot, new[] { token }, 0, 1, p0, null, null);

            var key = (slot.Id, DecodeRegime(p0));
            if (!_graphs.TryGetValue(key, out IntPtr[] execs))
            {
                try
                {
                    execs = CaptureDecode(slot, p0);
                }
                catch (Exception ex)
                {
                    _graphsFailed = true;
                    Console.Error.WriteLine($"[q4e-cuda] decode steps run uncaptured from here on: the capture failed ({ex.Message})");
                    return false;
                }
                _graphs[key] = execs;
            }

            for (int d = 0; d <= _lastDev; d++)
            {
                var dev = _devs[d];
                dev.MakeCurrent();
                if (d > 0)
                    CudaDriverApi.cuStreamWaitEvent(dev.Stream, _devs[d - 1].XsReadyEv, 0).ThrowOnError();
                CudaDriverApi.cuGraphLaunch(execs[d], dev.Stream).ThrowOnError();
                if (d < _lastDev)
                    CudaDriverApi.cuEventRecord(dev.XsReadyEv, dev.Stream).ThrowOnError();
            }
            if (_perf > 0)
                _issueEnd = System.Diagnostics.Stopwatch.GetTimestamp();
            FinishHead(1, logitsOut, false, null);
            return true;
        }

        private IntPtr[] CaptureDecode(Slot slot, int p0)
        {
            var execs = new IntPtr[_lastDev + 1];
            try
            {
                for (int d = 0; d <= _lastDev; d++)
                {
                    var dev = _devs[d];
                    dev.MakeCurrent();
                    execs[d] = CudaDecodeCapture.Capture(dev.Alloc, () => IssueDecodeStep(dev, slot, p0));
                }
                return execs;
            }
            catch
            {
                DestroyGraphs(execs);
                throw;
            }
        }

        /// <summary>One device's part of a decode step, as captured: the position (and on device 0 the
        /// token) from pinned memory, the streams from the previous device's boundary buffer, its
        /// layers, then the boundary copy or, on the last device, the head up to the pinned logits.</summary>
        private void IssueDecodeStep(Dev dev, Slot slot, int p0)
        {
            var m = _m;
            var boundary = new UIntPtr((ulong)HcDim * 4);
            CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.PosDev), _posPinned, new UIntPtr(4), dev.Stream).ThrowOnError();
            if (dev.Ordinal == 0)
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Tokens), _pinnedTokens, new UIntPtr(4), dev.Stream).ThrowOnError();
                dev.DK.Embed(_tokEmbd.Ptr, dev.Tokens, dev.Xs, _tokEmbd.Type, _tokEmbd.RowBytes, 1, m.NEmbd, dev.Stream);
            }
            else
            {
                CudaDriverApi.cuMemcpyHtoDAsync(Ptr(dev.Xs), _devs[dev.Ordinal - 1].BoundaryPinned, boundary, dev.Stream).ThrowOnError();
            }
            for (int il = 0; il < m.NLayer; il++)
                if (_layers[il].Device == dev.Ordinal)
                    RunLayer(dev, il, slot, 1, p0, null, null, Ptr(dev.PosDev));
            if (dev.Ordinal < _lastDev)
                CudaDriverApi.cuMemcpyDtoHAsync(dev.BoundaryPinned, Ptr(dev.Xs), boundary, dev.Stream).ThrowOnError();
            else
                IssueHead(1, logits: true, allRows: false, hidden: false);
        }

        private void DestroySlotGraphs(int slotId)
        {
            List<(int, int)> keys = null;
            foreach (var key in _graphs.Keys)
                if (key.Slot == slotId)
                    (keys ??= new List<(int, int)>()).Add(key);
            if (keys == null)
                return;
            foreach (var key in keys)
            {
                DestroyGraphs(_graphs[key]);
                _graphs.Remove(key);
            }
        }

        private void DestroyGraphs(IntPtr[] execs)
        {
            for (int d = 0; d < execs.Length; d++)
            {
                if (execs[d] == IntPtr.Zero)
                    continue;
                _devs[d].MakeCurrent();
                CudaDriverApi.cuGraphExecDestroy(execs[d]);
                execs[d] = IntPtr.Zero;
            }
        }
    }
}
