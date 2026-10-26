// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Host copies of Qwen3.5 / Qwen3.8 direct-CUDA engine slots.
//
// A slot that device memory can no longer hold (a retained prompt checkpoint or finished
// conversation) moves to pinned host memory and comes back when a request reuses it: copying a
// 17K-token checkpoint back costs about 60 ms, computing it again seconds. Its attention blocks are
// shared by content: every device block carries a key that changes whenever rows are written into
// it, a host copy is made once per key however many host slots hold it, and a slot coming back
// shares the device block that still carries the key instead of uploading the copy.
//
// Host memory comes in units of one attention block; a slot's recurrent states and drafter ring
// are packed into units too. Pinning is slow (about 2.5 GB/s), so a background thread keeps a
// reserve of pinned units ahead of the copies.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        /// <summary>A slot held in pinned host memory: its position, recurrent states, drafter ring
        /// and attention blocks (shared by content with other host slots).</summary>
        public sealed class HostSlot
        {
            public int NPast { get; internal set; }
            /// <summary>The slot's M-RoPE delta (see <see cref="Slot.RopeDelta"/>).</summary>
            public int RopeDelta { get; internal set; }
            internal HostBlock[] Blocks;
            internal IntPtr[] StateUnits;
        }

        internal sealed class HostBlock
        {
            internal IntPtr Data;
            internal long Key;
            internal int Refs;
        }

        // Content keys of live device blocks, and the device block holding each key.
        private readonly Dictionary<IntPtr, long> _blockKey = new Dictionary<IntPtr, long>();
        private readonly Dictionary<long, IntPtr> _deviceByKey = new Dictionary<long, IntPtr>();
        private long _nextBlockKey = 1;
        private readonly Dictionary<long, HostBlock> _hostBlocks = new Dictionary<long, HostBlock>();

        // Pinned units (one block's bytes), and the pinner that tops them up. Guarded by _hostLock.
        private readonly object _hostLock = new object();
        private readonly Stack<IntPtr> _freeUnits = new Stack<IntPtr>();
        private readonly List<IntPtr> _hostChunks = new List<IntPtr>();
        private long _pinnedBytes, _hostLimitBytes, _hostReserveBytes;
        private Thread _pinner;
        private bool _pinnerStop;
        /// <summary>Units one pinned allocation carves.</summary>
        private const int UnitsPerChunk = 16;

        // Where each recurrent-state part of a slot lands (unit, offset), and the units one slot's state takes.
        private (int Unit, long Offset)[] _stateLayout;
        private int _stateUnits;

        /// <summary>Pinned bytes the host slots hold (units kept for reuse excluded).</summary>
        public long HostBytesInUse { get; private set; }
        /// <summary>Milliseconds the engine thread spent pinning and copying to host (diagnostics).</summary>
        public double HostPinMs { get; private set; }
        public double HostCopyMs { get; private set; }

        /// <summary>Host bytes a slot's recurrent states and drafter ring take.</summary>
        public long HostStateBytes => StateLayout() * KvBlockBytes;

        /// <summary>Lets the background pinner keep <paramref name="reserveBytes"/> of pinned units
        /// free, pinning at most <paramref name="limitBytes"/> in all.</summary>
        public void ConfigureHostPool(long limitBytes, long reserveBytes)
        {
            lock (_hostLock)
            {
                _hostLimitBytes = Math.Max(0, limitBytes);
                _hostReserveBytes = Math.Max(0, Math.Min(reserveBytes, _hostLimitBytes));
                if (_hostLimitBytes > 0 && _pinner == null)
                {
                    _pinner = new Thread(PinLoop) { IsBackground = true, Name = "q35-host-pinner" };
                    _pinner.Start();
                }
                Monitor.PulseAll(_hostLock);
            }
        }

        private void PinLoop()
        {
            _alloc.Context.MakeCurrent();
            while (true)
            {
                lock (_hostLock)
                {
                    while (!_pinnerStop && (_freeUnits.Count * KvBlockBytes >= _hostReserveBytes
                        || _pinnedBytes + UnitsPerChunk * KvBlockBytes > _hostLimitBytes))
                        Monitor.Wait(_hostLock);
                    if (_pinnerStop)
                        return;
                }
                if (CudaDriverApi.cuMemHostAlloc(out IntPtr chunk, new UIntPtr((ulong)(KvBlockBytes * UnitsPerChunk)), 0) != 0)
                {
                    lock (_hostLock)
                        _hostLimitBytes = _pinnedBytes;   // the system refused: pin no more
                    continue;
                }
                lock (_hostLock)
                    AddChunk(chunk);
            }
        }

        private void AddChunk(IntPtr chunk)
        {
            _hostChunks.Add(chunk);
            _pinnedBytes += KvBlockBytes * UnitsPerChunk;
            for (int i = UnitsPerChunk - 1; i >= 0; i--)
                _freeUnits.Push(chunk + (nint)(i * KvBlockBytes));
        }

        private IntPtr TakeUnit()
        {
            lock (_hostLock)
            {
                if (_freeUnits.Count == 0)
                {
                    // The pinner fell behind: pin here.
                    long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    CudaDriverApi.cuMemHostAlloc(out IntPtr chunk, new UIntPtr((ulong)(KvBlockBytes * UnitsPerChunk)), 0).ThrowOnError();
                    HostPinMs += System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                    AddChunk(chunk);
                }
                IntPtr u = _freeUnits.Pop();
                if (_freeUnits.Count * KvBlockBytes < _hostReserveBytes)
                    Monitor.PulseAll(_hostLock);
                return u;
            }
        }

        private void ReturnUnit(IntPtr u)
        {
            lock (_hostLock)
                _freeUnits.Push(u);
        }

        private void SetBlockKey(IntPtr b, long key)
        {
            if (_blockKey.TryGetValue(b, out long old) && _deviceByKey.TryGetValue(old, out IntPtr holder) && holder == b)
                _deviceByKey.Remove(old);
            _blockKey[b] = key;
            _deviceByKey[key] = b;
        }

        private void DropBlockKey(IntPtr b)
        {
            if (_blockKey.Remove(b, out long key) && _deviceByKey.TryGetValue(key, out IntPtr holder) && holder == b)
                _deviceByKey.Remove(key);
        }

        /// <summary>Rows are about to be written into <paramref name="b"/>: host copies of its old
        /// content no longer describe it.</summary>
        private void RekeyBlock(IntPtr b) => SetBlockKey(b, _nextBlockKey++);

        /// <summary>Recurrent states and drafter ring of a slot, in the order a host slot stores them.</summary>
        private IEnumerable<(IntPtr Device, long Bytes)> StateParts(Slot s)
        {
            for (int i = 0; i < _nGdn; i++)
            {
                yield return (s.Conv[i], ConvBytes);
                yield return (s.State[i], StateBytes);
            }
            if (_d != null && s.RingPos != IntPtr.Zero)
            {
                long ring = (long)_d.NKvHead * _ringRows * _d.HeadDim * 2;
                for (int l = 0; l < _dl.Length; l++)
                {
                    yield return (s.RingK[l], ring);
                    yield return (s.RingV[l], ring);
                }
                yield return (s.RingPos, 4L * _ringRows);
            }
        }

        /// <summary>Packs the state parts into units in order; returns the units a slot's state takes.</summary>
        private int StateLayout()
        {
            if (_stateLayout != null)
                return _stateUnits;
            var layout = new List<(int, long)>();
            int unit = 0;
            long at = 0;
            foreach (var (_, bytes) in StateParts(_active))
            {
                if (bytes > KvBlockBytes)
                    throw new NotSupportedException($"[q35-cuda] a {bytes}-byte state part exceeds a host unit");
                if (at + bytes > KvBlockBytes)
                {
                    unit++;
                    at = 0;
                }
                layout.Add((unit, at));
                at += bytes;
            }
            _stateLayout = layout.ToArray();
            _stateUnits = unit + 1;
            return _stateUnits;
        }

        /// <summary>A host copy of <paramref name="s"/> (the slot itself is unchanged; free it after).</summary>
        public HostSlot CopyToHost(Slot s)
        {
            MakeCurrent();
            Sync();
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            double pinBefore = HostPinMs;
            HostSlot h = IssueCopyToHost(s, Stream);
            Sync();
            HostCopyMs += System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds - (HostPinMs - pinBefore);
            return h;
        }

        private IntPtr _copyStream;
        private readonly Stack<IntPtr> _freeEvents = new Stack<IntPtr>();

        /// <summary>Starts a host copy of <paramref name="s"/> on a copy stream, behind every kernel
        /// issued so far: the slot must not change until <paramref name="done"/> completes
        /// (<see cref="CopyDone"/>, <see cref="WaitCopy"/>), and the host slot is valid only then.</summary>
        public HostSlot BeginCopyToHost(Slot s, out IntPtr done)
        {
            MakeCurrent();
            if (_copyStream == IntPtr.Zero)
                CudaDriverApi.cuStreamCreate(out _copyStream, 1 /* CU_STREAM_NON_BLOCKING */).ThrowOnError();
            IntPtr ready = TakeEvent();
            CudaDriverApi.cuEventRecord(ready, Stream).ThrowOnError();
            CudaDriverApi.cuStreamWaitEvent(_copyStream, ready, 0).ThrowOnError();
            _freeEvents.Push(ready);   // reusable once recorded again: the wait already captured it
            HostSlot h = IssueCopyToHost(s, _copyStream);
            done = TakeEvent();
            CudaDriverApi.cuEventRecord(done, _copyStream).ThrowOnError();
            return h;
        }

        /// <summary>True once the copy <paramref name="done"/> marks has finished (the event is then released).</summary>
        public bool CopyDone(IntPtr done)
        {
            int r = CudaDriverApi.cuEventQuery(done);
            if (r == 600 /* CUDA_ERROR_NOT_READY */)
                return false;
            r.ThrowOnError();
            _freeEvents.Push(done);
            return true;
        }

        /// <summary>Waits for the copy <paramref name="done"/> marks (the event is then released).</summary>
        public void WaitCopy(IntPtr done)
        {
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            CudaDriverApi.cuEventSynchronize(done).ThrowOnError();
            HostCopyMs += System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            _freeEvents.Push(done);
        }

        private IntPtr TakeEvent()
        {
            if (_freeEvents.Count > 0)
                return _freeEvents.Pop();
            CudaDriverApi.cuEventCreate(out IntPtr e, 2 /* CU_EVENT_DISABLE_TIMING */).ThrowOnError();
            return e;
        }

        private HostSlot IssueCopyToHost(Slot s, IntPtr stream)
        {
            var h = new HostSlot { NPast = s.NPast, RopeDelta = s.RopeDelta };
            int nb = (s.NPast + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens;
            h.Blocks = new HostBlock[nb];
            for (int i = 0; i < nb; i++)
            {
                IntPtr b = s.Blocks[i];
                if (!_blockKey.TryGetValue(b, out long key))
                {
                    key = _nextBlockKey++;
                    SetBlockKey(b, key);
                }
                if (!_hostBlocks.TryGetValue(key, out HostBlock hb))
                {
                    hb = new HostBlock { Key = key, Data = TakeUnit() };
                    CudaDriverApi.cuMemcpyDtoHAsync(hb.Data, b, new UIntPtr((ulong)KvBlockBytes), stream).ThrowOnError();
                    _hostBlocks[key] = hb;
                    HostBytesInUse += KvBlockBytes;
                }
                hb.Refs++;
                h.Blocks[i] = hb;
            }
            h.StateUnits = new IntPtr[StateLayout()];
            for (int u = 0; u < h.StateUnits.Length; u++)
                h.StateUnits[u] = TakeUnit();
            int p = 0;
            foreach (var (dev, bytes) in StateParts(s))
            {
                var (unit, offset) = _stateLayout[p++];
                CudaDriverApi.cuMemcpyDtoHAsync(h.StateUnits[unit] + (nint)offset, dev, new UIntPtr((ulong)bytes), stream).ThrowOnError();
            }
            HostBytesInUse += h.StateUnits.Length * KvBlockBytes;
            return h;
        }

        /// <summary>Makes <paramref name="dst"/> the slot <paramref name="h"/> holds: blocks whose
        /// content is still on the device are shared, the others uploaded; states and ring uploaded.</summary>
        public void CopyFromHost(HostSlot h, Slot dst)
        {
            MakeCurrent();
            // Host blocks another slot's copy is still filling must land before they are read.
            if (_copyStream != IntPtr.Zero)
                CudaDriverApi.cuStreamSynchronize(_copyStream).ThrowOnError();
            ReleaseBlocks(dst);
            foreach (HostBlock hb in h.Blocks)
            {
                if (_deviceByKey.TryGetValue(hb.Key, out IntPtr b))
                    _blockRefs[b]++;
                else
                {
                    b = AllocBlock();
                    SetBlockKey(b, hb.Key);
                    CudaDriverApi.cuMemcpyHtoDAsync(b, hb.Data, new UIntPtr((ulong)KvBlockBytes), Stream).ThrowOnError();
                }
                dst.Blocks.Add(b);
            }
            dst.Version++;
            int p = 0;
            foreach (var (dev, bytes) in StateParts(dst))
            {
                var (unit, offset) = _stateLayout[p++];
                CudaDriverApi.cuMemcpyHtoDAsync(dev, h.StateUnits[unit] + (nint)offset, new UIntPtr((ulong)bytes), Stream).ThrowOnError();
            }
            dst.NPast = h.NPast;
            dst.RopeDelta = h.RopeDelta;
            Sync();
        }

        /// <summary>Releases a host slot; returns the pinned bytes that freed.</summary>
        public long FreeHostSlot(HostSlot h)
        {
            if (h == null || h.StateUnits == null)
                return 0;
            long freed = 0;
            foreach (HostBlock hb in h.Blocks)
            {
                if (--hb.Refs > 0)
                    continue;
                _hostBlocks.Remove(hb.Key);
                ReturnUnit(hb.Data);
                freed += KvBlockBytes;
            }
            foreach (IntPtr u in h.StateUnits)
                ReturnUnit(u);
            freed += h.StateUnits.Length * KvBlockBytes;
            h.StateUnits = null;
            h.Blocks = Array.Empty<HostBlock>();
            HostBytesInUse -= freed;
            return freed;
        }

        /// <summary>The content key of the last full block below <paramref name="tokens"/> (0 when there is none):
        /// two slots with the same key there share every row before it, so a longer slot with a slot's
        /// key at that index continues it.</summary>
        public long PrefixKey(Slot s, int tokens)
        {
            int i = tokens / Q35Kernels.KvBlockTokens - 1;
            return i >= 0 && i < s.Blocks.Count && _blockKey.TryGetValue(s.Blocks[i], out long key) ? key : 0;
        }

        /// <inheritdoc cref="PrefixKey(Slot, int)"/>
        public static long PrefixKey(HostSlot h, int tokens)
        {
            int i = tokens / Q35Kernels.KvBlockTokens - 1;
            return i >= 0 && i < h.Blocks.Length ? h.Blocks[i].Key : 0;
        }

        /// <summary>Host bytes freeing <paramref name="h"/> would release now (its state, and the blocks
        /// no other host slot holds).</summary>
        public long HostReleasableBytes(HostSlot h)
        {
            if (h?.StateUnits == null)
                return 0;
            long bytes = h.StateUnits.Length * KvBlockBytes;
            foreach (HostBlock hb in h.Blocks)
                if (hb.Refs == 1)
                    bytes += KvBlockBytes;
            return bytes;
        }

        /// <summary>Host bytes copying <paramref name="s"/> would add (blocks the host already holds
        /// by content are not counted).</summary>
        public long HostBytesToCopy(Slot s)
        {
            long bytes = HostStateBytes;
            int nb = (s.NPast + Q35Kernels.KvBlockTokens - 1) / Q35Kernels.KvBlockTokens;
            for (int i = 0; i < nb; i++)
                if (!_blockKey.TryGetValue(s.Blocks[i], out long key) || !_hostBlocks.ContainsKey(key))
                    bytes += KvBlockBytes;
            return bytes;
        }

        private void FreeHostPool()
        {
            if (_copyStream != IntPtr.Zero)
            {
                CudaDriverApi.cuStreamSynchronize(_copyStream);
                CudaDriverApi.cuStreamDestroy(_copyStream);
                _copyStream = IntPtr.Zero;
            }
            while (_freeEvents.Count > 0)
                CudaDriverApi.cuEventDestroy(_freeEvents.Pop());
            Thread pinner;
            lock (_hostLock)
            {
                _pinnerStop = true;
                Monitor.PulseAll(_hostLock);
                pinner = _pinner;
            }
            pinner?.Join();
            lock (_hostLock)
            {
                foreach (IntPtr p in _hostChunks)
                    CudaDriverApi.cuMemFreeHost(p);
                _hostChunks.Clear();
                _freeUnits.Clear();
                _pinnedBytes = 0;
            }
            _hostBlocks.Clear();
            HostBytesInUse = 0;
        }
    }
}
