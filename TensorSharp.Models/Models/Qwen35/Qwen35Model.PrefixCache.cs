// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen 3.5 / 3.6's radix prefix cache: complete attention and GatedDeltaNet
// holders, copied or donated at exact lengths. Released holders reuse the pool;
// device-authoritative state is settled before copying. Backend and tensor-
// parallel capabilities still determine which operations are available.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class Qwen35Model : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
    {
        // TS_PREFIX_RELEASE_DEBUG=1: log released payloads and holders that could not be settled.
        private static readonly bool PrefixReleaseDebug =
            Environment.GetEnvironmentVariable("TS_PREFIX_RELEASE_DEBUG") == "1";

        // Set by AttachPrefixCache: the prefix cache owns this model's reuse state.
        private IPrefixPayloadSink _prefixCacheSink;
        // >0 while a batched release has already reset the decode graphs.
        private int _holderGraphResetSuppressed;

        // ---------------------------------------------------------------- capabilities

        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        {
            bool holders = SupportsPerSequenceFusedForward;
            return new PrefixCacheCapabilities
            {
                Class = FamilyClass.R,
                // Backend-qualified on MLX (see CheckpointFingerprint): the checkpoint files
                // are named from it, and an MLX run must not replace a ggml checkpoint of the
                // same prompt that it cannot read, or the other way round.
                NamespaceFingerprint = CheckpointFingerprint,
                EndState = holders ? EndStateSupport.CopyAndDonate : EndStateSupport.None,
                // False under tensor parallelism (the cache lives on the ranks).
                CanCaptureCopy = holders && SupportsPrefixCheckpoints,
                AdoptPrimaryOnDisplacement = holders,
                DeferPrimaryConversion = holders,
                PrimaryResident = true,
                MinRetainTokens = 32,
                Truncation = TruncationKind.None,
                TruncationGranularity = Math.Max(1, KVCacheTruncationGranularity),
                Pages = PageSupport.None,
                // DEC-11: false until M8 lands a rope delta and flips it (Phase 0's value).
                ReuseAcrossMediaSpan = SupportsReuseAcrossMediaSpan,
                Persistable = holders && SupportsRetainedCacheSerialization,
            };
        }

        public void AttachPrefixCache(IPrefixPayloadSink sink)
            => _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));

        public void DetachPrefixCache() => _prefixCacheSink = null;

        /// <summary>A holder's recurrent state (delta tensors, conv rings and scratch) is host
        /// memory; its device copies are charged as <see cref="ResourceClass.DeviceKv"/> by
        /// <see cref="MeasureHolderFootprint"/>. Measuring <see cref="ResourceClass.StateSnapshot"/>
        /// against GPU headroom evicted retained conversations whenever concurrent requests
        /// filled the device. DeviceKv also counts the K/V device copies of idle retained
        /// holders: a clone demotes those to host memory before allocating
        /// (EnsureDeviceRetainHeadroom), so evicting payloads for the same room loses prefixes.</summary>
        public long QuerySpareBytes(ResourceClass cls)
        {
            if (cls == ResourceClass.StateSnapshot && IsGgmlBackend && _backend != BackendType.GgmlMetal)
                return QueryPrefixCacheSpareBytes(ResourceClass.HostKv);
            long spare = QueryPrefixCacheSpareBytes(cls);
            if (spare < 0 || cls != ResourceClass.DeviceKv)
                return spare;
            return spare + DemotableRetainedDeviceBytes();
        }

        private long DemotableRetainedDeviceBytes()
        {
            if (_retainedFusedHolders == null || !DeviceHolderCopiesEnabled)
                return 0;
            long bytes = 0;
            foreach (var holder in _retainedFusedHolders.Values)
                if (!holder.Retired && !holder.ArenaStateResident && holder.DeviceMirrored && CanCopyHolderKvOnDevice(holder))
                    bytes += HolderKvBytes(holder, holder.KvCapacity);
            return bytes;
        }

        // ---------------------------------------------------------------- end states

        /// <summary>The bytes adopting the current primary would publish, without
        /// allocating a replacement, settling native state or changing ownership.</summary>
        public bool TryMeasurePrimaryEndState(int length, out PayloadFootprint footprint)
        {
            footprint = default;
            if (_activeFusedKey != null || _kvCacheK == null || _kvCacheV == null
                || length <= 0 || _cacheSeqLen != length || length > _kvCacheCapacity)
                return false;
            footprint = MeasureHolderFootprint(SnapshotActiveCache());
            return true;
        }

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            return SupportsPerSequenceFusedForward
                && HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
        }

        /// <summary>A live holder of exactly <paramref name="payloadTokens"/> tokens, materialized at that
        /// length only: the GatedDeltaNet recurrence has no earlier state to rewind to.</summary>
        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
            => TryGetRetained(payloadKey, out var holder)
               && holder.CacheSeqLen == payloadTokens && targetTokens == payloadTokens;

        /// <summary>
        /// §6.1 for a hybrid holder: bind it (the arena slot is keyed by the holder's pointers), flush in the
        /// order <see cref="TryCheckpointActiveCache"/> uses — arena slot and K/V mirrors, fused-decode conv
        /// scratch and delta mirrors, verify-owned device state — record the flags, rebind the previous
        /// cache. The host copy is then current, so the device state is no longer marked resident: the
        /// holder's next fused decode re-seeds its device state from those same bytes.
        /// </summary>
        public bool SettleForCopy(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return false;
            if (!(holder.KvHostDirty || holder.GdnHostDirty || holder.ArenaStateResident || holder.FdStateResident))
                return true;
            // MLX never sets those flags (its holders' K/V and GDN caches are the only
            // copies), so a flagged holder there has nothing this could settle.
            if (!IsGgmlBackend || IsTensorParallel || _isRecurrent == null) return false;
            // A CUDA clone copies K/V between device copies, so device-authoritative K/V
            // needs no download; only device-held recurrent state must reach the host.
            bool kvOnDevice = CanCopyHolderKvOnDevice(holder);
            if (kvOnDevice && !(holder.GdnHostDirty || holder.ArenaStateResident || holder.FdStateResident))
                return true;
            // The active cache keeps its verify state on the device across the visit.
            SettleVerifyStateOnDevice();
            Qwen35KvCacheHolder previous = SnapshotActiveCache();
            LoadCacheHolder(holder);
            try
            {
                if (!kvOnDevice) EnsureKvCacheHostSynchronized();
                EnsureFusedDecodeStateHostSynchronized();
                DrainDeviceRecurrentState();
                // An arena flush writes K/V to the host and drops the device copies.
                if (kvOnDevice && !CanCopyHolderKvOnDevice(SnapshotActiveCache()))
                    EnsureKvCacheHostSynchronized();
                // Device-newer recurrent state lives in the fused-decode mirrors, an arena
                // slot or the verify owner's slices, all settled above. A holder retained
                // after its last verify kept the dirty flag it was snapshotted with, while
                // the next holder switch drained that verify state into its host bytes;
                // with none of the three left, those host bytes are current.
                if (_gdnStateHostDirty && !_fdStateResident && !_arenaStateResident && !_fvDeviceStateCurrent)
                    _gdnStateHostDirty = false;
                if (!_gdnStateHostDirty)
                    _fdStateResident = false;
                holder.KvHostDirty = _kvCacheHostDirty;
                holder.GdnHostDirty = _gdnStateHostDirty;
                holder.ArenaStateResident = _arenaStateResident;
                holder.FdStateResident = _fdStateResident;
            }
            finally
            {
                LoadCacheHolder(previous);
            }
            bool settled = !(holder.GdnHostDirty || holder.ArenaStateResident || holder.FdStateResident)
                && (!holder.KvHostDirty || CanCopyHolderKvOnDevice(holder));
            if (!settled && PrefixReleaseDebug)
                Console.Error.WriteLine($"[prefix-settle] {holder.CacheSeqLen} tokens unsettled: kv={holder.KvHostDirty} gdn={holder.GdnHostDirty} " +
                    $"arena={holder.ArenaStateResident} fd={holder.FdStateResident} fdSlot={_fdGdnSlot != null} kvOnDevice={kvOnDevice}");
            return settled;
        }

        /// <summary>Host bytes as <see cref="IdleHolderBytes"/> counts them, split into attention K/V
        /// (<see cref="ResourceVector.HostKv"/>) and recurrent state (<see cref="ResourceVector.StateSnapshot"/>),
        /// plus the device copies of a holder that was bound on a GPU backend.</summary>
        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            if (!TryGetRetained(payloadKey, out var holder)) return default;
            return MeasureHolderFootprint(holder);
        }

        private PayloadFootprint MeasureHolderFootprint(Qwen35KvCacheHolder holder)
        {
            MeasureHolder(holder, out long kvBytes, out long stateBytes, out long deviceBytes);
            var vector = new ResourceVector
            {
                HostKv = kvBytes,
                StateSnapshot = stateBytes,
                DeviceKv = holder.DeviceMirrored ? PinnedDeviceBytes(holder, kvBytes, deviceBytes) : 0,
            };
            return new PayloadFootprint(holder.CacheSeqLen, holder.KvCapacity, vector, PositionDelta: 0);
        }

        /// <summary>The device bytes of a retained holder that only eviction can free. With device
        /// holder copies, idle K/V device copies are demoted to host memory under pressure
        /// (<see cref="EnsureDeviceRetainHeadroom"/>) instead, so they are not charged; an
        /// arena-resident holder and device-only recurrent state still are.</summary>
        private long PinnedDeviceBytes(Qwen35KvCacheHolder holder, long kvBytes, long deviceBytes)
        {
            if (!DeviceHolderCopiesEnabled || holder.ArenaStateResident)
                return deviceBytes;
            return holder.FdStateResident || holder.GdnHostDirty ? Math.Max(0, deviceBytes - kvBytes) : 0;
        }

        /// <summary>What <see cref="DeepCopyHolder"/> allocates: attention rows sized to what the source
        /// holds plus a padded window, and a full set of recurrent state. This is a conservative
        /// allocation bound: an exact-capacity idle holder can satisfy an active clone without
        /// allocating again, but a budget check must remain safe if pressure trims that pool.</summary>
        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens)
            => EstimateCloneBytes(payloadKey, targetTokens, 0);

        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens, int requiredContextTokens)
        {
            if (!TryGetRetained(payloadKey, out var source)) return default;
            int cap = CloneCapacity(source, requiredContextTokens);
            long kv = 0;
            var seen = new HashSet<Storage>();
            foreach (Tensor[] set in new[] { source.K, source.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && seen.Add(t.Storage))
                            kv += t.Sizes[1] <= 0 ? t.Storage.ByteLength : t.Storage.ByteLength / t.Sizes[1] * cap;
            MeasureHolder(source, out _, out long state, out _);
            if (source.Logits != null) state -= (long)source.Logits.Length * sizeof(float);   // a copy has no logits buffer
            return new ResourceVector { HostKv = kv, StateSnapshot = state };
        }

        public int ResolveCloneReservationTokens(string payloadKey, int requiredContextTokens)
        {
            if (!TryGetRetained(payloadKey, out var source) || requiredContextTokens <= 0) return 0;
            int compact = CloneCapacity(source, 0);
            // A fresh clone has no resident allocation to credit against the budget.
            // Leave the same half-spare scratch allowance as PrepareForPrefill and
            // freeze this result before the coordinator estimates/evicts/materializes.
            long? spare = GetCacheMemorySpareBytes();
            int requested = Math.Min(requiredContextTokens, _maxContextLength);
            if (spare.HasValue)
                requested = ResolvePrefillReservationLength(spare.Value, KvCacheBytesPerToken,
                    requested, compact, CacheCapacityAlignment);
            return CloneCapacity(source, requested);
        }

        private int CloneCapacity(Qwen35KvCacheHolder source, int requiredContextTokens)
        {
            int rows = Math.Max(0, Math.Min(source.CacheSeqLen, source.KvCapacity));
            int compact = Math.Max(1, Math.Min(source.KvCapacity, CacheCapacityFor(rows)));
            if (requiredContextTokens <= compact) return compact;
            return ResolveAttentionCacheGrowthCapacity(compact, Math.Min(requiredContextTokens, _maxContextLength),
                _maxContextLength, KvCacheBytesPerToken, null, geometricGrowth: false);
        }

        /// <summary>The batched <see cref="DiscardRetainedCache"/> (DEC-24): recycle into the holder pool by
        /// <see cref="RecycleHolder"/>'s rules, and reset the decode graphs at most once for the whole batch,
        /// before its first disposal. Invalidation, pressure and reset dispose.</summary>
        public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            if (_retainedFusedHolders == null || payloadKeys.IsEmpty) return;
            if (PrefixReleaseDebug)
            {
                var described = new List<string>();
                foreach (string key in payloadKeys)
                    described.Add(key != null && _retainedFusedHolders.TryGetValue(key, out var h) ? $"{h.CacheSeqLen}" : "?");
                Console.Error.WriteLine($"[prefix-release] reason={reason} tokens=[{string.Join(",", described)}] retained={_retainedFusedHolders.Count}");
            }
            List<Qwen35KvCacheHolder> released = null;
            foreach (string key in payloadKeys)
            {
                if (key != null && _retainedFusedHolders.Remove(key, out var holder))
                {
                    holder.Retired = true;
                    (released ??= new List<Qwen35KvCacheHolder>()).Add(holder);
                }
            }
            if (released == null) return;

            bool dispose = reason is ReleaseReason.Invalidated or ReleaseReason.Pressure or ReleaseReason.Reset;
            var toPool = new List<Qwen35KvCacheHolder>();
            var toDispose = new List<Qwen35KvCacheHolder>();
            bool trimPool = false;
            int poolMax = ExecutionOptions.FromEnvironment().KvHolderPoolMax;
            long? spare = GetCacheMemorySpareBytes();
            long pooledBytes = PooledHolderBytes();
            foreach (var holder in released)
            {
                // Its state is no longer observable: retire an arena mapping before the
                // holder's stable pointers can be reassigned or freed.
                DiscardArenaSlotForHolder(holder);
                if (dispose || trimPool || holder.DisposalStarted)
                {
                    toDispose.Add(holder);
                    continue;
                }
                long bytes = IdleHolderBytes(holder);
                if (!CanPoolIdleCache(bytes, pooledBytes, 0, 1, spare))
                {
                    // As RecycleHolder: headroom is gone, so nothing more is parked.
                    trimPool = true;
                    toDispose.AddRange(toPool);
                    toPool.Clear();
                    toDispose.Add(holder);
                    continue;
                }
                if (CanPoolIdleCache(bytes, pooledBytes, (_holderPool?.Count ?? 0) + toPool.Count, poolMax, spare))
                {
                    toPool.Add(holder);
                    pooledBytes += bytes;
                }
                else
                {
                    toDispose.Add(holder);
                }
            }

            if (IsGgmlBackend && (toDispose.Count > 0 || (trimPool && _holderPool is { Count: > 0 })))
            {
                GgmlBasicOps.Qwen35ResetDecodeCache();
                InvalidateVerifyCache();
                CountDecodeGraphReset();
            }
            _holderGraphResetSuppressed++;
            try
            {
                foreach (var holder in toDispose)
                    DisposeHolder(holder);
                foreach (var holder in toPool)
                {
                    InvalidateHolderDeviceCopiesForReuse(holder);
                    _holderPool ??= new List<Qwen35KvCacheHolder>(Math.Max(1, poolMax));
                    _holderPool.Add(holder);
                    holder.Retired = false;
                }
                if (trimPool)
                    TrimIdleMemoryForPressure();
            }
            finally
            {
                _holderGraphResetSuppressed--;
            }
        }

        // ---------------------------------------------------------------- helpers

        /// <summary>
        /// Every allocation a holder owns, by class: attention K/V storages; the recurrent state (delta-state
        /// storages, host conv rings and their write indices, the fused-decode conv scratch — one float per
        /// ring float — and the holder's logits buffer); and what a bound holder mirrors on the device (K/V,
        /// delta state, conv scratch). Counted independently of <see cref="IdleHolderBytes"/>, which
        /// MeasureEndStateTests holds it to within 1%.
        /// </summary>
        private static void MeasureHolder(Qwen35KvCacheHolder holder, out long kvBytes, out long stateBytes, out long deviceBytes)
        {
            kvBytes = 0;
            long delta = 0, rings = 0;
            var seen = new HashSet<Storage>();
            foreach (Tensor[] set in new[] { holder.K, holder.V })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && seen.Add(t.Storage))
                            kvBytes = checked(kvBytes + t.Storage.ByteLength);
            if (holder.DeltaState != null)
                foreach (Tensor t in holder.DeltaState)
                    if (t != null && seen.Add(t.Storage))
                        delta = checked(delta + t.Storage.ByteLength);
            if (holder.ConvState != null)
                foreach (float[] ring in holder.ConvState)
                    if (ring != null) rings = checked(rings + (long)ring.Length * sizeof(float));
            long writeIndices = holder.ConvWriteIdx == null ? 0 : (long)holder.ConvWriteIdx.Length * sizeof(int);
            long scratch = holder.ConvScratch != IntPtr.Zero ? rings : 0;
            long logits = holder.Logits == null ? 0 : (long)holder.Logits.Length * sizeof(float);
            stateBytes = checked(delta + rings + writeIndices + scratch + logits + holder.MlxGdnStateBytes);
            deviceBytes = checked(kvBytes + delta + scratch);
        }

        private bool TryGetRetained(string payloadKey, out Qwen35KvCacheHolder holder)
        {
            holder = null;
            return payloadKey != null && _retainedFusedHolders != null
                && _retainedFusedHolders.TryGetValue(payloadKey, out holder)
                && !holder.Retired && !holder.DisposalStarted && holder.K != null;
        }

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys =>
            _retainedFusedHolders == null ? Array.Empty<string>() : _retainedFusedHolders.Keys.ToArray();

        public int PrivateHolderCount => _fusedHolders?.Count ?? 0;

        public int PrimaryCacheLength => _activeFusedKey == null ? _cacheSeqLen : (_primaryHolder?.CacheSeqLen ?? 0);

        // ---------------------------------------------------------------- test visibility

        /// <summary>The idle-holder byte count the holder pool budgets with, for a retained payload.</summary>
        internal long RetainedIdleHolderBytes(string payloadKey) => TryGetRetained(payloadKey, out var h) ? IdleHolderBytes(h) : -1;

        /// <summary>The same count for a request's private holder (a clone, before it is bound).</summary>
        internal long PrivateIdleHolderBytes(string requestId)
            => _fusedHolders != null && requestId != null && _fusedHolders.TryGetValue(requestId, out var h) ? IdleHolderBytes(h) : -1;

        /// <summary>Whether a retained holder has device state newer than, or not yet reflected in, its host bytes.</summary>
        internal bool IsRetainedDeviceAuthoritative(string payloadKey)
            => TryGetRetained(payloadKey, out var h) && (h.KvHostDirty || h.GdnHostDirty || h.ArenaStateResident || h.FdStateResident);
    }
}
