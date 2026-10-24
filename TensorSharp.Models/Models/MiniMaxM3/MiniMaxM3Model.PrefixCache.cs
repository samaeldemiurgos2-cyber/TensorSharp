// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models;

// Each request owns a native sequence slot; a finished request's slot is retained
// under a payload key and donated to the request that continues it, and the prefix
// every conversation shares (system prompt, tool schemas) is checkpointed once by a
// device-to-device slot copy and cloned into each new conversation (Muse Spark's
// CopyAndDonate contract over SlotRetention).
//
// Every layer keeps every K/V row and the index keys advance with them, so a slot
// rewinds exactly to any token. Positions equal cache rows (MSA blocks are anchored
// to absolute rows), which is why a donated or cloned slot is only ever continued
// at its own head, never shifted.
public partial class MiniMaxM3Model : IPrefixCacheModel, IPrefixCacheModelDiagnostics
{
    public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
    {
        Class = FamilyClass.N,
        NamespaceFingerprint = KVStateFingerprint,
        EndState = UsesNativeExecutor ? EndStateSupport.CopyAndDonate : EndStateSupport.None,
        CanCaptureCopy = UsesNativeExecutor,
        AdoptPrimaryOnDisplacement = UsesNativeExecutor,
        PrimaryResident = true,
        MinRetainTokens = 32,
        Truncation = TruncationKind.Any,
        TruncationGranularity = 1,
        RewindCapTokens = int.MaxValue,
        Pages = PageSupport.None,
        ReuseAcrossMediaSpan = false,
        Persistable = false,
        MaxRetainedNativeSlots = 0,
    };

    public void AttachPrefixCache(IPrefixPayloadSink sink)
    {
        if (!UsesNativeExecutor) return;
        lock (_nativeSync)
            _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));
    }

    public void DetachPrefixCache()
    {
        lock (_nativeSync)
            _prefixCacheSink = null;
    }

    public bool TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        if (!UsesNativeExecutor) return false;
        lock (_nativeSync)
        {
            if (string.IsNullOrEmpty(payloadKey)) return false;
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(payloadKey)) return false;
            int source = SlotOfRequestOrActive(requestId);
            int tokens = source >= 0 ? GgmlMiniMaxM3Native.SlotNPast(_native, source) : -1;
            if (tokens <= 0) return false;
            int copy = CopyIntoNewSlot(source);
            if (copy < 0) return false;
            _retainedSlots.Add(payloadKey, new SlotRetention.RetainedSlot(copy, tokens));
            _checkpointKeys.Add(payloadKey);
            footprint = NativeFootprint(_retainedSlots[payloadKey]);
            return true;
        }
    }

    public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        if (!UsesNativeExecutor) return false;
        lock (_nativeSync)
        {
            if (_slotByRequest == null) return false;
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            var store = new NativeSlotStore(_native);
            if (!SlotRetention.RetainSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, length, true,
                    ref _primarySlot, ref _activeSlotKey, store))
                return false;
            _cacheSeqLen = store.ActiveHead();
            footprint = NativeFootprint(_retainedSlots[payloadKey]);
            return true;
        }
    }

    public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        if (!UsesNativeExecutor) return false;
        lock (_nativeSync)
        {
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            var store = new NativeSlotStore(_native);
            if (!SlotRetention.ConvertPrimarySlot(_retainedSlots, payloadKey, length, ref _primarySlot, ref _activeSlotKey, store))
                return false;
            _cacheSeqLen = store.ActiveHead();
            footprint = NativeFootprint(_retainedSlots[payloadKey]);
            return true;
        }
    }

    public bool TryMaterialize(in MaterializeRequest request)
    {
        if (!UsesNativeExecutor) return false;
        if (request.Op == MaterializeOp.Clone)
        {
            lock (_nativeSync)
            {
                if (_retainedSlots == null || !_retainedSlots.TryGetValue(request.PayloadKey, out var payload)
                    || payload.Tokens != request.PayloadTokens || request.TargetTokens < 0 || request.TargetTokens > payload.Tokens
                    || string.IsNullOrEmpty(request.TargetRequestId))
                    return false;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                if (_slotByRequest.ContainsKey(request.TargetRequestId)) return false;
                int copy = CopyIntoNewSlot(payload.Slot);
                if (copy < 0) return false;
                // The clone is unbound; the executor truncates it to TargetTokens at the first bind.
                _slotByRequest.Add(request.TargetRequestId, copy);
                return true;
            }
        }
        if (request.Op != MaterializeOp.Donate) return false;
        lock (_nativeSync)
        {
            if (!SlotRetention.CanDonate(_retainedSlots, request.PayloadKey, request.PayloadTokens, request.TargetTokens, true))
                return false;
            _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
            if (!SlotRetention.DonateSlot(_slotByRequest, _retainedSlots, request.PayloadKey, request.TargetRequestId, out _))
                return false;
            _checkpointKeys.Remove(request.PayloadKey);
            return true;
        }
    }

    public bool TryReturnDonation(string requestId, string payloadKey)
    {
        if (!UsesNativeExecutor) return false;
        lock (_nativeSync)
        {
            if (_slotByRequest == null) return false;
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            return SlotRetention.ReturnSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, ref _primarySlot,
                ref _activeSlotKey, new NativeSlotStore(_native));
        }
    }

    public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
    {
        if (!UsesNativeExecutor) return false;
        lock (_nativeSync)
            return SlotRetention.CanDonate(_retainedSlots, payloadKey, payloadTokens, targetTokens, true);
    }

    public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
    {
        if (!UsesNativeExecutor) return;
        lock (_nativeSync)
        {
            SlotRetention.ReleaseSlots(_retainedSlots, payloadKeys, new NativeSlotStore(_native));
            foreach (string key in payloadKeys)
                if (key != null && (_retainedSlots == null || !_retainedSlots.ContainsKey(key))) _checkpointKeys.Remove(key);
        }
    }

    public PayloadFootprint MeasureEndState(string payloadKey)
    {
        if (!UsesNativeExecutor) return default;
        lock (_nativeSync)
            return payloadKey != null && _retainedSlots != null && _retainedSlots.TryGetValue(payloadKey, out var entry)
                ? NativeFootprint(entry) : default;
    }

    public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => default;
    public bool TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId) => false;
    public bool TryExport(string payloadKey, System.IO.Stream destination) => false;

    public bool TryImport(string payloadKey, int tokens, System.IO.Stream source, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    public bool TryBeginImport(int tokens, out object importTicket)
    {
        importTicket = null;
        return false;
    }

    public bool RunImportRead(object importTicket, System.IO.Stream source) => false;

    public bool TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    public void AbortImport(object importTicket) { }

    public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

    public IReadOnlyCollection<string> RetainedPayloadKeys
    {
        get
        {
            lock (_nativeSync)
                return _retainedSlots == null ? Array.Empty<string>() : _retainedSlots.Keys.ToArray();
        }
    }

    public int PrivateHolderCount
    {
        get
        {
            lock (_nativeSync)
                return _slotByRequest?.Count ?? 0;
        }
    }

    public int PrimaryCacheLength
    {
        get
        {
            if (!UsesNativeExecutor) return _cacheSeqLen;
            lock (_nativeSync)
                return _activeSlotKey == null && _primarySlot >= 0 ? GgmlMiniMaxM3Native.NPast(_native) : 0;
        }
    }

    /// <summary>The native slot holding <paramref name="requestId"/>'s cache: its own slot when it has one
    /// (also when it is not the active one, as after a batched step), else the active slot.</summary>
    private int SlotOfRequestOrActive(string requestId)
    {
        if (requestId != null && _slotByRequest != null && _slotByRequest.TryGetValue(requestId, out int slot)) return slot;
        if (_activeSlotKey != null && _slotByRequest != null && _slotByRequest.TryGetValue(_activeSlotKey, out slot)) return slot;
        return _primarySlot;
    }

    /// <summary>Copy a slot into a fresh one, reclaiming retained slots for memory when needed (never the
    /// source, which a clone takes from the retained slots); -1 on failure.</summary>
    private int CopyIntoNewSlot(int source)
    {
        if (source < 0) return -1;
        int fresh = GgmlMiniMaxM3Native.SlotAlloc(_native);
        if (fresh < 0 && _retainedSlots is { Count: > 0 }) fresh = AllocSlotReclaimingRetained(source);
        if (fresh < 0) return -1;
        // Growing the copy's rows may free retained slots on the native side, as a graph does: the
        // reclaimable list it walks must be the current one (a slot donated since the last graph call is
        // live now), and what it frees must be retired here, so the copy runs between the same two hooks.
        bool copied;
        BeforeGraphCall();
        try
        {
            copied = GgmlMiniMaxM3Native.SlotCopy(_native, source, fresh);
        }
        finally
        {
            AfterGraphCall();
        }
        if (!copied)
        {
            GgmlMiniMaxM3Native.SlotFree(_native, fresh);
            return -1;
        }
        return fresh;
    }

    private PayloadFootprint NativeFootprint(SlotRetention.RetainedSlot entry) => new(entry.Tokens, _maxContextLength, default, PositionDelta: 0);
}
