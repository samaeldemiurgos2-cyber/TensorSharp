// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models;

// Managed path: Spark retains every linear KV row even on sliding-attention
// layers, so a prefix can rewind to any token. F16 uses host slabs; F32 also
// exposes the model-paged rows produced by its packed forward path. Every
// end-state member refuses (page-only family).
//
// Native path: each request owns a native sequence slot; a finished request's
// slot is retained under a payload key and donated to the request that
// continues it (the GLM executor's donate-only contract, SlotRetention).
public partial class MuseSparkModel : IPrefixCacheModel, IPrefixCacheModelDiagnostics
{
    public PrefixCacheCapabilities GetPrefixCacheCapabilities()
        => UsesNativeExecutor ? NativePrefixCacheCapabilities() : PageFamilyCapabilities(FamilyClass.S, TruncationKind.Any);

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
        return UsesNativeExecutor && NativeTryCaptureCopy(requestId, payloadKey, out footprint);
    }

    public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        return UsesNativeExecutor && NativeTryCaptureDonate(requestId, payloadKey, length, out footprint);
    }

    public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        return UsesNativeExecutor && NativeTryConvertPrimary(payloadKey, length, out footprint);
    }

    public bool TryMaterialize(in MaterializeRequest request)
        => UsesNativeExecutor && NativeTryMaterialize(request);

    public bool TryReturnDonation(string requestId, string payloadKey)
        => UsesNativeExecutor && NativeTryReturnDonation(requestId, payloadKey);

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
                return _activeSlotKey == null && _primarySlot >= 0 ? GgmlMuseSparkNative.NPast(_native) : 0;
        }
    }
}
