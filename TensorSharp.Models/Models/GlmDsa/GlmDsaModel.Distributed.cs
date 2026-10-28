// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.GGML;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

public partial class GlmDsaModel
{
    private NativeDistributedExecution _distributedNative;
    private GgmlGlmNative.CrossNodeAllReduce _distributedReduce;
    private GgmlGlmNative.LayerTransfer _distributedTransfer;
    private ITensorParallelGroup GlmDistributedGroup => (ITensorParallelGroup)LayerSplitGroup ?? _tpGroup;
    private bool IsDistributedGlm => GlmDistributedGroup?.NodeCount > 1;

    private void CleanupFailedConstruction(Exception failure)
    {
        // Configuration/tokenizer parsing can fail before the bridge exists.
        // Unblock peers immediately and preserve that original constructor error.
        try
        {
            if (_distributedNative != null) _distributedNative.Abort(failure);
            else if (GlmDistributedGroup is IDistributedExecutionGroup execution) execution.Abort(failure);
        }
        catch { }
        try { Dispose(); }
        catch
        {
            try { _exec?.Dispose(); } catch { }
            try { base.Dispose(); } catch { }
        }
        finally
        {
            // A partially initialized derived resource must not keep the GGUF
            // mapping or peer transports alive if another cleanup step throws.
            try { _gguf?.Dispose(); } catch { }
            try { GlmDistributedGroup?.Dispose(); } catch { }
            GC.KeepAlive(_distributedReduce);
            GC.KeepAlive(_distributedTransfer);
        }
    }

    private static void ValidateNativeDistributedGroup(BackendType backend, int degree, ITensorParallelGroup group)
    {
        bool fixtureCpu = backend == BackendType.GgmlCpu &&
            Environment.GetEnvironmentVariable("TS_GLM_TP_OVERSUBSCRIBE") == "1";
        if (!NativeRequested(backend) || (backend != BackendType.GgmlCuda && !fixtureCpu) ||
            group.NodeCount < 2 || group is not INestedTensorParallelGroup || group.Degree is < 1 or > 8 ||
            group.GlobalDegree != (long)group.NodeCount * group.Degree ||
            group.GlobalRankOffset < 0 || group.GlobalRankOffset > group.GlobalDegree - group.Degree ||
            group.GlobalRankOffset % group.Degree != 0)
            throw new NotSupportedException("Distributed GLM requires the native ggml_cuda executor and a valid nested group with 1–8 local GPUs.");
        if (group is ILayerSplitGroup ? degree != 1 : degree != group.Degree)
            throw new ArgumentException("Distributed GLM mode/local degree disagrees with its execution group.");
        if (SpeculationOptions.FromEnvironment().Enabled)
            throw new NotSupportedException("Distributed GLM does not mirror speculative KDA snapshots; disable speculation.");
        if (group is not ILayerSplitGroup && ResolveCpuMoeLayers() != 0)
            throw new NotSupportedException("Distributed GLM tensor parallelism requires GPU-resident routed experts; use layer split for CPU expert offload.");
    }

    private IntPtr LoadDistributedNative(string path, BackendType backend, int context, int ubatch, int threads)
    {
        _distributedNative = new NativeDistributedExecution(GlmDistributedGroup);
        if (!IsGlm5NextArch)
            throw new NotSupportedException("Distributed GLM execution requires glm5next (GLM-5.3-Flash).");
        var group = GlmDistributedGroup;
        bool driverLast = LayerSplitGroup != null && LayerSplitPlan.DriverRunsLastStageFromEnvironment();
        _distributedNative.ValidateConfiguration(
            $"glm5next|header={NativeDistributedExecution.HeaderIdentity(path)}|metadata={NativeDistributedExecution.MetadataIdentity(_gguf)}|layers={_numTrunkLayers}" +
            $"|hidden={Config.HiddenSize}|heads={Config.NumHeads}|vocab={Config.VocabSize}" +
            $"|experts={_numExperts}|used={_numExpertsUsed}|ff={_expertFfnLength}|ctx={context}|ubatch={ubatch}" +
            $"|local={group.Degree}|global={group.GlobalDegree}|layer={LayerSplitGroup != null}|driverLast={driverLast}|cpu={ResolveCpuMoeLayers()}" +
            NativeDistributedExecution.EnvironmentIdentity(new[] { "TS_GLM_TP_SHARD", "TS_GLM_OP_OFFLOAD",
                "TS_GLM_TRACE", "TS_GLM_TP_OVERSUBSCRIBE", "TS_GLM_TP_GRAPH_OPTIMIZE", "GGML_CUDA_FORCE_MMQ",
                "GGML_CUDA_FORCE_CUBLAS", "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_GRAPH_OPT" }));
        // Every node executes identical chunks and cache positions. Preserve the
        // agreed context rather than independently auto-capping it on each VM.
        if (LayerSplitGroup is { } layers)
        {
            // The native pipeline addresses peers by stage. With the default
            // rotated order node 0 runs the last stage, so logits stay local.
            _distributedNative.UseStageOrder(driverLast);
            _distributedTransfer = _distributedNative.Transfer;
            int stage = LayerSplitPlan.StageOf(layers.NodeId, layers.NodeCount, driverLast);
            IntPtr handle = GgmlGlmNative.LoadModelLayerPipeline(path, layers.Degree, stage, layers.NodeCount,
                context, ubatch, threads, ResolveCpuMoeLayers(), BackendRegistryName(backend), _distributedTransfer);
            if (handle != IntPtr.Zero)
            {
                try { GgmlGlmNative.SetPipelineDriver(handle, LayerSplitPlan.StageOf(0, layers.NodeCount, driverLast)); }
                catch { GgmlGlmNative.Free(handle); throw; }
            }
            return handle;
        }
        _distributedReduce = _distributedNative.AllReduce;
        _nativeTp = group.GlobalDegree;
        return GgmlGlmNative.LoadModelDistributed(path, group.Degree, group.GlobalDegree, group.GlobalRankOffset,
            context, ubatch, threads, BackendRegistryName(backend), _distributedReduce);
    }

    private void RefuseUnmirroredGlmOperation(string operation)
    {
        if (IsDistributedGlm)
            throw new NotSupportedException($"Distributed GLM does not mirror {operation}; use ordinary text prefill/decode with reset/refill.");
    }
}
