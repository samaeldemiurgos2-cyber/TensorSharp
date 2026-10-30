using System;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public partial class Qwen4ExpModel
{
    private LayerSplitPlan _q4eStage;
    private NativeDistributedExecution _q4eDistributed;
    private GgmlBasicOps.CrossNodeAllReduce _q4eCrossNode;
    private bool HasDistributedQwen => LayerSplitGroup?.NodeCount > 1 || _tpGroup?.NodeCount > 1;
    private bool OwnsQwenLayer(int layer) => _q4eStage == null || _q4eStage.Owns(layer);
    private bool OwnsQwenPle => _isPle != null && Enumerable.Range(0, _isPle.Length).Any(l => _isPle[l] && OwnsQwenLayer(l));

    private void CleanupQwenFailedConstruction(Exception failure)
    {
        var group = (ITensorParallelGroup)LayerSplitGroup ?? _tpGroup;
        try
        {
            if (_q4eDistributed != null) _q4eDistributed.Abort(failure);
            else if (group is IDistributedExecutionGroup execution) execution.Abort(failure);
        }
        catch { }
        try { Dispose(); }
        catch { try { base.Dispose(); } catch { } }
        finally
        {
            // Preserve the original loader error even when a partially created
            // native state or custom group's teardown fails.
            try { _gguf?.Dispose(); } catch { }
            if (OwnsTensorParallelGroup) try { group?.Dispose(); } catch { }
            GC.KeepAlive(_q4eCrossNode);
        }
    }

    private void RefuseUnmirroredQwenOperation(string operation)
    {
        if (HasDistributedQwen)
            throw new NotSupportedException($"Distributed Qwen does not mirror {operation} across its workers.");
    }

    private void ConfigureQwenDistributed(string path)
    {
        if (!HasDistributedQwen) return;
        var group = (ITensorParallelGroup)LayerSplitGroup ?? _tpGroup;
        _q4eDistributed = new NativeDistributedExecution(group);
        _q4eCrossNode = _q4eDistributed.AllReduce;
        if (!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_Q4E_DEBUG_SPAN_INPUT")))
            throw new NotSupportedException("Distributed Qwen cannot replace a stage input with a local diagnostic file.");
        if (LayerSplitGroup != null)
        {
            if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_Q4E_LAYER_SPLIT")))
                throw new NotSupportedException("Distributed Qwen layer placement balances bytes across nodes; remove the local TS_Q4E_LAYER_SPLIT override.");
            ApplyModelAlignedKvCacheDefaultFromMetadata();
            var bytes = new long[Config.NumLayers];
            long embedding = 0, head = 0;
            foreach (var info in _gguf.Tensors.Values)
            {
                if (info.Name == "per_layer_token_embd.weight") continue; // host-mapped sparse PLE table
                long size = IsQuantizedLinearWeight(info) || info.Shape.Length > 2
                    ? _gguf.GetTensorByteCount(info) : checked(info.NumElements * 4);
                if (LayerSplitPlan.TryGetLayer(info.Name, out int layer) && layer < bytes.Length)
                {
                    if (MoeCpuOffloadConfig.IsExplicitlySet && MoeCpuOffloadConfig.IsLayerOnCpu(layer)
                        && info.Name.Contains("_exps.", StringComparison.Ordinal)) continue;
                    bytes[layer] = checked(bytes[layer] + size);
                }
                else if (info.Name == "token_embd.weight") embedding = checked(embedding + size);
                else head = checked(head + size);
            }
            if (!_gguf.Tensors.ContainsKey("output.weight")) head = checked(head + embedding);
            _q4eStage = LayerSplitPlan.CreateWeighted(bytes, LayerSplitDegree,
                LayerSplitGroup.NodeId, LayerSplitGroup.NodeCount, embedding, head,
                LayerSplitPlan.DriverRunsLastStageFromEnvironment());
            Console.WriteLine($"  Qwen distributed layer split: node {_q4eStage.NodeId}/{_q4eStage.NodeCount}, stage {_q4eStage.StageIndex}, layers [{_q4eStage.FirstLayer},{_q4eStage.EndLayer}), {LayerSplitDegree} local devices.");
        }
        _q4eDistributed.ValidateConfiguration(
            $"qwen4exp|{NativeDistributedExecution.HeaderIdentity(path)}|{NativeDistributedExecution.MetadataIdentity(_gguf)}|{ResolveConfiguredContextLength()}|{_kvCacheDtype}|{group.Degree}|{group.GlobalDegree}|{LayerSplitGroup != null}|{_q4eStage?.DriverRunsLastStage}|{MoeCpuOffloadConfig.IsExplicitlySet}|{MoeCpuOffloadConfig.CpuMoeLayers}|{MoeCpuOffloadConfig.AllLayers}|" +
            NativeDistributedExecution.EnvironmentIdentity(new[] { "TS_N_CPU_MOE", "TS_CPU_MOE", "TS_Q4E_GDN_VERIFY", "TS_Q4E_DEBUG_SPAN_LAYERS", "TS_Q4E_PREFILL_COMBINE", "TS_Q4E_TEST_MMVQ_CHANNELS", "TS_Q4E_TEST_BATCHED_VERIFY" }));
    }

    protected override bool ShouldLoadWeight(GgufTensorInfo info)
    {
        if (_q4eStage == null) return base.ShouldLoadWeight(info);
        if (LayerSplitPlan.TryGetLayer(info.Name, out int layer)) return _q4eStage.Owns(layer);
        if (info.Name == "per_layer_token_embd.weight") return OwnsQwenPle;
        if (info.Name == "token_embd.weight")
            return _q4eStage.IsFirst || (_q4eStage.IsLast && !_gguf.Tensors.ContainsKey("output.weight"));
        return _q4eStage.IsLast;
    }

    private float[] ForwardQwenDistributedStage(int[] tokens)
    {
        _q4eDistributed.ThrowIfAborted();
        if (_pendingMRoPEPositions != null || _visionEmbeddingsList.Count != 0 || _specForwardActive)
            throw new NotSupportedException("Distributed Qwen currently executes ordinary text forwards; media and speculative state require a distributed protocol.");
        int count = tokens.Length, position = _cacheSeqLen;
        if (count == 0) throw new ArgumentException("A Qwen forward needs tokens.", nameof(tokens));
        EnsureCacheCapacity(checked(position + count));
        PrepareQsaHistory(position, count);
        Tensor residual = null;
        try
        {
            if (_q4eStage.IsFirst)
            {
                using var embedded = Embedding(tokens);
                residual = BroadcastToStreams(embedded, count);
            }
            else
            {
                var values = new float[checked(count * _hcDim)];
                LayerSplitGroup.Receive(values, values.Length, _q4eStage.PreviousNode);
                residual = new Tensor(_allocator, DType.Float32, count, _hcDim);
                residual.SetElementsAsFloat(values);
            }
            if (!TryFusedTokenSpans(residual, tokens, count, position))
                throw new NotSupportedException("Distributed Qwen requires its complete native token-span executor.");
            _cacheSeqLen = checked(position + count);
            _forwardCount++;
            bool driver = LayerSplitGroup.NodeId == 0;
            if (_q4eStage.IsLast)
            {
                if (!_spanLogitsValid) throw new InvalidOperationException("Qwen output stage did not produce logits.");
                // With the default rotated order the driver is the last stage and
                // its logits stay local; otherwise they return to node 0.
                if (!driver) LayerSplitGroup.Send(_spanLogits, _spanLogits.Length, 0);
                return _spanLogits;
            }
            var outgoing = TensorToFloatArray(residual);
            LayerSplitGroup.Send(outgoing, outgoing.Length, _q4eStage.NextNode);
            if (!driver) return Array.Empty<float>();
            _logitsBuffer ??= new float[Config.VocabSize];
            LayerSplitGroup.Receive(_logitsBuffer, _logitsBuffer.Length, _q4eStage.NodeOfStage(_q4eStage.NodeCount - 1));
            return _logitsBuffer;
        }
        catch (Exception failure) { _q4eDistributed.Abort(failure); throw; }
        finally { residual?.Dispose(); }
    }
}
