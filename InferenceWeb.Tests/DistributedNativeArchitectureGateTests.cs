// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Host preflight only: no weight, native-library or device initialization.</summary>
public sealed class DistributedNativeArchitectureGateTests
{
    [Theory]
    [InlineData("deepseek41")]
    [InlineData("qwen4exp")]
    [InlineData("glm5next")]
    [InlineData("glm5-next")]
    public void NativeDistributedModesHaveExplicitBackendSupport(string id)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(id, out var architecture));
        ModelBase.ValidateDistributedTensorParallelism(architecture, BackendType.GgmlCuda);
        ModelBase.ValidateDistributedLayerSplit(architecture, BackendType.GgmlCuda);
        foreach (BackendType backend in new[] { BackendType.Cpu, BackendType.Cuda,
            BackendType.GgmlCpu, BackendType.GgmlVulkan, BackendType.GgmlMetal, BackendType.Mlx })
        {
            Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedTensorParallelism(architecture, backend));
            // DeepSeek V4.1 alone also runs its distributed layer-split stages on the direct-CUDA engine.
            if (id == "deepseek41" && backend == BackendType.Cuda)
                ModelBase.ValidateDistributedLayerSplit(architecture, backend);
            else
                Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedLayerSplit(architecture, backend));
        }
    }

    /// <summary>
    /// The direct-CUDA engines have no cross-node collectives, and only DeepSeek V4.1's has cross-node
    /// layer-split stages (Dsv4CudaEngine.Distributed.cs). The refusal has to
    /// name the backend that has them in the spelling --backend accepts (it used to print the enum
    /// name GgmlCuda, and the layer-split one named no backend at all).
    /// </summary>
    [Theory]
    [InlineData("deepseek41")]
    [InlineData("qwen4exp")]
    [InlineData("glm5next")]
    public void DirectCudaDistributedRefusalsNameGgmlCuda(string id)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(id, out var architecture));
        var tp = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ValidateDistributedTensorParallelism(architecture, BackendType.Cuda));
        Assert.Contains($"Architecture '{id}' does not implement distributed tensor parallelism on --backend cuda", tp.Message);
        Assert.Contains("Use one of its distributed backends: --backend ggml_cuda.", tp.Message);
        if (id == "deepseek41")
        {
            // Its direct-CUDA engine runs the stages of a distributed layer split (Dsv4CudaEngine.Distributed.cs).
            ModelBase.ValidateDistributedLayerSplit(architecture, BackendType.Cuda);
            return;
        }
        var layers = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ValidateDistributedLayerSplit(architecture, BackendType.Cuda));
        Assert.Contains($"Architecture '{id}' does not implement distributed --layer-split on --backend cuda", layers.Message);
        Assert.Contains("Its distributed stage executor runs on --backend ggml_cuda", layers.Message);
    }

    [Fact]
    public void FlashSupportDoesNotAdvertiseTheOlderGlmExecutionGraph()
    {
        Assert.True(ModelArchitectureRegistry.TryGet("glm5next", out var flash));
        Assert.True(ModelArchitectureRegistry.TryGet("glm5-next", out var flashAlias));
        Assert.True(ModelArchitectureRegistry.TryGet("glm-dsa", out var older));
        Assert.True(ModelArchitectureRegistry.TryGet("glm_dsa", out var olderAlias));
        Assert.NotSame(older, flash);
        Assert.Same(flash, flashAlias);
        Assert.Same(ChatProtocolRegistry.For("glm5next"), ChatProtocolRegistry.For("glm5-next"));
        Assert.Equal(OutputParserFactory.Create("glm5next").GetType(), OutputParserFactory.Create("glm5-next").GetType());
        Assert.Same(older, olderAlias);
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedLayerSplit(older, BackendType.GgmlCuda));
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedTensorParallelism(older, BackendType.GgmlCuda));
    }

    [Theory]
    [InlineData("deepseek41", 1, 3)]
    [InlineData("deepseek41", 2, 2)]
    [InlineData("qwen4exp", 1, 3)]
    [InlineData("qwen4exp", 2, 2)]
    [InlineData("glm5next", 1, 3)]
    [InlineData("glm5next", 2, 2)]
    public void PipelineSelectionPreservesTheLocalDegreeAndDoesNotEnableTensorSharding(string id, int local, int nodes)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(id, out var architecture));
        ITensorParallelGroup group = new Layers(local, nodes);
        var original = group;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(architecture,
            BackendType.GgmlCuda, 1, ref group, out int split, local));
        Assert.Equal(local, split);
        Assert.Same(original, group);
        Assert.False(group.IsActive);
        Assert.Equal(local * nodes, group.GlobalDegree);
        Assert.Throws<ArgumentException>(() => ModelBase.ResolveTensorParallelSupport(architecture,
            BackendType.GgmlCuda, 2, ref group, out _, local));
        Assert.Throws<ArgumentException>(() => ModelBase.ResolveTensorParallelSupport(architecture,
            BackendType.GgmlCuda, 1, ref group, out _, local + 1));
    }

    private sealed class Layers(int local, int nodes) : ILayerSplitGroup
    {
        public int Degree => local;
        public int GlobalDegree => local * nodes;
        public int GlobalRankOffset => 0;
        public int NodeCount => nodes;
        public int NodeId => 0;
        public bool IsActive => false;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() => throw new NotSupportedException();
        public void Barrier() => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Send(float[] values, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] values, int count, int source) => throw new NotSupportedException();
        public void Dispose() => throw new NotSupportedException();
    }
}
