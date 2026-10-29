using System.Reflection;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpDistributedPlacementTests
{
    [Theory]
    [InlineData(256, false, false)]
    [InlineData(512, true, false)]
    [InlineData(256, false, true)]
    public void StagesLoadOnlyOwnedLayersAndAllocateOnlyTheirPersistentState(int hiddenSize, bool q4kxlExperts, bool tiedOutput)
    {
        string path = Path.Combine(Path.GetTempPath(), $"qwen-stages-{Guid.NewGuid():N}.gguf");
        var dtype = KvCacheDtypeConfig.Current;
        bool explicitDtype = KvCacheDtypeConfig.IsExplicitlySet;
        var owners = new List<int[]>();
        try
        {
            Qwen4ExpSyntheticModelBuilder.Write(path, contextLength: 128, hiddenSize: hiddenSize, q4kxlExperts: q4kxlExperts, tiedOutput: tiedOutput);
            for (int node = 0; node < 2; node++)
            {
                KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, false);
                using var model = new InspectableQwen(path, new MetadataGroup(node, 2));
                int[] layers = model.LoadedLayers;
                Assert.NotEmpty(layers);
                owners.Add(layers);
                for (int layer = 0; layer < 8; layer++)
                {
                    bool owned = layers.Contains(layer);
                    Assert.Equal(owned && !Qwen4ExpSyntheticModelBuilder.IsRecurrent(layer), model.Keys[layer] != null);
                    Assert.Equal(owned && Qwen4ExpSyntheticModelBuilder.IsRecurrent(layer), model.Conv[layer] != null);
                }
                // Stage order, not node id, decides the endpoints: by default node 0
                // (the driver) runs the last stage and keeps the head.
                bool first = LayerSplitPlan.StageOf(node, 2, LayerSplitPlan.DriverRunsLastStageFromEnvironment()) == 0;
                Assert.Equal(first || tiedOutput, model.HasWeight("token_embd.weight"));
                if (first) Assert.Equal(0, model.PreloadRank("token_embd.weight"));
                if (!first && tiedOutput) Assert.Equal(1, model.PreloadRank("token_embd.weight"));
                if (!first) Assert.Equal(1, model.PreloadRank("output_hc_norm.weight"));
                Assert.Equal(!first, model.HasWeight("output_hc_norm.weight"));
                Assert.Equal(layers.Contains(1), model.HasWeight("per_layer_token_embd.weight"));
                Assert.False(model.SupportsPerSequenceFusedForward);
                Assert.False(model.SupportsRetainedFusedCache);
                Assert.False(model.SupportsKVCacheTruncation);
                Assert.Throws<NotSupportedException>(() => model.LoadMtpDraftWeights("not-opened.gguf"));
                Assert.Throws<NotSupportedException>(() => model.LoadVisionEncoder("not-opened.gguf"));
                Assert.False(model.SpeculationProfitable);
                Assert.False(model.SpecSupportsMultimodalPrefill);
                Assert.False(model.SpecTrunkFollowsBoundCache);
                Assert.Throws<NotSupportedException>(() => model.SpecForward([1], null, new float[256], false));
                Assert.Throws<NotSupportedException>(() => model.SpecEnsureCapacity(1));
                Assert.Throws<NotSupportedException>(() => model.SpecSnapshotRecurrentState());
                Assert.Throws<NotSupportedException>(() => model.SpecRestoreRecurrentState());
                Assert.Throws<NotSupportedException>(() => model.SpecRewindCache(0));
                Assert.Throws<NotSupportedException>(() => model.BindSequenceCache("request"));
                Assert.Throws<NotSupportedException>(() => model.AdoptPrimaryCacheToFused("request"));
                Assert.Throws<NotSupportedException>(() => model.SetMRoPEPositions([0, 0, 0]));
                using var embedding = model.NewEmbedding();
                Assert.True(embedding.Storage.IsOwnerExclusive());
                Assert.Throws<NotSupportedException>(() => model.SetVisionEmbeddings(embedding, 0));
                Assert.False(embedding.Storage.IsOwnerExclusive());
                Assert.False(model.HasFusedSequenceCache("request"));
                Assert.Null(model.PendingPositions);
                Assert.Null(model.SpeculativeMetadata);
            }
            Assert.Empty(owners[0].Intersect(owners[1]));
            Assert.Equal(Enumerable.Range(0, 8), owners.SelectMany(x => x).Order());
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(dtype, explicitDtype);
            File.Delete(path);
        }
    }

    [Fact]
    public void MalformedMetadataAbortsBeforeBridgeCreationAndPreservesOriginalError()
    {
        string path = Path.Combine(Path.GetTempPath(), $"qwen-malformed-{Guid.NewGuid():N}.gguf");
        var group = new MetadataGroup(0) { FailDisposal = true };
        try
        {
            SyntheticGguf.Write(path,
            [
                new SyntheticGguf.Str { Key = "general.architecture", V = "qwen4exp" },
                new SyntheticGguf.U32 { Key = "qwen4exp.block_count", V = 8 },
                new SyntheticGguf.U32 { Key = "qwen4exp.embedding_length", V = 256 },
                new SyntheticGguf.U32 { Key = "qwen4exp.attention.head_count", V = 4 },
                new SyntheticGguf.U32 { Key = "qwen4exp.context_length", V = 128 },
            ], []);
            var failure = Assert.Throws<NotSupportedException>(() => new InspectableQwen(path, group));
            Assert.Contains("hyper_connection.count", failure.Message);
            Assert.Same(failure, group.Failure);
            Assert.True(group.Disposed);
            using var reopened = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { File.Delete(path); }
    }

    private sealed class InspectableQwen(string path, ILayerSplitGroup group)
        : Qwen4ExpModel(path, BackendType.Cpu, tpGroup: group)
    {
        public int[] LoadedLayers => _weights.Keys.Concat(_quantWeights.Keys).Concat(_stackedExpertWeights.Keys)
            .Select(name => LayerSplitPlan.TryGetLayer(name, out int layer) ? layer : -1)
            .Where(layer => layer >= 0).Distinct().Order().ToArray();
        public bool HasWeight(string name) => _weights.ContainsKey(name) || _quantWeights.ContainsKey(name);
        public int PreloadRank(string name) => PreloadRankForWeight(name);
        public Tensor NewEmbedding() => new(_allocator, DType.Float32, 1, Config.HiddenSize);
        public object PendingPositions => Read("_pendingMRoPEPositions");
        public object SpeculativeMetadata => Read("_specMetadata");
        public Tensor[] Keys => (Tensor[])Read("_kCache");
        public float[][] Conv => (float[][])Read("_gdnConvState");
        private object Read(string field) => typeof(Qwen4ExpModel)
            .GetField(field, BindingFlags.Instance | BindingFlags.NonPublic).GetValue(this);
    }

    // Constructor-only metadata agreement stand-in. Actual pipeline transport and
    // lifecycle parity run in separate processes via QwenDistributedFixture.
    private sealed class MetadataGroup(int node, int localDegree = 1) : ILayerSplitGroup, INestedTensorParallelGroup, IDistributedExecutionGroup
    {
        private readonly CpuAllocator _allocator = new(BlasEnum.DotNet);
        public int NodeId => node;
        public int NodeCount => 2;
        public int Degree => localDegree;
        public int GlobalDegree => 2 * localDegree;
        public int GlobalRankOffset => node * localDegree;
        public bool IsActive => false;
        public ITensorParallelGroup LocalGroup => this;
        public IAllocator GetAllocator(int rank) => _allocator;
        public void CrossNodeAllReduce(float[] buffer, int count)
        { for (int i = 0; i < count; i++) buffer[i] *= NodeCount; }
        public void Send(float[] buffer, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] buffer, int count, int source) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public bool FailDisposal { get; init; }
        public bool Disposed { get; private set; }
        public Exception Failure { get; private set; }
        public void Abort(Exception failure) => Failure ??= failure;
        public void Dispose()
        {
            Disposed = true;
            if (FailDisposal) throw new InvalidOperationException("Deliberate teardown failure");
        }
    }
}
