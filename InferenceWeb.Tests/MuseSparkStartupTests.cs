// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkStartupTests
{
    [Theory]
    [InlineData(KvCacheDtype.F32, true, false)]
    [InlineData(KvCacheDtype.F32, true, true)]
    [InlineData(KvCacheDtype.F16, false, false)]
    [InlineData(KvCacheDtype.F16, false, true)]
    public async Task AnySplitShardLoadsThroughFactoryAndDeclaresActualRuntimeModes(
        KvCacheDtype dtype, bool paged, bool enterFromLastShard)
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-spark-startup-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string[] names = ["MAX_CONTEXT", "TS_KV_INITIAL_TOKENS", "TS_MUSE_SPARK_PREFILL_CHUNK",
            "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE"];
        string?[] saved = names.Select(Environment.GetEnvironmentVariable).ToArray();
        KvCacheDtype savedDtype = KvCacheDtypeConfig.Current;
        bool savedExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            string[] values = ["16", "2", "4", "1", "1"];
            for (int i = 0; i < names.Length; i++) Environment.SetEnvironmentVariable(names[i], values[i]);
            KvCacheDtypeConfig.Set(dtype);
            var fixture = new MuseSparkSyntheticModelBuilder();
            string[] paths = fixture.WriteSplit(directory);
            string first = paths[0], last = paths[1];
            string entry = paths[enterFromLastShard ? 1 : 0];
            using (var probe = GgufFile.OpenWithoutSiblingShards(first))
            {
                Assert.Empty(probe.Tensors);
                Assert.Equal("muse-spark", probe.Metadata["general.architecture"]);
                Assert.True(probe.Metadata.ContainsKey("tokenizer.ggml.tokens"));
                Assert.Equal((ushort)0, probe.Metadata["split.no"]);
                Assert.Equal((ushort)2, probe.Metadata["split.count"]);
                Assert.Equal(fixture.Weights.Count, probe.Metadata["split.tensors.count"]);
                Assert.Equal(new FileInfo(first).Length, probe.GetRequiredLength(out _));
            }
            using (var probe = GgufFile.OpenWithoutSiblingShards(last))
            {
                Assert.Equal(3, probe.Metadata.Count);
                Assert.False(probe.Metadata.ContainsKey("general.architecture"));
                Assert.False(probe.Metadata.ContainsKey("tokenizer.ggml.tokens"));
                Assert.Equal((ushort)1, probe.Metadata["split.no"]);
                Assert.Equal((ushort)2, probe.Metadata["split.count"]);
                Assert.Equal(fixture.Weights.Count, probe.Metadata["split.tensors.count"]);
                Assert.Equal(fixture.Weights.Count, probe.Tensors.Count);
                Assert.Equal(last, Assert.Single(probe.FilePaths));
            }
            using var loaded = ModelBase.Create(entry, BackendType.Cpu);
            using (var merged = new GgufFile(entry))
            {
                merged.ThrowIfTruncated();
                Assert.Equal(fixture.Weights.Count, merged.Tensors.Count);
                Assert.Equal(2, merged.FilePaths.Count);
                Assert.Equal(first, merged.FilePaths[0]);
                Assert.Equal("muse-spark", merged.Metadata["general.architecture"]);
            }
            var model = Assert.IsType<MuseSparkModel>(loaded);
            Assert.Equal(16, model.MaxContextLength);
            Assert.Equal(64, model.Config.DeclaredContextLength);
            ExecutionCapabilities capabilities = ExecutionCapabilities.FromModel(model);
            Assert.Equal(paged, capabilities.BatchedForwardAvailable);
            Assert.True(capabilities.SupportsKvStateSnapshot);
            Assert.True(capabilities.SupportsSpeculativeTrunk);
            Assert.False(capabilities.HasDraftHead);
            using var drafter = SpeculatorRegistry.Create(model,
                new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.NGram }, out string? refusal);
            Assert.Null(refusal);
            Assert.IsType<NGramSpeculator>(drafter);

            int[] tokens = [7, 21, 42, 11, 9];
            float[] actual = model.Forward(tokens), expected = fixture.Reference(tokens);
            Assert.All(actual, x => Assert.True(float.IsFinite(x)));
            Assert.True(actual.Zip(expected, (a, b) => MathF.Abs(a - b)).Max() < 3e-3f);
            Assert.Throws<InvalidOperationException>(() => model.Forward(new int[12]));
            model.ResetKVCache();
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                BlockSize = 2, NumBlocks = 32, EnablePrefixCaching = true,
                MaxNumBatchedTokens = 8, SoloPrefillChunkSize = 8,
                MaxPrefillChunkSize = 4, StopRepetition = false,
            });
            Assert.True(engine.PrefixCacheActive);
            async Task<SequenceState> Run(string id, int[] prompt)
            {
                var sequence = new SequenceState(id, prompt, 2, 2, SamplingConfig.Greedy, cacheScope: "startup");
                await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(20));
                Assert.True(sequence.OutputTokens.Count > 0);
                return sequence;
            }
            int[] sharedPrompt = [7, 21, 42, 11, 9, 86, 102, 68, 52];
            var cold = await Run("cold", sharedPrompt);
            await Run("displace", [132, 41, 24, 62, 13, 45, 16, 91, 8]);
            var warm = await Run("warm", sharedPrompt);
            Assert.Equal(0, cold.PrefixCacheReusedTokens);
            Assert.Equal(8, warm.PrefixCacheReusedTokens);
            Assert.Equal(cold.OutputTokens, warm.OutputTokens);
        }
        finally
        {
            for (int i = 0; i < names.Length; i++) Environment.SetEnvironmentVariable(names[i], saved[i]);
            KvCacheDtypeConfig.RestoreForTests(savedDtype, savedExplicit);
            Directory.Delete(directory, true);
        }
    }
}
