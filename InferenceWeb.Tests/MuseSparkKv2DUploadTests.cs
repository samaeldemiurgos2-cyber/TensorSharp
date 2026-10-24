// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

[Collection("Muse Spark model")]
public sealed class MuseSparkKv2DUploadTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeCpu2DUploadsPreserveChunkPrefixAndSpeculativeLogits() => Check(BackendType.GgmlCpu, 1);

    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void NativeCuda2DUploadsPreserveChunkPrefixAndSpeculativeLogits() => Check(BackendType.GgmlCuda, 1);

    [MuseSparkLayerSplitFact]
    public void NativeCudaTwoRank2DUploadsPreserveChunkPrefixAndSpeculativeLogits() => Check(BackendType.GgmlCuda, 2);

    private static void Check(BackendType backend, int split)
    {
        string directory = Path.Combine(Path.GetTempPath(), "spark-kv2d-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string[] names = ["TS_MUSE_SPARK_KV_UPLOAD_2D", "MAX_CONTEXT", "TS_MUSE_SPARK_PREFILL_CHUNK", "TS_KV_INITIAL_TOKENS"];
        string[] previous = names.Select(Environment.GetEnvironmentVariable).ToArray();
        var oldDtype = KvCacheDtypeConfig.Current;
        bool oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "64");
            Environment.SetEnvironmentVariable("TS_MUSE_SPARK_PREFILL_CHUNK", "2");
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "2");
            KvCacheDtypeConfig.Set(KvCacheDtype.F16);
            var fixture = new MuseSparkSyntheticModelBuilder();
            string path = fixture.Write(Path.Combine(directory, "tiny.gguf"), true);
            int[] tokens = [7,21,42,11,9,86,102,68];
            List<float[]> Run(string flag)
            {
                Environment.SetEnvironmentVariable("TS_MUSE_SPARK_KV_UPLOAD_2D", flag);
                using var model = new MuseSparkModel(path, backend, layerSplitDegree: split);
                var outputs = new List<float[]> { model.Forward(tokens[..4]).ToArray() };
                byte[] snapshot = new byte[checked((int)model.ComputeKVBlockByteSize(4))];
                Assert.True(model.TryExtractKVBlock(0,4,snapshot));
                outputs.Add(model.Forward(tokens[4..]).ToArray());
                model.ResetKVCache();
                Assert.True(model.TryInjectKVBlock(0,4,snapshot));
                outputs.Add(model.Forward(tokens[4..5]).ToArray());
                var logits = new float[2 * model.Config.VocabSize];
                model.SpecForward(tokens[5..7], null, logits, true);
                outputs.Add(logits);
                model.SpecRewindCache(5);
                outputs.Add(model.Forward(tokens[7..]).ToArray());
                Assert.Equal(flag == "1", model.Kv2DUploadCallCount > 0);
                return outputs;
            }
            var baseline = Run(null);
            foreach (var flag in new[] { "0", "true", "1" })
            {
                var actual = Run(flag);
                Assert.Equal(baseline.Count, actual.Count);
                for (int i = 0; i < baseline.Count; i++) Assert.Equal(baseline[i], actual[i]);
            }
        }
        finally
        {
            for (int i=0;i<names.Length;i++) Environment.SetEnvironmentVariable(names[i],previous[i]);
            KvCacheDtypeConfig.RestoreForTests(oldDtype,oldExplicit);
            Directory.Delete(directory,true);
        }
    }
}
