// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

/// <summary>
/// The native executor's tensor parallelism (--tp N on ggml_cuda): one graph per ubatch on
/// ggml's meta backend, every weight and K/V tensor split by ggml_ops_muse_spark_tp's plan.
/// Logits must match the independent double-precision decoder through a prefill whose
/// global K/V rows grow (copied GPU-locally into larger slices) while the sliding ring wraps,
/// decode, a rewind, batched decode over several slots and a prefix-checkpoint clone. At 10
/// GPUs the fixture's 16 KV groups split unevenly and rotate per layer, as the real model's do.
/// </summary>
[Collection("Muse Spark model")]
public sealed class MuseSparkNativeTensorParallelTests : IDisposable
{
    private static readonly string[] Saved =
    [
        "TS_MUSE_SPARK_NATIVE", "MAX_CONTEXT", "TS_MUSE_SPARK_KV_PAD", "TS_MUSE_SPARK_UBATCH", "TS_MUSE_SPARK_SWA_REWIND",
        "TS_MUSE_SPARK_FUSE_QKVG", "TS_MUSE_SPARK_SWA_FULL", "TS_MUSE_SPARK_SLOT_INIT_ROWS", "TS_MUSE_SPARK_LAYER_SPLIT",
        "TS_MUSE_SPARK_PIPELINE", "TS_MUSE_SPARK_PIPELINE_UBATCH", "TENSORSHARP_TP_DEVICES",
    ];
    private readonly Dictionary<string, string> _old = Saved.ToDictionary(n => n, Environment.GetEnvironmentVariable);
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-muse-spark-tp-" + Guid.NewGuid().ToString("N"));
    private const int Vocab = MuseSparkSyntheticModelBuilder.Vocab;
    private const int Context = 1024;
    // Hundreds of positions of F16 K/V against the double-precision oracle.
    private const float Tolerance = 2e-2f;

    public MuseSparkNativeTensorParallelTests()
    {
        Directory.CreateDirectory(_dir);
        MoeCpuOffloadConfig.Reset();
        foreach (string name in Saved) Environment.SetEnvironmentVariable(name, null);
        Environment.SetEnvironmentVariable("MAX_CONTEXT", Context.ToString());
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_UBATCH", "32");
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_SWA_REWIND", "0");
        // Global rows start at the 256-row pad and double while the prompt grows.
        Environment.SetEnvironmentVariable("TS_MUSE_SPARK_SLOT_INIT_ROWS", "8");
    }

    public void Dispose()
    {
        foreach (var (name, value) in _old) Environment.SetEnvironmentVariable(name, value);
        MoeCpuOffloadConfig.Reset();
        Directory.Delete(_dir, true);
    }

    private static int[] Tokens(int count, int seed)
    {
        var tokens = new int[count];
        uint state = (uint)seed * 2654435761u + 7;
        for (int i = 0; i < count; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            tokens[i] = (int)(state % (Vocab - 2)) + 2;
        }
        return tokens;
    }

    private static void Close(float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float error = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]));
            error = MathF.Max(error, MathF.Abs(expected[i] - actual[i]));
        }
        Assert.True(error <= Tolerance, $"maximum logit error {error} > {Tolerance}");
    }

    // Head dim 64, as the real model: a tensor-parallel layer cannot fall back to the
    // CPU, and CUDA's flash attention has no kernel for the default fixture's 8.
    private static MuseSparkSyntheticModelBuilder Fixture(int gpus)
        => gpus <= 2
            ? new MuseSparkSyntheticModelBuilder(headDim: 64, contextLength: Context)
            : new MuseSparkSyntheticModelBuilder(heads: 32, kvHeads: 16, headDim: 64, contextLength: Context);

    [MultiCudaFact(2, GgmlBackend = BackendType.GgmlCuda)]
    public void TwoGpusMatchReference() => Check(2);

    [MultiCudaFact(4, GgmlBackend = BackendType.GgmlCuda)]
    public void FourGpusMatchReference() => Check(4);

    [MultiCudaFact(10, GgmlBackend = BackendType.GgmlCuda)]
    public void TenGpusWithUnevenKvGroupsMatchReference() => Check(10);

    private void Check(int gpus)
    {
        var fixture = Fixture(gpus);
        using var model = new MuseSparkModel(fixture.Write(Path.Combine(_dir, Guid.NewGuid() + ".gguf")), BackendType.GgmlCuda,
            tpDegree: gpus);
        Assert.True(model.UsesNativeExecutor);
        long[] stats = model.NativeStatsForTests();
        Assert.NotNull(stats);
        Assert.Equal(gpus, stats[11]);
        Assert.Equal(0, stats[9]);   // no pipelined prefill: every GPU works on every layer

        int[] tokens = Tokens(440, 21);
        // 400 tokens in 32-token ubatches: the global rows grow 256 -> 512 (copied
        // on the GPUs) and the 256-row sliding ring wraps.
        Close(fixture.Reference(tokens[..400]), model.Forward(tokens[..400]));
        for (int end = 401; end <= 404; end++)
            Close(fixture.Reference(tokens[..end]), model.Forward([tokens[end - 1]]));
        // A short rewind keeps the window the next query reads.
        Assert.True(model.TryTruncateKVCache(380));
        Close(fixture.Reference(tokens[..440]), model.Forward(tokens[380..440]));

        // Several slots: separate prefills, then batched decode steps.
        model.ResetKVCache();
        int[][] prompts = [Tokens(40, 2), Tokens(40, 3), Tokens(40, 4)];
        int[] lengths = [6, 9, 3];
        string[] ids = ["a", "b", "c"];
        for (int s = 0; s < ids.Length; s++)
        {
            Assert.True(model.BindSequenceCache(ids[s]));
            Close(fixture.Reference(prompts[s][..lengths[s]]), model.Forward(prompts[s][..lengths[s]]));
        }
        for (int step = 0; step < 6; step++)
        {
            var next = new int[ids.Length];
            var positions = new int[ids.Length];
            for (int s = 0; s < ids.Length; s++) { next[s] = prompts[s][lengths[s]]; positions[s] = lengths[s]; }
            var logits = new float[ids.Length][];
            Assert.True(model.TryForwardBatchedFusedDecode(ids, next, positions, logits));
            for (int s = 0; s < ids.Length; s++)
            {
                lengths[s]++;
                Close(fixture.Reference(prompts[s][..lengths[s]]), logits[s]);
            }
        }

        // A prefix checkpoint (a GPU-local copy of every K/V slice) and a clone of it.
        var prefixCache = (IPrefixCacheModel)model;
        int[] shared = Tokens(30, 10);
        int[] suffix = shared.Concat(Tokens(6, 11)).ToArray();
        Assert.True(model.BindSequenceCache("owner"));
        Close(fixture.Reference(shared), model.Forward(shared));
        Assert.True(prefixCache.TryCaptureCopy("owner", "pc:tp:1", out _));
        Assert.True(prefixCache.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone, "pc:tp:1", "clone", shared.Length, shared.Length)));
        Assert.False(model.BindSequenceCache("clone"));
        Close(fixture.Reference(suffix), model.Forward(suffix[shared.Length..]));
        model.BindSequenceCache("owner");
        Close(fixture.Reference(shared.Concat(new[] { 5 }).ToArray()), model.Forward([5]));
        foreach (string id in ids.Concat(new[] { "owner", "clone" })) model.OnSequenceReleased(id);
        prefixCache.ReleasePayloads(new[] { "pc:tp:1" }, ReleaseReason.Invalidated);
    }

    /// <summary>What native tensor parallelism cannot honour is a load refusal, not a quiet change of placement.</summary>
    [MultiCudaFact(2, GgmlBackend = BackendType.GgmlCuda)]
    public void UnsupportedCombinationsAreRefused()
    {
        var fixture = Fixture(2);
        string path = fixture.Write(Path.Combine(_dir, Guid.NewGuid() + ".gguf"));
        void Refused(string name, string value, string expect)
        {
            Environment.SetEnvironmentVariable(name, value);
            try
            {
                var ex = Assert.ThrowsAny<Exception>(() => new MuseSparkModel(path, BackendType.GgmlCuda, tpDegree: 2).Dispose());
                Assert.True(ModelLoadRefusal.TryDescribe(ex, out _), ex.ToString());
                Assert.Contains(expect, ex.ToString());
            }
            finally
            {
                Environment.SetEnvironmentVariable(name, null);
            }
        }
        Refused("TS_MUSE_SPARK_LAYER_SPLIT", "1,1", "TS_MUSE_SPARK_LAYER_SPLIT");
        Refused("TENSORSHARP_TP_DEVICES", "0,1", "TENSORSHARP_TP_DEVICES");
        Refused("TS_MUSE_SPARK_PIPELINE", "1", "TS_MUSE_SPARK_PIPELINE=1");
        Refused("TS_MUSE_SPARK_FUSE_QKVG", "1", "TS_MUSE_SPARK_FUSE_QKVG=1");
        // More GPUs than the fixture's two KV heads.
        var ex = Assert.ThrowsAny<Exception>(() => new MuseSparkModel(path, BackendType.GgmlCuda, tpDegree: 3).Dispose());
        Assert.True(ModelLoadRefusal.TryDescribe(ex, out _), ex.ToString());
    }
}
