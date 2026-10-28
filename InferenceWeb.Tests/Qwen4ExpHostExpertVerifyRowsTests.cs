// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A qwen4exp token span of 2..8 rows runs the one-token device kernels (Q4eRowKernels in
/// ggml_ops_qwen4exp.cpp), so a prompt prefilled in chunks of that width - the shape of a
/// speculative verify forward - reproduces token-by-token decode bit for bit. Host-offloaded
/// experts (--cpu-moe / --n-cpu-moe) must keep that contract: a decode token's offloaded layer runs
/// on the one-token host team, and the span used to hand the same rows of a wider span to the
/// multi-token host expert graph instead, whose SiLU (and so the Q8_0 down-projection input)
/// rounds differently.
/// On Qwen3.8-Flash-Next UD-IQ1_M (40 of 48 layers offloaded) the first 8-token chunk already
/// drifted from token-by-token decode by KL 2.3e-3 and max |dlogit| 1.3.
/// </summary>
[Collection("MoeCpuOffloadConfig")]
public sealed class Qwen4ExpHostExpertVerifyRowsTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qwen4exp-verify-rows-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    public Qwen4ExpHostExpertVerifyRowsTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "1024");
        _env.Set("TS_PREFILL_WARMUP", "0");
        _env.ClearSpeculationVars();
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] Prompt = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % 250).ToArray();

    [GgmlFact(BackendType.GgmlCuda)]
    public void VerifyWidthChunks_WithHostExperts_MatchTokenByTokenBitwise_Cuda()
        => VerifyWidthChunksMatchTokenByToken(BackendType.GgmlCuda);

    private void VerifyWidthChunksMatchTokenByToken(BackendType backend)
    {
        // Every layer's routed experts on the host; the indexer keeps the dense
        // window the trained model uses below 2048 tokens. The two host paths
        // differ by ULPs in the expert activation (the one-token team evaluates
        // SiLU with std::exp, the graph with ggml's swiglu kernel) before the down
        // projection quantizes it to Q8_0, so the fixture needs enough host dot
        // products (1024-wide, ten of 64 UD-style IQ2_XS experts) for one such ULP
        // to flip a Q8_0 rounding. At the default 256-wide, four-expert size both
        // paths happen to round identically and this check passes without the fix.
        MoeCpuOffloadConfig.SetAllLayers();
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "qwen4exp.gguf"), indexerTopK: 2048,
            q2kxlExperts: true, expertCount: 64, expertUsedCount: 10, hiddenSize: 1024);
        using ModelBase model = ModelBase.Create(path, backend);
        Assert.IsType<Qwen4ExpModel>(model);

        model.ResetKVCache();
        var solo = new float[Prompt.Length][];
        for (int i = 0; i < Prompt.Length; i++)
            solo[i] = (float[])model.Forward(new[] { Prompt[i] }).Clone();

        foreach (int chunk in new[] { 2, 5, 8 })
        {
            model.ResetKVCache();
            for (int at = 0; at < Prompt.Length; at += chunk)
            {
                int n = Math.Min(chunk, Prompt.Length - at);
                float[] got = model.Forward(Prompt[at..(at + n)]);
                AssertBitwiseEqual(solo[at + n - 1], got, $"chunk={chunk}, rows [{at},{at + n})");
            }
            _output.WriteLine($"chunk={chunk}: every chunk's last logits bitwise equal to token-by-token decode");
        }
    }

    private static void AssertBitwiseEqual(float[] expected, float[] actual, string phase)
    {
        Assert.Equal(expected.Length, actual.Length);
        int differing = 0;
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{phase}: non-finite logit at {i}");
            if (BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i])) continue;
            differing++;
            worst = Math.Max(worst, Math.Abs((double)expected[i] - actual[i]));
        }
        Assert.True(differing == 0, $"{phase}: {differing}/{expected.Length} logits differ bitwise; max |dlogit|={worst:G9}.");
    }
}
