// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Globalization;
using System.Text.RegularExpressions;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// How many GPUs the Muse Spark native executor is asked for. A count it cannot
/// honour must refuse the load (exit 2) instead of running the layers on fewer
/// GPUs, which is how ten A40s once became eight and the model stopped fitting.
/// </summary>
public sealed class MuseSparkNativeGpuCountTests
{
    /// <summary>Muse Spark's lone-prompt steps: 8192 tokens, a remainder of at most 2048 joined to the
    /// step before it (a short call would fill and drain the GPU pipeline alone), and the scheduler's
    /// own step whenever the native executor says the call would not pipeline.</summary>
    [Fact]
    public void LonePromptStepsAreFullWithShortRemaindersAbsorbed()
    {
        static int Plan(long p0, long n) => n >= 1024 ? 512 : 0;
        Assert.Equal(8192, MuseSparkModel.SoloPrefillStep(0, 20_000, Plan));
        Assert.Equal(10_000, MuseSparkModel.SoloPrefillStep(0, 10_000, Plan));   // 1808 left would not pipeline well alone
        Assert.Equal(8192, MuseSparkModel.SoloPrefillStep(0, 10_241, Plan));     // 2049 left is a call of its own
        Assert.Equal(3000, MuseSparkModel.SoloPrefillStep(0, 3000, Plan));
        Assert.Equal(0, MuseSparkModel.SoloPrefillStep(0, 600, Plan));           // too short to pipeline
        Assert.Equal(0, MuseSparkModel.SoloPrefillStep(5, 0, Plan));
        // The executor's verdict for the position decides (masks of a long context, a remembered misfit).
        Assert.Equal(0, MuseSparkModel.SoloPrefillStep(500_000, 9000, (p0, n) => p0 > 400_000 ? 0 : 512));
    }

    [Fact]
    public void LayerSplitOfTenUsesTenVisibleGpus()
        => Assert.Equal(10, MuseSparkModel.ResolveNativeGpuCount(10, null, 10, "CUDA"));

    [Fact]
    public void ExplicitPerGpuCountsRaiseTheRequest()
        => Assert.Equal(10, MuseSparkModel.ResolveNativeGpuCount(1, "7,7,6,6,6,6,6,6,6,6", 10, "CUDA"));

    [Fact]
    public void MoreGpusThanVisibleIsRefused()
    {
        var refusal = Assert.Throws<ModelLoadRefusedException>(() => MuseSparkModel.ResolveNativeGpuCount(10, null, 8, "CUDA"));
        Assert.Contains("sees only 8", refusal.Message);
    }

    [Fact]
    public void MoreGpusThanTheExecutorSpansIsRefused()
    {
        int beyond = MuseSparkModel.NativeMaxGpus + 1;
        var refusal = Assert.Throws<ModelLoadRefusedException>(() => MuseSparkModel.ResolveNativeGpuCount(beyond, null, 32, "CUDA"));
        Assert.Contains($"at most {MuseSparkModel.NativeMaxGpus}", refusal.Message);
    }

    [Fact]
    public void CpuBackendHasNoGpuCountToCheck()
        => Assert.Equal(3, MuseSparkModel.ResolveNativeGpuCount(3, null, null, "CPU"));

    [Fact]
    public void ManagedLimitMatchesTheNativeExecutor()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        if (dir is null) return;   // running outside a source checkout
        string source = File.ReadAllText(Path.Combine(dir.FullName, "TensorSharp.GGML.Native", "ggml_ops_muse_spark_exec.cpp"));
        Match match = Regex.Match(source, @"static constexpr int MAX_GPUS\s*=\s*(\d+)\s*;");
        Assert.True(match.Success, "MAX_GPUS not found in ggml_ops_muse_spark_exec.cpp");
        Assert.Equal(MuseSparkModel.NativeMaxGpus, int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture));
    }
}
