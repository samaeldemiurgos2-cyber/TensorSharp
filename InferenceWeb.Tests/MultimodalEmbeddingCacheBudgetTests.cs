// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class MultimodalEmbeddingCacheBudgetTests
{
    private const long MiB = 1024 * 1024;
    private const long GiB = 1024 * MiB;

    [Theory]
    [InlineData(16L * 1024 * 1024 * 1024, 512L * 1024 * 1024)]    // small hosts keep the old floor
    [InlineData(128L * 1024 * 1024 * 1024, 2L * 1024 * 1024 * 1024)]
    [InlineData(512L * 1024 * 1024 * 1024, 4L * 1024 * 1024 * 1024)]  // capped
    [InlineData(0, 512L * 1024 * 1024)]                           // unknown host memory
    public void DefaultScalesWithHostMemory(long available, long expected)
        => Assert.Equal(expected, ModelMultimodalInjector.ResolveEmbeddingCacheBudgetBytes(null, available));

    [Fact]
    public void DefaultHoldsFourLanesOfSevenQwenScreenshots()
    {
        // 1008 merged tokens x 5120 hidden x F32 per image.
        const long image = 1008L * 5120 * 4;
        long budget = ModelMultimodalInjector.ResolveEmbeddingCacheBudgetBytes(null, 128 * GiB);
        Assert.True(budget >= 4 * 7 * image);
    }

    [Theory]
    [InlineData("64", 64 * MiB)]
    [InlineData("8192", 8192 * MiB)]
    public void ExplicitSettingWins(string setting, long expected)
        => Assert.Equal(expected, ModelMultimodalInjector.ResolveEmbeddingCacheBudgetBytes(setting, 512 * GiB));

    [Theory]
    [InlineData("0")]
    [InlineData("-5")]
    [InlineData("lots")]
    public void InvalidSettingFallsBackToTheDefault(string setting)
        => Assert.Equal(4 * GiB, ModelMultimodalInjector.ResolveEmbeddingCacheBudgetBytes(setting, 512 * GiB));
}
