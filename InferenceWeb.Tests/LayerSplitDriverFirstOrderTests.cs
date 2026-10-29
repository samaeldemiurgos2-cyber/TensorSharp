// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy order (node 0 runs the first stage and receives the logits from
/// the last node) stays selectable with TENSORSHARP_LAYER_SPLIT_DRIVER_STAGE=first.
/// The variable is read at model construction, so these run alone.
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class LayerSplitDriverFirstOrderTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"layer-pipeline-first-{Guid.NewGuid():N}.gguf");
    private readonly string _previous = Environment.GetEnvironmentVariable(LayerSplitPlan.DriverStageVariable);

    public LayerSplitDriverFirstOrderTests()
    {
        BackendFailureWarmupTests.WriteProbeGguf(_path);
        Environment.SetEnvironmentVariable(LayerSplitPlan.DriverStageVariable, "first");
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(LayerSplitPlan.DriverStageVariable, _previous);
        File.Delete(_path);
    }

    [Theory]
    [InlineData(2, 2, 7)]
    [InlineData(3, 2, 11)]
    public async Task LegacyOrderReturnsLogitsToTheDriverAndMatchesLocal(int nodeCount, int localDegree, int layerCount)
    {
        Assert.False(LayerSplitPlan.DriverRunsLastStageFromEnvironment());
        await LayerSplitPipelineTests.RunPipelineParity(_path, nodeCount, localDegree, layerCount);
    }

    [Fact]
    public void InvalidDriverStageIsRefused()
    {
        Environment.SetEnvironmentVariable(LayerSplitPlan.DriverStageVariable, "middle");
        Assert.Throws<ArgumentException>(() => LayerSplitPlan.DriverRunsLastStageFromEnvironment());
    }

    [Fact]
    public void DefaultOrderRunsTheDriverAsTheLastStage()
    {
        Environment.SetEnvironmentVariable(LayerSplitPlan.DriverStageVariable, null);
        Assert.True(LayerSplitPlan.DriverRunsLastStageFromEnvironment());
    }

    [Fact]
    public void SocketBufferTooSmallForTwoInlineFramesIsRefused()
    {
        const string variable = "TENSORSHARP_TP_SOCKET_BUFFER_BYTES";
        string previous = Environment.GetEnvironmentVariable(variable);
        try
        {
            Environment.SetEnvironmentVariable(variable, "8192");
            var endpoints = new[] { new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 1), new System.Net.IPEndPoint(System.Net.IPAddress.Loopback, 2) };
            var options = new TensorSharp.Distributed.TcpTransportOptions { SymmetricInlineBytes = 32 * 1024 };
            var error = Assert.Throws<ArgumentException>(() =>
                new TensorSharp.Distributed.TcpCommunicator(0, endpoints, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(1), null, options));
            Assert.Contains("TENSORSHARP_TP_SYMMETRIC_INLINE_BYTES", error.Message);
        }
        finally { Environment.SetEnvironmentVariable(variable, previous); }
    }
}
