// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>
/// The pure C# V4.1 executor against the independent PyTorch oracle.
///
/// <para>The fixture is a five-layer numerical model that exercises both
/// compression ratios, the shared compressed caches, candidate pruning, the
/// Engram table and the delayed hyper-connection gates. Generate it with
/// <c>python eng/dsv41-fixture.py DIR --f32</c>, write the reference logits
/// beside it, and point <c>TS_DSV41_FIXTURE_DIR</c> at DIR. Without that the
/// tests skip, saying what is missing (<see cref="Dsv41OracleFactAttribute"/>):
/// the fixture needs numpy, torch, gguf and tokenizers.</para>
///
/// <para>The oracle is <c>eng/dsv41-reference.py</c>, not the native executor,
/// so a shared misreading of the architecture cannot pass.</para>
/// </summary>
public class Dsv41CpuExecutorTests
{
    private sealed record Fixture(string Gguf, int[] Tokens, int[] Other, float[][] Expected, float[][] ExpectedOther);

    /// <summary>
    /// TS_DSV41_FIXTURE_ORACLE=native compares against native-logits.json
    /// instead. Use it where the PyTorch reference and ggml legitimately part
    /// company -- quantized fixture weights, and index top-k large enough that
    /// tie order in the sparse selection is observable -- so the pure C#
    /// executor is still pinned to something rather than left unchecked there.
    /// </summary>
    private static (string Gguf, string Logits) Paths()
    {
        string dir = Environment.GetEnvironmentVariable("TS_DSV41_FIXTURE_DIR") ?? string.Empty;
        string oracle = Environment.GetEnvironmentVariable("TS_DSV41_FIXTURE_ORACLE") ?? "torch";
        return (Path.Combine(dir, "deepseek41-fixture.gguf"), Path.Combine(dir, oracle.ToLowerInvariant() switch
        {
            "native" => "native-logits.json",
            "csharp" => "csharp-logits.json",
            _ => "expected-logits.json",
        }));
    }

    /// <summary>Why the oracle tests cannot run here, or null: the fixture directory, its GGUF and the
    /// chosen oracle's logits must all be there. Decided when the tests are discovered, so a missing
    /// fixture is a visible skip and never a test that returns early and counts as a pass.</summary>
    internal static string OracleSkip()
    {
        string dir = Environment.GetEnvironmentVariable("TS_DSV41_FIXTURE_DIR");
        if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
            return "Requires the V4.1 oracle fixture: TS_DSV41_FIXTURE_DIR is unset or missing " +
                   "(python eng/dsv41-fixture.py DIR --f32, then the reference logits beside it).";
        var (gguf, logits) = Paths();
        if (!File.Exists(gguf))
            return $"Requires the V4.1 oracle fixture: no {Path.GetFileName(gguf)} in TS_DSV41_FIXTURE_DIR.";
        return File.Exists(logits) ? null
            : $"Requires the oracle's logits: no {Path.GetFileName(logits)} in TS_DSV41_FIXTURE_DIR " +
              "(TS_DSV41_FIXTURE_ORACLE picks torch, native or csharp).";
    }

    private static Fixture Load()
    {
        var (gguf, logits) = Paths();
        Assert.True(OracleSkip() == null, OracleSkip());

        using var doc = JsonDocument.Parse(File.ReadAllBytes(logits));
        JsonElement root = doc.RootElement;
        static int[] Ids(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();
        static float[][] Rows(JsonElement e) => e.EnumerateArray()
            .Select(row => row.EnumerateArray().Select(v => v.GetSingle()).ToArray()).ToArray();
        return new Fixture(gguf, Ids(root.GetProperty("tokens")), Ids(root.GetProperty("other")),
            Rows(root.GetProperty("expected")), Rows(root.GetProperty("expected_other")));
    }

    /// <summary>The oracle stores F32; 2e-5 is the tolerance the native
    /// executor is held to on the same fixture.</summary>
    private static void AssertClose(string what, float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double tolerance = 2e-5 + 2e-5 * Math.Abs(expected[i]);
            double error = Math.Abs(actual[i] - expected[i]) - tolerance;
            if (error > worst) { worst = error; at = i; }
        }
        Assert.True(worst <= 0,
            $"{what}: logit {at} was {actual[at]}, reference {expected[at]} (excess over tolerance {worst:E3})");
        int actualArgmax = Array.IndexOf(actual, actual.Max());
        int expectedArgmax = Array.IndexOf(expected, expected.Max());
        Assert.Equal(expectedArgmax, actualArgmax);
    }

    private static DeepSeek4CpuExecutor Open(Fixture fixture) =>
        new(fixture.Gguf, maxContext: 256, nUbatch: 32, nThreads: 4, new CpuAllocator(BlasEnum.DotNet));

    /// <summary>
    /// The same fixture through the DIRECT-CUDA engine, which owns its own
    /// kernels and shares nothing with ggml. Opt-in with TS_DSV41_FIXTURE_CUDA=1:
    /// it needs a GPU and a fixture written with <c>--cuda-attn</c> (the engine
    /// takes only 512-wide heads); without either it skips, saying which.
    /// </summary>
    [Dsv41OracleTheory(Cuda = true)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    public void TheDirectCudaEngineMatchesTheSameOracle(int chunk)
    {
        Fixture fixture = Load();

        using var executor = new DeepSeek4CudaExecutor(fixture.Gguf, maxContext: 256, nUbatch: 32, nGpu: 1);
        var logits = new float[executor.VocabSize];
        int step = chunk == 0 ? fixture.Tokens.Length : chunk;
        for (int start = 0; start < fixture.Tokens.Length; start += step)
        {
            int stop = Math.Min(start + step, fixture.Tokens.Length);
            executor.Forward(fixture.Tokens[start..stop], logits);
            AssertClose($"cuda chunk {step} position {stop}", logits, fixture.Expected[stop - 1]);
            Assert.Equal(stop, executor.NPast);
        }

        executor.Reset();
        Assert.Equal(0, executor.NPast);
        executor.Forward(fixture.Other, logits);
        AssertClose("cuda after reset", logits, fixture.ExpectedOther[^1]);
    }

    /// <summary>
    /// Writes the managed executor's per-position logits beside the fixture so a
    /// backend that cannot be held to the PyTorch reference directly can be held
    /// to THIS executor, which can.
    ///
    /// <para>The reference computes every matmul in float. Any backend that
    /// quantizes activations - all of them, on a quantized fixture - therefore
    /// drifts from it by far more than 2e-5 on a five-layer model of random
    /// weights, and the drift says nothing about whether the backend is correct.
    /// Comparing against the managed executor instead removes that common-mode
    /// term: it is the same arithmetic, and it is pinned to the reference by the
    /// F32 fixture.</para>
    /// </summary>
    [Dsv41OracleFact(DumpVariable = "TS_DSV41_DUMP_CSHARP_LOGITS")]
    public void DumpsTheManagedExecutorsLogitsWhenAsked()
    {
        Fixture fixture = Load();
        string outPath = Environment.GetEnvironmentVariable("TS_DSV41_DUMP_CSHARP_LOGITS");

        using var executor = Open(fixture);
        var rows = new List<float[]>();
        var logits = new float[executor.VocabSize];
        foreach (int _ in fixture.Tokens)
        {
            executor.Forward(fixture.Tokens[rows.Count..(rows.Count + 1)], logits);
            rows.Add((float[])logits.Clone());
        }
        executor.Reset();
        var other = new List<float[]>();
        foreach (int _ in fixture.Other)
        {
            executor.Forward(fixture.Other[other.Count..(other.Count + 1)], logits);
            other.Add((float[])logits.Clone());
        }

        File.WriteAllText(outPath, JsonSerializer.Serialize(new
        {
            tokens = fixture.Tokens,
            other = fixture.Other,
            expected = rows,
            expected_other = other,
        }));
    }

    [Dsv41OracleFact]
    public void PrefillMatchesTheReferenceOracle()
    {
        Fixture fixture = Load();

        using var executor = Open(fixture);
        var logits = new float[executor.VocabSize];
        executor.Forward(fixture.Tokens, logits);
        AssertClose("prefill", logits, fixture.Expected[^1]);
    }

    /// <summary>
    /// Splitting a prompt must not change a single logit. This is where the
    /// V4.1 port is most fragile: the ratio-2 compressor's window straddles the
    /// boundary, the Engram history is indexed by absolute position, and the
    /// delayed hyper-connection gates carry across the layer loop.
    /// </summary>
    [Dsv41OracleTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(8)]
    public void ChunkedPrefillMatchesTheReferenceOracleAtEveryPosition(int chunk)
    {
        Fixture fixture = Load();

        using var executor = Open(fixture);
        var logits = new float[executor.VocabSize];
        for (int start = 0; start < fixture.Tokens.Length; start += chunk)
        {
            int stop = Math.Min(start + chunk, fixture.Tokens.Length);
            executor.Forward(fixture.Tokens[start..stop], logits);
            AssertClose($"chunk {chunk} position {stop}", logits, fixture.Expected[stop - 1]);
            Assert.Equal(stop, executor.NPast);
        }
    }

    /// <summary>Reset has to clear the Engram token history and the compressor
    /// state ring, not only the KV rows.</summary>
    [Dsv41OracleFact]
    public void ResetClearsEveryPieceOfSequenceState()
    {
        Fixture fixture = Load();

        using var executor = Open(fixture);
        var logits = new float[executor.VocabSize];
        executor.Forward(fixture.Tokens, logits);
        executor.Reset();
        Assert.Equal(0, executor.NPast);
        executor.Forward(fixture.Other, logits);
        AssertClose("after reset", logits, fixture.ExpectedOther[^1]);
    }
}

/// <summary>[Fact] over the V4.1 oracle fixture: skips, saying what is missing, unless
/// <c>TS_DSV41_FIXTURE_DIR</c> holds the fixture and the chosen oracle's logits
/// (<see cref="Dsv41CpuExecutorTests.OracleSkip"/>). With <see cref="DumpVariable"/>, a tool rather than
/// a check: it also skips unless that variable names a file to write.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class Dsv41OracleFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Models";

    public Dsv41OracleFactAttribute() => Skip = Dsv41CpuExecutorTests.OracleSkip();

    public string DumpVariable
    {
        get => _dumpVariable;
        set
        {
            _dumpVariable = value;
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(value)))
                Skip ??= $"Writes the managed executor's logits only when {value} names the file to write.";
        }
    }
    private string _dumpVariable;
}

/// <summary>[Theory] variant of <see cref="Dsv41OracleFactAttribute"/>. With <see cref="Cuda"/>, the
/// rows run the direct-CUDA engine, which also needs a GPU and the opt-in TS_DSV41_FIXTURE_CUDA=1.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class Dsv41OracleTheoryAttribute : TheoryAttribute, ITraitAttribute
{
    public string RequiresValue => _cuda ? "Cuda,Models" : "Models";

    public Dsv41OracleTheoryAttribute() => Skip = Dsv41CpuExecutorTests.OracleSkip();

    public bool Cuda
    {
        get => _cuda;
        set
        {
            _cuda = value;
            if (!value) return;
            Skip ??= TestGates.CudaSkip;
            if (Environment.GetEnvironmentVariable("TS_DSV41_FIXTURE_CUDA") != "1")
                Skip ??= "Runs the direct-CUDA engine only with TS_DSV41_FIXTURE_CUDA=1 and a fixture written with " +
                         "--cuda-attn (the engine takes only 512-wide heads).";
        }
    }
    private bool _cuda;
}
