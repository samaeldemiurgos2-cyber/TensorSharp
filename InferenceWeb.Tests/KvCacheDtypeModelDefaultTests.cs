// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// The K/V cache dtype default is decided per model, from the model being loaded.
///
/// <para>It used to be one process-wide default that only ever went up
/// (<c>KvCacheDtypeConfig.ApplyModelDtypeDefault</c>): the first model with sub-F32 weights
/// set it to F16 and nothing set it back, so every later model in the process got an F16
/// cache, all-F32 models included - a server switching models, Studio, and a test run that
/// loads several fixtures. The cuda-f32 work found it as order-dependent LoRA parity tests.
/// Likewise a family that cannot read a block-quantized cache rewrote the operator's q8_0 /
/// q4_0 request to an explicit f16 for every later model.</para>
///
/// <para>Each row loads several tiny synthetic checkpoints in one process and checks every
/// model gets the cache it gets as the first load of a fresh process: F16 for sub-F32
/// weights, F32 for all-F32 weights, and the user's explicit choice over both.</para>
/// </summary>
public sealed class KvCacheDtypeModelDefaultTests : IDisposable
{
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-kv-default-").FullName;
    private readonly EnvScope _env = new();

    public KvCacheDtypeModelDefaultTests()
    {
        // Every row starts where a fresh process does: no dtype chosen.
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, explicitlySet: false);
        _env.ClearSpeculationVars();
        _env.Set("MAX_CONTEXT", "256");
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ------------------------------------------------------------------ checkpoints

    /// <summary>mistral3 with every matrix F32: an all-F32 model.</summary>
    private string Mistral3F32() => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }
        .Write(Path.Combine(_dir, "mistral3-f32.gguf"));

    /// <summary>mistral3 with layer 1's matrices F16, the only weights below F32, so its
    /// dominant quantized type is F16 (a sub-F32 model).</summary>
    private string Mistral3F16() => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", HalfPrecisionLayer = 1 }
        .Write(Path.Combine(_dir, "mistral3-f16.gguf"));

    private string HunyuanF16() => new DenseDecoderSyntheticModelBuilder
    {
        Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense", IncludeMistralControlTokens = false,
        IncludeQkNorms = true, RopeBase = 10000f, HalfPrecisionLayer = 1,
    }.Write(Path.Combine(_dir, "hunyuan-f16.gguf"));

    /// <summary>Dense Nemotron-H (Mamba2, attention, ReLU^2 FFN, Mamba2): its own cache
    /// initialisation, and a real quantized weight type.</summary>
    private string Nemotron(SyntheticGguf.GgmlType linearType) => new NemotronSyntheticModelBuilder
    {
        Layout = "SAFS", LinearType = linearType,
    }.Write(Path.Combine(_dir, $"nemotron-{linearType}.gguf"));

    private (string Quantized, string F32) Pair(string family) => family switch
    {
        "mistral3" => (Mistral3F16(), Mistral3F32()),
        "nemotron" => (Nemotron(SyntheticGguf.GgmlType.Q8_0), Nemotron(SyntheticGguf.GgmlType.F32)),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    private static KvCacheDtype LoadedCache(string path, BackendType backend)
    {
        using ModelBase model = ModelBase.Create(path, backend);
        return model.KvCacheDtype;
    }

    /// <summary>The user's request is what it was: models never write their own choice back.</summary>
    private static void AssertRequestUnchanged(KvCacheDtype dtype, bool explicitlySet)
    {
        Assert.Equal(dtype, KvCacheDtypeConfig.Current);
        Assert.Equal(explicitlySet, KvCacheDtypeConfig.IsExplicitlySet);
    }

    // ------------------------------------------------------------------ the default

    /// <summary>
    /// The regression: a quantized model, then an all-F32 one, in one process. The F32 model
    /// got the quantized model's F16 cache; it gets its own F32 cache, and the quantized model
    /// loaded again still gets F16.
    /// </summary>
    [Theory]
    [InlineData("mistral3")]
    [InlineData("nemotron")]
    public void AQuantizedModelThenAnF32Model_EachGetsItsOwnDefault(string family)
        => AssertQuantizedThenF32(family, BackendType.Cpu);

    [Theory]
    [InlineData("mistral3")]
    [InlineData("nemotron")]
    public void AnF32ModelThenAQuantizedModel_EachGetsItsOwnDefault(string family)
    {
        var (quantized, f32) = Pair(family);
        Assert.Equal(KvCacheDtype.F32, LoadedCache(f32, BackendType.Cpu));
        Assert.Equal(KvCacheDtype.F16, LoadedCache(quantized, BackendType.Cpu));
        Assert.Equal(KvCacheDtype.F32, LoadedCache(f32, BackendType.Cpu));
        AssertRequestUnchanged(KvCacheDtype.F32, explicitlySet: false);
    }

    /// <summary>The leak also crossed architectures (a mistral3 F16 row turned hunyuan-dense's
    /// F32 rows F16 in the LoRA parity tests): each family keeps its own default.</summary>
    [Fact]
    public void TheDefaultDoesNotCarryAcrossArchitectures()
    {
        Assert.Equal(KvCacheDtype.F16, LoadedCache(HunyuanF16(), BackendType.Cpu));
        Assert.Equal(KvCacheDtype.F32, LoadedCache(Mistral3F32(), BackendType.Cpu));
        Assert.Equal(KvCacheDtype.F16, LoadedCache(Nemotron(SyntheticGguf.GgmlType.Q8_0), BackendType.Cpu));
        Assert.Equal(KvCacheDtype.F32, LoadedCache(Nemotron(SyntheticGguf.GgmlType.F32), BackendType.Cpu));
        AssertRequestUnchanged(KvCacheDtype.F32, explicitlySet: false);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void AQuantizedModelThenAnF32Model_EachGetsItsOwnDefault_GgmlCpu()
        => AssertQuantizedThenF32("nemotron", BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlCuda)]
    public void AQuantizedModelThenAnF32Model_EachGetsItsOwnDefault_GgmlCuda()
        => AssertQuantizedThenF32("nemotron", BackendType.GgmlCuda);

    private void AssertQuantizedThenF32(string family, BackendType backend)
    {
        var (quantized, f32) = Pair(family);
        Assert.Equal(KvCacheDtype.F16, LoadedCache(quantized, backend));
        Assert.Equal(KvCacheDtype.F32, LoadedCache(f32, backend));
        Assert.Equal(KvCacheDtype.F16, LoadedCache(quantized, backend));
        AssertRequestUnchanged(KvCacheDtype.F32, explicitlySet: false);
    }

    // ------------------------------------------------------------------ explicit choices

    /// <summary>An explicit choice (KV_CACHE_DTYPE, --kv-cache-dtype, a server config) still
    /// wins over the model-aligned default for every model, in both orders.</summary>
    [Theory]
    [InlineData(KvCacheDtype.F32)]
    [InlineData(KvCacheDtype.F16)]
    public void AnExplicitChoiceWinsForEveryModel(KvCacheDtype chosen)
    {
        KvCacheDtypeConfig.Set(chosen);
        var (quantized, f32) = Pair("mistral3");
        Assert.Equal(chosen, LoadedCache(quantized, BackendType.Cpu));
        Assert.Equal(chosen, LoadedCache(f32, BackendType.Cpu));
        Assert.Equal(chosen, LoadedCache(quantized, BackendType.Cpu));
        AssertRequestUnchanged(chosen, explicitlySet: true);
    }

    /// <summary>
    /// The server's own path: <c>--kv-cache-dtype</c> through
    /// <see cref="ServerOptionsBuilder.ApplyKvCacheDtypeCliFlag"/>, the environment variable
    /// through <see cref="KvCacheDtypeConfig.ConfigureFromEnvironment"/>, and a model switch
    /// through <see cref="ModelLifecycleService.LoadModel"/>, which disposes the old model and
    /// creates the new one with the real factory. Without a choice each model gets its own
    /// default; with one, every model gets the choice.
    /// </summary>
    [Theory]
    [InlineData(null, null)]
    [InlineData("f32", null)]
    [InlineData("f16", null)]
    [InlineData(null, "f32")]
    public void AServerModelSwitchGivesEachModelItsOwnDefault(string? flag, string? environment)
    {
        _env.Set("KV_CACHE_DTYPE", environment);
        KvCacheDtypeConfig.ConfigureFromEnvironment();
        Assert.Equal(flag != null, ServerOptionsBuilder.ApplyKvCacheDtypeCliFlag(
            flag != null ? new[] { "--kv-cache-dtype", flag } : Array.Empty<string>()));
        KvCacheDtype? chosen = null;
        if (KvCacheDtypeConfig.TryParse(flag ?? environment, out KvCacheDtype parsed))
            chosen = parsed;

        var (quantized, f32) = Pair("mistral3");
        string nemotronF32 = Nemotron(SyntheticGguf.GgmlType.F32);
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance);
        lifecycle.LoadModel(quantized, null, "cpu");
        Assert.Equal(chosen ?? KvCacheDtype.F16, lifecycle.Model.KvCacheDtype);
        lifecycle.LoadModel(f32, null, "cpu");
        Assert.Equal(chosen ?? KvCacheDtype.F32, lifecycle.Model.KvCacheDtype);
        lifecycle.LoadModel(quantized, null, "cpu");
        Assert.Equal(chosen ?? KvCacheDtype.F16, lifecycle.Model.KvCacheDtype);
        lifecycle.LoadModel(nemotronF32, null, "cpu");
        Assert.Equal(chosen ?? KvCacheDtype.F32, lifecycle.Model.KvCacheDtype);
        AssertRequestUnchanged(chosen ?? KvCacheDtype.F32, explicitlySet: chosen != null);
    }

    // ------------------------------------------------------------------ block-quantized downgrade

    /// <summary>
    /// A family that cannot read a block-quantized cache loads f16 for an explicit q8_0 / q4_0
    /// request, and says so. That is this model's substitution: it used to be written back as
    /// an explicit f16, so every later model in the process lost the operator's choice.
    /// </summary>
    [Theory]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.Q4_0)]
    public void ABlockQuantizedDowngradeLeavesTheRequestForTheNextModel(KvCacheDtype requested)
    {
        string path = Path.Combine(_dir, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(path);
        KvCacheDtypeConfig.Set(requested);

        var stderr = new StringWriter();
        TextWriter original = Console.Error;
        Console.SetError(stderr);
        try
        {
            using (var declines = new ProbeModel(path, readsBlockQuantizedCache: false))
                Assert.Equal(KvCacheDtype.F16, declines.KvCacheDtype);
        }
        finally { Console.SetError(original); }
        Assert.Contains($"cannot read a {requested.ToShortString()} K/V cache on any of its attention paths; using f16 instead",
            stderr.ToString());
        AssertRequestUnchanged(requested, explicitlySet: true);

        using (var reads = new ProbeModel(path, readsBlockQuantizedCache: true))
            Assert.Equal(requested, reads.KvCacheDtype);
        AssertRequestUnchanged(requested, explicitlySet: true);
    }

    /// <summary>Stands in for a family's constructor: it runs the model-aligned default at the
    /// point a family does (before its caches are allocated), over no quantized weights.</summary>
    private sealed class ProbeModel : ModelBase
    {
        private readonly bool _readsBlockQuantizedCache;

        public ProbeModel(string path, bool readsBlockQuantizedCache) : base(path, BackendType.Cpu)
        {
            _readsBlockQuantizedCache = readsBlockQuantizedCache;
            Config = new ModelConfig { Architecture = "probe", VocabSize = 4 };
            ApplyModelAlignedKvCacheDefault(null);
        }

        protected override bool SupportsBlockQuantizedKvCache => _readsBlockQuantizedCache;

        protected override float[] ForwardCore(int[] tokens) => new float[4];

        protected override void ResetKVCacheCore()
        {
        }
    }
}
