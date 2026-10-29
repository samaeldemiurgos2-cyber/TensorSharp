// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// What the two managed V4.1 executors -- <c>--backend cuda</c>'s direct-CUDA engine and
/// <c>--backend cpu</c>'s pure C# one -- refuse before a weight is read: llama.cpp's combined
/// Engram layout (unsloth's DeepSeek-V4.1-Flash-1017 GGUFs), which only the native loader reads,
/// and routed-expert quantizations the direct-CUDA engine cannot run where its split puts them.
/// Header-only synthetic GGUFs and descriptors: no weights, native library or GPU.
/// </summary>
public sealed class DeepSeek41ManagedExecutorLayoutTests : IDisposable
{
    private readonly EnvScope _env = new();
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dsv41-layout-" + Guid.NewGuid().ToString("N"));

    public DeepSeek41ManagedExecutorLayoutTests()
    {
        _env.ClearSpeculationVars();
        foreach (string name in new[] { "TS_DSV41_YOCO_KEEP", "TS_DSV41_ENGRAM_DEVICE", "TS_DSV41_ALLOW_NON_CUDA_GPU",
                     "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE" })
            _env.Set(name, null);
        _env.Set("KV_CACHE_DTYPE", "f16");
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { /* best effort */ }
    }

    public enum Layout { TensorSharp, Combined, Both, CombinedTableOnly }

    /// <summary>A deepseek41 header in one Engram layout, keys only as the native loader names them
    /// (dsv41_engram_gguf.h). The combined one carries what the 1017 files do: the <c>ple</c> hash
    /// geometry and the raw pad token that the token map compresses.</summary>
    private string Write(Layout layout)
    {
        var kv = new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.Str { Key = "general.architecture", V = "deepseek41" },
            new SyntheticGguf.U32 { Key = "deepseek41.block_count", V = 43 },
        };
        SyntheticGguf.AddByteTokenizer(kv, 16, 0, 1);
        kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.engram.token_map", V = Enumerable.Range(0, 16).Select(i => (uint)(i % 8)).ToArray() });
        if (layout is Layout.TensorSharp or Layout.Both)
        {
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.pad_id", V = 2 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.max_ngram_size", V = 3 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.head_count", V = 1 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.key_length", V = 32 });
            kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.engram.layer_ids", V = new uint[] { 1, 14 } });
        }
        if (layout is Layout.Combined or Layout.Both)
        {
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.ple.ngram_size", V = 3 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.ple.heads_per_ngram", V = 1 });
            kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.ple.layers", V = new uint[] { 1, 14 } });
            kv.Add(new SyntheticGguf.U64Arr { Key = "deepseek41.ple.layer_multipliers", V = new ulong[] { 7, 9, 11, 13, 15, 17 } });
            kv.Add(new SyntheticGguf.U32Arr { Key = "deepseek41.ple.head_vocab_sizes", V = new uint[] { 11, 13, 17, 19 } });
            kv.Add(new SyntheticGguf.U64Arr { Key = "deepseek41.ple.head_offsets", V = new ulong[] { 0, 11, 24, 41 } });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.embedding_length_per_layer_input", V = 32 });
            kv.Add(new SyntheticGguf.U32 { Key = "deepseek41.engram.pad_token_id", V = 3 });
        }
        var tensors = new List<SyntheticGguf.Tensor>();
        if (layout == Layout.CombinedTableOnly)
            tensors.Add(SyntheticGguf.Gen("per_layer_token_embd.weight", 0.1f, 32, 60));
        string path = Path.Combine(_dir, layout + ".gguf");
        SyntheticGguf.Write(path, kv, tensors);
        return path;
    }

    [Theory]
    [InlineData(BackendType.Cuda, "cuda", "the direct-CUDA engine")]
    [InlineData(BackendType.Cpu, "cpu", "the pure C# executor")]
    public void CombinedEngramLayoutIsRefusedFromTheHeaderOnTheManagedExecutors(BackendType backend, string option, string executor)
    {
        string path = Write(Layout.Combined);
        var error = Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateLoad(path, backend, null));
        Assert.Contains("llama.cpp's combined layout", error.Message);
        Assert.Contains("deepseek41.ple.*", error.Message);
        Assert.Contains("per_layer_token_embd.weight", error.Message);
        Assert.Contains("DeepSeek-V4.1-Flash-1017", error.Message);
        Assert.Contains($"--backend {option} ({executor}) reads only TensorSharp's per-layer layout", error.Message);
        Assert.Contains("Use --backend ggml_cuda", error.Message);
        // The executor's own failure named a metadata key, not the layout or the backend that loads it.
        Assert.DoesNotContain("Missing or invalid", error.Message);
        Assert.Equal(backend == BackendType.Cpu, error.Message.Contains("--backend ggml_cpu for a CPU-only run"));

        // The header the caller already holds gives the same answer without reopening the file.
        using var header = new GgufFile(path);
        var again = Assert.Throws<NotSupportedException>(() =>
            DeepSeek41Architecture.ValidateLoad("not-opened.gguf", backend, null, header: header));
        Assert.Equal(error.Message, again.Message);
    }

    /// <summary>The refusal is the descriptor's pre-load hook, so ModelBase.Create stops before the
    /// factory builds an executor -- this header has no weights for one to read.</summary>
    [Theory]
    [InlineData(BackendType.Cuda)]
    [InlineData(BackendType.Cpu)]
    public void TheRegistryLoadStopsBeforeTheExecutorIsBuilt(BackendType backend)
    {
        string path = Write(Layout.Combined);
        using (var probe = new GgufFile(path))
        {
            var context = new ModelCreateContext(path, backend, probe);
            Assert.Contains("combined layout", Assert.Throws<NotSupportedException>(() =>
                DeepSeek41Architecture.Descriptor.ApplyNativeTunables(context)).Message);
        }
        var error = Assert.Throws<NotSupportedException>(() => ModelBase.Create(path, backend));
        Assert.Contains("llama.cpp's combined layout", error.Message);
        Assert.Contains("Use --backend ggml_cuda", error.Message);
    }

    /// <summary>ggml_cuda and ggml_cpu go through the native loader, which reads both layouts.</summary>
    [Theory]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCpu)]
    public void CombinedEngramLayoutStillReachesTheNativeLoader(BackendType backend)
    {
        string path = Write(Layout.Combined);
        DeepSeek41Architecture.ValidateLoad(path, backend, null);
        using var header = new GgufFile(path);
        DeepSeek41Architecture.ValidateLoad(path, backend, null, header: header);
    }

    /// <summary>Regression guard: TensorSharp's layout, including a header that carries both key
    /// sets (the native loader then reads TensorSharp's), still reaches the managed executors.</summary>
    [Theory]
    [InlineData(BackendType.Cuda, Layout.TensorSharp)]
    [InlineData(BackendType.Cpu, Layout.TensorSharp)]
    [InlineData(BackendType.Cuda, Layout.Both)]
    [InlineData(BackendType.Cpu, Layout.Both)]
    public void TensorSharpEngramLayoutReachesTheManagedExecutors(BackendType backend, Layout layout)
    {
        string path = Write(layout);
        DeepSeek41Architecture.ValidateLoad(path, backend, null);
        using var header = new GgufFile(path);
        Assert.False(DeepSeek41Architecture.UsesCombinedEngramLayout(header));
        DeepSeek41Architecture.ValidateLoad(path, backend, null, header: header);
    }

    [Fact]
    public void TheCombinedTableAloneMarksTheLayout()
    {
        string path = Write(Layout.CombinedTableOnly);
        using var header = new GgufFile(path);
        Assert.True(DeepSeek41Architecture.UsesCombinedEngramLayout(header));
        Assert.Throws<NotSupportedException>(() => DeepSeek41Architecture.ValidateLoad(path, BackendType.Cuda, null));
    }

    // ---- routed-expert types on the direct-CUDA engine ----

    private const int Embd = 512, Ff = 256;

    private static CudaWeightDesc Expert(GgmlTensorType type, string name, int ne0, int ne1)
        => new() { GgmlType = (int)type, Name = name, Ne0 = ne0, Ne1 = ne1, Ne2 = 4 };

    private static Dsv4CudaEngine.LayerDesc Layer(int il, GgmlTensorType gateUp, GgmlTensorType down, string prefix = null)
    {
        prefix ??= $"blk.{il}.";
        return new Dsv4CudaEngine.LayerDesc
        {
            GateExps = Expert(gateUp, prefix + "ffn_gate_exps.weight", Embd, Ff),
            UpExps = Expert(gateUp, prefix + "ffn_up_exps.weight", Embd, Ff),
            DownExps = Expert(down, prefix + "ffn_down_exps.weight", Ff, Embd),
        };
    }

    private static Dsv4CudaEngine.ModelDesc Model(params Dsv4CudaEngine.LayerDesc[] layers) => new()
    {
        NLayer = layers.Length,
        Layers = layers,
        HeadDim = 512,
        NExpertUsed = 6,
        HostMatMul = DeepSeek4CudaExecutor.HostMatMul.Instance,
    };

    /// <summary>
    /// The routed-expert types of unsloth's 1017 UD-IQ2_XXS file against both places the engine can
    /// run them. docs/models/deepseek41.md quotes this table; IQ1_S and IQ2_S run in neither.
    /// </summary>
    [Theory]
    [InlineData(GgmlTensorType.IQ1_S, false, false)]
    [InlineData(GgmlTensorType.IQ2_S, false, false)]
    [InlineData(GgmlTensorType.IQ2_XS, false, true)]
    [InlineData(GgmlTensorType.IQ2_XXS, true, false)]
    [InlineData(GgmlTensorType.IQ3_XXS, false, true)]
    [InlineData(GgmlTensorType.Q2_K, true, true)]
    public void The1017ExpertTypesAgainstTheGpuKernelsAndTheHostMatmul(GgmlTensorType type, bool gpu, bool host)
    {
        Assert.Equal(gpu, Dsv4CudaEngine.SupportsResidentExpertType((int)type));
        Assert.Equal(host, DeepSeek4CudaExecutor.HostMatMul.Instance.SupportsType((int)type, Embd));
        Assert.Equal(host, DeepSeek4CudaExecutor.HostMatMul.Instance.SupportsType((int)type, Ff));
    }

    [Fact]
    public void UnsupportedGpuResidentExpertTypesAreRefusedByTensorName()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ1_S, GgmlTensorType.IQ2_S),
                      Layer(1, GgmlTensorType.IQ2_XXS, GgmlTensorType.Q2_K),
                      Layer(2, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS));
        var error = Assert.Throws<NotSupportedException>(() => Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 0));
        Assert.Contains("cannot run 4 routed-expert tensor(s)", error.Message);
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ1_S (GPU-resident)", error.Message);
        Assert.Contains("blk.0.ffn_up_exps.weight is IQ1_S (GPU-resident)", error.Message);
        Assert.Contains("blk.0.ffn_down_exps.weight is IQ2_S (GPU-resident)", error.Message);
        Assert.Contains("blk.2.ffn_down_exps.weight is IQ3_XXS (GPU-resident)", error.Message);
        Assert.DoesNotContain("blk.1.", error.Message);
        Assert.DoesNotContain("blk.2.ffn_gate_exps", error.Message);
        Assert.Contains("Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ3_S, IQ2_XXS, MXFP4", error.Message);
        Assert.Contains("Use --backend ggml_cuda", error.Message);
    }

    [Fact]
    public void SupportedExpertTypesPassOnTheGpuAndTheHost()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ2_XS, GgmlTensorType.IQ3_XXS),
                      Layer(1, GgmlTensorType.Q2_K, GgmlTensorType.Q3_K),
                      Layer(2, GgmlTensorType.IQ2_XXS, GgmlTensorType.MXFP4),
                      Layer(3, GgmlTensorType.Q8_0, GgmlTensorType.IQ3_S));
        // Layer 0's types have only a host kernel, so they pass offloaded and are refused resident.
        Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 1);
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ2_XS (GPU-resident)",
            Assert.Throws<NotSupportedException>(() => Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 0)).Message);
    }

    /// <summary>An offloaded layer is checked against the host matmul instead: IQ2_XXS has GPU
    /// kernels but no host one, and Q3_K the same.</summary>
    [Fact]
    public void OffloadedLayersAreCheckedAgainstTheHostMatmul()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ2_XXS, GgmlTensorType.Q2_K),
                      Layer(1, GgmlTensorType.Q4_K, GgmlTensorType.Q3_K),
                      Layer(2, GgmlTensorType.IQ2_XXS, GgmlTensorType.Q2_K));
        Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 0);
        var error = Assert.Throws<NotSupportedException>(() => Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 2));
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ2_XXS (offloaded by --n-cpu-moe)", error.Message);
        Assert.Contains("blk.1.ffn_down_exps.weight is Q3_K (offloaded by --n-cpu-moe)", error.Message);
        Assert.DoesNotContain("blk.2.", error.Message);
        // Without a host matmul the offload is refused by the engine's own check, not here.
        m.HostMatMul = null;
        Dsv4CudaEngine.ValidateRoutedExpertTypes(m, 2);
    }

    [Fact]
    public void GateAndUpMustShareOneType()
    {
        var layer = Layer(0, GgmlTensorType.Q4_K, GgmlTensorType.Q4_K);
        layer.UpExps = Expert(GgmlTensorType.Q5_K, "blk.0.ffn_up_exps.weight", Embd, Ff);
        var error = Assert.Throws<NotSupportedException>(() => Dsv4CudaEngine.ValidateRoutedExpertTypes(Model(layer), 0));
        Assert.Contains("blk.0.ffn_gate_exps.weight (Q4_K) and blk.0.ffn_up_exps.weight (Q5_K) differ", error.Message);
    }

    [Fact]
    public void DrafterStagesAreAlwaysGpuResident()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ2_XS, GgmlTensorType.IQ2_XS));
        m.Dspark = new Dsv4CudaEngine.DsparkDesc
        {
            Stages = new[] { Layer(0, GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ2_XS, "mtp.0.") },
        };
        var error = Assert.Throws<NotSupportedException>(() => Dsv4CudaEngine.ValidateRoutedExpertTypes(m, int.MaxValue));
        Assert.Contains("cannot run 1 routed-expert tensor(s)", error.Message);
        Assert.Contains("mtp.0.ffn_down_exps.weight is IQ2_XS (GPU-resident)", error.Message);
    }

    [Fact]
    public void LongListsNameTheFirstTensorsAndCountTheRest()
    {
        var layers = Enumerable.Range(0, 40).Select(il => Layer(il, GgmlTensorType.IQ1_S, GgmlTensorType.IQ2_S)).ToArray();
        var error = Assert.Throws<NotSupportedException>(() => Dsv4CudaEngine.ValidateRoutedExpertTypes(Model(layers), 0));
        Assert.Contains("cannot run 120 routed-expert tensor(s)", error.Message);
        Assert.Contains("blk.1.ffn_up_exps.weight is IQ1_S", error.Message);
        Assert.Contains("; and 114 more", error.Message);
        Assert.DoesNotContain("blk.2.", error.Message);
    }

    /// <summary>An explicit --n-cpu-moe is checked before the engine touches CUDA, so this refusal
    /// needs no device.</summary>
    [Fact]
    public void TheEngineRefusesBeforeAnyDeviceWorkWithAnExplicitOffload()
    {
        var m = Model(Layer(0, GgmlTensorType.IQ1_S, GgmlTensorType.Q2_K));
        var error = Assert.Throws<NotSupportedException>(() => new Dsv4CudaEngine(m, 1, 0));
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ1_S (GPU-resident)", error.Message);
    }

    [Fact]
    public void TypeNamesMatchGgml()
    {
        foreach (GgmlTensorType type in Enum.GetValues<GgmlTensorType>().Where(t => (int)t < 100))
            Assert.Equal(type.ToString(), Dsv4CudaEngine.GgmlTypeName((int)type));
        Assert.Equal("type 99", Dsv4CudaEngine.GgmlTypeName(99));
    }
}
