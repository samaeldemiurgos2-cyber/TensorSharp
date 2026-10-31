// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.RegularExpressions;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The direct-CUDA DeepSeek engine once its devices exist: the routed-expert check against the kernel
/// module those devices actually loaded, the refusal a model too big for the GPUs gets, and the expert
/// route its forwards actually run. The first two need a GPU but no weights (descriptor-only models,
/// refused before a byte is uploaded); the last runs the V4.1 fixture (see <see cref="Dsv41CudaSlotTests"/>).
/// </summary>
public sealed class Dsv4CudaEngineLoadTests
{
    private const string FixtureVar = "TS_TEST_DSV41_CUDA_FIXTURE_DIR";
    private const string Fixture = "deepseek41-fixture";
    private const int Embd = 512, Ff = 256, Experts = 8;

    private readonly ITestOutputHelper _output;

    public Dsv4CudaEngineLoadTests(ITestOutputHelper output) => _output = output;

    /// <summary>A model of routed experts only, nothing else to upload. Each layer's experts claim
    /// <paramref name="expertRowBytes"/> per row: bytes the layer split prices and nothing allocates.</summary>
    private static Dsv4CudaEngine.ModelDesc Model(IEnumerable<(GgmlTensorType GateUp, GgmlTensorType Down)> layers,
        long expertRowBytes = 0)
    {
        CudaWeightDesc Expert(GgmlTensorType type, string name, int ne0, int ne1)
            => new() { GgmlType = (int)type, Name = name, Ne0 = ne0, Ne1 = ne1, Ne2 = Experts, RowBytes = expertRowBytes };
        var descs = layers.Select((l, il) => new Dsv4CudaEngine.LayerDesc
        {
            GateExps = Expert(l.GateUp, $"blk.{il}.ffn_gate_exps.weight", Embd, Ff),
            UpExps = Expert(l.GateUp, $"blk.{il}.ffn_up_exps.weight", Embd, Ff),
            DownExps = Expert(l.Down, $"blk.{il}.ffn_down_exps.weight", Ff, Embd),
        }).ToArray();
        return new Dsv4CudaEngine.ModelDesc
        {
            NLayer = descs.Length,
            Layers = descs,
            HeadDim = 512,
            NExpertUsed = 6,
            HostMatMul = DeepSeek4CudaExecutor.HostMatMul.Instance,
        };
    }

    /// <summary>The committed tensorsharp_dsv4_kernels PTX (what a machine without nvcc loads) without the
    /// low-bit IQ marker: a module built from kernel source older than the IQ1_S and IQ2_S decoders, which
    /// reads those rows as Q8_0. (Today's baseline has no marker to remove; once it is regenerated with
    /// one, removing it keeps this the stale module.)</summary>
    internal static byte[] ModuleWithoutLowBitIqDecoders()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        string path = null;
        for (; dir != null && path == null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "TensorSharp.Backends.Cuda", "native", "ptx", "tensorsharp_dsv4_kernels.ptx");
            if (File.Exists(candidate))
                path = candidate;
        }
        Assert.True(path != null, "the committed tensorsharp_dsv4_kernels.ptx is not above " + AppContext.BaseDirectory);
        string ptx = Regex.Replace(File.ReadAllText(path),
            @"\.visible\s+\.entry\s+ts_dsv4_caps_lowbit_iq_experts\s*\(\s*\)\s*\{[^}]*\}", string.Empty);
        Assert.DoesNotContain("ts_dsv4_caps_lowbit_iq_experts(", ptx);
        return Encoding.ASCII.GetBytes(ptx);
    }

    /// <summary>
    /// The engine checks the routed experts against the kernel module its devices loaded, not against the
    /// kernel source: with a module older than the IQ1_S and IQ2_S decoders, a resident IQ1_S layer is
    /// refused by name, both with an explicit offload count (checked as soon as the devices have loaded the
    /// module) and with the automatic one (checked once the split has chosen it), before anything is
    /// uploaded. IQ2_XXS, which such a module still decodes on its warp-per-row kernels, passes that check
    /// and goes on to the layer split. Before, nothing exercised the engine's use of the loaded module's
    /// markers: every stale-module test passed the markers in by hand.
    /// </summary>
    [CudaFact]
    public void TheEngine_ChecksTheExpertsAgainstTheModuleItLoaded()
    {
        byte[] stale = ModuleWithoutLowBitIqDecoders();
        var loaded = new List<Dsv4Kernels>();
        Dsv4CudaEngine.KernelModuleLoader = () =>
        {
            Dsv4Kernels kernels = Dsv4Kernels.FromModule(CudaModule.LoadFromBytes(stale));
            loaded.Add(kernels);
            return kernels;
        };
        try
        {
            foreach (int nCpuMoe in new[] { 0, -1 })
            {
                var m = Model(new[] { (GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS), (GgmlTensorType.IQ1_S, GgmlTensorType.IQ2_S) });
                var error = Assert.Throws<NotSupportedException>(() => new Dsv4CudaEngine(m, 1, nCpuMoe));
                Assert.StartsWith("[dsv4-cuda] blk.1.ffn_gate_exps.weight is IQ1_S, and the loaded tensorsharp_dsv4_kernels " +
                    "module was built from kernel source older than its IQ1_S expert decoder", error.Message);
                Assert.True(ModelLoadRefusal.TryDescribe(error, out _));
            }

            // The same module takes IQ2_XXS: the load gets as far as the split, which refuses this one's
            // 16 GiB of experts per layer, 256 GiB, on any one GPU.
            var iq2xxs = Model(Enumerable.Repeat((GgmlTensorType.IQ2_XXS, GgmlTensorType.IQ3_XXS), 16), expertRowBytes: 1L << 21);
            var vram = Assert.Throws<InvalidOperationException>(() => new Dsv4CudaEngine(iq2xxs, 1, 0));
            Assert.StartsWith("[dsv4-cuda] not enough VRAM", vram.Message);

            Assert.Equal(3, loaded.Count);
            Assert.All(loaded, k => Assert.False(k.DecodesLowBitIqExperts));
            Assert.All(loaded, k => Assert.True(k.DecodesQ5_1Experts));
        }
        finally
        {
            Dsv4CudaEngine.KernelModuleLoader = null;
        }
    }

    /// <summary>
    /// A model the GPUs cannot hold is refused with the offload that would make it fit only when the host
    /// matmul runs the experts that offload would move. The 1017's gate/up experts are IQ1_S, IQ2_S or
    /// IQ2_XXS in every layer, none of which it runs, so the refusal names the first such tensor and the
    /// ways that can work instead; it used to advise <c>--n-cpu-moe N</c> or <c>--cpu-moe</c>, which the
    /// engine then refused by that same tensor. Here the 1017's layouts at 8 GiB of experts per layer, 320
    /// GiB, against one GPU.
    /// </summary>
    [CudaFact]
    public void TheVramRefusal_OffersNoOffloadTheHostCannotRun()
    {
        var gateUp = Enumerable.Repeat(GgmlTensorType.IQ1_S, 10).Concat(Enumerable.Repeat(GgmlTensorType.IQ2_S, 15))
            .Concat(Enumerable.Repeat(GgmlTensorType.IQ2_XXS, 15));
        var down = Enumerable.Repeat(GgmlTensorType.IQ2_S, 15).Concat(Enumerable.Repeat(GgmlTensorType.IQ3_XXS, 15))
            .Concat(Enumerable.Repeat(GgmlTensorType.IQ2_XS, 6)).Concat(Enumerable.Repeat(GgmlTensorType.IQ2_XXS, 3))
            .Append(GgmlTensorType.Q2_K);
        var m = Model(gateUp.Zip(down), expertRowBytes: 1L << 20);
        var error = Assert.Throws<InvalidOperationException>(() => new Dsv4CudaEngine(m, 1, 0));
        _output.WriteLine(error.Message);
        Assert.StartsWith("[dsv4-cuda] not enough VRAM: 320.0 GiB of weights", error.Message.Replace(',', '.'));
        Assert.Contains("Offloading routed experts cannot make it fit", error.Message);
        Assert.Contains("blk.0.ffn_gate_exps.weight is IQ1_S, which the host matmul behind --n-cpu-moe does not implement",
            error.Message);
        Assert.Contains("--backend ggml_cuda", error.Message);
        Assert.DoesNotContain("Re-run with", error.Message);
        Assert.DoesNotContain("--cpu-moe", error.Message);
    }

    // ---------------------------------------------------------------- forced expert routes (fixture)

    private static int[] Stream(int seed, int count)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => rng.Next(3, 256)).ToArray();
    }

    private static float[] Forward(DeepSeek4CudaExecutor ex, int[] tokens)
    {
        var logits = new float[ex.VocabSize];
        ex.Forward(tokens, logits);
        return logits;
    }

    private static double RelativeL2(float[] actual, float[] expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - (double)expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>
    /// The test-only route override reaches the routed-expert blocks of a real forward: forced, every GPU
    /// expert block of a 24-token prefill and two decode steps runs on that route and no other
    /// (<see cref="Dsv4CudaEngine.MoeRouteRuns"/>), on the V4.1 fixture's Q8_0 experts at width 256, which
    /// every route takes. Unforced, the prefill takes the grouped route the module picks (tensor cores, or
    /// register-staged without their body) and the decode steps the per-slot one, and a rerun reproduces it
    /// bit for bit; forced to that grouped route, the prefill is the unforced one bit for bit. The per-slot,
    /// warp-per-row and register-staged routes all read q8_1 activations and decode each Q8_0 row alike, so
    /// they give one set of logits to the bit. The tensor cores read F16 activations instead, a rounding
    /// difference these random 5-layer fixtures (top-2 sparse attention selection, hyper-connection mixing)
    /// carry a long way: 9.5e-2 relative L2 on the Q8_0 fixture's prefill logits and 5e-1 on its dense-mix
    /// variant, whose argmax moves; so the two families are not held to each other here, which is the
    /// parity gate's job on a fixture built for it. A module without the tensor-core body refuses the forced
    /// tensor-core route rather than running another.
    /// </summary>
    [CudaFact(FixtureVar, Fixture)]
    public void AForcedExpertRoute_IsTheRouteEveryExpertBlockRuns()
    {
        using var ex = new DeepSeek4CudaExecutor(TestGates.FindGguf(Environment.GetEnvironmentVariable(FixtureVar), Fixture),
            maxContext: 1024, nUbatch: 32, nGpu: 1);
        Dsv4CudaEngine engine = ex.Engine;
        int[] prompt = Stream(41, 24);
        int[] next = Stream(42, 2);

        (float[][] Logits, int[] Runs) Run(CudaMoe.ExpertRoute? route)
        {
            int slot = ex.SlotAlloc();
            Assert.True(slot >= 0);
            Assert.True(ex.SetActiveSlot(slot));
            int[] before = engine.MoeRouteRuns.ToArray();
            Dsv4CudaEngine.ForceMoeRoute = route;
            try
            {
                var rows = new List<float[]> { Forward(ex, prompt) };
                foreach (int token in next)
                    rows.Add(Forward(ex, new[] { token }));
                return (rows.ToArray(), engine.MoeRouteRuns.Zip(before, (after, was) => after - was).ToArray());
            }
            finally
            {
                Dsv4CudaEngine.ForceMoeRoute = null;
                Assert.True(ex.SetActiveSlot(0));
                Assert.True(ex.SlotFree(slot));
            }
        }

        var (want, defaultRuns) = Run(null);
        int blocks = defaultRuns.Sum() / (1 + next.Length);
        Assert.True(blocks > 0, "the fixture ran no GPU expert block");
        Assert.Equal(blocks * next.Length, defaultRuns[(int)CudaMoe.ExpertRoute.PerSlot]);
        CudaMoe.ExpertRoute grouped = defaultRuns[(int)CudaMoe.ExpertRoute.TensorCore] == blocks
            ? CudaMoe.ExpertRoute.TensorCore : CudaMoe.ExpertRoute.Staged;
        Assert.Equal(blocks, defaultRuns[(int)grouped]);
        _output.WriteLine($"unforced: {blocks} expert blocks per forward; prefill on {grouped}, decode on PerSlot");
        var (again, _) = Run(null);
        for (int i = 0; i < want.Length; i++)
            Assert.Equal(want[i], again[i]);

        var results = new Dictionary<CudaMoe.ExpertRoute, float[][]>();
        foreach (CudaMoe.ExpertRoute route in Enum.GetValues<CudaMoe.ExpertRoute>())
        {
            if (route == CudaMoe.ExpertRoute.TensorCore && grouped != CudaMoe.ExpertRoute.TensorCore)
            {
                var error = Assert.Throws<InvalidOperationException>(() => Run(route));
                Assert.Contains("ForceMoeRoute = TensorCore cannot run here", error.Message);
                _output.WriteLine("TensorCore: refused (the module has no tensor-core body)");
                continue;
            }
            var (got, runs) = Run(route);
            results[route] = got;
            for (int r = 0; r < runs.Length; r++)
                Assert.True(runs[r] == (r == (int)route ? blocks * (1 + next.Length) : 0),
                    $"forced {route}: {runs[r]} block(s) ran on {(CudaMoe.ExpertRoute)r}");
            Assert.All(got, row => Assert.All(row, v => Assert.True(float.IsFinite(v), $"forced {route}: a non-finite logit")));
            for (int i = 0; i < want.Length; i++)
                _output.WriteLine($"{route}: forward {i} relative L2 {RelativeL2(got[i], want[i]):E3} against the unforced routes");
        }

        Assert.Equal(want[0], results[grouped][0]);
        float[][] perSlot = results[CudaMoe.ExpertRoute.PerSlot];
        foreach (CudaMoe.ExpertRoute route in new[] { CudaMoe.ExpertRoute.Warp, CudaMoe.ExpertRoute.Staged })
            for (int i = 0; i < want.Length; i++)
                Assert.True(perSlot[i].SequenceEqual(results[route][i]),
                    $"forced {route}: forward {i} is {RelativeL2(results[route][i], perSlot[i]):E3} from forced PerSlot");
    }
}
