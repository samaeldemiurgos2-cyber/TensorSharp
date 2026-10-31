// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// DeepSeek V4.1 on the direct-CUDA engine (<c>--backend cuda</c>) held to the native executor, on the
/// 1017-shaped checkpoint of <see cref="Dsv41SyntheticModelBuilder"/>: llama.cpp's combined Engram layout
/// and every routed-expert layout of unsloth's DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS (IQ1_S, IQ2_S, IQ2_XXS
/// gate/up; IQ2_S, IQ3_XXS, IQ2_XS, IQ2_XXS, Q2_K down), its Q6_K/Q8_0 attention, Q5_K/Q8_0 shared experts
/// and BF16 compressor and indexer, with a prompt long enough for the sparse selection and the candidate
/// mask to engage. The oracles are <c>ggml_cuda</c> (the engine the 1017 is served on today) and
/// <c>ggml_cpu</c> (whose kernels share nothing with ggml-cuda's), both with exact prefill
/// (<c>TS_DSV41_YOCO_KEEP=0</c>, the only prefill the direct engine has).
///
/// <para>The engines read the same quantized bytes and differ in their kernels and in how they round
/// activations for them (q8_1 for ggml's MMQ and the direct engine's decode kernels, F16 for its
/// tensor-core experts and dequantized dense GEMMs, q8_K on ggml-cpu), so the logit gate is a relative L2
/// of 2e-2 per logit vector with the same argmax: the bound the other direct engines are held to. The
/// fixture keeps every block's update small but the routed experts' (see the builder), so the logits see
/// the last row of each call, and the attention path, only coarsely. The stages are therefore also held
/// to the oracle's own trace of them (TS_DSV41_TRACE_DIR against TS_DSV4_CUDA_TRACE_DIR), row by row:
/// every routed-expert block's output on every row of every call, on each forced route; the indexer's
/// visible rows, scores and selection and the compressor's latents at every index and cache source; the
/// Engram stage.</para>
///
/// <para>The native oracles need the process to pin their GGML backend (<c>TS_TEST_GGML_BACKEND=cuda</c>
/// for ggml_cuda, the default lane for ggml_cpu); everything else needs only a CUDA device. These tests
/// set engine variables (<c>TS_DSV4_UBATCH</c>, <c>TS_DSV41_ENGRAM_DEVICE</c>, the trace directories) and
/// the engine's route override, so they run alone. <c>TS_TEST_DSV41_TRACE_OUT</c> keeps the traces in
/// that directory (emptied first) instead of the process's temp directory.</para>
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class Dsv41CudaEngineParityTests : IDisposable
{
    private const int Context = 1024, Ubatch = 64;
    private const double LogitBound = 2e-2;

    /// <summary>The raw ids the Engram token map compresses to the pad (declared first: the streams use them).</summary>
    private static readonly int[] RawPadIds = Enumerable.Range(0, Dsv41SyntheticModelBuilder.Vocab)
        .Where(t => Dsv41SyntheticModelBuilder.TokenMap[t] == Dsv41SyntheticModelBuilder.CompressedPad).ToArray();

    /// <summary>150 tokens at a ubatch of 64: 64 + 64 + 22, the last also past the 16-row per-slot bound,
    /// so every prefill ubatch runs the grouped expert routes. Ratio 1 and 2 caches then hold 150 and 75
    /// rows, past the indexer's 16 and the candidate mask's 4 blocks of 8. The prompt carries both raw ids
    /// the Engram token map sends to the compressed pad (1 at 70, 201 at 131), and a decode step and the
    /// append one each, so a real token that hashes as padding is in the gate as well as the lookback
    /// padding at positions 0-1.</summary>
    private static readonly int[] Prompt = Tokens(1017, 150, 70, 131);
    private static readonly int[] DecodeTokens = Tokens(1018, 17, 4);
    private static readonly int[] Append = Tokens(1019, 21, 9);

    private readonly ITestOutputHelper _output;
    private readonly NativeEnvScope _env = new();
    private readonly EnvScope _speculation = new();
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-dsv41-parity-" + Guid.NewGuid().ToString("N"));

    public Dsv41CudaEngineParityTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", Context.ToString());
        _env.Set("TS_DSV4_UBATCH", Ubatch.ToString());
        _env.Set("TS_DSV41_YOCO_KEEP", "0");
        foreach (string name in new[] { "TS_DSV41_ENGRAM_DEVICE", "TS_DSV4_MMAP", "TS_DSV41_TRACE_DIR", "TS_DSV4_CUDA_TRACE_DIR" })
            _env.Set(name, null);
        _speculation.ClearSpeculationVars();
    }

    public void Dispose()
    {
        Dsv4CudaEngine.ForceMoeRoute = null;
        _speculation.Dispose();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>A reproducible stream over the fixture's vocabulary, with the raw ids the Engram token map
    /// compresses to the pad (alternately) at <paramref name="padsAt"/>.</summary>
    private static int[] Tokens(int seed, int count, params int[] padsAt)
    {
        var rng = new Random(seed);
        int[] tokens = Enumerable.Range(0, count).Select(_ => rng.Next(0, Dsv41SyntheticModelBuilder.Vocab)).ToArray();
        for (int i = 0; i < padsAt.Length; i++)
            tokens[padsAt[i]] = RawPadIds[i % RawPadIds.Length];
        return tokens;
    }

    // ---------------------------------------------------------------- the script every engine runs

    /// <summary>One forward call: its name, what it feeds, and whether a reset comes first.</summary>
    private readonly record struct Call(string Phase, int[] Tokens, bool Reset = false);

    /// <summary>The prompt; 15 decode steps; an append of 21 tokens at the odd position 165, so a multi-row
    /// call completes the ratio-2 block (164, 165) and the candidate block 160..167 that earlier calls
    /// began, as a multi-turn append after an odd-length history does; 2 more decode steps; then a reset,
    /// the prompt again and the first 3 decode steps again. One logit vector per call.</summary>
    private static readonly Call[] Calls = BuildCalls();

    private const int DecodesBeforeAppend = 15, Redecodes = 3;

    private static Call[] BuildCalls()
    {
        var calls = new List<Call> { new("prefill", Prompt) };
        for (int i = 0; i < DecodeTokens.Length; i++)
        {
            if (i == DecodesBeforeAppend)
                calls.Add(new("append", Append));
            calls.Add(new($"decode {i + 1}", new[] { DecodeTokens[i] }));
        }
        calls.Add(new("re-prefill", Prompt, Reset: true));
        for (int i = 0; i < Redecodes; i++)
            calls.Add(new($"re-decode {i + 1}", new[] { DecodeTokens[i] }));
        return calls.ToArray();
    }

    /// <summary>Each graph of the script before its reset, as both engines split it: (first position, rows).
    /// The calls after the reset repeat these positions and must answer to the bit as the first ones did,
    /// so the traces they rewrite hold the same values.</summary>
    private static readonly (int P0, int Rows)[] Graphs = BuildGraphs();

    private static (int P0, int Rows)[] BuildGraphs()
    {
        var graphs = new List<(int P0, int Rows)>();
        int p = 0;
        foreach (Call call in Calls.TakeWhile(c => !c.Reset))
        {
            for (int off = 0; off < call.Tokens.Length; off += Ubatch)
                graphs.Add((p + off, Math.Min(Ubatch, call.Tokens.Length - off)));
            p += call.Tokens.Length;
        }
        return graphs.ToArray();
    }

    private static float[][] Script(Func<int[], float[]> forward, Action reset)
    {
        var rows = new float[Calls.Length][];
        for (int i = 0; i < Calls.Length; i++)
        {
            if (Calls[i].Reset)
                reset();
            rows[i] = forward(Calls[i].Tokens);
        }
        return rows;
    }

    private static float[][] Native(string path, BackendType backend)
    {
        using ModelBase model = ModelBase.Create(path, backend);
        return Script(tokens => (float[])model.Forward(tokens).Clone(), model.ResetKVCache);
    }

    private static readonly Dictionary<BackendType, float[][]> NativeRuns = new();

    /// <summary>The native script on the combined fixture, once per backend per process.</summary>
    private static float[][] NativeReference(BackendType backend)
    {
        lock (NativeRuns)
        {
            if (!NativeRuns.TryGetValue(backend, out float[][] rows))
                NativeRuns[backend] = rows = Native(Dsv41SyntheticModelBuilder.Fixture(), backend);
            return rows;
        }
    }

    private static float[] Forward(DeepSeek4CudaExecutor ex, int[] tokens)
    {
        var logits = new float[ex.VocabSize];
        ex.Forward(tokens, logits);
        return logits;
    }

    /// <summary>The script on the direct engine, with every routed-expert block forced onto
    /// <paramref name="route"/> (null: the engine's own choice) and its stages traced into
    /// <paramref name="traceDir"/> (emptied first) when given; also how many blocks ran on each route.</summary>
    private (float[][] Logits, int[] Runs, bool EngramResident) Direct(string path, CudaMoe.ExpertRoute? route = null,
        string traceDir = null)
    {
        if (traceDir != null)
        {
            EmptyDirectory(traceDir);
            _env.Set("TS_DSV4_CUDA_TRACE_DIR", traceDir);
        }
        try
        {
            using var ex = new DeepSeek4CudaExecutor(path, Context, Ubatch, nGpu: 1);
            Dsv4CudaEngine.ForceMoeRoute = route;
            float[][] rows = Script(tokens => Forward(ex, tokens), ex.Reset);
            return (rows, ex.Engine.MoeRouteRuns.ToArray(), ex.Engine.EngramResident);
        }
        finally
        {
            Dsv4CudaEngine.ForceMoeRoute = null;
            if (traceDir != null)
                _env.Set("TS_DSV4_CUDA_TRACE_DIR", null);
        }
    }

    // ---------------------------------------------------------------- traces

    /// <summary>Where the traces go: TS_TEST_DSV41_TRACE_OUT, or a directory of this process's. Each engine's
    /// run empties its own subdirectory before it writes, and <see cref="StageTrace"/> deletes it once read
    /// (the process's exit deletes what a failed run left).</summary>
    private static readonly Lazy<(string Root, bool Keep)> TraceRoot = new(() =>
    {
        string keep = Environment.GetEnvironmentVariable("TS_TEST_DSV41_TRACE_OUT");
        if (!string.IsNullOrEmpty(keep))
            return (keep, true);
        string root = Path.Combine(Path.GetTempPath(), $"ts-dsv41-traces-{Environment.ProcessId}-{Guid.NewGuid():N}");
        AppDomain.CurrentDomain.ProcessExit += (_, _) =>
        {
            try { Directory.Delete(root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        };
        return (root, false);
    });

    private static string TraceDir(string engine) => Path.Combine(TraceRoot.Value.Root, engine);

    private static void EmptyDirectory(string dir)
    {
        if (Directory.Exists(dir))
            Directory.Delete(dir, recursive: true);
        Directory.CreateDirectory(dir);
    }

    private static readonly Dictionary<BackendType, StageTrace> NativeTraces = new();

    /// <summary>The native script on the combined fixture with TS_DSV41_TRACE_DIR set, once per backend per
    /// process: its trace.</summary>
    private StageTrace NativeTrace(BackendType backend)
    {
        lock (NativeTraces)
        {
            if (NativeTraces.TryGetValue(backend, out StageTrace trace))
                return trace;
            string dir = TraceDir(backend == BackendType.GgmlCuda ? "ggml_cuda" : "ggml_cpu");
            EmptyDirectory(dir);
            _env.Set("TS_DSV41_TRACE_DIR", dir);
            try { Native(Dsv41SyntheticModelBuilder.Fixture(), backend); }
            finally { _env.Set("TS_DSV41_TRACE_DIR", null); }
            return NativeTraces[backend] = new StageTrace(dir);
        }
    }

    private static readonly object DirectTraceLock = new();
    private static (StageTrace Trace, float[][] Logits)? _directTrace;

    /// <summary>The direct engine's script on the combined fixture on its own routes, traced, once per
    /// process: its trace, and its logits.</summary>
    private (StageTrace Trace, float[][] Logits) DirectTrace()
    {
        lock (DirectTraceLock)
        {
            if (_directTrace == null)
            {
                string dir = TraceDir("cuda");
                float[][] logits = Direct(Dsv41SyntheticModelBuilder.Fixture(), traceDir: dir).Logits;
                _directTrace = (new StageTrace(dir), logits);
            }
            return _directTrace.Value;
        }
    }

    /// <summary>
    /// One engine's trace of the script: the stages the checks read (the direct engine's
    /// p&lt;p0&gt;_blk&lt;NN&gt;_&lt;stage&gt; files or the native executor's p&lt;p0&gt;_v41.&lt;NN&gt;.&lt;stage&gt; ones,
    /// not its views), read into memory as soon as the run ends; the directory is deleted then, unless
    /// TS_TEST_DSV41_TRACE_OUT keeps it.
    /// </summary>
    private sealed class StageTrace
    {
        private static readonly Regex FileName = new(@"^p(\d+)_(?:blk(\d+)_|v41\.(\d+)\.)(.+)\.(?:f32|i32)$");
        private static readonly HashSet<string> Stages = new()
        {
            "attn_input", "attn_out", "attn_output", "ffn_out", "ffn_output", "hidden", "output", "engram_kv",
            "engram_output", "latent", "index_scores", "index_top", "index_count", "router", "experts",
        };
        private readonly Dictionary<(int P0, int Layer, string Stage), byte[]> _stages = new();
        private readonly string _dir;

        public StageTrace(string dir)
        {
            _dir = dir;
            foreach (string path in Directory.GetFiles(dir))
            {
                Match m = FileName.Match(Path.GetFileName(path));
                if (!m.Success || !Stages.Contains(m.Groups[4].Value))
                    continue;
                int layer = int.Parse((m.Groups[2].Success ? m.Groups[2] : m.Groups[3]).Value);
                _stages[(int.Parse(m.Groups[1].Value), layer, m.Groups[4].Value)] = File.ReadAllBytes(path);
            }
            if (!TraceRoot.Value.Keep)
                Directory.Delete(dir, recursive: true);
        }

        public bool Has(int p0, int layer, string stage) => _stages.ContainsKey((p0, layer, stage));

        public float[] F32(int p0, int layer, string stage)
        {
            byte[] bytes = Bytes(p0, layer, stage);
            var values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        public int[] I32(int p0, int layer, string stage)
        {
            byte[] bytes = Bytes(p0, layer, stage);
            var values = new int[bytes.Length / sizeof(int)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }

        private byte[] Bytes(int p0, int layer, string stage)
        {
            Assert.True(_stages.TryGetValue((p0, layer, stage), out byte[] bytes), $"{_dir} holds no p{p0} layer {layer} {stage}");
            return bytes;
        }
    }

    /// <summary>|got - want| over |want| for each of the rows of <paramref name="width"/> values both hold
    /// (every value finite on both sides).</summary>
    private static double[] RowErrors(float[] got, float[] want, int width, string what)
    {
        Assert.True(want.Length == got.Length && want.Length % width == 0,
            $"{what}: {got.Length} values against the reference's {want.Length} (rows of {width})");
        var errors = new double[want.Length / width];
        for (int row = 0; row < errors.Length; row++)
        {
            double diff = 0, norm = 0;
            for (int i = row * width; i < (row + 1) * width; i++)
            {
                Assert.True(float.IsFinite(got[i]) && float.IsFinite(want[i]), $"{what}: value {i} is {got[i]} against {want[i]}");
                double d = got[i] - (double)want[i];
                diff += d * d;
                norm += (double)want[i] * want[i];
            }
            errors[row] = Math.Sqrt(diff / Math.Max(norm, 1e-30));
        }
        return errors;
    }

    private readonly record struct Gap(double RelativeL2, double MaxAbs, int Argmax, int ReferenceArgmax);

    private static Gap Compare(float[] got, float[] want)
    {
        double num = 0, den = 0, maxAbs = 0;
        for (int i = 0; i < want.Length; i++)
        {
            // Both sides: a NaN in the reference would make the relative L2 NaN, which no bound fails.
            Assert.True(float.IsFinite(got[i]) && float.IsFinite(want[i]), $"logit {i} is {got[i]} against {want[i]}");
            double d = got[i] - (double)want[i];
            num += d * d;
            den += (double)want[i] * want[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
        }
        return new Gap(Math.Sqrt(num / Math.Max(den, 1e-30)), maxAbs, ArgMax(got), ArgMax(want));
    }

    private static int ArgMax(float[] v) => Array.IndexOf(v, v.Max());

    private static bool Same(float[] a, float[] b) => MemoryMarshal.AsBytes(a.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(b.AsSpan()));

    /// <summary>Every vector within <paramref name="bound"/> relative L2 of the reference, with its argmax;
    /// logs each one, adds what strays to <paramref name="failures"/> and returns the worst relative L2 and
    /// |dlogit|.</summary>
    private (double RelativeL2, double MaxAbs) LogitParity(string what, float[][] got, float[][] want, double bound,
        List<string> failures)
    {
        Assert.Equal(want.Length, got.Length);
        double worst = 0, worstAbs = 0;
        for (int i = 0; i < want.Length; i++)
        {
            Gap gap = Compare(got[i], want[i]);
            worst = Math.Max(worst, gap.RelativeL2);
            worstAbs = Math.Max(worstAbs, gap.MaxAbs);
            _output.WriteLine($"{what} {Calls[i].Phase,-11} relative L2 {gap.RelativeL2:E3}  max |dlogit| {gap.MaxAbs:E3}  " +
                $"argmax {gap.Argmax}/{gap.ReferenceArgmax}");
            if (gap.RelativeL2 > bound || gap.Argmax != gap.ReferenceArgmax)
                failures.Add($"{what} {Calls[i].Phase}: relative L2 {gap.RelativeL2:E3}, argmax {gap.Argmax} vs {gap.ReferenceArgmax}");
        }
        return (worst, worstAbs);
    }

    private void AssertParity(string what, float[][] got, float[][] want, double bound)
    {
        var failures = new List<string>();
        var (worst, worstAbs) = LogitParity(what, got, want, bound, failures);
        _output.WriteLine($"[dsv41-parity] {what}: worst relative L2 {worst:E3}, max |dlogit| {worstAbs:E3}");
        Assert.True(failures.Count == 0, $"{what} strayed past {bound:E1} or changed the argmax: {string.Join("; ", failures)}");
    }

    /// <summary>The calls after the reset answer to the bit as the calls they repeat.</summary>
    private static void AssertTheResetReproduces(string what, float[][] rows)
    {
        for (int i = 0; i < Calls.Length; i++)
        {
            if (!Calls[i].Phase.StartsWith("re-", StringComparison.Ordinal))
                continue;
            int first = Array.FindIndex(Calls, c => c.Phase == Calls[i].Phase[3..]);
            Assert.True(Same(rows[i], rows[first]), $"{what}: {Calls[i].Phase} after a reset does not reproduce {Calls[first].Phase}");
        }
    }

    // ---------------------------------------------------------------- (a), (b): every expert route vs ggml

    /// <summary>
    /// Each routed-expert route against ggml_cuda: the tensor-core kernels (F16 tiles), the register-staged
    /// and warp-per-row grouped kernels, and the per-slot decode kernels, each forced on every block of the
    /// prefill ubatches, the append and the decode steps. The register-staged route is what a machine
    /// without nvcc runs in place of the tensor cores (the committed PTX has no sm_80 body), so it is held
    /// to the same bounds. A reset must also reproduce the prompt's and the first decode steps' answers to
    /// the bit.
    /// <list type="bullet">
    /// <item>The logits: every vector within 2e-2 relative L2 and with the same argmax.</item>
    /// <item>Every routed-expert block's output (ffn_out against ggml's v41.NN.ffn_output: the routed
    /// experts and the shared one), on every row of every call and every layer (see <see cref="ExpertRows"/>).
    /// The logits see only each call's last row, and the other rows only through attention's small update;
    /// a decoder wrong on some of an expert's members (past the first pass of a busy expert, say) or on one
    /// layer's layout is wrong by a whole expert's share of these rows.</item>
    /// </list>
    /// Measured on an RTX 3080 (sm_86): every route within 8.2e-3 of ggml_cuda's logits and 9.7e-3 of
    /// ggml_cpu's; every expert block row within 4.5e-2 of ggml_cuda's and 2.5e-2 of ggml_cpu's but one routing
    /// flip against each (router margins 2.6e-3 to 5.2e-3) and, against ggml_cpu, the 6% that flip leaves in its
    /// token's later layers. Dropping the members of a busy expert past its sixteenth in layers 3-5 on the
    /// grouped routes leaves the logits within the bound and fails here.
    /// </summary>
    [CudaTheory(GgmlBackend = BackendType.GgmlCuda)]
    [TensorCoreInlineData("tensorsharp_dsv4_kernels", "TensorCore")]
    [InlineData("Staged")]
    [InlineData("Warp")]
    [InlineData("PerSlot")]
    public void EveryExpertRoute_AnswersAsGgmlCuda(string route) => Parity(BackendType.GgmlCuda, Enum.Parse<CudaMoe.ExpertRoute>(route));

    /// <summary>The same against ggml_cpu, an oracle whose expert and dense kernels (and q8_K activation
    /// rounding) are independent of ggml-cuda's.</summary>
    [CudaTheory(GgmlBackend = BackendType.GgmlCpu)]
    [TensorCoreInlineData("tensorsharp_dsv4_kernels", "TensorCore")]
    [InlineData("Staged")]
    [InlineData("Warp")]
    [InlineData("PerSlot")]
    public void EveryExpertRoute_AnswersAsGgmlCpu(string route) => Parity(BackendType.GgmlCpu, Enum.Parse<CudaMoe.ExpertRoute>(route));

    private void Parity(BackendType backend, CudaMoe.ExpertRoute route)
    {
        float[][] reference = NativeReference(backend);
        StageTrace native = NativeTrace(backend);
        string dir = TraceDir($"cuda-{route}");
        var (rows, runs, _) = Direct(Dsv41SyntheticModelBuilder.Fixture(), route, dir);
        var traces = new StageTrace(dir);

        // Every GPU expert block ran on the forced route and no other: each layer of each ubatch.
        int ubatches = Calls.Sum(c => (c.Tokens.Length + Ubatch - 1) / Ubatch);
        int blocks = Dsv41SyntheticModelBuilder.Layers * ubatches;
        for (int r = 0; r < runs.Length; r++)
            Assert.True(runs[r] == (r == (int)route ? blocks : 0), $"forced {route}: {runs[r]} block(s) ran on {(CudaMoe.ExpertRoute)r}");
        AssertTheResetReproduces(route.ToString(), rows);

        var failures = new List<string>();
        var (worst, worstAbs) = LogitParity($"cuda/{route} vs {backend}", rows, reference, LogitBound, failures);
        _output.WriteLine($"[dsv41-parity] {route,-10} vs {backend}: worst relative L2 {worst:E3}, max |dlogit| {worstAbs:E3}" +
            (failures.Count == 0 ? " (every vector within the bound)" : $" ({failures.Count} vector(s) past the bound)"));

        ExpertRows($"{route,-10} vs {backend}", traces, native, failures);
        Assert.True(failures.Count == 0, $"cuda/{route} strayed from {backend}: {string.Join("; ", failures.Take(20))}" +
            (failures.Count > 20 ? $" (and {failures.Count - 20} more)" : ""));
    }

    /// <summary>
    /// Every row of every routed-expert block against the oracle's, and the experts the direct engine's
    /// router chose against the top four of its own scores. A row may stray <see cref="ExpertRowBound"/>
    /// past <see cref="StreamGain"/> times the error of the stream it was given (the layer's attention input,
    /// which attention's small update leaves the block's input to within its rounding): a block adds no more
    /// than that to what it is handed. A row past it is a routing flip only where the direct engine's own
    /// fourth and fifth choices stood within <see cref="RouterTieMargin"/>: the two engines' streams differ
    /// by a few 1e-3 before every block, which settles such a near tie either way, and the row then sums
    /// another expert (and carries a stream a few percent off into the later layers, which the stream term
    /// covers). No more than <see cref="RouterFlipShare"/> of the rows may be flips.
    /// </summary>
    private void ExpertRows(string what, StageTrace direct, StageTrace native, List<string> failures)
    {
        float[][] bias = RouterBias.Value;
        int experts = Dsv41SyntheticModelBuilder.Experts, used = Dsv41SyntheticModelBuilder.ExpertsUsed;
        int rows = 0, biasDecides = 0, nearTies = 0;
        double minMargin = double.MaxValue;
        var flips = new List<string>();
        for (int il = 0; il < Dsv41SyntheticModelBuilder.Layers; il++)
        {
            double layerWorst = 0;
            string layerAt = null;
            Dictionary<int, double> stream = StreamErrors(direct, native, il);
            foreach (var (p0, n) in Graphs)
            {
                double[] errors = RowErrors(direct.F32(p0, il, "ffn_out"), native.F32(p0, il, "ffn_output"),
                    Dsv41SyntheticModelBuilder.Hidden, $"layer {il} p{p0} ffn_out");
                float[] logits = direct.F32(p0, il, "router");
                int[] chosen = direct.I32(p0, il, "experts");
                Assert.True(logits.Length == n * experts && chosen.Length == n * used, $"layer {il} p{p0}: router trace sizes");
                for (int t = 0; t < n; t++)
                {
                    rows++;
                    double[] scores = SelectionScores(logits, t, experts, bias[il]);
                    int[] ranked = Enumerable.Range(0, experts).OrderByDescending(e => scores[e]).ThenBy(e => e).ToArray();
                    double margin = scores[ranked[used - 1]] - scores[ranked[used]];
                    minMargin = Math.Min(minMargin, margin);
                    if (margin <= RouterTieMargin)
                        nearTies++;
                    var top = ranked.Take(used).ToHashSet();
                    // The kernel ranks in float, this in double: only a tie within float rounding may differ.
                    if (!top.SetEquals(chosen.Skip(t * used).Take(used)) && margin > 1e-5)
                        failures.Add($"layer {il} p{p0 + t}: the router chose [{string.Join(",", chosen.Skip(t * used).Take(used))}], " +
                            $"not the top {used} of its own scores [{string.Join(",", top)}]");
                    double[] unbiased = SelectionScores(logits, t, experts, null);
                    if (!top.SetEquals(Enumerable.Range(0, experts).OrderByDescending(e => unbiased[e]).ThenBy(e => e).Take(used)))
                        biasDecides++;
                    double allowed = ExpertRowBound + StreamGain * stream[p0 + t];
                    if (errors[t] <= allowed)
                    {
                        if (errors[t] > layerWorst) { layerWorst = errors[t]; layerAt = $"p{p0 + t}"; }
                    }
                    else if (margin <= RouterTieMargin)
                        flips.Add($"layer {il} p{p0 + t} {errors[t]:E2} (router margin {margin:E2})");
                    else
                        failures.Add($"layer {il} p{p0 + t}: expert block {errors[t]:E3} > {allowed:E2} (router margin {margin:E2})");
                }
            }
            _output.WriteLine($"[dsv41-parity] {what}: layer {il} expert block, worst row {layerWorst:E3} at {layerAt}");
        }
        _output.WriteLine($"[dsv41-parity] {what}: {rows} expert block rows; routing flips: {flips.Count} " +
            $"[{string.Join("; ", flips)}] of {nearTies} near ties; smallest router margin {minMargin:E2}; rows the selection bias decides: {biasDecides}");
        if (flips.Count > rows * RouterFlipShare)
            failures.Add($"{flips.Count} routing flips in {rows} rows: more than near ties explain");
    }

    private const double ExpertRowBound = 1e-1, RouterTieMargin = 2e-2, RouterFlipShare = 1e-2;

    /// <summary>How much of the error of the stream a stage is given its output may carry, on top of the
    /// stage's own bound.</summary>
    private const double StreamGain = 2;

    /// <summary>The error of the stream entering layer <paramref name="layer"/> (its attention input, per
    /// row, against the oracle's), by position, over the graphs of the script.</summary>
    private static Dictionary<int, double> StreamErrors(StageTrace direct, StageTrace native, int layer)
    {
        var errors = new Dictionary<int, double>();
        foreach (var (p0, n) in Graphs)
        {
            double[] rows = RowErrors(direct.F32(p0, layer, "attn_input"), native.F32(p0, layer, "attn_input"),
                Dsv41SyntheticModelBuilder.Hidden, $"layer {layer} p{p0} attn_input");
            for (int t = 0; t < n; t++)
                errors[p0 + t] = rows[t];
        }
        return errors;
    }

    /// <summary>What DeepSeek V4's router ranks the experts by: sqrt(softplus(logit)) plus the selection bias,
    /// which picks and does not weigh (null: the scores alone).</summary>
    private static double[] SelectionScores(float[] logits, int row, int experts, float[] bias)
    {
        var scores = new double[experts];
        for (int e = 0; e < experts; e++)
        {
            double x = logits[row * experts + e];
            scores[e] = Math.Sqrt(x > 20 ? x : Math.Log(1 + Math.Exp(x))) + (bias?[e] ?? 0);
        }
        return scores;
    }

    /// <summary>The fixture's exp_probs_b, per layer.</summary>
    private static readonly Lazy<float[][]> RouterBias = new(() =>
    {
        using var file = GgufFile.OpenWithoutSiblingShards(Dsv41SyntheticModelBuilder.Fixture());
        return Enumerable.Range(0, Dsv41SyntheticModelBuilder.Layers).Select(il =>
        {
            byte[] bytes = file.ReadTensorData(file.Tensors[$"blk.{il}.exp_probs_b.bias"]);
            var values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
            return values;
        }).ToArray();
    });

    // ---------------------------------------------------------------- (c) the layout adapter, exactly

    /// <summary>
    /// The combined layout (views of one per_layer_token_embd in the second shard, engram_kv, the fused
    /// engram_gate, a raw pad the token map compresses) and TensorSharp's per-layer layout of the same
    /// weights (per-layer tables, engram_wkv, engram_q = the gate, engram_k = ones, the compressed pad)
    /// give the direct engine the same bytes, so every logit is the same to the bit. So is the traced run
    /// the stage tests read: tracing only copies the stages out.
    /// </summary>
    [CudaFact]
    public void TheCombinedAndPerLayerLayouts_AnswerIdentically()
    {
        float[][] combined = Direct(Dsv41SyntheticModelBuilder.Fixture(Dsv41SyntheticModelBuilder.Layout.Combined)).Logits;
        float[][] perLayer = Direct(Dsv41SyntheticModelBuilder.Fixture(Dsv41SyntheticModelBuilder.Layout.PerLayer)).Logits;
        float[][] traced = DirectTrace().Logits;
        for (int i = 0; i < combined.Length; i++)
        {
            Assert.True(Same(combined[i], perLayer[i]),
                $"{Calls[i].Phase}: the combined layout's logits differ from the per-layer layout's ({Compare(combined[i], perLayer[i]).RelativeL2:E3})");
            Assert.True(Same(combined[i], traced[i]), $"{Calls[i].Phase}: tracing the stages changed the logits");
        }
    }

    // ---------------------------------------------------------------- (d) the Engram stage, traced

    /// <summary>
    /// The Engram stage at both Engram layers against ggml_cuda's own trace of it, at every graph of the
    /// script (the prefill ubatches, the decode steps, the append). The stage's output
    /// (v41.NN.engram_output, the hidden streams after the gated update) also carries every earlier
    /// layer's rounding: measured per token, its error is the error of the streams it was given to three
    /// digits, since the update is under 2% of the stream. So the stage is held to ggml where it is the
    /// stage alone:
    /// <list type="bullet">
    /// <item>engram_kv (the gathered Q5_K rows through the Q6_K projection): at the decode steps, where both
    /// engines feed it q8_1 activations, within <see cref="EngramKvDecodeBound"/> (measured 2.0e-7: the same
    /// rows, decoded and projected alike); at the multi-row calls, where the direct engine's dequantized GEMM
    /// reads F16 activations against ggml's q8_1, within <see cref="EngramKvPrefillBound"/> (measured
    /// 5.4e-3). A wrong row (a hash, pad, offset or view error) is wrong by its whole size.</item>
    /// <item>the update each engine applied (its engram_output minus its own input, the previous layer's
    /// output), per token, relative to the stream: within <see cref="EngramUpdateDecodeBound"/> at the
    /// decode steps (measured 2.2e-5) and <see cref="EngramUpdatePrefillBound"/> at the multi-row calls,
    /// whose engram_kv differs more (measured 4.8e-4). A wrong gate or value, or a token reading another
    /// token's key or value, moves it by a few percent of the stream.</item>
    /// <item>engram_output itself within the logit bound per token, as a sanity check (measured
    /// 3.1e-3).</item>
    /// </list>
    /// </summary>
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void TheEngramStage_TracesAsGgmlCuda()
    {
        StageTrace native = NativeTrace(BackendType.GgmlCuda), direct = DirectTrace().Trace;
        var failures = new List<string>();
        void Check(string what, double value, double bound)
        {
            _output.WriteLine($"{what,-44} {value:E3} (bound {bound:E0})");
            if (!(value <= bound))
                failures.Add($"{what} {value:E3} > {bound:E0}");
        }

        int width = Dsv41SyntheticModelBuilder.Hc * Dsv41SyntheticModelBuilder.Hidden;
        foreach (int layer in Dsv41SyntheticModelBuilder.EngramLayers)
        foreach (var (p0, rows) in Graphs)
        {
            bool decode = rows == 1;
            string at = $"layer {layer} p{p0}";
            Check($"{at} engram_kv", MaxRelative(direct.F32(p0, layer, "engram_kv"), native.F32(p0, layer, "engram_kv")),
                decode ? EngramKvDecodeBound : EngramKvPrefillBound);
            float[] output = direct.F32(p0, layer, "engram_output"), nativeOutput = native.F32(p0, layer, "engram_output");
            float[] input = direct.F32(p0, layer - 1, "hidden"), nativeInput = native.F32(p0, layer - 1, "output");
            Assert.True(output.Length == rows * width && nativeOutput.Length == output.Length
                && input.Length == output.Length && nativeInput.Length == output.Length, $"{at}: trace sizes");
            double update = 0, stream = 0;
            for (int t = 0; t < rows; t++)
            {
                double diff = 0, outDiff = 0, norm = 0;
                for (int i = t * width; i < (t + 1) * width; i++)
                {
                    double d = (output[i] - input[i]) - (double)(nativeOutput[i] - nativeInput[i]);
                    double o = output[i] - (double)nativeOutput[i];
                    diff += d * d;
                    outDiff += o * o;
                    norm += (double)nativeInput[i] * nativeInput[i];
                }
                update = Math.Max(update, Math.Sqrt(diff / norm));
                stream = Math.Max(stream, Math.Sqrt(outDiff / norm));
            }
            Check($"{at} engram update / stream", update, decode ? EngramUpdateDecodeBound : EngramUpdatePrefillBound);
            Check($"{at} engram_output per token", stream, LogitBound);
        }
        Assert.True(failures.Count == 0, "the Engram stage strayed from ggml_cuda's: " + string.Join("; ", failures));
    }

    private const double EngramKvDecodeBound = 1e-4, EngramKvPrefillBound = 2e-2;
    private const double EngramUpdateDecodeBound = 1e-4, EngramUpdatePrefillBound = 2e-3;

    /// <summary>max |a - b| over max |b|, as compare-dsv4-traces.py reports it.</summary>
    private static double MaxRelative(float[] got, float[] want)
    {
        Assert.Equal(want.Length, got.Length);
        double error = 0, scale = 0;
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]) && float.IsFinite(want[i]), $"value {i} is {got[i]} against {want[i]}");
            error = Math.Max(error, Math.Abs(got[i] - (double)want[i]));
            scale = Math.Max(scale, Math.Abs((double)want[i]));
        }
        return error / Math.Max(scale, 1e-30);
    }

    // ---------------------------------------------------------------- the sparse attention, traced

    /// <summary>
    /// The V4.1 sparse attention against the oracle's trace of it, at every graph of the script. The
    /// fixture's attention writes a small update (see the builder), so the logits cannot see a selection or
    /// a candidate mask that is applied but wrong. Nor can the two engines be asked to select the same rows:
    /// the indexer's queries and keys are rounded to the trained 4-bit cache format, where a 1e-3 change of
    /// an input moves a value a whole step, so the engines' scores differ by about 6% (relative L2, per
    /// token) and their selections by a row or two on about half of the tokens. What both engines must do
    /// exactly is the arithmetic on their own scores; that, the scores, and the attention over a selection
    /// both share are held to the oracle:
    /// <list type="bullet">
    /// <item>the compressor's latents (v41.NN.latent: one normalized row per block a call completes, the
    /// block (164, 165) the append completes included) at each cache source, per row, within
    /// <see cref="LatentRowBound"/> past <see cref="StreamGain"/> times the error of the streams it
    /// compresses (a token whose routing flipped carries a stream a few percent off);</item>
    /// <item>each index source scores every visible row, and one after the candidate layer only the rows
    /// <see cref="CandidateRows"/> leaves it from the candidate layer's scores of the same token; and each
    /// selects <see cref="SelectedRows"/> of its scores. Both engines, each on its own scores: ggml's trace is
    /// what shows the rules are its arithmetic;</item>
    /// <item>the scores against the oracle's, per token, within <see cref="IndexScoreTokenBound"/> relative
    /// L2 and <see cref="IndexScoreMeanBound"/> on average per layer; the selections' overlap with the
    /// oracle's at least <see cref="SelectionOverlapFloor"/> on average;</item>
    /// <item>the attention output (attn_out against v41.NN.attn_output) within <see cref="AttnRowBound"/>
    /// past the same share of its input's error, per row, on every row whose selection is the oracle's
    /// (every row of the window-only layers): there the
    /// two engines attend over the same rows, so a consumer layer reading the wrong cache, a gather off by a
    /// row or a mask the kernel ignores is wrong by the attended rows' share.</item>
    /// </list>
    /// </summary>
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void TheSparseAttention_SelectsAndAttendsAsGgmlCuda() => SparseAttention(BackendType.GgmlCuda);

    /// <summary>The same against ggml_cpu's trace.</summary>
    [CudaFact(GgmlBackend = BackendType.GgmlCpu)]
    public void TheSparseAttention_SelectsAndAttendsAsGgmlCpu() => SparseAttention(BackendType.GgmlCpu);

    private const double LatentRowBound = 1e-2, IndexScoreTokenBound = 0.35, IndexScoreMeanBound = 0.12;
    private const double SelectionOverlapFloor = 0.85, AttnRowBound = 6e-2;

    private void SparseAttention(BackendType backend)
    {
        StageTrace native = NativeTrace(backend), direct = DirectTrace().Trace;
        var failures = new List<string>();
        int k = Dsv41SyntheticModelBuilder.IndexerTopK, head = Dsv41SyntheticModelBuilder.HeadDim;
        int[] ratios = Dsv41SyntheticModelBuilder.Ratios;

        // The compressor, per source and row.
        foreach (int source in Dsv41SyntheticModelBuilder.KvSources)
        {
            int ratio = ratios[source];
            Dictionary<int, double> stream = StreamErrors(direct, native, source);
            double worst = 0;
            int latents = 0;
            foreach (var (p0, rows) in Graphs)
            {
                bool d = direct.Has(p0, source, "latent"), n = native.Has(p0, source, "latent");
                if (d != n)
                {
                    failures.Add($"layer {source} p{p0}: {(n ? "only ggml" : "only the direct engine")} completed a compressed block");
                    continue;
                }
                if (!d)
                    continue;
                double[] errors = RowErrors(direct.F32(p0, source, "latent"), native.F32(p0, source, "latent"), head, $"layer {source} p{p0} latent");
                // The blocks this call completes, from the one holding p0: (p0 + 1) / ratio rounded up, minus one.
                int first = (p0 + ratio) / ratio - 1;
                Assert.True((p0 + rows) / ratio - first == errors.Length, $"layer {source} p{p0}: {errors.Length} latents");
                for (int j = 0; j < errors.Length; j++)
                {
                    int block = first + j;
                    double given = Enumerable.Range(block * ratio, ratio).Max(p => stream[p]);
                    double allowed = LatentRowBound + StreamGain * given;
                    worst = Math.Max(worst, errors[j]);
                    if (!(errors[j] <= allowed))
                        failures.Add($"layer {source} block {block} (p{block * ratio}..{block * ratio + ratio - 1}): latent {errors[j]:E3} > {allowed:E2}");
                }
                latents += errors.Length;
            }
            _output.WriteLine($"[dsv41-parity] {backend}: layer {source} compressor, {latents} latents, worst row {worst:E3}");
        }

        // The indexer, per source and token: the rows scored, the selection, the scores.
        var agrees = new Dictionary<(int Layer, int Pos), bool>();
        foreach (int source in Dsv41SyntheticModelBuilder.IndexSources)
        {
            int ratio = ratios[source], candidate = Dsv41SyntheticModelBuilder.CandidateSource;
            bool pruned = source > candidate;
            double worstScore = 0, sumScore = 0, sumOverlap = 0;
            int tokens = 0, selecting = 0, same = 0;
            foreach (var (p0, n) in Graphs)
            {
                float[] ns = native.F32(p0, source, "index_scores"), ds = direct.F32(p0, source, "index_scores");
                int[] nTop = native.I32(p0, source, "index_top"), dTop = direct.I32(p0, source, "index_top");
                int[] dCount = direct.I32(p0, source, "index_count");
                float[] nCand = pruned ? native.F32(p0, candidate, "index_scores") : null;
                float[] dCand = pruned ? direct.F32(p0, candidate, "index_scores") : null;
                int nw = ns.Length / n, nk = nTop.Length / n, dw = ds.Length / n, dk = dTop.Length / n;
                for (int t = 0; t < n; t++)
                {
                    int pos = p0 + t, visible = (pos + 1) / ratio;
                    string at = $"layer {source} p{pos}";
                    var nScored = Enumerable.Range(0, nw).Where(j => float.IsFinite(ns[t * nw + j])).ToHashSet();
                    var dScored = Enumerable.Range(0, visible).Where(j => float.IsFinite(ds[t * dw + j])).ToHashSet();
                    var all = Enumerable.Range(0, visible).ToHashSet();
                    var nWant = pruned ? CandidateRows(nCand, t * (nCand.Length / n), pos, visible) : all;
                    var dWant = pruned ? CandidateRows(dCand, t * (dCand.Length / n), pos, visible) : all;
                    if (!nScored.SetEquals(nWant))
                        failures.Add($"{at}: ggml scored {nScored.Count} rows, not the {nWant.Count} the rule leaves it");
                    if (!dScored.SetEquals(dWant))
                        failures.Add($"{at}: the direct engine scored {dScored.Count} rows, not the {dWant.Count} its candidate layer leaves it");

                    var nSelected = nTop.Skip(t * nk).Take(nk).Where(j => j < nw && float.IsFinite(ns[t * nw + j])).ToHashSet();
                    var dSelected = dTop.Skip(t * dk).Take(dCount[t]).ToHashSet();
                    if (!nSelected.SetEquals(SelectedRows(ns, t * nw, visible)))
                        failures.Add($"{at}: ggml's selection is not the top {k} of its scores");
                    if (!dSelected.SetEquals(SelectedRows(ds, t * dw, visible)))
                        failures.Add($"{at}: the direct engine selected [{string.Join(",", dSelected.Order())}], not the top {k} of its own scores");

                    double diff = 0, norm = 0;
                    foreach (int j in nScored.Intersect(dScored))
                    {
                        double d = ds[t * dw + j] - (double)ns[t * nw + j];
                        diff += d * d;
                        norm += (double)ns[t * nw + j] * ns[t * nw + j];
                    }
                    double score = Math.Sqrt(diff / Math.Max(norm, 1e-30));
                    worstScore = Math.Max(worstScore, score);
                    sumScore += score;
                    tokens++;
                    if (!(score <= IndexScoreTokenBound))
                        failures.Add($"{at}: scores {score:E3} from ggml's > {IndexScoreTokenBound}");
                    bool agree = dSelected.SetEquals(nSelected);
                    agrees[(source, pos)] = agree;
                    if (agree)
                        same++;
                    if (visible > k)
                    {
                        selecting++;
                        sumOverlap += (double)dSelected.Intersect(nSelected).Count() / nSelected.Count;
                    }
                }
            }
            double meanScore = sumScore / tokens, overlap = sumOverlap / Math.Max(selecting, 1);
            _output.WriteLine($"[dsv41-parity] {backend}: layer {source} indexer, {tokens} tokens: scores within {worstScore:E3} " +
                $"(mean {meanScore:E3}) of ggml's; {selecting} select, overlapping ggml's {overlap:F3} on average; {same} select as ggml does");
            if (!(meanScore <= IndexScoreMeanBound))
                failures.Add($"layer {source} scores {meanScore:E3} from ggml's on average > {IndexScoreMeanBound}");
            if (selecting == 0 || !(overlap >= SelectionOverlapFloor))
                failures.Add($"layer {source} selections overlap ggml's {overlap:F3} on average < {SelectionOverlapFloor}");
        }

        // The attention, on the rows that attend over the oracle's own selection.
        for (int il = 0; il < Dsv41SyntheticModelBuilder.Layers; il++)
        {
            int layer = il;
            int source = ratios[il] == 0 ? -1
                : Dsv41SyntheticModelBuilder.IndexSources.Where(s => s <= layer && ratios[s] == ratios[layer]).Max();
            Dictionary<int, double> stream = StreamErrors(direct, native, il);
            double worst = 0;
            int compared = 0, rows = 0;
            foreach (var (p0, n) in Graphs)
            {
                double[] errors = RowErrors(direct.F32(p0, il, "attn_out"), native.F32(p0, il, "attn_output"),
                    Dsv41SyntheticModelBuilder.Hidden, $"layer {il} p{p0} attn_out");
                for (int t = 0; t < n; t++, rows++)
                {
                    if (source >= 0 && !agrees[(source, p0 + t)])
                        continue;
                    compared++;
                    worst = Math.Max(worst, errors[t]);
                    double allowed = AttnRowBound + StreamGain * stream[p0 + t];
                    if (!(errors[t] <= allowed))
                        failures.Add($"layer {il} p{p0 + t}: attention {errors[t]:E3} from ggml's over the same selection > {allowed:E2}");
                }
            }
            _output.WriteLine($"[dsv41-parity] {backend}: layer {il} attention, {compared} of {rows} rows over ggml's selection: worst {worst:E3}");
            if (compared * 4 < rows)
                failures.Add($"layer {il}: only {compared} of {rows} rows attend over ggml's selection");
        }
        Assert.True(failures.Count == 0, $"the sparse attention strayed from {backend}'s: {string.Join("; ", failures.Take(20))}" +
            (failures.Count > 20 ? $" (and {failures.Count - 20} more)" : ""));
    }

    /// <summary>The rows an index source selects from its scores of one token (at <paramref name="offset"/>):
    /// the indexer's top-k of the scored (finite) visible rows, the higher score first, then the lower row;
    /// every scored row when there are no more than k.</summary>
    private static HashSet<int> SelectedRows(float[] scores, int offset, int visible)
        => Enumerable.Range(0, visible).Where(j => float.IsFinite(scores[offset + j]))
            .OrderByDescending(j => scores[offset + j]).ThenBy(j => j)
            .Take(Dsv41SyntheticModelBuilder.IndexerTopK).ToHashSet();

    /// <summary>The rows the candidate layer leaves the index sources after it, for the token at
    /// <paramref name="pos"/>, from its own scores of that token (at <paramref name="offset"/>): blocks of
    /// CandidateBlockSize rows ranked by their best visible score, the block holding the token's position
    /// first, then the higher score, then the lower block; the visible rows of the best
    /// CandidateTopkBlocks.</summary>
    private static HashSet<int> CandidateRows(float[] scores, int offset, int pos, int visible)
    {
        int block = Dsv41SyntheticModelBuilder.CandidateBlockSize;
        int blocks = (visible + block - 1) / block, pinned = pos / block;
        var best = new double[blocks];
        for (int b = 0; b < blocks; b++)
        {
            best[b] = double.NegativeInfinity;
            for (int j = b * block; j < Math.Min(visible, (b + 1) * block); j++)
                best[b] = Math.Max(best[b], scores[offset + j]);
            if (b == pinned)
                best[b] = double.PositiveInfinity;
        }
        return Enumerable.Range(0, blocks).Where(b => !double.IsNegativeInfinity(best[b]))
            .OrderByDescending(b => best[b]).ThenBy(b => b).Take(Dsv41SyntheticModelBuilder.CandidateTopkBlocks)
            .SelectMany(b => Enumerable.Range(b * block, Math.Min(visible, (b + 1) * block) - b * block)).ToHashSet();
    }

    // ---------------------------------------------------------------- (f) the host-mapped Engram tables

    /// <summary>
    /// The 1017 on 46 GB A40s cannot hold its 62.9 GiB Engram slices in VRAM, so it runs the host-mapped
    /// path: each slice is a row view of the combined table in its shard (MapEngramTable), the executor
    /// dequantizes the Q5_K rows a token selects on the host (GatherEngramRows) and they cross the bus. Forced
    /// with TS_DSV41_ENGRAM_DEVICE=0 on the combined fixture, it answers as ggml_cuda does, and to the bit as
    /// the device-resident tables do: the host's Q5_K row decode and the device gather's give the same rows.
    /// </summary>
    [CudaFact(GgmlBackend = BackendType.GgmlCuda)]
    public void TheHostMappedEngramPath_AnswersAsGgmlCudaAndAsTheDeviceTables()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture();
        var (device, _, resident) = Direct(path);
        Assert.True(resident, "the fixture's tables should fit in VRAM");
        _env.Set("TS_DSV41_ENGRAM_DEVICE", "0");
        var (host, _, hostResident) = Direct(path);
        _env.Set("TS_DSV41_ENGRAM_DEVICE", null);
        Assert.False(hostResident, "TS_DSV41_ENGRAM_DEVICE=0 left the tables in VRAM");

        AssertParity("host-mapped Engram vs ggml_cuda", host, NativeReference(BackendType.GgmlCuda), LogitBound);
        for (int i = 0; i < host.Length; i++)
            Assert.True(Same(host[i], device[i]),
                $"{Calls[i].Phase}: host-mapped Engram answers {Compare(host[i], device[i]).RelativeL2:E3} away from the device tables");
    }

    // ---------------------------------------------------------------- (f) the pure C# executor's buffered path

    /// <summary>
    /// TS_DSV4_MMAP=0 copies every tensor the pure C# executor reads into RAM. On llama.cpp's layout that
    /// must be the per-layer row views of the combined table, never the combined table as well: on the 1017
    /// that would be its 126 GiB twice. The buffered run answers exactly as the mapped one.
    /// </summary>
    [Fact]
    public void TheBufferedCpuExecutor_ReadsTheRowViews_NeverTheWholeTable()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture();
        int[] prompt = Prompt[..40];
        float[][] Run(string mmap, out IReadOnlyDictionary<string, long> buffered)
        {
            _env.Set("TS_DSV4_MMAP", mmap);
            using var ex = new DeepSeek4CpuExecutor(path, maxContext: 256, nUbatch: 32, nThreads: 8, new CpuAllocator(BlasEnum.DotNet));
            buffered = ex.BufferedTensorBytes();
            var rows = new List<float[]>();
            var logits = new float[ex.VocabSize];
            ex.Forward(prompt, logits);
            rows.Add((float[])logits.Clone());
            foreach (int token in DecodeTokens[..3])
            {
                ex.Forward(new[] { token }, logits);
                rows.Add((float[])logits.Clone());
            }
            return rows.ToArray();
        }
        float[][] mapped = Run("1", out var none);
        float[][] copied = Run("0", out var buffered);
        _env.Set("TS_DSV4_MMAP", null);
        Assert.Empty(none);
        for (int i = 0; i < mapped.Length; i++)
            Assert.True(Same(mapped[i], copied[i]), $"call {i}: the buffered executor answers differently from the mapped one");

        // What the files hold, against what was buffered: every tensor once, the combined table only as its views.
        var shards = Directory.GetFiles(Path.GetDirectoryName(path), "*.gguf");
        long fileBytes = 0, tableBytes = 0;
        foreach (string shard in shards)
        {
            using var file = GgufFile.OpenWithoutSiblingShards(shard);
            foreach (var (name, info) in file.Tensors)
            {
                long bytes = file.GetTensorByteCount(info);
                fileBytes += bytes;
                if (name == Dsv41EngramData.CombinedTableName)
                    tableBytes = bytes;
            }
        }
        Assert.True(tableBytes > 0);
        Assert.DoesNotContain(Dsv41EngramData.CombinedTableName, buffered.Keys);
        const long Q5KRow = 176;
        long viewBytes = 0;
        for (int e = 0; e < Dsv41SyntheticModelBuilder.EngramLayers.Length; e++)
        {
            string view = Dsv41EngramData.TableName(Dsv41SyntheticModelBuilder.EngramLayers[e]);
            Assert.True(buffered.TryGetValue(view, out long bytes), view + " was not buffered");
            viewBytes += bytes;
        }
        Assert.Equal(tableBytes, viewBytes);
        Assert.Equal(fileBytes, buffered.Values.Sum());
        Assert.Equal(0, viewBytes % Q5KRow);
        _output.WriteLine($"[dsv41-parity] TS_DSV4_MMAP=0 buffered {buffered.Count} tensors, {buffered.Values.Sum()} bytes: " +
            $"the table's {tableBytes} once, as {Dsv41SyntheticModelBuilder.EngramLayers.Length} row views");
    }

    // ---------------------------------------------------------------- the fixture reaches what it is for

    /// <summary>
    /// The prompt runs the sparse selection and the candidate mask, which is what the tests above claim to
    /// cover: with an indexer top-k above every cache's rows (no selection) or without the candidate budget
    /// (the loaders' 2048 blocks, no pruning), the prompt's logits change (attention writes a small update
    /// into this fixture, see the builder: the traced test above holds the selection and the mask to ggml);
    /// a prompt too short to reach either answers identically in all three.
    /// </summary>
    [CudaFact]
    public void ThePrompt_RunsTheSparseSelectionAndTheCandidateMask()
    {
        float[] Prefill(string path, int[] tokens)
        {
            using var ex = new DeepSeek4CudaExecutor(path, Context, Ubatch, nGpu: 1);
            return Forward(ex, tokens);
        }
        string fixture = Dsv41SyntheticModelBuilder.Fixture();
        string dense = Dsv41SyntheticModelBuilder.Variant(new Dsv41SyntheticModelBuilder.Options { IndexerTopK = Context });
        string unpruned = Dsv41SyntheticModelBuilder.Variant(new Dsv41SyntheticModelBuilder.Options { CandidateTopkBlocks = null });

        // 12 tokens: 6 ratio-2 and 12 ratio-1 rows, under the top-k of 16 and the 4 x 8 candidate rows.
        int[] shortPrompt = Prompt[..12];
        float[] shortLogits = Prefill(fixture, shortPrompt);
        Assert.True(Same(shortLogits, Prefill(dense, shortPrompt)), "a prompt below the top-k already selects");
        Assert.True(Same(shortLogits, Prefill(unpruned, shortPrompt)), "a prompt below the budget is already pruned");

        float[] logits = Prefill(fixture, Prompt);
        double selection = Compare(Prefill(dense, Prompt), logits).RelativeL2;
        double pruning = Compare(Prefill(unpruned, Prompt), logits).RelativeL2;
        _output.WriteLine($"[dsv41-parity] removing the selection moves the prompt's logits {selection:E3}, removing the candidate mask {pruning:E3}");
        Assert.True(selection > 0, "the prompt never reaches the indexer's top-k");
        Assert.True(pruning > 0, "the prompt never reaches the candidate mask");
    }
}
