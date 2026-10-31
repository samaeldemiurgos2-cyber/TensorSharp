// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// The direct-CUDA DeepSeek engine's sequence slots: the per-request caches, rewind checkpoints
/// and batched decode that let <c>--backend cuda</c> serve parallel conversations and keep each
/// one's KV between turns, as the native ggml executor does.
///
/// <para>The GPU tests run twice: on the 1017-shaped checkpoint <see cref="Dsv41SyntheticModelBuilder"/>
/// writes (llama.cpp's combined Engram layout, the 1017's IQ1_S / IQ2_S / IQ2_XXS experts, Q5_K shared
/// experts and BF16 compressor and indexer, whose dense projections are what batched decode has to keep
/// row-invariant), and on the V4.1 fixture generated for this engine
/// (<c>python eng/dsv41-fixture.py DIR --cuda-attn --q8</c>, then
/// <c>TS_TEST_DSV41_CUDA_FIXTURE_DIR=DIR</c>; that row skips without it). They compare the engine
/// with itself on a fresh slot, so the reference does not depend on the fixture's quantization.
/// "Exactly" is held to 1e-5 of the largest logit rather than to the last bit, which no kernel
/// promises; a wrong cache row moves logits by orders of magnitude more.</para>
/// </summary>
public sealed class Dsv41CudaSlotTests
{
    private const string FixtureVar = "TS_TEST_DSV41_CUDA_FIXTURE_DIR";
    private const string Fixture = "deepseek41-fixture";

    // ---------------------------------------------------------------- the truncation planner

    /// <summary>
    /// Every rewind <see cref="Dsv4CudaEngine.PlanTruncate"/> allows reads only rows that hold what
    /// the attention mask believes they hold. Checked against a simulation of the raw ring and the
    /// mask rather than the planner's own arithmetic (the native dsv41_truncate_test.cpp, ported).
    /// </summary>
    [Fact]
    public void ThePlanner_AllowsOnlyRewindsWhoseReadsAreSound_AndNeedlessRefusalsAreNone()
    {
        int allowed = 0, refused = 0;
        foreach ((long ring, long window) in new[] { (512L, 128L), (1280L, 128L), (256L, 8L), (512L, 512L) })
        {
            long span = ring - window + 1;
            foreach (long align in new long[] { 1, 2, 128 })
            foreach (long nPast in new[] { span / 2 + 1, span, span + 1, 3 * ring + 7, 9 * ring })
            {
                if (nPast <= 0) continue;
                foreach (long cp in new[] { -1, 0, nPast / 3, nPast - span - 1, nPast - span, nPast - 1, nPast })
                {
                    if (cp > nPast) continue;
                    foreach (long target in Targets(nPast, cp, span, ring))
                    {
                        var route = Dsv4CudaEngine.PlanTruncate(target, nPast, cp, span, align);
                        if (route == Dsv4CudaEngine.TruncateRoute.Refuse)
                        {
                            refused++;
                            // A refusal is only needless when the target is aligned and a sound source exists.
                            if (target % align == 0 && target > 0 && target < nPast)
                                Assert.False(Sound(Ring(nPast, ring), target, ring, window)
                                             && nPast - target <= span,
                                    $"refused a live rewind {nPast}->{target} (ring {ring}, window {window})");
                            continue;
                        }
                        allowed++;
                        Assert.True(target % align == 0 || target == 0 || target == nPast);
                        if (route is Dsv4CudaEngine.TruncateRoute.None or Dsv4CudaEngine.TruncateRoute.Reset)
                            continue;
                        long[] slots = route == Dsv4CudaEngine.TruncateRoute.Checkpoint ? Ring(cp, ring) : Ring(nPast, ring);
                        foreach (long nt in new long[] { 1, 2, Math.Max(1, ring - window) })
                            Assert.True(Sound(slots, target, ring, window, nt),
                                $"{route} rewind {nPast}->{target} (cp {cp}, ring {ring}, window {window}, ubatch {nt}) reads a stale row");
                    }
                }
            }
        }
        Assert.True(allowed > 1000 && refused > 1000, $"allowed {allowed}, refused {refused}");
    }

    private static IEnumerable<long> Targets(long nPast, long cp, long span, long ring)
    {
        var t = new SortedSet<long>();
        foreach (long anchor in new[] { 0, cp, nPast, nPast - span, cp - span, ring, 2 * ring })
            for (long d = -2; d <= 2; d++)
                if (anchor + d >= 0 && anchor + d <= nPast) t.Add(anchor + d);
        for (long p = 0; p <= nPast; p += Math.Max(1, nPast / 24)) t.Add(p);
        return t;
    }

    /// <summary>The raw ring after positions [0, filled) were written in order.</summary>
    private static long[] Ring(long filled, long ring)
    {
        var slots = Enumerable.Repeat(-1L, (int)ring).ToArray();
        for (long p = Math.Max(0, filled - ring); p < filled; p++) slots[p % ring] = p;
        return slots;
    }

    /// <summary>After a ubatch of <paramref name="nt"/> tokens at <paramref name="target"/> is
    /// written, every slot the mask treats as visible holds the position it believes.</summary>
    private static bool Sound(long[] ringSlots, long target, long ring, long window, long nt = 1)
    {
        long[] slots = (long[])ringSlots.Clone();
        long last = target + nt - 1;
        for (long p = target; p <= last; p++) slots[p % ring] = p;
        for (long s = 0; s < ring; s++)
        {
            long believed = last - (((last - s) % ring) + ring) % ring;
            if (believed < 0 || believed > last || believed <= target - window) continue;
            if (slots[s] != believed) return false;
        }
        return true;
    }

    // ---------------------------------------------------------------- the engine (GPU)

    /// <summary>The theory rows: the synthetic 1017-shaped fixture, and the one the environment names.</summary>
    private const string Synthetic = "synthetic", FromEnvironment = "environment";

    private static string FixturePath(string fixture) => fixture == Synthetic
        ? Dsv41SyntheticModelBuilder.Fixture()
        : TestGates.FindGguf(Environment.GetEnvironmentVariable(FixtureVar), Fixture);

    private static DeepSeek4CudaExecutor Open(string fixture, int ubatch = 32) =>
        new(FixturePath(fixture), maxContext: 1024, nUbatch: ubatch, nGpu: 1);

    /// <summary>A reproducible token stream over the fixture's ordinary ids.</summary>
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

    private static int NewActiveSlot(DeepSeek4CudaExecutor ex)
    {
        int slot = ex.SlotAlloc();
        Assert.True(slot >= 0);
        Assert.True(ex.SetActiveSlot(slot));
        return slot;
    }

    /// <summary>Two sequences whose forwards alternate on their own slots answer exactly as
    /// each does alone: no cache, position, Engram history or checkpoint leaks across slots.</summary>
    [CudaTheory]
    [InlineData(Synthetic)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment)]
    public void InterleavedSlots_AnswerExactlyAsEachSequenceAlone(string fixture)
    {
        using var ex = Open(fixture);
        int[][] a = { Stream(1, 23), Stream(2, 1), Stream(3, 9), Stream(4, 1), Stream(5, 1) };
        int[][] b = { Stream(6, 5), Stream(7, 31), Stream(8, 1), Stream(9, 4), Stream(10, 1) };

        List<float[]> Alone(int[][] chunks)
        {
            int slot = NewActiveSlot(ex);
            var rows = chunks.Select(c => Forward(ex, c)).ToList();
            Assert.True(ex.SetActiveSlot(0));
            Assert.True(ex.SlotFree(slot));
            return rows;
        }
        List<float[]> expectedA = Alone(a), expectedB = Alone(b);

        int sa = ex.SlotAlloc(), sb = ex.SlotAlloc();
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(ex.SetActiveSlot(sa));
            AssertClose($"sequence A chunk {i}", Forward(ex, a[i]), expectedA[i], Exact);
            Assert.True(ex.SetActiveSlot(sb));
            AssertClose($"sequence B chunk {i}", Forward(ex, b[i]), expectedB[i], Exact);
        }
        Assert.Equal(a.Sum(c => c.Length), StatusHead(ex, sa));
        Assert.Equal(b.Sum(c => c.Length), StatusHead(ex, sb));
        Assert.False(ex.SlotFree(sb), "the active slot must not be freed");
    }

    private static int StatusHead(DeepSeek4CudaExecutor ex, int slot)
    {
        Assert.True(ex.SlotStatus(slot, out int head, out _, out bool healthy));
        Assert.True(healthy);
        return head;
    }

    /// <summary>
    /// A row of a batched decode step does not depend on its batchmates: four sequences decoded
    /// together answer exactly as the same four decoded as two pairs. Both batches take the same
    /// multi-row kernels, so this holds to rounding; it is what catches a row reading another row's
    /// slot, cache rows or position.
    /// </summary>
    [CudaTheory]
    [InlineData(Synthetic)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment)]
    public void BatchedDecode_EveryRowIsIndependentOfItsBatchmates(string fixture)
    {
        using var ex = Open(fixture);
        int[] prompts = { 20, 13, 31, 2 };
        int n = prompts.Length, steps = 12;
        var together = new int[n];
        var paired = new int[n];
        for (int i = 0; i < n; i++)
        {
            together[i] = NewActiveSlot(ex);
            Forward(ex, Stream(100 + i, prompts[i]));
            paired[i] = NewActiveSlot(ex);
            Forward(ex, Stream(100 + i, prompts[i]));
        }

        for (int step = 0; step < steps; step++)
        {
            int[] tokens = Enumerable.Range(0, n).Select(i => Stream(200 + i, steps)[step]).ToArray();
            int[] positions = prompts.Select(p => p + step).ToArray();
            var all = new float[n * ex.VocabSize];
            Assert.True(ex.ForwardBatchedDecode(together, tokens, positions, all));
            for (int pair = 0; pair < n; pair += 2)
            {
                var rows = new float[2 * ex.VocabSize];
                Assert.True(ex.ForwardBatchedDecode(paired[pair..(pair + 2)], tokens[pair..(pair + 2)],
                    positions[pair..(pair + 2)], rows));
                for (int j = 0; j < 2; j++)
                {
                    int i = pair + j;
                    AssertClose($"step {step} sequence {i}", all.AsSpan(i * ex.VocabSize, ex.VocabSize).ToArray(),
                        rows.AsSpan(j * ex.VocabSize, ex.VocabSize).ToArray(), Exact);
                }
            }
        }
        for (int i = 0; i < n; i++)
            Assert.Equal(prompts[i] + steps, StatusHead(ex, together[i]));
    }

    /// <summary>
    /// A batched decode step computes exactly what each sequence's own one-token forward computes, to the
    /// bit, step after step. Every dense projection of a decode-size step takes a row-invariant route (each
    /// row of a 2..16-row call is the one-row call's arithmetic: Q8_0 and Q6_K on their q8_1 row kernels,
    /// Q5_K and the low-bit layouts on DenseQ81Rows, BF16 one row at a time; see DenseMatMul), the experts
    /// run per (token, expert) slot, and attention, the caches and Engram go row by row. So a batched row
    /// equals its own forward exactly, its cache writes do too, and the next step compares like with like.
    /// Measured: identical on both fixtures, whose dense layouts span Q8_0, Q6_K, Q5_K and BF16; before the
    /// row-invariant routes a Q5_K / BF16 fixture's batched rows strayed 0.65 of the largest logit. A row
    /// answering from the wrong slot, position or cache is wrong at once.
    /// </summary>
    [CudaTheory]
    [InlineData(Synthetic)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment)]
    public void BatchedDecode_ComputesEverySequencesOwnForward(string fixture)
    {
        using var ex = Open(fixture);
        int[] prompts = { 20, 13, 31, 2 };
        int n = prompts.Length, steps = 4;
        var batched = new int[n];
        var serial = new int[n];
        for (int i = 0; i < n; i++)
        {
            batched[i] = NewActiveSlot(ex);
            Forward(ex, Stream(100 + i, prompts[i]));
            serial[i] = NewActiveSlot(ex);
            Forward(ex, Stream(100 + i, prompts[i]));
        }

        double worst = 0;
        for (int step = 0; step < steps; step++)
        {
            int[] tokens = Enumerable.Range(0, n).Select(i => Stream(200 + i, steps)[step]).ToArray();
            int[] positions = prompts.Select(p => p + step).ToArray();
            var rows = new float[n * ex.VocabSize];
            Assert.True(ex.ForwardBatchedDecode(batched, tokens, positions, rows));
            for (int i = 0; i < n; i++)
            {
                Assert.True(ex.SetActiveSlot(serial[i]));
                float[] want = Forward(ex, new[] { tokens[i] });
                float[] got = rows.AsSpan(i * ex.VocabSize, ex.VocabSize).ToArray();
                worst = Math.Max(worst, RelativeError(got, want));
                Assert.True(MemoryMarshal.AsBytes(got.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(want.AsSpan())),
                    $"step {step} sequence {i}: the batched row is {RelativeError(got, want):E3} from its own forward");
            }
        }
        Console.WriteLine($"[dsv41-cuda] batched decode vs one-row forwards over {steps} steps: worst relative error {worst:E3}");
    }

    private static double RelativeError(float[] actual, float[] expected)
    {
        double scale = Math.Max(1.0, expected.Max(v => Math.Abs((double)v)));
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
            worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]) / scale);
        return worst;
    }

    /// <summary>
    /// A layer split answers as one device does. V4.1 layers read what earlier layers published: a
    /// ratio group's compressed and indexer caches (written by the group's first layer), the sparse
    /// selection an index source publishes, the candidate mask, and the gates each block collapses
    /// with. Across a split each has to reach the device that reads it; before it did, the published
    /// Q2_K checkpoint faulted on its 6-way split (a group read another device's cache pointer). Five
    /// devices put one layer of the five-layer fixture on each, so every group spans a boundary;
    /// prefill ubatches, single-token steps and a batched step all cross them. A host with fewer
    /// devices runs the splits it can hold, so two GPUs still check a boundary.
    /// </summary>
    [MultiCudaTheory(2)]
    [InlineData(Synthetic)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment)]
    public void ALayerSplit_AnswersAsOneDevice(string fixture)
    {
        int[] prompt = Stream(31, 45), steps = Stream(32, 6), other = Stream(33, 17);
        List<float[]> Run(int gpus)
        {
            using var ex = new DeepSeek4CudaExecutor(FixturePath(fixture), maxContext: 1024, nUbatch: 32, nGpu: gpus);
            var rows = new List<float[]> { Forward(ex, prompt) };
            foreach (int t in steps)
                rows.Add(Forward(ex, new[] { t }));
            int first = ex.ActiveSlot;
            int second = NewActiveSlot(ex);
            rows.Add(Forward(ex, other));
            var batched = new float[2 * ex.VocabSize];
            Assert.True(ex.ForwardBatchedDecode(new[] { first, second }, new[] { 9, 10 },
                new[] { prompt.Length + steps.Length, other.Length }, batched));
            rows.Add(batched);
            return rows;
        }

        List<float[]> single = Run(1);
        int devices = CudaDevice.GetDeviceCount();
        foreach (int gpus in new[] { 2, 3, 5 })
        {
            if (gpus > devices)
                continue;
            List<float[]> split = Run(gpus);
            for (int i = 0; i < single.Count; i++)
                AssertClose($"{gpus} devices, forward {i}", split[i], single[i], Exact);
        }
    }

    /// <summary>A step the engine cannot batch is declined before anything is written.</summary>
    [CudaTheory]
    [InlineData(Synthetic)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment)]
    public void BatchedDecode_DeclinesAStepItCannotServe_WithoutWriting(string fixture)
    {
        using var ex = Open(fixture);
        int s1 = NewActiveSlot(ex);
        Forward(ex, Stream(1, 6));
        int s2 = NewActiveSlot(ex);
        Forward(ex, Stream(2, 4));
        var rows = new float[2 * ex.VocabSize];
        Assert.False(ex.ForwardBatchedDecode(new[] { s1, s1 }, new[] { 5, 6 }, new[] { 6, 6 }, rows), "duplicate slot");
        Assert.False(ex.ForwardBatchedDecode(new[] { s1, s2 }, new[] { 5, 6 }, new[] { 6, 5 }, rows), "wrong position");
        Assert.False(ex.ForwardBatchedDecode(new[] { s1, 999 }, new[] { 5, 6 }, new[] { 6, 4 }, rows), "unknown slot");
        Assert.False(ex.ForwardBatchedDecode(new[] { s1 }, new[] { 5 }, new[] { 6 }, rows), "a single row");
        Assert.Equal(6, StatusHead(ex, s1));
        Assert.Equal(4, StatusHead(ex, s2));
    }

    /// <summary>
    /// A conversational rewind is exact. The answer runs the head far past the raw ring's reach
    /// (<paramref name="answer"/> 260 against a span of 249 positions on the generated fixture's
    /// 8-token window, 241 on the synthetic one's 16), so the rewind to the prompt boundary is
    /// served by the checkpoint the prompt's forward took; a short answer is served by the live
    /// ring. Either way the next turn answers exactly as a fresh prefill.
    /// </summary>
    [CudaTheory]
    [InlineData(Synthetic, 30)]
    [InlineData(Synthetic, 260)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment, 30)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment, 260)]
    public void Truncate_ToThePromptBoundary_ContinuesExactlyAsAFreshPrefill(string fixture, int answer)
    {
        using var ex = Open(fixture);
        int[] prompt = Stream(11, 24), reply = Stream(12, answer), next = Stream(13, 9);
        Assert.Equal(2, ex.TruncateAlign);

        int warm = ex.ActiveSlot;
        Forward(ex, prompt);
        foreach (int t in reply)
            Forward(ex, new[] { t });
        Assert.True(ex.SlotStatus(warm, out int head, out int checkpoint, out _));
        Assert.Equal(prompt.Length + answer, head);
        Assert.Equal(prompt.Length, checkpoint);
        Assert.True(ex.SlotCanReuse(warm, head, prompt.Length));
        Assert.False(ex.Truncate(prompt.Length - 1), "an unaligned target must be refused");
        Assert.True(ex.Truncate(prompt.Length));
        Assert.Equal(prompt.Length, ex.NPast);
        float[] got = Forward(ex, next);

        NewActiveSlot(ex);
        Forward(ex, prompt);
        AssertClose($"next turn after a {answer}-token answer", got, Forward(ex, next), Exact);
    }

    /// <summary>A rewind neither the live ring nor the checkpoint can serve is refused, and
    /// refusing it leaves the slot exactly where it was.</summary>
    [CudaTheory]
    [InlineData(Synthetic)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment)]
    public void Truncate_RefusesAReachNothingHolds_AndLeavesTheSlotUntouched(string fixture)
    {
        using var ex = Open(fixture);
        int[] prompt = Stream(21, 300), reply = Stream(22, 3);
        Forward(ex, prompt);
        foreach (int t in reply)
            Forward(ex, new[] { t });
        int slot = ex.ActiveSlot;
        Assert.False(ex.SlotCanReuse(slot, prompt.Length + reply.Length, 2));
        Assert.False(ex.Truncate(2));
        Assert.Equal(prompt.Length + reply.Length, ex.NPast);

        float[] got = Forward(ex, new[] { 7 });
        NewActiveSlot(ex);
        Forward(ex, prompt);
        foreach (int t in reply)
            Forward(ex, new[] { t });
        AssertClose("after the refusal", got, Forward(ex, new[] { 7 }), Exact);
    }

    /// <summary>The model's retained-conversation contract on the direct-CUDA executor, the
    /// same script DeepSeekNativeRetentionFixtureTests runs on the native one.</summary>
    [CudaTheory]
    [InlineData(Synthetic)]
    [ModelInlineData(FixtureVar, Fixture, FromEnvironment)]
    public void RetainedSlot_ContinuesTheConversationExactly_AndOutlivesANewPrimary(string fixture)
    {
        string path = FixturePath(fixture);
        string[] keys = { "TS_DSV41_RETAINED_CACHE_MB", "MAX_CONTEXT", "TS_DSV4_UBATCH" };
        string[] values = { "2048", "512", "8" };
        var old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], values[i]);
            using var model = new DeepSeek4Model(path, BackendType.Cuda);
            Assert.True(model.SupportsPerSequenceFusedForward);
            Assert.True(model.SupportsRetainedFusedCache);
            Assert.True(model.SupportsKVCacheTruncation);
            int[] prefix = { 0, 15, 32, 64, 128, 13, 254, 18 };
            int[] suffix = { 9, 21, 85 };
            model.Forward(prefix);
            var expected = model.Forward(suffix);
            model.ResetKVCache();
            model.Forward(prefix);
            model.AdoptPrimaryCacheToFused("first");
            Assert.True(model.RetainSequenceCache("first"));
            model.OnSequenceReleased("first");
            Assert.True(model.CanReuseRetainedPrefix("first", prefix.Length, prefix.Length));
            Assert.False(model.CanReuseRetainedPrefix("first", prefix.Length + 1, prefix.Length));
            Assert.True(model.TryRebindRetainedCache("first", "follow"));
            Assert.False(model.BindSequenceCache("follow"));
            AssertClose("the retained conversation's next turn", model.Forward(suffix), expected, Exact);
            Assert.True(model.RetainSequenceCache("follow"));
            model.OnSequenceReleased("follow");
            // The device has room for another slot: the single-stream path gets a new primary and the
            // retained conversation stays for its next turn.
            model.RestorePrimaryCache();
            Assert.True(model.CanReuseRetainedPrefix("follow", prefix.Length + suffix.Length, prefix.Length));
            Assert.True(model.CanReuseLivePrefix(0, 0));
            var unrelated = model.Forward(suffix);
            model.ResetKVCache();
            AssertClose("a reset primary", model.Forward(suffix), unrelated, Exact);
        }
        finally
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], old[i]);
        }
    }

    private const double Exact = 1e-5;

    /// <summary>Asserts every logit is within <paramref name="tolerance"/> of the largest one's
    /// magnitude and the chosen token agrees; returns the worst relative error.</summary>
    private static double AssertClose(string what, float[] actual, float[] expected, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1.0, expected.Max(v => Math.Abs((double)v)));
        double worst = 0;
        int at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            // Both sides: a NaN makes the error NaN, which no comparison with the tolerance catches.
            Assert.True(float.IsFinite(actual[i]) && float.IsFinite(expected[i]),
                $"{what}: logit {i} was {actual[i]}, reference {expected[i]}");
            double error = Math.Abs(actual[i] - expected[i]) / scale;
            if (error > worst) { worst = error; at = i; }
        }
        Assert.True(worst <= tolerance,
            $"{what}: logit {at} was {actual[at]}, reference {expected[at]} (relative error {worst:E3})");
        Assert.Equal(Array.IndexOf(expected, expected.Max()), Array.IndexOf(actual, actual.Max()));
        return worst;
    }
}
