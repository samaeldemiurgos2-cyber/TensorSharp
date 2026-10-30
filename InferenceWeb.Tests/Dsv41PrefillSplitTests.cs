// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A DeepSeek V4.1 prompt prefilled in one forward call and the same prompt prefilled in several calls that
/// end on micro-batch boundaries must leave the same caches, so the same logits, bit for bit: both passes run
/// the same micro-batches, so only something taken from the forward call rather than from the micro-batch can
/// make them differ. A call that ends inside a micro-batch (calls shorter than the micro-batch, a prefix reuse
/// or a rewind resuming at an arbitrary position) regroups the positions after it into different
/// micro-batches; the projections and MoE are not invariant to the micro-batch width, so that is outside this
/// contract.
///
/// <para>The native executor once padded every micro-batch's attention width to the bucket of the WHOLE
/// call's end. The owned CUDA attention chooses its kernel from that width (sparse from 2,048 keys, tiled
/// below), so a whole-prompt call ran its early micro-batches sparse while 256-token calls ran the same
/// micro-batches tiled. Both kernels are F32-accurate but round differently, and V4.1 quantizes attention
/// outputs into its caches: on the lite 1017 checkpoint the two passes' logits were 4.5e-2..1.6e-1 apart at
/// 4,204 tokens, with changed argmax tokens, and the boundaries were exactly 1,025 tokens (ratio-1 layers)
/// and 2,050 (ratio-2 layers). <see cref="Dsv41PrefillSplitFixture"/> has the checkpoint's attention
/// geometry, so the same lengths cross the same boundaries here. 3,000 tokens also compares micro-batches
/// whose OWN width reaches the sparse kernel at the ratio-2 layers (from position 2,048), and 512-token calls
/// show the contract does not depend on a call being exactly one micro-batch.</para>
///
/// <para>YOCO bounded replay is off (TS_DSV41_YOCO_KEEP=0): it runs the replay layers over each call's own
/// tail by design, so it is call-dependent on purpose and outside this contract.</para>
/// </summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class Dsv41PrefillSplitTests : IDisposable
{
    private const int Ubatch = 256;
    private const int DecodeSteps = 3;

    private readonly ITestOutputHelper _output;
    private readonly EnvScope _managed = new();
    private readonly NativeEnvScope _native = new();

    public Dsv41PrefillSplitTests(ITestOutputHelper output)
    {
        _output = output;
        _managed.ClearSpeculationVars();
        // Read by the managed loader (MAX_CONTEXT, TS_DSV4_UBATCH) and the native one (the rest).
        _native.Set("MAX_CONTEXT", Dsv41PrefillSplitFixture.ContextLength.ToString());
        _native.Set("TS_DSV4_UBATCH", Ubatch.ToString());
        _native.Set("TS_DSV41_YOCO_KEEP", "0");
        _native.Set("TS_DSV41_SPARSE_FA", null);
        _native.Set("TS_DSV41_ENGRAM_DEVICE", null);
        _native.Set("TS_DSV41_ENGRAM_WARM", "0");
        _native.Set("TS_N_CPU_MOE", null);
    }

    public void Dispose()
    {
        _native.Dispose();
        _managed.Dispose();
    }

    /// <summary>The serving path, where the owned attention's kernel choice made the split visible.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void GgmlCuda_PromptPrefilledInChunks_MatchesTheWholePromptBitForBit()
        => AssertSplitInvariant(BackendType.GgmlCuda, (1025, 256), (2050, 256), (3000, 256), (3000, 512));

    /// <summary>The CPU backend has no owned attention gate, but shares the per-micro-batch widths.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void GgmlCpu_PromptPrefilledInChunks_MatchesTheWholePromptBitForBit()
        => AssertSplitInvariant(BackendType.GgmlCpu, (2050, 256));

    private void AssertSplitInvariant(BackendType backend, params (int Prompt, int Chunk)[] cases)
    {
        using var model = new DeepSeek4Model(Dsv41PrefillSplitFixture.Path, backend);
        var failures = new List<string>();
        foreach ((int prompt, int chunk) in cases)
        {
            Assert.True(chunk % Ubatch == 0, $"chunk {chunk} must end on micro-batch boundaries (ubatch {Ubatch})");
            int[] tokens = Tokens(prompt + DecodeSteps);
            float[][] whole = Run(model, tokens, prompt, prompt);
            // Nor if the fixture's logits are degenerate (a dead graph gives equal zeros both ways).
            foreach (float[] row in whole)
                Assert.True(row.All(float.IsFinite) && row.Max() - row.Min() > 1e-3f,
                    $"{backend}: the fixture's logits at {prompt} tokens are not finite and distinct");
            // The comparison means nothing unless a pass reproduces itself.
            float[][] again = Run(model, tokens, prompt, prompt);
            for (int row = 0; row < whole.Length; row++)
                Assert.True(Diff.Of(whole[row], again[row]).Differing == 0,
                    $"{backend}: two whole-prompt passes of {prompt} tokens differ at row {row}; the backend is not deterministic here");

            float[][] split = Run(model, tokens, prompt, chunk);
            for (int row = 0; row < whole.Length; row++)
            {
                Diff d = Diff.Of(whole[row], split[row]);
                string name = row == 0 ? "prefill" : $"decode {row - 1}";
                _output.WriteLine($"{backend} prompt {prompt}, chunk {chunk}, {name}: {d}");
                if (d.Differing != 0)
                    failures.Add($"prompt {prompt} chunk {chunk} {name}: {d}");
            }
        }
        Assert.True(failures.Count == 0,
            $"{backend}: a prompt prefilled in several calls left different logits than the same prompt in one call " +
            "(whole vs chunked):\n" + string.Join("\n", failures));
    }

    /// <summary>Resets, prefills <paramref name="prompt"/> tokens in calls of <paramref name="chunk"/>, then
    /// decodes the stream's next tokens; returns the prefill's last logits and each decode step's.</summary>
    private static float[][] Run(DeepSeek4Model model, int[] tokens, int prompt, int chunk)
    {
        model.ResetKVCache();
        var rows = new List<float[]>();
        float[] last = null;
        for (int start = 0; start < prompt; start += chunk)
            last = model.Forward(tokens[start..Math.Min(prompt, start + chunk)]);
        rows.Add(last);
        for (int i = 0; i < DecodeSteps; i++)
            rows.Add(model.Forward(new[] { tokens[prompt + i] }));
        return rows.ToArray();
    }

    /// <summary>A reproducible stream over the fixture's ordinary ids (3..255).</summary>
    private static int[] Tokens(int count)
        => Enumerable.Range(0, count).Select(i => (int)((7L * i * i + 13L * i + 5) % 253) + 3).ToArray();

    private readonly record struct Diff(int Differing, double MaxAbs, double RelL2, bool ArgmaxSame)
    {
        public static Diff Of(float[] a, float[] b)
        {
            Assert.Equal(a.Length, b.Length);
            int differing = 0;
            double maxAbs = 0, num = 0, den = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) differing++;
                double d = (double)a[i] - b[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(d));
                num += d * d;
                den += (double)a[i] * a[i];
            }
            return new Diff(differing, maxAbs, Math.Sqrt(num / Math.Max(den, 1e-30)), Argmax(a) == Argmax(b));
        }

        private static int Argmax(float[] v)
        {
            int best = 0;
            for (int i = 1; i < v.Length; i++)
                if (v[i] > v[best]) best = i;
            return best;
        }

        public override string ToString()
            => $"{Differing} of the logits differ, max |d| {MaxAbs:G3}, rel L2 {RelL2:G3}, argmax {(ArgmaxSame ? "same" : "CHANGED")}";
    }
}
