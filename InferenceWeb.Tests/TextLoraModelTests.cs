// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A text-LoRA adapter is applied unmerged: LinearForward adds x @ AT @ BT on top of the base
// projection. These tests pin that the result is the adapter, not something near it: on a tiny
// synthetic hunyuan-dense model (which fuses attn_q/k/v and ffn_gate/up at load), base + LoRA
// gives the same logits as a model file with the deltas merged into its weights, including
// adapters on the projections that were fused away. They also pin the name mapping a PEFT
// file goes through and the directory confinement of runtime loading.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class TextLoraModelTests : IDisposable
{
    private const int Rank = 4;
    private const float Alpha = 8f; // scale = alpha / rank = 2
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-lora-model-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private static BackendType Backend =>
        TestGates.PinnedGgmlBackend is BackendType.GgmlCpu ? BackendType.Cpu : TestGates.PinnedGgmlBackend;

    // ---- Binder ---------------------------------------------------------------

    [Fact]
    public void Binder_ComposesSplitProjectionsIntoTheFusedWeightExactly()
    {
        // attn_q [8 <- 4], attn_k [2 <- 4], attn_v [2 <- 4], fused at load into attn_qkv [12 <- 4].
        const int inDim = 4, q = 8, kv = 2;
        var deltas = new Dictionary<string, LoraDelta>
        {
            ["blk.0.attn_q.weight"] = RandomDelta(inDim, q, 2, seed: 1),
            ["blk.0.attn_v.weight"] = RandomDelta(inDim, kv, 3, seed: 2),
        };
        var adapter = new TextLoraAdapter("qv", 2, 2, deltas, "");
        var loaded = new Dictionary<string, (long, long)> { ["blk.0.attn_qkv.weight"] = (inDim, q + 2 * kv) };
        var file = new Dictionary<string, (long, long)>
        {
            ["blk.0.attn_q.weight"] = (inDim, q), ["blk.0.attn_k.weight"] = (inDim, kv), ["blk.0.attn_v.weight"] = (inDim, kv),
        };

        var binding = TextLoraBinder.Bind(adapter, n => loaded.TryGetValue(n, out var d) ? d : null,
            n => file.TryGetValue(n, out var d) ? d : null);

        Assert.Empty(binding.Errors);
        var target = Assert.Single(binding.Targets);
        Assert.Equal("blk.0.attn_qkv.weight", target.WeightName);
        Assert.Equal(5, target.Rank);
        Assert.Equal(new[] { "blk.0.attn_q.weight", "blk.0.attn_v.weight" }, target.Sources);

        // x @ AT @ BT over the fused output equals each delta in its own rows, and zero in K's.
        float[] x = { 0.3f, -1.1f, 0.7f, 2.0f };
        float[] fused = Apply(x, target.AT, target.BT, inDim, target.Rank, (int)target.Out);
        float[] qOut = Apply(x, deltas["blk.0.attn_q.weight"].AT, deltas["blk.0.attn_q.weight"].BT, inDim, 2, q);
        float[] vOut = Apply(x, deltas["blk.0.attn_v.weight"].AT, deltas["blk.0.attn_v.weight"].BT, inDim, 3, kv);
        for (int i = 0; i < q; i++) Assert.Equal(qOut[i], fused[i], 5);
        for (int i = 0; i < kv; i++) Assert.Equal(0f, fused[q + i]);
        for (int i = 0; i < kv; i++) Assert.Equal(vOut[i], fused[q + kv + i], 5);
    }

    [Fact]
    public void Binder_RefusesModulesTheModelDoesNotHave_OrWithAnotherShape()
    {
        var deltas = new Dictionary<string, LoraDelta>
        {
            ["blk.0.ffn_down.weight"] = RandomDelta(4, 6, 2, seed: 3),
            ["blk.0.ssm_in.weight"] = RandomDelta(4, 6, 2, seed: 4),
            ["blk.0.ffn_up.weight"] = RandomDelta(4, 5, 2, seed: 5),
        };
        var adapter = new TextLoraAdapter("bad", 2, 2, deltas, "");
        var loaded = new Dictionary<string, (long, long)>
        {
            ["blk.0.ffn_down.weight"] = (4, 7), ["blk.0.ffn_gate_up.weight"] = (4, 12),
        };
        var file = new Dictionary<string, (long, long)> { ["blk.0.ffn_gate.weight"] = (4, 6), ["blk.0.ffn_up.weight"] = (4, 6) };

        var binding = TextLoraBinder.Bind(adapter, n => loaded.TryGetValue(n, out var d) ? d : null,
            n => file.TryGetValue(n, out var d) ? d : null);

        Assert.Empty(binding.Targets);
        Assert.Equal(3, binding.Errors.Count);
        Assert.Contains(binding.Errors, e => e.StartsWith("blk.0.ffn_down.weight:") && e.Contains("[7, 4]"));
        Assert.Contains(binding.Errors, e => e.StartsWith("blk.0.ssm_in.weight: not a weight"));
        Assert.Contains(binding.Errors, e => e.StartsWith("blk.0.ffn_up.weight:") && e.Contains("ffn_gate_up"));
    }

    // ---- File names -----------------------------------------------------------

    [Theory]
    [InlineData("blk.3.attn_q.weight.lora_a", "blk.3.attn_q.weight", LoraFactor.A)]
    [InlineData("lora.blk.3.ffn_down.weight.lora_b", "blk.3.ffn_down.weight", LoraFactor.B)]
    [InlineData("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight", "blk.3.attn_q.weight", LoraFactor.A)]
    [InlineData("base_model.model.model.layers.12.mlp.down_proj.lora_B.weight", "blk.12.ffn_down.weight", LoraFactor.B)]
    [InlineData("base_model.model.backbone.layers.7.mixer.in_proj.lora_A.weight", "blk.7.ssm_in.weight", LoraFactor.A)]
    [InlineData("base_model.model.lm_head.lora_B.weight", "output.weight", LoraFactor.B)]
    public void Names_MapPeftAndPrefixedSpellingsToGgufWeights(string tensor, string target, LoraFactor factor)
    {
        Assert.True(TextLoraNames.TryParse(tensor, out string mapped, out LoraFactor parsed, out string unmapped));
        Assert.Null(unmapped);
        Assert.Equal(target, mapped);
        Assert.Equal(factor, parsed);
    }

    [Fact]
    public void Loader_ReadsAPeftFile_AndRefusesModulesItCannotName()
    {
        string path = Path.Combine(_dir, "peft", "adapter_model.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var a = new float[2 * 4];
        var b = new float[6 * 2];
        WriteSafetensors(path,
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", new long[] { 2, 4 }, a),
            ("base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight", new long[] { 6, 2 }, b));
        var adapter = TextLoraLoader.Load("peft", path);
        Assert.Equal(new[] { "blk.0.attn_q.weight" }, adapter.Deltas.Keys.ToArray());

        string odd = Path.Combine(_dir, "odd.safetensors");
        WriteSafetensors(odd,
            ("base_model.model.model.layers.0.self_attn.rotary.lora_A.weight", new long[] { 2, 4 }, a),
            ("base_model.model.model.layers.0.self_attn.rotary.lora_B.weight", new long[] { 6, 2 }, b));
        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("odd", odd));
        Assert.Contains("self_attn.rotary", ex.Message);
    }

    // ---- Runtime directory ----------------------------------------------------

    [Fact]
    public void Paths_ConfineLoadingToTheConfiguredDirectory()
    {
        string root = Path.Combine(_dir, "loras");
        Directory.CreateDirectory(Path.Combine(root, "sub"));
        File.WriteAllText(Path.Combine(_dir, "secret.safetensors"), "x");

        Assert.Equal(Path.Combine(root, "sub", "a.safetensors"),
            TextLoraPaths.ResolveWithin(root, "sub/a.safetensors", ".safetensors"));

        Assert.Throws<LoraPathException>(() => TextLoraPaths.ResolveWithin(null, "a.safetensors", ".safetensors"));
        Assert.Throws<LoraPathException>(() => TextLoraPaths.ResolveWithin(root, "../secret.safetensors", ".safetensors"));
        Assert.Throws<LoraPathException>(() => TextLoraPaths.ResolveWithin(root, "sub/../../secret.safetensors", ".safetensors"));
        Assert.Throws<LoraPathException>(() =>
            TextLoraPaths.ResolveWithin(root, Path.Combine(_dir, "secret.safetensors"), ".safetensors"));
        Assert.Throws<LoraPathException>(() => TextLoraPaths.ResolveWithin(root, "sub/passwd", ".safetensors"));

        // The refusal for a path outside the directory is the same whether or not it exists.
        var outsideExisting = Assert.Throws<LoraPathException>(() => TextLoraPaths.ResolveWithin(root, "../secret.safetensors", ".safetensors"));
        var outsideMissing = Assert.Throws<LoraPathException>(() => TextLoraPaths.ResolveWithin(root, "../nope.safetensors", ".safetensors"));
        Assert.Equal(outsideExisting.Message, outsideMissing.Message);

        Assert.True(TextLoraPaths.IsValidName("style-v2.1"));
        Assert.False(TextLoraPaths.IsValidName("none"));
        Assert.False(TextLoraPaths.IsValidName("../x"));
        Assert.False(TextLoraPaths.IsValidName(""));
    }

    [Fact]
    public void Paths_RefuseALinkThatLeavesTheDirectory()
    {
        string root = Path.Combine(_dir, "linked");
        Directory.CreateDirectory(root);
        string outside = Path.Combine(_dir, "outside");
        Directory.CreateDirectory(outside);
        File.WriteAllText(Path.Combine(outside, "a.safetensors"), "x");
        try
        {
            Directory.CreateSymbolicLink(Path.Combine(root, "escape"), outside);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        {
            return; // Creating links needs a privilege this machine does not grant; nothing to check.
        }
        Assert.Throws<LoraPathException>(() => TextLoraPaths.ResolveWithin(root, "escape/a.safetensors", ".safetensors"));
    }

    // ---- LinearForward hook on a real model -----------------------------------

    /// <summary>The adapter of the parity tests: a delta on attn_q and attn_v (fused into
    /// attn_qkv at load), on ffn_up (fused into ffn_gate_up), on ffn_down and on output.</summary>
    private static readonly (string Module, int In, int Out)[] AdapterModules =
    {
        ("blk.0.attn_q.weight", DenseDecoderSyntheticModelBuilder.Hidden, DenseDecoderSyntheticModelBuilder.NumHeads * DenseDecoderSyntheticModelBuilder.HeadDim),
        ("blk.0.attn_v.weight", DenseDecoderSyntheticModelBuilder.Hidden, DenseDecoderSyntheticModelBuilder.NumKvHeads * DenseDecoderSyntheticModelBuilder.HeadDim),
        ("blk.1.ffn_up.weight", DenseDecoderSyntheticModelBuilder.Hidden, DenseDecoderSyntheticModelBuilder.FfnLength),
        ("blk.1.ffn_down.weight", DenseDecoderSyntheticModelBuilder.FfnLength, DenseDecoderSyntheticModelBuilder.Hidden),
        ("output.weight", DenseDecoderSyntheticModelBuilder.Hidden, DenseDecoderSyntheticModelBuilder.ByteTokens),
    };

    private Dictionary<string, (float[] A, float[] B, int In, int Out)> AdapterFactors()
    {
        var factors = new Dictionary<string, (float[], float[], int, int)>();
        int seed = 11;
        foreach (var (module, inDim, outDim) in AdapterModules)
            factors[module] = (Gen(Rank * inDim, 0.15f, seed++), Gen(outDim * Rank, 0.15f, seed++), inDim, outDim);
        return factors;
    }

    private string WriteAdapter(Dictionary<string, (float[] A, float[] B, int In, int Out)> factors)
    {
        string dir = Path.Combine(_dir, "adapter");
        Directory.CreateDirectory(dir);
        var tensors = new List<(string, long[], float[])>();
        foreach (var (module, (a, b, inDim, outDim)) in factors)
        {
            tensors.Add((module + ".lora_a", new long[] { Rank, inDim }, a));
            tensors.Add((module + ".lora_b", new long[] { outDim, Rank }, b));
        }
        string path = Path.Combine(dir, "adapter.safetensors");
        WriteSafetensors(path, tensors.ToArray());
        File.WriteAllText(Path.Combine(dir, "adapter_config.json"), $"{{ \"r\": {Rank}, \"lora_alpha\": {Alpha} }}");
        return path;
    }

    private string WriteModel(string name, Dictionary<string, (float[] A, float[] B, int In, int Out)> mergeFactors) =>
        new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "hunyuan-dense",
            PreTokenizer = "hunyuan-dense",
            IncludeMistralControlTokens = false,
            IncludeQkNorms = true,
            RopeBase = 10000f,
            TransformTensor = mergeFactors == null ? null : (tensorName, dims, data) =>
            {
                if (!mergeFactors.TryGetValue(tensorName, out var f)) return data;
                // GGUF [in, out] with in fastest: W[o, i] at o * in + i. W += (alpha / r) * B @ A.
                var merged = (float[])data.Clone();
                float scale = Alpha / Rank;
                for (int o = 0; o < f.Out; o++)
                    for (int i = 0; i < f.In; i++)
                    {
                        float d = 0;
                        for (int k = 0; k < Rank; k++) d += f.B[o * Rank + k] * f.A[k * f.In + i];
                        merged[o * f.In + i] += scale * d;
                    }
                return merged;
            },
        }.Write(Path.Combine(_dir, name));

    [Fact]
    public void LinearForwardLora_MatchesTheMergedWeights_IncludingFusedTargets()
    {
        var factors = AdapterFactors();
        var adapter = TextLoraLoader.Load("parity", WriteAdapter(factors));
        int[] prompt = { 72, 101, 108, 108, 111, 33, 7, 200 };
        int next = 42;

        float[] mergedPrefill, mergedDecode;
        using (var merged = ModelBase.Create(WriteModel("merged.gguf", factors), Backend))
        {
            mergedPrefill = (float[])merged.Forward(prompt).Clone();
            mergedDecode = (float[])merged.Forward(new[] { next }).Clone();
        }

        using var model = ModelBase.Create(WriteModel("base.gguf", null), Backend);
        float[] basePrefill = (float[])model.Forward(prompt).Clone();
        model.ResetKVCache();

        var targets = model.PrepareLora(adapter);
        Assert.Equal(
            new[] { "blk.0.attn_qkv.weight", "blk.1.ffn_down.weight", "blk.1.ffn_gate_up.weight", "output.weight" },
            targets.OrderBy(t => t, StringComparer.Ordinal).ToArray());

        var lora = (ILoraAdapterModel)model;
        lora.ActivateLoraAdapter(adapter);
        lora.TakeAppliedLoraTargets();
        float[] loraPrefill = (float[])model.Forward(prompt).Clone();
        float[] loraDecode = (float[])model.Forward(new[] { next }).Clone();
        var applied = lora.TakeAppliedLoraTargets();
        lora.ActivateLoraAdapter(null);

        Assert.True(MaxAbsDiff(basePrefill, mergedPrefill) > 1e-2f, "the adapter must change the logits for the test to mean anything");
        AssertClose(mergedPrefill, loraPrefill, "prefill");
        AssertClose(mergedDecode, loraDecode, "decode");
        Assert.Equal(targets.OrderBy(t => t).ToArray(), applied.OrderBy(t => t).ToArray());

        // Deactivated, the model is the base model again.
        model.ResetKVCache();
        AssertClose(basePrefill, model.Forward(prompt), "base after deactivation");
    }

    [Fact]
    public void PrepareLora_RefusesAnAdapterThatDoesNotFit_NamingTheModules()
    {
        var factors = AdapterFactors();
        factors["blk.9.attn_q.weight"] = factors["blk.0.attn_q.weight"];
        var adapter = TextLoraLoader.Load("misfit", WriteAdapter(factors));
        using var model = ModelBase.Create(WriteModel("base.gguf", null), Backend);

        var ex = Assert.Throws<InvalidDataException>(() => model.PrepareLora(adapter));
        Assert.Contains("blk.9.attn_q.weight", ex.Message);
        Assert.False(model.IsLoraPrepared(adapter));
    }

    // ---- helpers --------------------------------------------------------------

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = MaxAbsDiff(expected, actual);
        float scale = expected.Max(MathF.Abs);
        Assert.True(worst <= 1e-4f * Math.Max(1f, scale), $"{what}: max |diff| {worst} (logit scale {scale})");
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    private static float[] Apply(float[] x, float[] at, float[] bt, int inDim, int rank, int outDim)
    {
        var h = new float[rank];
        for (int r = 0; r < rank; r++)
            for (int i = 0; i < inDim; i++) h[r] += x[i] * at[i * rank + r];
        var y = new float[outDim];
        for (int o = 0; o < outDim; o++)
            for (int r = 0; r < rank; r++) y[o] += h[r] * bt[r * outDim + o];
        return y;
    }

    private static LoraDelta RandomDelta(int inDim, int outDim, int rank, int seed) => new()
    {
        In = inDim, Out = outDim, Rank = rank,
        AT = Gen(inDim * rank, 1f, seed), BT = Gen(rank * outDim, 1f, seed + 100),
    };

    private static float[] Gen(int n, float scale, int seed)
    {
        uint state = (uint)(seed * 2654435761u) | 1u;
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            data[i] = scale * ((state & 0xFFFF) / 32768f - 1f);
        }
        return data;
    }

    internal static void WriteSafetensors(string path, params (string Name, long[] Shape, float[] Data)[] tensors)
    {
        var header = new JsonObject();
        long offset = 0;
        var blobs = new List<byte[]>();
        foreach (var (name, shape, data) in tensors)
        {
            var bytes = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
            var shapeArr = new JsonArray();
            foreach (long d in shape) shapeArr.Add(d);
            header[name] = new JsonObject
            {
                ["dtype"] = "F32",
                ["shape"] = shapeArr,
                ["data_offsets"] = new JsonArray(offset, offset + bytes.Length),
            };
            offset += bytes.Length;
            blobs.Add(bytes);
        }
        byte[] hb = Encoding.UTF8.GetBytes(header.ToJsonString());
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)hb.Length));
        fs.Write(hb);
        foreach (var blob in blobs) fs.Write(blob);
    }
}
