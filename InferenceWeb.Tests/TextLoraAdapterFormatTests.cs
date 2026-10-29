// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The two adapter containers the text-LoRA loader reads (PR #282 review item 8): llama.cpp's
/// GGUF adapter and a safetensors file of <c>&lt;tensor&gt;.lora_a</c> / <c>.lora_b</c> pairs,
/// both named by the base GGUF's tensor names. What cannot be applied exactly is refused with
/// a reason, never skipped: PEFT module names (with the converter to use), DoRA and the other
/// PEFT settings, activated LoRA, and tensors that are not factors. Expected factors are
/// re-derived here with nested loops, not with loader code.
/// </summary>
public sealed class TextLoraAdapterFormatTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-lora-format-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ------------------------------------------------------------------ GGUF adapters

    [Theory]
    [InlineData("F32")]
    [InlineData("F16")]
    [InlineData("BF16")]
    [InlineData("Q8_0")]
    public void GgufAdapter_LlamaCppLayout_MatchesItsSafetensorsTwin(string factorType)
    {
        var type = Enum.Parse<SyntheticGguf.GgmlType>(factorType);
        // Q8_0 rows are 32-value blocks, so both factors need a row of 32 or more.
        int rank = type == SyntheticGguf.GgmlType.Q8_0 ? 32 : 8, inDim = 64, outDim = 48;
        const float alpha = 16f;
        var (a, b) = TextLoraFactors.Random(7, rank, inDim, outDim);
        string gguf = TextLoraFactors.WriteGguf(Path.Combine(_dir, "a.gguf"), "mistral3", alpha,
            new[] { ("blk.0.attn_q.weight", a, b, rank, inDim, outDim) }, type);
        string safetensors = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "a.safetensors"),
            new[] { ("blk.0.attn_q.weight", a, b, rank, inDim, outDim) }, alpha);

        var fromGguf = TextLoraLoader.Load("g", gguf);
        var fromSafetensors = TextLoraLoader.Load("s", safetensors);

        Assert.Equal("mistral3", fromGguf.Architecture);
        Assert.Null(fromSafetensors.Architecture);
        Assert.Equal(alpha, fromGguf.Alpha);
        var dg = fromGguf.Deltas["blk.0.attn_q.weight"];
        var ds = fromSafetensors.Deltas["blk.0.attn_q.weight"];
        Assert.Equal((inDim, outDim, rank), (dg.In, dg.Out, dg.Rank));
        // F32 is exact; the stored types round each factor once.
        float tolerance = type switch
        {
            SyntheticGguf.GgmlType.F32 => 0f,
            SyntheticGguf.GgmlType.F16 => 2e-3f,
            SyntheticGguf.GgmlType.BF16 => 1.6e-2f,
            _ => 2e-2f,
        };
        for (int i = 0; i < ds.AT.Length; i++)
            Assert.True(Math.Abs(ds.AT[i] - dg.AT[i]) <= tolerance * Math.Max(1f, Math.Abs(ds.AT[i])), $"AT[{i}]");
        float scale = alpha / rank;
        for (int i = 0; i < ds.BT.Length; i++)
            Assert.True(Math.Abs(ds.BT[i] - dg.BT[i]) <= tolerance * scale * Math.Max(1f, Math.Abs(ds.BT[i]) / scale), $"BT[{i}]");
        if (type == SyntheticGguf.GgmlType.F32)
        {
            Assert.Equal(ds.AT, dg.AT);
            Assert.Equal(ds.BT, dg.BT);
        }
    }

    [Fact]
    public void GgufAdapter_ScaleIsAlphaOverRank_AndZeroAlphaMeansOne_TimesTheCallersScale()
    {
        var (a, b) = TextLoraFactors.Random(3, 4, 32, 32);
        var targets = new[] { ("output.weight", a, b, 4, 32, 32) };
        string withAlpha = TextLoraFactors.WriteGguf(Path.Combine(_dir, "alpha.gguf"), "mistral3", 12f, targets);
        string noAlpha = TextLoraFactors.WriteGguf(Path.Combine(_dir, "zero.gguf"), "mistral3", 0f, targets);

        float b00 = b[0];   // B[0, 0] lands at BT[0, 0]
        Assert.Equal(3f * b00, TextLoraLoader.Load("x", withAlpha).Deltas["output.weight"].BT[0], 5);
        Assert.Equal(1f * b00, TextLoraLoader.Load("x", noAlpha).Deltas["output.weight"].BT[0], 5);
        Assert.Equal(0.5f * 3f * b00, TextLoraLoader.Load("x", withAlpha, scaleOverride: 0.5f).Deltas["output.weight"].BT[0], 5);
    }

    [Fact]
    public void GgufAdapter_ThatIsNotALoraAdapter_IsRefusedWithTheReason()
    {
        var (a, b) = TextLoraFactors.Random(1, 2, 32, 32);
        var targets = new[] { ("output.weight", a, b, 2, 32, 32) };

        string model = TextLoraFactors.WriteGguf(Path.Combine(_dir, "model.gguf"), "mistral3", 1f, targets, generalType: "model");
        var notAdapter = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("m", model));
        Assert.Contains("is not a LoRA adapter", notAdapter.Message);
        Assert.Contains("'model'", notAdapter.Message);

        string vector = TextLoraFactors.WriteGguf(Path.Combine(_dir, "cv.gguf"), "mistral3", 1f, targets, adapterType: "control_vector");
        Assert.Contains("'control_vector' adapter", Assert.Throws<NotSupportedException>(() => TextLoraLoader.Load("c", vector)).Message);

        string alora = TextLoraFactors.WriteGguf(Path.Combine(_dir, "alora.gguf"), "mistral3", 1f, targets, alora: true);
        Assert.Contains("activated LoRA", Assert.Throws<NotSupportedException>(() => TextLoraLoader.Load("a", alora)).Message);
    }

    [Fact]
    public void GgufAdapter_WithAFactorTypeItCannotRead_IsRefusedNamingTheTensor()
    {
        string path = Path.Combine(_dir, "k.gguf");
        var kv = TextLoraFactors.AdapterMetadata("mistral3", 8f);
        SyntheticGguf.Write(path, kv, new List<SyntheticGguf.Tensor>
        {
            SyntheticGguf.Blocks("output.weight.lora_a", SyntheticGguf.GgmlType.Q4_K, 0.1f, 256, 8),
            new() { Name = "output.weight.lora_b", Dims = new ulong[] { 8, 4 }, Data = new float[32] },
        });

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("k", path));
        Assert.Contains("'output.weight.lora_a'", ex.Message);
        Assert.Contains("F32, F16, BF16 or Q8_0", ex.Message);
    }

    /// <summary>
    /// A header that claims more than its file holds is refused as malformed before anything is
    /// sized from the claim: never an OutOfMemoryException, which reads as the host running out
    /// rather than the file being corrupt.
    /// </summary>
    [Theory]
    [InlineData("dims", "has 2147483647 dimensions; at most 8 are supported")]
    [InlineData("ne0", "GGUF tensor 'w.lora_a' (F32, ne [536870911, 1] at data offset 0) does not fit in the file's")]
    [InlineData("ne0-wrap", "GGUF tensor 'w.lora_a' (F32, ne [4611686018427387905, 1] at data offset 0) does not fit in the file's")]
    [InlineData("array", "GGUF array of 2147483647 Uint32 values runs past the end of the file")]
    [InlineData("string", "GGUF string length 2147483647 runs past the end of the file")]
    [InlineData("alignment", "GGUF general.alignment 0 is not a power of 2")]
    public void GgufAdapter_WhoseHeaderClaimsMoreThanTheFileHolds_IsRefusedBeforeAnythingIsSized(string kind, string reason)
    {
        string path = TextLoraFactors.WriteCorruptGguf(Path.Combine(_dir, kind + ".gguf"), kind);

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path));

        Assert.Contains(reason, ex.Message);
    }

    /// <summary>
    /// The reader also serves files ggml cannot open, so it is not held to ggml's four
    /// dimensions: Wan's DiT stores its conv3d patch embedding as one 5-D tensor
    /// [kw, kh, kd, ic, oc], and refusing it refused every Wan DiT at probe time. All five
    /// dimensions come back, and the tensor's data and the next one's read at their offsets.
    /// </summary>
    [Fact]
    public void GgufFile_WithAFiveDimensionalTensor_ReadsEveryDimensionAndItsData()
    {
        string path = Path.Combine(_dir, "wan.gguf");
        var patch = SyntheticGguf.Gen("patch_embedding.weight", 0.1f, 2, 2, 1, 4, 8);
        patch.Type = SyntheticGguf.GgmlType.F16;
        var bias = SyntheticGguf.Gen("patch_embedding.bias", 0.1f, 8);
        SyntheticGguf.Write(path, new List<SyntheticGguf.Kv> { new SyntheticGguf.Str { Key = "general.architecture", V = "wan" } },
            new List<SyntheticGguf.Tensor> { patch, bias });

        using var gguf = new GgufFile(path);

        var info = gguf.Tensors["patch_embedding.weight"];
        Assert.Equal(new ulong[] { 2, 2, 1, 4, 8 }, info.Shape);
        Assert.Equal(GgmlTensorType.F16, info.Type);
        Assert.Equal(128, info.NumElements);
        Assert.Equal(256, gguf.GetTensorByteCount(info));
        Assert.Equal(patch.Raw(), gguf.ReadTensorData(info));
        Assert.Equal(bias.Raw(), gguf.ReadTensorData(gguf.Tensors["patch_embedding.bias"]));
        gguf.ThrowIfTruncated();
    }

    [Fact]
    public void GgufFile_NamedWithoutTheExtension_IsStillReadAsGguf()
    {
        var (a, b) = TextLoraFactors.Random(5, 2, 32, 32);
        string path = TextLoraFactors.WriteGguf(Path.Combine(_dir, "adapter.bin"), "mistral3", 2f,
            new[] { ("output.weight", a, b, 2, 32, 32) });

        Assert.Equal("mistral3", TextLoraLoader.Load("x", path).Architecture);
    }

    // ------------------------------------------------------------------ PEFT names and settings

    [Fact]
    public void PeftModuleNames_AreRefused_PointingAtTheConverter()
    {
        string path = LoraFixtures.WriteRaw(Path.Combine(_dir, "adapter_model.safetensors"),
            "{\"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]}," +
            "\"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[4,8]}}",
            new byte[8]);

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("peft", path));
        Assert.Contains("PEFT / Hugging Face module names", ex.Message);
        Assert.Contains("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", ex.Message);
        Assert.Contains("convert_lora_to_gguf.py", ex.Message);
    }

    [Fact]
    public void PeftModuleNames_InAGguf_AreRefusedToo()
    {
        string path = Path.Combine(_dir, "peft.gguf");
        SyntheticGguf.Write(path, TextLoraFactors.AdapterMetadata("llama", 8f), new List<SyntheticGguf.Tensor>
        {
            new() { Name = "base_model.model.lm_head.lora_A.weight", Dims = new ulong[] { 32, 2 }, Data = new float[64] },
        });
        Assert.Contains("convert_lora_to_gguf.py", Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("p", path)).Message);
    }

    /// <summary>
    /// Each PEFT setting that changes the adapter's math beyond <c>alpha / r</c> is refused,
    /// naming it, from an <c>adapter_config.json</c> beside the weights and from a named config.
    /// Activated LoRA (any <c>alora_invocation_tokens</c> but null, or the older
    /// <c>invocation_string</c> of IBM's granite aLoRA adapters, which llama.cpp's converter
    /// writes as <c>adapter.alora.*</c>), trained token embeddings
    /// (<c>trainable_token_indices</c>), trained biases (<c>bias</c> other than <c>"none"</c>,
    /// <c>lora_bias</c>), <c>layer_replication</c>, the KaSA and Arrow variants and a pattern
    /// that is not an object (a malformed one, never read as empty) loaded and applied the
    /// adapter to every token as plain <c>alpha / r</c> without them. A variant config that is
    /// an empty object is the variant with its defaults; BD-LoRA is refused because it stores a
    /// block-diagonal factor (<c>[out, in / nblocks]</c>), not a full one.
    /// </summary>
    [Theory]
    [InlineData("\"use_dora\": true", "use_dora")]
    [InlineData("\"fan_in_fan_out\": true", "fan_in_fan_out")]
    [InlineData("\"alpha_pattern\": {\"q_proj\": 32}", "alpha_pattern")]
    [InlineData("\"rank_pattern\": {\"q_proj\": 16}", "rank_pattern")]
    [InlineData("\"rank_pattern\": 16", "rank_pattern")]
    [InlineData("\"modules_to_save\": [\"lm_head\"]", "modules_to_save")]
    [InlineData("\"alora_invocation_tokens\": [1, 2]", "alora_invocation_tokens (activated LoRA)")]
    [InlineData("\"alora_invocation_tokens\": []", "alora_invocation_tokens (activated LoRA)")]
    [InlineData("\"bias\": \"all\"", "bias other than \"none\"")]
    [InlineData("\"bias\": \"lora_only\"", "bias other than \"none\"")]
    [InlineData("\"bias\": true", "bias other than \"none\"")]
    [InlineData("\"lora_bias\": true", "lora_bias")]
    [InlineData("\"layer_replication\": [[0, 4], [2, 6]]", "layer_replication")]
    [InlineData("\"peft_type\": \"LORA\", \"invocation_string\": \"<|start_of_role|>check_requirement<|end_of_role|>\"", "invocation_string (activated LoRA)")]
    [InlineData("\"invocation_string\": 7", "invocation_string (activated LoRA)")]
    [InlineData("\"trainable_token_indices\": [128256, 128257]", "trainable_token_indices (trained token embeddings)")]
    [InlineData("\"trainable_token_indices\": {\"embed_tokens\": [128256]}", "trainable_token_indices (trained token embeddings)")]
    [InlineData("\"kasa_config\": {}", "kasa_config (KaSA)")]
    [InlineData("\"kasa_config\": true", "kasa_config (KaSA)")]
    [InlineData("\"arrow_config\": {\"top_k\": 3, \"router_temperature\": 1.0}", "arrow_config (Arrow)")]
    [InlineData("\"arrow_config\": {}", "arrow_config (Arrow)")]
    [InlineData("\"use_bdlora\": true", "use_bdlora (BD-LoRA)")]
    [InlineData("\"use_bdlora\": {}", "use_bdlora (BD-LoRA)")]
    public void PeftSettingsThatCannotBeAppliedExactly_AreRefused(string setting, string named)
    {
        foreach (var (path, config, file) in ConfigForms("{ \"r\": 1, \"lora_alpha\": 1, " + setting + " }"))
        {
            var ex = Assert.Throws<NotSupportedException>(() => TextLoraLoader.Load("x", path, config));
            Assert.StartsWith($"LoRA config '{file}' uses ", ex.Message);
            Assert.Contains(named, ex.Message);
        }
    }

    /// <summary>
    /// An <c>init_lora_weights</c> that rewrites the base weights when PEFT creates the adapter
    /// (PiSSA and its fast-SVD form, OLoRA, CorDA, Astra, LoftQ, LoRA-GA) makes an adapter that
    /// applies on top of <c>W - s * B0 * A0</c> (LoftQ: a quantized residual), not on top of
    /// <c>W</c>, unless PEFT converted it on save, which writes <c>init_lora_weights</c> true.
    /// Every one loaded and was added to the original weights, PiSSA's top-r SVD component
    /// twice. Refused, beside the weights and named, saying how to convert it. OLoRA is matched
    /// in any case, as PEFT's layer matches it.
    /// </summary>
    [Theory]
    [InlineData("pissa")]
    [InlineData("pissa_niter_4")]
    [InlineData("olora")]
    [InlineData("OLoRA")]
    [InlineData("corda")]
    [InlineData("astra")]
    [InlineData("loftq")]
    [InlineData("lora_ga")]
    public void PeftInitsThatRewriteTheBaseWeights_AreRefused_WhenSavedWithoutConversion(string init)
    {
        foreach (var (path, config, file) in ConfigForms("{ \"r\": 1, \"lora_alpha\": 1, \"init_lora_weights\": \"" + init + "\" }"))
        {
            var ex = Assert.Throws<NotSupportedException>(() => TextLoraLoader.Load("x", path, config));
            Assert.StartsWith($"LoRA config '{file}' has init_lora_weights \"{init}\": the adapter was saved without weight conversion", ex.Message);
            Assert.Contains("save_pretrained(..., path_initial_model_for_weight_conversion=...)", ex.Message);
            Assert.Contains(init == "loftq" ? "does not undo LoftQ's quantized residual" : "Convert it to a plain LoRA adapter before use", ex.Message);
        }
    }

    /// <summary>
    /// What leaves the adapter plain LoRA at inference loads at <c>alpha / r</c> (8 / 4 on
    /// all-ones factors here), beside the weights and named. VeLoRA changes only the backward
    /// pass and MonteCLoRA perturbs <c>lora_A</c> only while training: PEFT's forward when not
    /// training is <c>lora_B(lora_A(x)) * scaling</c> for both, and both merge as plain LoRA.
    /// Their configs are as PEFT saves them (MonteCLoRA's at its defaults); both were refused.
    /// The other <c>init_lora_weights</c> only initialize the factors and leave the base weights
    /// alone.
    /// </summary>
    [Theory]
    [InlineData("\"velora_config\": {\"num_groups\": 32, \"scale\": 1.0, \"init_type\": \"batch_average_once\"}")]
    [InlineData("\"monteclora_config\": {\"num_samples\": 8, \"use_entropy\": false, \"dirichlet_prior\": 0.1, " +
                "\"sample_scaler\": 0.0001, \"kl_loss_weight\": 1e-05, \"buffer_size\": 150}")]
    [InlineData("\"init_lora_weights\": true")]
    [InlineData("\"init_lora_weights\": false")]
    [InlineData("\"init_lora_weights\": \"gaussian\"")]
    [InlineData("\"init_lora_weights\": \"eva\"")]
    [InlineData("\"init_lora_weights\": \"nora\"")]
    [InlineData("\"init_lora_weights\": \"orthogonal\"")]
    [InlineData("\"init_lora_weights\": \"mica\"")]
    public void PeftSettings_ThatArePlainLoraAtInference_LoadAtAlphaOverRank(string setting)
    {
        foreach (var (path, config, _) in ConfigForms("{ \"r\": 4, \"lora_alpha\": 8, " + setting + " }", rank: 4))
        {
            var adapter = TextLoraLoader.Load("x", path, config);
            Assert.Equal(8f, adapter.Alpha);
            var delta = Assert.Single(adapter.Deltas).Value;
            Assert.Equal(4, delta.Rank);
            Assert.All(delta.BT, v => Assert.Equal(2f, v));   // B at 8 / 4
        }
    }

    /// <summary>
    /// An adapter whose config is <paramref name="json"/>, in both forms the loader reads: an
    /// <c>adapter_config.json</c> beside the weights, which it finds itself (no config named),
    /// and a config the caller names, with nothing beside its weights.
    /// </summary>
    private List<(string Path, string? Config, string File)> ConfigForms(string json, int rank = 1)
    {
        string sibling = Directory.CreateDirectory(Path.Combine(_dir, "sibling")).FullName;
        File.WriteAllText(Path.Combine(sibling, "adapter_config.json"), json);
        string named = Directory.CreateDirectory(Path.Combine(_dir, "named")).FullName;
        string config = Path.Combine(_dir, "cfg.json");
        File.WriteAllText(config, json);
        return new()
        {
            (LoraFixtures.WriteAdapter(Path.Combine(sibling, "a.safetensors"), "w", rank), null, "adapter_config.json"),
            (LoraFixtures.WriteAdapter(Path.Combine(named, "a.safetensors"), "w", rank), config, "cfg.json"),
        };
    }

    /// <summary>
    /// A boolean PEFT flag the loader reads must be <c>true</c> or <c>false</c>. PEFT tests it
    /// by Python truth, so <c>1</c> or <c>"true"</c> turns DoRA or rsLoRA on there, while the
    /// loader read anything but JSON <c>true</c> as off and applied the adapter at the wrong scale
    /// (rsLoRA) or without the setting.
    /// </summary>
    [Theory]
    [InlineData("use_dora", "1")]
    [InlineData("use_rslora", "\"true\"")]
    [InlineData("use_rslora", "1")]
    [InlineData("use_rslora", "0")]
    [InlineData("fan_in_fan_out", "1")]
    [InlineData("lora_bias", "\"yes\"")]
    public void PeftFlags_ThatAreNotBooleans_AreRefused(string flag, string value)
    {
        string path = LoraFixtures.WriteAdapter(Path.Combine(_dir, "a.safetensors"), "w");
        string config = Path.Combine(_dir, "cfg.json");
        File.WriteAllText(config, "{ \"r\": 1, \"lora_alpha\": 1, \"" + flag + "\": " + value + " }");

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path, config));
        Assert.Equal($"LoRA config 'cfg.json' has a {flag} that is not true or false.", ex.Message);
    }

    /// <summary>
    /// The keys a PEFT <c>LoraConfig</c> writes, at their defaults, from a stock
    /// <c>adapter_config.json</c> beside the weights and from a named config: each refused
    /// setting is off when null, false or empty (a variant config only when null or false),
    /// so a plain adapter still loads.
    /// </summary>
    [Theory]
    [InlineData(PeftStockSettings)]
    [InlineData("\"use_dora\": false, \"fan_in_fan_out\": false, \"alpha_pattern\": {}, \"rank_pattern\": {}, " +
                "\"modules_to_save\": null, \"bias\": \"none\", \"lora_bias\": false, \"layer_replication\": null, " +
                "\"alora_invocation_tokens\": null, \"use_rslora\": null")]
    [InlineData("\"invocation_string\": null")]
    [InlineData("\"invocation_string\": \"\"")]
    [InlineData("\"trainable_token_indices\": null")]
    [InlineData("\"trainable_token_indices\": []")]
    [InlineData("\"trainable_token_indices\": {}")]
    [InlineData("\"modules_to_save\": []")]
    [InlineData("\"kasa_config\": null, \"arrow_config\": null, \"velora_config\": null, \"monteclora_config\": null, \"use_bdlora\": null")]
    [InlineData("\"kasa_config\": false, \"arrow_config\": false, \"velora_config\": false, \"monteclora_config\": false, \"use_bdlora\": false")]
    public void PeftSettings_ThatAreEmptyOrOff_AreAccepted(string settings)
    {
        foreach (var (path, config, _) in ConfigForms("{ \"r\": 1, \"lora_alpha\": 1, " + settings + " }"))
            Assert.Single(TextLoraLoader.Load("x", path, config).Deltas);
    }

    // A PEFT LoraConfig as saved for a causal LM, every key at its default (r and lora_alpha
    // come from the test).
    private const string PeftStockSettings =
        "\"alora_invocation_tokens\": null, \"alpha_pattern\": {}, \"arrow_config\": null, \"auto_mapping\": null, " +
        "\"base_model_name_or_path\": \"meta-llama/Llama-3.2-1B\", \"bias\": \"none\", \"corda_config\": null, " +
        "\"ensure_weight_tying\": false, \"eva_config\": null, \"exclude_modules\": null, \"fan_in_fan_out\": false, " +
        "\"inference_mode\": true, \"init_lora_weights\": true, \"kasa_config\": null, \"layer_replication\": null, " +
        "\"layers_pattern\": null, \"layers_to_transform\": null, \"loftq_config\": {}, \"lora_bias\": false, " +
        "\"lora_dropout\": 0.05, \"megatron_config\": null, \"megatron_core\": \"megatron.core\", " +
        "\"modules_to_save\": null, \"monteclora_config\": null, \"peft_type\": \"LORA\", \"qalora_group_size\": 16, " +
        "\"rank_pattern\": {}, \"revision\": null, \"target_modules\": [\"q_proj\", \"v_proj\"], " +
        "\"target_parameters\": null, \"task_type\": \"CAUSAL_LM\", \"trainable_token_indices\": null, " +
        "\"use_bdlora\": false, \"use_dora\": false, \"use_qalora\": false, \"use_rslora\": false, \"velora_config\": null";

    [Fact]
    public void Rslora_ScalesByAlphaOverSquareRootOfRank()
    {
        const int rank = 4;
        var (a, b) = TextLoraFactors.Random(11, rank, 3, 2);
        string path = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "r.safetensors"),
            new[] { ("w", a, b, rank, 3, 2) }, alpha: null);
        string config = Path.Combine(_dir, "cfg.json");
        File.WriteAllText(config, "{ \"r\": 4, \"lora_alpha\": 8, \"use_rslora\": true }");

        var delta = TextLoraLoader.Load("x", path, config).Deltas["w"];

        // BT[j, o] = (alpha / sqrt(r)) * B[o, j] = 4 * B[o, j].
        for (int o = 0; o < 2; o++)
            for (int j = 0; j < rank; j++)
                Assert.Equal(4f * b[o * rank + j], delta.BT[j * 2 + o], 5);
    }

    [Fact]
    public void TensorsThatAreNotFactors_AreRefusedByName()
    {
        string path = LoraFixtures.WriteRaw(Path.Combine(_dir, "mixed.safetensors"),
            "{\"w.lora_a\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]}," +
            "\"w.lora_b\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[4,8]}," +
            "\"output.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[8,12]}}",
            new byte[12]);

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path));
        Assert.Contains("'output.weight'", ex.Message);
        Assert.Contains("not LoRA factors", ex.Message);
    }

    [Fact]
    public void ModulesMayHaveTheirOwnRank_EachScaledByItsOwn()
    {
        var (a4, b4) = TextLoraFactors.Random(1, 4, 6, 5);
        var (a8, b8) = TextLoraFactors.Random(2, 8, 6, 7);
        string path = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "mixed.safetensors"),
            new[] { ("blk.0.attn_q.weight", a4, b4, 4, 6, 5), ("blk.0.attn_k.weight", a8, b8, 8, 6, 7) }, alpha: 16f);

        var adapter = TextLoraLoader.Load("x", path);

        Assert.Equal(4, adapter.Deltas["blk.0.attn_q.weight"].Rank);
        Assert.Equal(8, adapter.Deltas["blk.0.attn_k.weight"].Rank);
        Assert.Equal(8, adapter.Rank);
        Assert.Contains("r=4-8", adapter.Summary);
        Assert.Equal(4f * b4[0], adapter.Deltas["blk.0.attn_q.weight"].BT[0], 5);   // 16 / 4
        Assert.Equal(2f * b8[0], adapter.Deltas["blk.0.attn_k.weight"].BT[0], 5);   // 16 / 8

        // A config's r is PEFT's one rank for every module: a module of another rank does
        // not match it.
        string config = Path.Combine(_dir, "cfg.json");
        File.WriteAllText(config, "{ \"r\": 8, \"lora_alpha\": 16 }");
        Assert.Contains("config declares 8", Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path, config)).Message);
    }

    /// <summary>
    /// A safetensors factor pair with an empty dimension is refused, as a GGUF one is. A zero
    /// dimension leaves both factors (or one) with zero bytes, which every size check accepts:
    /// rank 0 loaded an adapter scaled by alpha / 0, an empty in or out dimension one that
    /// changes nothing, and [0, 2^60] / [2^60, 0] looped 2^60 times building the delta, so the
    /// load never returned (hence the timeout).
    /// </summary>
    [Theory]
    [InlineData(0L, 4L, 3L)]
    [InlineData(2L, 0L, 3L)]
    [InlineData(2L, 4L, 0L)]
    [InlineData(0L, 1L << 60, 1L << 60)]
    public async Task SafetensorsAdapter_WithAnEmptyDimension_IsRefused(long rank, long inDim, long outDim)
    {
        long aBytes = rank * inDim * 4, bBytes = outDim * rank * 4;
        string path = LoraFixtures.WriteRaw(Path.Combine(_dir, "empty.safetensors"),
            "{\"w.lora_a\":{\"dtype\":\"F32\",\"shape\":[" + rank + "," + inDim + "],\"data_offsets\":[0," + aBytes + "]},"
            + "\"w.lora_b\":{\"dtype\":\"F32\",\"shape\":[" + outDim + "," + rank + "],\"data_offsets\":[" + aBytes + "," + (aBytes + bBytes) + "]}}",
            new byte[aBytes + bBytes]);

        var ex = await Assert.ThrowsAsync<InvalidDataException>(
            () => Task.Run(() => TextLoraLoader.Load("x", path)).WaitAsync(TimeSpan.FromSeconds(30)));

        Assert.Equal("LoRA tensor 'w' in 'empty.safetensors' has an empty dimension.", ex.Message);
    }

    // ------------------------------------------------------------------ alpha and scale

    /// <summary>
    /// A safetensors <c>__metadata__</c> alpha that is not a finite number >= 0 is refused with
    /// the reason, as a GGUF adapter's <c>adapter.lora.alpha</c> is. An infinite one (spelled
    /// out, or past float's range) loaded with <c>BT = +Inf</c> and an alpha no adapter listing
    /// could serialize; a negative or NaN one was replaced by the rank, applying the adapter at
    /// scale +1 where PEFT applies <c>alpha / r</c>.
    /// </summary>
    [Theory]
    [InlineData("Infinity", "Infinity")]
    [InlineData("1e39", "Infinity")]
    [InlineData("NaN", "NaN")]
    [InlineData("-16", "-16")]
    [InlineData("-Infinity", "-Infinity")]
    public void SafetensorsMetadataAlpha_ThatIsNotAFiniteNumberAtLeastZero_IsRefused(string alpha, string read)
    {
        string path = LoraFixtures.WriteAdapter(Path.Combine(_dir, "meta.safetensors"), "w", metadataAlpha: alpha);

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path));

        Assert.Equal($"'meta.safetensors' has __metadata__ alpha {read}; it must be a finite number >= 0.", ex.Message);
    }

    /// <summary>One that is not a number at all is refused too, as in a GGUF adapter, never read as no alpha.</summary>
    [Fact]
    public void SafetensorsMetadataAlpha_ThatIsNotANumber_IsRefused()
    {
        string path = LoraFixtures.WriteAdapter(Path.Combine(_dir, "meta.safetensors"), "w", metadataAlpha: "sixteen");

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path));

        Assert.Equal("'meta.safetensors' has a __metadata__ alpha that is not a number.", ex.Message);
    }

    /// <summary>
    /// The same for <c>adapter_config.json</c>'s <c>lora_alpha</c>, beside the weights or
    /// named, and although the file's own metadata alpha is valid: the config is the more
    /// specific source, and its bad value is refused rather than passed over. JSON has no NaN
    /// or infinity, so those arrive as strings (not a number), and a number past float's range
    /// reads as infinity.
    /// </summary>
    [Theory]
    [InlineData("1e39", "has lora_alpha Infinity; it must be a finite number >= 0.")]
    [InlineData("-16", "has lora_alpha -16; it must be a finite number >= 0.")]
    [InlineData("-1e39", "has lora_alpha -Infinity; it must be a finite number >= 0.")]
    [InlineData("\"Infinity\"", "has a lora_alpha that is not a number.")]
    [InlineData("\"NaN\"", "has a lora_alpha that is not a number.")]
    [InlineData("\"16\"", "has a lora_alpha that is not a number.")]
    [InlineData("null", "has a lora_alpha that is not a number.")]
    public void AdapterConfigLoraAlpha_ThatIsNotAFiniteNumberAtLeastZero_IsRefused(string alpha, string reason)
    {
        string path = LoraFixtures.WriteAdapter(Path.Combine(_dir, "a.safetensors"), "w", metadataAlpha: "8");
        string config = Path.Combine(_dir, "adapter_config.json");
        File.WriteAllText(config, "{ \"r\": 1, \"lora_alpha\": " + alpha + " }");

        foreach (string? named in new[] { null, config })
        {
            var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path, named));
            Assert.Equal("LoRA config 'adapter_config.json' " + reason, ex.Message);
        }
    }

    [Theory]
    [InlineData(float.PositiveInfinity, "Infinity")]
    [InlineData(float.NaN, "NaN")]
    [InlineData(-16f, "-16")]
    public void GgufAdapterAlpha_ThatIsNotAFiniteNumberAtLeastZero_IsRefusedTheSameWay(float alpha, string read)
    {
        string path = TextLoraFactors.WriteGguf(Path.Combine(_dir, "a.gguf"), "mistral3", alpha,
            new[] { ("w", new[] { 1f }, new[] { 1f }, 1, 1, 1) });

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path));

        Assert.Equal($"'a.gguf' has adapter.lora.alpha {read}; it must be a finite number >= 0.", ex.Message);
    }

    /// <summary>
    /// <c>adapter.lora.alpha</c> must be one FLOAT32 value: llama.cpp reads it with
    /// <c>gguf_get_val_f32</c>, which asserts that type and converts no other, numeric or not.
    /// It was read through <c>Convert.ToSingle</c>, so a BOOL loaded as alpha 1, a STRING was
    /// parsed in the current culture ("16,5" as 165 where the comma groups thousands), an
    /// integer gave its value and an array its first element. A one-element FLOAT32 array, which
    /// <c>gguf_get_val_f32</c> would also take, is refused too: no converter writes one.
    /// </summary>
    [Theory]
    [InlineData("BOOL", "BOOL")]
    [InlineData("STRING", "STRING")]
    [InlineData("ARRAY", "ARRAY")]
    [InlineData("ARRAY1", "ARRAY")]
    [InlineData("UINT32", "UINT32")]
    public void GgufAdapterAlpha_ThatIsNotOneFloat32_IsRefused(string kind, string type)
    {
        var kv = TextLoraFactors.AdapterMetadata("mistral3", 1f);
        kv.RemoveAll(k => k.Key == "adapter.lora.alpha");
        kv.Add(kind switch
        {
            "BOOL" => new SyntheticGguf.Bool { Key = "adapter.lora.alpha", V = true },
            "STRING" => new SyntheticGguf.Str { Key = "adapter.lora.alpha", V = "16" },
            "ARRAY" => new SyntheticGguf.F32Arr { Key = "adapter.lora.alpha", V = new[] { 16f, 8f } },
            "ARRAY1" => new SyntheticGguf.F32Arr { Key = "adapter.lora.alpha", V = new[] { 16f } },
            _ => new SyntheticGguf.U32 { Key = "adapter.lora.alpha", V = 16 },
        });
        string path = Path.Combine(_dir, "a.gguf");
        SyntheticGguf.Write(path, kv, new List<SyntheticGguf.Tensor>
        {
            new() { Name = "w.lora_a", Dims = new ulong[] { 1, 1 }, Data = new[] { 1f } },
            new() { Name = "w.lora_b", Dims = new ulong[] { 1, 1 }, Data = new[] { 1f } },
        });

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path));

        Assert.Equal($"'a.gguf' has an adapter.lora.alpha of GGUF type {type}; it must be a FLOAT32 number.", ex.Message);
    }

    /// <summary>
    /// An alpha of 0 means none, <c>alpha = rank</c> (scale 1), in every source, as llama.cpp
    /// reads <c>adapter.lora.alpha</c>, and so does no alpha at all. A config's 0 supplies none,
    /// so the file's metadata alpha still applies under it.
    /// </summary>
    [Fact]
    public void ZeroOrMissingAlpha_MeansAlphaIsTheRank_InEverySource()
    {
        const int rank = 4;
        var (a, b) = TextLoraFactors.Random(5, rank, 3, 2);
        var targets = new[] { ("w", a, b, rank, 3, 2) };
        string gguf = TextLoraFactors.WriteGguf(Path.Combine(_dir, "zero.gguf"), "mistral3", 0f, targets);
        string metadataZero = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "zero.safetensors"), targets, alpha: 0f);
        string none = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "none.safetensors"), targets, alpha: null);
        string metadataEight = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "eight.safetensors"), targets, alpha: 8f);
        string configZero = Path.Combine(_dir, "zero.json");
        File.WriteAllText(configZero, "{ \"r\": 4, \"lora_alpha\": 0 }");

        foreach (var adapter in new[]
                 {
                     TextLoraLoader.Load("x", gguf), TextLoraLoader.Load("x", metadataZero),
                     TextLoraLoader.Load("x", none), TextLoraLoader.Load("x", none, configZero),
                 })
        {
            Assert.Equal(rank, adapter.Alpha);
            Assert.Equal(b[0], adapter.Deltas["w"].BT[0]);   // B[0, 0] at scale 1
        }
        var underZero = TextLoraLoader.Load("x", metadataEight, configZero);
        Assert.Equal(8f, underZero.Alpha);
        Assert.Equal(2f * b[0], underZero.Deltas["w"].BT[0], 5);   // 8 / 4
    }

    /// <summary>
    /// Scale 1 holds under <c>use_rslora</c> too, as the docs promise and as a GGUF adapter
    /// reads it. Standing in the rank for the missing alpha gave <c>r / sqrt(r) = sqrt(r)</c>:
    /// twice the delta at rank 4. The adapter is still listed with <c>alpha = r</c>.
    /// </summary>
    [Fact]
    public void ZeroOrMissingAlpha_IsScaleOne_UnderRsloraToo()
    {
        const int rank = 4;
        var (a, b) = TextLoraFactors.Random(13, rank, 3, 2);
        var targets = new[] { ("w", a, b, rank, 3, 2) };
        string none = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "none.safetensors"), targets, alpha: null);
        string metadataZero = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "zero.safetensors"), targets, alpha: 0f);
        string missing = Path.Combine(_dir, "missing.json"), zero = Path.Combine(_dir, "zero.json");
        File.WriteAllText(missing, "{ \"r\": 4, \"use_rslora\": true }");
        File.WriteAllText(zero, "{ \"r\": 4, \"lora_alpha\": 0, \"use_rslora\": true }");

        foreach (var adapter in new[]
                 {
                     TextLoraLoader.Load("x", none, missing), TextLoraLoader.Load("x", none, zero),
                     TextLoraLoader.Load("x", metadataZero, missing),
                 })
        {
            Assert.Equal(rank, adapter.Alpha);
            Assert.Contains("alpha=rank, rslora", adapter.Summary);
            for (int o = 0; o < 2; o++)
                for (int j = 0; j < rank; j++)
                    Assert.Equal(b[o * rank + j], adapter.Deltas["w"].BT[j * 2 + o]);   // B at scale 1
        }
    }

    /// <summary>
    /// A finite <c>alpha / rank</c> times the scale given at load can still overflow: refused
    /// in both containers, never loaded as a delta of infinities. At its own scale the same
    /// file loads.
    /// </summary>
    [Fact]
    public void AScaleThatOverflows_IsRefused_InBothContainers()
    {
        var targets = new[] { ("w", new[] { 1f }, new[] { 1f }, 1, 1, 1) };
        string gguf = TextLoraFactors.WriteGguf(Path.Combine(_dir, "big.gguf"), "mistral3", 3e38f, targets);
        string safetensors = TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "big.safetensors"), targets, alpha: 3e38f);

        foreach (string path in new[] { gguf, safetensors })
        {
            var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path, scaleOverride: 10f));
            Assert.Equal($"LoRA module 'w' in '{Path.GetFileName(path)}' has scale Infinity " +
                         "(alpha / rank times the scale given at load); it must be a finite number.", ex.Message);
            Assert.Equal(3e38f, TextLoraLoader.Load("x", path).Deltas["w"].BT[0]);
        }
    }

    /// <summary>
    /// A factor holding NaN or infinity, or one that overflows once the scale (16 here) is
    /// folded in, would turn every output of its module into NaN or infinity: refused in both
    /// containers, naming the tensor.
    /// </summary>
    [Theory]
    [InlineData("gguf", "a-nan", "LoRA tensor 'w.lora_a' in '{0}' holds a value that is not a finite number.")]
    [InlineData("gguf", "b-infinity", "LoRA tensor 'w.lora_b' in '{0}' holds a value that is not a finite number.")]
    [InlineData("gguf", "b-overflow", "LoRA tensor 'w.lora_b' in '{0}' overflows once scaled by 16.")]
    [InlineData("safetensors", "a-nan", "LoRA tensor 'w.lora_a' in '{0}' holds a value that is not a finite number.")]
    [InlineData("safetensors", "b-infinity", "LoRA tensor 'w.lora_b' in '{0}' holds a value that is not a finite number.")]
    [InlineData("safetensors", "b-overflow", "LoRA tensor 'w.lora_b' in '{0}' overflows once scaled by 16.")]
    public void AFactorThatIsNotFinite_IsRefused_InBothContainers(string container, string kind, string reason)
    {
        float[] a = { 1f, kind == "a-nan" ? float.NaN : 1f };   // [rank 1, in 2]
        float[] b = { 1f, kind switch { "b-infinity" => float.PositiveInfinity, "b-overflow" => 3e38f, _ => 1f } };   // [out 2, rank 1]
        var targets = new[] { ("w", a, b, 1, 2, 2) };
        string path = container == "gguf"
            ? TextLoraFactors.WriteGguf(Path.Combine(_dir, "f.gguf"), "mistral3", 16f, targets)
            : TextLoraFactors.WriteSafetensors(Path.Combine(_dir, "f.safetensors"), targets, alpha: 16f);

        var ex = Assert.Throws<InvalidDataException>(() => TextLoraLoader.Load("x", path));

        Assert.Equal(string.Format(System.Globalization.CultureInfo.InvariantCulture, reason, Path.GetFileName(path)), ex.Message);
    }

    /// <summary>The listings write alpha as a JSON number, which has no NaN or infinity: no
    /// adapter carries such an alpha, whatever made it.</summary>
    [Theory]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NaN)]
    [InlineData(-1f)]
    public void AnAdapter_WithAnAlphaThatIsNotAFiniteNumberAtLeastZero_CannotBeMade(float alpha)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new TextLoraAdapter("x", 1, alpha, new Dictionary<string, LoraDelta>(), "x"));
    }
}

/// <summary>Random LoRA factors and the two adapter containers, for the text-LoRA tests.</summary>
internal static class TextLoraFactors
{
    /// <summary>Seeded <c>A [rank, in]</c> and <c>B [out, rank]</c>, row-major.</summary>
    public static (float[] A, float[] B) Random(int seed, int rank, int inDim, int outDim, float scale = 0.5f)
    {
        var rng = new System.Random(seed);
        float Next() => (float)(rng.NextDouble() * 2 - 1) * scale;
        var a = new float[rank * inDim];
        var b = new float[outDim * rank];
        for (int i = 0; i < a.Length; i++) a[i] = Next();
        for (int i = 0; i < b.Length; i++) b[i] = Next();
        return (a, b);
    }

    public static List<SyntheticGguf.Kv> AdapterMetadata(string architecture, float alpha,
        string generalType = "adapter", string adapterType = "lora", bool alora = false)
    {
        var kv = new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.Str { Key = "general.architecture", V = architecture },
            new SyntheticGguf.Str { Key = "general.type", V = generalType },
            new SyntheticGguf.Str { Key = "adapter.type", V = adapterType },
            new SyntheticGguf.F32 { Key = "adapter.lora.alpha", V = alpha },
        };
        if (alora)
            kv.Add(new SyntheticGguf.U32Arr { Key = "adapter.alora.invocation_tokens", V = new uint[] { 1, 2 } });
        return kv;
    }

    /// <summary>A llama.cpp GGUF adapter: <c>lora_a</c> ne <c>[in, rank]</c>, <c>lora_b</c> ne <c>[rank, out]</c>.</summary>
    public static string WriteGguf(string path, string architecture, float alpha,
        IEnumerable<(string Target, float[] A, float[] B, int Rank, int In, int Out)> targets,
        SyntheticGguf.GgmlType type = SyntheticGguf.GgmlType.F32,
        string generalType = "adapter", string adapterType = "lora", bool alora = false)
    {
        var tensors = new List<SyntheticGguf.Tensor>();
        foreach (var (target, a, b, rank, inDim, outDim) in targets)
        {
            tensors.Add(new SyntheticGguf.Tensor { Name = target + ".lora_a", Dims = new[] { (ulong)inDim, (ulong)rank }, Data = a, Type = type });
            tensors.Add(new SyntheticGguf.Tensor { Name = target + ".lora_b", Dims = new[] { (ulong)rank, (ulong)outDim }, Data = b, Type = type });
        }
        SyntheticGguf.Write(path, AdapterMetadata(architecture, alpha, generalType, adapterType, alora), tensors);
        return path;
    }

    /// <summary>
    /// A one-module GGUF adapter (<c>w</c>, rank 1, F32) with one corrupt header field:
    /// <c>dims</c> gives <c>w.lora_a</c> int.MaxValue dimensions, <c>ne0</c> a row of
    /// 0x1FFFFFFF values that its 4 bytes of data cannot hold, <c>array</c> a metadata array
    /// int.MaxValue values long and <c>string</c> an architecture int.MaxValue bytes long. A
    /// reader that sizes an array from any of these throws OutOfMemoryException (each is past
    /// the largest .NET array) rather than allocating. <c>ne0-wrap</c> is a row of 2^62 + 1
    /// values, whose F32 byte count wraps round to exactly those 4 bytes, and
    /// <c>alignment</c> a <c>general.alignment</c> of 0, which the reader divides by.
    /// </summary>
    public static string WriteCorruptGguf(string path, string kind)
    {
        var kv = AdapterMetadata("probe", 1f);
        kv.Add(new SyntheticGguf.U32Arr { Key = "adapter.lora.probe", V = new uint[] { 7 } });
        if (kind == "alignment")
            kv.Add(new SyntheticGguf.U32 { Key = "general.alignment", V = 0 });
        SyntheticGguf.Write(path, kv, new List<SyntheticGguf.Tensor>
        {
            new()
            {
                Name = "w.lora_a",
                Dims = new ulong[] { kind switch { "ne0" => 0x1FFF_FFFFUL, "ne0-wrap" => 0x4000_0000_0000_0001UL, _ => 1UL }, 1 },
                Data = new[] { 1f },
            },
            new() { Name = "w.lora_b", Dims = new ulong[] { 1, 1 }, Data = new[] { 1f } },
        });
        switch (kind)
        {
            case "dims":
                Patch("w.lora_a", 0, BitConverter.GetBytes((uint)int.MaxValue));             // n_dims follows the name
                break;
            case "array":
                Patch("adapter.lora.probe", 8, BitConverter.GetBytes((ulong)int.MaxValue));   // after the value and element types
                break;
            case "string":
                Patch("general.architecture", 4, BitConverter.GetBytes((ulong)int.MaxValue)); // the value's length, after its type
                break;
            case "ne0":
            case "ne0-wrap":
            case "alignment":
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
        }
        return path;

        void Patch(string after, int skip, byte[] value)
        {
            byte[] file = File.ReadAllBytes(path);
            int at = file.AsSpan().IndexOf(Encoding.UTF8.GetBytes(after)) + after.Length + skip;
            value.CopyTo(file, at);
            File.WriteAllBytes(path, file);
        }
    }

    /// <summary>An F32 safetensors adapter, with alpha in <c>__metadata__</c> when given.</summary>
    public static string WriteSafetensors(string path,
        IEnumerable<(string Target, float[] A, float[] B, int Rank, int In, int Out)> targets, float? alpha)
    {
        var header = new JsonObject();
        if (alpha is { } al)
            header["__metadata__"] = new JsonObject { ["alpha"] = al.ToString(System.Globalization.CultureInfo.InvariantCulture) };
        var data = new List<byte>();
        foreach (var (target, a, b, rank, inDim, outDim) in targets)
        {
            Add(target + ".lora_a", a, rank, inDim);
            Add(target + ".lora_b", b, outDim, rank);
        }
        byte[] hb = Encoding.UTF8.GetBytes(header.ToJsonString());
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)hb.Length));
        fs.Write(hb);
        fs.Write(data.ToArray());
        return path;

        void Add(string name, float[] values, int rows, int cols)
        {
            header[name] = new JsonObject
            {
                ["dtype"] = "F32",
                ["shape"] = new JsonArray(rows, cols),
                ["data_offsets"] = new JsonArray(data.Count, data.Count + values.Length * 4),
            };
            foreach (float v in values) data.AddRange(BitConverter.GetBytes(v));
        }
    }

    /// <summary><c>W + scale * B @ A</c> for a row-major <c>W [out, in]</c>, accumulated in double.</summary>
    public static float[] Merge(float[] w, float[] a, float[] b, int rank, int inDim, int outDim, float scale)
    {
        var merged = new float[w.Length];
        for (int o = 0; o < outDim; o++)
            for (int i = 0; i < inDim; i++)
            {
                double acc = 0;
                for (int j = 0; j < rank; j++)
                    acc += (double)b[o * rank + j] * a[j * inDim + i];
                merged[o * inDim + i] = (float)(w[o * inDim + i] + scale * acc);
            }
        return merged;
    }
}
