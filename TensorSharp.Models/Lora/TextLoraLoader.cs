// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// Loads text-LoRA adapters. Both accepted containers name each target by the base
    /// GGUF's own tensor name with a <c>.lora_a</c> / <c>.lora_b</c> suffix, which is
    /// llama.cpp's convention (and PEFT's factor orientation, <c>lora_A [rank, in]</c>,
    /// <c>lora_B [out, rank]</c>):
    ///
    /// <list type="bullet">
    /// <item>A llama.cpp GGUF adapter, as <c>convert_lora_to_gguf.py</c> writes it:
    /// <c>general.type = adapter</c>, <c>adapter.type = lora</c>, alpha in
    /// <c>adapter.lora.alpha</c> (0 means scale 1), and <c>general.architecture</c>, which the
    /// model checks against its own. F32, F16, BF16 and Q8_0 factors are read.</item>
    /// <item>A <c>.safetensors</c> file of F32/F16/BF16/F64 factors, with <c>r</c> and
    /// <c>lora_alpha</c> from an optional PEFT <c>adapter_config.json</c> or the
    /// <c>__metadata__</c> <c>rank</c> / <c>alpha</c> keys (without an alpha, scale 1).</item>
    /// </list>
    ///
    /// <code>
    /// blk.0.attn_q.weight.lora_a     [rank, in]
    /// blk.0.attn_q.weight.lora_b     [out, rank]
    /// blk.0.ffn_down.weight.lora_a
    /// blk.0.ffn_down.weight.lora_b
    /// output.weight.lora_a
    /// output.weight.lora_b
    /// </code>
    ///
    /// <para>
    /// Projections TensorSharp fuses at load (attn_q/k/v into attn_qkv, ffn_gate/up into
    /// ffn_gate_up) keep their GGUF names here; the model resolves them to the fused weight
    /// (see <see cref="TextLoraBinding"/>). The fused names are accepted too, but whether a
    /// layer is fused depends on the quantization, so the GGUF names are the stable ones.
    /// </para>
    ///
    /// <para>
    /// Each module may have its own rank; its scale is <c>alpha / rank</c>
    /// (<c>alpha / sqrt(rank)</c> with PEFT's <c>use_rslora</c>), times the caller's scale.
    /// In both containers an alpha of 0, or none, means scale 1, with or without
    /// <c>use_rslora</c>, which is how llama.cpp reads <c>adapter.lora.alpha</c>; the adapter
    /// then reports <c>alpha = rank</c>. A GGUF alpha must be one FLOAT32 value, as llama.cpp's
    /// <c>gguf_get_val_f32</c> requires. An alpha that is negative or not a finite
    /// number is refused from any source, never replaced by the rank, and so is a module whose
    /// scale or factors are not finite (a NaN or infinity in the file, or a value that overflows
    /// once the scale is folded in).
    /// What cannot be applied exactly is refused, never skipped: PEFT / Hugging Face module
    /// names (mapping them onto GGUF tensors needs each architecture's conversion, applied to
    /// A and B, which is what llama.cpp's <c>convert_lora_to_gguf.py</c> does), DoRA,
    /// <c>fan_in_fan_out</c>, <c>alpha_pattern</c>, <c>rank_pattern</c>,
    /// <c>modules_to_save</c>, <c>trainable_token_indices</c>, trained biases (<c>bias</c>
    /// other than <c>"none"</c>, <c>lora_bias</c>), <c>layer_replication</c>, activated LoRA
    /// (GGUF <c>adapter.alora.*</c>, PEFT <c>alora_invocation_tokens</c> or the older
    /// <c>invocation_string</c>), the PEFT LoRA variants KaSA, Arrow and BD-LoRA, a PEFT
    /// <c>init_lora_weights</c> that rewrites the base weights (PiSSA, OLoRA, CorDA, Astra,
    /// LoftQ, LoRA-GA) in an adapter saved without weight conversion, a PEFT flag that is not
    /// <c>true</c> or <c>false</c>, and any tensor that is not a factor.
    /// </para>
    /// </summary>
    public static class TextLoraLoader
    {
        internal const string ConverterAdvice =
            "convert the PEFT adapter with llama.cpp's convert_lora_to_gguf.py " +
            "(python convert_lora_to_gguf.py <peft_adapter_dir> --base <hf_base_model_dir> --outtype f16) " +
            "and load the .gguf it writes";

        /// <param name="discoverSiblingConfig">
        /// When <paramref name="configPath"/> is null, look for an <c>adapter_config.json</c>
        /// next to the weights. A caller that confines file access (the server's
        /// <c>--loras-dir</c>) resolves that sibling through its own guard and passes false,
        /// so the loader never opens a file the guard did not check. A GGUF adapter carries its
        /// own alpha and reads no config.
        /// </param>
        public static TextLoraAdapter Load(
            string name,
            string safetensorsPath,
            string? configPath = null,
            float? scaleOverride = null,
            bool discoverSiblingConfig = true)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("A LoRA adapter name is required.", nameof(name));
            if (!File.Exists(safetensorsPath))
                throw new FileNotFoundException($"LoRA weights file not found: {safetensorsPath}", safetensorsPath);
            if (IsGguf(safetensorsPath))
                return LoadGguf(name, safetensorsPath, scaleOverride);

            configPath = discoverSiblingConfig ? ResolveConfigPath(safetensorsPath, configPath) : configPath;
            int configuredRank = 0;
            float configuredAlpha = 0f;
            bool rslora = false;
            if (configPath != null)
                ReadConfig(configPath, ref configuredRank, ref configuredAlpha, ref rslora);

            using var st = OpenSafetensors(safetensorsPath);
            string file = Path.GetFileName(safetensorsPath);

            // A lora_a tensor pairs with a lora_b tensor on the same target weight.
            var pairs = PairFactors(st.Tensors, file);

            // Metadata rank/alpha are the least specific source, so only use them when
            // the config did not already supply a value (an alpha of 0 supplies none).
            if (configuredRank <= 0 || configuredAlpha <= 0f)
            {
                st.Metadata.TryGetValue("rank", out string? metaRank);
                st.Metadata.TryGetValue("alpha", out string? metaAlpha);
                if (configuredRank <= 0 && metaRank != null && int.TryParse(metaRank, NumberStyles.Integer, CultureInfo.InvariantCulture, out int mr))
                    configuredRank = mr;
                if (configuredAlpha <= 0f && metaAlpha != null)
                {
                    // float.TryParse reads "Infinity" and "NaN", and a value past float's range as
                    // infinity; RequireAlpha refuses all three.
                    if (!float.TryParse(metaAlpha, NumberStyles.Float, CultureInfo.InvariantCulture, out float ma))
                        throw new InvalidDataException($"'{file}' has a __metadata__ alpha that is not a number.");
                    configuredAlpha = RequireAlpha(ma, $"'{file}' has __metadata__ alpha");
                }
            }

            var deltas = new Dictionary<string, LoraDelta>(StringComparer.Ordinal);
            foreach (var (target, (aInfo, bInfo)) in pairs)
            {
                RequireFloatFactor(aInfo);
                RequireFloatFactor(bInfo);

                if (aInfo.Shape.Length != 2 || bInfo.Shape.Length != 2)
                    throw new InvalidDataException(
                        $"LoRA tensor '{target}.lora_a/lora_b' must be rank-2, got {string.Join("x", aInfo.Shape)} / {string.Join("x", bInfo.Shape)}.");

                long rankA = aInfo.Shape[0];
                long inDim = aInfo.Shape[1];
                long outDim = bInfo.Shape[0];
                long rankB = bInfo.Shape[1];

                if (rankA != rankB)
                    throw new InvalidDataException(
                        $"LoRA rank mismatch for '{target}': lora_a rank {rankA} != lora_b rank {rankB}.");
                // As in a GGUF adapter. A zero dimension makes both factors zero bytes, which
                // the size checks accept, while the other dimensions stay unbounded: BuildDelta
                // would loop over them ([0, 2^60] / [2^60, 0] never returns), and rank 0 scales
                // by alpha / 0. Every dimension at least 1 bounds each one by the bytes present.
                if (rankA < 1 || inDim < 1 || outDim < 1)
                    throw new InvalidDataException($"LoRA tensor '{target}' in '{file}' has an empty dimension.");
                // A declared rank is PEFT's one r for every module (rank_pattern is refused),
                // so a module of another rank is a file that does not match its config.
                if (configuredRank > 0 && configuredRank != rankA)
                    throw new InvalidDataException(
                        $"LoRA rank mismatch for '{target}': file has rank {rankA}, config declares {configuredRank}.");

                float[] a = ReadFactor(st, aInfo, rankA, inDim);
                float[] b = ReadFactor(st, bInfo, outDim, rankB);
                deltas[target] = BuildDelta(a, b, checked((int)rankA), inDim, outDim,
                    ModuleScale(configuredAlpha, (int)rankA, rslora, scaleOverride), target, file);
            }

            int rank = configuredRank > 0 ? configuredRank : deltas.Values.Max(d => d.Rank);
            float resolvedAlpha = configuredAlpha > 0f ? configuredAlpha : rank;
            string summary =
                $"lora {name}: {file} (r={DescribeRanks(deltas)}, " +
                $"alpha={(configuredAlpha > 0f ? configuredAlpha.ToString(CultureInfo.InvariantCulture) : "rank")}" +
                $"{(rslora ? ", rslora" : "")}, modules={deltas.Count})" +
                (configPath != null ? $" config={Path.GetFileName(configPath)}" : string.Empty);
            return new TextLoraAdapter(name, rank, resolvedAlpha, deltas, summary);
        }

        // ------------------------------------------------------------------ GGUF

        private static bool IsGguf(string path)
        {
            if (string.Equals(Path.GetExtension(path), ".gguf", StringComparison.OrdinalIgnoreCase))
                return true;
            Span<byte> magic = stackalloc byte[4];
            using var fs = File.OpenRead(path);
            return fs.Read(magic) == 4 && magic.SequenceEqual("GGUF"u8);
        }

        private static TextLoraAdapter LoadGguf(string name, string path, float? scaleOverride)
        {
            string file = Path.GetFileName(path);
            using var gguf = OpenGguf(path, file);

            string? type = gguf.GetString("general.type");
            if (!string.Equals(type, "adapter", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException(
                    $"'{file}' is not a LoRA adapter: its general.type is {(type == null ? "missing" : $"'{type}'")}, not 'adapter'. " +
                    "A model GGUF cannot be loaded as an adapter.");
            string? adapterType = gguf.GetString("adapter.type");
            if (!string.Equals(adapterType, "lora", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"'{file}' is a {(adapterType == null ? "GGUF adapter with no adapter.type" : $"'{adapterType}' adapter")}; only LoRA adapters (adapter.type = lora) are supported.");
            if (gguf.Metadata.Keys.Any(k => k.StartsWith("adapter.alora.", StringComparison.Ordinal)))
                throw new NotSupportedException(
                    $"'{file}' is an activated LoRA (aLoRA), which applies only after its invocation tokens; text LoRA applies an adapter to every token, so it is refused.");

            float alpha = 0f;
            if (gguf.Metadata.TryGetValue("adapter.lora.alpha", out object? alphaValue))
            {
                // llama.cpp reads it with gguf_get_val_f32, which asserts the FLOAT32 type and one
                // element (no other type, numeric or not, is converted), and
                // convert_lora_to_gguf.py writes one FLOAT32. GetFloat32 would convert anything:
                // a BOOL to 1, a STRING parsed in the current culture ("16,5" as 165), an array's
                // first element. An array is refused at any length, the one-element FLOAT32 array
                // gguf_get_val_f32 would also take included: no converter writes one.
                if (alphaValue is not float value)
                    throw new InvalidDataException(
                        $"'{file}' has an adapter.lora.alpha of GGUF type {GgufTypeName(alphaValue)}; it must be a FLOAT32 number.");
                alpha = RequireAlpha(value, $"'{file}' has adapter.lora.alpha");
            }
            string? architecture = gguf.GetString("general.architecture");

            var pairs = PairFactors(gguf.Tensors, file);
            var deltas = new Dictionary<string, LoraDelta>(StringComparer.Ordinal);
            foreach (var (target, (aInfo, bInfo)) in pairs)
            {
                // GGUF lists ne0 first: lora_a is ne [in, rank] (row-major [rank, in]) and
                // lora_b is ne [rank, out] (row-major [out, rank]).
                if (aInfo.Shape.Length != 2 || bInfo.Shape.Length != 2)
                    throw new InvalidDataException(
                        $"LoRA tensor '{target}.lora_a/lora_b' in '{file}' must be 2-D, got ne [{string.Join(", ", aInfo.Shape)}] / [{string.Join(", ", bInfo.Shape)}].");
                long inDim = (long)aInfo.Shape[0];
                long rankA = (long)aInfo.Shape[1];
                long rankB = (long)bInfo.Shape[0];
                long outDim = (long)bInfo.Shape[1];
                if (rankA != rankB)
                    throw new InvalidDataException(
                        $"LoRA rank mismatch for '{target}' in '{file}': lora_a rank {rankA} != lora_b rank {rankB}.");
                if (rankA < 1 || inDim < 1 || outDim < 1)
                    throw new InvalidDataException($"LoRA tensor '{target}' in '{file}' has an empty dimension.");

                float[] a = ReadGgufFactor(gguf, aInfo, file);
                float[] b = ReadGgufFactor(gguf, bInfo, file);
                // llama.cpp: scale = alpha ? alpha / rank : 1 (times the caller's scale).
                float scale = alpha > 0f ? alpha / rankA : 1f;
                if (scaleOverride is { } userScale)
                    scale *= userScale;
                deltas[target] = BuildDelta(a, b, checked((int)rankA), inDim, outDim, scale, target, file);
            }

            int rank = deltas.Values.Max(d => d.Rank);
            float resolvedAlpha = alpha > 0f ? alpha : rank;
            string summary =
                $"lora {name}: {file} (gguf{(architecture != null ? ", " + architecture : "")}, r={DescribeRanks(deltas)}, " +
                $"alpha={(alpha > 0f ? alpha.ToString(CultureInfo.InvariantCulture) : "rank")}, modules={deltas.Count})";
            return new TextLoraAdapter(name, rank, resolvedAlpha, deltas, summary, architecture);
        }

        /// <summary>The GGUF value type <see cref="GgufFile.Metadata"/> read
        /// <paramref name="value"/> from, by its CLR type.</summary>
        private static string GgufTypeName(object value) => value switch
        {
            byte => "UINT8",
            sbyte => "INT8",
            ushort => "UINT16",
            short => "INT16",
            uint => "UINT32",
            int => "INT32",
            ulong => "UINT64",
            long => "INT64",
            float => "FLOAT32",
            double => "FLOAT64",
            bool => "BOOL",
            string => "STRING",
            Array => "ARRAY",
            _ => value.GetType().Name,
        };

        /// <summary>
        /// Open the adapter as one GGUF file (never its "-00001-of-" siblings: a caller that
        /// confines file access checked only this one). A header the reader rejects is a
        /// malformed adapter; a short file stays an <see cref="EndOfStreamException"/>.
        /// </summary>
        private static GgufFile OpenGguf(string path, string file)
        {
            try
            {
                return GgufFile.OpenWithoutSiblingShards(path);
            }
            catch (Exception ex) when (ex is InvalidDataException or NotSupportedException or FormatException
                                           or InvalidCastException or OverflowException or ArgumentException)
            {
                throw new InvalidDataException($"'{file}' is not a valid GGUF file: {ex.Message}", ex);
            }
        }

        private static float[] ReadGgufFactor(GgufFile gguf, GgufTensorInfo info, string file)
        {
            var type = info.Type;
            if (type is not (GgmlTensorType.F32 or GgmlTensorType.F16 or GgmlTensorType.BF16 or GgmlTensorType.Q8_0))
                throw new InvalidDataException(
                    $"LoRA tensor '{info.Name}' in '{file}' is {type}; GGUF LoRA factors must be F32, F16, BF16 or Q8_0.");
            long ne0 = (long)info.Shape[0];
            if (ne0 % GgufFile.GetBlockSize(type) != 0)
                throw new InvalidDataException(
                    $"LoRA tensor '{info.Name}' in '{file}' is {type} with a row of {ne0}, not a multiple of its {GgufFile.GetBlockSize(type)}-value block.");
            // Through the managed dequantizer for every type: GgufFile's own float read is a
            // raw copy that is right only for F32.
            byte[] raw = gguf.ReadTensorData(info);
            var dst = new float[info.NumElements];
            ManagedQuantizedOps.DequantizeToFloat32((int)type, raw, 0, dst, 0, info.NumElements);
            return dst;
        }

        // ------------------------------------------------------------------ shared

        /// <summary>
        /// Pair every <c>&lt;target&gt;.lora_a</c> with its <c>.lora_b</c>. PEFT / Hugging
        /// Face module names, tensors that are not factors, and a factor without its partner are
        /// refused with the names, so nothing in the file is silently dropped.
        /// </summary>
        private static Dictionary<string, (T A, T B)> PairFactors<T>(IEnumerable<KeyValuePair<string, T>> tensors, string file)
            where T : class
        {
            var a = new Dictionary<string, T>(StringComparer.Ordinal);
            var b = new Dictionary<string, T>(StringComparer.Ordinal);
            var peft = new List<string>();
            var other = new List<string>();
            foreach (var (tensorName, info) in tensors)
            {
                if (IsPeftName(tensorName))
                    peft.Add(tensorName);
                else if (tensorName.EndsWith(".lora_a", StringComparison.Ordinal))
                    a[tensorName[..^".lora_a".Length]] = info;
                else if (tensorName.EndsWith(".lora_b", StringComparison.Ordinal))
                    b[tensorName[..^".lora_b".Length]] = info;
                else
                    other.Add(tensorName);
            }

            if (peft.Count > 0)
                throw new InvalidDataException(
                    $"LoRA file '{file}' uses PEFT / Hugging Face module names (e.g. '{peft.OrderBy(n => n, StringComparer.Ordinal).First()}'), " +
                    "not the base GGUF's tensor names. Those names cannot be mapped onto the model without the architecture's own " +
                    $"conversion of A and B, so they are refused: {ConverterAdvice}.");
            if (a.Count == 0 && b.Count == 0)
                throw new InvalidDataException(
                    $"LoRA file '{file}' contains no '<weight>.lora_a' / '<weight>.lora_b' tensor pairs.");
            if (other.Count > 0)
                throw new InvalidDataException(
                    $"LoRA file '{file}' holds tensors that are not LoRA factors ({NameList(other)}). Only " +
                    "'<weight>.lora_a' / '<weight>.lora_b' pairs are applied, and a fully trained module " +
                    "(modules_to_save) or a DoRA magnitude cannot be ignored without changing the result.");

            var pairs = new Dictionary<string, (T A, T B)>(StringComparer.Ordinal);
            foreach (var (target, aInfo) in a.OrderBy(p => p.Key, StringComparer.Ordinal))
            {
                if (!b.TryGetValue(target, out T? bInfo))
                    throw new InvalidDataException($"LoRA file '{file}' has no '{target}.lora_b' tensor for '{target}.lora_a'.");
                pairs[target] = (aInfo, bInfo);
            }
            foreach (string target in b.Keys)
            {
                if (!a.ContainsKey(target))
                    throw new InvalidDataException($"LoRA file '{file}' has no '{target}.lora_a' tensor for '{target}.lora_b'.");
            }
            return pairs;
        }

        /// <summary>PEFT's module paths (<c>base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight</c>),
        /// its embedding factors and its DoRA magnitudes.</summary>
        private static bool IsPeftName(string name)
            => name.StartsWith("base_model.", StringComparison.Ordinal)
               || name.Contains(".lora_A", StringComparison.Ordinal)
               || name.Contains(".lora_B", StringComparison.Ordinal)
               || name.Contains("lora_embedding_", StringComparison.Ordinal)
               || name.Contains("lora_magnitude_vector", StringComparison.Ordinal);

        private static string NameList(List<string> names)
        {
            const int Shown = 6;
            names.Sort(StringComparer.Ordinal);
            string list = string.Join(", ", names.Take(Shown).Select(n => $"'{n}'"));
            return names.Count > Shown ? $"{list} and {names.Count - Shown} more" : list;
        }

        /// <summary>
        /// An adapter's alpha, from any source. 0 means none (scale 1, listed as <c>alpha = rank</c>). A
        /// negative or non-finite value is refused rather than replaced by the rank, which would
        /// apply the adapter with another sign or magnitude than it was trained with; an infinite
        /// alpha would also be listed (<c>GET /v1/loras</c>, <c>GET /v1/models</c>) as a JSON
        /// number, which cannot hold one.
        /// </summary>
        private static float RequireAlpha(float alpha, string source)
        {
            if (!float.IsFinite(alpha) || alpha < 0f)
                throw new InvalidDataException(
                    $"{source} {alpha.ToString(CultureInfo.InvariantCulture)}; it must be a finite number >= 0.");
            return alpha;
        }

        /// <summary>
        /// The <see cref="LoraDelta"/> of <c>lora_A</c> (<paramref name="a"/>, row-major
        /// <c>[rank, in]</c>) and <c>lora_B</c> (<paramref name="b"/>, <c>[out, rank]</c>):
        /// both transposed, so the forward runs <c>x @ AT @ BT</c> as two plain GEMMs, with the
        /// scale folded into <c>BT</c>. A scale or factor value that is not finite is refused:
        /// it would turn every output of the module into NaN or infinity.
        /// </summary>
        private static LoraDelta BuildDelta(float[] a, float[] b, int rank, long inDim, long outDim, float scale,
            string target, string file)
        {
            // alpha / rank is finite for a finite alpha; times the caller's scale it can overflow.
            if (!float.IsFinite(scale))
                throw new InvalidDataException(
                    $"LoRA module '{target}' in '{file}' has scale {scale.ToString(CultureInfo.InvariantCulture)} " +
                    "(alpha / rank times the scale given at load); it must be a finite number.");
            var at = new float[a.Length];
            for (long r = 0; r < rank; r++)
                for (long c = 0; c < inDim; c++)
                {
                    float v = a[r * inDim + c];
                    if (!float.IsFinite(v))
                        throw new InvalidDataException($"LoRA tensor '{target}.lora_a' in '{file}' holds a value that is not a finite number.");
                    at[c * rank + r] = v;
                }
            var bt = new float[b.Length];
            for (long r = 0; r < outDim; r++)
                for (long c = 0; c < rank; c++)
                {
                    float v = scale * b[r * rank + c];
                    if (!float.IsFinite(v))
                        throw new InvalidDataException(float.IsFinite(b[r * rank + c])
                            ? $"LoRA tensor '{target}.lora_b' in '{file}' overflows once scaled by {scale.ToString(CultureInfo.InvariantCulture)}."
                            : $"LoRA tensor '{target}.lora_b' in '{file}' holds a value that is not a finite number.");
                    bt[c * outDim + r] = v;
                }
            return new LoraDelta { In = inDim, Out = outDim, Rank = rank, AT = at, BT = bt };
        }

        /// <summary>
        /// PEFT's scale: <c>alpha / r</c>, or <c>alpha / sqrt(r)</c> under rsLoRA. Without an
        /// alpha (0) it is 1 in either mode, as in a GGUF adapter: standing in the rank for the
        /// alpha would give <c>r / sqrt(r) = sqrt(r)</c> under rsLoRA.
        /// </summary>
        private static float ModuleScale(float alpha, int rank, bool rslora, float? scaleOverride)
        {
            float scale = alpha <= 0f ? 1f : rslora ? alpha / MathF.Sqrt(rank) : alpha / rank;
            return scaleOverride is { } userScale ? scale * userScale : scale;
        }

        private static string DescribeRanks(Dictionary<string, LoraDelta> deltas)
        {
            int min = deltas.Values.Min(d => d.Rank), max = deltas.Values.Max(d => d.Rank);
            return min == max
                ? max.ToString(CultureInfo.InvariantCulture)
                : $"{min.ToString(CultureInfo.InvariantCulture)}-{max.ToString(CultureInfo.InvariantCulture)}";
        }

        /// <summary>
        /// LoRA factors are real-valued matrices. An integer, bool or FP8 tensor under a
        /// <c>lora_a</c>/<c>lora_b</c> name is not one, and would otherwise be converted
        /// silently (integers) or fail half-way through conversion (FP8).
        /// </summary>
        private static void RequireFloatFactor(SafetensorTensorInfo info)
        {
            if (info.Dtype is not (SafetensorDtype.F32 or SafetensorDtype.F16 or SafetensorDtype.BF16 or SafetensorDtype.F64))
                throw new InvalidDataException(
                    $"LoRA tensor '{info.Name}' is {info.Dtype}; LoRA factors must be F32, F16, BF16 or F64.");
        }

        /// <summary>
        /// The safetensors reader reports a malformed header with whatever the JSON reader
        /// threw: a parse error, a missing member, a value of the wrong kind. To a caller that
        /// is one failure, a file that is not a valid safetensors adapter, so it is reported as
        /// such, without quoting the file's text.
        /// </summary>
        private static SafetensorsFile OpenSafetensors(string path)
        {
            try
            {
                return new SafetensorsFile(path);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"'{Path.GetFileName(path)}' is not a valid safetensors file: its header is not valid JSON.", ex);
            }
            catch (Exception ex) when (ex is KeyNotFoundException or InvalidOperationException or FormatException)
            {
                throw new InvalidDataException(
                    $"'{Path.GetFileName(path)}' is not a valid safetensors file: its header does not have the expected fields and types.", ex);
            }
        }

        private static JsonDocument ParseConfig(string configPath)
        {
            try
            {
                return JsonDocument.Parse(File.ReadAllText(configPath));
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"LoRA config '{Path.GetFileName(configPath)}' is not valid JSON.", ex);
            }
        }

        /// <summary>
        /// Read PEFT's <c>r</c>, <c>lora_alpha</c> and <c>use_rslora</c>, and refuse the
        /// settings whose effect this loader cannot reproduce exactly: DoRA, transposed
        /// factors (<c>fan_in_fan_out</c>), per-module alpha or rank patterns keyed by PEFT
        /// module names (applying them would need those names mapped onto GGUF tensors), fully
        /// trained modules (<c>modules_to_save</c>), trained token embeddings
        /// (<c>trainable_token_indices</c>), trained biases (<c>bias</c> other than
        /// <c>"none"</c>, <c>lora_bias</c>), <c>layer_replication</c>, activated LoRA
        /// (<c>alora_invocation_tokens</c>, <c>invocation_string</c>), the LoRA variants
        /// (<c>kasa_config</c>, <c>arrow_config</c>, <c>use_bdlora</c>), an
        /// <c>init_lora_weights</c> that rewrites the base weights (the adapter applies on top of
        /// the rewritten ones), a boolean flag that is not <c>true</c> or <c>false</c>, and a
        /// <c>lora_alpha</c> that is not a finite number >= 0.
        /// </summary>
        private static void ReadConfig(string configPath, ref int rank, ref float alpha, ref bool rslora)
        {
            if (!File.Exists(configPath))
                throw new FileNotFoundException($"LoRA config file not found: {configPath}", configPath);

            using var doc = ParseConfig(configPath);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"LoRA config '{configPath}' must be a JSON object.");

            string file = Path.GetFileName(configPath);
            bool dora = ReadFlag(root, "use_dora", file);
            bool fanInFanOut = ReadFlag(root, "fan_in_fan_out", file);
            bool loraBias = ReadFlag(root, "lora_bias", file);
            rslora = ReadFlag(root, "use_rslora", file);

            var unsupported = new List<string>();
            if (dora) unsupported.Add("use_dora (DoRA)");
            if (fanInFanOut) unsupported.Add("fan_in_fan_out");
            if (IsSet(root, "alpha_pattern")) unsupported.Add("alpha_pattern");
            if (IsSet(root, "rank_pattern")) unsupported.Add("rank_pattern");
            if (IsSet(root, "modules_to_save")) unsupported.Add("modules_to_save");
            // Trained embedding rows (a list of token ids, or a dict of them per embedding
            // module), which no factor carries, as with modules_to_save.
            if (IsSet(root, "trainable_token_indices")) unsupported.Add("trainable_token_indices (trained token embeddings)");
            // "all" and "lora_only" train the base model's biases, which no factor carries.
            if (root.TryGetProperty("bias", out var biasEl) && biasEl.ValueKind != JsonValueKind.Null
                && !(biasEl.ValueKind == JsonValueKind.String && biasEl.GetString() == "none"))
                unsupported.Add("bias other than \"none\" (trained biases)");
            if (loraBias) unsupported.Add("lora_bias");
            if (IsSet(root, "layer_replication")) unsupported.Add("layer_replication");
            // Any value but null (an empty list too) makes an activated LoRA, which applies only
            // after its invocation tokens; text LoRA applies an adapter to every token.
            if (root.TryGetProperty("alora_invocation_tokens", out var aloraEl) && aloraEl.ValueKind != JsonValueKind.Null)
                unsupported.Add("alora_invocation_tokens (activated LoRA)");
            // The older aLoRA form (IBM's granite aLoRA adapters): llama.cpp's converter reads a
            // non-empty invocation_string as activated LoRA and writes adapter.alora.*.
            if (IsSet(root, "invocation_string")) unsupported.Add("invocation_string (activated LoRA)");
            // PEFT's other LoRA variants. KaSA applies lora_B(lora_A(x) * lora_diag), Arrow
            // routes each token to its top-k LoRA experts and BD-LoRA stores a block-diagonal
            // factor ([out, in / nblocks]). VeLoRA and MonteCLoRA are accepted: PEFT's forward
            // when not training is lora_B(lora_A(x)) * scaling, and both merge as plain LoRA.
            if (IsVariant(root, "kasa_config")) unsupported.Add("kasa_config (KaSA)");
            if (IsVariant(root, "arrow_config")) unsupported.Add("arrow_config (Arrow)");
            if (IsVariant(root, "use_bdlora")) unsupported.Add("use_bdlora (BD-LoRA)");
            if (unsupported.Count > 0)
                throw new NotSupportedException(
                    $"LoRA config '{file}' uses {string.Join(", ", unsupported)}, which text LoRA " +
                    "cannot apply exactly, so the adapter is refused rather than applied without it.");

            // PiSSA, OLoRA, CorDA, Astra, LoftQ and LoRA-GA set the base weight to W - s * B0 * A0
            // (LoftQ to a quantized residual) when the adapter is created, so a trained adapter
            // applies on top of that, not on top of W. PEFT's save_pretrained writes it as a plain
            // LoRA adapter (init_lora_weights true) only when given
            // path_initial_model_for_weight_conversion. The values are tested as PEFT's
            // LoraLayer.update_layer tests them; every other init leaves W alone.
            if (root.TryGetProperty("init_lora_weights", out var initEl) && initEl.ValueKind == JsonValueKind.String
                && initEl.GetString() is string init && RewritesBaseWeights(init))
                throw new NotSupportedException(
                    $"LoRA config '{file}' has init_lora_weights \"{init}\": the adapter was saved without weight " +
                    "conversion, so it applies on top of base weights this initialization rewrote, not the original " +
                    "ones, and is refused. " +
                    (init == "loftq"
                        ? "PEFT's weight conversion (save_pretrained(..., path_initial_model_for_weight_conversion=...)) " +
                          "does not undo LoftQ's quantized residual, so the adapter cannot be applied to the original model."
                        : "Convert it to a plain LoRA adapter before use: PEFT " +
                          "save_pretrained(..., path_initial_model_for_weight_conversion=...)."));

            if (root.TryGetProperty("r", out var rEl) && rEl.ValueKind == JsonValueKind.Number && rEl.TryGetInt32(out int r))
                rank = r;
            if (root.TryGetProperty("lora_alpha", out var aEl))
            {
                // As adapter.lora.alpha in a GGUF adapter: anything but a number is refused, never
                // read as no alpha. JSON has no NaN or infinity, but a number past float's range
                // parses to infinity; RequireAlpha refuses that, and a negative alpha.
                if (aEl.ValueKind != JsonValueKind.Number)
                    throw new InvalidDataException($"LoRA config '{file}' has a lora_alpha that is not a number.");
                alpha = RequireAlpha(float.Parse(aEl.GetRawText(), NumberStyles.Float, CultureInfo.InvariantCulture),
                    $"LoRA config '{file}' has lora_alpha");
            }

            // A boolean PEFT flag: true, or false (absent or null). PEFT tests it by Python
            // truth, so 1 or "true" turns DoRA or rsLoRA on there; any value that is not a JSON
            // boolean is refused rather than read as off.
            static bool ReadFlag(JsonElement root, string key, string file)
            {
                if (!root.TryGetProperty(key, out var el))
                    return false;
                return el.ValueKind switch
                {
                    JsonValueKind.True => true,
                    JsonValueKind.False or JsonValueKind.Null => false,
                    _ => throw new InvalidDataException($"LoRA config '{file}' has a {key} that is not true or false."),
                };
            }

            // A setting is off when absent, null, false or empty (string, list or object); any
            // other value is set, a malformed one (a number, say) included.
            static bool IsSet(JsonElement root, string key)
            {
                if (!root.TryGetProperty(key, out var el))
                    return false;
                return el.ValueKind switch
                {
                    JsonValueKind.Null or JsonValueKind.False => false,
                    JsonValueKind.Object => el.EnumerateObject().Any(),
                    JsonValueKind.Array => el.GetArrayLength() > 0,
                    JsonValueKind.String => !string.IsNullOrEmpty(el.GetString()),
                    _ => true,
                };
            }

            // A LoRA variant is off only when absent, null or false: an empty object is the
            // variant with its defaults, never read as no variant.
            static bool IsVariant(JsonElement root, string key) =>
                root.TryGetProperty(key, out var el) && el.ValueKind is not (JsonValueKind.Null or JsonValueKind.False);

            // "pissa" and "pissa_niter_<N>" (any other "pissa..." PEFT rejects), "corda", "astra",
            // "olora" in any case, "loftq" and "lora_ga".
            static bool RewritesBaseWeights(string init) =>
                init.StartsWith("pissa", StringComparison.Ordinal) || init.StartsWith("corda", StringComparison.Ordinal)
                || init.StartsWith("astra", StringComparison.Ordinal) || init.Equals("olora", StringComparison.OrdinalIgnoreCase)
                || init == "loftq" || init == "lora_ga";
        }

        private static string? ResolveConfigPath(string weightsPath, string? configPath)
        {
            if (configPath != null)
                return configPath;

            string dir = Path.GetDirectoryName(Path.GetFullPath(weightsPath)) ?? string.Empty;
            string sibling = Path.Combine(dir, "adapter_config.json");
            return File.Exists(sibling) ? sibling : null;
        }

        /// <summary>A <c>[rows, cols]</c> factor as row-major floats.</summary>
        private static float[] ReadFactor(SafetensorsFile st, SafetensorTensorInfo info, long rows, long cols)
        {
            if (info.NumElements != rows * cols)
                throw new InvalidDataException(
                    $"LoRA tensor '{info.Name}' has {info.NumElements} elements, expected {rows}x{cols}.");
            return st.ReadFloat32(info.Name);
        }
    }
}
