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
using System.Text.RegularExpressions;

namespace TensorSharp.Models.Lora
{
    /// <summary>Which factor of a LoRA pair a tensor holds.</summary>
    public enum LoraFactor { A, B }

    /// <summary>
    /// Maps the tensor names of a text-LoRA <c>.safetensors</c> file to the model-file (GGUF)
    /// weight each factor updates. Three spellings are accepted:
    /// <list type="bullet">
    /// <item><c>blk.0.attn_q.weight.lora_a</c> - the GGUF weight name plus a factor suffix;</item>
    /// <item><c>lora.blk.0.attn_q.weight.lora_a</c> - the same with a <c>lora.</c> prefix;</item>
    /// <item><c>base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight</c> - a standard
    /// Hugging Face PEFT file, whose module path is translated to the GGUF name the way
    /// llama.cpp's converter names it.</item>
    /// </list>
    /// </summary>
    public static partial class TextLoraNames
    {
        // HF module path inside a decoder layer -> the GGUF weight suffix llama.cpp writes.
        private static readonly Dictionary<string, string> LayerModules = new(StringComparer.Ordinal)
        {
            ["self_attn.q_proj"] = "attn_q",
            ["self_attn.k_proj"] = "attn_k",
            ["self_attn.v_proj"] = "attn_v",
            ["self_attn.o_proj"] = "attn_output",
            ["mlp.gate_proj"] = "ffn_gate",
            ["mlp.up_proj"] = "ffn_up",
            ["mlp.down_proj"] = "ffn_down",
            // Nemotron-H: every block is a "mixer" (attention, Mamba2 or MLP).
            ["mixer.q_proj"] = "attn_q",
            ["mixer.k_proj"] = "attn_k",
            ["mixer.v_proj"] = "attn_v",
            ["mixer.o_proj"] = "attn_output",
            ["mixer.up_proj"] = "ffn_up",
            ["mixer.down_proj"] = "ffn_down",
            ["mixer.in_proj"] = "ssm_in",
            ["mixer.out_proj"] = "ssm_out",
        };

        [GeneratedRegex(@"(?:^|\.)layers\.(\d+)\.(.+)$")]
        private static partial Regex LayerPath();

        /// <summary>
        /// The GGUF weight and factor <paramref name="tensorName"/> holds, or false when it is not
        /// a LoRA factor at all (metadata, a DoRA magnitude, a saved embedding). A factor whose
        /// module has no known GGUF name is reported in <paramref name="unmapped"/> so the load
        /// can refuse it instead of silently dropping part of the adapter.
        /// </summary>
        public static bool TryParse(string tensorName, out string target, out LoraFactor factor, out string? unmapped)
        {
            target = string.Empty;
            factor = LoraFactor.A;
            unmapped = null;

            // TensorSharp spelling: the target is the model-file weight name, verbatim (the
            // binder checks it against the loaded model).
            if (tensorName.EndsWith(".lora_a", StringComparison.Ordinal) || tensorName.EndsWith(".lora_b", StringComparison.Ordinal))
            {
                factor = tensorName[^1] == 'a' ? LoraFactor.A : LoraFactor.B;
                target = tensorName[..^".lora_a".Length];
                if (target.StartsWith("lora.", StringComparison.Ordinal))
                    target = target["lora.".Length..];
                return true;
            }

            string module;
            if (tensorName.EndsWith(".lora_A.weight", StringComparison.Ordinal)) { module = tensorName[..^".lora_A.weight".Length]; factor = LoraFactor.A; }
            else if (tensorName.EndsWith(".lora_B.weight", StringComparison.Ordinal)) { module = tensorName[..^".lora_B.weight".Length]; factor = LoraFactor.B; }
            else return false;

            // A PEFT module path.
            if (module.EndsWith("lm_head", StringComparison.Ordinal))
            {
                target = "output.weight";
                return true;
            }
            var m = LayerPath().Match(module);
            if (m.Success && LayerModules.TryGetValue(m.Groups[2].Value, out string? suffix))
            {
                target = $"blk.{m.Groups[1].Value}.{suffix}.weight";
                return true;
            }
            unmapped = module;
            return true;
        }
    }
}
