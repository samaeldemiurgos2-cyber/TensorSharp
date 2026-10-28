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
using System.Linq;

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// One loaded weight an adapter updates, with the factors the forward applies
    /// (<c>x @ AT @ BT</c>, scale folded into <see cref="BT"/>). For a weight fused at load
    /// (<c>attn_q/k/v -> attn_qkv</c>, <c>ffn_gate/up -> ffn_gate_up</c>) the adapter's
    /// per-projection deltas are composed into one exact delta for the fused weight: the
    /// ranks are stacked in <see cref="AT"/> and each <c>B</c> lands in its projection's rows
    /// of the fused output, so the fused projection plus this delta equals the separate
    /// projections plus theirs.
    /// </summary>
    public sealed class LoraBoundTarget
    {
        public required string WeightName { get; init; }
        public required long In { get; init; }
        public required long Out { get; init; }
        public required int Rank { get; init; }

        /// <summary><c>[In, Rank]</c> row-major.</summary>
        public required float[] AT { get; init; }

        /// <summary><c>[Rank, Out]</c> row-major.</summary>
        public required float[] BT { get; init; }

        /// <summary>The adapter module names composed into this target.</summary>
        public required IReadOnlyList<string> Sources { get; init; }
    }

    /// <summary>The result of <see cref="TextLoraBinder.Bind"/>.</summary>
    public sealed class LoraBinding
    {
        public required IReadOnlyList<LoraBoundTarget> Targets { get; init; }

        /// <summary>One line per adapter module that cannot be applied to the loaded model.
        /// An adapter with any is refused: applying the rest would answer with a model that
        /// is neither the base nor the adapter.</summary>
        public required IReadOnlyList<string> Errors { get; init; }
    }

    /// <summary>
    /// Maps an adapter's modules (named by the model file's weights) onto the weights a
    /// model holds after load. Model-free, so the composition is testable without weights.
    /// </summary>
    public static class TextLoraBinder
    {
        /// <summary>Fused weight suffix, then the file's projections it concatenates along the
        /// output dimension, in order. These are the fusions <c>ModelBase</c> and the model
        /// families perform at load.</summary>
        private static readonly (string Fused, string[] Parts)[] FusedGroups =
        {
            ("attn_qkv.weight", new[] { "attn_q.weight", "attn_k.weight", "attn_v.weight" }),
            ("ffn_gate_up.weight", new[] { "ffn_gate.weight", "ffn_up.weight" }),
        };

        /// <param name="adapter">The adapter to bind.</param>
        /// <param name="loadedDims">A weight's <c>(in, out)</c> as the model holds it now, or null
        /// when the model holds no such weight.</param>
        /// <param name="fileDims">A weight's <c>(in, out)</c> as the model file stores it (before
        /// any fusion), or null when the file has no such tensor.</param>
        public static LoraBinding Bind(
            TextLoraAdapter adapter,
            Func<string, (long In, long Out)?> loadedDims,
            Func<string, (long In, long Out)?> fileDims)
        {
            ArgumentNullException.ThrowIfNull(adapter);
            ArgumentNullException.ThrowIfNull(loadedDims);
            ArgumentNullException.ThrowIfNull(fileDims);

            var errors = new List<string>();
            var components = new Dictionary<string, List<(string Source, LoraDelta Delta, long Offset)>>(StringComparer.Ordinal);
            var fusedDims = new Dictionary<string, (long In, long Out)>(StringComparer.Ordinal);

            foreach (var (module, delta) in adapter.Deltas.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                if (loadedDims(module) is { } dims)
                {
                    if (delta.In != dims.In || delta.Out != dims.Out)
                    {
                        errors.Add($"{module}: the delta is [{delta.Out}, {delta.In}] but the weight is [{dims.Out}, {dims.In}]");
                        continue;
                    }
                    Add(module, module, delta, 0, dims);
                    continue;
                }

                if (!TryLocateFusedSegment(module, loadedDims, fileDims, out string? fusedName,
                        out (long In, long Out) fused, out long offset, out long segmentOut, out string? why))
                {
                    errors.Add(why ?? $"{module}: not a weight of the loaded model");
                    continue;
                }
                if (delta.In != fused.In || delta.Out != segmentOut)
                {
                    errors.Add($"{module}: the delta is [{delta.Out}, {delta.In}] but the projection is " +
                        $"[{segmentOut}, {fused.In}] (rows {offset}..{offset + segmentOut} of {fusedName})");
                    continue;
                }
                Add(fusedName!, module, delta, offset, fused);
            }

            var targets = new List<LoraBoundTarget>(components.Count);
            foreach (var (weightName, parts) in components.OrderBy(kv => kv.Key, StringComparer.Ordinal))
                targets.Add(Compose(weightName, fusedDims[weightName], parts));
            return new LoraBinding { Targets = targets, Errors = errors };

            void Add(string weightName, string source, LoraDelta delta, long offset, (long In, long Out) dims)
            {
                if (!components.TryGetValue(weightName, out var list))
                    components[weightName] = list = new List<(string, LoraDelta, long)>();
                list.Add((source, delta, offset));
                fusedDims[weightName] = dims;
            }
        }

        private static bool TryLocateFusedSegment(
            string module,
            Func<string, (long In, long Out)?> loadedDims,
            Func<string, (long In, long Out)?> fileDims,
            out string? fusedName,
            out (long In, long Out) fused,
            out long offset,
            out long segmentOut,
            out string? why)
        {
            fusedName = null; fused = default; offset = 0; segmentOut = 0; why = null;
            foreach (var (fusedSuffix, parts) in FusedGroups)
            {
                int index = Array.FindIndex(parts, p => module.EndsWith("." + p, StringComparison.Ordinal));
                if (index < 0) continue;

                string prefix = module[..^parts[index].Length];
                string candidate = prefix + fusedSuffix;
                if (loadedDims(candidate) is not { } dims)
                    return false;

                long total = 0;
                for (int i = 0; i < parts.Length; i++)
                {
                    if (fileDims(prefix + parts[i]) is not { } partDims)
                    {
                        why = $"{module}: {candidate} was fused at load, but the model file has no {prefix + parts[i]} " +
                            "to locate this projection's rows";
                        return false;
                    }
                    if (i == index) { offset = total; segmentOut = partDims.Out; }
                    total += partDims.Out;
                }
                if (total != dims.Out)
                {
                    why = $"{module}: {candidate} has {dims.Out} output rows, but its projections in the file sum to {total}";
                    return false;
                }
                fusedName = candidate;
                fused = dims;
                return true;
            }
            return false;
        }

        private static LoraBoundTarget Compose(
            string weightName, (long In, long Out) dims, List<(string Source, LoraDelta Delta, long Offset)> parts)
        {
            if (parts.Count == 1 && parts[0].Offset == 0 && parts[0].Delta.Out == dims.Out)
            {
                var only = parts[0].Delta;
                return new LoraBoundTarget
                {
                    WeightName = weightName, In = dims.In, Out = dims.Out, Rank = only.Rank,
                    AT = only.AT, BT = only.BT, Sources = new[] { parts[0].Source },
                };
            }

            int rank = parts.Sum(p => p.Delta.Rank);
            long inDim = dims.In, outDim = dims.Out;
            var at = new float[checked(inDim * rank)];
            var bt = new float[checked(rank * outDim)];
            int rankOffset = 0;
            foreach (var (_, delta, offset) in parts)
            {
                int r = delta.Rank;
                // AT: this delta's columns rankOffset..rankOffset+r of every input row.
                for (long x = 0; x < inDim; x++)
                    Array.Copy(delta.AT, x * r, at, x * rank + rankOffset, r);
                // BT: this delta's rows land in its projection's columns of the fused output.
                for (int k = 0; k < r; k++)
                    Array.Copy(delta.BT, (long)k * delta.Out, bt, (rankOffset + k) * outDim + offset, delta.Out);
                rankOffset += r;
            }
            return new LoraBoundTarget
            {
                WeightName = weightName, In = inDim, Out = outDim, Rank = rank,
                AT = at, BT = bt, Sources = parts.Select(p => p.Source).ToArray(),
            };
        }
    }
}
