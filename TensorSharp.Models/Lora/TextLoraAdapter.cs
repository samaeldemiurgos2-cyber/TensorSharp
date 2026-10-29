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
using System.Threading;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// One low-rank update term for a single base-model projection, in a form that
    /// makes the forward path two small GEMMs with no per-call transpose.
    ///
    /// <para>
    /// A PEFT adapter stores <c>lora_A</c> as <c>[rank, in]</c> and <c>lora_B</c> as
    /// <c>[out, rank]</c>. The delta for input <c>x</c> (<c>[seq, in]</c>) is
    /// <c>(alpha / rank) * x @ A^T @ B^T</c>. This type stores <see cref="AT"/>
    /// (<c>[in, rank]</c>) and <see cref="BT"/> (<c>[rank, out]</c>) with the effective
    /// scale already folded into <c>BT</c>, so the forward computes
    /// <c>x @ AT @ BT</c> and adds it to the base projection.
    /// </para>
    /// </summary>
    public sealed class LoraDelta
    {
        /// <summary>Input feature count of the base projection.</summary>
        public long In { get; init; }

        /// <summary>Output feature count of the base projection.</summary>
        public long Out { get; init; }

        /// <summary>Low rank.</summary>
        public int Rank { get; init; }

        /// <summary><c>[In, Rank]</c> row-major — the transpose of <c>lora_A</c>.</summary>
        public float[] AT { get; init; } = Array.Empty<float>();

        /// <summary><c>[Rank, Out]</c> row-major — the transpose of <c>lora_B</c>, scale folded in.</summary>
        public float[] BT { get; init; } = Array.Empty<float>();
    }

    /// <summary>
    /// A loaded LoRA adapter: a name, the low rank and alpha, and one
    /// <see cref="LoraDelta"/> per target projection keyed by the base GGUF's tensor name
    /// (e.g. <c>"blk.0.attn_q.weight"</c>). Immutable after load, so it is safe to
    /// share between the registry, the model and concurrent inference readers.
    ///
    /// <para>
    /// A request carries the instance it resolved (<see cref="SequenceState.Adapter"/>),
    /// and the engine compares adapters by reference: reloading a name loads a new
    /// instance with a new <see cref="Generation"/> and retires the old one, which keeps
    /// serving the requests that already hold it.
    /// </para>
    /// </summary>
    public sealed class TextLoraAdapter : IRequestAdapter
    {
        private static long s_generation;
        private volatile bool _retired;

        public string Name { get; }

        /// <summary>Process-wide load counter: tells two loads of one name apart in logs.</summary>
        public long Generation { get; }

        /// <inheritdoc/>
        public string Id => $"{Name}#{Generation}";

        /// <inheritdoc/>
        public bool IsRetired => _retired;

        /// <summary>Low rank from the PEFT config, or the largest rank of the <c>lora_A</c>
        /// tensors (modules may differ; each <see cref="LoraDelta.Rank"/> is its own).</summary>
        public int Rank { get; }

        /// <summary>The PEFT <c>lora_alpha</c>, or <see cref="Rank"/> when unspecified or 0;
        /// always a finite number >= 0.</summary>
        public float Alpha { get; }

        /// <summary>Per-target deltas keyed by base-model weight name.</summary>
        public IReadOnlyDictionary<string, LoraDelta> Deltas { get; }

        /// <summary>One human-readable line summarising the source files.</summary>
        public string Summary { get; }

        /// <summary>
        /// The <c>general.architecture</c> a GGUF adapter was converted for, or null when the
        /// file does not say (safetensors). A model of another architecture refuses it.
        /// </summary>
        public string? Architecture { get; }

        public TextLoraAdapter(
            string name,
            int rank,
            float alpha,
            IReadOnlyDictionary<string, LoraDelta> deltas,
            string summary,
            string? architecture = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("A LoRA adapter name is required.", nameof(name));
            // Listed as a JSON number (GET /v1/loras, GET /v1/models), which has no NaN or
            // infinity. The loader refuses such an alpha with the reason; this keeps any other
            // source from registering one that would fail every listing.
            if (!float.IsFinite(alpha) || alpha < 0f)
                throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "A LoRA alpha must be a finite number >= 0.");
            Name = name;
            Rank = rank;
            Alpha = alpha;
            Deltas = deltas ?? throw new ArgumentNullException(nameof(deltas));
            Summary = summary ?? string.Empty;
            Architecture = string.IsNullOrWhiteSpace(architecture) ? null : architecture;
            Generation = Interlocked.Increment(ref s_generation);
        }

        /// <summary>Mark the adapter unloaded or replaced (see <see cref="IRequestAdapter.IsRetired"/>).</summary>
        internal void Retire() => _retired = true;
    }

    /// <summary>Public metadata for <c>GET /v1/models</c>, sized to stay cheap to list.</summary>
    public sealed record LoraAdapterInfo(string Name, int Rank, float Alpha, int Modules);
}