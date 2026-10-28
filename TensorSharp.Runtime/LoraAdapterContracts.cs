// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// A loaded text-LoRA adapter as the scheduler sees it: an opaque, immutable identity
    /// carried on one request (<see cref="SamplingConfig.LoraAdapter"/>). Two requests use the
    /// same adapter only when they carry the same instance - reloading an adapter under the
    /// same name produces a new one, so a request that started on the old weights never
    /// shares a forward pass with one on the new.
    /// </summary>
    public interface ILoraAdapterHandle
    {
        /// <summary>The name the adapter was registered under.</summary>
        string Name { get; }
    }

    /// <summary>
    /// A model that applies a text-LoRA adapter on top of its base weights. The engine owns
    /// the selection: it activates the adapter of the sequences a step runs immediately before
    /// that step's forward, on the engine thread and under the model's GPU lock, and clears
    /// it afterwards, so no forward ever runs under another request's adapter.
    /// </summary>
    public interface ILoraAdapterModel
    {
        /// <summary>Select <paramref name="adapter"/> for the forwards that follow, or null
        /// to run the base weights.</summary>
        void ActivateLoraAdapter(ILoraAdapterHandle? adapter);

        /// <summary>The loaded weight names <paramref name="adapter"/> applies to on this
        /// model. A request is refused before it emits a token unless every one of them was
        /// applied while its prompt was processed.</summary>
        IReadOnlyCollection<string> GetLoraTargets(ILoraAdapterHandle adapter);

        /// <summary>The weight names a LoRA delta was actually applied to since the last
        /// call, which this call resets.</summary>
        IReadOnlyCollection<string> TakeAppliedLoraTargets();

        /// <summary>Record that this model's forward does not apply <paramref name="missing"/>
        /// for <paramref name="adapter"/>, so later requests naming it are refused up front.</summary>
        void MarkLoraUnsupported(ILoraAdapterHandle adapter, IReadOnlyCollection<string> missing);
    }
}
