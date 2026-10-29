// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// A change to the model's weights that one request asks for (a text LoRA adapter),
    /// carried on its <see cref="SequenceState.Adapter"/>. The engine never lets two
    /// adapters share a forward: the scheduler admits sequences in "adapter epochs" (the
    /// running set only ever holds sequences of one adapter, or of none), and the executor
    /// selects the epoch's adapter on the model for the length of one step
    /// (<see cref="IAdapterAwareModel.SelectStepAdapter"/>). Identity is by reference: a
    /// reloaded adapter is a new instance and a new epoch, even under the same name.
    /// </summary>
    public interface IRequestAdapter
    {
        /// <summary>Name plus load generation (for logs), e.g. <c>"style#3"</c>.</summary>
        string Id { get; }

        /// <summary>
        /// True once the adapter was unloaded or replaced. Requests that already hold it
        /// keep running with it; the model releases its copy of the factors once none of
        /// them is left (<see cref="InferenceEngine.ReleaseRetiredAdapters"/>).
        /// </summary>
        bool IsRetired { get; }
    }

    /// <summary>
    /// A model that can apply an <see cref="IRequestAdapter"/> to a forward. Every member
    /// that touches the model's tensors runs under the model's GPU compute lock: the
    /// engine worker binds, selects and releases inside its step, and a host that binds
    /// eagerly (at adapter load) takes the lock itself.
    /// </summary>
    public interface IAdapterAwareModel
    {
        /// <summary>
        /// Validate <paramref name="adapter"/> against this model and materialize what the
        /// forward needs, once per (model, adapter); idempotent. False, with the reason,
        /// when any part of the adapter cannot be applied: an adapter is applied whole or
        /// refused, never partially.
        /// </summary>
        bool TryBindAdapter(IRequestAdapter adapter, out string? error);

        /// <summary>
        /// Select the bound <paramref name="adapter"/> for the forwards of the current step,
        /// or null for the base model. The executor clears it before the step releases the
        /// GPU lock, so nothing outside an engine step ever sees an adapter selected.
        /// </summary>
        void SelectStepAdapter(IRequestAdapter? adapter);

        /// <summary>The adapter selected for the current step, or null.</summary>
        IRequestAdapter? StepAdapter { get; }

        /// <summary>A snapshot of the adapters this model holds state for (bound or merely
        /// validated).</summary>
        IReadOnlyCollection<IRequestAdapter> BoundAdapters { get; }

        /// <summary>Drop this model's state for <paramref name="adapter"/>. Throws while it is
        /// selected for a step.</summary>
        void ReleaseAdapter(IRequestAdapter adapter);
    }
}
