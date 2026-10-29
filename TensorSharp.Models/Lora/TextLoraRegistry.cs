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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// Owns loaded <see cref="TextLoraAdapter"/> instances independently of any model, so
    /// adapters can be loaded and unloaded at runtime without reloading the base model.
    ///
    /// <para>
    /// Adapters are immutable after load, so the read path is lock-free: <see cref="Get"/>
    /// and <see cref="List"/> only observe the <see cref="ConcurrentDictionary{TKey, TValue}"/>.
    /// <see cref="Load"/> / <see cref="Register"/> replace an existing entry under the same
    /// name (reload semantics) and <see cref="Unload"/> removes one; either way the instance
    /// that leaves the registry is retired (<see cref="TextLoraAdapter.IsRetired"/>). A
    /// request that already resolved it keeps it to the end, and the model releases its
    /// copy of the factors only after the last such request (see
    /// <c>InferenceEngine.ReleaseRetiredAdapters</c>).
    /// </para>
    /// </summary>
    public sealed class TextLoraRegistry : IDisposable
    {
        private readonly ConcurrentDictionary<string, TextLoraAdapter> _adapters = new(StringComparer.Ordinal);

        /// <summary>
        /// Load (or reload) an adapter under <paramref name="name"/> and register it.
        /// </summary>
        /// <exception cref="InvalidDataException">The file or config is malformed.</exception>
        /// <exception cref="FileNotFoundException">A named file is missing.</exception>
        public TextLoraAdapter Load(
            string name, string safetensorsPath, string? configPath = null, float? scaleOverride = null,
            bool discoverSiblingConfig = true)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("A LoRA adapter name is required.", nameof(name));

            return Register(TextLoraLoader.Load(name, safetensorsPath, configPath, scaleOverride, discoverSiblingConfig));
        }

        /// <summary>
        /// Register an already loaded adapter under its name, retiring the instance it
        /// replaces (if any). Returns <paramref name="adapter"/>.
        /// </summary>
        public TextLoraAdapter Register(TextLoraAdapter adapter)
        {
            ArgumentNullException.ThrowIfNull(adapter);
            while (true)
            {
                if (_adapters.TryGetValue(adapter.Name, out var previous))
                {
                    if (!_adapters.TryUpdate(adapter.Name, adapter, previous))
                        continue;
                    if (!ReferenceEquals(previous, adapter))
                        previous.Retire();
                    return adapter;
                }
                if (_adapters.TryAdd(adapter.Name, adapter))
                    return adapter;
            }
        }

        /// <summary>Unload, retire and stop advertising an adapter by name. Returns true when
        /// it was loaded.</summary>
        public bool Unload(string name)
        {
            if (!_adapters.TryRemove(name, out var removed))
                return false;
            removed.Retire();
            return true;
        }

        /// <summary>The adapter registered under <paramref name="name"/>, or null.</summary>
        public TextLoraAdapter? Get(string name) => _adapters.TryGetValue(name, out var adapter) ? adapter : null;

        /// <summary>True when an adapter with this name is loaded.</summary>
        public bool Contains(string name) => _adapters.ContainsKey(name);

        /// <summary>How many adapters are registered.</summary>
        public int Count => _adapters.Count;

        /// <summary>Snapshot of loaded adapters, ordered by name for a stable listing.</summary>
        public IReadOnlyList<LoraAdapterInfo> List() =>
            _adapters.Values
                .OrderBy(a => a.Name, StringComparer.Ordinal)
                .Select(a => new LoraAdapterInfo(a.Name, a.Rank, a.Alpha, a.Deltas.Count))
                .ToList();

        public void Dispose()
        {
            // The registry holds no unmanaged resources: the adapters own only managed
            // float[] buffers, so clearing the dictionary releases them once no request
            // or model reference remains.
            foreach (var adapter in _adapters.Values)
                adapter.Retire();
            _adapters.Clear();
        }
    }
}