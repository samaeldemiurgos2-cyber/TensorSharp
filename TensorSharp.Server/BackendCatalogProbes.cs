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
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Server;

/// <summary>
/// The Server's half of <see cref="BackendCatalog"/>: the availability probes
/// that touch the CUDA, MLX and GGML backends. They live here rather than next
/// to the catalog because TensorSharp.Chat must not reference
/// TensorSharp.Backends.Cuda or TensorSharp.Backends.MLX (a static partial class
/// cannot span two assemblies, hence a sibling class rather than a partial).
/// </summary>
public static class BackendCatalogProbes
{
    /// <summary>The backends this machine can actually run, in UI order.</summary>
    internal static IReadOnlyList<BackendOption> GetSupportedBackends()
    {
        return BackendCatalog.GetSupportedBackends(
            IsGgmlBackendAvailable,
            CudaBackend.IsAvailable,
            MlxBackend.IsAvailable);
    }

    // The reason a GGML backend probe threw, per backend, so the startup banner
    // can say WHY a backend is missing instead of just omitting it from the list.
    // First failure wins; the probe may run more than once.
    private static readonly Dictionary<GgmlBackendType, string> ProbeFailures = new();

    private static bool IsGgmlBackendAvailable(GgmlBackendType backendType)
    {
        try
        {
            // Backend discovery runs at web-app startup, so it must not spin up
            // any GGML device — otherwise picking a non-GGML backend (MLX,
            // direct CUDA) would still trigger `ggml_metal_device_init` / etc.
            // logs at startup. CanInitializeBackend is a lightweight compile-flag
            // + platform check; the real GGML init is deferred until a GGML
            // backend is actually selected.
            return GgmlBasicOps.CanInitializeBackend(backendType);
        }
        catch (Exception ex)
        {
            lock (ProbeFailures)
            {
                ProbeFailures.TryAdd(backendType, ex.Message);
            }
            return false;
        }
    }

    /// <summary>
    /// The probe exceptions swallowed above, one <c>"ggml_metal: reason"</c> line
    /// per backend, so the startup banner's backend list carries a cause. The direct
    /// CUDA probe adds a <c>"cuda: reason"</c> line when the driver loaded but is older
    /// than CUDA 13, or every GPU is below compute capability 7.5.
    /// </summary>
    public static IReadOnlyList<string> DescribeProbeFailures()
    {
        lock (ProbeFailures)
        {
            var failures = ProbeFailures
                .Select(kv => "ggml_" + kv.Key.ToString().ToLowerInvariant() + ": " + kv.Value)
                .ToList();
            if (CudaDevice.UnavailableReason is string cudaReason)
                failures.Add("cuda: " + cudaReason);
            return failures;
        }
    }
}
