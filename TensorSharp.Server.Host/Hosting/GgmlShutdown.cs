// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.GGML;

namespace TensorSharp.Server.Host.Hosting;

internal static class GgmlShutdown
{
    internal static void ShutdownIfLoaded()
    {
        // Shutdown is itself a P/Invoke. Cleanup must not be the first native
        // call in a managed CPU process, which may have no GgmlOps library.
        // Read the non-loading flag at teardown time, including ProcessExit.
        if (GgmlBasicOps.IsNativeLibraryLoaded)
            GgmlBasicOps.Shutdown();
    }
}
