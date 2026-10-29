// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Server.Hosting;

/// <summary>
/// The server's text-model LoRA adapter directory: <c>--loras-dir &lt;dir&gt;</c>, or
/// <see cref="EnvironmentVariable"/> when the flag is absent (the flag wins).
///
/// <para>
/// <c>POST /v1/loras</c> may only load adapter files inside this directory, named relative
/// to it, and every adapter at its top level is loaded at startup. Without it, runtime
/// loading is refused (403), because the route would otherwise open any file the server
/// process can read on behalf of an unauthenticated client.
/// </para>
/// <para>
/// Deliberately separate from <see cref="TensorSharp.Runtime.LoraCliFlags"/>: those are
/// the Qwen-Image-2.1 / MiniMax-H3 plug-in flags, shared with the CLI and pinned to exactly
/// three spellings by the flag-drift tests, and a text model refuses them.
/// </para>
/// </summary>
public static class TextLoraHostOptions
{
    public const string DirectoryFlag = "--loras-dir";

    public const string EnvironmentVariable = "TS_LORAS_DIR";
}
