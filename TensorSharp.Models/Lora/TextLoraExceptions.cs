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

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// The loaded model cannot apply a text-LoRA adapter in full (see
    /// <see cref="TextLoraBinding.RefusalMessage"/>). A <see cref="NotSupportedException"/>, so
    /// callers that answer an adapter they cannot use with 400 keep doing so;
    /// <see cref="UnsupportedModules"/> lets them name the modules in a structured error.
    /// </summary>
    public sealed class TextLoraNotApplicableException : NotSupportedException
    {
        public TextLoraNotApplicableException(string message, IReadOnlyList<string> unsupportedModules)
            : base(message)
        {
            UnsupportedModules = unsupportedModules ?? Array.Empty<string>();
        }

        /// <summary>The modules the model cannot apply, layers collapsed.</summary>
        public IReadOnlyList<string> UnsupportedModules { get; }
    }

    /// <summary>
    /// A forward ran with a text-LoRA adapter selected but did not apply it to every weight
    /// the plan put it on: some path bypassed the <c>LinearForward</c> hook. The forward's
    /// output is discarded and its requests fail, rather than return partly adapted text.
    /// </summary>
    public sealed class TextLoraNotAppliedException : InvalidOperationException
    {
        public TextLoraNotAppliedException(string message) : base(message) { }
    }
}
