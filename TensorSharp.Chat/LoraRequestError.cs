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
using TensorSharp.Models.Lora;

namespace TensorSharp.Server
{
    /// <summary>
    /// Why a request cannot run with the adapter its <c>lora</c> field names, for a 400 in the
    /// OpenAI error shape (<c>param: "lora"</c>, <see cref="Code"/>, and
    /// <c>unsupported_modules</c> when the loaded model cannot apply the adapter).
    /// </summary>
    public sealed record LoraRequestError(string Message, string Code, IReadOnlyList<string> UnsupportedModules)
    {
        /// <summary>No adapter of that name is loaded.</summary>
        public const string NotFound = "lora_not_found";

        /// <summary>No model is loaded to apply it to.</summary>
        public const string ModelNotLoaded = "model_not_loaded";

        /// <summary>The loaded model cannot apply the adapter in full (family, backend,
        /// multi-device mode, architecture, or modules it does not run through the hook).</summary>
        public const string NotApplicable = "lora_not_applicable";

        /// <summary>The body of the OpenAI <c>error</c> member: <c>message</c>, <c>type</c>,
        /// <c>param</c>, <c>code</c>, and <c>unsupported_modules</c> for
        /// <see cref="NotApplicable"/>.</summary>
        public Dictionary<string, object?> ToOpenAIError()
        {
            var error = new Dictionary<string, object?>
            {
                ["message"] = Message,
                ["type"] = "invalid_request_error",
                ["param"] = "lora",
                ["code"] = Code,
            };
            if (Code == NotApplicable)
                error["unsupported_modules"] = UnsupportedModules;
            return error;
        }
    }

    /// <summary>A loaded adapter, with whether the loaded model can apply it and, when it
    /// cannot, why.</summary>
    public sealed record LoraListing(LoraAdapterInfo Info, bool Applicable, string? Reason);
}
