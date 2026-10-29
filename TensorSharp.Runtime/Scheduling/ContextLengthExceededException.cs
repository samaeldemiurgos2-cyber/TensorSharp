// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// A request the engine refuses at submission because its prompt does not fit the model's
    /// context window (<see cref="IModelArchitecture.MaxContextLength"/>, MAX_CONTEXT): the prompt
    /// must leave room for at least one generated token, or every token of it would be computed
    /// only for the first decode step to overrun the KV cache. Nothing about the request was
    /// computed or allocated when this is thrown.
    /// <para>
    /// This is an input the caller can fix (shorten the prompt or raise MAX_CONTEXT), not an
    /// engine failure: hosts report it in one line - the CLI as a configuration error, the
    /// server as a 400 - instead of a stack trace. A reply that merely RUNS into the window is
    /// not refused; the engine ends it there with the length-cap finish reason.
    /// </para>
    /// </summary>
    public sealed class ContextLengthExceededException : InvalidOperationException
    {
        public ContextLengthExceededException(int promptTokens, int contextTokens, string? message = null)
            : base(message ?? Describe(promptTokens, contextTokens))
        {
            PromptTokens = promptTokens;
            ContextTokens = contextTokens;
        }

        /// <summary>The refused prompt's length, in tokens.</summary>
        public int PromptTokens { get; }

        /// <summary>The context window the prompt was measured against, in tokens.</summary>
        public int ContextTokens { get; }

        private static string Describe(int promptTokens, int contextTokens) =>
            $"The prompt is {promptTokens} tokens, but the model's context window is {contextTokens} tokens " +
            "and must leave room for at least one generated token. Shorten the input, or raise MAX_CONTEXT " +
            "if the model supports a longer context.";
    }
}
