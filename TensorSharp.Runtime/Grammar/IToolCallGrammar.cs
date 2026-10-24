// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorSharp.Runtime.Grammar;

/// <summary>
/// A family's compiled tool-call grammar (<c>tool_choice</c> required, named or auto) as an
/// immutable recipe: every generation gets an independent constraint, armed where that
/// family's reply leaves its reasoning.
/// </summary>
public interface IToolCallGrammar
{
    /// <param name="thinking">Whether the request turned reasoning on; families whose reply
    /// ends its reasoning the same way in both modes ignore it.</param>
    GrammarConstraint NewConstraint(ITokenizer tokenizer, bool thinking);
}
