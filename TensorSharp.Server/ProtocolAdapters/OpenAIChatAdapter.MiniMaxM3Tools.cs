// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace TensorSharp.Server.ProtocolAdapters;

public sealed partial class OpenAIChatAdapter
{
    private static bool IsMiniMaxM3(string architecture)
        => ChatProtocolRegistry.For(architecture)?.Id == "minimax-m3";

    /// <summary>
    /// <c>tool_choice</c> <c>required</c> or a named function on MiniMax-M3: the reply must
    /// open a <c>]&lt;]minimax[&gt;[&lt;tool_call&gt;</c> block of well-formed calls to the
    /// client's functions straight after its <c>&lt;/mm:think&gt;</c>. <c>auto</c> and
    /// <c>none</c> stay unconstrained (the parser reads whatever the model writes), as for
    /// every family without a tool grammar. <see cref="ValidateClientToolChoice"/> has
    /// already checked the choice against the declared functions.
    /// </summary>
    internal static MiniMaxM3ToolGrammar? PrepareMiniMaxM3ToolGrammar(JsonElement body,
        List<ToolFunction>? clientTools, StructuredOutputFormat? responseFormat)
    {
        if (!body.TryGetProperty("tool_choice", out var requested))
            return null;
        var choice = MiniMaxM3ToolChoice.Auto;
        string? name = null;
        if (requested.ValueKind == JsonValueKind.String && requested.GetString() == "required")
            choice = MiniMaxM3ToolChoice.Required;
        else if (requested.ValueKind == JsonValueKind.Object)
        {
            choice = MiniMaxM3ToolChoice.Named;
            name = requested.GetProperty("function").GetProperty("name").GetString();
        }
        if (choice == MiniMaxM3ToolChoice.Auto)
            return null;
        if (responseFormat != null)
            throw new NotSupportedException("A required tool call cannot be combined with response_format.");

        bool parallel = !body.TryGetProperty("parallel_tool_calls", out var parallelProperty)
            || parallelProperty.ValueKind != JsonValueKind.False;
        var plan = MiniMaxM3ToolGrammar.Compile(clientTools ?? [], choice, name, parallel);
        _ = Grammar.Parse(plan.Source); // Reject unsupported recipes before any SSE headers or inference.
        return plan;
    }
}
