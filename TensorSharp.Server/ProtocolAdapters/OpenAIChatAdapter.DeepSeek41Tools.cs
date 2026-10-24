// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace TensorSharp.Server.ProtocolAdapters;

public sealed partial class OpenAIChatAdapter
{
    private static bool IsDeepSeek41(string architecture)
        => ChatProtocolRegistry.For(architecture)?.Id == "deepseek41";

    internal static DeepSeek41ToolGrammar? PrepareDeepSeek41ToolGrammar(JsonElement body,
        List<ToolFunction>? clientTools, List<ToolFunction>? effectiveTools, StructuredOutputFormat? responseFormat)
    {
        var choice = DeepSeek41ToolChoice.Auto;
        string? name = null;
        bool explicitChoice = body.TryGetProperty("tool_choice", out var requested);
        if (explicitChoice)
        {
            if (requested.ValueKind == JsonValueKind.String)
                choice = requested.GetString() switch
                {
                    "auto" => DeepSeek41ToolChoice.Auto,
                    "none" => DeepSeek41ToolChoice.None,
                    "required" => DeepSeek41ToolChoice.Required,
                    _ => throw new NotSupportedException("DeepSeek V4.1 tool_choice must be auto, none, required, or a named function."),
                };
            else if (requested.ValueKind == JsonValueKind.Object &&
                     requested.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String && type.GetString() == "function" &&
                     requested.TryGetProperty("function", out var function) && function.ValueKind == JsonValueKind.Object &&
                     function.TryGetProperty("name", out var named) && named.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(named.GetString()))
            {
                choice = DeepSeek41ToolChoice.Named;
                name = named.GetString();
            }
            else throw new NotSupportedException("DeepSeek V4.1 tool_choice names a function as {type: function, function: {name: ...}}.");
        }

        bool parallel = true;
        if (body.TryGetProperty("parallel_tool_calls", out var parallelProperty))
        {
            if (parallelProperty.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                throw new JsonException("parallel_tool_calls must be a boolean.");
            parallel = parallelProperty.GetBoolean();
        }
        if (body.TryGetProperty("tools", out var declared))
        {
            if (declared.ValueKind != JsonValueKind.Array) throw new JsonException("tools must be an array.");
            foreach (var tool in declared.EnumerateArray())
            {
                if (tool.ValueKind != JsonValueKind.Object ||
                    (tool.TryGetProperty("type", out var type) && (type.ValueKind != JsonValueKind.String || type.GetString() != "function")) ||
                    !tool.TryGetProperty("function", out var function) || function.ValueKind != JsonValueKind.Object)
                    throw new NotSupportedException("DeepSeek V4.1 tools must be function declarations.");
                if (function.TryGetProperty("parameters", out var parameters) && parameters.ValueKind is not (JsonValueKind.Null or JsonValueKind.Object))
                    throw new JsonException("Function parameters must be a JSON Schema object or null.");
            }
        }
        if (choice is DeepSeek41ToolChoice.Required or DeepSeek41ToolChoice.Named && clientTools is not { Count: > 0 })
            throw new NotSupportedException("tool_choice requires at least one client-declared function.");

        // JSON response grammar already excludes tool syntax; history and the
        // original catalog remain available for explicit final-answer turns.
        if (responseFormat != null)
        {
            if (choice is DeepSeek41ToolChoice.Required or DeepSeek41ToolChoice.Named)
                throw new NotSupportedException("A required tool call cannot be combined with response_format.");
            return null;
        }

        // Required/named calls refer to the client's functions. An internal skill
        // round must not satisfy that contract invisibly on the caller's behalf.
        var tools = choice is DeepSeek41ToolChoice.Required or DeepSeek41ToolChoice.Named ? clientTools : effectiveTools;
        if (choice == DeepSeek41ToolChoice.Auto && tools is not { Count: > 0 })
            return explicitChoice ? DeepSeek41ToolGrammar.Compile([], DeepSeek41ToolChoice.None) : null;
        var plan = DeepSeek41ToolGrammar.Compile(tools ?? [], choice, name, parallel);
        _ = Grammar.Parse(plan.Source); // Reject unsupported recipes before any SSE headers or inference.
        return plan;
    }

    private SamplingConfig? WithToolCallGrammar(SamplingConfig? config,
        IToolCallGrammar? plan, bool thinking)
    {
        if (plan == null) return config;
        var tokenizer = _svc.Model?.Tokenizer
            ?? throw new InvalidOperationException("The tool-call grammar requires a loaded tokenizer.");
        var result = (config ?? SamplingConfig.Default).Clone();
        result.Grammar = plan.NewConstraint(tokenizer, thinking);
        return result;
    }
}
