// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    public static partial class ChatTemplate
    {
        public const string DeepSeek41ImagePlaceholder = "<｜deepseek_image｜>";
        /// <summary>
        /// DeepSeek V4.1 text protocol, following DeepSeek's reference encoder as
        /// adapted in vLLM's deepseek_v41_encoding.py. The published V4.1 GGUF
        /// disables automatic BOS insertion, so the prompt contains BOS explicitly.
        /// Tool use preserves past reasoning; ordinary chat drops past reasoning.
        /// This protocol is independent of model execution support.
        /// </summary>
        public static string RenderDeepSeek41(List<ChatMessage> messages, bool addGenerationPrompt = true,
            bool enableThinking = false, List<ToolFunction>? tools = null,
            int reasoningEffort = 50, bool dropThinking = true)
        {
            ArgumentNullException.ThrowIfNull(messages);
            if (reasoningEffort < 1 || reasoningEffort > 100)
                throw new ArgumentOutOfRangeException(nameof(reasoningEffort), "Reasoning effort must be within [1, 100].");

            bool hasTools = tools is { Count: > 0 };
            var normalized = InjectMultimodalTokens(messages, "deepseek41");
            if (hasTools && !normalized.Any(m => m.Role == "system"))
                normalized.Insert(0, new ChatMessage { Role = "system" });
            OrderDeepSeek41ToolResults(normalized);

            // Tool results share the user role. Merge before applying the history
            // policy. A merged user turn that holds only tool results does not
            // start a new turn (the 1017 reference encoder's find_last_user_index),
            // so the reasoning of the assistant that called those tools stays current.
            var turns = new List<ChatMessage>();
            var toolOnlyTurns = new HashSet<ChatMessage>(ReferenceEqualityComparer.Instance);
            foreach (ChatMessage message in normalized)
            {
                bool mergeable = message.Role is "user" or "tool";
                string content = message.Role == "tool"
                    ? "<tool_result>" + (message.Content ?? "") + "</tool_result>"
                    : message.Content ?? "";
                if (mergeable && turns.Count > 0 && turns[^1].Role == "user")
                {
                    turns[^1].Content += "\n\n" + content;
                    if (message.Role == "user") toolOnlyTurns.Remove(turns[^1]);
                }
                else if (mergeable)
                {
                    turns.Add(new ChatMessage { Role = "user", Content = content });
                    if (message.Role == "tool") toolOnlyTurns.Add(turns[^1]);
                }
                else
                    turns.Add(message);
            }

            bool effectiveDropThinking = dropThinking && !hasTools;
            int lastUser = FindLastDeepSeek41User(turns, toolOnlyTurns);
            if (enableThinking && effectiveDropThinking)
            {
                turns = turns.Where((m, i) => i >= lastUser || m.Role != "developer").ToList();
                lastUser = FindLastDeepSeek41User(turns, toolOnlyTurns);
            }

            var sb = new StringBuilder("<｜begin▁of▁sentence｜>");
            bool toolsRendered = false;
            for (int i = 0; i < turns.Count; i++)
            {
                ChatMessage message = turns[i];
                if (i == 0)
                {
                    if (enableThinking || message.Role == "system")
                        sb.Append("<｜System｜>");
                    if (enableThinking)
                        sb.Append("Reasoning Effort: ").Append(reasoningEffort)
                            .Append(" (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
                }

                switch (message.Role)
                {
                    case "system":
                        if (i > 0) sb.Append("<｜System｜>");
                        sb.Append(message.Content ?? "");
                        if (tools is { Count: > 0 } && !toolsRendered)
                        {
                            sb.Append("\n\n").Append(DeepSeek41ToolsHeader);
                            sb.AppendJoin('\n', tools.Select(t => SpaceDeepSeek41Json(DeepSeek41ToolSchema(t))));
                            sb.Append("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n");
                            toolsRendered = true;
                        }
                        break;
                    case "user":
                    case "developer":
                        sb.Append("<｜User｜>").Append(message.Content ?? "");
                        break;
                    case "latest_reminder":
                        sb.Append("<｜latest_reminder｜>").Append(message.Content ?? "");
                        break;
                    case "assistant":
                        if (enableThinking && (!effectiveDropThinking || i > lastUser))
                            sb.Append(message.Thinking ?? "").Append("</think>");
                        sb.Append(message.Content ?? "");
                        if (message.ToolCalls is { Count: > 0 })
                            AppendDeepSeek41ToolCalls(sb, message.ToolCalls);
                        sb.Append("<｜end▁of▁sentence｜>");
                        break;
                    default:
                        throw new ArgumentException($"Unsupported DeepSeek V4.1 message role: {message.Role}.", nameof(messages));
                }

                bool followedByAssistant = i + 1 < turns.Count && turns[i + 1].Role is "assistant" or "latest_reminder";
                bool generation = i == turns.Count - 1 && addGenerationPrompt;
                if ((followedByAssistant || generation) &&
                    (message.Role is "user" or "developer" || message.Role == "system" && i > 0))
                {
                    sb.Append("<｜Assistant｜>");
                    sb.Append(enableThinking && (!effectiveDropThinking || i >= lastUser) ? "<think>" : "</think>");
                }
            }
            return sb.ToString();
        }

        private static int FindLastDeepSeek41User(List<ChatMessage> messages, HashSet<ChatMessage> toolOnlyTurns)
        {
            for (int i = messages.Count - 1; i >= 0; i--)
                if ((messages[i].Role is "user" or "developer" || messages[i].Role == "system" && i > 0) &&
                    !toolOnlyTurns.Contains(messages[i]))
                    return i;
            return -1;
        }

        private static string DeepSeek41ToolSchema(ToolFunction tool)
        {
            if (tool.ParametersSchemaJson == null) return ToolFunctionToJson(tool);
            using var schema = JsonDocument.Parse(tool.ParametersSchemaJson);
            var function = new Dictionary<string, object?> { ["name"] = tool.Name };
            if (!string.IsNullOrEmpty(tool.Description)) function["description"] = tool.Description;
            function["parameters"] = schema.RootElement;
            return JsonSerializer.Serialize(function);
        }

        private static void OrderDeepSeek41ToolResults(List<ChatMessage> messages)
        {
            var callOrder = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < messages.Count; i++)
            {
                if (messages[i].Role == "assistant" && messages[i].ToolCalls is { Count: > 0 } calls)
                {
                    callOrder.Clear();
                    for (int c = 0; c < calls.Count; c++)
                        if (!string.IsNullOrEmpty(calls[c].Id)) callOrder[calls[c].Id!] = c;
                }
                if (messages[i].Role is not ("user" or "tool")) continue;
                var positions = new List<int>();
                int end = i;
                while (end < messages.Count && messages[end].Role is "user" or "tool")
                {
                    if (messages[end].Role == "tool") positions.Add(end);
                    end++;
                }
                if (positions.Count > 1 && callOrder.Count > 0)
                {
                    var ordered = positions.Select(p => messages[p])
                        .OrderBy(m => m.ToolCallId != null && callOrder.TryGetValue(m.ToolCallId, out int n) ? n : 0).ToArray();
                    for (int p = 0; p < positions.Count; p++) messages[positions[p]] = ordered[p];
                }
                i = end - 1;
            }
        }

        private static void AppendDeepSeek41ToolCalls(StringBuilder sb, List<ToolCall> calls)
        {
            // Parsed assistant content already includes whitespace before the tool
            // block. Reuse that separator so replaying history does not insert tokens.
            // Preserve longer whitespace runs and all tool argument bytes.
            int trailingNewlines = 0;
            for (int i = sb.Length - 1; i >= 0 && sb[i] == '\n' && trailingNewlines < 2; i--)
                trailingNewlines++;
            sb.Append('\n', 2 - trailingNewlines);
            sb.Append("<｜DSML｜ calls>\n");
            for (int i = 0; i < calls.Count; i++)
            {
                ToolCall call = calls[i];
                sb.Append("<｜DSML｜ invoke name=\"").Append(call.Name).Append("\">\n");
                bool first = true;
                foreach (var parameter in call.Arguments ?? new Dictionary<string, object?>())
                {
                    if (!first) sb.Append('\n');
                    first = false;
                    string? text = parameter.Value switch
                    {
                        string value => value,
                        JsonElement { ValueKind: JsonValueKind.String } value => value.GetString(),
                        _ => null,
                    };
                    bool rawString = text != null && !Grammar.DeepSeek41ToolGrammar.ContainsReservedMarkup(text);
                    sb.Append("<｜DSML｜ parameter name=\"").Append(parameter.Key)
                        .Append("\" string=\"").Append(rawString ? "true" : "false").Append("\">");
                    sb.Append(rawString ? text : SpaceDeepSeek41Json(JsonSerializer.Serialize(parameter.Value), protectToolDelimiters: true));
                    sb.Append("</｜DSML｜ parameter>");
                }
                sb.Append("\n</｜DSML｜ invoke>\n");
            }
            sb.Append("</｜DSML｜ calls>");
        }

        // Python json.dumps uses ', ' and ': ' while retaining Unicode. Match
        // those bytes: punctuation inside strings must remain untouched.
        private static string SpaceDeepSeek41Json(string json, bool protectToolDelimiters = false)
        {
            using JsonDocument document = JsonDocument.Parse(json);
            string compact = JsonSerializer.Serialize(document.RootElement, DeepSeek41JsonOptions);
            // Replaying parsed arguments must not undo the JSON-string escape
            // that kept a literal marker inside the parameter's value. Apply
            // protection after Unicode-preserving normalization, including keys
            // and strings inside nested JSON. Ordinary history bytes are retained.
            if (protectToolDelimiters && Grammar.DeepSeek41ToolGrammar.ContainsReservedMarkup(compact))
                compact = compact.Replace("<", "\\u003c", StringComparison.Ordinal);
            var sb = new StringBuilder(compact.Length);
            bool quoted = false, escaped = false;
            foreach (char c in compact)
            {
                sb.Append(c);
                if (escaped) { escaped = false; continue; }
                if (quoted && c == '\\') { escaped = true; continue; }
                if (c == '"') quoted = !quoted;
                if (!quoted && c is ',' or ':') sb.Append(' ');
            }
            return sb.ToString();
        }

        private static readonly JsonSerializerOptions DeepSeek41JsonOptions = new()
        {
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        private const string DeepSeek41ToolsHeader = 
            "## Tools\n"
            +"\n"
            +"You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:\n"
            +"\n"
            +"<｜DSML｜ calls>\n"
            +"<｜DSML｜ invoke name=\"$TOOL_NAME\">\n"
            +"<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n"
            + "...\n"
            + "</｜DSML｜ invoke>\n"
            +"<｜DSML｜ invoke name=\"$TOOL_NAME2\">\n"
            + "...\n"
            + "</｜DSML｜ invoke>\n"
            + "</｜DSML｜ calls>\n"
            +"\n"
            +"String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n"
            + "\n"
            +"If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n"
            + "\n"
            +"Otherwise, output directly after </think> with tool calls or final response.\n"
            + "\n"
            + "### Available Tool Schemas"
            + "\n"
            + "\n";
    }
}
