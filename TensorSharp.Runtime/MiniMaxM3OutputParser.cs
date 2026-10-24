// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// MiniMax-M3 replies, streamed in chunks cut anywhere:
    /// <code>
    /// [&lt;mm:think&gt;reasoning&lt;/mm:think&gt; | &lt;/mm:think&gt;] content
    /// ]&lt;]minimax[&gt;[&lt;tool_call&gt;
    /// ]&lt;]minimax[&gt;[&lt;invoke name="f"&gt;]&lt;]minimax[&gt;[&lt;city&gt;Paris]&lt;]minimax[&gt;[&lt;/city&gt;]&lt;]minimax[&gt;[&lt;/invoke&gt;
    /// ]&lt;]minimax[&gt;[&lt;/tool_call&gt;
    /// </code>
    /// The adaptive generation prompt ends at <c>]~b]ai\n</c>, so the model writes its own
    /// opener: a reasoning block, or the bare <c>&lt;/mm:think&gt;</c> its history renders for a
    /// turn without reasoning, which is dropped. A prompt that already opened the block
    /// (thinking requested) starts the reply inside it.
    ///
    /// <para>
    /// Argument values are untyped XML text, so they are typed from the declared JSON Schema
    /// the way llama.cpp's <c>peg-minimax-m3</c> parser does: a property that may be a string
    /// keeps the raw text up to its close tag (even text that looks like markup); numbers,
    /// booleans and null are JSON literals; objects and arrays are nested elements
    /// (<c>&lt;item&gt;</c> for array members), or JSON text. A repeated element becomes a
    /// list and an element no schema describes stays a string, as in vLLM. A block that
    /// cannot be read is returned as content, never thrown and never dropped; an
    /// unterminated last call at the end of the reply is not dispatched.
    /// </para>
    /// </summary>
    public sealed class MiniMaxM3OutputParser : IOutputParser
    {
        private enum State { Start, Thinking, Content, ToolCalls }

        private const string Ns = ChatTemplate.MiniMaxM3Namespace;
        private const string ThinkOpen = ChatTemplate.MiniMaxM3ThinkOpen;
        private const string ThinkClose = ChatTemplate.MiniMaxM3ThinkClose;
        internal const string CallsOpen = Ns + "<tool_call>";
        private const string CallsClose = Ns + "</tool_call>";
        private const string InvokeOpen = Ns + "<invoke name=";
        private const string InvokeClose = Ns + "</invoke>";
        private const string ElementOpen = Ns + "<";
        private const string ElementCloseOpen = Ns + "</";
        private const int MaxDepth = 64;

        private State _state;
        private readonly StringBuilder _buffer = new();
        private bool _started;
        private bool _startOfThinking;
        private int _callIndex;
        // Watermark into the open block already surfaced as ParsedOutput.ToolCallText: the
        // buffer keeps the whole block for the parse, a UI is shown it as it is written.
        private int _blockReportedChars;
        private readonly Dictionary<string, JsonElement> _schemas = new(StringComparer.Ordinal);

        public bool HasThinkingSupport => true;
        public bool HasToolSupport => true;
        public bool AlwaysRequired => true;
        public bool IsThinkingChannelOpen => _state == State.Thinking;

        public void Init(bool enableThinking, List<ToolFunction>? tools)
        {
            _buffer.Clear();
            _started = false;
            _callIndex = 0;
            _blockReportedChars = 0;
            // Thinking on renders `]~b]ai\n<mm:think>`; off renders the adaptive prompt.
            _state = enableThinking ? State.Thinking : State.Start;
            _startOfThinking = enableThinking;
            _schemas.Clear();
            if (tools == null)
                return;
            foreach (ToolFunction tool in tools)
            {
                if (string.IsNullOrEmpty(tool.Name) || _schemas.ContainsKey(tool.Name))
                    continue;
                JsonElement? schema = ParameterSchema(tool);
                if (schema != null)
                    _schemas[tool.Name] = schema.Value;
            }
        }

        /// <summary>What the generation prompt ended with: <c>&lt;mm:think&gt;</c> starts the
        /// reply inside the reasoning block, <c>&lt;/mm:think&gt;</c> in the answer, and an
        /// empty suffix lets the model's own opener decide. Null is "unknown" and keeps the
        /// state <see cref="Init"/> chose.</summary>
        public void SetGenerationPromptSuffix(string? suffix)
        {
            if (suffix == null || _started)
                return;
            string tail = suffix.TrimEnd();
            if (tail.EndsWith(ThinkOpen, StringComparison.Ordinal))
            {
                _state = State.Thinking;
                _startOfThinking = true;
            }
            else if (tail.EndsWith(ThinkClose, StringComparison.Ordinal))
                _state = State.Content;
            else
            {
                _state = State.Start;
                _startOfThinking = false;
            }
        }

        public ParsedOutput Add(string text, bool done)
        {
            if (!string.IsNullOrEmpty(text))
            {
                _buffer.Append(text);
                _started = true;
            }
            var content = new StringBuilder();
            var thinking = new StringBuilder();
            var calls = new List<ToolCall>();
            var callText = new StringBuilder();
            string? callName = null;

            bool progress = true;
            while (progress)
            {
                progress = false;
                string buf = _buffer.ToString();
                if (buf.Length == 0)
                    break;

                switch (_state)
                {
                    case State.Start:
                    {
                        // Whitespace the model may put before its opener is held with it;
                        // without an opener it is the answer's own.
                        int at = 0;
                        while (at < buf.Length && char.IsWhiteSpace(buf[at]))
                            at++;
                        if (StartsWithAt(buf, at, ThinkOpen))
                        {
                            Consume(at + ThinkOpen.Length);
                            _state = State.Thinking;
                            progress = true;
                        }
                        else if (StartsWithAt(buf, at, ThinkClose))
                        {
                            Consume(at + ThinkClose.Length);
                            _state = State.Content;
                            progress = true;
                        }
                        else if (!done && (at == buf.Length || IsPrefixAt(buf, at, ThinkOpen) || IsPrefixAt(buf, at, ThinkClose)))
                        {
                            // Undecided: hold it until the next piece.
                        }
                        else
                        {
                            _state = State.Content;
                            progress = true;
                        }
                        break;
                    }

                    case State.Thinking:
                    {
                        // A prompt that opened the block may meet the model's own opener.
                        if (_startOfThinking)
                        {
                            if (StartsWithAt(buf, 0, ThinkOpen))
                            {
                                Consume(ThinkOpen.Length);
                                _startOfThinking = false;
                                progress = true;
                                break;
                            }
                            if (!done && IsPrefixAt(buf, 0, ThinkOpen))
                                break;
                            _startOfThinking = false;
                        }
                        int close = buf.IndexOf(ThinkClose, StringComparison.Ordinal);
                        if (close >= 0)
                        {
                            thinking.Append(buf, 0, close);
                            Consume(close + ThinkClose.Length);
                            _state = State.Content;
                            progress = true;
                        }
                        else
                        {
                            int hold = done ? 0 : HeldPrefix(buf, ThinkClose);
                            thinking.Append(buf, 0, buf.Length - hold);
                            Consume(buf.Length - hold);
                        }
                        break;
                    }

                    case State.Content:
                    {
                        int open = buf.IndexOf(CallsOpen, StringComparison.Ordinal);
                        if (open >= 0)
                        {
                            content.Append(buf, 0, open);
                            Consume(open);
                            _state = State.ToolCalls;
                            progress = true;
                        }
                        else
                        {
                            int hold = done ? 0 : HeldPrefix(buf, CallsOpen);
                            content.Append(buf, 0, buf.Length - hold);
                            Consume(buf.Length - hold);
                        }
                        break;
                    }

                    case State.ToolCalls:
                    {
                        // The buffer starts with CallsOpen.
                        int close = buf.IndexOf(CallsClose, CallsOpen.Length, StringComparison.Ordinal);
                        if (close >= 0)
                        {
                            int end = close + CallsClose.Length;
                            ReadBlock(buf, CallsOpen.Length, close, end, closed: true, calls, content);
                            Consume(end);
                            _state = State.Content;
                            _blockReportedChars = 0;
                            progress = true;
                        }
                        else if (done)
                        {
                            // The reply stopped inside the block (max_tokens, or EOS right
                            // after the last </invoke>): every complete call still counts.
                            ReadBlock(buf, CallsOpen.Length, buf.Length, buf.Length, closed: false, calls, content);
                            Consume(buf.Length);
                            _state = State.Content;
                            _blockReportedChars = 0;
                        }
                        else
                        {
                            // Progress only: the calls themselves surface when the block is read.
                            int reportable = buf.Length - HeldPrefix(buf, CallsClose);
                            int from = Math.Max(_blockReportedChars, CallsOpen.Length);
                            if (reportable > from)
                            {
                                callText.Append(buf, from, reportable - from);
                                _blockReportedChars = reportable;
                            }
                            callName = OpenInvokeName(buf);
                        }
                        break;
                    }
                }
            }

            return new ParsedOutput
            {
                Content = content.ToString(),
                Thinking = thinking.ToString(),
                ToolCalls = calls.Count > 0 ? calls : null,
                ToolCallText = callText.ToString(),
                ToolCallName = callName,
            };
        }

        /// <summary>The name of the last invoke the open block has started.</summary>
        private static string? OpenInvokeName(string block)
        {
            int start = block.LastIndexOf(InvokeOpen, StringComparison.Ordinal);
            if (start < 0)
                return null;
            int nameStart = start + InvokeOpen.Length;
            int nameEnd = block.IndexOf('>', nameStart);
            return nameEnd < 0 ? null : block.Substring(nameStart, nameEnd - nameStart).Trim().Trim('"', '\'');
        }

        private void Consume(int count) => _buffer.Remove(0, count);

        private static bool StartsWithAt(string text, int at, string tag)
            => at + tag.Length <= text.Length && string.CompareOrdinal(text, at, tag, 0, tag.Length) == 0;

        /// <summary>The rest of <paramref name="text"/> from <paramref name="at"/> is a proper
        /// prefix of <paramref name="tag"/>.</summary>
        private static bool IsPrefixAt(string text, int at, string tag)
        {
            int length = text.Length - at;
            return length > 0 && length < tag.Length && string.CompareOrdinal(text, at, tag, 0, length) == 0;
        }

        /// <summary>Length of the longest end of <paramref name="buffer"/> that could start
        /// <paramref name="tag"/>.</summary>
        private static int HeldPrefix(string buffer, string tag)
        {
            for (int length = Math.Min(tag.Length - 1, buffer.Length); length > 0; length--)
                if (string.CompareOrdinal(buffer, buffer.Length - length, tag, 0, length) == 0)
                    return length;
            return 0;
        }

        /// <summary>
        /// Read the invokes between <paramref name="bodyStart"/> and <paramref name="bodyEnd"/>.
        /// Anything that is not a readable call - stray text, a malformed invoke - goes to
        /// <paramref name="content"/> verbatim, framed as the model wrote it, so nothing is
        /// lost and nothing is invented; a block with no readable call at all is returned
        /// whole. The exception is an unclosed block's unterminated last invoke after calls
        /// that did complete: the model was cut off writing it, and half a call is not a call.
        /// </summary>
        private void ReadBlock(string buf, int bodyStart, int bodyEnd, int blockEnd, bool closed,
            List<ToolCall> calls, StringBuilder content)
        {
            var parsed = new List<ToolCall>();
            var leftovers = new StringBuilder();
            int pos = bodyStart;
            while (pos < bodyEnd)
            {
                int start = buf.IndexOf(InvokeOpen, pos, bodyEnd - pos, StringComparison.Ordinal);
                if (start < 0)
                {
                    leftovers.Append(buf, pos, bodyEnd - pos);
                    break;
                }
                leftovers.Append(buf, pos, start - pos);
                int end = buf.IndexOf(InvokeClose, start, bodyEnd - start, StringComparison.Ordinal);
                if (end < 0)
                {
                    if (closed)
                        leftovers.Append(buf, start, bodyEnd - start);
                    break;
                }
                int invokeEnd = end + InvokeClose.Length;
                ToolCall? call = ReadInvoke(buf, start + InvokeOpen.Length, end);
                if (call != null)
                    parsed.Add(call);
                else
                    leftovers.Append(buf, start, invokeEnd - start);
                pos = invokeEnd;
            }

            if (parsed.Count == 0)
            {
                // Nothing in it was a call: the block is text the model wrote.
                int blockStart = bodyStart - CallsOpen.Length;
                content.Append(buf, blockStart, blockEnd - blockStart);
                return;
            }
            foreach (ToolCall call in parsed)
            {
                call.Index = _callIndex++;
                calls.Add(call);
            }
            // The line breaks between invokes are the block's layout, not text.
            string rest = leftovers.ToString().Trim();
            if (rest.Length > 0)
                content.Append(rest);
        }

        private ToolCall? ReadInvoke(string buf, int nameStart, int bodyEnd)
        {
            int headerEnd = buf.IndexOf('>', nameStart, bodyEnd - nameStart);
            if (headerEnd < 0)
                return null;
            string name = buf.Substring(nameStart, headerEnd - nameStart).Trim();
            if (name.Length >= 2 && (name[0] == '"' || name[0] == '\'') && name[^1] == name[0])
                name = name.Substring(1, name.Length - 2);
            if (name.Length == 0 || name.Contains('<') || name.Contains('"'))
                return null;

            _schemas.TryGetValue(name, out JsonElement schema);
            JsonElement? root = schema.ValueKind == JsonValueKind.Object ? schema : null;
            int pos = headerEnd + 1;
            Dictionary<string, object?>? arguments = ReadMembers(buf, ref pos, bodyEnd, null, root, root, 0);
            if (arguments == null)
                return null;
            SkipWhitespace(buf, ref pos, bodyEnd);
            return pos == bodyEnd ? new ToolCall { Name = name, Arguments = arguments } : null;
        }

        /// <summary>Elements up to <paramref name="close"/> (or <paramref name="end"/> when
        /// null) as an object; a repeated name collects into a list.</summary>
        private static Dictionary<string, object?>? ReadMembers(string buf, ref int pos, int end, string? close,
            JsonElement? schema, JsonElement? root, int depth)
        {
            var members = new Dictionary<string, object?>(StringComparer.Ordinal);
            var repeated = new HashSet<string>(StringComparer.Ordinal);
            while (true)
            {
                SkipWhitespace(buf, ref pos, end);
                if (close == null ? pos == end : StartsWithWithin(buf, pos, end, close))
                {
                    if (close != null)
                        pos += close.Length;
                    return members;
                }
                if (!ReadElementName(buf, ref pos, end, out string name))
                    return null;
                object? value;
                if (!ReadValue(buf, ref pos, end, ElementCloseOpen + name + ">",
                        SchemaProperty(schema, name, root), root, depth + 1, out value))
                    return null;
                if (!members.TryGetValue(name, out object? existing))
                    members[name] = value;
                else if (repeated.Contains(name))
                    ((List<object?>)existing!).Add(value);
                else
                {
                    members[name] = new List<object?> { existing, value };
                    repeated.Add(name);
                }
            }
        }

        private static bool ReadElementName(string buf, ref int pos, int end, out string name)
        {
            name = "";
            if (!StartsWithWithin(buf, pos, end, ElementOpen) || StartsWithWithin(buf, pos, end, ElementCloseOpen))
                return false;
            int nameStart = pos + ElementOpen.Length;
            int nameEnd = buf.IndexOf('>', nameStart, end - nameStart);
            if (nameEnd <= nameStart)
                return false;
            name = buf.Substring(nameStart, nameEnd - nameStart);
            if (name.Trim().Length == 0 || name.Contains('<') || name.Contains(Ns, StringComparison.Ordinal))
                return false;
            pos = nameEnd + 1;
            return true;
        }

        private static bool ReadValue(string buf, ref int pos, int end, string close,
            JsonElement? schema, JsonElement? root, int depth, out object? value)
        {
            value = null;
            if (depth > MaxDepth)
                return false;
            int closeAt = buf.IndexOf(close, pos, end - pos, StringComparison.Ordinal);
            if (closeAt < 0)
                return false;

            // A value that may be a string is everything up to its close tag, markup and
            // all (llama.cpp: a union with a string alternative is a string).
            if (schema != null && MayBeString(schema.Value, root, 0))
            {
                value = buf.Substring(pos, closeAt - pos);
                pos = closeAt + close.Length;
                return true;
            }

            int at = pos;
            SkipWhitespace(buf, ref at, end);
            if (StartsWithWithin(buf, at, end, ElementOpen) && !StartsWithWithin(buf, at, end, ElementCloseOpen))
            {
                int after = at;
                if (TryReadContainer(buf, ref after, end, close, schema, root, depth, out value))
                {
                    pos = after;
                    return true;
                }
                // Not well-formed nested elements: the raw text, as for a string.
                value = buf.Substring(pos, closeAt - pos);
                pos = closeAt + close.Length;
                return true;
            }

            value = schema == null ? buf.Substring(pos, closeAt - pos) : Coerce(buf.Substring(pos, closeAt - pos), schema.Value, root);
            pos = closeAt + close.Length;
            return true;
        }

        private static bool TryReadContainer(string buf, ref int pos, int end, string close,
            JsonElement? schema, JsonElement? root, int depth, out object? value)
        {
            value = null;
            JsonElement? container = null;
            bool isArray = false;
            if (schema != null)
                container = ContainerSchema(schema.Value, root, out isArray);
            if (!isArray)
            {
                Dictionary<string, object?>? members = ReadMembers(buf, ref pos, end, close, container, root, depth);
                value = members;
                return members != null;
            }

            JsonElement? items = container!.Value.TryGetProperty("items", out JsonElement itemSchema)
                && itemSchema.ValueKind == JsonValueKind.Object ? Resolve(itemSchema, root, 0) : null;
            var list = new List<object?>();
            while (true)
            {
                SkipWhitespace(buf, ref pos, end);
                if (StartsWithWithin(buf, pos, end, close))
                {
                    pos += close.Length;
                    value = list;
                    return true;
                }
                if (!ReadElementName(buf, ref pos, end, out string name))
                    return false;
                if (!ReadValue(buf, ref pos, end, ElementCloseOpen + name + ">", items, root, depth + 1, out object? item))
                    return false;
                list.Add(item);
            }
        }

        private static bool StartsWithWithin(string text, int at, int end, string tag)
            => at + tag.Length <= end && string.CompareOrdinal(text, at, tag, 0, tag.Length) == 0;

        private static void SkipWhitespace(string text, ref int pos, int end)
        {
            while (pos < end && char.IsWhiteSpace(text[pos]))
                pos++;
        }

        // ---- schema-driven typing -------------------------------------------------------

        /// <summary>The tool's declared parameters schema; a flat declaration (no JSON Schema
        /// object) is rebuilt from its parsed parameter types.</summary>
        private static JsonElement? ParameterSchema(ToolFunction tool)
        {
            string json;
            if (tool.ParametersSchemaJson != null)
                json = tool.ParametersSchemaJson;
            else if (tool.Parameters is { Count: > 0 })
            {
                var properties = new Dictionary<string, object>();
                foreach (var parameter in tool.Parameters)
                    properties[parameter.Key] = string.IsNullOrEmpty(parameter.Value.Type)
                        ? new Dictionary<string, object>()
                        : new Dictionary<string, object> { ["type"] = parameter.Value.Type };
                json = JsonSerializer.Serialize(new Dictionary<string, object>
                {
                    ["type"] = "object",
                    ["properties"] = properties,
                });
            }
            else
                return null;
            try
            {
                using JsonDocument document = JsonDocument.Parse(json);
                return document.RootElement.ValueKind == JsonValueKind.Object ? document.RootElement.Clone() : null;
            }
            catch (JsonException)
            {
                return null;
            }
        }

        /// <summary>A local <c>$ref</c> (<c>#/$defs/X</c>, <c>#/definitions/X</c>) resolved
        /// against the parameters schema.</summary>
        private static JsonElement? Resolve(JsonElement schema, JsonElement? root, int depth)
        {
            if (depth > MaxDepth || schema.ValueKind != JsonValueKind.Object)
                return schema.ValueKind == JsonValueKind.Object ? schema : null;
            if (!schema.TryGetProperty("$ref", out JsonElement reference) || reference.ValueKind != JsonValueKind.String
                || root == null)
                return schema;
            string path = reference.GetString() ?? "";
            if (!path.StartsWith("#/", StringComparison.Ordinal))
                return null;
            JsonElement target = root.Value;
            foreach (string segment in path.Substring(2).Split('/'))
            {
                string key = segment.Replace("~1", "/", StringComparison.Ordinal).Replace("~0", "~", StringComparison.Ordinal);
                if (target.ValueKind != JsonValueKind.Object || !target.TryGetProperty(key, out target))
                    return null;
            }
            return Resolve(target, root, depth + 1);
        }

        private static JsonElement? SchemaProperty(JsonElement? schema, string name, JsonElement? root)
        {
            if (schema == null)
                return null;
            JsonElement? resolved = Resolve(schema.Value, root, 0);
            if (resolved == null)
                return null;
            if (resolved.Value.TryGetProperty("properties", out JsonElement properties)
                && properties.ValueKind == JsonValueKind.Object
                && properties.TryGetProperty(name, out JsonElement property))
                return Resolve(property, root, 0);
            if (resolved.Value.TryGetProperty("additionalProperties", out JsonElement additional)
                && additional.ValueKind == JsonValueKind.Object)
                return Resolve(additional, root, 0);
            return null;
        }

        /// <summary>
        /// Caps the TOTAL work of one schema walk. Tool schemas come from the client and may share or
        /// cycle <c>$ref</c> targets; a depth cap alone still lets a union of two refs to the same
        /// target fan out 2^MaxDepth calls (a hung request thread), and a cycle through a union with
        /// no depth guard overflows the stack (a dead process). An exhausted walk answers "no
        /// information", the same as a schema that says nothing.
        /// </summary>
        private sealed class SchemaWalk
        {
            private int _remaining = MaxSchemaVisits;

            internal bool Visit() => --_remaining >= 0;
        }

        private const int MaxSchemaVisits = 4096;

        /// <summary>llama.cpp's <c>common_chat_schema::may_be_string</c>.</summary>
        internal static bool MayBeString(JsonElement schema, JsonElement? root, int depth)
            => MayBeString(schema, root, depth, new SchemaWalk());

        private static bool MayBeString(JsonElement schema, JsonElement? root, int depth, SchemaWalk walk)
        {
            if (!walk.Visit())
                return false;
            JsonElement? resolved = Resolve(schema, root, depth);
            if (resolved == null || depth > MaxDepth)
                return false;
            JsonElement s = resolved.Value;
            if (s.TryGetProperty("const", out JsonElement constant))
                return constant.ValueKind == JsonValueKind.String;
            if (s.TryGetProperty("enum", out JsonElement values) && values.ValueKind == JsonValueKind.Array)
                return values.EnumerateArray().Any(v => v.ValueKind == JsonValueKind.String);
            foreach (string union in new[] { "anyOf", "oneOf" })
                if (s.TryGetProperty(union, out JsonElement alternatives) && alternatives.ValueKind == JsonValueKind.Array)
                    return alternatives.EnumerateArray().Any(a => MayBeString(a, root, depth + 1, walk));
            if (s.TryGetProperty("allOf", out JsonElement all) && all.ValueKind == JsonValueKind.Array)
            {
                bool any = false;
                foreach (JsonElement part in all.EnumerateArray())
                {
                    if (Types(part, root, depth + 1, walk).Count == 0)
                        continue;
                    if (!MayBeString(part, root, depth + 1, walk))
                        return false;
                    any = true;
                }
                return any;
            }
            if (s.TryGetProperty("type", out JsonElement type))
            {
                if (type.ValueKind == JsonValueKind.String)
                    return type.GetString() == "string";
                if (type.ValueKind == JsonValueKind.Array)
                    return type.EnumerateArray().Any(t => t.ValueKind == JsonValueKind.String && t.GetString() == "string");
            }
            return false;
        }

        /// <summary>vLLM's <c>extract_types_from_schema</c>, without its string default: an
        /// empty set means the schema says nothing about the type.</summary>
        private static HashSet<string> Types(JsonElement schema, JsonElement? root, int depth)
            => Types(schema, root, depth, new SchemaWalk());

        private static HashSet<string> Types(JsonElement schema, JsonElement? root, int depth, SchemaWalk walk)
        {
            var types = new HashSet<string>(StringComparer.Ordinal);
            if (!walk.Visit())
                return types;
            JsonElement? resolved = Resolve(schema, root, depth);
            if (resolved == null || depth > MaxDepth)
                return types;
            JsonElement s = resolved.Value;
            if (s.TryGetProperty("type", out JsonElement type))
            {
                if (type.ValueKind == JsonValueKind.String)
                    types.Add(type.GetString()!);
                else if (type.ValueKind == JsonValueKind.Array)
                    foreach (JsonElement t in type.EnumerateArray())
                        if (t.ValueKind == JsonValueKind.String)
                            types.Add(t.GetString()!);
            }
            if (s.TryGetProperty("enum", out JsonElement values) && values.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement v in values.EnumerateArray())
                {
                    types.Add(v.ValueKind switch
                    {
                        JsonValueKind.Null => "null",
                        JsonValueKind.True or JsonValueKind.False => "boolean",
                        JsonValueKind.Number => v.TryGetInt64(out _) ? "integer" : "number",
                        JsonValueKind.String => "string",
                        JsonValueKind.Array => "array",
                        _ => "object",
                    });
                }
            }
            if (s.TryGetProperty("const", out JsonElement constant))
                types.Add(constant.ValueKind switch
                {
                    JsonValueKind.Null => "null",
                    JsonValueKind.True or JsonValueKind.False => "boolean",
                    JsonValueKind.Number => "number",
                    JsonValueKind.String => "string",
                    JsonValueKind.Array => "array",
                    _ => "object",
                });
            foreach (string union in new[] { "anyOf", "oneOf", "allOf" })
                if (s.TryGetProperty(union, out JsonElement alternatives) && alternatives.ValueKind == JsonValueKind.Array)
                    foreach (JsonElement alternative in alternatives.EnumerateArray())
                        types.UnionWith(Types(alternative, root, depth + 1, walk));
            if (types.Count == 0)
            {
                if (s.TryGetProperty("items", out _)) types.Add("array");
                else if (s.TryGetProperty("properties", out _) || s.TryGetProperty("additionalProperties", out _)) types.Add("object");
            }
            return types;
        }

        /// <summary>vLLM's <c>_container_schema</c>: the first array or object alternative.</summary>
        private static JsonElement? ContainerSchema(JsonElement schema, JsonElement? root, out bool isArray)
            => ContainerSchema(schema, root, 0, new SchemaWalk(), out isArray);

        private static JsonElement? ContainerSchema(JsonElement schema, JsonElement? root, int depth, SchemaWalk walk,
            out bool isArray)
        {
            isArray = false;
            if (depth > MaxDepth || !walk.Visit())
                return null;
            JsonElement? resolved = Resolve(schema, root, depth);
            if (resolved == null)
                return null;
            JsonElement s = resolved.Value;
            if (IsArraySchema(s))
            {
                isArray = true;
                return s;
            }
            if (HasType(s, "object") || (!s.TryGetProperty("type", out _)
                && (s.TryGetProperty("properties", out _) || s.TryGetProperty("additionalProperties", out _))))
                return s;
            foreach (string union in new[] { "anyOf", "oneOf" })
            {
                if (!s.TryGetProperty(union, out JsonElement alternatives) || alternatives.ValueKind != JsonValueKind.Array)
                    continue;
                foreach (JsonElement alternative in alternatives.EnumerateArray())
                {
                    JsonElement? found = ContainerSchema(alternative, root, depth + 1, walk, out isArray);
                    if (found != null)
                        return found;
                }
            }
            return null;
        }

        private static bool IsArraySchema(JsonElement s)
            => HasType(s, "array") || (!s.TryGetProperty("type", out _) && s.TryGetProperty("items", out _));

        private static bool HasType(JsonElement s, string name)
        {
            if (!s.TryGetProperty("type", out JsonElement type))
                return false;
            if (type.ValueKind == JsonValueKind.String)
                return type.GetString() == name;
            return type.ValueKind == JsonValueKind.Array
                && type.EnumerateArray().Any(t => t.ValueKind == JsonValueKind.String && t.GetString() == name);
        }

        /// <summary>
        /// Type a leaf's text by its schema, in vLLM's priority order (null, integer, number,
        /// boolean, object, array) with llama.cpp's spelling (JSON literals: <c>1</c>,
        /// <c>-14</c>, <c>3.14</c>, <c>true</c>). An empty element of a container type is the
        /// empty container (the template renders <c>[]</c> and <c>{}</c> that way). Text the
        /// schema cannot type stays the model's text.
        /// </summary>
        private static object? Coerce(string raw, JsonElement schema, JsonElement? root)
        {
            HashSet<string> types = Types(schema, root, 0);
            string text = raw.Trim();
            if (types.Contains("null") && text == "null")
                return null;
            if (types.Contains("integer")
                && long.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out long integer))
                return integer;
            if (types.Contains("number") && TryJson(text, out object? number) && number is long or double)
                return number;
            if (types.Contains("boolean"))
            {
                if (string.Equals(text, "true", StringComparison.OrdinalIgnoreCase)) return true;
                if (string.Equals(text, "false", StringComparison.OrdinalIgnoreCase)) return false;
            }
            bool isObject = types.Contains("object"), isArray = types.Contains("array");
            if ((isObject || isArray) && text.Length == 0)
                return isArray && !isObject ? new List<object?>() : new Dictionary<string, object?>();
            if (isObject && TryJson(text, out object? map) && map is Dictionary<string, object?>)
                return map;
            if (isArray && TryJson(text, out object? list) && list is List<object?>)
                return list;
            if (types.Count > 0 && !types.Contains("string"))
                return TryJson(text, out object? literal) ? literal : raw;
            return types.Count == 0 && TryJson(text, out object? any) ? any : raw;
        }

        private static bool TryJson(string text, out object? value)
        {
            value = null;
            if (text.Length == 0)
                return false;
            try
            {
                using JsonDocument document = JsonDocument.Parse(text);
                value = ChatMlOutputParser.JsonElementToObject(document.RootElement);
                return true;
            }
            catch (JsonException)
            {
                return false;
            }
        }
    }
}
