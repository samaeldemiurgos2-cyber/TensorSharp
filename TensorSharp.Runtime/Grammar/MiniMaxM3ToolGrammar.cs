// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace TensorSharp.Runtime.Grammar;

public enum MiniMaxM3ToolChoice { Auto, None, Required, Named }

/// <summary>
/// The grammar llama.cpp's <c>peg-minimax-m3</c> format builds for MiniMax-M3 tool calls,
/// as an immutable recipe; every generation gets its own parser.
/// <code>
/// ]&lt;]minimax[&gt;[&lt;tool_call&gt; ws INVOKE (ws INVOKE)* ws ]&lt;]minimax[&gt;[&lt;/tool_call&gt;
/// INVOKE  := ]&lt;]minimax[&gt;[&lt;invoke name="f"&gt; MEMBERS ws ]&lt;]minimax[&gt;[&lt;/invoke&gt;
/// MEMBERS := required properties in schema order, then optional ones in any order
/// ELEMENT := ]&lt;]minimax[&gt;[&lt;k&gt; VALUE ]&lt;]minimax[&gt;[&lt;/k&gt;
/// VALUE   := raw text (a schema that may be a string) | MEMBERS (object) |
///            (ws ]&lt;]minimax[&gt;[&lt;item&gt; VALUE ]&lt;]minimax[&gt;[&lt;/item&gt;)* (array) |
///            a choice of the alternatives (anyOf/oneOf) | a JSON literal (anything else)
/// </code>
/// It arms after the reply's <c>&lt;/mm:think&gt;</c> - the end of a reasoning block or the
/// bare prefix of a reply without one, in either thinking mode - and, for
/// <c>tool_choice=auto</c>, only once the model opens a block itself (llama.cpp's lazy
/// trigger). <c>required</c> or a named function must open the block straight after it.
/// </summary>
public sealed class MiniMaxM3ToolGrammar : IToolCallGrammar
{
    private const string Ns = ChatTemplate.MiniMaxM3Namespace;
    public const string CallsOpen = Ns + "<tool_call>";
    private const string CallsClose = Ns + "</tool_call>";
    private static readonly JsonSerializerOptions JsonOptions = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    public string Source { get; }
    public MiniMaxM3ToolChoice Choice { get; }

    private MiniMaxM3ToolGrammar(string source, MiniMaxM3ToolChoice choice) => (Source, Choice) = (source, choice);

    /// <summary>The reply ends its reasoning with `&lt;/mm:think&gt;` in both thinking modes.</summary>
    public GrammarConstraint NewConstraint(ITokenizer tokenizer, bool thinking) => NewConstraint(tokenizer);

    public GrammarConstraint NewConstraint(ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        // The namespace marker and the block tags are USER_DEFINED tokens, which the grammar
        // trie leaves out unless they are allowed by name.
        var allowed = new List<int>();
        if (Choice != MiniMaxM3ToolChoice.None)
        {
            foreach (string marker in new[] { Ns, "<tool_call>", "</tool_call>" })
            {
                int id = tokenizer.LookupToken(marker);
                if (id >= 0) allowed.Add(id);
                else if (marker == Ns)
                    throw new NotSupportedException("MiniMax-M3 tool grammar requires the tokenizer's ]<]minimax[>[ marker.");
            }
        }
        var constraint = GrammarLibrary.NewConstraint(GrammarLibrary.ForGbnf(Source, tokenizer, allowed), tokenizer);
        string reasoningEnd = ChatTemplate.MiniMaxM3ThinkClose;
        if (Choice == MiniMaxM3ToolChoice.Auto) constraint.ActivateAfterTriggers(reasoningEnd, CallsOpen);
        else constraint.ActivateAfter(reasoningEnd);
        return constraint;
    }

    public static MiniMaxM3ToolGrammar Compile(IReadOnlyList<ToolFunction> tools,
        MiniMaxM3ToolChoice choice = MiniMaxM3ToolChoice.Auto, string? namedTool = null, bool parallelToolCalls = true)
    {
        var rules = new Rules();
        if (choice == MiniMaxM3ToolChoice.None)
        {
            rules.ForbiddenText("answer", new[] { CallsOpen });
            rules.Add("root", "answer");
            return new(rules.Emit(), choice);
        }
        if (tools == null || tools.Count == 0) throw Unsupported("tool_choice requires at least one declared function");
        if (tools.Count > 64) throw Unsupported("at most 64 functions are supported by the MiniMax-M3 tool grammar");
        var names = new HashSet<string>(StringComparer.Ordinal);
        var invokes = new List<string>();
        foreach (ToolFunction tool in tools)
        {
            ValidateName(tool.Name, "function");
            if (!names.Add(tool.Name)) throw Unsupported($"duplicate function name '{tool.Name}'");
            if (choice == MiniMaxM3ToolChoice.Named && tool.Name != namedTool) continue;
            using JsonDocument document = JsonDocument.Parse(ParameterSchema(tool));
            JsonElement root = document.RootElement;
            string members = rules.Members(root, root, 0);
            string invoke = rules.Next("invoke");
            rules.Add(invoke, Literal(Ns + "<invoke name=\"" + tool.Name + "\">") + " " + members + " ws " +
                Literal(Ns + "</invoke>"));
            invokes.Add(invoke);
        }
        if (invokes.Count == 0) throw Unsupported($"tool_choice names undeclared function '{namedTool}'");
        rules.Add("invoke", string.Join(" | ", invokes));
        string calls = "ws invoke" + (parallelToolCalls ? " (ws invoke)*" : "") + " ws " + Literal(CallsClose) + " ws";
        // Auto arms after the block's opener; required/named must write it first.
        rules.Add("root", choice == MiniMaxM3ToolChoice.Auto ? calls : "ws " + Literal(CallsOpen) + " " + calls);
        return new(rules.Emit(), choice);
    }

    private static string ParameterSchema(ToolFunction tool)
    {
        if (tool.ParametersSchemaJson != null) return tool.ParametersSchemaJson;
        var properties = new Dictionary<string, object>();
        foreach (var (name, parameter) in tool.Parameters ?? new())
        {
            var schema = new Dictionary<string, object> { ["type"] = string.IsNullOrEmpty(parameter.Type) ? "string" : parameter.Type };
            if (parameter.Enum is { Count: > 0 } && (string)schema["type"] == "string") schema["enum"] = parameter.Enum;
            properties[name] = schema;
        }
        return JsonSerializer.Serialize(new { type = "object", properties, required = tool.Required ?? new() });
    }

    private static NotSupportedException Unsupported(string detail)
        => new("Unsupported MiniMax-M3 tool schema/policy: " + detail + ".");

    private static void ValidateName(string name, string kind)
    {
        if (string.IsNullOrEmpty(name) || name.Length > 128 || name.Any(c => char.IsControl(c) || c is '"' or '<' or '>'))
            throw Unsupported($"{kind} names must be nonempty and contain no quotes, angle brackets or control characters");
    }

    private static string Literal(string value) => JsonSerializer.Serialize(value, JsonOptions);

    private sealed class Rules
    {
        private readonly Dictionary<string, string> _rules = new(StringComparer.Ordinal);
        private int _next;
        private readonly Dictionary<string, string> _rawRules = new(StringComparer.Ordinal);

        public Rules()
        {
            foreach (string line in JsonSchemaGrammarCompiler.JsonValueGrammar.Split('\n'))
            {
                int separator = line.IndexOf("::=", StringComparison.Ordinal);
                if (separator < 0) continue;
                string name = line[..separator].Trim();
                if (name != "root") Add(name, line[(separator + 3)..].Trim());
            }
            Add("number", "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?");
            Add("integer", "\"-\"? (\"0\" | [1-9] [0-9]*)");
            Add("boolean", "(\"true\" | \"false\")");
            Add("null", "\"null\"");
        }

        public string Next(string prefix) => prefix + "-" + _next++;
        public void Add(string name, string body) => _rules[name] = body;
        public string Emit() => string.Join("\n", _rules.Select(p => p.Key + " ::= " + p.Value)) + "\n";

        /// <summary>An object's elements: required properties in schema order, then any
        /// number of optional ones in any order (llama.cpp's members_of).</summary>
        public string Members(JsonElement schema, JsonElement root, int depth)
        {
            if (depth > 16) throw Unsupported("schema nesting exceeds 16 levels");
            JsonElement s = Resolve(schema, root, 0);
            if (!s.TryGetProperty("properties", out JsonElement properties) || properties.ValueKind != JsonValueKind.Object)
                return "\"\"";
            var required = new HashSet<string>(StringComparer.Ordinal);
            if (s.TryGetProperty("required", out JsonElement list) && list.ValueKind == JsonValueKind.Array)
                foreach (JsonElement name in list.EnumerateArray())
                    if (name.ValueKind == JsonValueKind.String) required.Add(name.GetString()!);
            var mandatory = new List<string>();
            var optional = new List<string>();
            foreach (JsonProperty property in properties.EnumerateObject())
            {
                ValidateName(property.Name, "parameter");
                string element = Next("element");
                string close = Ns + "</" + property.Name + ">";
                Add(element, Literal(Ns + "<" + property.Name + ">") + " " + Value(property.Value, root, depth + 1, close) + " " + Literal(close));
                (required.Contains(property.Name) ? mandatory : optional).Add(element);
            }
            string body = string.Join(" ", mandatory.Select(e => "ws " + e));
            if (optional.Count > 0)
                body += (body.Length > 0 ? " " : "") + "(ws (" + string.Join(" | ", optional) + "))*";
            if (body.Length == 0) return "\"\"";
            string rule = Next("members");
            Add(rule, body);
            return rule;
        }

        /// <param name="close">The close tag that ends this value.</param>
        private string Value(JsonElement schema, JsonElement root, int depth, string close)
        {
            if (depth > 16) throw Unsupported("schema nesting exceeds 16 levels");
            JsonElement s = Resolve(schema, root, 0);
            if (s.ValueKind != JsonValueKind.Object) throw Unsupported("schemas must be JSON objects");
            if (MiniMaxM3OutputParser.MayBeString(s, root, 0))
                return StringValue(s, close);
            foreach (string union in new[] { "anyOf", "oneOf" })
            {
                if (!s.TryGetProperty(union, out JsonElement alternatives) || alternatives.ValueKind != JsonValueKind.Array)
                    continue;
                string rule = Next("union");
                Add(rule, string.Join(" | ", alternatives.EnumerateArray().Select(a => "(" + Value(a, root, depth + 1, close) + ")")));
                return rule;
            }
            if (s.TryGetProperty("enum", out JsonElement values) && values.ValueKind == JsonValueKind.Array && values.GetArrayLength() > 0)
                return "(" + string.Join(" | ", values.EnumerateArray().Select(v => Literal(v.GetRawText()))) + ")";
            if (s.TryGetProperty("const", out JsonElement constant))
                return Literal(constant.GetRawText());
            string type = Type(s);
            if (type.Contains('|'))
                return "ws (" + type.Replace("|", " | ", StringComparison.Ordinal) + ") ws";
            switch (type)
            {
                case "integer":
                case "number":
                case "boolean":
                case "null":
                    return "ws " + type + " ws";
                case "object":
                    if (s.TryGetProperty("properties", out JsonElement p) && p.ValueKind == JsonValueKind.Object && p.EnumerateObject().Any())
                        return Members(s, root, depth) + " ws";
                    return "ws object";
                case "array":
                {
                    string item = s.TryGetProperty("items", out JsonElement items) && items.ValueKind == JsonValueKind.Object
                        ? Value(items, root, depth + 1, Ns + "</item>") : "ws value";
                    string rule = Next("items");
                    Add(rule, "(ws " + Literal(Ns + "<item>") + " " + item + " " + Literal(Ns + "</item>") + ")* ws");
                    return rule;
                }
                default:
                    return "ws value";
            }
        }

        private string StringValue(JsonElement s, string close)
        {
            // A pure string enum or const keeps its listed spellings; anything else that may
            // be a string is raw text up to its own close tag (llama.cpp's until(close), which
            // the parser mirrors), so it may hold the marker and markup of other elements.
            if (s.TryGetProperty("enum", out JsonElement values) && values.ValueKind == JsonValueKind.Array
                && values.GetArrayLength() > 0 && values.EnumerateArray().All(v => v.ValueKind == JsonValueKind.String
                    && !v.GetString()!.Contains(Ns, StringComparison.Ordinal)))
                return "(" + string.Join(" | ", values.EnumerateArray().Select(v => Literal(v.GetString()!))) + ")";
            if (s.TryGetProperty("const", out JsonElement constant) && constant.ValueKind == JsonValueKind.String
                && !constant.GetString()!.Contains(Ns, StringComparison.Ordinal))
                return Literal(constant.GetString()!);
            if (!_rawRules.TryGetValue(close, out string? raw))
            {
                raw = "raw-" + _rawRules.Count;
                _rawRules[close] = raw;
                ForbiddenText(raw, new[] { close });
            }
            return raw;
        }

        private static JsonElement Resolve(JsonElement schema, JsonElement root, int depth)
        {
            if (schema.ValueKind != JsonValueKind.Object || !schema.TryGetProperty("$ref", out JsonElement reference))
                return schema;
            if (depth > 16) throw Unsupported("recursive $ref schemas are not supported");
            string path = reference.ValueKind == JsonValueKind.String ? reference.GetString()! : "";
            if (!path.StartsWith("#/", StringComparison.Ordinal)) throw Unsupported($"$ref '{path}' is not a local reference");
            JsonElement target = root;
            foreach (string segment in path[2..].Split('/'))
            {
                string key = segment.Replace("~1", "/", StringComparison.Ordinal).Replace("~0", "~", StringComparison.Ordinal);
                if (target.ValueKind != JsonValueKind.Object || !target.TryGetProperty(key, out target))
                    throw Unsupported($"$ref '{path}' does not resolve");
            }
            return Resolve(target, root, depth + 1);
        }

        private static string Type(JsonElement s)
        {
            if (s.TryGetProperty("type", out JsonElement type))
            {
                if (type.ValueKind == JsonValueKind.String) return type.GetString()!;
                // A union of literal types, such as the nullable ["integer", "null"], is a
                // choice of those literals ("integer|null"); one with a container is any JSON.
                if (type.ValueKind == JsonValueKind.Array)
                {
                    string[] names = type.EnumerateArray().Where(t => t.ValueKind == JsonValueKind.String)
                        .Select(t => t.GetString()!).Distinct().ToArray();
                    if (names.Length == 1) return names[0];
                    return names.Length > 0 && names.All(n => n is "integer" or "number" or "boolean" or "null")
                        ? string.Join("|", names) : "any";
                }
            }
            if (s.TryGetProperty("properties", out _)) return "object";
            if (s.TryGetProperty("items", out _)) return "array";
            return "any";
        }

        // A small DFA excludes the reserved marker while keeping literal newlines, quotes,
        // Unicode and unrelated markup in raw values (as DeepSeek41ToolGrammar does).
        public void ForbiddenText(string prefix, IReadOnlyList<string> forbidden)
        {
            var states = new HashSet<string>(StringComparer.Ordinal) { "" };
            foreach (string word in forbidden)
                for (int i = 1; i < word.Length; ++i) states.Add(word[..i]);
            string[] ordered = states.OrderBy(s => s.Length).ThenBy(s => s, StringComparer.Ordinal).ToArray();
            var indexes = ordered.Select((s, i) => (s, i)).ToDictionary(x => x.s, x => x.i, StringComparer.Ordinal);
            char[] alphabet = forbidden.SelectMany(s => s).Distinct().OrderBy(c => c).ToArray();
            string excluded = new(alphabet.SelectMany(c => c is '\\' or ']' or '[' or '^' or '-' ? new[] { '\\', c } : new[] { c }).ToArray());
            for (int i = 0; i < ordered.Length; ++i)
            {
                var alternatives = new List<string> { "\"\"", "[^" + excluded + "] " + prefix + "-0" };
                foreach (char c in alphabet)
                {
                    string next = ordered[i] + c;
                    if (forbidden.Any(s => next.EndsWith(s, StringComparison.Ordinal))) continue;
                    string suffix = ordered.Where(s => next.EndsWith(s, StringComparison.Ordinal)).OrderByDescending(s => s.Length).First();
                    alternatives.Add(Literal(c.ToString(CultureInfo.InvariantCulture)) + " " + prefix + "-" + indexes[suffix]);
                }
                Add(prefix + "-" + i, string.Join(" | ", alternatives));
            }
            Add(prefix, prefix + "-0");
        }
    }
}
