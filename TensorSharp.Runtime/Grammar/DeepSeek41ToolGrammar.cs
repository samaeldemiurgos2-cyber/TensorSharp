// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;

namespace TensorSharp.Runtime.Grammar;

public enum DeepSeek41ToolChoice { Auto, None, Required, Named }

/// <summary>Immutable grammar recipe; every generation gets an independent parser.</summary>
public sealed class DeepSeek41ToolGrammar : IToolCallGrammar
{
    public const string CallsOpen = "<｜DSML｜ calls>";
    private const string CallsClose = "</｜DSML｜ calls>";
    private const string ParameterClose = "</｜DSML｜ parameter>";
    // Reserve tag families as well as the canonical DSML delimiters. Otherwise
    // a mistyped closer such as </parameter_name> remains an unfinished raw
    // argument and can absorb the rest of the response. Literal values bearing
    // these prefixes still use the lossless JSON-string representation below.
    private static readonly string[] RawReserved = { ParameterClose, "<param", "</param", "<invoke", "</invoke", "<｜DSML｜ parameter", "<｜DSML｜ invoke", "</｜DSML｜ invoke>", CallsClose };
    private static readonly JsonSerializerOptions JsonOptions = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    public string Source { get; }
    public DeepSeek41ToolChoice Choice { get; }

    private DeepSeek41ToolGrammar(string source, DeepSeek41ToolChoice choice)
        => (Source, Choice) = (source, choice);

    public GrammarConstraint NewConstraint(ITokenizer tokenizer, bool thinking)
    {
        int[] allowed = Array.Empty<int>();
        if (Choice != DeepSeek41ToolChoice.None)
        {
            int dsml = tokenizer.LookupToken("｜DSML｜");
            if (dsml < 0)
                throw new NotSupportedException("DeepSeek V4.1 tool grammar requires the tokenizer's ｜DSML｜ marker.");
            allowed = new[] { dsml };
        }
        var constraint = GrammarLibrary.NewConstraint(GrammarLibrary.ForGbnf(Source, tokenizer, allowed), tokenizer);
        if (Choice == DeepSeek41ToolChoice.Auto && thinking) constraint.ActivateAfterTriggers("</think>", CallsOpen);
        else if (Choice == DeepSeek41ToolChoice.Auto) constraint.ActivateAfter(CallsOpen);
        else if (thinking) constraint.ActivateAfter("</think>");
        return constraint;
    }

    public static DeepSeek41ToolGrammar Compile(IReadOnlyList<ToolFunction> tools,
        DeepSeek41ToolChoice choice = DeepSeek41ToolChoice.Auto, string? namedTool = null,
        bool parallelToolCalls = true)
    {
        var compiler = new Compiler();
        if (choice == DeepSeek41ToolChoice.None)
        {
            compiler.ForbiddenText("answer", new[] { CallsOpen });
            compiler.Add("root", "answer");
            return new(compiler.Emit(), choice);
        }
        if (tools == null || tools.Count == 0) throw Unsupported("tool_choice requires at least one declared function");
        if (tools.Count > 64) throw Unsupported("at most 64 functions are supported by the V4.1 tool grammar");
        var names = new HashSet<string>(StringComparer.Ordinal);
        var invokeRules = new List<string>();
        foreach (var tool in tools)
        {
            ValidateName(tool.Name, "function");
            if (!names.Add(tool.Name)) throw Unsupported($"duplicate function name '{tool.Name}'");
            if (choice == DeepSeek41ToolChoice.Named && tool.Name != namedTool) continue;
            using var document = JsonDocument.Parse(ParameterSchema(tool));
            string parameters = compiler.Parameters(document.RootElement);
            string invoke = compiler.Next("invoke");
            compiler.Add(invoke, Literal("<｜DSML｜ invoke name=\"" + tool.Name + "\">") +
                " ws " + parameters + " ws " + Literal("</｜DSML｜ invoke>") + " ws");
            invokeRules.Add(invoke);
        }
        if (invokeRules.Count == 0) throw Unsupported($"tool_choice names undeclared function '{namedTool}'");
        compiler.Add("invoke", string.Join(" | ", invokeRules));
        compiler.Add("root", (choice == DeepSeek41ToolChoice.Auto ? "ws " : "ws " + Literal(CallsOpen) + " ws ") +
            "invoke" + (parallelToolCalls ? " invoke*" : "") + " " + Literal(CallsClose) + " ws");
        return new(compiler.Emit(), choice);
    }

    private static string ParameterSchema(ToolFunction tool)
    {
        if (tool.ParametersSchemaJson != null) return tool.ParametersSchemaJson;
        var properties = new Dictionary<string, object>();
        foreach (var (name, parameter) in tool.Parameters ?? new())
        {
            string type = string.IsNullOrEmpty(parameter.Type) ? "string" : parameter.Type;
            var schema = new Dictionary<string, object> { ["type"] = type };
            if (parameter.Enum is { Count: > 0 })
            {
                if (type == "string") schema["enum"] = parameter.Enum;
                else schema["enum"] = parameter.Enum.Select(v => JsonDocument.Parse(v).RootElement.Clone()).ToArray();
            }
            properties[name] = schema;
        }
        return JsonSerializer.Serialize(new { type = "object", properties, required = tool.Required ?? new() });
    }

    private static NotSupportedException Unsupported(string detail)
        => new("Unsupported DeepSeek V4.1 tool schema/policy: " + detail + ".");

    private static void ValidateName(string name, string kind)
    {
        if (string.IsNullOrEmpty(name) || name.Length > 128 || name.Any(c => char.IsControl(c) || c is '"' or '<' or '>'))
            throw Unsupported($"{kind} names must be nonempty and contain no quotes, angle brackets or control characters");
    }

    private static string Literal(string value) => JsonSerializer.Serialize(value, JsonOptions);

    internal static bool ContainsReservedMarkup(string value)
        => RawReserved.Any(marker => value.Contains(marker, StringComparison.Ordinal));

    private sealed class Compiler
    {
        private readonly Dictionary<string, string> _rules = new(StringComparer.Ordinal);
        private int _next;
        private bool _raw;
        public Compiler()
        {
            // Shared JSON primitives accept JSON syntax; stricter property types
            // below select the appropriate primitive or recursively built rule.
            foreach (string line in JsonSchemaGrammarCompiler.JsonValueGrammar.Split('\n'))
            {
                int separator = line.IndexOf("::=", StringComparison.Ordinal);
                if (separator < 0) continue;
                string name = line[..separator].Trim();
                if (name != "root") Add(name, line[(separator + 3)..].Trim());
            }
            // JSON permits arbitrary digit counts. Do not inherit the generic
            // JSON grammar's practical numeric-length caps for tool parameters.
            Add("number", "\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)? ws");
            Add("integer", "\"-\"? (\"0\" | [1-9] [0-9]*) ws");
            // The DSML parser locates delimiters before parsing embedded JSON.
            // Require JSON strings to spell '<' as \u003c so a string value can
            // contain any XML/DSML text without becoming envelope markup.
            Add("char", "[^<\"\\\\\\x7F\\x00-\\x1F] | \"\\\\\" ([\"\\\\bfnrt/] | \"u\" [0-9a-fA-F]{4})");
        }
        public string Next(string prefix) => prefix + "-" + _next++;
        public void Add(string name, string body) => _rules[name] = body;
        public string Emit() => string.Join("\n", _rules.Select(p => p.Key + " ::= " + p.Value)) + "\n";

        public string Parameters(JsonElement schema)
        {
            CheckKeywords(schema, new[] { "type", "properties", "required", "additionalProperties" });
            if (Type(schema, "object") != "object") throw Unsupported("function parameters must have type object");
            var properties = Properties(schema);
            var required = Required(schema, properties.Select(p => p.Name));
            CheckAdditionalProperties(schema);
            var result = new List<string>();
            foreach (var property in properties)
            {
                ValidateName(property.Name, "parameter");
                JsonElement value = property.Value;
                bool isString = Type(value) == "string";
                string parameter = Next("parameter");
                string close = " " + Literal(ParameterClose) + " ws";
                string json = Literal("<｜DSML｜ parameter name=\"" + property.Name + "\" string=\"false\">") +
                    " " + JsonValue(value, 0) + close;
                string? raw = isString ? RawString(value) : null;
                // string=false carries a JSON value, including a JSON string.
                // That existing protocol convention preserves strings containing
                // reserved delimiters through escaped '<', while ordinary strings
                // can retain the trained raw string=true representation.
                Add(parameter, raw == null ? json :
                    Literal("<｜DSML｜ parameter name=\"" + property.Name + "\" string=\"true\">") +
                    " " + raw + close + " | " + json);
                result.Add(parameter + (required.Contains(property.Name) ? "" : "?"));
            }
            return result.Count == 0 ? "\"\"" : string.Join(" ", result);
        }

        private string? RawString(JsonElement schema)
        {
            CheckKeywords(schema, new[] { "type", "enum", "const" });
            if (schema.TryGetProperty("enum", out var values)) return RawChoices(values);
            if (schema.TryGetProperty("const", out var constant)) return CanRenderRaw(constant) ? Literal(constant.GetString()!) : null;
            if (!_raw)
            {
                _raw = true;
                ForbiddenText("raw", RawReserved);
            }
            return "raw";
        }

        private static string? RawChoices(JsonElement values)
        {
            if (values.ValueKind != JsonValueKind.Array || values.GetArrayLength() == 0) throw Unsupported("enum must be a nonempty array");
            string[] literals = values.EnumerateArray().Where(CanRenderRaw).Select(v => Literal(v.GetString()!)).ToArray();
            return literals.Length == 0 ? null : "(" + string.Join(" | ", literals) + ")";
        }

        private static bool CanRenderRaw(JsonElement value)
        {
            if (value.ValueKind != JsonValueKind.String) throw Unsupported("string enum/const must contain strings");
            string text = value.GetString()!;
            return !ContainsReservedMarkup(text);
        }

        private string JsonValue(JsonElement schema, int depth)
        {
            if (depth > 16) throw Unsupported("schema nesting exceeds 16 levels");
            string type = Type(schema);
            string[] typeKeywords = type switch
            {
                "object" => new[] { "type", "properties", "required", "additionalProperties" },
                "array" => new[] { "type", "items" },
                _ => new[] { "type", "enum", "const" },
            };
            CheckKeywords(schema, typeKeywords);
            if (schema.TryGetProperty("enum", out var values))
            {
                if (values.ValueKind != JsonValueKind.Array || values.GetArrayLength() == 0) throw Unsupported("enum must be a nonempty array");
                return "(" + string.Join(" | ", values.EnumerateArray().Select(v => JsonLiteral(v, type))) + ") ws";
            }
            if (schema.TryGetProperty("const", out var constant)) return JsonLiteral(constant, type) + " ws";
            if (type is "any" or "string" or "integer" or "number" or "boolean" or "null")
                return type == "any" ? "value" : type;
            if (type == "array")
            {
                string item = schema.TryGetProperty("items", out var items) ? JsonValue(items, depth + 1) : "value";
                string rule = Next("items");
                Add(rule, "\"[\" ws (" + item + " (\",\" ws " + item + ")*)? \"]\" ws");
                return rule;
            }
            if (type == "object")
            {
                var properties = Properties(schema);
                var required = Required(schema, properties.Select(p => p.Name));
                CheckAdditionalProperties(schema);
                // A nested open map with no named properties can carry arbitrary
                // JSON members. Keep the separate top-level Parameters() path's
                // no-argument function convention unchanged. Generic JSON string
                // rules still escape '<' to protect the enclosing DSML envelope.
                if (properties.Count == 0 &&
                    (!schema.TryGetProperty("additionalProperties", out var extras) || extras.ValueKind == JsonValueKind.True))
                    return "object";
                // Two suffix states per property preserve arbitrary optional
                // omissions while commas appear only between emitted members.
                string prefix = Next("members");
                var valueRules = properties.Select(p => JsonValue(p.Value, depth + 1)).ToArray();
                for (int i = properties.Count; i >= 0; --i)
                    for (int hasPrevious = 0; hasPrevious <= 1; ++hasPrevious)
                    {
                        string name = prefix + "-" + i + "-" + hasPrevious;
                        if (i == properties.Count) { Add(name, "\"\""); continue; }
                        var property = properties[i];
                        string value = valueRules[i];
                        string present = (hasPrevious != 0 ? "\",\" ws " : "") +
                            Literal(JsonSerializer.Serialize(property.Name)) + " \":\" ws " + value + " " + prefix + "-" + (i + 1) + "-1";
                        Add(name, required.Contains(property.Name) ? present : present + " | " + prefix + "-" + (i + 1) + "-" + hasPrevious);
                    }
                string rule = Next("object");
                Add(rule, "\"{\" ws " + prefix + "-0-0 \"}\" ws");
                return rule;
            }
            throw Unsupported($"parameter type '{type}' is not supported");
        }

        private static string JsonLiteral(JsonElement value, string type)
        {
            if (type == "integer")
            {
                // The tool parser retains Int64 values exactly and otherwise
                // falls back to double. Never promise an integer constant that
                // would become a different value during that conversion.
                if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt64(out long integer))
                    throw Unsupported("integer enum/const values must be JSON integer literals in the Int64 range");
                return Literal(integer.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }
            bool valid = type switch
            {
                "any" => value.ValueKind is not (JsonValueKind.Object or JsonValueKind.Array),
                "string" => value.ValueKind == JsonValueKind.String,
                "number" => value.ValueKind == JsonValueKind.Number,
                "boolean" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
                "null" => value.ValueKind == JsonValueKind.Null,
                _ => false,
            };
            if (!valid) throw Unsupported("enum/const value does not match its primitive parameter type");
            return Literal(JsonSerializer.Serialize(value));
        }

        private static string Type(JsonElement schema, string fallback = "any")
        {
            if (schema.ValueKind != JsonValueKind.Object) throw Unsupported("schemas must be JSON objects");
            if (schema.TryGetProperty("type", out var type))
            {
                if (type.ValueKind != JsonValueKind.String) throw Unsupported("union parameter types are not supported");
                return type.GetString()!;
            }
            if (schema.TryGetProperty("properties", out _)) return "object";
            return fallback;
        }

        private static void CheckKeywords(JsonElement schema, IEnumerable<string> permitted)
        {
            if (schema.ValueKind != JsonValueKind.Object) throw Unsupported("schemas must be JSON objects");
            if (schema.TryGetProperty("enum", out _) && schema.TryGetProperty("const", out _))
                throw Unsupported("combined enum and const assertions are not supported");
            var allowed = new HashSet<string>(permitted, StringComparer.Ordinal);
            allowed.UnionWith(new[] { "description", "title", "default", "examples", "$comment" });
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in schema.EnumerateObject())
            {
                if (!seen.Add(property.Name)) throw Unsupported($"duplicate schema keyword '{property.Name}'");
                if (!allowed.Contains(property.Name)) throw Unsupported($"schema keyword '{property.Name}' cannot be enforced");
            }
        }

        private static List<JsonProperty> Properties(JsonElement schema)
        {
            if (!schema.TryGetProperty("properties", out var properties)) return new();
            if (properties.ValueKind != JsonValueKind.Object) throw Unsupported("properties must be an object");
            var result = properties.EnumerateObject().ToList();
            if (result.Count > 64 || result.Select(p => p.Name).Distinct(StringComparer.Ordinal).Count() != result.Count)
                throw Unsupported("properties must have at most 64 distinct names");
            return result;
        }

        private static HashSet<string> Required(JsonElement schema, IEnumerable<string> names)
        {
            var result = new HashSet<string>(StringComparer.Ordinal);
            if (!schema.TryGetProperty("required", out var required)) return result;
            if (required.ValueKind != JsonValueKind.Array) throw Unsupported("required must be an array");
            var available = names.ToHashSet(StringComparer.Ordinal);
            foreach (var value in required.EnumerateArray())
                if (value.ValueKind != JsonValueKind.String || !available.Contains(value.GetString()!) || !result.Add(value.GetString()!))
                    throw Unsupported("required must name distinct declared properties");
            return result;
        }

        private static void CheckAdditionalProperties(JsonElement schema)
        {
            if (schema.TryGetProperty("additionalProperties", out var extra) && extra.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                throw Unsupported("additionalProperties schemas are not supported");
        }

        // A small DFA excludes reserved delimiters while retaining literal
        // newlines, quotes, Unicode, and unrelated XML in raw string values.
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
                    alternatives.Add(Literal(c.ToString()) + " " + prefix + "-" + indexes[suffix]);
                }
                Add(prefix + "-" + i, string.Join(" | ", alternatives));
            }
            Add(prefix, prefix + "-0");
        }
    }
}
