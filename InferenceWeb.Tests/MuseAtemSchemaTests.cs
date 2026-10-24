// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class MuseAtemSchemaTests
{
    [Theory]
    [InlineData("muse-spark", "MuseSpark.unsloth_chat_template.jinja", 1)]
    [InlineData("muse-spark", "MuseSpark.unsloth_chat_template.jinja", 10000)]
    [InlineData("muse-glimmer", "MuseGlimmer.chat_template.jinja", 1)]
    [InlineData("muse-glimmer", "MuseGlimmer.chat_template.jinja", 10000)]
    public void ReleasedTemplateToolArgumentsRoundTripWithoutChangingStringTypes(
        string architecture, string fixture, int chunkSize)
    {
        var tools = ToolFunction.ParseList("""
            [{"type":"function","function":{"name":"echo","parameters":{
              "type":"object","properties":{
                "null_text":{"type":"string"},"bool_text":{"type":"string"},
                "number_text":{"type":"string"},"quoted_text":{"type":"string"},
                "json_text":{"type":"string"},"spaced_text":{"type":"string"},
                "nullable_text":{"type":["string","null"]},
                "null_value":{"type":"null"},"number_value":{"type":"integer"},
                "bool_value":{"type":"boolean"}
              }}}}]
            """);
        var arguments = new Dictionary<string, object?>
        {
            ["null_text"] = "null", ["bool_text"] = "true", ["number_text"] = "123",
            ["quoted_text"] = "\"Paris\"", ["json_text"] = "{\"n\":1}",
            ["spaced_text"] = "  keep whitespace  ", ["nullable_text"] = "null",
            ["null_value"] = null, ["number_value"] = 123L, ["bool_value"] = true,
        };
        using var stream = typeof(MuseAtemSchemaTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures." + fixture)!;
        using var reader = new StreamReader(stream);
        string rendered = ChatTemplate.RenderFromGgufTemplate(reader.ReadToEnd(),
            [new() { Role = "user", Content = "Echo these values." },
             new() { Role = "assistant", ToolCalls = [new() { Name = "echo", Arguments = arguments }] }],
            addGenerationPrompt: false, architecture: architecture, tools: tools);
        const string marker = "<|start|>assistant to=echo<|message|>";
        int start = rendered.LastIndexOf(marker, StringComparison.Ordinal);
        Assert.True(start >= 0, rendered);
        var parser = OutputParserFactory.Create(architecture);
        parser.Init(false, tools);
        var (content, call) = ParseCall(parser, rendered[start..], chunkSize);
        Assert.Equal("", content);
        Assert.Equal("echo", call.Name);
        Assert.Equal(arguments.Count, call.Arguments.Count);
        foreach (var (key, expected) in arguments)
        {
            Assert.Equal(expected, call.Arguments[key]);
            if (expected is string) Assert.IsType<string>(call.Arguments[key]);
        }
    }

    [Theory]
    [InlineData("muse-spark")]
    [InlineData("muse-glimmer")]
    public void ReinitializingClearsPriorSchemasAndUnknownArgumentsKeepBestEffortDecoding(string architecture)
    {
        const string generated = " to=echo<|message|><atem:function_calls><atem:invoke name=\"echo\">"
            + "<atem:parameter name=\"value\">123</atem:parameter>"
            + "</atem:invoke></atem:function_calls><|eot|>";
        var parser = OutputParserFactory.Create(architecture);
        parser.Init(false, [new() { Name = "echo", Parameters = new() { ["value"] = new() { Type = "string" } } }]);
        Assert.Equal("123", ParseCall(parser, generated, 1).call.Arguments["value"]);
        parser.Init(false, null);
        Assert.Equal(123L, ParseCall(parser, generated, 1).call.Arguments["value"]);
    }

    private static (string content, ToolCall call) ParseCall(IOutputParser parser, string generated, int chunkSize)
    {
        var content = new StringBuilder();
        var calls = new List<ToolCall>();
        for (int offset = 0; offset < generated.Length; offset += chunkSize)
        {
            int count = Math.Min(chunkSize, generated.Length - offset);
            var parsed = parser.Add(generated.Substring(offset, count), offset + count == generated.Length);
            content.Append(parsed.Content);
            if (parsed.ToolCalls != null) calls.AddRange(parsed.ToolCalls);
        }
        return (content.ToString(), Assert.Single(calls));
    }
}
