using System.Numerics;
using System.Text.Json;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

public class Qwen35ToolHistoryTemplateTests
{
    // The Qwen 3.5/3.8 XML argument branch: strings are emitted verbatim,
    // while other JSON values use tojson. Keep that distinction explicit.
    private const string Template = """
        {%- for message in messages -%}
          {%- if message.tool_calls -%}
            {%- for call in message.tool_calls -%}
              {{- '<tool_call>\n<function=' + call.function.name + '>' -}}
              {%- for key, value in call.function.arguments.items() -%}
                {{- '\n<parameter=' + key + '>\n' -}}
                {%- if value is string -%}{{- value -}}{%- else -%}{{- value | tojson -}}{%- endif -%}
                {{- '\n</parameter>' -}}
              {%- endfor -%}
              {{- '\n</function>\n</tool_call>' -}}
            {%- endfor -%}
          {%- endif -%}
        {%- endfor -%}
        """;

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OpenAiHistoryRendersCanonicalXmlWithoutAddingStringQuotes(bool argumentObject)
    {
        const string arguments = """
            {"code":"CARD-0042","color":"green","quoted":"\"literal\"","path":"C:\\tmp\\file","payload":{"text":"a\"b\\c","values":[true,null,3,9007199254740993,18446744073709551615,1234567890123456789012345678901234567890]},"wide":9007199254740993,"negative":-9223372036854775808,"fraction":1.25,"tiny":1e-100,"precise":1.1234567890123456789012345678901234,"huge":1e400}
            """;
        string wireArguments = argumentObject ? arguments : JsonSerializer.Serialize(arguments);
        string request = "[{\"role\":\"assistant\",\"tool_calls\":[{\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"inspect\",\"arguments\":"
            + wireArguments + "}}]},{\"role\":\"tool\",\"tool_call_id\":\"call-1\",\"content\":\"ok\"}]";
        List<ChatMessage> messages;
        using (var document = JsonDocument.Parse(request))
            messages = ChatMessageParser.ParseOpenAI(document.RootElement, null);

        var original = messages[0].ToolCalls![0].Arguments;
        string originalJson = JsonSerializer.Serialize(original);
        Assert.IsType<JsonElement>(original["code"]); // Exercise the actual HTTP parser output.
        string rendered = Render(messages);
        Assert.Contains(Parameter("code", "CARD-0042"), rendered);
        Assert.Contains(Parameter("color", "green"), rendered);
        Assert.Contains(Parameter("quoted", "\"literal\""), rendered);
        Assert.Contains(Parameter("path", "C:\\tmp\\file"), rendered);
        Assert.Contains(Parameter("wide", "9007199254740993"), rendered);
        Assert.Contains(Parameter("negative", "-9223372036854775808"), rendered);
        Assert.Contains(Parameter("fraction", "1.25"), rendered);
        Assert.Contains(Parameter("tiny", "1e-100"), rendered);
        Assert.Contains(Parameter("precise", "1.1234567890123456789012345678901234"), rendered);
        Assert.Contains(Parameter("huge", "1e400"), rendered);
        string payload = rendered.Split("<parameter=payload>\n")[1].Split("\n</parameter>")[0];
        using var expected = JsonDocument.Parse(arguments);
        using var observed = JsonDocument.Parse(payload);
        Assert.True(JsonElement.DeepEquals(expected.RootElement.GetProperty("payload"), observed.RootElement));
        Assert.Contains("\"values\": [true, null, 3, 9007199254740993, 18446744073709551615, 1234567890123456789012345678901234567890]", payload);
        Assert.Same(original, messages[0].ToolCalls![0].Arguments);
        Assert.Equal(originalJson, JsonSerializer.Serialize(original));
        Assert.Equal(rendered, Render(messages));
    }

    [Fact]
    public void NormalizedContainersExposeJinjaTypesWithoutMutatingSavedArguments()
    {
        var nested = new Dictionary<string, object?> { ["text"] = JsonSerializer.SerializeToElement("before") };
        var values = new List<object?> { JsonSerializer.SerializeToElement(new { text = "nested" }) };
        var arguments = new Dictionary<string, object?> { ["object"] = nested, ["array"] = values,
            ["boolean"] = JsonSerializer.SerializeToElement(true), ["empty"] = JsonSerializer.SerializeToElement<object?>(null) };
        List<ChatMessage> messages = [new() { Role = "assistant", ToolCalls = [new() { Name = "inspect", Arguments = arguments }] }];
        const string types = "{% set a = messages[0].tool_calls[0].function.arguments %}"
            + "{{ a.object is mapping }}:{{ a.array is sequence }}:{{ a.array[0].text is string }}:{{ a.boolean is boolean }}:{{ a.empty is none }}"
            + "{% set child = a.object %}{% set child.text = 'changed' %}:{{ child.text }}";
        Assert.Equal("True:True:True:True:True:changed", ChatTemplate.RenderFromGgufTemplate(types, messages, architecture: "history-normalization-test"));
        Assert.Equal("before", ((JsonElement)nested["text"]!).GetString());
        Assert.IsType<JsonElement>(Assert.Single(values));
    }

    [Fact]
    public void ClrIntegersAndDecimalsStayJsonNumbersAndLiteralStringsStayExact()
    {
        var arguments = new Dictionary<string, object?> { ["wide"] = long.MaxValue, ["unsigned"] = ulong.MaxValue,
            ["arbitrary"] = BigInteger.Parse("1234567890123456789012345678901234567890"),
            ["decimal"] = 12.125m, ["literal"] = "\"\\d+\"" };
        List<ChatMessage> messages = [new() { Role = "assistant", ToolCalls = [new() { Name = "inspect", Arguments = arguments }] }];
        string rendered = Render(messages);
        Assert.Contains(Parameter("wide", "9223372036854775807"), rendered);
        Assert.Contains(Parameter("unsigned", "18446744073709551615"), rendered);
        Assert.Contains(Parameter("arbitrary", "1234567890123456789012345678901234567890"), rendered);
        Assert.Contains(Parameter("decimal", "12.125"), rendered);
        Assert.Contains(Parameter("literal", "\"\\d+\""), rendered);
    }

    [Theory]
    [InlineData("\"CARD-0042\"")]
    [InlineData("\"\\d+\"")]
    [InlineData("C:\\tmp\\file")]
    public void OutputParserKeepsDeliberateQuotesAndBackslashesInDeclaredStrings(string literal)
    {
        var parser = OutputParserFactory.Create("qwen35");
        parser.Init(enableThinking: false, tools: [new() { Name = "inspect",
            Parameters = new() { ["code"] = new() { Type = "string" } } }]);
        string text = "<tool_call>\n<function=inspect>\n" + Parameter("code", literal) + "\n</function>\n</tool_call>";
        var first = parser.Add(text, false);
        var final = parser.Add("", true);
        var call = Assert.Single(final.ToolCalls ?? first.ToolCalls!);
        Assert.Equal(literal, call.Arguments["code"]);
    }

    private static string Render(List<ChatMessage> messages) => ChatTemplate.RenderFromGgufTemplate(
        Template, messages, architecture: "qwen35", enableThinking: false);
    private static string Parameter(string name, string value) => $"<parameter={name}>\n{value}\n</parameter>";
}
