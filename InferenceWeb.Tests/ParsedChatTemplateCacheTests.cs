namespace InferenceWeb.Tests;

public class ParsedChatTemplateCacheTests
{
    private const string Template =
        "{% set ns = namespace(count=0) %}"
        + "{% macro emit(content) %}[{{ content }}]{% endmacro %}"
        + "{% for message in messages %}{% set ns.count = ns.count + 1 %}{{ emit(message.content) }}{% endfor %}"
        + "{{ ns.count }}:{{ enable_thinking }}:"
        + "{% for tool in tools %}{{ tool.function.name }}={{ tool.function.description }};{% endfor %}";

    [Fact]
    public void SharedAstDoesNotShareNamespacesMessagesToolsOrThinkingAcrossRequests()
    {
        Parallel.For(0, 64, i =>
        {
            var messages = new List<ChatMessage> { new() { Role = "user", Content = $"request-{i}" } };
            if (i % 3 == 0) messages.Insert(0, new() { Role = "system", Content = "rules" });
            var tools = new List<ToolFunction> { new() { Name = $"tool_{i}", Description = $"description-{i}" } };
            bool thinking = i % 2 == 0;
            string expected = (i % 3 == 0 ? "[rules]" : "")
                + $"[request-{i}]{messages.Count}:{thinking}:tool_{i}=description-{i};";
            for (int attempt = 0; attempt < 4; attempt++)
                Assert.Equal(expected, ChatTemplate.RenderFromGgufTemplate(Template, messages,
                    architecture: "parsed-cache-test", tools: tools, enableThinking: thinking));
        });
    }

    [Fact]
    public void ReusedAstStillReadsMutatedMessagesToolsAndChangedTemplate()
    {
        var message = new ChatMessage { Role = "user", Content = "before" };
        var tool = new ToolFunction { Name = "inspect", Description = "old" };
        List<ChatMessage> messages = [message];
        List<ToolFunction> tools = [tool];
        string Render(string template, bool thinking) => ChatTemplate.RenderFromGgufTemplate(
            template, messages, architecture: "parsed-cache-test", tools: tools, enableThinking: thinking);
        Assert.Equal("[before]1:False:inspect=old;", Render(Template, false));
        message.Content = "after";
        tool.Description = "new";
        Assert.Equal("[after]1:True:inspect=new;", Render(Template, true));
        Assert.Equal("[after]1:True:inspect=new;changed", Render(Template + "changed", true));
        Assert.Equal("[after]1:False:inspect=new;", Render(Template, false));
    }
}
