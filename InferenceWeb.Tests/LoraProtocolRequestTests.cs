// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

/// <summary>
/// The <c>lora</c> request field on <c>/v1/chat/completions</c> and <c>/v1/responses</c>: a
/// request that cannot run with its adapter is a 400 before any streamed output (never a 200
/// stream that carries base-model text or an error event), and an accepted adapter reaches
/// that request's generation and no other request's.
/// </summary>
public sealed class LoraProtocolRequestTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("ts-lora-protocol-").FullName;
    private readonly string _gguf;
    private readonly string _adapter;

    public LoraProtocolRequestTests()
    {
        _gguf = Path.Combine(_root, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(_gguf);
        _adapter = WriteOnesAdapter(Path.Combine(_root, "canary.safetensors"), LoraProbeModel.Target);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    [Theory]
    [InlineData("chat", true)]
    [InlineData("chat", false)]
    [InlineData("responses", true)]
    [InlineData("responses", false)]
    public async Task UnknownAdapter_Is400BeforeAnyStreamedOutput(string surface, bool stream)
    {
        using var service = Service(supportsLora: true);

        var (context, body) = await Invoke(service, surface, stream, "\"lora\":\"missing\"");

        var error = AssertRefusedBeforeStreaming(context, body, "No LoRA adapter named 'missing'");
        Assert.Equal("lora", error.GetProperty("param").GetString());
        Assert.Equal("lora_not_found", error.GetProperty("code").GetString());
        Assert.False(error.TryGetProperty("unsupported_modules", out _));
    }

    [Theory]
    [InlineData("chat")]
    [InlineData("responses")]
    public async Task AdapterTheModelCannotApply_Is400NamingTheReason_BeforeStreaming(string surface)
    {
        using var service = Service(supportsLora: false);
        // Registered behind the model's back (as an adapter loaded before a model switch is):
        // the request still has to be refused, not answered by the base model.
        service.LoraRegistry.Load("canary", _adapter);

        var (context, body) = await Invoke(service, surface, stream: true, "\"lora\":\"canary\"");

        var error = AssertRefusedBeforeStreaming(context, body, "nemotron_h, mistral3 and hunyuan-dense");
        Assert.Equal("lora", error.GetProperty("param").GetString());
        Assert.Equal("lora_not_applicable", error.GetProperty("code").GetString());
        Assert.Equal(new[] { "blk.{0}.attn_output.weight" },
            error.GetProperty("unsupported_modules").EnumerateArray().Select(m => m.GetString()).ToArray());
    }

    /// <summary>
    /// A model that takes adapters, and an adapter with one module it cannot apply (the
    /// embedding, which a tied LM head would otherwise pick up): the request is refused before
    /// any streamed byte, naming the module, never answered by the base model or in part.
    /// </summary>
    [Theory]
    [InlineData("chat")]
    [InlineData("responses")]
    public async Task AdapterWithAModuleTheModelCannotApply_Is400NamingIt_BeforeStreaming(string surface)
    {
        using var service = Service(supportsLora: true);
        string partial = WriteOnesAdapter(Path.Combine(_root, "partial.safetensors"), "token_embd.weight");
        service.LoraRegistry.Load("partial", partial);

        var (context, body) = await Invoke(service, surface, stream: true, "\"lora\":\"partial\"");

        var error = AssertRefusedBeforeStreaming(context, body, "refused rather than applied in part");
        Assert.Equal("lora_not_applicable", error.GetProperty("code").GetString());
        Assert.Equal(new[] { "token_embd.weight" },
            error.GetProperty("unsupported_modules").EnumerateArray().Select(m => m.GetString()).ToArray());
    }

    [Fact]
    public void ModelsListing_SaysWhetherEachAdapterApplies_AndWhyNot()
    {
        using var service = Service(supportsLora: true);
        service.LoadLora("canary", _adapter);
        service.LoraRegistry.Load("partial", WriteOnesAdapter(Path.Combine(_root, "partial.safetensors"), "token_embd.weight"));
        var options = ServerOptionsBuilder.Build(new[] { "--model", _gguf, "--backend", "cpu", "--no-skills" }, _root);
        var adapter = new OpenAIChatAdapter(service, options, new UploadStoragePolicy(Path.Combine(_root, "uploads")),
            new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }), null,
            new SessionWorkspaceManager(Path.Combine(_root, "workspaces")), NullLoggerFactory.Instance);

        string json = JsonSerializer.Serialize(((IValueHttpResult)adapter.ListModels()).Value);

        using var doc = JsonDocument.Parse(json);
        var loras = doc.RootElement.GetProperty("data")[0].GetProperty("loras").EnumerateArray()
            .ToDictionary(l => l.GetProperty("id").GetString()!);
        Assert.True(loras["canary"].GetProperty("applicable").GetBoolean());
        Assert.False(loras["canary"].TryGetProperty("reason", out _));
        Assert.False(loras["partial"].GetProperty("applicable").GetBoolean());
        Assert.Contains("token_embd.weight", loras["partial"].GetProperty("reason").GetString());
    }

    [Fact]
    public async Task NonStringLora_Is400()
    {
        using var service = Service(supportsLora: true);
        var (context, body) = await Invoke(service, "chat", stream: true, "\"lora\":42");
        var error = AssertRefusedBeforeStreaming(context, body, "lora must be a string");
        Assert.Equal("lora", error.GetProperty("param").GetString());
    }

    /// <summary>As documented, and as an SDK that writes an unset optional field sends it:
    /// <c>"lora": null</c> is the base model, like an omitted field, on both surfaces.</summary>
    [Theory]
    [InlineData("chat")]
    [InlineData("responses")]
    public async Task NullLora_IsTheBaseModel(string surface)
    {
        using var service = Service(supportsLora: true);
        service.LoadLora("canary", _adapter);

        var (context, body) = await Invoke(service, surface, stream: false, "\"lora\":null");

        Assert.Equal(200, context.Response.StatusCode);
        Assert.Contains("BBBBBB", body);
        Assert.DoesNotContain("LLL", body);
    }

    /// <summary>On both surfaces, streamed or not: the adapter reaches the generation of the
    /// request that names it (an argument dropped on the way still compiles, and serves the
    /// base model), and not the requests that name none.</summary>
    [Theory]
    [InlineData("chat", false)]
    [InlineData("chat", true)]
    [InlineData("responses", false)]
    [InlineData("responses", true)]
    public async Task AcceptedAdapter_ReachesThisRequestOnly(string surface, bool stream)
    {
        using var service = Service(supportsLora: true);
        service.LoadLora("canary", _adapter);

        var (adaptedContext, adapted) = await Invoke(service, surface, stream, "\"lora\":\"canary\"");
        var (plainContext, plain) = await Invoke(service, surface, stream, "\"lora\":\"none\"");
        var (absentContext, absent) = await Invoke(service, surface, stream, "");

        Assert.Equal(200, adaptedContext.Response.StatusCode);
        Assert.Equal(200, plainContext.Response.StatusCode);
        Assert.Equal(200, absentContext.Response.StatusCode);
        Assert.Equal("LLLLLL", Content(surface, adapted, stream));
        Assert.Equal("BBBBBB", Content(surface, plain, stream));
        Assert.Equal("BBBBBB", Content(surface, absent, stream));
    }

    // ------------------------------------------------------------------ helpers

    private static JsonElement AssertRefusedBeforeStreaming(DefaultHttpContext context, string body, string reason)
    {
        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        Assert.StartsWith("application/json", context.Response.ContentType);
        Assert.DoesNotContain("data:", body);
        var error = JsonDocument.Parse(body).RootElement.GetProperty("error");
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Contains(reason, error.GetProperty("message").GetString());
        return error.Clone();
    }

    /// <summary>The generated text: the message content (chat) or the <c>output_text</c> parts
    /// of the message items (responses), or the concatenated deltas of a stream.</summary>
    private static string Content(string surface, string body, bool stream)
    {
        var text = new StringBuilder();
        if (!stream)
        {
            using var doc = JsonDocument.Parse(body);
            if (surface == "chat")
                return doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? "";
            foreach (var item in doc.RootElement.GetProperty("output").EnumerateArray())
            {
                if (item.GetProperty("type").GetString() != "message")
                    continue;
                foreach (var part in item.GetProperty("content").EnumerateArray())
                    if (part.GetProperty("type").GetString() == "output_text")
                        text.Append(part.GetProperty("text").GetString());
            }
            return text.ToString();
        }
        foreach (string line in body.Split('\n'))
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal) || line.StartsWith("data: [DONE]", StringComparison.Ordinal))
                continue;
            using var doc = JsonDocument.Parse(line.Substring(6));
            var root = doc.RootElement;
            if (surface == "chat")
            {
                if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0
                    && choices[0].TryGetProperty("delta", out var delta)
                    && delta.TryGetProperty("content", out var content) && content.ValueKind == JsonValueKind.String)
                    text.Append(content.GetString());
            }
            else if (root.TryGetProperty("type", out var type) && type.GetString() == "response.output_text.delta")
            {
                text.Append(root.GetProperty("delta").GetString());
            }
        }
        return text.ToString();
    }

    private async Task<(DefaultHttpContext Context, string Body)> Invoke(ModelService service, string surface, bool stream, string extra)
    {
        string fields = (extra.Length == 0 ? "" : "," + extra) + ",\"stream\":" + (stream ? "true" : "false");
        string request = surface == "chat"
            ? "{\"model\":\"probe.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":6" + fields + "}"
            : "{\"model\":\"probe.gguf\",\"input\":\"hello\",\"store\":false,\"max_output_tokens\":6" + fields + "}";
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", _gguf, "--backend", "cpu", "--no-skills" }, _root);
        var uploads = new UploadStoragePolicy(Path.Combine(_root, "uploads"));
        var workspaces = new SessionWorkspaceManager(Path.Combine(_root, "workspaces"));
        if (surface == "chat")
        {
            await new OpenAIChatAdapter(service, options, uploads, registry, null, workspaces, NullLoggerFactory.Instance)
                .ChatCompletionsAsync(context);
        }
        else
        {
            using var store = new InMemoryResponsesStore();
            await new OpenAIResponsesAdapter(service, options, uploads, registry, null, workspaces, NullLoggerFactory.Instance, store)
                .CreateResponseAsync(context);
        }
        context.Response.Body.Position = 0;
        return (context, await new StreamReader(context.Response.Body).ReadToEndAsync());
    }

    private ModelService Service(bool supportsLora)
    {
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new LoraProbeModel(path, new float[1], 1, 1, supportsLora, LoraProbeModel.Template));
        service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 1024, MaxNumRunningSequences = 2,
            MaxNumBatchedTokens = 1024, MaxPrefillChunkSize = 1024, SoloPrefillChunkSize = 1024,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        service.LoadModel(_gguf, null, "cpu");
        return service;
    }

    /// <summary>A rank-1 adapter of ones on <paramref name="target"/> ([1, 1], scale 1).</summary>
    private static string WriteOnesAdapter(string path, string target)
    {
        string header = "{\"" + target + ".lora_a\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]},"
                      + "\"" + target + ".lora_b\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[4,8]}}";
        byte[] hb = Encoding.UTF8.GetBytes(header);
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)hb.Length));
        fs.Write(hb);
        fs.Write(BitConverter.GetBytes(1f));
        fs.Write(BitConverter.GetBytes(1f));
        return path;
    }
}
