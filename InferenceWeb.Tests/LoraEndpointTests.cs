// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Logging;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

/// <summary>
/// <c>/v1/loras</c> over real HTTP (Kestrel on 127.0.0.1:0, Production environment, the
/// server's own exception middleware): runtime loading is confined to <c>--loras-dir</c> and
/// refused without it, every path that does not resolve there gets one byte-identical answer,
/// a file that does not load is a 400 that names no server path (never the middleware's
/// "Could not read the request body"), a failure on the server's side is a 5xx that quotes
/// nothing, and every error has the OpenAI shape.
/// </summary>
public sealed class LoraEndpointTests : IDisposable
{
    private readonly string _base = Directory.CreateTempSubdirectory("ts-lora-http-").FullName;
    private readonly string _root;
    private readonly string _outside;

    public LoraEndpointTests()
    {
        _root = Path.Combine(_base, "loras");
        _outside = Path.Combine(_base, "outside");
        Directory.CreateDirectory(Path.Combine(_root, "sub"));
        Directory.CreateDirectory(_outside);
        Directory.CreateDirectory(_root + "-evil");
        LoraFixtures.WriteAdapter(Path.Combine(_root, "a.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(_root, "r2.safetensors"), rank: 2);
        LoraFixtures.WriteAdapter(Path.Combine(_outside, "secret.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(_root + "-evil", "a.safetensors"));
        File.WriteAllText(Path.Combine(_outside, "config.json"), "{\"r\":1,\"lora_alpha\":4}");
    }

    public void Dispose() => LoraLinks.DeleteTree(_base);

    [Fact]
    public async Task Post_WithoutLoraDirectory_Returns403_AndRegistryStaysEmpty()
    {
        await using var server = await LoraServer.StartAsync(directory: null);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras",
            new { name = "a", path = Path.Combine(_outside, "secret.safetensors") });

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
        var error = await ErrorOf(response);
        Assert.Equal("lora_loading_disabled", error.GetProperty("code").GetString());
        Assert.Contains("--loras-dir", error.GetProperty("message").GetString());
        Assert.Equal(0, server.Service.LoraRegistry.Count);
        Assert.Empty(await ListAsync(server));
    }

    [Fact]
    public async Task Post_RelativeAdapter_LoadsListsAndDeletes()
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));

        using var posted = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "math", path = "r2.safetensors", scale = 0.5 });
        Assert.Equal(HttpStatusCode.OK, posted.StatusCode);
        using (var body = JsonDocument.Parse(await posted.Content.ReadAsStringAsync()))
        {
            Assert.Equal("math", body.RootElement.GetProperty("id").GetString());
            Assert.Equal(2, body.RootElement.GetProperty("rank").GetInt32());
            Assert.Equal(1, body.RootElement.GetProperty("modules").GetInt32());
            LoraFixtures.AssertNoPathInJson(body.RootElement.GetRawText(), _root);
        }

        var listed = Assert.Single(await ListAsync(server));
        Assert.Equal("math", listed.GetProperty("id").GetString());
        Assert.Equal(0, listed.GetProperty("in_flight").GetInt32());

        using var deleted = await server.Client.DeleteAsync("/v1/loras/math");
        Assert.Equal(HttpStatusCode.OK, deleted.StatusCode);
        Assert.True(JsonDocument.Parse(await deleted.Content.ReadAsStringAsync()).RootElement.GetProperty("removed").GetBoolean());

        using var again = await server.Client.DeleteAsync("/v1/loras/math");
        Assert.Equal(HttpStatusCode.NotFound, again.StatusCode);
        Assert.Equal("lora_not_found", (await ErrorOf(again)).GetProperty("code").GetString());
        Assert.Empty(await ListAsync(server));
    }

    [Fact]
    public async Task Post_AbsolutePathInsideTheDirectory_IsAccepted()
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));
        using var posted = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "abs", path = Path.Combine(_root, "a.safetensors") });
        Assert.Equal(HttpStatusCode.OK, posted.StatusCode);
        LoraFixtures.AssertNoPathInJson(await posted.Content.ReadAsStringAsync(), _root);
    }

    [Fact]
    public async Task Post_ReloadSameName_ReplacesTheEntry()
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));
        Assert.Equal(HttpStatusCode.OK, (await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "a.safetensors" })).StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "r2.safetensors" })).StatusCode);

        var entry = Assert.Single(await ListAsync(server));
        Assert.Equal(2, entry.GetProperty("rank").GetInt32());
    }

    [Fact]
    public async Task Post_SiblingAdapterConfig_IsReadThroughTheDirectory()
    {
        LoraFixtures.WriteAdapter(Path.Combine(_root, "sub", "adapter.safetensors"));
        File.WriteAllText(Path.Combine(_root, "sub", "adapter_config.json"), "{\"r\":1,\"lora_alpha\":4}");
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));

        using var posted = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "cfg", path = "sub/adapter.safetensors" });

        Assert.Equal(HttpStatusCode.OK, posted.StatusCode);
        Assert.Equal(4f, JsonDocument.Parse(await posted.Content.ReadAsStringAsync()).RootElement.GetProperty("alpha").GetSingle());
    }

    /// <summary>
    /// Missing, outside, traversal, sibling-prefix, a directory, the wrong extension, rooted,
    /// UNC and drive paths: one byte-identical body, which names neither the directory nor
    /// anything about the target, so it cannot tell a caller whether a file exists.
    /// </summary>
    [Fact]
    public async Task Post_EveryPathThatDoesNotResolve_GetsAByteIdenticalBody()
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));
        string[] paths =
        {
            "missing.safetensors",
            Path.Combine(_outside, "secret.safetensors"),
            Path.Combine(_outside, "missing.safetensors"),
            "../outside/secret.safetensors",
            "..\\outside\\secret.safetensors",
            Path.Combine(_root + "-evil", "a.safetensors"),
            "sub",
            "a.json",
            "/etc/passwd",
            "../../../../../../etc/passwd",
            "/proc/self/environ",
            "C:\\Windows\\win.ini",
            "\\\\server\\share\\x.safetensors",
            "a.safetensors:s",
        };

        var bodies = new System.Collections.Generic.List<string>();
        foreach (string path in paths)
        {
            using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path });
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
            bodies.Add(await response.Content.ReadAsStringAsync());
        }

        string body = Assert.Single(bodies.Distinct());
        using var parsed = JsonDocument.Parse(body);
        var error = parsed.RootElement.GetProperty("error");
        Assert.Equal(LoraAdapterDirectory.ResolutionRefusal, error.GetProperty("message").GetString());
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Equal("path", error.GetProperty("param").GetString());
        LoraFixtures.AssertNoPathInJson(body, _base);
        Assert.DoesNotContain("not found", body, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("bytes", body, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(0, server.Service.LoraRegistry.Count);
    }

    [LoraDirectoryLinkFact]
    public async Task Post_DirectoryLinkOutOfTheDirectory_GetsTheSameBodyAsAMissingFile()
    {
        Assert.True(LoraLinks.TryLinkDirectory(Path.Combine(_root, "escape"), _outside));
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));

        using var linked = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "escape/secret.safetensors" });
        using var missing = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "missing.safetensors" });

        Assert.Equal(HttpStatusCode.BadRequest, linked.StatusCode);
        Assert.Equal(await missing.Content.ReadAsStringAsync(), await linked.Content.ReadAsStringAsync());
        Assert.Equal(0, server.Service.LoraRegistry.Count);
    }

    [LoraFileSymlinkFact]
    public async Task Post_FileSymlinkOutOfTheDirectory_GetsTheSameBodyAsAMissingFile()
    {
        Assert.True(LoraLinks.TryLinkFile(Path.Combine(_root, "leak.safetensors"), Path.Combine(_outside, "secret.safetensors")));
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));

        using var linked = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "leak.safetensors" });
        using var missing = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "missing.safetensors" });

        Assert.Equal(HttpStatusCode.BadRequest, linked.StatusCode);
        Assert.Equal(await missing.Content.ReadAsStringAsync(), await linked.Content.ReadAsStringAsync());
    }

    public static TheoryData<string, string> CorruptAdapters => new()
    {
        { "truncated", "the file is truncated" },
        { "huge-header", "header length" },
        { "header-not-json", "not valid JSON" },
        { "dtype-q4k", "unknown safetensors dtype 'Q4_K'" },
        { "f8-factors", "LoRA factors must be F32, F16, BF16 or F64" },
        { "i32-factors", "LoRA factors must be F32, F16, BF16 or F64" },
        { "b-without-a", "has no 'w.lora_a' tensor" },
        { "no-pairs", "contains no '<weight>.lora_a' / '<weight>.lora_b' tensor pairs" },
        { "missing-dtype", "does not have the expected fields and types" },
        { "dtype-number", "does not have the expected fields and types" },
        { "shape-fraction", "does not have the expected fields and types" },
        { "rank-mismatch", "rank mismatch" },
        { "gguf", "is not a valid GGUF file" },
        { "gguf-alignment", "GGUF general.alignment 0 is not a power of 2" },
        { "peft", "convert_lora_to_gguf.py <peft_adapter_dir> --base <hf_base_model_dir>" },
        // Shapes whose byte count wraps round to the bytes present: refused by the reader, not
        // left to fail when the factor's array is sized from them.
        { "gguf-ne0-wrap", "GGUF tensor 'w.lora_a' (F32, ne [4611686018427387905, 1] at data offset 0) does not fit in the file's" },
        { "shape-wrap", "safetensors tensor 'w.lora_a' has shape [1,4611686018427387905], which is negative or too large for F32" },
        // Headers that size an allocation from their own claims: refused before it is made,
        // never the OutOfMemoryException that read as the server running out (a 503).
        { "gguf-dims", "has 2147483647 dimensions; at most 8 are supported" },
        { "gguf-ne0", "GGUF tensor 'w.lora_a' (F32, ne [536870911, 1] at data offset 0) does not fit in the file's" },
    };

    [Theory]
    [MemberData(nameof(CorruptAdapters))]
    public async Task Post_CorruptAdapter_Is400NamingTheReason_NeverTheRequestBodyMessageOrAServerPath(string kind, string reason)
    {
        string file = WriteCorrupt(kind);
        var logger = new LoraRecordingLogger();
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), logger);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "bad", path = file });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        var error = JsonDocument.Parse(body).RootElement.GetProperty("error");
        string message = error.GetProperty("message").GetString()!;
        Assert.StartsWith($"LoRA adapter '{file}' could not be loaded: ", message);
        Assert.Contains(reason, message);
        Assert.Equal("path", error.GetProperty("param").GetString());
        Assert.DoesNotContain("Could not read the request body", body);
        LoraFixtures.AssertNoPathInJson(body, _base);
        Assert.Equal(0, server.Service.LoraRegistry.Count);
        // The full detail, absolute path included, is in the server log instead.
        Assert.Contains(logger.Entries, e => e.Level == LogLevel.Warning && e.Message.Contains(Path.Combine(_root, file)));
    }

    /// <summary>
    /// A file that fails to read in a way the loader did not anticipate is still the file's
    /// fault: a 400 that quotes nothing, never a 5xx <c>server_error</c>, which is kept for the
    /// server running out of memory and for the bind. A <c>data_offsets</c> of one element,
    /// which the safetensors reader indexes past, is one.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Post_ACorruptAdapterThatFailsUnexpectedly_Is400QuotingNothing_WithOrWithoutAModel(bool modelLoaded)
    {
        LoraFixtures.WriteRaw(Path.Combine(_root, "offsets.safetensors"),
            "{\"w.lora_a\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0]}}", new byte[4]);
        var service = modelLoaded ? ProbeService() : null;
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), service: service);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "offsets.safetensors" });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string raw = await response.Content.ReadAsStringAsync();
        Assert.DoesNotContain(nameof(IndexOutOfRangeException), raw);
        LoraFixtures.AssertNoPathInJson(raw, _base);
        var error = await ErrorOf(response);
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Equal("path", error.GetProperty("param").GetString());
        Assert.Equal(JsonValueKind.Null, error.GetProperty("code").ValueKind);
        Assert.Equal("LoRA adapter 'offsets.safetensors' could not be loaded: reading it failed unexpectedly; the server log has the details",
            error.GetProperty("message").GetString());
        Assert.Equal(0, server.Service.LoraRegistry.Count);
    }

    /// <summary>
    /// The host running out of memory while the file is read (its factors are converted to
    /// floats then) is the server's fault, as it is during the bind: a 503 <c>server_error</c>
    /// that quotes nothing, never the 400 of a bad file. A corrupt header cannot get here; the
    /// readers refuse one that would size an allocation (see <see cref="CorruptAdapters"/>).
    /// </summary>
    [Theory]
    [InlineData("out-of-memory")]
    [InlineData("insufficient-memory")]
    public async Task Post_HostMemoryRunningOutReadingTheAdapter_Is503QuotingNothing(string kind)
    {
        var service = ProbeService();
        var directory = new LoraAdapterDirectory(_root)
        {
            ReadAdapter = (_, weights, _, _) => throw (kind == "out-of-memory"
                ? new OutOfMemoryException($"Array dimensions exceeded reading {weights}")
                : new InsufficientMemoryException($"Array dimensions exceeded reading {weights}")),
        };
        await using var server = await LoraServer.StartAsync(directory, service: service);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "a.safetensors" });

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        string raw = await response.Content.ReadAsStringAsync();
        LoraFixtures.AssertNoPathInJson(raw, _root);
        Assert.DoesNotContain("Array dimensions", raw);
        var error = await ErrorOf(response);
        Assert.Equal("server_error", error.GetProperty("type").GetString());
        Assert.Equal(LoraAdapterDirectory.LoadFailedCode, error.GetProperty("code").GetString());
        Assert.Equal(JsonValueKind.Null, error.GetProperty("param").ValueKind);
        Assert.Equal("LoRA adapter 'a.safetensors' could not be loaded: the server ran out of memory for its factors; the server log has the details",
            error.GetProperty("message").GetString());
        Assert.Equal(0, service.LoraRegistry.Count);
        Assert.Empty(((TensorSharp.Runtime.Scheduling.IAdapterAwareModel)service.Model).BoundAdapters);
    }

    [Fact]
    public async Task Post_ConfigThatDoesNotResolve_GetsTheUniformBody()
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));
        var bodies = new System.Collections.Generic.List<string>();
        foreach (string config in new[] { Path.Combine(_outside, "config.json"), "../outside/config.json", "missing.json", "a.safetensors" })
        {
            using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "a.safetensors", config });
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
            bodies.Add(await response.Content.ReadAsStringAsync());
        }
        string body = Assert.Single(bodies.Distinct());
        Assert.Equal("config", JsonDocument.Parse(body).RootElement.GetProperty("error").GetProperty("param").GetString());
        LoraFixtures.AssertNoPathInJson(body, _base);
    }

    [Fact]
    public async Task Post_MalformedConfigInsideTheDirectory_Is400_NotTheRequestBodyMessage()
    {
        File.WriteAllText(Path.Combine(_root, "bad.json"), "{ \"r\": 1, oops");
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "a.safetensors", config = "bad.json" });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        Assert.Contains("not valid JSON", body);
        Assert.DoesNotContain("Could not read the request body", body);
        Assert.DoesNotContain("oops", body);
        LoraFixtures.AssertNoPathInJson(body, _base);
    }

    [Theory]
    [InlineData("[]", null)]
    [InlineData("\"a\"", null)]
    [InlineData("not json", null)]
    [InlineData("", null)]
    [InlineData("{}", "name")]
    [InlineData("{\"name\":\"a\"}", "path")]
    [InlineData("{\"name\":\"none\",\"path\":\"a.safetensors\"}", "name")]
    [InlineData("{\"name\":\"NONE\",\"path\":\"a.safetensors\"}", "name")]
    [InlineData("{\"name\":\"a/b\",\"path\":\"a.safetensors\"}", "name")]
    [InlineData("{\"name\":7,\"path\":\"a.safetensors\"}", "name")]
    [InlineData("{\"name\":\"a\",\"path\":5}", "path")]
    [InlineData("{\"name\":\"a\",\"path\":\"\"}", "path")]
    [InlineData("{\"name\":\"a\",\"path\":\"a.safetensors\",\"config\":5}", "config")]
    [InlineData("{\"name\":\"a\",\"path\":\"a.safetensors\",\"scale\":1e60}", "scale")]
    [InlineData("{\"name\":\"a\",\"path\":\"a.safetensors\",\"scale\":\"2\"}", "scale")]
    public async Task Post_InvalidBody_Is400InTheOpenAIShape(string json, string? param)
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));

        using var response = await server.Client.PostAsync("/v1/loras", new StringContent(json, Encoding.UTF8, "application/json"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = await ErrorOf(response);
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        if (param == null)
            Assert.Equal(JsonValueKind.Null, error.GetProperty("param").ValueKind);
        else
            Assert.Equal(param, error.GetProperty("param").GetString());
        Assert.Equal(0, server.Service.LoraRegistry.Count);
    }

    [Fact]
    public async Task Post_NullConfigAndScale_AreTheDefaults()
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));
        using var response = await server.Client.PostAsync("/v1/loras", new StringContent(
            "{\"name\":\"a\",\"path\":\"a.safetensors\",\"config\":null,\"scale\":null}", Encoding.UTF8, "application/json"));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task Post_WhenTheRegistryIsFull_Is409_ButAReloadStillWorks()
    {
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));
        for (int i = 0; i < LoraAdapterDirectory.MaxAdapters; i++)
            Assert.Equal(HttpStatusCode.OK, (await server.Client.PostAsJsonAsync("/v1/loras", new { name = "n" + i, path = "a.safetensors" })).StatusCode);

        using var full = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "extra", path = "a.safetensors" });
        Assert.Equal(HttpStatusCode.Conflict, full.StatusCode);
        Assert.Equal("lora_limit_reached", (await ErrorOf(full)).GetProperty("code").GetString());

        using var reload = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "n3", path = "r2.safetensors" });
        Assert.Equal(HttpStatusCode.OK, reload.StatusCode);
        Assert.Equal(LoraAdapterDirectory.MaxAdapters, (await ListAsync(server)).Length);
    }

    [Fact]
    public async Task Post_AnAdapterTheLoadedModelCannotApply_Is400_AndRegistersNothing()
    {
        string gguf = Path.Combine(_base, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(gguf);
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new LoraProbeModel(path, new float[1], 1, 1, supportsLora: false, LoraProbeModel.Template));
        service.LoadModel(gguf, null, "cpu");
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), service: service);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "a.safetensors" });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = await ErrorOf(response);
        string message = error.GetProperty("message").GetString()!;
        Assert.StartsWith("LoRA adapter 'a.safetensors' could not be loaded: ", message);
        Assert.Contains("nemotron_h, mistral3 and hunyuan-dense", message);
        Assert.Equal("lora_not_applicable", error.GetProperty("code").GetString());
        Assert.Equal(new[] { "blk.{0}.attn_output.weight" },
            error.GetProperty("unsupported_modules").EnumerateArray().Select(m => m.GetString()).ToArray());
        Assert.Equal(0, service.LoraRegistry.Count);
    }

    [Fact]
    public async Task Post_AnAdapterOnANormWeight_Is400NotApplicable_NeverA500()
    {
        // The target is a 1-D tensor of the loaded model; planning it used to index a second
        // dimension it does not have, and the IndexOutOfRangeException became a 500.
        var service = ProbeService();
        LoraFixtures.WriteAdapter(Path.Combine(_root, "norm.safetensors"), LoraProbeModel.Norm);
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), service: service);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "norm", path = "norm.safetensors" });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = await ErrorOf(response);
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Equal("lora_not_applicable", error.GetProperty("code").GetString());
        Assert.Contains("not a 2-D projection weight", error.GetProperty("message").GetString());
        Assert.Equal(new[] { "blk.{0}.attn_norm.weight" },
            error.GetProperty("unsupported_modules").EnumerateArray().Select(m => m.GetString()).ToArray());
        Assert.Equal(0, service.LoraRegistry.Count);
    }

    /// <summary>
    /// A target whose layer number does not fit an int is a target no model has: a 400
    /// <c>lora_not_applicable</c> that lists it as written, never the 500 its
    /// OverflowException became once the bind was blamed on the server, and nothing is left
    /// bound or registered, whether the model refuses it per target or (a family that has not
    /// opted in) as a whole.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Post_ATargetWhoseLayerNumberOverflows_Is400NotApplicable_AndLeavesNoBinding(bool supportsLora)
    {
        const string target = "blk.99999999999.attn_output.weight";
        LoraFixtures.WriteAdapter(Path.Combine(_root, "huge.safetensors"), target);
        var service = ProbeService(supportsLora);
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), service: service);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "huge", path = "huge.safetensors" });

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = await ErrorOf(response);
        Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
        Assert.Equal("lora_not_applicable", error.GetProperty("code").GetString());
        Assert.Contains(supportsLora ? target + " (not a weight of the loaded model)" : "nemotron_h, mistral3 and hunyuan-dense",
            error.GetProperty("message").GetString());
        Assert.Equal(new[] { target },
            error.GetProperty("unsupported_modules").EnumerateArray().Select(m => m.GetString()).ToArray());
        Assert.Equal(0, service.LoraRegistry.Count);
        Assert.Empty(((TensorSharp.Runtime.Scheduling.IAdapterAwareModel)service.Model).BoundAdapters);
    }

    /// <summary>
    /// Factors with an empty dimension ([0, 2^60] and [2^60, 0], no data bytes) passed every
    /// size check on zero bytes, then building the delta looped 2^60 times inside the load
    /// gate: this request never answered and every later load waited behind it. They are
    /// refused as a GGUF adapter's are, at once, and the gate is free for the next load.
    /// </summary>
    [Fact]
    public async Task Post_AnAdapterWithAnEmptyDimension_Is400AtOnce_NotAHang()
    {
        const long huge = 1L << 60;
        LoraFixtures.WriteRaw(Path.Combine(_root, "empty.safetensors"),
            "{\"w.lora_a\":{\"dtype\":\"F32\",\"shape\":[0," + huge + "],\"data_offsets\":[0,0]},"
            + "\"w.lora_b\":{\"dtype\":\"F32\",\"shape\":[" + huge + ",0],\"data_offsets\":[0,0]}}",
            Array.Empty<byte>());
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root));

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "empty", path = "empty.safetensors" })
            .WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var error = await ErrorOf(response);
        Assert.Equal("path", error.GetProperty("param").GetString());
        Assert.Equal("LoRA adapter 'empty.safetensors' could not be loaded: LoRA tensor 'w' in 'empty.safetensors' has an empty dimension.",
            error.GetProperty("message").GetString());
        Assert.Equal(0, server.Service.LoraRegistry.Count);

        using var next = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "a", path = "a.safetensors" })
            .WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Equal(HttpStatusCode.OK, next.StatusCode);
    }

    public static TheoryData<string, int> ServerFaults => new()
    {
        { "invalid-operation", 500 },
        { "disposed", 500 },
        { "cuda-launch", 500 },
        { "out-of-memory", 503 },
        { "insufficient-memory", 503 },
        { "cuda-out-of-memory", 503 },
    };

    /// <summary>
    /// A bind that fails on the server's side (memory, a device, a model disposed under the
    /// load) is a 5xx <c>server_error</c>, never a 400 that blames the request: 503 when
    /// memory ran out, else 500. It quotes neither the exception nor a path.
    /// </summary>
    [Theory]
    [MemberData(nameof(ServerFaults))]
    public async Task Post_AServerSideBindFailure_Is5xxQuotingNothing_AndLeavesNoBinding(string kind, int status)
    {
        var service = ProbeService();
        var model = (LoraProbeModel)service.Model;
        string where = Path.Combine(_root, "a.safetensors");
        model.BindFailure = kind switch
        {
            "invalid-operation" => new InvalidOperationException($"cudaMalloc failed uploading {where}"),
            "disposed" => new ObjectDisposedException($"cudaMalloc model {where}"),
            "cuda-launch" => new TensorSharp.Cuda.Interop.CudaException(700, $"cudaMalloc illegal address uploading {where}"),
            "out-of-memory" => new OutOfMemoryException($"cudaMalloc failed uploading {where}"),
            "insufficient-memory" => new InsufficientMemoryException($"cudaMalloc failed uploading {where}"),
            "cuda-out-of-memory" => new TensorSharp.Cuda.Interop.CudaException(2, $"cudaMalloc out of memory uploading {where}"),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
        };
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), service: service);

        using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "x", path = "a.safetensors" });

        Assert.Equal((HttpStatusCode)status, response.StatusCode);
        string raw = await response.Content.ReadAsStringAsync();
        LoraFixtures.AssertNoPathInJson(raw, _root);
        Assert.DoesNotContain("cudaMalloc", raw);
        var error = await ErrorOf(response);
        Assert.Equal("server_error", error.GetProperty("type").GetString());
        Assert.Equal(LoraAdapterDirectory.LoadFailedCode, error.GetProperty("code").GetString());
        Assert.Equal(JsonValueKind.Null, error.GetProperty("param").ValueKind);
        Assert.StartsWith("LoRA adapter 'a.safetensors' could not be loaded: ", error.GetProperty("message").GetString());
        Assert.Equal(0, service.LoraRegistry.Count);
        Assert.Empty(((TensorSharp.Runtime.Scheduling.IAdapterAwareModel)model).BoundAdapters);
    }

    [Fact]
    public async Task Get_SaysWhetherEachAdapterAppliesToTheLoadedModel()
    {
        string gguf = Path.Combine(_base, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(gguf);
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new LoraProbeModel(path, new float[1], 1, 1, supportsLora: true, LoraProbeModel.Template));
        service.LoadModel(gguf, null, "cpu");
        // A GGUF adapter loads like a safetensors one.
        TextLoraFactors.WriteGguf(Path.Combine(_root, "g.gguf"), "probe", 1f,
            new[] { (LoraFixtures.Target, new[] { 1f }, new[] { 1f }, 1, 1, 1) });
        LoraFixtures.WriteAdapter(Path.Combine(_root, "embedding.safetensors"), "token_embd.weight");
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), service: service);

        using var good = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "good", path = "a.safetensors" });
        using var gg = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "gg", path = "g.gguf" });
        // Registered behind the model's back, as an adapter loaded before a model switch is.
        service.LoraRegistry.Load("embedding", Path.Combine(_root, "embedding.safetensors"));

        Assert.Equal(HttpStatusCode.OK, good.StatusCode);
        Assert.Equal(HttpStatusCode.OK, gg.StatusCode);
        var listed = (await ListAsync(server)).ToDictionary(e => e.GetProperty("id").GetString()!);
        Assert.True(listed["good"].GetProperty("applicable").GetBoolean());
        Assert.True(listed["gg"].GetProperty("applicable").GetBoolean());
        Assert.False(listed["good"].TryGetProperty("reason", out _));
        Assert.False(listed["embedding"].GetProperty("applicable").GetBoolean());
        Assert.Contains("token_embd.weight", listed["embedding"].GetProperty("reason").GetString());
    }

    public static TheoryData<string, string, string> AlphasThatAreNotFinite => new()
    {
        { "metadata", "Infinity", "Infinity" },
        { "metadata", "1e39", "Infinity" },
        { "metadata", "NaN", "NaN" },
        { "metadata", "-16", "-16" },
        { "config", "1e39", "Infinity" },
        { "config", "-16", "-16" },
    };

    /// <summary>
    /// An adapter whose alpha (in its metadata or its <c>adapter_config.json</c>) is not a
    /// finite number >= 0 is a 400 naming the reason, under a new name or a loaded one, and
    /// nothing is registered or replaced. An infinite alpha used to load and register first:
    /// the response could not serialize it (a 500), and from then on neither could
    /// <c>GET /v1/loras</c> nor <c>GET /v1/models</c>, until the adapter was deleted. A
    /// negative or NaN one loaded at scale +1.
    /// </summary>
    [Theory]
    [MemberData(nameof(AlphasThatAreNotFinite))]
    public async Task Post_AnAdapterWhoseAlphaIsNotAFiniteNumber_Is400_AndTheListingsStay200(string source, string alpha, string read)
    {
        string path;
        string reason;
        if (source == "metadata")
        {
            path = "alpha.safetensors";
            LoraFixtures.WriteAdapter(Path.Combine(_root, path), metadataAlpha: alpha);
            reason = $"'alpha.safetensors' has __metadata__ alpha {read}; it must be a finite number >= 0.";
        }
        else
        {
            Directory.CreateDirectory(Path.Combine(_root, "cfg"));
            path = "cfg/alpha.safetensors";
            LoraFixtures.WriteAdapter(Path.Combine(_root, "cfg", "alpha.safetensors"));
            File.WriteAllText(Path.Combine(_root, "cfg", "adapter_config.json"), "{\"r\":1,\"lora_alpha\":" + alpha + "}");
            reason = $"LoRA config 'adapter_config.json' has lora_alpha {read}; it must be a finite number >= 0.";
        }
        var service = ProbeService();
        await using var server = await LoraServer.StartAsync(new LoraAdapterDirectory(_root), service: service, listModels: _base);
        using (var good = await server.Client.PostAsJsonAsync("/v1/loras", new { name = "good", path = "a.safetensors" }))
            Assert.Equal(HttpStatusCode.OK, good.StatusCode);
        var registered = service.LoraRegistry.Get("good");

        foreach (string name in new[] { "bad", "good" })
        {
            using var response = await server.Client.PostAsJsonAsync("/v1/loras", new { name, path });

            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
            var error = await ErrorOf(response);
            Assert.Equal("invalid_request_error", error.GetProperty("type").GetString());
            Assert.Equal("path", error.GetProperty("param").GetString());
            Assert.Equal(JsonValueKind.Null, error.GetProperty("code").ValueKind);
            string message = error.GetProperty("message").GetString()!;
            Assert.StartsWith("LoRA adapter '", message);
            Assert.EndsWith(" could not be loaded: " + reason, message);
        }

        Assert.Same(registered, service.LoraRegistry.Get("good"));
        Assert.Equal(1, service.LoraRegistry.Count);
        await AssertListingsServe(server, "good");
    }

    /// <summary>
    /// The startup load reads such a file the same way: it is refused with the reason and the
    /// server comes up with both listings serving, where it used to come up with the adapter
    /// registered and both listings failing.
    /// </summary>
    [Fact]
    public async Task Preload_AnAdapterWhoseAlphaIsInfinite_IsRefused_AndTheListingsServeFromStartup()
    {
        string root = Path.Combine(_base, "preload");
        Directory.CreateDirectory(root);
        LoraFixtures.WriteAdapter(Path.Combine(root, "good.safetensors"));
        LoraFixtures.WriteAdapter(Path.Combine(root, "infinite.safetensors"), metadataAlpha: "Infinity");
        var service = ProbeService();
        var directory = new LoraAdapterDirectory(root);
        var logger = new LoraRecordingLogger();

        LoraPreloadReport report = directory.Preload(service, logger);

        Assert.Equal(new[] { "good" }, report.Loaded);
        var (file, reason) = Assert.Single(report.Refused);
        Assert.Equal("infinite.safetensors", file);
        Assert.Contains("'infinite.safetensors' has __metadata__ alpha Infinity; it must be a finite number >= 0.", reason);
        Assert.Single(logger.Entries, e => e.Level == LogLevel.Warning);
        await using var server = await LoraServer.StartAsync(directory, service: service, listModels: _base);
        await AssertListingsServe(server, "good");
    }

    // ------------------------------------------------------------------ helpers

    /// <summary><c>GET /v1/loras</c> and <c>GET /v1/models</c> both answer 200, listing exactly
    /// <paramref name="adapter"/>.</summary>
    private static async Task AssertListingsServe(LoraServer server, string adapter)
    {
        using var loras = await server.Client.GetAsync("/v1/loras");
        Assert.Equal(HttpStatusCode.OK, loras.StatusCode);
        using (var doc = JsonDocument.Parse(await loras.Content.ReadAsStringAsync()))
            Assert.Equal(adapter, Assert.Single(doc.RootElement.GetProperty("data").EnumerateArray()).GetProperty("id").GetString());

        using var models = await server.Client.GetAsync("/v1/models");
        Assert.Equal(HttpStatusCode.OK, models.StatusCode);
        using (var doc = JsonDocument.Parse(await models.Content.ReadAsStringAsync()))
            Assert.Equal(adapter, Assert.Single(doc.RootElement.GetProperty("data")[0].GetProperty("loras").EnumerateArray())
                .GetProperty("id").GetString());
    }

    /// <summary>A service with a probe model loaded that takes adapters (or, without
    /// <paramref name="supportsLora"/>, refuses every one as a family that has not opted in).</summary>
    private ModelService ProbeService(bool supportsLora = true)
    {
        string gguf = Path.Combine(_base, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(gguf);
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new LoraProbeModel(path, new float[1], 1, 1, supportsLora, LoraProbeModel.Template));
        service.LoadModel(gguf, null, "cpu");
        return service;
    }

    /// <summary>Write one kind of broken adapter into the directory; returns its relative name.</summary>
    private string WriteCorrupt(string kind)
    {
        string name = kind + (kind.StartsWith("gguf", StringComparison.Ordinal) ? ".gguf" : ".safetensors");
        string path = Path.Combine(_root, name);
        switch (kind)
        {
            case "truncated":
                File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });
                break;
            case "huge-header":
                var bytes = new byte[64];
                new Random(7).NextBytes(bytes);
                BitConverter.GetBytes(0x7FFF_FFFF_FFFFUL).CopyTo(bytes, 0);
                File.WriteAllBytes(path, bytes);
                break;
            case "header-not-json":
                LoraFixtures.WriteRaw(path, "{\"w.lora_a\": oops", new byte[8]);
                break;
            case "dtype-q4k":
                LoraFixtures.WriteAdapter(path, "w", dtype: "Q4_K");
                break;
            case "f8-factors":
                LoraFixtures.WriteAdapter(path, "w", dtype: "F8_E4M3");
                break;
            case "i32-factors":
                LoraFixtures.WriteAdapter(path, "w", dtype: "I32");
                break;
            case "b-without-a":
                LoraFixtures.WriteRaw(path, "{\"w.lora_b\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}", new byte[4]);
                break;
            case "no-pairs":
                LoraFixtures.WriteRaw(path, "{\"w.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]}}", new byte[4]);
                break;
            case "missing-dtype":
                LoraFixtures.WriteRaw(path, "{\"w.lora_a\":{\"shape\":[1,1],\"data_offsets\":[0,4]}}", new byte[4]);
                break;
            case "dtype-number":
                LoraFixtures.WriteRaw(path, "{\"w.lora_a\":{\"dtype\":5,\"shape\":[1,1],\"data_offsets\":[0,4]}}", new byte[4]);
                break;
            case "shape-fraction":
                LoraFixtures.WriteRaw(path, "{\"w.lora_a\":{\"dtype\":\"F32\",\"shape\":[1.5,1],\"data_offsets\":[0,4]}}", new byte[4]);
                break;
            case "rank-mismatch":
                LoraFixtures.WriteRaw(path,
                    "{\"w.lora_a\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[0,8]},\"w.lora_b\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[8,12]}}",
                    new byte[12]);
                break;
            case "gguf":
                File.WriteAllBytes(path, Encoding.ASCII.GetBytes("GGUF").Concat(new byte[60]).ToArray());
                break;
            case "gguf-dims":
            case "gguf-ne0":
            case "gguf-ne0-wrap":
            case "gguf-alignment":
                TextLoraFactors.WriteCorruptGguf(path, kind["gguf-".Length..]);
                break;
            case "shape-wrap":
                // 2^62 + 1 F32 values wrap round to the 4 bytes given.
                LoraFixtures.WriteRaw(path,
                    "{\"w.lora_a\":{\"dtype\":\"F32\",\"shape\":[1,4611686018427387905],\"data_offsets\":[0,4]},"
                    + "\"w.lora_b\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[4,8]}}",
                    new byte[8]);
                break;
            case "peft":
                // As PEFT saves it: Hugging Face module names, refused with the converter's command.
                LoraFixtures.WriteRaw(path,
                    "{\"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[0,4]},"
                    + "\"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight\":{\"dtype\":\"F32\",\"shape\":[1,1],\"data_offsets\":[4,8]}}",
                    new byte[8]);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(kind), kind, null);
        }
        return name;
    }

    private static async Task<JsonElement[]> ListAsync(LoraServer server)
    {
        using var doc = JsonDocument.Parse(await server.Client.GetStringAsync("/v1/loras"));
        return doc.RootElement.GetProperty("data").EnumerateArray().Select(e => e.Clone()).ToArray();
    }

    private static async Task<JsonElement> ErrorOf(HttpResponseMessage response)
    {
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        var error = doc.RootElement.GetProperty("error");
        // The OpenAI shape: every member present, param and code null when they do not apply.
        foreach (string member in new[] { "message", "type", "param", "code" })
            Assert.True(error.TryGetProperty(member, out _), $"error.{member} is missing");
        return error.Clone();
    }

    /// <summary>The LoRA routes on real Kestrel, behind the server's own exception middleware;
    /// with <c>listModels</c> (the directory holding the probe model), <c>GET /v1/models</c>
    /// too, as the server maps it.</summary>
    private sealed class LoraServer : IAsyncDisposable
    {
        private readonly WebApplication _app;

        private LoraServer(WebApplication app, HttpClient client, ModelService service)
        {
            _app = app;
            Client = client;
            Service = service;
        }

        public HttpClient Client { get; }
        public ModelService Service { get; }

        public static async Task<LoraServer> StartAsync(
            LoraAdapterDirectory? directory, LoraRecordingLogger? logger = null, ModelService? service = null,
            string? listModels = null)
        {
            service ??= new ModelService(NullLogger<ModelService>.Instance);
            var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
            builder.Logging.ClearProviders();
            if (logger != null)
                builder.Logging.AddProvider(logger);
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            builder.Services.AddSingleton(service);
            if (directory != null)
                builder.Services.AddSingleton(directory);
            if (listModels != null)
            {
                var options = ServerOptionsBuilder.Build(
                    new[] { "--model", Path.Combine(listModels, "probe.gguf"), "--backend", "cpu", "--no-skills" }, listModels);
                builder.Services.AddSingleton(new OpenAIChatAdapter(service, options,
                    new UploadStoragePolicy(Path.Combine(listModels, "uploads")),
                    new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }), null,
                    new SessionWorkspaceManager(Path.Combine(listModels, "workspaces")), NullLoggerFactory.Instance));
            }
            var app = builder.Build();
            app.UseApiExceptionHandling();
            app.MapLoraEndpoints();
            if (listModels != null)
                app.MapGet("/v1/models", (OpenAIChatAdapter chat) => chat.ListModels());
            await app.StartAsync();
            string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            return new LoraServer(app, new HttpClient { BaseAddress = new Uri(address) }, service);
        }

        public async ValueTask DisposeAsync()
        {
            Client.Dispose();
            await _app.StopAsync();
            await _app.DisposeAsync();
            // Registered as an instance, so the container does not own it.
            Service.Dispose();
        }
    }
}
