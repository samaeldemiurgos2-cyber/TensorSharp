// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// POST /v1/loras used to open any path on the server, and its error text ("LoRA weights file
// not found: ...") told a client whether the path existed. It now loads only from the
// directory the operator names at startup (--text-lora-dir / TS_TEXT_LORA_DIR), refuses
// everything when none is named, and answers a file it cannot use with 400, not 500.
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
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class LoraEndpointsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-lora-http-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private async Task<(WebApplication App, HttpClient Client, ModelService Service)> StartAsync(string loraDirectory)
    {
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Production" });
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        var service = new ModelService { LoraDirectory = loraDirectory };
        builder.Services.AddSingleton(service);
        var app = builder.Build();
        app.MapLoraEndpoints();
        await app.StartAsync();
        string address = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.Single();
        return (app, new HttpClient { BaseAddress = new Uri(address) }, service);
    }

    private string WriteAdapter(string relative)
    {
        string path = Path.Combine(_dir, "loras", relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        TextLoraModelTests.WriteSafetensors(path,
            ("blk.0.ffn_down.weight.lora_a", new long[] { 2, 4 }, new float[8]),
            ("blk.0.ffn_down.weight.lora_b", new long[] { 3, 2 }, new float[6]));
        return path;
    }

    private static async Task<string> ErrorOf(HttpResponseMessage response)
        => JsonDocument.Parse(await response.Content.ReadAsStringAsync()).RootElement.GetProperty("error").GetString()!;

    [Fact]
    public async Task Load_IsConfinedToTheConfiguredDirectory()
    {
        string root = Path.Combine(_dir, "loras");
        WriteAdapter("style/adapter.safetensors");
        File.WriteAllText(Path.Combine(_dir, "secret.safetensors"), "not an adapter");
        var (app, client, service) = await StartAsync(root);
        await using (app)
        using (client)
        {
            using var ok = await client.PostAsJsonAsync("/v1/loras", new { name = "style", path = "style/adapter.safetensors" });
            Assert.Equal(HttpStatusCode.OK, ok.StatusCode);
            Assert.NotNull(service.LoraRegistry.Get("style"));

            using var list = await client.GetAsync("/v1/loras");
            var listed = JsonDocument.Parse(await list.Content.ReadAsStringAsync()).RootElement;
            Assert.Equal("style", listed.GetProperty("data")[0].GetProperty("id").GetString());
            Assert.True(listed.GetProperty("loading_enabled").GetBoolean());

            // Outside the directory, existing or not, the answer is the same and names no host path.
            foreach (string path in new[] { "../secret.safetensors", "../missing.safetensors", Path.Combine(_dir, "secret.safetensors") })
            {
                using var refused = await client.PostAsJsonAsync("/v1/loras", new { name = "x", path });
                Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
                string error = await ErrorOf(refused);
                Assert.DoesNotContain(_dir, error, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain("not found", error, StringComparison.OrdinalIgnoreCase);
            }

            // Inside it, a missing file or one that is not an adapter is 400, without the directory's location.
            using var missing = await client.PostAsJsonAsync("/v1/loras", new { name = "m", path = "nope.safetensors" });
            Assert.Equal(HttpStatusCode.BadRequest, missing.StatusCode);
            Assert.DoesNotContain(root, await ErrorOf(missing), StringComparison.OrdinalIgnoreCase);

            File.WriteAllText(Path.Combine(root, "junk.safetensors"), "garbage that is not a safetensors header");
            using var junk = await client.PostAsJsonAsync("/v1/loras", new { name = "junk", path = "junk.safetensors" });
            Assert.Equal(HttpStatusCode.BadRequest, junk.StatusCode);

            using var badName = await client.PostAsJsonAsync("/v1/loras", new { name = "../x", path = "style/adapter.safetensors" });
            Assert.Equal(HttpStatusCode.BadRequest, badName.StatusCode);

            using var badScale = await client.PostAsJsonAsync("/v1/loras", new { name = "s", path = "style/adapter.safetensors", scale = "big" });
            Assert.Equal(HttpStatusCode.BadRequest, badScale.StatusCode);

            using var deleted = await client.DeleteAsync("/v1/loras/style");
            Assert.Equal(HttpStatusCode.OK, deleted.StatusCode);
            Assert.Null(service.LoraRegistry.Get("style"));
            using var deletedAgain = await client.DeleteAsync("/v1/loras/style");
            Assert.Equal(HttpStatusCode.NotFound, deletedAgain.StatusCode);
        }
    }

    [Fact]
    public async Task Load_IsRefusedWithoutAConfiguredDirectory()
    {
        string previous = Environment.GetEnvironmentVariable("TS_TEXT_LORA_DIR");
        Environment.SetEnvironmentVariable("TS_TEXT_LORA_DIR", null);
        try
        {
            WriteAdapter("adapter.safetensors");
            var (app, client, service) = await StartAsync(null);
            await using (app)
            using (client)
            {
                using var refused = await client.PostAsJsonAsync("/v1/loras",
                    new { name = "a", path = Path.Combine(_dir, "loras", "adapter.safetensors") });
                Assert.Equal(HttpStatusCode.Forbidden, refused.StatusCode);
                Assert.Contains("--text-lora-dir", await ErrorOf(refused));
                Assert.Empty(service.LoraRegistry.List());
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_TEXT_LORA_DIR", previous);
        }
    }
}
