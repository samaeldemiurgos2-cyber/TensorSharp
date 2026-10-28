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
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace TensorSharp.Server.Endpoints;

/// <summary>
/// Administrative routes for loading and unloading text-LoRA adapters at runtime,
/// without reloading the base model. A loaded adapter can then be selected per
/// request with the <c>lora</c> field on <c>/v1/chat/completions</c> or
/// <c>/v1/responses</c>, and is listed under <c>GET /v1/models</c>.
///
/// <para>
/// <c>POST</c> reads only from the directory the operator named at startup
/// (<c>--text-lora-dir</c> / <c>TS_TEXT_LORA_DIR</c>): <c>path</c> and <c>config</c> are
/// relative to it, and anything resolving outside it is refused before the file system
/// is consulted (<see cref="TextLoraPaths.ResolveWithin"/>). Without a directory the route
/// refuses every load. Error text never carries the directory's location.
/// </para>
/// </summary>
public static class LoraEndpoints
{
    public static IEndpointRouteBuilder MapLoraEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/v1/loras", (ModelService svc) =>
        {
            var data = svc.LoraRegistry.List()
                .Select(a => new Dictionary<string, object>
                {
                    ["id"] = a.Name,
                    ["rank"] = a.Rank,
                    ["alpha"] = a.Alpha,
                    ["modules"] = a.Modules,
                })
                .ToList();
            return Results.Json(new { @object = "list", data, loading_enabled = svc.LoraDirectory != null });
        });

        endpoints.MapPost("/v1/loras", async (HttpContext ctx, ModelService svc) =>
        {
            string name;
            string path;
            string? config = null;
            float? scale = null;
            try
            {
                var doc = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body).ConfigureAwait(false);
                if (doc.ValueKind != JsonValueKind.Object)
                {
                    await BadAsync(ctx, "request body must be a JSON object").ConfigureAwait(false);
                    return;
                }
                if (!doc.TryGetProperty("name", out var nameEl) || nameEl.ValueKind != JsonValueKind.String ||
                    string.IsNullOrWhiteSpace(nameEl.GetString()))
                {
                    await BadAsync(ctx, "a non-empty \"name\" is required").ConfigureAwait(false);
                    return;
                }
                name = nameEl.GetString()!;

                if (!doc.TryGetProperty("path", out var pathEl) || pathEl.ValueKind != JsonValueKind.String ||
                    string.IsNullOrWhiteSpace(pathEl.GetString()))
                {
                    await BadAsync(ctx, "a non-empty \"path\" to a .safetensors file in the server's LoRA directory is required").ConfigureAwait(false);
                    return;
                }
                path = pathEl.GetString()!;

                if (doc.TryGetProperty("config", out var configEl))
                {
                    if (configEl.ValueKind != JsonValueKind.String && configEl.ValueKind != JsonValueKind.Null)
                    {
                        await BadAsync(ctx, "\"config\" must be a string").ConfigureAwait(false);
                        return;
                    }
                    config = configEl.ValueKind == JsonValueKind.String ? configEl.GetString() : null;
                }
                if (doc.TryGetProperty("scale", out var scaleEl))
                {
                    if (scaleEl.ValueKind != JsonValueKind.Number || !scaleEl.TryGetSingle(out float s) || !float.IsFinite(s))
                    {
                        await BadAsync(ctx, "\"scale\" must be a finite number").ConfigureAwait(false);
                        return;
                    }
                    scale = s;
                }
            }
            catch (JsonException)
            {
                await BadAsync(ctx, "request body must be a JSON object").ConfigureAwait(false);
                return;
            }

            if (svc.LoraDirectory == null)
            {
                await WriteErrorAsync(ctx, 403, TextLoraPaths.ResolveWithinDisabledMessage).ConfigureAwait(false);
                return;
            }

            try
            {
                var adapter = svc.LoadLora(name, path, config, scale);
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsJsonAsync(new
                {
                    id = adapter.Name,
                    rank = adapter.Rank,
                    alpha = adapter.Alpha,
                    modules = adapter.Deltas.Count,
                    summary = adapter.Summary,
                }).ConfigureAwait(false);
            }
            catch (Exception ex) when (IsLoadFailure(ex))
            {
                // A file the server will open but cannot use: malformed, an unsupported
                // dtype, unreadable, or an adapter that does not fit the loaded model.
                await BadAsync(ctx, TextLoraPaths.Redact(ex.Message, svc.LoraDirectory)).ConfigureAwait(false);
            }
        });

        endpoints.MapDelete("/v1/loras/{name}", (ModelService svc, string name) =>
        {
            bool removed = svc.UnloadLora(name);
            return removed
                ? Results.Json(new { removed = true })
                : Results.Json(new { error = $"No LoRA adapter named '{name}' is loaded." }, statusCode: 404);
        });

        return endpoints;
    }

    /// <summary>Failures that are the request's (or the file's) fault, answered with 400.</summary>
    internal static bool IsLoadFailure(Exception ex) =>
        ex is LoraPathException or InvalidDataException or ArgumentException or IOException
            or UnauthorizedAccessException or NotSupportedException or JsonException or OverflowException
            or InvalidOperationException or FormatException;

    private static Task BadAsync(HttpContext ctx, string message) => WriteErrorAsync(ctx, 400, message);

    private static async Task WriteErrorAsync(HttpContext ctx, int status, string message)
    {
        ctx.Response.StatusCode = status;
        await ctx.Response.WriteAsJsonAsync(new { error = message }).ConfigureAwait(false);
    }
}
