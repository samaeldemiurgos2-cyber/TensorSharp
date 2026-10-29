// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Server.Endpoints;

/// <summary>
/// Administrative routes for loading and unloading text-LoRA adapters at runtime,
/// without reloading the base model. A loaded adapter can then be selected per
/// request with the <c>lora</c> field on <c>/v1/chat/completions</c> or
/// <c>/v1/responses</c>, and is listed under <c>GET /v1/models</c>.
///
/// <para>
/// <c>POST /v1/loras</c> reads only from the operator's LoRA directory
/// (<c>--loras-dir</c> / <c>TS_LORAS_DIR</c>, see <see cref="LoraAdapterDirectory"/>): its
/// <c>path</c> is relative to that directory, and without one the route answers 403. The
/// routes are unauthenticated like the rest of the server, so a deployment that must stop
/// clients from loading and unloading adapters at all keeps them behind its own boundary.
/// Errors use the OpenAI shape <c>{"error":{"message","type","param","code"}}</c>.
/// </para>
/// </summary>
public static class LoraEndpoints
{
    internal const string LoadingDisabled =
        "Runtime LoRA loading is disabled. Start the server with --loras-dir <dir> (or TS_LORAS_DIR) "
        + "to load adapters from that directory.";

    public static IEndpointRouteBuilder MapLoraEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/v1/loras", (ModelService svc) =>
        {
            var data = svc.ListLoras()
                .Select(l =>
                {
                    var entry = new Dictionary<string, object>
                    {
                        ["id"] = l.Info.Name,
                        ["rank"] = l.Info.Rank,
                        ["alpha"] = l.Info.Alpha,
                        ["modules"] = l.Info.Modules,
                        // Adapters are per request: how many requests in flight run with this one.
                        ["in_flight"] = svc.LoraInFlight(l.Info.Name),
                        // Whether the loaded model can apply it, and why not.
                        ["applicable"] = l.Applicable,
                    };
                    if (!l.Applicable)
                        entry["reason"] = l.Reason ?? string.Empty;
                    return entry;
                })
                .ToList();
            return Results.Json(new { @object = "list", data });
        });

        endpoints.MapPost("/v1/loras", async (HttpContext ctx, ModelService svc) =>
        {
            ILogger logger = Logger(ctx);
            var directory = ctx.RequestServices.GetService<LoraAdapterDirectory>();
            if (directory == null)
            {
                logger.LogWarning(LogEventIds.HttpRequestRejected,
                    "POST /v1/loras refused: no LoRA directory is configured (--loras-dir / TS_LORAS_DIR)");
                await ErrorAsync(ctx, StatusCodes.Status403Forbidden, LoadingDisabled, code: "lora_loading_disabled").ConfigureAwait(false);
                return;
            }

            JsonElement doc;
            try
            {
                doc = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body, cancellationToken: ctx.RequestAborted).ConfigureAwait(false);
            }
            catch (JsonException)
            {
                doc = default;
            }
            if (doc.ValueKind != JsonValueKind.Object)
            {
                await ErrorAsync(ctx, 400, "The request body must be a JSON object with \"name\" and \"path\".").ConfigureAwait(false);
                return;
            }

            if (!TryGetOptionalString(doc, "name", out string? name) || name == null)
            {
                await ErrorAsync(ctx, 400, LoraAdapterDirectory.NameRefusal, "name").ConfigureAwait(false);
                return;
            }
            if (!TryGetOptionalString(doc, "path", out string? path) || path == null)
            {
                await ErrorAsync(ctx, 400, LoraAdapterDirectory.ResolutionRefusal, "path").ConfigureAwait(false);
                return;
            }
            if (!TryGetOptionalString(doc, "config", out string? config))
            {
                await ErrorAsync(ctx, 400, LoraAdapterDirectory.ResolutionRefusal, "config").ConfigureAwait(false);
                return;
            }
            float? scale = null;
            if (doc.TryGetProperty("scale", out var scaleEl) && scaleEl.ValueKind != JsonValueKind.Null)
            {
                if (scaleEl.ValueKind != JsonValueKind.Number || !scaleEl.TryGetSingle(out float s) || !float.IsFinite(s))
                {
                    await ErrorAsync(ctx, 400, "\"scale\" must be a finite number.", "scale").ConfigureAwait(false);
                    return;
                }
                scale = s;
            }

            // A reload replaces the adapter for new requests only; requests that already
            // run with the old one finish with it, and its factors are released after them.
            LoraLoadResult result = await directory.LoadAsync(svc, name, path, config, scale, ctx.RequestAborted).ConfigureAwait(false);
            if (result.Adapter == null)
            {
                // The detail (absolute paths, the exception) stays in the log; the client
                // gets the sanitized message only.
                logger.LogWarning(LogEventIds.HttpRequestRejected, result.Exception,
                    "POST /v1/loras refused ({Status}) for adapter {Name}: {Detail}", result.StatusCode, name, result.Detail);
                await ErrorAsync(ctx, result.StatusCode, result.Message!, result.Param, result.Code, result.UnsupportedModules).ConfigureAwait(false);
                return;
            }

            var adapter = result.Adapter;
            logger.LogInformation(LogEventIds.ModelLoadCompleted,
                "LoRA adapter loaded through POST /v1/loras: {Summary} from {Path}", adapter.Summary, result.RelativePath);
            await ctx.Response.WriteAsJsonAsync(new
            {
                id = adapter.Name,
                rank = adapter.Rank,
                alpha = adapter.Alpha,
                modules = adapter.Deltas.Count,
                summary = adapter.Summary,
            }).ConfigureAwait(false);
        });

        endpoints.MapDelete("/v1/loras/{name}", async (HttpContext ctx, ModelService svc, string name) =>
        {
            if (!svc.UnloadLora(name))
            {
                await ErrorAsync(ctx, StatusCodes.Status404NotFound,
                    $"No LoRA adapter named '{name}' is loaded.", code: "lora_not_found").ConfigureAwait(false);
                return;
            }
            Logger(ctx).LogInformation(LogEventIds.ModelUnloaded, "LoRA adapter {Name} unloaded through DELETE /v1/loras", name);
            await ctx.Response.WriteAsJsonAsync(new { removed = true }).ConfigureAwait(false);
        });

        return endpoints;
    }

    /// <summary>
    /// Read an optional string member: true with null when it is absent or JSON null, true with
    /// the value when it is a non-blank string, false for any other kind or a blank string.
    /// </summary>
    private static bool TryGetOptionalString(JsonElement doc, string property, out string? value)
    {
        value = null;
        if (!doc.TryGetProperty(property, out var el) || el.ValueKind == JsonValueKind.Null)
            return true;
        if (el.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(el.GetString()))
            return false;
        value = el.GetString();
        return true;
    }

    private static ILogger Logger(HttpContext ctx)
        => ctx.RequestServices.GetService<ILoggerFactory>()?.CreateLogger("TensorSharp.Server.Lora") ?? NullLogger.Instance;

    /// <summary>The OpenAI error shape every <c>/v1/loras</c> route answers with (a 5xx is a
    /// <c>server_error</c>, the rest <c>invalid_request_error</c>), plus
    /// <c>unsupported_modules</c> when the loaded model cannot apply the adapter.</summary>
    private static Task ErrorAsync(HttpContext ctx, int status, string message, string? param = null, string? code = null,
        IReadOnlyList<string>? unsupportedModules = null)
    {
        ctx.Response.StatusCode = status;
        var error = new Dictionary<string, object?>
        {
            ["message"] = message,
            ["type"] = status >= 500 ? "server_error" : "invalid_request_error",
            ["param"] = param,
            ["code"] = code,
        };
        if (unsupportedModules != null)
            error["unsupported_modules"] = unsupportedModules;
        return ctx.Response.WriteAsJsonAsync(new { error });
    }
}
