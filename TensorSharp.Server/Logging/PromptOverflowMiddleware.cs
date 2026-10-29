using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace TensorSharp.Server.Logging;

/// <summary>
/// Maps a <see cref="PromptContextOverflowException"/> (or the engine's
/// <see cref="TensorSharp.Runtime.Scheduling.ContextLengthExceededException"/>) — a client input
/// error (the prompt does not fit the context window) — to a 400 with a
/// protocol-appropriate JSON body. The overflow surfaces during prompt
/// preparation, before any response body is written, so the status is still
/// settable. Once the response has started, the exception is left to
/// propagate.
/// </summary>
public static class PromptOverflowMiddleware
{
    public static IApplicationBuilder UsePromptOverflowHandling(this IApplicationBuilder app)
    {
        return app.Use(next => context => InvokeAsync(context, () => next(context)));
    }

    internal static async Task InvokeAsync(HttpContext context, System.Func<Task> next)
    {
        try
        {
            await next().ConfigureAwait(false);
        }
        catch (PromptContextOverflowException ex) when (!context.Response.HasStarted)
        {
            await WriteBadRequestAsync(context, ex.Message).ConfigureAwait(false);
        }
        // The engine's own refusal of a prompt that leaves no room for a reply. The chat
        // pipeline trims or rejects such a prompt before submitting it, so this is the
        // backstop for any caller that reaches the engine without that step: the same
        // client error, not a 500.
        catch (TensorSharp.Runtime.Scheduling.ContextLengthExceededException ex) when (!context.Response.HasStarted)
        {
            await WriteBadRequestAsync(context, ex.Message).ConfigureAwait(false);
        }
    }

    private static async Task WriteBadRequestAsync(HttpContext context, string message)
    {
        context.Response.Clear();
        context.Response.StatusCode = StatusCodes.Status400BadRequest;

        // OpenAI SDKs parse {error:{message,type}}; the Ollama and Web UI
        // clients expect a flat {error:"..."}. Shape by route prefix.
        if (context.Request.Path.StartsWithSegments("/v1"))
        {
            await context.Response.WriteAsJsonAsync(new
            {
                error = new { message, type = "invalid_request_error", code = "context_length_exceeded" }
            }).ConfigureAwait(false);
        }
        else
        {
            await context.Response.WriteAsJsonAsync(new { error = message }).ConfigureAwait(false);
        }
    }
}
