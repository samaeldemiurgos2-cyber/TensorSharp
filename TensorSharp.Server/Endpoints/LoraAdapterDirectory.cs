// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Server.Endpoints;

/// <summary>
/// The server's text-LoRA adapter directory (<c>--loras-dir</c> / <c>TS_LORAS_DIR</c>): the
/// only place <c>POST /v1/loras</c> and the startup load may read adapter files from.
///
/// <para>
/// <c>POST /v1/loras</c> is unauthenticated and its <c>path</c> comes from the client, so
/// without this every file the server process can read was one request away, and the error
/// text told the caller whether a file existed, whether it was readable and what its first
/// eight bytes were. Now a path is accepted only when it names an existing <c>.safetensors</c>
/// or <c>.gguf</c> file inside the directory (relative to it, or absolute and inside it), and
/// <c>config</c> an existing <c>.json</c> there. Containment is
/// <see cref="SkillPathGuard"/>'s: <c>..</c> segments, rooted, drive-qualified and UNC paths,
/// a sibling directory that merely shares the prefix (<c>loras-evil</c>), and a symlink or
/// junction anywhere on the path that leaves the directory are all refused. An
/// <c>adapter_config.json</c> beside the weights is read only after the same check.
/// </para>
/// <para>
/// Every way resolution can fail answers with one fixed message that echoes nothing, so a
/// missing file, a file outside the directory, a directory and a link escape cannot be told
/// apart. A file that resolves but does not load is described without absolute paths or
/// exception text from the I/O layer; the full exception goes to the server log only.
/// </para>
/// <para>
/// The directory must be writable only by the operator: a link swapped in between the check
/// and the open, or a hard link, is not something a path check can see.
/// </para>
/// </summary>
public sealed class LoraAdapterDirectory
{
    /// <summary>Most adapters a server holds at once. Each one is a full float copy of its
    /// factors, so without a bound a client could exhaust memory by loading one file under
    /// fresh names. A reload under a loaded name does not count.</summary>
    public const int MaxAdapters = 32;

    /// <summary>The one answer to every path or config that does not resolve inside the directory.</summary>
    internal const string ResolutionRefusal =
        "\"path\" must name an existing .safetensors or .gguf file inside the server's LoRA directory, "
        + "given relative to it, and \"config\", when given, an existing .json file there.";

    internal const string NameRefusal =
        "\"name\" must be 1-64 letters, digits, '.', '_' or '-', start with a letter or digit, and not be \"none\".";

    /// <summary>The <c>code</c> of a load that failed on the server's side (a 5xx).</summary>
    internal const string LoadFailedCode = "lora_load_failed";

    private const string SiblingConfigName = "adapter_config.json";
    private static readonly string[] WeightExtensions = { ".safetensors", ".gguf" };
    private static readonly string[] ConfigExtensions = { ".json" };
    private static readonly Regex NamePattern = new(@"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\z", RegexOptions.CultureInvariant);

    // Loads one at a time: the capacity check and the registration it guards must not
    // interleave with another load, and two adapters' factors are never converted at once.
    private readonly SemaphoreSlim _loadGate = new(1, 1);

    /// <param name="root">The directory, as an absolute path (<see cref="Hosting.ServerOptionsBuilder"/>
    /// canonicalizes <c>--loras-dir</c> and checks that it exists).</param>
    public LoraAdapterDirectory(string root)
    {
        if (string.IsNullOrWhiteSpace(root) || !Path.IsPathFullyQualified(root))
            throw new ArgumentException("The LoRA directory must be an absolute path.", nameof(root));
        Root = SkillPathGuard.NormalizeDirectory(root);
        RealRoot = ResolveRealRoot(Root);
    }

    /// <summary>The directory, canonical and absolute.</summary>
    public string Root { get; }

    /// <summary>The directory with any link on it followed; resolved paths sit under this one.</summary>
    private string RealRoot { get; }

    /// <summary>
    /// Reads a resolved adapter file: <see cref="TextLoraLoader.Load"/>. The sibling config was
    /// resolved through the guard, so the loader must not look for one on its own. Tests replace
    /// it to fail the read in ways no file on disk can (the host running out of memory).
    /// </summary>
    internal Func<string, string, string?, float?, TextLoraAdapter> ReadAdapter { get; init; } =
        (name, weights, config, scale) => TextLoraLoader.Load(name, weights, config, scale, discoverSiblingConfig: false);

    /// <summary>True for a name an adapter can be registered under: it must fit a URL path
    /// segment (<c>DELETE /v1/loras/{name}</c>) and must not be <c>none</c>, which a
    /// request's <c>lora</c> field reads as "the base model" in any case.</summary>
    public static bool IsValidName(string? name)
        => name != null && NamePattern.IsMatch(name) && !string.Equals(name, "none", StringComparison.OrdinalIgnoreCase);

    /// <summary>Resolve a weights path from a request. On success <paramref name="fullPath"/> is
    /// the file to open (links followed) and <paramref name="relative"/> the client's own path,
    /// relative to the directory, with forward slashes. <paramref name="why"/> is for the server
    /// log only.</summary>
    internal bool TryResolveWeights(string? requested, out string? fullPath, out string? relative, out string? why)
        => TryResolveFile(requested, WeightExtensions, out fullPath, out relative, out why);

    /// <summary>
    /// Resolve the adapter's config: an explicit <paramref name="requested"/> path under the
    /// same rules as the weights (extension <c>.json</c>), else an <c>adapter_config.json</c>
    /// beside the weights. That sibling passes through the guard too; when it exists but
    /// leaves the directory the load is refused rather than run without it. True with a null
    /// <paramref name="configFull"/> when there is no config.
    /// </summary>
    internal bool TryResolveConfig(
        string? requested, string weightsRelative, out string? configFull, out string? configRelative, out string? why)
    {
        if (requested != null)
            return TryResolveFile(requested, ConfigExtensions, out configFull, out configRelative, out why);

        configFull = null;
        configRelative = null;
        int slash = weightsRelative.LastIndexOf('/');
        string sibling = slash < 0 ? SiblingConfigName : weightsRelative[..(slash + 1)] + SiblingConfigName;
        if (!SkillPathGuard.TryResolve(Root, sibling, out string? resolved, out string? guardError))
        {
            why = $"'{sibling}' beside the weights: {guardError}";
            return false;
        }
        why = null;
        if (resolved != null && File.Exists(resolved))
        {
            configFull = resolved;
            configRelative = sibling;
        }
        return true;
    }

    private bool TryResolveFile(
        string? requested, string[] extensions, out string? fullPath, out string? relative, out string? why)
    {
        fullPath = null;
        relative = null;
        if (string.IsNullOrWhiteSpace(requested))
        {
            why = "no path was given";
            return false;
        }
        if (requested.IndexOf('\0') >= 0)
        {
            why = "the path contains a NUL character";
            return false;
        }

        string candidate = requested.Trim();
        // An absolute path is accepted only inside the directory, and from here on it is
        // handled in its relative form, so it meets exactly the checks a relative one does.
        if (Path.IsPathFullyQualified(candidate))
        {
            string full;
            try
            {
                full = Path.GetFullPath(candidate);
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                why = "the absolute path is not usable: " + ex.Message;
                return false;
            }
            if (!SkillPathGuard.IsUnder(Root, full))
            {
                why = "the absolute path is outside the directory";
                return false;
            }
            candidate = Path.GetRelativePath(Root, full);
        }

        // A colon is a drive prefix or an NTFS alternate data stream on Windows; neither
        // belongs in an adapter's name relative to the directory.
        if (candidate.IndexOf(':') >= 0)
        {
            why = "the path contains a ':'";
            return false;
        }
        if (!SkillPathGuard.TryResolveExistingFile(Root, candidate, out string? resolved, out string? guardError))
        {
            why = guardError;
            return false;
        }
        if (!extensions.Any(e => resolved!.EndsWith(e, StringComparison.OrdinalIgnoreCase)))
        {
            why = $"'{Path.GetFileName(resolved)}' does not have a {string.Join(" or ", extensions)} extension";
            return false;
        }

        why = null;
        fullPath = resolved;
        relative = SkillPathGuard.ToSkillRelative(Root, Path.GetFullPath(Path.Combine(Root, candidate.Replace('\\', '/'))));
        return true;
    }

    /// <summary>
    /// Load (or reload) an adapter from this directory and register it on
    /// <paramref name="service"/>: name check, path and config resolution, the
    /// <see cref="MaxAdapters"/> bound, then <see cref="TextLoraLoader.Load"/> and
    /// <see cref="ModelService.RegisterLora"/> (which also refuses an adapter the loaded model
    /// cannot apply in full). Never throws for a bad
    /// request, a bad file or a failed bind; the result carries the status, the client-safe
    /// message and the server-only detail. Shared by <c>POST /v1/loras</c> and the startup
    /// load so both apply the same rules.
    /// </summary>
    public async Task<LoraLoadResult> LoadAsync(
        ModelService service, string? name, string? path, string? config, float? scale,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(service);
        if (!IsValidName(name))
            return LoraLoadResult.Refused(400, NameRefusal, "name", null, null, "invalid adapter name");
        if (!TryResolveWeights(path, out string? weightsFull, out string? weightsRelative, out string? why))
            return LoraLoadResult.Refused(400, ResolutionRefusal, "path", null, null, $"path '{path}' refused: {why}");
        if (!TryResolveConfig(config, weightsRelative!, out string? configFull, out string? configRelative, out why))
            return LoraLoadResult.Refused(400, ResolutionRefusal, "config", null, weightsRelative, $"config refused: {why}");

        await _loadGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (!service.LoraRegistry.Contains(name!) && service.LoraRegistry.Count >= MaxAdapters)
            {
                return LoraLoadResult.Refused(409,
                    $"The server already holds {MaxAdapters} LoRA adapters, its limit. Unload one with DELETE /v1/loras/{{name}} first.",
                    null, "lora_limit_reached", weightsRelative, $"limit of {MaxAdapters} adapters reached");
            }

            // Two phases, blamed apart. Reading the file is the adapter's: what that throws is a
            // 400, except the host running out of memory for its factors, which is the server's
            // (a 503, as in the bind). A corrupt header no longer fails that way: the readers
            // refuse one that claims more than its file holds before anything is sized from it.
            // Binding it to the model is the server's, so from there only an adapter the model
            // cannot apply is a 400.
            TextLoraAdapter adapter;
            try
            {
                adapter = ReadAdapter(name!, weightsFull!, configFull, scale);
            }
            catch (Exception ex) when (IsAdapterFault(ex))
            {
                string reason = DescribeLoadFailure(ex, weightsFull!, weightsRelative!, configFull, configRelative);
                return LoraLoadResult.Refused(400,
                    $"LoRA adapter '{weightsRelative}' could not be loaded: {reason}",
                    "path", null, weightsRelative, $"{weightsFull}: {ex.GetType().Name}: {ex.Message}", ex);
            }
            catch (Exception ex) when (IsOutOfMemory(ex))
            {
                return ServerFault(ex, weightsFull!, weightsRelative!);
            }
            catch (Exception ex)
            {
                // A fault the loader did not anticipate is still the file's: a 400 that
                // quotes nothing.
                return LoraLoadResult.Refused(400,
                    $"LoRA adapter '{weightsRelative}' could not be loaded: reading it failed unexpectedly; the server log has the details",
                    "path", null, weightsRelative, $"{weightsFull}: {ex.GetType().Name}: {ex.Message}", ex);
            }

            try
            {
                return LoraLoadResult.Loaded(service.RegisterLora(adapter), weightsRelative!);
            }
            catch (TextLoraNotApplicableException ex)
            {
                // The file is a valid adapter; the loaded model cannot apply all of it.
                string reason = Sanitize(ex.Message, weightsFull!, weightsRelative!, configFull, configRelative);
                return LoraLoadResult.Refused(400,
                    $"LoRA adapter '{weightsRelative}' could not be loaded: {reason}",
                    "path", LoraRequestError.NotApplicable, weightsRelative, $"{weightsFull}: {ex.Message}", ex)
                    with { UnsupportedModules = ex.UnsupportedModules };
            }
            catch (Exception ex)
            {
                // Anything else is the server's fault, not the adapter's: memory or a device
                // allocation for its factors, a model disposed under the load, a fault the
                // bind did not anticipate. Never a 400, which would blame the request. The
                // plan reads the adapter's own target names, but it does not throw on them:
                // one the model does not have (an out-of-range layer number included) is an
                // unsupported module above, so no exception type is mapped back to the file.
                return ServerFault(ex, weightsFull!, weightsRelative!);
            }
        }
        finally
        {
            _loadGate.Release();
        }
    }

    /// <summary>
    /// A load that failed on the server's side. It still fails only this load (never the
    /// startup load), with a 5xx that quotes nothing: 503 when memory ran out, so a client can
    /// retry, else 500.
    /// </summary>
    private static LoraLoadResult ServerFault(Exception ex, string weightsFull, string weightsRelative)
    {
        bool outOfMemory = IsOutOfMemory(ex);
        string reason = outOfMemory
            ? "the server ran out of memory for its factors"
            : "it failed unexpectedly";
        return LoraLoadResult.Refused(outOfMemory ? 503 : 500,
            $"LoRA adapter '{weightsRelative}' could not be loaded: {reason}; the server log has the details",
            null, LoadFailedCode, weightsRelative, $"{weightsFull}: {ex.GetType().Name}: {ex.Message}", ex);
    }

    /// <summary>
    /// Load every adapter at the top level of the directory (<c>*.safetensors</c> and
    /// <c>*.gguf</c>, in ordinal file-name order), each under its file name without the
    /// extension, through <see cref="LoadAsync"/>. Run after the startup model has loaded, so
    /// each adapter is validated against it and bound once. A file that does not load (or a
    /// name that is not valid, or a second file with the same name) is reported and skipped;
    /// it never stops the server. Loading stops at <see cref="MaxAdapters"/>.
    /// </summary>
    public LoraPreloadReport Preload(ModelService service, ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(service);
        ArgumentNullException.ThrowIfNull(logger);
        var loaded = new List<string>();
        var refused = new List<(string File, string Reason)>();

        List<string> files;
        try
        {
            files = Directory.EnumerateFiles(Root, "*", SearchOption.TopDirectoryOnly)
                .Select(Path.GetFileName)
                .Where(f => f != null && WeightExtensions.Any(e => f.EndsWith(e, StringComparison.OrdinalIgnoreCase)))
                .Select(f => f!)
                .OrderBy(f => f, StringComparer.Ordinal)
                .ToList();
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            logger.LogWarning(LogEventIds.ModelLoadFailed, ex,
                "LoRA directory {Directory} could not be listed; no adapter was loaded at startup", Root);
            return new LoraPreloadReport(loaded, refused);
        }

        foreach (string file in files)
        {
            string name = Path.GetFileNameWithoutExtension(file);
            // Two files with one stem (a.gguf, a.safetensors): the first that loads keeps the
            // name, rather than the later one silently replacing it.
            if (loaded.Contains(name, StringComparer.Ordinal))
            {
                Refuse(file, $"'{file}' was skipped: another file in the directory is already loaded as '{name}'");
                continue;
            }
            if (service.LoraRegistry.Count >= MaxAdapters && !service.LoraRegistry.Contains(name))
            {
                logger.LogWarning(LogEventIds.ModelLoadFailed,
                    "LoRA directory {Directory}: stopped at the limit of {Max} adapters; {File} and later files were not loaded",
                    Root, MaxAdapters, file);
                break;
            }

            LoraLoadResult result;
            try
            {
                result = LoadAsync(service, name, file, null, null).GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                // LoadAsync answers a bad file itself; whatever else escapes still costs this
                // one file, never the server's start.
                Refuse(file, $"'{file}' was skipped: it could not be loaded", $"{ex.GetType().Name}: {ex.Message}", ex);
                continue;
            }
            if (result.Adapter != null)
            {
                loaded.Add(result.Adapter.Name);
                logger.LogInformation(LogEventIds.ModelLoadCompleted, "LoRA adapter loaded at startup: {Summary}", result.Adapter.Summary);
            }
            else
            {
                // Each reason names the file once: the load failures already read
                // "LoRA adapter '<file>' could not be loaded: ...".
                string reason = result.Param switch
                {
                    "name" => $"'{file}' was skipped: '{name}' is not a valid adapter name. {NameRefusal}",
                    "path" or "config" when result.StatusCode == 400 && result.Message == ResolutionRefusal
                        => $"'{file}' was skipped: it does not resolve to a file inside the directory",
                    _ => result.Message!,
                };
                Refuse(file, reason, result.Detail, result.Exception);
            }
        }

        logger.LogInformation(LogEventIds.HostConfiguration,
            "LoRA directory {Directory}: {Loaded} adapter(s) loaded at startup, {Refused} refused; POST /v1/loras loads more from it",
            Root, loaded.Count, refused.Count);
        return new LoraPreloadReport(loaded, refused);

        void Refuse(string file, string reason, string? detail = null, Exception? ex = null)
        {
            refused.Add((file, reason));
            logger.LogWarning(LogEventIds.ModelLoadFailed, ex,
                "LoRA directory {Directory}: {Reason}{Detail}",
                Root, reason, detail != null ? $" ({detail})" : string.Empty);
        }
    }

    /// <summary>
    /// The failures reading an adapter file that say why, answered with a 400 that gives the
    /// reason: malformed or unsupported content (the loader reports a header or config the JSON
    /// reader rejects as <see cref="InvalidDataException"/>), and a truncated or unreadable
    /// file. Anything else reading it is a 400 that quotes nothing, unless memory ran out
    /// (<see cref="IsOutOfMemory"/>), which is the server's.
    /// </summary>
    private static bool IsAdapterFault(Exception ex)
        => ex is InvalidDataException or NotSupportedException or JsonException or IOException
            or UnauthorizedAccessException or OverflowException or ArgumentException;

    /// <summary>Host memory (<see cref="InsufficientMemoryException"/> is one too) or, on the
    /// direct <c>cuda</c> backend, device memory (<c>CUDA_ERROR_OUT_OF_MEMORY</c>) ran out.</summary>
    private static bool IsOutOfMemory(Exception ex)
        => ex is OutOfMemoryException or CudaException { ErrorCode: 2 };

    /// <summary>
    /// A client-safe reason. I/O failures say only that the file could not be read, and JSON
    /// failures do not quote the offending text; the loader's own messages (which name tensors,
    /// dtypes and shapes) are kept, with every path rewritten relative to the directory.
    /// </summary>
    private string DescribeLoadFailure(
        Exception ex, string weightsFull, string weightsRelative, string? configFull, string? configRelative) => ex switch
    {
        EndOfStreamException => "the file is truncated",
        IOException or UnauthorizedAccessException => "the file could not be read",
        JsonException => "its safetensors header or config is not valid JSON",
        OverflowException => "a tensor shape is out of range",
        _ => Sanitize(ex.Message, weightsFull, weightsRelative, configFull, configRelative),
    };

    /// <summary>Rewrite the resolved absolute paths in <paramref name="message"/> to the
    /// client's relative forms, then drop any other mention of the directory itself.</summary>
    internal string Sanitize(string message, string weightsFull, string weightsRelative, string? configFull, string? configRelative)
    {
        string result = message;
        if (configFull != null && configRelative != null)
            result = ReplacePath(result, configFull, configRelative);
        result = ReplacePath(result, weightsFull, weightsRelative);
        // Longest first: with a linked directory one root can be a prefix of the other's text
        // (loras and loras-link).
        foreach (string root in new[] { RealRoot, Root }.Distinct().OrderByDescending(r => r.Length))
        {
            result = ReplacePath(result, root + Path.DirectorySeparatorChar, string.Empty);
            result = ReplacePath(result, root, ".");
        }
        return result;
    }

    private static string ReplacePath(string text, string path, string replacement)
    {
        foreach (string form in new[] { path, path.Replace('\\', '/'), path.Replace('/', '\\') }.Distinct())
            text = text.Replace(form, replacement, SkillPathGuard.PathComparison);
        return text;
    }

    private static string ResolveRealRoot(string root)
    {
        try
        {
            FileSystemInfo? target = new DirectoryInfo(root).ResolveLinkTarget(returnFinalTarget: true);
            return target != null ? SkillPathGuard.NormalizeDirectory(target.FullName) : root;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return root;
        }
    }
}

/// <summary>
/// The outcome of <see cref="LoraAdapterDirectory.LoadAsync"/>. <see cref="Message"/>,
/// <see cref="Param"/> and <see cref="Code"/> are safe to return to the client;
/// <see cref="Detail"/> and <see cref="Exception"/> name absolute paths and belong in the
/// server log only.
/// </summary>
public sealed record LoraLoadResult(
    TextLoraAdapter? Adapter,
    int StatusCode,
    string? Message,
    string? Param,
    string? Code,
    string? RelativePath,
    string? Detail,
    Exception? Exception)
{
    /// <summary>For <c>lora_not_applicable</c>: the modules the loaded model cannot apply.</summary>
    public IReadOnlyList<string>? UnsupportedModules { get; init; }

    internal static LoraLoadResult Loaded(TextLoraAdapter adapter, string relativePath)
        => new(adapter, 200, null, null, null, relativePath, null, null);

    internal static LoraLoadResult Refused(
        int status, string message, string? param, string? code, string? relativePath, string detail, Exception? ex = null)
        => new(null, status, message, param, code, relativePath, detail, ex);
}

/// <summary>What <see cref="LoraAdapterDirectory.Preload"/> loaded (adapter names) and refused
/// (file names relative to the directory, with a client-safe reason).</summary>
public sealed record LoraPreloadReport(IReadOnlyList<string> Loaded, IReadOnlyList<(string File, string Reason)> Refused);
