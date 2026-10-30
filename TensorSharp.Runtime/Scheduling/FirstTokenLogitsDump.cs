// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// <c>TS_DUMP_LOGITS=&lt;path&gt;</c>: the raw float32 logits that choose the run's FIRST GENERATED
    /// TOKEN - the last prompt row once the WHOLE prompt has been processed - written once per process,
    /// plus a JSON sidecar (<c>&lt;path&gt;.json</c>) saying how that row was produced.
    ///
    /// <para>
    /// Comparing two executors by their generated TEXT is a poor test: greedy decoding turns a
    /// near-tie into a visibly different sentence. The logit vector separates a genuine bug from a
    /// close call - but only when both sides dump the SAME row. The dump used to be taken inside
    /// <c>ModelBase.Forward</c>, of the first non-warm-up forward, and the first forward is not the
    /// prompt's last row whenever a prompt runs as more than one forward: a prefix-cache checkpoint
    /// boundary, a recurrent block boundary or a chunk cap splits it, and paths that never call
    /// <c>Forward</c> for the prompt (batched / fused prefill) left the first DECODE forward to be
    /// dumped. Two configurations of one model then compared different computations (Nemotron
    /// <c>--tp 2</c> against one GPU, Qwen3.8 <c>cuda</c> against <c>ggml_cuda</c>).
    /// </para>
    /// <para>
    /// The vector is therefore taken where the first token is CHOSEN: the engine's sampler, the
    /// moment a sequence with no output yet samples from its last logits (whichever path, chunking
    /// or prefill entry point produced them), and the CLI's direct prefill loops (<c>--benchmark</c>,
    /// <c>--test</c>), which call <see cref="TryCapture"/> after their prompt forward. Warm-ups never
    /// reach either point (<c>ModelBase.WarmUpKernels</c> forwards are not sampled by
    /// the engine), and whole warm-up REQUESTS - the CLI's <c>--warmup-runs</c>, the server's
    /// prefix-cache warm-up, the benchmarks' warm-up passes - are submitted inside
    /// <see cref="Suppress"/>, which excludes those requests for their whole life.
    /// </para>
    /// <para>
    /// The sidecar carries the SHA-256 of the vector's bytes, so a tool can tell a sidecar that
    /// describes the vector beside it from a stale one. Stale outputs are removed when a process
    /// first arms the dump (<see cref="ClearStaleOutputs"/>), so a run that never reaches a first
    /// token leaves no file behind instead of an earlier run's.
    /// </para>
    /// <para>
    /// Opt-in and free when unset: the variable is read once, and the hot paths test
    /// <see cref="IsArmed"/> only after a cheaper per-sequence check.
    /// </para>
    /// </summary>
    public static class FirstTokenLogitsDump
    {
        /// <summary>The environment variable naming the dump path.</summary>
        public const string EnvironmentVariable = "TS_DUMP_LOGITS";

        /// <summary>Written into every sidecar: what the vector is.</summary>
        public const string Semantics = "first-generated-token";

        /// <summary>The sidecar is the dump path with this suffix.</summary>
        public const string SidecarSuffix = ".json";

        private static string? s_path = Normalize(Environment.GetEnvironmentVariable(EnvironmentVariable));
        private static int s_claimed;
        private static int s_cleared;
        private static readonly AsyncLocal<int> s_suppressedHere = new();

        /// <summary>True while a dump is still owed: the variable names a path and nothing has been
        /// captured yet. (Suppression is per request, see <see cref="Suppress"/>.)</summary>
        public static bool IsArmed => s_path != null && Volatile.Read(ref s_claimed) == 0;

        /// <summary>The configured dump path, or null when the dump is off.</summary>
        public static string? DumpPath => s_path;

        /// <summary>True inside a <see cref="Suppress"/> scope on the current async flow.</summary>
        public static bool IsSuppressedHere => s_suppressedHere.Value > 0;

        /// <summary>
        /// Exclude the requests SUBMITTED inside the returned scope from the dump: a warm-up request
        /// is a real request whose first token is worthless, and the dump must hold the first token
        /// of the run being measured. The scope belongs to the current async flow (an
        /// <see cref="AsyncLocal{T}"/>), and a <see cref="SequenceState"/> records it when it is
        /// created, so the exclusion follows the request: a warm-up that is still in the engine when
        /// its caller gives up (a cancelled background warm-up) stays excluded, and a real request
        /// submitted from another flow while the warm-up runs is still eligible. The CLI's direct
        /// prefill loops, which capture on the caller's flow, honour it through
        /// <see cref="TryCapture"/>. Nests; dispose on the flow that opened it.
        /// </summary>
        public static IDisposable Suppress()
        {
            s_suppressedHere.Value = s_suppressedHere.Value + 1;
            return new SuppressScope();
        }

        /// <summary>
        /// Remove an earlier run's <c>&lt;path&gt;</c> and <c>&lt;path&gt;.json</c>, once per process
        /// and only while nothing has been captured: a run that fails before its first token must not
        /// leave the previous run's pair looking like its own. Called when a process starts serving
        /// (CLI start-up, every <see cref="InferenceEngine"/>); a no-op when the dump is off.
        /// </summary>
        public static void ClearStaleOutputs()
        {
            string? path = s_path;
            if (path == null || Volatile.Read(ref s_claimed) != 0)
                return;
            if (Interlocked.CompareExchange(ref s_cleared, 1, 0) != 0)
                return;
            foreach (string stale in new[] { path, path + SidecarSuffix })
            {
                try
                {
                    if (File.Exists(stale))
                        File.Delete(stale);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
                {
                    try { Console.Error.WriteLine($"[{EnvironmentVariable}] could not remove the stale {stale}: {ex.Message}"); }
                    catch (Exception) { }
                }
            }
        }

        /// <summary>
        /// Write <paramref name="logits"/> as the run's first-token vector, once per process: the
        /// first call wins, every later call (and every call inside a <see cref="Suppress"/> scope on
        /// this flow, or while the dump is unset) returns null. For a caller that runs the prompt on
        /// its own flow (the CLI's direct loops); the engine, which samples on its worker thread,
        /// checks the request's own exclusion instead. The binary is written before anything samples
        /// from the vector, because samplers mask and penalize in place; call
        /// <see cref="Capture.Complete"/> with the token that was chosen to finish the sidecar.
        /// </summary>
        public static Capture? TryCapture(ReadOnlySpan<float> logits, FirstTokenLogitsInfo info)
            => IsSuppressedHere ? null : TryCaptureCore(logits, info);

        /// <summary><see cref="TryCapture"/> without the current-flow suppression check: the engine's
        /// sampler runs on its own thread, so it decides from the sequence's
        /// <see cref="SequenceState.ExcludedFromFirstTokenDump"/>.</summary>
        internal static Capture? TryCaptureCore(ReadOnlySpan<float> logits, FirstTokenLogitsInfo info)
        {
            string? path = s_path;
            if (path == null || logits.IsEmpty)
                return null;
            if (Interlocked.CompareExchange(ref s_claimed, 1, 0) != 0)
                return null;

            int argmax = 0;
            for (int i = 1; i < logits.Length; i++)
                if (logits[i] > logits[argmax]) argmax = i;
            byte[] row = new byte[logits.Length * sizeof(float)];
            for (int i = 0; i < logits.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(row.AsSpan(i * sizeof(float)), logits[i]);
            var capture = new Capture(path, info ?? new FirstTokenLogitsInfo(), logits.Length, argmax, logits[argmax],
                Convert.ToHexString(SHA256.HashData(row)).ToLowerInvariant());
            try
            {
                string? directory = Path.GetDirectoryName(Path.GetFullPath(path));
                if (!string.IsNullOrEmpty(directory))
                    Directory.CreateDirectory(directory);
                // An older sidecar must never sit beside this vector, even if writing the new one fails.
                string sidecar = path + SidecarSuffix;
                if (File.Exists(sidecar))
                    File.Delete(sidecar);
                using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
                    stream.Write(row, 0, row.Length);
                capture.WriteSidecar();
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
            {
                // A diagnostic must not fail the request it observes. The comparison tools treat a
                // missing vector or sidecar as "not dumped", never as a match.
                try { Console.Error.WriteLine($"[{EnvironmentVariable}] could not write {path}: {ex.Message}"); }
                catch (Exception) { }
                return null;
            }
            return capture;
        }

        /// <summary>FNV-1a 64 over the prompt token ids, so two dumps can be shown to come from the
        /// same prompt before their vectors are compared.</summary>
        public static string Fingerprint(IReadOnlyList<int>? tokens)
        {
            ulong hash = 14695981039346656037UL;
            if (tokens != null)
            {
                for (int i = 0; i < tokens.Count; i++)
                {
                    uint v = unchecked((uint)tokens[i]);
                    for (int b = 0; b < 4; b++)
                    {
                        hash ^= (byte)(v >> (8 * b));
                        hash *= 1099511628211UL;
                    }
                }
            }
            return hash.ToString("x16");
        }

        /// <summary>Test hook: re-arm with <paramref name="path"/> (null turns the dump off) and
        /// clear this flow's suppression; pass <see cref="EnvironmentPath"/> to restore the process
        /// setting.</summary>
        internal static void ResetForTests(string? path)
        {
            s_path = Normalize(path);
            Volatile.Write(ref s_claimed, 0);
            Volatile.Write(ref s_cleared, 0);
            s_suppressedHere.Value = 0;
        }

        /// <summary>The path the environment variable names, for <see cref="ResetForTests"/>.</summary>
        internal static string? EnvironmentPath => Normalize(Environment.GetEnvironmentVariable(EnvironmentVariable));

        private static string? Normalize(string? path) => string.IsNullOrWhiteSpace(path) ? null : path;

        private sealed class SuppressScope : IDisposable
        {
            private int _disposed;

            public void Dispose()
            {
                if (Interlocked.Exchange(ref _disposed, 1) == 0)
                    s_suppressedHere.Value = Math.Max(0, s_suppressedHere.Value - 1);
            }
        }

        /// <summary>A vector that has been written; <see cref="Complete"/> records the token chosen from it.</summary>
        public sealed class Capture
        {
            private readonly string _path;
            private readonly FirstTokenLogitsInfo _info;
            private readonly int _count;
            private readonly int _argmax;
            private readonly float _argmaxLogit;
            private readonly string _vectorSha256;
            private int? _token;
            private string? _tokenSource;

            internal Capture(string path, FirstTokenLogitsInfo info, int count, int argmax, float argmaxLogit, string vectorSha256)
            {
                _path = path;
                _info = info;
                _count = count;
                _argmax = argmax;
                _argmaxLogit = argmaxLogit;
                _vectorSha256 = vectorSha256;
            }

            /// <summary>Index of the largest logit in the written vector.</summary>
            public int Argmax => _argmax;

            /// <summary>Record the first generated token and how it was chosen: "sampled" (the
            /// request's sampler over this vector), "argmax", or a token the vector did not choose
            /// ("forced-thinking-end", "device-pending"). Rewrites the sidecar; never throws.</summary>
            public void Complete(int token, string source)
            {
                _token = token;
                _tokenSource = source;
                try { WriteSidecar(); }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                {
                    try { Console.Error.WriteLine($"[{EnvironmentVariable}] could not update {_path}{SidecarSuffix}: {ex.Message}"); }
                    catch (Exception) { }
                }
            }

            internal void WriteSidecar()
            {
                using var stream = new FileStream(_path + SidecarSuffix, FileMode.Create, FileAccess.Write, FileShare.Read);
                using var json = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
                json.WriteStartObject();
                json.WriteString("semantics", Semantics);
                json.WriteString("format", "f32le");
                json.WriteNumber("count", _count);
                json.WriteString("vector_sha256", _vectorSha256);
                json.WriteString("source", _info.Source ?? "unknown");
                if (_info.RequestId != null) json.WriteString("request_id", _info.RequestId);
                json.WriteNumber("prompt_tokens", _info.PromptTokens);
                if (_info.PromptFingerprint != null) json.WriteString("prompt_fingerprint", _info.PromptFingerprint);
                json.WriteNumber("media_spans", _info.MediaSpans);
                if (_info.Adapter != null) json.WriteString("adapter", _info.Adapter);
                json.WriteNumber("prefix_reused_tokens", _info.PrefixReusedTokens);
                json.WriteNumber("prefill_forwards", _info.PrefillSteps?.Count ?? 0);
                json.WriteStartArray("prefill_steps");
                if (_info.PrefillSteps != null)
                {
                    foreach (var step in _info.PrefillSteps)
                    {
                        json.WriteStartObject();
                        json.WriteString("path", step.Path);
                        json.WriteNumber("tokens", step.Tokens);
                        json.WriteEndObject();
                    }
                }
                json.WriteEndArray();
                if (_info.PrefillSteps is { Count: > 0 })
                    json.WriteString("final_prefill_path", _info.PrefillSteps[_info.PrefillSteps.Count - 1].Path);
                json.WriteNumber("argmax", _argmax);
                json.WriteNumber("argmax_logit", _argmaxLogit);
                if (_token.HasValue)
                {
                    json.WriteNumber("first_token", _token.Value);
                    json.WriteString("first_token_source", _tokenSource ?? "unknown");
                }
                else
                {
                    json.WriteNull("first_token");
                }
                json.WriteNumber("pid", Environment.ProcessId);
                json.WriteString("written_utc", DateTime.UtcNow.ToString("O"));
                json.WriteEndObject();
            }
        }
    }

    /// <summary>One forward that carried part of a prompt: the scheduler plan path it ran under (a
    /// label, not the kernel that executed it) and how many prompt tokens it ran. A prompt restored
    /// from the prefix cache has no step for the reused part.</summary>
    public readonly record struct PrefillStepTrace(string Path, int Tokens);

    /// <summary>What <see cref="FirstTokenLogitsDump"/> records beside the vector.</summary>
    public sealed record FirstTokenLogitsInfo
    {
        /// <summary>Who captured it: "engine" (the inference engine's sampler), "cli-benchmark", ...</summary>
        public string? Source { get; init; }

        /// <summary>The request whose first token this is, when there is one.</summary>
        public string? RequestId { get; init; }

        /// <summary>Prompt length in tokens; the vector is the row after the last of them.</summary>
        public int PromptTokens { get; init; }

        /// <summary><see cref="FirstTokenLogitsDump.Fingerprint"/> of the prompt token ids. Media
        /// placeholders are ids too, so two prompts that differ only in an image's content share
        /// it; <see cref="MediaSpans"/> at least says media were present.</summary>
        public string? PromptFingerprint { get; init; }

        /// <summary>How many media spans (image / audio / video embeddings) the prompt carried.</summary>
        public int MediaSpans { get; init; }

        /// <summary>The per-request LoRA adapter the row was computed under, when there was one.</summary>
        public string? Adapter { get; init; }

        /// <summary>Leading prompt tokens restored from a cache instead of forwarded.</summary>
        public int PrefixReusedTokens { get; init; }

        /// <summary>The forwards that ran the prompt, in order (replays included).</summary>
        public IReadOnlyList<PrefillStepTrace>? PrefillSteps { get; init; }
    }
}
