// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Studio;

public sealed record StudioRenderCapabilities(bool Available, string? Reason, string? Codec, bool DrawText, bool ProbeAvailable = false);
public sealed record StudioRenderPlan(long Revision, long DurationFrames, string FilterGraph, IReadOnlyList<string> Arguments);

/// <summary>Local subprocess adapter. Every argument uses ArgumentList; project data never enters a shell.
/// Network protocols and playlist demuxers are excluded. Cancellation kills and joins the process tree.</summary>
public sealed class StudioMedia(StudioOptions options)
{
    const string Formats = "mov,matroska,avi,wav,mp3,flac,ogg,image2,png_pipe,jpeg_pipe,webp_pipe,gif,aiff";
    readonly SemaphoreSlim capabilityGate = new(1, 1);
    StudioRenderCapabilities? cachedCapabilities;
    string? cachedFfmpegPath, cachedFfprobePath;
    static string N(double n) => n.ToString("0.#########", CultureInfo.InvariantCulture);

    public async Task<StudioRenderCapabilities> GetCapabilitiesAsync(CancellationToken ct = default)
    {
        if (!options.EnableExternalProcesses) return new(false, "Local media probing and rendering require desktop subprocess support on this host.", null, false);
        await capabilityGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            string ffmpeg = options.FfmpegPath, ffprobe = options.FfprobePath;
            if (cachedCapabilities != null && cachedFfmpegPath == ffmpeg && cachedFfprobePath == ffprobe) return cachedCapabilities;
            cachedCapabilities = null;
            bool probed = false, drawText = false;
            try
            {
                await RunAsync(ffprobe, ["-version"], null, ct, TimeSpan.FromSeconds(10));
                probed = true;
                var encoders = await RunAsync(ffmpeg, ["-hide_banner", "-encoders"], null, ct, TimeSpan.FromSeconds(10));
                var filters = await RunAsync(ffmpeg, ["-hide_banner", "-filters"], null, ct, TimeSpan.FromSeconds(10));
                drawText = filters.Contains(" drawtext ", StringComparison.Ordinal);
                List<string> candidates = [];
                if ((OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst()) && encoders.Contains(" h264_videotoolbox ", StringComparison.Ordinal)) candidates.Add("h264_videotoolbox");
                if (encoders.Contains(" libx264 ", StringComparison.Ordinal)) candidates.Add("libx264");
                if (candidates.Count == 0 || !encoders.Contains(" aac ", StringComparison.Ordinal)) return new(false, "The installed FFmpeg needs an H.264 encoder (VideoToolbox or libx264) and AAC encoder. No codec substitution is performed.", null, drawText, true);
                // Being compiled into FFmpeg does not mean an encoder can start on this host. In particular,
                // VideoToolbox can be listed in a headless/restricted session where no encoding session opens.
                // Verify the actual MP4/AAC path before admitting native inference, trying software H.264 if needed.
                var failures = new List<string>();
                foreach (var codec in candidates)
                {
                    try { await VerifyEncoderAsync(ffmpeg, ffprobe, codec, ct).ConfigureAwait(false); }
                    catch (Exception e) when (e is not OperationCanceledException) { failures.Add($"{codec}: {e.Message}"); continue; }
                    cachedFfmpegPath = ffmpeg; cachedFfprobePath = ffprobe;
                    return cachedCapabilities = new(true, null, codec, drawText, true);
                }
                return new(false, "FFmpeg could not encode and probe an H.264/AAC MP4: " + string.Join("; ", failures), null, drawText, true);
            }
            catch (Exception e) when (e is not OperationCanceledException)
            { return new(false, $"FFmpeg/ffprobe could not be used: {e.Message}", null, drawText, probed); }
        }
        finally { capabilityGate.Release(); }
    }

    static async Task VerifyEncoderAsync(string ffmpeg, string ffprobe, string codec, CancellationToken ct)
    {
        string directory = Path.Combine(Path.GetTempPath(), "tensorstudio-media-check-" + StudioJson.NewId());
        Directory.CreateDirectory(directory);
        try
        {
            string output = Path.Combine(directory, "check.mp4");
            List<string> arguments = ["-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24:d=0.125",
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "0.125", "-pix_fmt", "yuv420p", "-c:v", codec];
            AddVideoEncodingArguments(arguments, codec, 64, 64, 24);
            arguments.AddRange(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", output]);
            await RunAsync(ffmpeg, arguments, directory, ct, TimeSpan.FromSeconds(10)).ConfigureAwait(false);
            var json = await RunAsync(ffprobe, ["-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov",
                "-show_entries", "stream=codec_name,codec_type", "-of", "json", output], directory, ct, TimeSpan.FromSeconds(10)).ConfigureAwait(false);
            using var doc = JsonDocument.Parse(json);
            if (!doc.RootElement.TryGetProperty("streams", out var streams) ||
                !streams.EnumerateArray().Any(s => S(s, "codec_type") == "video" && S(s, "codec_name") == "h264") ||
                !streams.EnumerateArray().Any(s => S(s, "codec_type") == "audio" && S(s, "codec_name") == "aac"))
                throw new StudioException("media_encode_validation_failed", "The encoder test did not produce H.264 video and AAC audio.");
        }
        finally { try { Directory.Delete(directory, true); } catch (IOException) { } catch (UnauthorizedAccessException) { } }
    }

    static void AddVideoEncodingArguments(List<string> arguments, string codec, int width, int height, double fps)
    {
        if (codec == "libx264") arguments.AddRange(["-preset", "veryfast", "-crf", "20"]);
        else
        {
            // Keep the existing quality budget for small projects, scaling it with pixels and frame rate for HD/4K.
            long bitrate = Math.Max(6_000_000, (long)Math.Ceiling(width * (double)height * fps * .12));
            arguments.AddRange(["-b:v", bitrate.ToString(CultureInfo.InvariantCulture), "-allow_sw", "1"]);
        }
    }

    public async Task<StudioAsset> ProbeAsync(string path, string displayName, CancellationToken ct)
    {
        if (!options.EnableExternalProcesses) throw new StudioException("unsupported_platform", "Media import requires ffprobe on this host; existing projects can still be inspected and edited.");
        var json = await RunAsync(options.FfprobePath, ["-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", Formats,
            "-show_format", "-show_streams", "-of", "json", path], null, ct, TimeSpan.FromSeconds(30));
        using var doc = JsonDocument.Parse(json);
        var streams = doc.RootElement.GetProperty("streams").EnumerateArray().ToArray();
        var video = streams.FirstOrDefault(s => S(s, "codec_type") == "video" && (!s.TryGetProperty("disposition", out var d) || !d.TryGetProperty("attached_pic", out var pic) || pic.GetInt32() == 0));
        var audio = streams.Any(s => S(s, "codec_type") == "audio");
        var image = Path.GetExtension(path).ToLowerInvariant() is ".png" or ".jpg" or ".jpeg" or ".webp";
        if (video.ValueKind == JsonValueKind.Undefined && !audio) throw new StudioException("invalid_media", "File has no supported video, image, or audio stream.");
        if (image && video.ValueKind == JsonValueKind.Undefined) throw new StudioException("invalid_media", "Image file does not contain a decodable image.");
        var duration = doc.RootElement.TryGetProperty("format", out var format) ? D(format, "duration") : 0;
        duration = Math.Max(duration, streams.Select(s => D(s, "duration")).DefaultIfEmpty().Max());
        var width = I(video, "width"); var height = I(video, "height");
        if (width > 16384 || height > 16384 || (long)width * height > 100_000_000) throw new StudioException("media_too_large", "Media dimensions exceed the decoder safety limit.");
        // A phone photo is often stored on its side with an EXIF orientation. Record the size it is shown and
        // generated at: the video models decode it upright, and a canvas sized from the first frame must agree.
        if (image && await QuarterTurnAsync(path, ct)) (width, height) = (height, width);
        if (!image && (!double.IsFinite(duration) || duration <= 0 || duration > 24 * 60 * 60)) throw new StudioException("invalid_duration", "Media duration must be positive and no longer than 24 hours.");
        var rate = S(video, "r_frame_rate"); var average = S(video, "avg_frame_rate");
        return new StudioAsset
        {
            Name = displayName, FileName = Path.GetFileName(path), Kind = image ? "image" : video.ValueKind != JsonValueKind.Undefined ? "video" : "audio",
            DurationSeconds = image ? 0 : duration, Width = width, Height = height, HasAudio = audio,
            SourceFrameRate = average, VariableFrameRate = !image && rate != average,
            ContentType = ContentType(Path.GetExtension(path)), Bytes = new FileInfo(path).Length
        };
    }

    /// <summary>True when the image's display matrix (EXIF orientation 5-8, as ffprobe reports it per frame) turns it
    /// a quarter turn. A probe that fails leaves the stored size, which is what the import recorded before.</summary>
    async Task<bool> QuarterTurnAsync(string path, CancellationToken ct)
    {
        try
        {
            var json = await RunAsync(options.FfprobePath, ["-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", Formats,
                "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_entries", "frame_side_data=rotation", "-of", "json", path], null, ct, TimeSpan.FromSeconds(30));
            using var doc = JsonDocument.Parse(json);
            foreach (var frame in doc.RootElement.TryGetProperty("frames", out var frames) ? frames.EnumerateArray() : default)
                foreach (var side in frame.TryGetProperty("side_data_list", out var list) ? list.EnumerateArray() : default)
                    if (side.TryGetProperty("rotation", out var r) && r.TryGetDouble(out double degrees) && Math.Abs(Math.Round(degrees / 90)) % 2 == 1)
                        return true;
        }
        catch (Exception e) when (e is not OperationCanceledException) { }
        return false;
    }

    public static string ContentType(string extension) => extension.ToLowerInvariant() switch
    {
        ".mp4" or ".m4v" => "video/mp4", ".mov" => "video/quicktime", ".webm" => "video/webm", ".mkv" => "video/x-matroska", ".avi" => "video/x-msvideo",
        ".wav" => "audio/wav", ".mp3" => "audio/mpeg", ".m4a" => "audio/mp4", ".flac" => "audio/flac", ".ogg" => "audio/ogg", ".aiff" => "audio/aiff",
        ".png" => "image/png", ".jpg" or ".jpeg" => "image/jpeg", ".webp" => "image/webp", ".gif" => "image/gif",
        _ => throw new StudioException("unsupported_media", "Supported media: MP4/MOV/WebM/MKV/AVI, WAV/MP3/M4A/FLAC/OGG/AIFF, PNG/JPEG/WebP/GIF.")
    };

    public StudioRenderPlan CreateRenderPlan(StudioProject p, Func<string, string> assetPath, string workDirectory, string codec)
    {
        StudioCommands.Validate(p, options);
        if (p.DurationFrames <= 0) throw new StudioException("empty_timeline", "Add at least one clip or caption before rendering.");
        if (p.Clips.Count + p.Captions.Count > 120) throw new StudioException("render_limit", "The local renderer currently supports at most 120 combined clips and captions per render.");
        var fps = $"{p.FrameRate.Numerator}/{p.FrameRate.Denominator}";
        var duration = N(p.FrameRate.Seconds(p.DurationFrames));
        List<string> args = ["-hide_banner", "-nostdin", "-y", "-filter_complex_threads", "1", "-f", "lavfi", "-i", $"color=c=black:s={p.Width}x{p.Height}:r={fps}:d={duration}"];
        var graph = new StringBuilder("[0:v]format=yuv420p[base0];");
        var audioLabels = new List<string>(); var input = 1; var visual = 0;
        foreach (var track in p.Tracks)
        foreach (var clip in p.Clips.Where(c => c.TrackId == track.Id).OrderBy(c => c.StartFrame).ThenBy(c => c.Id, StringComparer.Ordinal))
        {
            var asset = p.Assets.Single(a => a.Id == clip.AssetId);
            bool show = track.Kind == "video" && track.Visible;
            bool hear = asset.HasAudio && !track.Muted && track.Volume > 0 && clip.Volume > 0;
            if (!show && !hear) continue;
            var path = assetPath(asset.Id);
            if (!File.Exists(path)) throw new StudioException("missing_media", $"Media '{asset.Name}' is missing; relink it before rendering.");
            args.AddRange(["-protocol_whitelist", "file,pipe", "-format_whitelist", Formats]);
            if (asset.Kind == "image") args.AddRange(["-loop", "1", "-framerate", fps]);
            args.AddRange(["-i", path]);
            var d = N(p.FrameRate.Seconds(clip.DurationFrames)); var source = N(p.FrameRate.Seconds(clip.SourceStartFrame));
            var fadeIn = N(p.FrameRate.Seconds(clip.FadeInFrames)); var fadeOut = N(p.FrameRate.Seconds(clip.FadeOutFrames)); var fadeStart = N(p.FrameRate.Seconds(clip.DurationFrames - clip.FadeOutFrames));
            if (show)
            {
                var scale = clip.Fit == "cover"
                    ? $"scale={p.Width}:{p.Height}:force_original_aspect_ratio=increase,crop={p.Width}:{p.Height}"
                    : $"scale={p.Width}:{p.Height}:force_original_aspect_ratio=decrease,pad={p.Width}:{p.Height}:(ow-iw)/2:(oh-ih)/2:color=black";
                // Sample source presentation intervals before trimming. Trimming raw VFR frames
                // by timestamp first would discard a frame whose display interval covers the
                // in-point and shift the first surviving frame (and thus video/audio sync).
                graph.Append($"[{input}:v]fps=fps={fps}:start_time=0:round=near,trim=start_frame={clip.SourceStartFrame}:end_frame={checked(clip.SourceStartFrame + clip.DurationFrames)},setpts=PTS-STARTPTS,{scale},setsar=1,format=yuva420p");
                if (clip.FadeInFrames > 0) graph.Append($",fade=t=in:st=0:d={fadeIn}:alpha=1");
                if (clip.FadeOutFrames > 0) graph.Append($",fade=t=out:st={fadeStart}:d={fadeOut}:alpha=1");
                // Frame-exact placement. Seconds printed to nine decimals round: 22/24 s became 0.916666667, later than the
                // base frame at 22/24 s, so a clip starting there lost its first frame to the black base, and 8/24 s became
                // 0.333333333, which truncated the offset a frame early and skipped the clip's first source frame. The offset
                // is rounded to whole ticks, and the overlay is bounded half a frame either side of its first and last frames.
                // Not by frame number: before FFmpeg 7.0 a timeline expression also ran on the overlay input's own frames,
                // whose n restarts at 0, which dropped a clip that does not start at frame 0 after its first frame.
                graph.Append($",setpts=PTS+round({clip.StartFrame}*{p.FrameRate.Denominator}/{p.FrameRate.Numerator}/TB)[v{input}];[base{visual}][v{input}]overlay=eof_action=pass:repeatlast=0:enable='{FrameWindow(p.FrameRate, clip.StartFrame, clip.DurationFrames)}'[base{visual + 1}];");
                visual++;
            }
            if (hear)
            {
                graph.Append($"[{input}:a]atrim=start={source}:duration={d},asetpts=PTS-STARTPTS,aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={N(track.Volume * clip.Volume)}");
                if (clip.FadeInFrames > 0) graph.Append($",afade=t=in:st=0:d={fadeIn}");
                var audioFadeFrames = Math.Max(clip.FadeOutFrames, clip.AudioFadeOutFrames);
                if (audioFadeFrames > 0) graph.Append($",afade=t=out:st={N(p.FrameRate.Seconds(clip.DurationFrames - audioFadeFrames))}:d={N(p.FrameRate.Seconds(audioFadeFrames))}");
                // Sample-based delay avoids millisecond rounding drift at fractional frame rates.
                var delaySamples = (long)Math.Round(p.FrameRate.Seconds(clip.StartFrame) * 48000, MidpointRounding.AwayFromZero);
                graph.Append($",adelay={delaySamples}S:all=1[a{input}];"); audioLabels.Add($"[a{input}]");
            }
            input++;
        }
        foreach (var caption in p.Captions)
        {
            var name = $"caption_{input}.png";
            StudioCaptionBitmap.Write(Path.Combine(workDirectory, name), caption.Text, p.Width, p.Height);
            args.AddRange(["-loop", "1", "-framerate", fps, "-i", Path.Combine(workDirectory, name)]);
            graph.Append($"[base{visual}][{input}:v]overlay=enable='{FrameWindow(p.FrameRate, caption.StartFrame, caption.DurationFrames)}':eof_action=pass[base{visual + 1}];");
            visual++; input++;
        }
        graph.Append($"[base{visual}]trim=duration={duration},format=yuv420p[outv];");
        if (audioLabels.Count > 0)
            graph.Append($"{string.Join("", audioLabels)}amix=inputs={audioLabels.Count}:duration=longest:normalize=0,alimiter=limit=0.95:latency=1,apad,atrim=duration={duration}[outa]");
        else graph.Append($"anullsrc=r=48000:cl=stereo,atrim=duration={duration}[outa]");
        args.AddRange(["-filter_complex", graph.ToString(), "-map", "[outv]", "-map", "[outa]", "-c:v", codec]);
        AddVideoEncodingArguments(args, codec, p.Width, p.Height, p.FrameRate.Value);
        args.AddRange(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-frames:v", p.DurationFrames.ToString(CultureInfo.InvariantCulture), "-t", duration, "-movflags", "+faststart", "-fs", options.MaxImportBytes.ToString(CultureInfo.InvariantCulture), "-progress", "pipe:1", "-nostats", Path.Combine(workDirectory, "output.mp4")]);
        return new StudioRenderPlan(p.Revision, p.DurationFrames, graph.ToString(), args);
    }

    /// <summary>A timeline enable expression true on frames start..start+duration-1 of the project clock: half a frame
    /// either side, in exact rational seconds, so no rounding of either bound can drop or add a frame.</summary>
    static string FrameWindow(StudioFrameRate rate, long start, long duration) =>
        $"gte(t,({start}-0.5)*{rate.Denominator}/{rate.Numerator})*lt(t,({checked(start + duration)}-0.5)*{rate.Denominator}/{rate.Numerator})";

    public async Task<string> RenderAsync(StudioProject p, Func<string, string> assetPath, string workDirectory, IProgress<StudioProgress> progress, CancellationToken ct)
    {
        var capability = await GetCapabilitiesAsync(ct);
        if (!capability.Available) throw new StudioException("render_unavailable", capability.Reason!);
        Directory.CreateDirectory(workDirectory);
        var plan = CreateRenderPlan(p, assetPath, workDirectory, capability.Codec!);
        progress.Report(new("rendering", 0.05));
        await RunAsync(options.FfmpegPath, plan.Arguments, workDirectory, ct, TimeSpan.FromHours(2), line =>
        {
            if (line.StartsWith("out_time_us=", StringComparison.Ordinal) && long.TryParse(line.AsSpan(12), out var us))
                progress.Report(new("rendering", Math.Clamp((us / 1_000_000d) / p.FrameRate.Seconds(p.DurationFrames), 0, 0.94)));
        });
        progress.Report(new("validating", 0.95));
        var result = Path.Combine(workDirectory, "output.mp4");
        var metadata = await ProbeAsync(result, "Rendered video", ct);
        if (Math.Abs(metadata.DurationSeconds - p.FrameRate.Seconds(p.DurationFrames)) > p.FrameRate.Seconds(2) + .05 || metadata.Width != p.Width || metadata.Height != p.Height)
            throw new StudioException("render_validation_failed", "Rendered dimensions or duration do not match the captured project.");
        // Decode every output frame and audio packet; existence alone never means a successful export.
        await RunAsync(options.FfmpegPath, ["-v", "error", "-xerror", "-nostdin", "-protocol_whitelist", "file,pipe", "-i", result, "-f", "null", "-"], workDirectory, ct, TimeSpan.FromMinutes(30));
        return result;
    }

    static string? S(JsonElement e, string key) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) ? v.ToString() : null;
    static double D(JsonElement e, string key) => double.TryParse(S(e, key), NumberStyles.Float, CultureInfo.InvariantCulture, out var n) ? n : 0;
    static int I(JsonElement e, string key) => int.TryParse(S(e, key), out var n) ? n : 0;

    internal static async Task<string> RunAsync(string executable, IEnumerable<string> arguments, string? workingDirectory, CancellationToken ct, TimeSpan timeout, Action<string>? onLine = null)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(ct); linked.CancelAfter(timeout);
        using var process = new Process { StartInfo = new ProcessStartInfo(executable) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true, WorkingDirectory = workingDirectory ?? Environment.CurrentDirectory } };
        foreach (var argument in arguments) process.StartInfo.ArgumentList.Add(argument);
        try { process.Start(); }
        catch (Exception e) { throw new StudioException("media_tool_unavailable", $"Unable to start {Path.GetFileName(executable)}: {e.Message}"); }
        var stdout = new StringBuilder(); var stderr = new StringBuilder();
        Exception? callbackFailure = null;
        async Task Read(StreamReader reader, StringBuilder buffer, Action<string>? callback)
        {
            while (await reader.ReadLineAsync(CancellationToken.None) is { } line)
            {
                if (callbackFailure == null && callback != null)
                {
                    try { callback(line); }
                    catch (Exception e) { Interlocked.CompareExchange(ref callbackFailure, e, null); linked.Cancel(); }
                }
                if (buffer.Length < 128 * 1024) buffer.AppendLine(line.Length > 8192 ? line[..8192] : line);
            }
        }
        var outTask = Read(process.StandardOutput, stdout, onLine); var errTask = Read(process.StandardError, stderr, null);
        try { await process.WaitForExitAsync(linked.Token); }
        catch (OperationCanceledException)
        {
            try { if (!process.HasExited) process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            await process.WaitForExitAsync(CancellationToken.None); await Task.WhenAll(outTask, errTask);
            if (callbackFailure != null) throw new StudioException("progress_failed", callbackFailure.Message);
            if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
            throw new StudioException("media_timeout", $"{Path.GetFileName(executable)} exceeded its time limit.");
        }
        await Task.WhenAll(outTask, errTask);
        if (callbackFailure != null) throw new StudioException("progress_failed", callbackFailure.Message);
        if (process.ExitCode != 0)
        {
            var error = stderr.ToString();
            throw new StudioException("media_failed", $"{Path.GetFileName(executable)} exited with code {process.ExitCode}: {error[Math.Max(0, error.Length - 6000)..]}");
        }
        return stdout.ToString();
    }
}
