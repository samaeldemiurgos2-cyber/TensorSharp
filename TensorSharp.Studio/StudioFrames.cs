// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;
using System.Text.Json;

namespace TensorSharp.Studio;

/// <summary>Stills taken from videos in the library, so a long video can be made as a chain: the next segment starts on
/// the previous segment's last frame (or on the frame a timeline clip ends on). The frame becomes an ordinary image asset
/// that records its source (<see cref="StudioAsset.DerivedFrom"/>), so generation requests, fingerprints, job snapshots,
/// packages and the model adapters treat it exactly as a photo.</summary>
public sealed partial class StudioService
{
    // One decoder at a time, apart from thumbnails and waveforms, which can run for minutes.
    readonly SemaphoreSlim frameGate = new(1, 1);
    const string FrameFormats = "mov,matroska,avi,wav,mp3,flac,ogg,image2,png_pipe,jpeg_pipe,webp_pipe,gif,aiff";

    /// <summary>Puts frame <paramref name="frame"/> of video <paramref name="assetId"/> in the library as a still
    /// (<see cref="StudioFrameSource.LastFrame"/> for its last frame, else frame N on the project clock as the timeline shows
    /// it) and returns its id. A frame taken before is returned as it is. A take's last frame is its own lossless frame when
    /// the take kept one; any other frame is decoded from the video file. Like an import, this adds to the library without
    /// an Undo step and moves the project to a new revision.</summary>
    public async Task<StudioFrameExtraction> ExtractFrameAsync(string projectId, string assetId, long frame, CancellationToken cancellationToken = default)
    {
        if (frame < StudioFrameSource.LastFrame || frame > Options.MaxTimelineFrames)
            throw new StudioException("invalid_frame", "Choose the video's last frame, or a frame from 0 within the supported timeline range.");
        var project = await GetProjectAsync(projectId, cancellationToken);
        var source = project.Assets.FirstOrDefault(a => a.Id == StudioPaths.Id(assetId)) ?? throw new StudioException("asset_not_found", "Asset does not belong to this project.");
        if (source.Kind != "video")
            throw new StudioException("invalid_frame_source", source.Kind == "image"
                ? "A photo can be the first frame as it is; only a video has frames to continue from."
                : "Audio has no picture to continue from. Choose a video.");
        var selector = new StudioFrameSource { AssetId = source.Id, Name = source.Name, Sha256 = source.Sha256, Frame = frame,
            FrameRate = frame == StudioFrameSource.LastFrame ? null : project.FrameRate };
        if (FindStill(project, selector) is { } existing) return new(project, existing.Id, true);
        string sourcePath = GetAssetPath(projectId, source.Id);
        // Even a take's kept frame is probed on import, as every library asset is.
        if (!Options.EnableExternalProcesses)
            throw new StudioException("unsupported_platform", "Taking a frame from a video needs the desktop media decoder (FFmpeg) on this host. " +
                "On this device Studio reviews and edits projects; taking frames, importing and generating are unavailable.");
        string? kept = frame == StudioFrameSource.LastFrame ? KeptLastFrame(projectId, source) : null;
        CheckDisk();
        var scratch = StudioPaths.SafeChild(RootPath, "imports", StudioJson.NewId());
        Directory.CreateDirectory(scratch);
        try
        {
            string still;
            if (kept != null) { selector.Method = StudioFrameSource.GenerationMethod; still = kept; }
            else
            {
                selector.Method = StudioFrameSource.DecodeMethod;
                await frameGate.WaitAsync(cancellationToken);
                try { still = await DecodeFrameAsync(source, sourcePath, frame, project.FrameRate, scratch, cancellationToken); }
                finally { frameGate.Release(); }
            }
            string baseName = Path.GetFileNameWithoutExtension(source.Name);
            if (baseName.Length > 150) baseName = baseName[..150];
            string name = baseName + (frame == StudioFrameSource.LastFrame ? " · last frame" : $" · frame {frame.ToString(CultureInfo.InvariantCulture)}") + ".png";
            await using var stream = File.OpenRead(still);
            var imported = await ImportAssetInternalAsync(projectId, stream, name, null, cancellationToken, selector);
            return new(imported.Project, imported.AssetId, imported.Existing);
        }
        finally { try { Directory.Delete(scratch, true); } catch (IOException) { } catch (UnauthorizedAccessException) { } }
    }

    /// <summary>A still in <paramref name="project"/> that is the frame <paramref name="selector"/> names, with its file present.</summary>
    StudioAsset? FindStill(StudioProject project, StudioFrameSource selector) => project.Assets.FirstOrDefault(a =>
        a.Kind == "image" && a.DerivedFrom is { } d && d.AssetId == selector.AssetId && d.Sha256 == selector.Sha256 && d.Frame == selector.Frame &&
        Equals(d.FrameRate, selector.FrameRate) && File.Exists(StudioPaths.SafeChild(ProjectDirectory(project.Id), "assets", a.FileName)));

    /// <summary>The lossless last frame the job that made <paramref name="take"/> kept, or null.</summary>
    string? KeptLastFrame(string projectId, StudioAsset take)
    {
        if (take.Generation == null) return null;
        lock (jobSync)
        {
            var record = jobs.Values.FirstOrDefault(r => r.Job.ProjectId == projectId && r.Job.Kind == "generate" && r.Job.State == "completed"
                && r.Job.OutputAssetId == take.Id && r.LastFrameFileName != null);
            if (record == null) return null;
            var path = StudioPaths.SafeChild(JobDirectory(record.Job.Id), record.LastFrameFileName!);
            return File.Exists(path) ? path : null;
        }
    }

    /// <summary>Decodes one frame of a video into a full-size PNG in <paramref name="scratch"/>, upright (the stream's
    /// rotation applied) with square pixels. The last frame is the last frame the decoder shows, edit lists honoured.
    /// Frame N is sampled exactly as the timeline renderer samples the video at the project rate.</summary>
    async Task<string> DecodeFrameAsync(StudioAsset source, string path, long frame, StudioFrameRate rate, string scratch, CancellationToken ct)
    {
        var stream = await ProbeVideoStreamAsync(path, ct);
        // Pixels are copied, never re-mapped: a tone map or a deinterlacer would be a choice made for the person.
        if (stream.Transfer is "smpte2084" or "arib-std-b67")
            throw new StudioException("unsupported_frame_source", "This video is HDR (PQ or HLG). Studio does not tone-map frames; export an SDR version and continue from that.");
        if (stream.FieldOrder is "tt" or "bb" or "tb" or "bt")
            throw new StudioException("unsupported_frame_source", "This video is interlaced. Studio does not deinterlace frames; export a progressive version and continue from that.");
        // Non-square pixels are widened to square, the way a player shows them; the decoders that read the still ignore aspect metadata.
        string square = stream.SquarePixels ? "" : "scale='trunc(iw*sar/2+0.5)*2':ih:flags=lanczos,setsar=1,";
        string output = Path.Combine(scratch, "frame.png");
        IEnumerable<string> Common(IEnumerable<string> seek, string filter) =>
            new[] { "-v", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-format_whitelist", FrameFormats }
                .Concat(seek).Concat(new[] { "-i", path, "-map", "0:V:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-vf", filter,
                    "-c:v", "png", "-f", "image2", "-update", "1" });
        var timeout = TimeSpan.FromMinutes(Math.Clamp(source.DurationSeconds / 30, 2, 30));
        if (frame == StudioFrameSource.LastFrame)
        {
            // Every frame from the seek point on overwrites the file, so the frame left is the last one decoded. Seeking
            // counts from the container's end, which a longer soundtrack moves past the picture's end: seek from the video
            // stream's own end when it is known, then from the container's end, then decode it all.
            var attempts = new List<string[]>();
            if (stream.SeekEnd is double end) attempts.Add(["-ss", Math.Max(0, end - 2).ToString("0.######", CultureInfo.InvariantCulture)]);
            attempts.Add(["-sseof", "-2"]);
            attempts.Add([]);
            for (int i = 0; i < attempts.Count; i++)
            {
                if (File.Exists(output)) File.Delete(output);
                // A seek that lands past the picture decodes nothing; the next attempt starts earlier. The last decodes it all.
                try { await StudioMedia.RunAsync(Options.FfmpegPath, Common(attempts[i], square + "format=rgb24").Append(output), scratch, ct, timeout); }
                catch (StudioException e) when (e.Code == "media_failed" && i < attempts.Count - 1) { continue; }
                if (StillWritten(output)) return output;
            }
            throw new StudioException("frame_extraction_failed", "The video's last frame could not be decoded.");
        }
        // Frame N of a clip is the frame the renderer shows there, so the fps grid is built from the start, unseeked.
        string sample = $"fps=fps={rate.Numerator}/{rate.Denominator}:start_time=0:round=near,trim=start_frame={frame}:end_frame={frame + 1},";
        await StudioMedia.RunAsync(Options.FfmpegPath, Common([], sample + square + "format=rgb24").Concat(new[] { "-frames:v", "1", output }), scratch, ct, timeout);
        if (StillWritten(output)) return output;
        throw new StudioException("invalid_frame", $"Frame {frame} is past the end of this video's picture. Choose an earlier frame, or the last frame.");
    }

    static bool StillWritten(string path) => File.Exists(path) && new FileInfo(path).Length is >= 8 and <= 1024L * 1024 * 1024;

    /// <param name="SeekEnd">Where the picture ends, in the seconds an input <c>-ss</c> counts (from the file's start
    /// time); null when the file does not say.</param>
    sealed record VideoStream(double? SeekEnd, bool SquarePixels, string? Transfer, string? FieldOrder);

    async Task<VideoStream> ProbeVideoStreamAsync(string path, CancellationToken ct)
    {
        var json = await StudioMedia.RunAsync(Options.FfprobePath, ["-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", FrameFormats,
            "-select_streams", "V:0", "-show_entries", "stream=width,height,sample_aspect_ratio,color_transfer,field_order,start_time,duration:stream_side_data=rotation:stream_tags=DURATION:format=start_time",
            "-of", "json", path], null, ct, TimeSpan.FromSeconds(30));
        using var doc = JsonDocument.Parse(json);
        var stream = doc.RootElement.TryGetProperty("streams", out var streams) ? streams.EnumerateArray().FirstOrDefault() : default;
        if (stream.ValueKind != JsonValueKind.Object) throw new StudioException("invalid_frame_source", "This file has no video picture to take a frame from.");
        string? Text(JsonElement e, string key) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind is JsonValueKind.String or JsonValueKind.Number ? v.ToString() : null;
        static double? Number(string? value) => double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var n) && double.IsFinite(n) ? n : null;
        double? streamStart = Number(Text(stream, "start_time")), duration = Number(Text(stream, "duration"));
        double? end = duration is double d && d > 0 ? (streamStart ?? 0) + d : null;
        // Matroska writes the stream's end as an HH:MM:SS.fraction tag, counted on the file's own clock (start offset included).
        if (end == null && stream.TryGetProperty("tags", out var tags) && Text(tags, "DURATION") is { } tag && TimeSpan.TryParse(tag[..Math.Min(tag.Length, 16)], CultureInfo.InvariantCulture, out var span))
            end = span.TotalSeconds;
        // An input -ss counts from the file's start time, not zero.
        double? fileStart = Number(Text(doc.RootElement.TryGetProperty("format", out var format) ? format : default, "start_time")) ?? streamStart;
        double? seekEnd = end is double e ? Math.Max(0, e - (fileStart ?? 0)) : null;
        // Non-square pixels (a stated ratio n:d, n != d) are widened before the still is checked, so the widened size is
        // bounded here, with the real reason. "0:1", "N/A" or no ratio mean square.
        string sar = Text(stream, "sample_aspect_ratio") ?? "1:1";
        bool square = true;
        if (sar.Split(':') is [var ns, var ds] && long.TryParse(ns, out long sn) && long.TryParse(ds, out long sd) && sn > 0 && sd > 0 && sn != sd)
        {
            square = false;
            double ratio = (double)sn / sd, codedWidth = Number(Text(stream, "width")) ?? 0, codedHeight = Number(Text(stream, "height")) ?? 0;
            // A quarter-turn display matrix is applied first: the frame is transposed and its pixel ratio inverted.
            bool quarterTurn = stream.TryGetProperty("side_data_list", out var sides) && sides.ValueKind == JsonValueKind.Array && sides.EnumerateArray()
                .Any(side => Number(Text(side, "rotation")) is double degrees && Math.Abs(Math.Round(degrees / 90)) % 2 == 1);
            double width = quarterTurn ? codedHeight / ratio : codedWidth * ratio, height = quarterTurn ? codedWidth : codedHeight;
            if (quarterTurn) ratio = 1 / ratio;
            if (ratio is < .25 or > 4 || width > 16384 || width * height > 100_000_000)
                throw new StudioException("unsupported_frame_source", $"This video's pixel aspect ratio ({sar}) would make its frames {Math.Round(width)} × {height} on square pixels, beyond what Studio takes. " +
                    "Export it with square pixels and continue from that.");
        }
        return new(seekEnd, square, Text(stream, "color_transfer"), Text(stream, "field_order"));
    }
}
