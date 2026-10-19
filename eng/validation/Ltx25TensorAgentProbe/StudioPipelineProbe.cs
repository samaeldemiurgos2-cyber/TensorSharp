using System.Diagnostics;
using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;
using TensorSharp.Studio;

/// <summary>Real model generation and the same HTTP import, timeline and export operations as Video Studio.</summary>
internal static class StudioPipelineProbe
{
    public static async Task<object> RunAsync(HttpClient client, string workspace, string prompt, string mode,
        int width, int height, int frames, int fps, long seed, string firstImage, string endImage,
        string[] referenceImages, Action<string> stage)
    {
        using var stop = new CancellationTokenSource(TimeSpan.FromMinutes(30));
        var ct = stop.Token;
        client.DefaultRequestHeaders.Add("X-TensorSharp-Studio", "1");
        var requests = new List<object>();
        async Task<JsonElement> Api(string action, object payload)
        {
            var body = new { action, payload };
            using var response = await client.PostAsJsonAsync("/api/studio", body, StudioJson.Options, ct);
            string text = await response.Content.ReadAsStringAsync(ct);
            if (!response.IsSuccessStatusCode) throw new InvalidOperationException($"Studio {action}: {(int)response.StatusCode}: {text}");
            using var parsed = JsonDocument.Parse(text);
            requests.Add(new { action, payload, response = parsed.RootElement.Clone() });
            return parsed.RootElement.Clone();
        }
        T Read<T>(JsonElement value) => value.Deserialize<T>(StudioJson.Options) ?? throw new InvalidDataException("Missing Studio response.");
        async Task<StudioProject> Project(string id) => Read<StudioProject>(await Api("projects/get", new { projectId = id }));
        async Task<string> Download(string url, string name)
        {
            string path = Path.Combine(workspace, name);
            await File.WriteAllBytesAsync(path, await client.GetByteArrayAsync(url, ct), ct);
            if (new FileInfo(path).Length == 0) throw new InvalidDataException("Empty Studio media response: " + url);
            return path;
        }

        stage("Studio media and model capability preflight");
        var capability = await Api("capabilities", new { });
        foreach (string flag in new[] { "externalProcesses", "importAvailable", "renderAvailable", "generationAvailable" })
            if (!capability.GetProperty(flag).GetBoolean()) throw new NotSupportedException("Studio capability unavailable: " + flag + "; " + capability);
        var model = Read<StudioModelCapability>(capability.GetProperty("models").EnumerateArray().Single());
        if (!model.Available || model.Family != "ltx-2.5" || model.FixedSteps != 8 || model.FixedCfgScale != 1
            || model.DefaultWidth != 1024 || model.DefaultHeight != 1024 || !model.DefaultGenerateAudio)
            throw new InvalidDataException("Studio did not preserve the loaded LTX high-quality generation defaults: " + capability);
        var project = Read<StudioProject>(await Api("projects/create", new { name = "Real LTX Studio validation", width, height, fpsNumerator = fps, fpsDenominator = 1 }));

        stage("Studio HTTP media import and byte-preserving download");
        var imports = new List<object>();
        async Task<string> Import(string source)
        {
            string absolute = Path.GetFullPath(source);
            byte[] bytes = await File.ReadAllBytesAsync(absolute, ct);
            using var content = new ByteArrayContent(bytes);
            content.Headers.ContentType = new("application/octet-stream");
            using var response = await client.PostAsync($"/api/studio/projects/{project.Id}/import?fileName={Uri.EscapeDataString(Path.GetFileName(absolute))}", content, ct);
            string text = await response.Content.ReadAsStringAsync(ct);
            if (!response.IsSuccessStatusCode) throw new InvalidOperationException("Studio import: " + text);
            using var parsed = JsonDocument.Parse(text);
            var updated = Read<StudioProject>(parsed.RootElement);
            var asset = updated.Assets.Single(a => a.Sha256 == Hash(absolute));
            project = updated;
            string served = await Download($"/api/studio/projects/{project.Id}/assets/{asset.Id}", $"import-{asset.Id}{Path.GetExtension(absolute)}");
            if (Hash(served) != Hash(absolute) || asset.Sha256 != Hash(absolute)) throw new InvalidDataException("Studio import changed source bytes.");
            string thumbnail = await Download($"/api/studio/thumbnail/{project.Id}/{asset.Id}", $"import-{asset.Id}.jpg");
            if (ImageIO.Load(thumbnail).Width <= 0) throw new InvalidDataException("Imported asset thumbnail is unreadable.");
            imports.Add(new { source = absolute, source_sha256 = Hash(absolute), asset, served, thumbnail });
            return asset.Id;
        }
        // Even text-only generation tests the originally failing import route with an actual encoded image.
        if (string.IsNullOrWhiteSpace(firstImage) && string.IsNullOrWhiteSpace(endImage) && referenceImages.Length == 0)
        {
            string fixture = Path.Combine(workspace, "studio-import-fixture.png");
            var pixels = Enumerable.Range(0, 96 * 64 * 3).Select(i => (i % 251) / 250f).ToArray();
            ImageIO.SavePng(fixture, new(96, 64, pixels));
            await Import(fixture);
        }
        string? first = string.IsNullOrWhiteSpace(firstImage) ? null : await Import(firstImage);
        string? end = string.IsNullOrWhiteSpace(endImage) ? null : await Import(endImage);
        var refs = new List<string>();
        foreach (string image in referenceImages) refs.Add(await Import(image));
        var request = new StudioGenerationRequest
        {
            ProjectId = project.Id, ModelId = model.ModelId, Prompt = prompt, Seed = seed,
            Width = width, Height = height, FrameCount = frames, Fps = fps, Steps = 8, CfgScale = 1, GenerateAudio = true,
            Mode = string.IsNullOrWhiteSpace(mode) ? refs.Count > 0 ? "ref" : end != null ? "fl2v" : first != null ? "i2v" : "t2v" : mode,
            ConditioningAssetId = first, EndConditioningAssetId = end, ReferenceAssetIds = refs,
        };
        var progress = new List<StudioJob>();
        async Task<StudioJob> Complete(StudioJob job)
        {
            string? previous = null;
            try
            {
                while (true)
                {
                    ct.ThrowIfCancellationRequested();
                    using var response = await client.PostAsJsonAsync("/api/studio", new { action = "jobs/list", payload = new { projectId = project.Id } }, StudioJson.Options, ct);
                    response.EnsureSuccessStatusCode();
                    var jobs = await response.Content.ReadFromJsonAsync<StudioJob[]>(StudioJson.Options, ct);
                    job = jobs!.Single(j => j.Id == job.Id);
                    string current = $"{job.State}|{job.Stage}|{job.Step}|{job.TotalSteps}";
                    if (current != previous)
                    {
                        progress.Add(job); previous = current;
                        Console.WriteLine($"Studio {job.Kind}: {job.State}, {job.Stage}, {job.Step}/{job.TotalSteps}: {job.Detail}");
                    }
                    if (job.State == "completed") return job;
                    if (job.State is "failed" or "cancelled" or "interrupted") throw new InvalidOperationException("Studio job " + job.State + ": " + job.Error);
                    await Task.Delay(1000, ct);
                }
            }
            catch
            {
                // Join cancellation before the host is disposed, using a separate bounded token after a probe timeout.
                if (job.State is "queued" or "running" or "cancelling")
                {
                    try
                    {
                        using var cleanup = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                        using var cancel = await client.PostAsJsonAsync("/api/studio", new { action = "jobs/cancel", payload = new { jobId = job.Id } }, StudioJson.Options, cleanup.Token);
                        cancel.EnsureSuccessStatusCode();
                        while (true)
                        {
                            using var pending = await client.PostAsJsonAsync("/api/studio", new { action = "jobs/list", payload = new { projectId = project.Id } }, StudioJson.Options, cleanup.Token);
                            pending.EnsureSuccessStatusCode();
                            var jobs = await pending.Content.ReadFromJsonAsync<StudioJob[]>(StudioJson.Options, cleanup.Token);
                            if (jobs!.Single(j => j.Id == job.Id).State is not ("queued" or "running" or "cancelling")) break;
                            await Task.Delay(250, cleanup.Token);
                        }
                    }
                    catch (Exception cleanupError) { Console.Error.WriteLine("Studio cancellation cleanup did not complete: " + cleanupError.Message); }
                }
                throw;
            }
        }

        stage("Studio queued native LTX generation and automatic asset ingestion");
        var generationTimer = Stopwatch.StartNew();
        var generation = await Complete(Read<StudioJob>(await Api("jobs/generate", request)));
        generationTimer.Stop();
        project = await Project(project.Id);
        var generated = project.Assets.Single(a => a.Id == generation.OutputAssetId);
        if (generated.Kind != "video" || !generated.HasAudio || generated.Generation?.Request.Seed != seed
            || generated.Width != width || generated.Height != height || generated.Generation.ModelVersion.Contains("sha256=", StringComparison.Ordinal) != true)
            throw new InvalidDataException("Generated Studio asset or provenance is incomplete.");
        string generatedPath = await Download($"/api/studio/projects/{project.Id}/assets/{generated.Id}", "studio-generated.mp4");
        if (Hash(generatedPath) != generated.Sha256) throw new InvalidDataException("Generated Studio asset download hash differs.");
        string videoThumbnail = await Download($"/api/studio/thumbnail/{project.Id}/{generated.Id}", "studio-generated.jpg");
        if (ImageIO.Load(videoThumbnail).Width <= 0) throw new InvalidDataException("Generated video preview is unreadable.");
        var frame = await Api("assets/extract-frame", new { projectId = project.Id, assetId = generated.Id, frame = "last" });
        project = Read<StudioProject>(frame.GetProperty("project"));
        var last = project.Assets.Single(a => a.Id == frame.GetProperty("assetId").GetString());
        if (last.DerivedFrom?.Method != StudioFrameSource.GenerationMethod || last.Width != width || last.Height != height)
            throw new InvalidDataException("Studio lost the native lossless continuation frame.");
        string lastFrame = await Download($"/api/studio/projects/{project.Id}/assets/{last.Id}", "studio-last-frame.png");
        var lastPixels = ImageIO.Load(lastFrame);
        if (lastPixels.Width != width || lastPixels.Height != height) throw new InvalidDataException("Native last-frame image dimensions differ.");

        stage("Studio timeline insertion, captured revision and HTTP H.264/AAC export");
        project = Read<StudioProject>(await Api("commands/apply", new { projectId = project.Id, expectedRevision = project.Revision,
            commandId = StudioJson.NewId(), commands = new[] { new { type = "insertClip", assetId = generated.Id,
                trackId = project.Tracks.First(t => t.Kind == "video").Id, startFrame = 0, durationFrames = frames } } }));
        var renderTimer = Stopwatch.StartNew();
        var render = await Complete(Read<StudioJob>(await Api("jobs/render", new { projectId = project.Id, expectedRevision = project.Revision, requestId = StudioJson.NewId() })));
        renderTimer.Stop();
        if (render.ProjectRevision != project.Revision) throw new InvalidDataException("Export captured another project revision.");
        string exported = await Download($"/api/studio/projects/{project.Id}/jobs/{render.Id}/output", "studio-export.mp4");

        stage("Full decode of generated and exported RGB frames and stereo PCM");
        var sourceDecode = await DecodeAsync(generatedPath, width, height, frames, fps, ct);
        var exportDecode = await DecodeAsync(exported, width, height, frames, fps, ct);
        double psnr = ComparePixels(sourceDecode.Rgb, exportDecode.Rgb);
        if (psnr < 25) throw new InvalidDataException($"Timeline export changed the unedited generated picture excessively: PSNR {psnr:F2} dB.");
        double correlation = AudioCorrelation(sourceDecode.Pcm, exportDecode.Pcm);
        if (correlation < .9) throw new InvalidDataException($"Timeline export changed the generated soundtrack excessively: correlation {correlation:F4}.");
        return new
        {
            status = "passed", scope = "Real AgentAppHost Studio HTTP capabilities, import, native LTX generation, automatic library ingestion, video thumbnail, lossless last-frame extraction, timeline insertion, captured-revision H.264/AAC export, complete RGB and PCM decode.",
            limitations = "CoreCLR HTTP host only; excludes native MAUI controls, iOS, CUDA, long clips and subjective perceptual quality. File/JIT caches are uncontrolled; timings include checkpoint provenance hashing, encoding and Studio ingestion rather than matched inference benchmarks.",
            capability, request, imports, requests, progress, project, generation, render,
            generation_http_job_ms = generationTimer.Elapsed.TotalMilliseconds, render_http_job_ms = renderTimer.Elapsed.TotalMilliseconds,
            generatedPath, generated_sha256 = Hash(generatedPath), exported, exported_sha256 = Hash(exported), videoThumbnail, lastFrame,
            source_decode = sourceDecode, export_decode = exportDecode, export_pixel_psnr_db = psnr, export_audio_correlation = correlation,
        };
    }

    sealed record DecodeResult(string Rgb, string Pcm, int Frames, int AudioSamples, double AudioRms, int MinPixel, int MaxPixel, JsonElement Metadata);
    static async Task<DecodeResult> DecodeAsync(string path, int width, int height, int frames, int fps, CancellationToken ct)
    {
        string ffmpeg = MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG") ?? throw new FileNotFoundException("ffmpeg unavailable.");
        string ffprobe = MediaExecutableLocator.Find("ffprobe", "TS_FFPROBE", ffmpeg) ?? throw new FileNotFoundException("ffprobe unavailable.");
        string metadataText = await RunAsync(ffprobe, ["-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", path], ct);
        using var parsed = JsonDocument.Parse(metadataText);
        var metadata = parsed.RootElement.Clone();
        var video = metadata.GetProperty("streams").EnumerateArray().Single(s => s.GetProperty("codec_type").GetString() == "video");
        var audio = metadata.GetProperty("streams").EnumerateArray().Single(s => s.GetProperty("codec_type").GetString() == "audio");
        if (video.GetProperty("codec_name").GetString() != "h264" || video.GetProperty("width").GetInt32() != width || video.GetProperty("height").GetInt32() != height
            || int.Parse(video.GetProperty("nb_read_frames").GetString()!, CultureInfo.InvariantCulture) != frames
            || video.GetProperty("r_frame_rate").GetString() != $"{fps}/1" || audio.GetProperty("codec_name").GetString() != "aac"
            || audio.GetProperty("sample_rate").GetString() != "48000" || audio.GetProperty("channels").GetInt32() != 2)
            throw new InvalidDataException("Studio MP4 stream geometry or codecs differ: " + metadata);
        string rgb = path + ".rgb24", pcm = path + ".stereo.f32";
        await RunAsync(ffmpeg, ["-v", "error", "-xerror", "-nostdin", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", rgb,
            "-map", "0:a:0", "-acodec", "pcm_f32le", "-f", "f32le", pcm], ct);
        if (new FileInfo(rgb).Length != (long)width * height * frames * 3) throw new InvalidDataException("RGB decode frame count differs.");
        int min = 255, max = 0;
        using (var file = File.OpenRead(rgb))
        {
            var buffer = new byte[1024 * 1024]; int count;
            while ((count = file.Read(buffer)) > 0)
                for (int i = 0; i < count; i++) { min = Math.Min(min, buffer[i]); max = Math.Max(max, buffer[i]); }
        }
        if (max - min < 8) throw new InvalidDataException("Generated video has no meaningful pixel range.");
        float[] samples = MemoryMarshal.Cast<byte, float>(await File.ReadAllBytesAsync(pcm, ct)).ToArray();
        double duration = frames / (double)fps;
        if (samples.Any(value => !float.IsFinite(value)) || samples.Length < Math.Max(0, duration - .15) * 48000 * 2
            || samples.Length > (duration + .15) * 48000 * 2) throw new InvalidDataException("Stereo PCM decode has invalid samples or duration.");
        double rms = Math.Sqrt(samples.Average(value => value * (double)value));
        if (rms < 1e-6) throw new InvalidDataException("Generated soundtrack is silent.");
        return new(rgb, pcm, frames, samples.Length / 2, rms, min, max, metadata);
    }

    static double ComparePixels(string source, string output)
    {
        using var a = File.OpenRead(source); using var b = File.OpenRead(output);
        if (a.Length != b.Length) throw new InvalidDataException("Exported RGB geometry changed.");
        var left = new byte[1024 * 1024]; var right = new byte[left.Length];
        double squared = 0; int count;
        while ((count = a.Read(left)) > 0)
        {
            b.ReadExactly(right.AsSpan(0, count));
            for (int i = 0; i < count; i++) { int delta = left[i] - right[i]; squared += delta * delta; }
        }
        return squared == 0 ? 999 : 10 * Math.Log10(255d * 255 * a.Length / squared);
    }

    static double AudioCorrelation(string source, string output)
    {
        float[] a = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(source)).ToArray();
        float[] b = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(output)).ToArray();
        int count = Math.Min(a.Length, b.Length); double dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < count; i++) { dot += a[i] * (double)b[i]; aa += a[i] * (double)a[i]; bb += b[i] * (double)b[i]; }
        return dot / Math.Sqrt(aa * bb);
    }

    static async Task<string> RunAsync(string executable, string[] arguments, CancellationToken ct)
    {
        using var process = new Process { StartInfo = new(executable) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true } };
        foreach (string arg in arguments) process.StartInfo.ArgumentList.Add(arg);
        process.Start();
        using var onCancel = ct.Register(() => { try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { } });
        var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync();
        try { await process.WaitForExitAsync(ct); }
        catch (OperationCanceledException)
        {
            try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
            await process.WaitForExitAsync(CancellationToken.None);
            await Task.WhenAll(stdout, stderr);
            throw;
        }
        string result = await stdout, error = await stderr;
        if (process.ExitCode != 0) throw new InvalidOperationException($"{Path.GetFileName(executable)} exited {process.ExitCode}: {error}");
        return result;
    }
    static string Hash(string path) { using var input = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(input)); }
}
