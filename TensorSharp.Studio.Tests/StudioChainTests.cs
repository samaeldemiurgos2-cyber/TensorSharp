// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

/// <summary>Chains (a long video made segment by segment, each starting on the previous one's last frame) and reference
/// photos beside a first frame. Frames are compared bit for bit with what FFmpeg's decoder (and the timeline renderer)
/// shows; a test double stands in for the model, so no inference runs.</summary>
public sealed class StudioChainTests : IAsyncLifetime
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-chain-" + StudioJson.NewId());
    StudioService service = null!;
    ChainAdapter adapter = null!;
    public Task InitializeAsync() { adapter = new ChainAdapter(); service = new StudioService(Path.Combine(root, "studio"), adapter); return Task.CompletedTask; }
    public async Task DisposeAsync() { await service.DisposeAsync(); if (Directory.Exists(root)) Directory.Delete(root, true); }

    /// <summary>Copies a fixture video as the take and, when asked, writes a "lossless last frame" that differs from the
    /// video's own last frame, so a test can tell which one a chain started on.</summary>
    sealed class ChainAdapter : IStudioGenerationAdapter
    {
        public string? Video, LastFrame; public int Calls; public StudioGenerationRequest? Validated, Generated;
        public StudioModelCapability Capability = new()
        {
            ModelId = "chain-double", Family = "chain-family", Available = true, TextToVideo = true, ImageConditioning = true, LastFrameConditioning = true,
            IdentityReferences = true, MaxReferences = 4, DefaultWidth = 64, DefaultHeight = 64, DefaultFrameCount = 17, MaxDimension = 1024,
        };
        public Task<IReadOnlyList<StudioModelCapability>> GetCapabilitiesAsync(CancellationToken ct) => Task.FromResult<IReadOnlyList<StudioModelCapability>>([Capability]);
        public Task ValidateAsync(StudioGenerationRequest r, IReadOnlyDictionary<string, string> assets, CancellationToken ct) { Validated = StudioJson.Clone(r); return Task.CompletedTask; }
        public Task<StudioGenerationResult> GenerateAsync(StudioGenerationRequest r, string dir, IReadOnlyDictionary<string, string> assets, IProgress<StudioProgress> progress, CancellationToken ct)
        {
            Calls++; Generated = StudioJson.Clone(r); Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, "generated.mp4"); File.Copy(Video!, path);
            string? last = null;
            if (LastFrame != null) { last = Path.Combine(dir, "last-frame.png"); File.Copy(LastFrame, last); }
            return Task.FromResult(new StudioGenerationResult(path, "TEST DOUBLE; no model inference", new(), last));
        }
    }

    static bool MediaInstalled() => new StudioMediaFactAttribute().Skip == null;
    static async Task<string> Run(string tool, params string[] args)
    {
        using var p = new Process { StartInfo = new(tool) { UseShellExecute = false, RedirectStandardError = true, RedirectStandardOutput = true } };
        foreach (var a in args) p.StartInfo.ArgumentList.Add(a); p.Start();
        var stdout = p.StandardOutput.ReadToEndAsync(); var stderr = p.StandardError.ReadToEndAsync();
        await p.WaitForExitAsync(); Assert.True(p.ExitCode == 0, await stderr); return await stdout;
    }
    // The MD5 of every frame the decoder shows, in order (rgb24, as a still is stored).
    static async Task<string[]> FrameMd5s(string path, params string[] filter) =>
        (await Run("ffmpeg", ["-v", "error", "-i", path, "-map", "0:V:0", "-fps_mode", "passthrough", .. filter, "-pix_fmt", "rgb24", "-f", "framemd5", "-"]))
        .Split('\n').Where(l => l.Length > 0 && !l.StartsWith('#')).Select(l => l.Split(',')[^1].Trim()).ToArray();
    async Task<string> Clip(string name, params string[] args) { var path = Path.Combine(root, name); Directory.CreateDirectory(root); await Run("ffmpeg", ["-v", "error", "-y", .. args, path]); return path; }
    async Task<StudioProject> Import(StudioProject p, string path) { await using var input = File.OpenRead(path); return await service.ImportAssetAsync(p.Id, input, Path.GetFileName(path)); }
    string AssetFile(StudioProject p, string assetId) => service.GetAssetPath(p.Id, assetId);
    async Task<StudioJob> Complete(string projectId, string jobId)
    {
        using var stop = new CancellationTokenSource(TimeSpan.FromSeconds(45));
        while (true)
        {
            var job = (await service.ListJobsAsync(projectId, stop.Token)).Single(j => j.Id == jobId);
            if (job.State is "completed" or "failed" or "cancelled") { Assert.True(job.State == "completed", job.Error); return job; }
            await Task.Delay(30, stop.Token);
        }
    }

    [SkippableTheory]
    [InlineData("bframes")]     // reordered frames: the last decoded is not the last stored
    [InlineData("longaudio")]   // a soundtrack longer than the picture moves the container's end past the last frame
    [InlineData("vfr")]         // variable frame rate: a constant-rate output would drop the true last frame
    [InlineData("rotated")]     // a display matrix: the still is upright, as the models decode it
    [InlineData("single")]      // one frame
    [InlineData("long")]        // longer than the seek window, a soundtrack 4 s longer than the picture
    [InlineData("offset-mp4")]  // a container that starts 3 s in: a seek counts from the file's start, not zero
    [InlineData("offset-mkv")]  // Matroska states the picture's end as a tag on the file's own clock
    public async Task ALastFrameIsTheLastFrameTheDecoderShows(string kind)
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        string source = kind switch
        {
            "bframes" => await Clip("b.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=2", "-c:v", "libx264", "-bf", "3", "-pix_fmt", "yuv420p"),
            "longaudio" => await Clip("a.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=2", "-f", "lavfi", "-i", "sine=d=5", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"),
            // Frames bunched a quarter frame apart at the end, B-frames reordering them: a constant-rate output drops the last.
            "vfr" => await Clip("v.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=2", "-vf", "setpts='if(gt(N,40),(40+(N-40)*0.25)/24/TB,N/24/TB)'", "-fps_mode", "passthrough", "-c:v", "libx264", "-bf", "3", "-pix_fmt", "yuv420p"),
            "rotated" => await Clip("r.mp4", "-display_rotation", "90", "-i", await Clip("plain.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p"), "-c", "copy"),
            "long" => await Clip("l.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=10", "-f", "lavfi", "-i", "sine=d=14", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"),
            "offset-mp4" => await Clip("o.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=10", "-f", "lavfi", "-i", "sine=d=14", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-output_ts_offset", "3"),
            "offset-mkv" => await Clip("o.mkv", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=10", "-f", "lavfi", "-i", "sine=d=14", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-output_ts_offset", "3"),
            _ => await Clip("s.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=1", "-frames:v", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p"),
        };
        var p = await Import(await service.CreateProjectAsync(), source);
        var video = p.Assets.Single();
        long before = p.Revision;
        var taken = await service.ExtractFrameAsync(p.Id, video.Id, StudioFrameSource.LastFrame);
        Assert.False(taken.Reused); Assert.True(taken.Project.Revision > before);
        var still = taken.Project.Assets.Single(a => a.Id == taken.AssetId);
        Assert.Equal("image", still.Kind); Assert.Null(still.Generation);
        Assert.Equal(Path.GetFileNameWithoutExtension(video.Name) + " · last frame.png", still.Name);
        var from = Assert.IsType<StudioFrameSource>(still.DerivedFrom);
        Assert.Equal((video.Id, video.Sha256, StudioFrameSource.LastFrame, StudioFrameSource.DecodeMethod), (from.AssetId, from.Sha256, from.Frame, from.Method));
        Assert.Null(from.FrameRate);
        Assert.Equal((await FrameMd5s(source))[^1], Assert.Single(await FrameMd5s(AssetFile(taken.Project, still.Id))));
        if (kind == "rotated") Assert.Equal((64, 96), (still.Width, still.Height));
        // The same frame again is the same still: a double tap adds nothing and does not move the revision.
        var again = await service.ExtractFrameAsync(p.Id, video.Id, StudioFrameSource.LastFrame);
        Assert.True(again.Reused); Assert.Equal(still.Id, again.AssetId); Assert.Equal(taken.Project.Revision, again.Project.Revision);
        Assert.Equal(2, again.Project.Assets.Count);
    }

    [SkippableFact]
    public async Task FrameNIsTheFrameTheTimelineShowsThere()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        // A 24 fps clip on a 30000/1001 project: frame N is resampled exactly as the renderer resamples the clip.
        string source = await Clip("n.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p");
        var p = await Import(await service.CreateProjectAsync(fpsNumerator: 30000, fpsDenominator: 1001), source);
        var video = p.Assets.Single();
        var taken = await service.ExtractFrameAsync(p.Id, video.Id, 37);
        var still = taken.Project.Assets.Single(a => a.Id == taken.AssetId);
        Assert.Equal(37, still.DerivedFrom!.Frame); Assert.Equal(new StudioFrameRate(30000, 1001), still.DerivedFrom.FrameRate);
        Assert.EndsWith(" · frame 37.png", still.Name);
        var timeline = await FrameMd5s(source, "-vf", "fps=fps=30000/1001:start_time=0:round=near");
        Assert.Equal(timeline[37], Assert.Single(await FrameMd5s(AssetFile(taken.Project, still.Id))));
        // Past the picture's end is refused, never clamped to the last frame.
        var past = await Assert.ThrowsAsync<StudioException>(() => service.ExtractFrameAsync(p.Id, video.Id, timeline.Length + 5));
        Assert.Equal("invalid_frame", past.Code);
        // Another frame, or the same number at another rate, is another still.
        Assert.NotEqual(still.Id, (await service.ExtractFrameAsync(p.Id, video.Id, 36)).AssetId);
    }

    [SkippableFact]
    public async Task OnlyAVideoOfThisProjectHasFramesToContinueFrom()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        var p = await service.CreateProjectAsync();
        p = await Import(p, await Clip("photo.png", "-f", "lavfi", "-i", "testsrc2=s=96x64", "-frames:v", "1"));
        p = await Import(p, await Clip("tone.wav", "-f", "lavfi", "-i", "sine=d=1"));
        p = await Import(p, await Clip("clip.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p"));
        string photo = p.Assets.Single(a => a.Kind == "image").Id, tone = p.Assets.Single(a => a.Kind == "audio").Id, clip = p.Assets.Single(a => a.Kind == "video").Id;
        Assert.Equal("invalid_frame_source", (await Assert.ThrowsAsync<StudioException>(() => service.ExtractFrameAsync(p.Id, photo, -1))).Code);
        Assert.Equal("invalid_frame_source", (await Assert.ThrowsAsync<StudioException>(() => service.ExtractFrameAsync(p.Id, tone, -1))).Code);
        Assert.Equal("invalid_frame", (await Assert.ThrowsAsync<StudioException>(() => service.ExtractFrameAsync(p.Id, clip, -2))).Code);
        var other = await service.CreateProjectAsync();
        Assert.Equal("asset_not_found", (await Assert.ThrowsAsync<StudioException>(() => service.ExtractFrameAsync(other.Id, clip, -1))).Code);
        Assert.Equal(3, (await service.GetProjectAsync(p.Id)).Assets.Count);
        // A host that runs no media subprocesses refuses before decoding anything.
        await service.DisposeAsync();
        service = new StudioService(Path.Combine(root, "studio"), adapter, new StudioOptions { EnableExternalProcesses = false });
        Assert.Equal("unsupported_platform", (await Assert.ThrowsAsync<StudioException>(() => service.ExtractFrameAsync(p.Id, clip, -1))).Code);
        Assert.Equal(3, (await service.GetProjectAsync(p.Id)).Assets.Count);
    }

    [SkippableFact]
    public async Task ATakeContinuesFromItsOwnLosslessLastFrameAndTheChainNamesItsParts()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        adapter.Video = await Clip("take.mp4", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24", "-frames:v", "17", "-c:v", "libx264", "-pix_fmt", "yuv420p");
        // Unlike the video's last frame, so the test can tell the kept frame was used.
        adapter.LastFrame = await Clip("kept.png", "-f", "lavfi", "-i", "color=c=magenta:s=64x64", "-frames:v", "1");
        var p = await service.CreateProjectAsync();
        var first = await Complete(p.Id, (await service.StartGenerationAsync(new() { ProjectId = p.Id, ModelId = "chain-double", Prompt = "A lantern drifts over the lake", FrameCount = 17 })).Id);
        p = await service.GetProjectAsync(p.Id);
        Assert.Equal("A lantern drifts over the lake.mp4", p.Assets.Single(a => a.Id == first.OutputAssetId).Name);
        var taken = await service.ExtractFrameAsync(p.Id, first.OutputAssetId!, StudioFrameSource.LastFrame);
        var still = taken.Project.Assets.Single(a => a.Id == taken.AssetId);
        Assert.Equal(StudioFrameSource.GenerationMethod, still.DerivedFrom!.Method);
        Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(await File.ReadAllBytesAsync(adapter.LastFrame))), still.Sha256);

        // The next segment starts on that still, and is named as the chain's second part.
        var second = await Complete(p.Id, (await service.StartGenerationAsync(new() { ProjectId = p.Id, ModelId = "chain-double", Prompt = "A lantern drifts over the lake",
            FrameCount = 17, Mode = StudioGenerationModes.ImageToVideo, ConditioningAssetId = still.Id })).Id);
        Assert.Equal(still.Id, adapter.Generated!.ConditioningAssetId);
        p = await service.GetProjectAsync(p.Id);
        Assert.Equal("A lantern drifts over the lake · part 2.mp4", p.Assets.Single(a => a.Id == second.OutputAssetId).Name);
        var third = await Complete(p.Id, (await service.StartGenerationAsync(new() { ProjectId = p.Id, ModelId = "chain-double", Prompt = "A lantern drifts over the lake",
            FrameCount = 17, Mode = StudioGenerationModes.ImageToVideo, ConditioningAssetId = (await service.ExtractFrameAsync(p.Id, second.OutputAssetId!, -1)).AssetId })).Id);
        Assert.EndsWith(" · part 3.mp4", (await service.GetProjectAsync(p.Id)).Assets.Single(a => a.Id == third.OutputAssetId).Name);

        // The kept frame survives a restart; a frame other than the last is decoded from the video.
        await service.DisposeAsync(); service = new StudioService(Path.Combine(root, "studio"), adapter);
        p = await service.GetProjectAsync(p.Id);
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "deleteAsset", AssetId = still.Id }]);
        var retaken = await service.ExtractFrameAsync(p.Id, first.OutputAssetId!, -1);
        Assert.False(retaken.Reused);
        Assert.Equal(StudioFrameSource.GenerationMethod, retaken.Project.Assets.Single(a => a.Id == retaken.AssetId).DerivedFrom!.Method);
        var decoded = await service.ExtractFrameAsync(p.Id, first.OutputAssetId!, 5);
        Assert.Equal(StudioFrameSource.DecodeMethod, decoded.Project.Assets.Single(a => a.Id == decoded.AssetId).DerivedFrom!.Method);
    }

    [SkippableFact]
    public async Task AKeptLastFrameOfTheWrongSizeIsNotUsed()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        adapter.Video = await Clip("take.mp4", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24", "-frames:v", "17", "-c:v", "libx264", "-pix_fmt", "yuv420p");
        adapter.LastFrame = await Clip("wrong.png", "-f", "lavfi", "-i", "color=c=magenta:s=32x32", "-frames:v", "1");
        var p = await service.CreateProjectAsync();
        var job = await Complete(p.Id, (await service.StartGenerationAsync(new() { ProjectId = p.Id, ModelId = "chain-double", Prompt = "A take", FrameCount = 17 })).Id);
        var taken = await service.ExtractFrameAsync(p.Id, job.OutputAssetId!, -1);
        var still = taken.Project.Assets.Single(a => a.Id == taken.AssetId);
        Assert.Equal(StudioFrameSource.DecodeMethod, still.DerivedFrom!.Method);
        Assert.Equal((await FrameMd5s(adapter.Video))[^1], Assert.Single(await FrameMd5s(AssetFile(taken.Project, still.Id))));
    }

    [Theory]
    [InlineData("R", true)]
    [InlineData("FR", true)]   // a chain segment that keeps a character
    [InlineData("LR", false)]  // a reference is never a last frame
    [InlineData("FLR", false)]
    [InlineData("GR", false)]
    public void RefMayStartOnAFirstFrameButNeverPinsALastOrTimedFrame(string inputs, bool accepted)
    {
        var request = new StudioGenerationRequest
        {
            Prompt = "A shot", Mode = StudioGenerationModes.Reference, ReferenceAssetIds = ["ref"],
            ConditioningAssetId = inputs.Contains('F') ? "first" : null, EndConditioningAssetId = inputs.Contains('L') ? "last" : null,
            TemporalConditions = inputs.Contains('G') ? [new("guide", 4)] : [],
        };
        if (accepted) Assert.Equal(StudioGenerationModes.Reference, StudioGenerationModes.Resolve(request));
        else Assert.Equal("invalid_mode_inputs", Assert.Throws<StudioException>(() => StudioGenerationModes.Resolve(request)).Code);
        // A request saved without a mode that has both a first frame and references is a ref take.
        if (accepted) { request.Mode = null; Assert.Equal(StudioGenerationModes.Reference, StudioGenerationModes.Resolve(request)); }
    }

    [SkippableTheory]
    [InlineData(false, true)]
    [InlineData(false, false)]  // a frame-pinning checkpoint with no ref mode at all (MiniMax-H3 FL2VA)
    [InlineData(true, true)]
    public async Task AdmissionRunsRefWithAFirstFrameOnlyOnACheckpointThatKeepsBoth(bool keepsBoth, bool takesReferences)
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        adapter.Video = await Clip("take.mp4", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24", "-frames:v", "17", "-c:v", "libx264", "-pix_fmt", "yuv420p");
        adapter.Capability.ReferencesWithFirstFrame = keepsBoth;
        // Without ref mode, the refusal is still why the family cannot keep a character while starting on a frame, never
        // "load the reference checkpoint", which would refuse the pinned first frame in turn.
        adapter.Capability.IdentityReferences = takesReferences;
        if (!takesReferences) adapter.Capability.UnavailableModes[StudioGenerationModes.Reference] = "Load the reference checkpoint to use references.";
        adapter.Capability.ReferencesWithFirstFrameUnavailableReason = keepsBoth ? null : "This family pins frames on one checkpoint and keeps references on another.";
        var p = await service.CreateProjectAsync();
        p = await Import(p, await Clip("first.png", "-f", "lavfi", "-i", "testsrc2=s=64x64", "-frames:v", "1"));
        p = await Import(p, await Clip("face.png", "-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1"));
        var request = new StudioGenerationRequest { ProjectId = p.Id, ModelId = "chain-double", Prompt = "She walks on", FrameCount = 17, Mode = StudioGenerationModes.Reference,
            ConditioningAssetId = p.Assets[0].Id, ReferenceAssetIds = [p.Assets[1].Id] };
        if (!keepsBoth)
        {
            var refused = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationAsync(request));
            Assert.Equal(("unsupported_conditioning", adapter.Capability.ReferencesWithFirstFrameUnavailableReason), (refused.Code, refused.Message));
            Assert.Null(adapter.Validated); Assert.Empty(await service.ListJobsAsync(p.Id));
            return;
        }
        var job = await Complete(p.Id, (await service.StartGenerationAsync(request)).Id);
        Assert.Equal((p.Assets[0].Id, p.Assets[1].Id), (adapter.Generated!.ConditioningAssetId, adapter.Generated.ReferenceAssetIds.Single()));
        Assert.NotNull(job.OutputAssetId);
    }

    [SkippableFact]
    public async Task APlanThatContinuesAVideoTakesTheFrameOnlyWhenItIsStarted()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        adapter.Video = await Clip("take.mp4", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24", "-frames:v", "17", "-c:v", "libx264", "-pix_fmt", "yuv420p");
        var p = await Import(await service.CreateProjectAsync(), adapter.Video);
        var video = p.Assets.Single();
        var api = new StudioApi(service);
        var created = await api.DispatchAsync("generation-drafts/create", JsonSerializer.SerializeToElement(new
        {
            projectId = p.Id, expectedRevision = p.Revision, summary = "Segment two",
            request = new { prompt = "It keeps going", mode = "i2v", width = 64, height = 64 }, firstFrameFrom = new { assetId = video.Id, frame = "last" },
        }));
        var draft = created.Deserialize<StudioGenerationDraft>(StudioJson.Options)!;
        Assert.Equal((video.Id, StudioFrameSource.LastFrame), (draft.FirstFrameFrom!.AssetId, draft.FirstFrameFrom.Frame));
        Assert.Null(draft.Request.ConditioningAssetId);
        // Planning changes nothing in the library.
        var unchanged = await service.GetProjectAsync(p.Id);
        Assert.Equal((p.Revision, 1), (unchanged.Revision, unchanged.Assets.Count));
        // A plan names its first frame one way; a photo is not a video to continue; text-only takes no frame.
        foreach (var (request, from, code) in new (object, object, string)[]
        {
            (new { prompt = "x", mode = "i2v", conditioningAssetId = video.Id }, new { assetId = video.Id }, "invalid_draft"),
            (new { prompt = "x", mode = "t2v" }, new { assetId = video.Id }, "invalid_mode_inputs"),
        })
            Assert.Equal(code, (await Assert.ThrowsAsync<StudioException>(() => api.DispatchAsync("generation-drafts/create", JsonSerializer.SerializeToElement(new
            { projectId = p.Id, expectedRevision = p.Revision, summary = "Bad", request, firstFrameFrom = from })))).Code);
        Assert.Equal("invalid_request", (await Assert.ThrowsAsync<StudioException>(() => api.DispatchAsync("generation-drafts/create", JsonSerializer.SerializeToElement(new
        { projectId = p.Id, expectedRevision = p.Revision, summary = "Bad", request = new { prompt = "x", mode = "i2v" }, firstFrameFrom = new { assetId = video.Id, frame = "first" } })))).Code);

        var job = await service.StartGenerationDraftAsync(p.Id, draft.Id, p.Revision, "chain-double");
        await Complete(p.Id, job.Id);
        p = await service.GetProjectAsync(p.Id);
        var still = p.Assets.Single(a => a.DerivedFrom != null);
        Assert.Equal(video.Id, still.DerivedFrom!.AssetId);
        Assert.Equal(still.Id, adapter.Generated!.ConditioningAssetId);
        // Starting the plan again finds the same job, and takes no second frame.
        Assert.Equal(job.Id, (await service.StartGenerationDraftAsync(p.Id, draft.Id, p.Revision, "chain-double")).Id);
        Assert.Single((await service.GetProjectAsync(p.Id)).Assets, a => a.DerivedFrom != null);
    }

    [SkippableFact]
    public async Task TheExtractActionTakesLastOrAFrameNumberAndNothingElse()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        var p = await Import(await service.CreateProjectAsync(), await Clip("c.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p"));
        var api = new StudioApi(service); string video = p.Assets.Single().Id;
        var last = await api.DispatchAsync("assets/extract-frame", JsonSerializer.SerializeToElement(new { projectId = p.Id, assetId = video, frame = "last" }));
        Assert.False(last.GetProperty("reused").GetBoolean());
        string still = last.GetProperty("assetId").GetString()!;
        Assert.Contains(last.GetProperty("project").GetProperty("assets").EnumerateArray(), a => a.GetProperty("id").GetString() == still && a.GetProperty("derivedFrom").GetProperty("frame").GetInt64() == -1);
        Assert.True((await api.DispatchAsync("assets/extract-frame", JsonSerializer.SerializeToElement(new { projectId = p.Id, assetId = video, frame = "last" }))).GetProperty("reused").GetBoolean());
        Assert.Equal(3, (await api.DispatchAsync("assets/extract-frame", JsonSerializer.SerializeToElement(new { projectId = p.Id, assetId = video, frame = 3 }))).GetProperty("project").GetProperty("assets").GetArrayLength());
        foreach (var payload in new object[] { new { projectId = p.Id, assetId = video, frame = "first" }, new { projectId = p.Id, assetId = video, frame = -1 },
                     new { projectId = p.Id, assetId = video, frame = 1.5 }, new { projectId = p.Id, assetId = video, frame = "last", extra = true } })
            Assert.Equal("invalid_request", (await Assert.ThrowsAsync<StudioException>(() => api.DispatchAsync("assets/extract-frame", JsonSerializer.SerializeToElement(payload)))).Code);
        Assert.Equal("missing_parameter", (await Assert.ThrowsAsync<StudioException>(() => api.DispatchAsync("assets/extract-frame", JsonSerializer.SerializeToElement(new { projectId = p.Id, assetId = video })))).Code);
    }

    [SkippableFact]
    public async Task AStillKeepsItsSourceThroughAPackageAndAForgedRecordIsRefused()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        var p = await Import(await service.CreateProjectAsync(), await Clip("c.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p"));
        var taken = await service.ExtractFrameAsync(p.Id, p.Assets.Single().Id, 4);
        var package = await service.ExportPackageAsync(p.Id);
        await using (var input = File.OpenRead(package))
        {
            var imported = await service.ImportPackageAsync(input);
            var still = imported.Assets.Single(a => a.Id == taken.AssetId);
            Assert.Equal((p.Assets[0].Id, 4L, new StudioFrameRate(24, 1)), (still.DerivedFrom!.AssetId, still.DerivedFrom.Frame, still.DerivedFrom.FrameRate));
        }
        // A last frame never names a frame rate, and a frame number always does.
        string forged = Path.Combine(root, "forged.tsstudio"); File.Copy(package, forged);
        using (var archive = ZipFile.Open(forged, ZipArchiveMode.Update))
        {
            var entry = archive.GetEntry("project.json")!; JsonNode project;
            using (var reader = new StreamReader(entry.Open())) project = JsonNode.Parse(await reader.ReadToEndAsync())!;
            foreach (var asset in project["assets"]!.AsArray()) if (asset!["derivedFrom"] is JsonObject derived) derived["frame"] = -1;
            entry.Delete();
            using var writer = new StreamWriter(archive.CreateEntry("project.json").Open()); await writer.WriteAsync(project.ToJsonString());
        }
        await using var forgedInput = File.OpenRead(forged);
        Assert.Equal("invalid_asset", (await Assert.ThrowsAsync<StudioException>(() => service.ImportPackageAsync(forgedInput))).Code);
    }

    [SkippableFact]
    public async Task ALastFrameThumbnailShowsTheEndOfTheVideo()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        // Red for a second, then blue: the opening and the end differ.
        // Ten seconds, a 14 s soundtrack and a 3 s start offset: the end is found where the picture ends.
        var p = await Import(await service.CreateProjectAsync(), await Clip("rb.mp4", "-f", "lavfi", "-i", "color=c=red:s=64x64:r=24:d=5", "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=24:d=5",
            "-f", "lavfi", "-i", "sine=d=14", "-filter_complex", "[0][1]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-output_ts_offset", "3"));
        string video = p.Assets.Single().Id;
        async Task<byte[]> Center(string jpeg) { var raw = Path.Combine(root, StudioJson.NewId() + ".rgb"); await Run("ffmpeg", "-v", "error", "-i", jpeg, "-vf", "crop=2:2:159:89,scale=1:1", "-pix_fmt", "rgb24", "-f", "rawvideo", raw); return await File.ReadAllBytesAsync(raw); }
        var opening = await Center(await service.GetThumbnailAsync(p.Id, video));
        var end = await Center(await service.GetThumbnailAsync(p.Id, video, lastFrame: true, default));
        Assert.True(opening[0] > 180 && opening[2] < 80, "opening is red");
        Assert.True(end[2] > 180 && end[0] < 80, "end is blue");
    }

    [SkippableFact]
    public async Task NonSquarePixelsAreWidenedAndAnUnboundedRatioIsRefused()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        var p = await service.CreateProjectAsync();
        p = await Import(p, await Clip("dv.mp4", "-f", "lavfi", "-i", "testsrc2=s=720x480:r=24:d=1", "-vf", "setsar=32/27", "-c:v", "libx264", "-pix_fmt", "yuv420p"));
        p = await Import(p, await Clip("thin.mp4", "-f", "lavfi", "-i", "testsrc2=s=96x64:r=24:d=1", "-vf", "setsar=8/1", "-c:v", "libx264", "-pix_fmt", "yuv420p"));
        var taken = await service.ExtractFrameAsync(p.Id, p.Assets.Single(a => a.Name == "dv.mp4").Id, -1);
        var still = taken.Project.Assets.Single(a => a.Id == taken.AssetId);
        Assert.Equal((854, 480), (still.Width, still.Height)); // shown as a player shows it, on square pixels
        var refused = await Assert.ThrowsAsync<StudioException>(() => service.ExtractFrameAsync(p.Id, p.Assets.Single(a => a.Name == "thin.mp4").Id, -1));
        Assert.Equal("unsupported_frame_source", refused.Code); Assert.Contains("8:1", refused.Message); // beyond 4:1, the widest Studio widens
        // A quarter turn is applied first (transposing the frame and inverting the ratio), so the bound is on the upright
        // frame: 4200×200 at 4:1 would be 16800 wide unturned, but turned it is 50×4200.
        p = await Import(p, await Clip("turned.mp4", "-display_rotation", "90", "-i",
            await Clip("wide.mp4", "-f", "lavfi", "-i", "testsrc2=s=4200x200:r=24:d=1", "-frames:v", "2", "-vf", "setsar=4/1", "-c:v", "libx264", "-pix_fmt", "yuv420p"), "-c", "copy"));
        var turned = await service.ExtractFrameAsync(p.Id, p.Assets.Single(a => a.Name == "turned.mp4").Id, -1);
        var upright = turned.Project.Assets.Single(a => a.Id == turned.AssetId);
        Assert.Equal((50, 4200), (upright.Width, upright.Height));
    }

    [SkippableFact]
    public async Task APlanTheLoadedModelRefusesLeavesTheLibraryAloneAndAStartedPlanStaysStarted()
    {
        Skip.IfNot(MediaInstalled(), "Operator-installed ffmpeg and ffprobe are required for real media validation.");
        adapter.Video = await Clip("take.mp4", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24", "-frames:v", "17", "-c:v", "libx264", "-pix_fmt", "yuv420p");
        adapter.Capability.ReferencesWithFirstFrameUnavailableReason = "This family cannot keep references while starting on a frame.";
        var p = await Import(await service.CreateProjectAsync(), adapter.Video);
        p = await Import(p, await Clip("face.png", "-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1"));
        string video = p.Assets.Single(a => a.Kind == "video").Id, face = p.Assets.Single(a => a.Kind == "image").Id;
        var refusedPlan = await service.CreateGenerationDraftAsync(p.Id, p.Revision, new() { Prompt = "She walks on", Mode = "ref", ReferenceAssetIds = [face], Width = 64, Height = 64 },
            "Part 2 with her face", new StudioFrameSelector { AssetId = video, Frame = StudioFrameSource.LastFrame });
        var refused = await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationDraftAsync(p.Id, refusedPlan.Id, p.Revision, "chain-double"));
        Assert.Equal("unsupported_conditioning", refused.Code);
        var unchanged = await service.GetProjectAsync(p.Id);
        Assert.Equal((p.Revision, 2), (unchanged.Revision, unchanged.Assets.Count)); // no still was taken for a plan that cannot run
        // Nor does a plan refused for its other inputs: a negative prompt where guidance 1 runs none, or a photo deleted since.
        adapter.Capability.DefaultCfgScale = 1;
        var negative = await service.CreateGenerationDraftAsync(p.Id, p.Revision, new() { Prompt = "On", Mode = "i2v", NegativePrompt = "blur", Width = 64, Height = 64 },
            "Part 2", new StudioFrameSelector { AssetId = video, Frame = StudioFrameSource.LastFrame });
        Assert.Equal("unsupported_negative_prompt", (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationDraftAsync(p.Id, negative.Id, p.Revision, "chain-double"))).Code);
        var ending = await service.CreateGenerationDraftAsync(p.Id, p.Revision, new() { Prompt = "On", Mode = "fl2v", EndConditioningAssetId = face, Width = 64, Height = 64 },
            "Part 2 ending on her face", new StudioFrameSelector { AssetId = video, Frame = StudioFrameSource.LastFrame });
        p = await service.ApplyCommandsAsync(p.Id, p.Revision, StudioJson.NewId(), [new() { Type = "deleteAsset", AssetId = face }]);
        Assert.Equal("asset_not_found", (await Assert.ThrowsAsync<StudioException>(() => service.StartGenerationDraftAsync(p.Id, ending.Id, p.Revision, "chain-double"))).Code);
        Assert.DoesNotContain((await service.GetProjectAsync(p.Id)).Assets, a => a.DerivedFrom != null);
        p = await service.UndoAsync(p.Id, p.Revision, StudioJson.NewId());
        adapter.Capability.DefaultCfgScale = 5;
        // A plan out of the video's range is refused when it is saved, not when it is started.
        Assert.Equal("invalid_frame", (await Assert.ThrowsAsync<StudioException>(() => service.CreateGenerationDraftAsync(p.Id, p.Revision, new() { Prompt = "x", Mode = "i2v" },
            "Too far", new StudioFrameSelector { AssetId = video, Frame = 5000 }))).Code);
        // A started plan returns its job when the start is retried, although its frame moved the project to a new revision.
        var plan = await service.CreateGenerationDraftAsync(p.Id, p.Revision, new() { Prompt = "It goes on", Mode = "i2v", Width = 64, Height = 64 }, "Part 2",
            new StudioFrameSelector { AssetId = video, Frame = StudioFrameSource.LastFrame });
        var job = await service.StartGenerationDraftAsync(p.Id, plan.Id, p.Revision, "chain-double");
        Assert.True((await service.GetProjectAsync(p.Id)).Revision > p.Revision);
        Assert.Equal(job.Id, (await service.StartGenerationDraftAsync(p.Id, plan.Id, p.Revision, "chain-double")).Id);
        await Complete(p.Id, job.Id);
    }

    [Fact]
    public void NewFieldsAreOmittedWhenUnset()
    {
        // Assets, capabilities and plans saved before chains serialize exactly as before.
        string asset = JsonSerializer.Serialize(new StudioAsset(), StudioJson.Options), capability = JsonSerializer.Serialize(new StudioModelCapability(), StudioJson.Options);
        Assert.DoesNotContain("derivedFrom", asset);
        Assert.DoesNotContain("referencesWithFirstFrameUnavailableReason", capability); Assert.DoesNotContain("referenceDefaultWidth", capability);
        Assert.DoesNotContain("firstFrameFrom", JsonSerializer.Serialize(new StudioGenerationDraft(), StudioJson.Options));
    }
}
