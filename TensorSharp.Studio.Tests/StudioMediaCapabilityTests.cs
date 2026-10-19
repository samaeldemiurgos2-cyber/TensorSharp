// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Studio;

namespace TensorSharp.Studio.Tests;

public sealed class StudioMediaCapabilityTests : IDisposable
{
    readonly string root = Path.Combine(Path.GetTempPath(), "studio-media-capabilities-" + StudioJson.NewId());
    public StudioMediaCapabilityTests() => Directory.CreateDirectory(root);
    public void Dispose() => Directory.Delete(root, true);

    static string Tool(string name) => (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator)
        .Select(p => Path.Combine(p, name + (OperatingSystem.IsWindows() ? ".exe" : ""))).First(File.Exists);
    static string Quote(string value) => "'" + value.Replace("'", "'\"'\"'") + "'";

    async Task RequireWrapperToolsAsync()
    {
        Skip.If(OperatingSystem.IsWindows(), "The executable fault-injection fixture requires a Unix host.");
        Skip.IfNot(new StudioMediaFactAttribute().Skip == null, "Installed ffmpeg and ffprobe are required for real encoding validation.");
        var encoders = await StudioMedia.RunAsync(Tool("ffmpeg"), ["-hide_banner", "-encoders"], null, default, TimeSpan.FromSeconds(10));
        Skip.IfNot(encoders.Contains(" libx264 ", StringComparison.Ordinal), "The fault-injection fixture uses installed libx264 for its successful encoder path.");
    }

    async Task<string> WrapperAsync(bool failAll = false, bool advertiseVideoToolbox = false)
    {
        string path = Path.Combine(root, "ffmpeg-fixture");
        // Only the encoder-list/startup boundary is injected. Successful encodes and every output probe use real tools.
        string script = "#!/bin/sh\n" +
            "case \"$*\" in\n" +
            "  *-encoders*) printf '%s\\n' ' V..... libx264 fixture' ' A..... aac fixture'" + (advertiseVideoToolbox ? " ' V..... h264_videotoolbox fixture'" : "") + "; exit 0 ;;\n" +
            "  *-filters*) printf '%s\\n' ' ... drawtext fixture'; exit 0 ;;\n" +
            "esac\n" +
            "for arg in \"$@\"; do\n" +
            "  case \"$arg\" in libx264|h264_videotoolbox) printf '%s\\n' \"$arg\" >> " + Quote(Path.Combine(root, "encoder-attempts")) + ";; esac\n" +
            (failAll ? "  case \"$arg\" in libx264|h264_videotoolbox) printf '%s\\n' 'Fixture encoder cannot open' >&2; exit 42;; esac\n" :
                "  if [ \"$arg\" = h264_videotoolbox ]; then printf '%s\\n' 'Fixture VideoToolbox session unavailable' >&2; exit 42; fi\n") +
            "done\nexec " + Quote(Tool("ffmpeg")) + " \"$@\"\n";
        await File.WriteAllTextAsync(path, script);
        if (!OperatingSystem.IsWindows()) File.SetUnixFileMode(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        return path;
    }

    [SkippableFact]
    public async Task ConcurrentCapabilityChecksEncodeOnceAndChangedToolSettingsAreRevalidated()
    {
        await RequireWrapperToolsAsync();
        var options = new StudioOptions { FfmpegPath = await WrapperAsync(), FfprobePath = Tool("ffprobe") };
        var media = new StudioMedia(options);
        var capabilities = await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => media.GetCapabilitiesAsync()));
        Assert.All(capabilities, capability => { Assert.True(capability.Available, capability.Reason); Assert.Equal("libx264", capability.Codec); });
        Assert.Equal(["libx264"], await File.ReadAllLinesAsync(Path.Combine(root, "encoder-attempts")));
        options.FfprobePath = Path.Combine(root, "missing-ffprobe");
        var changed = await media.GetCapabilitiesAsync();
        Assert.False(changed.Available); Assert.False(changed.ProbeAvailable);
        Assert.Contains("missing-ffprobe", changed.Reason);
    }

    [SkippableFact]
    public async Task ListedEncoderMustReallyEncodeAndFailureCanRecoverWithoutRestart()
    {
        await RequireWrapperToolsAsync();
        var media = new StudioMedia(new() { FfmpegPath = await WrapperAsync(failAll: true), FfprobePath = Tool("ffprobe") });
        var failed = await media.GetCapabilitiesAsync();
        Assert.False(failed.Available); Assert.True(failed.ProbeAvailable);
        Assert.Contains("Fixture encoder cannot open", failed.Reason);
        await WrapperAsync(); // Same configured path after the tool/session is repaired.
        var recovered = await media.GetCapabilitiesAsync();
        Assert.True(recovered.Available, recovered.Reason); Assert.Equal("libx264", recovered.Codec);
        Assert.Equal(["libx264", "libx264"], await File.ReadAllLinesAsync(Path.Combine(root, "encoder-attempts")));
    }

    [SkippableFact]
    public async Task UnavailableAppleEncodingSessionFallsBackToWorkingSoftwareH264()
    {
        await RequireWrapperToolsAsync();
        Skip.IfNot(OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst(), "VideoToolbox is selected only on Apple desktop hosts.");
        var media = new StudioMedia(new() { FfmpegPath = await WrapperAsync(advertiseVideoToolbox: true), FfprobePath = Tool("ffprobe") });
        var capability = await media.GetCapabilitiesAsync();
        Assert.True(capability.Available, capability.Reason); Assert.True(capability.ProbeAvailable); Assert.Equal("libx264", capability.Codec);
        Assert.Equal(["h264_videotoolbox", "libx264"], await File.ReadAllLinesAsync(Path.Combine(root, "encoder-attempts")));
    }

    [Theory]
    [InlineData(1280, 720, 24, "6000000")]
    [InlineData(3840, 2160, 24, "23887872")]
    [InlineData(3840, 2160, 60, "59719680")]
    public void HardwareExportBitrateScalesWithResolutionAndFrameRate(int width, int height, int fps, string bitrate)
    {
        var project = new StudioProject { Width = width, Height = height, FrameRate = new(fps) };
        project.Tracks.Add(new() { Name = "Video" });
        project.Assets.Add(new() { FileName = "fixture.mp4", DurationSeconds = 1, Sha256 = new('0', 64), Bytes = 1 });
        project.Clips.Add(new() { TrackId = project.Tracks[0].Id, AssetId = project.Assets[0].Id, DurationFrames = fps });
        string source = Path.Combine(root, "fixture.mp4"); File.WriteAllBytes(source, [0]);
        var media = new StudioMedia(new());
        var plan = media.CreateRenderPlan(project, _ => source, root, "h264_videotoolbox");
        int index = plan.Arguments.ToList().IndexOf("-b:v");
        Assert.True(index > 0); Assert.Equal(bitrate, plan.Arguments[index + 1]);
    }
}
