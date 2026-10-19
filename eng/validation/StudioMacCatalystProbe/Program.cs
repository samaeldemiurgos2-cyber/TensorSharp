// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Chat.Studio;
using TensorSharp.Studio;

// Run the built .app's Contents/MacOS/StudioMacCatalystProbe directly. An actual
// Catalyst runtime is essential: the ordinary macOS test host reports different OS flags.
// Use a worker because UIKit installs a main-thread synchronization context, while
// this command-line probe deliberately runs without a UIKit event loop.
return Task.Run(async () =>
{
    var result = new Dictionary<string, object?>
    {
        ["isIOS"] = OperatingSystem.IsIOS(),
        ["isMacCatalyst"] = OperatingSystem.IsMacCatalyst(),
        ["externalProcesses"] = new StudioOptions().EnableExternalProcesses
    };
    try
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo("/usr/bin/true")
            { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true }
        };
        process.Start();
        await process.WaitForExitAsync();
        result["processExitCode"] = process.ExitCode;
        var options = StudioHostOptions.Create();
        result["ffmpegPath"] = options.FfmpegPath;
        result["ffprobePath"] = options.FfprobePath;
        var capabilities = await new StudioMedia(options).GetCapabilitiesAsync();
        result["capabilities"] = capabilities;
        if (!capabilities.Available) throw new InvalidOperationException(capabilities.Reason);
        string workspace = Path.Combine(Path.GetTempPath(), "studio-catalyst-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(workspace);
        try
        {
            string source = Path.Combine(workspace, "source clip.mp4");
            using var fixture = new Process
            {
                StartInfo = new ProcessStartInfo(options.FfmpegPath)
                { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true }
            };
            foreach (string argument in new[] { "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24:d=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-c:a", "aac", "-shortest", source })
                fixture.StartInfo.ArgumentList.Add(argument);
            fixture.Start();
            Task<string> fixtureOutput = fixture.StandardOutput.ReadToEndAsync(), fixtureError = fixture.StandardError.ReadToEndAsync();
            await fixture.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(30));
            await fixtureOutput;
            if (fixture.ExitCode != 0) throw new InvalidOperationException(await fixtureError);

            await using var studio = new StudioService(Path.Combine(workspace, "studio"), options: options);
            var project = await studio.CreateProjectAsync("Catalyst media round trip", 64, 64);
            await using (var input = File.OpenRead(source)) project = await studio.ImportAssetAsync(project.Id, input, "Source clip.mp4");
            var asset = project.Assets.Single();
            result["import"] = new { asset.Kind, asset.Width, asset.Height, asset.HasAudio, asset.DurationSeconds };
            result["thumbnailBytes"] = new FileInfo(await studio.GetThumbnailAsync(project.Id, asset.Id)).Length;
            result["waveformPeaks"] = (await studio.GetWaveformAsync(project.Id, asset.Id)).Peaks.Count;
            project = await studio.ApplyCommandsAsync(project.Id, project.Revision, StudioJson.NewId(),
                [new() { Type = "insertClip", AssetId = asset.Id, TrackId = project.Tracks.First(t => t.Kind == "video").Id, StartFrame = 0, DurationFrames = 24 }]);
            string output = await studio.Media.RenderAsync(project, id => studio.GetAssetPath(project.Id, id), Path.Combine(workspace, "render"), new Progress<StudioProgress>(), CancellationToken.None);
            result["export"] = await studio.Media.ProbeAsync(output, "Exported video", CancellationToken.None);
            result["exportBytes"] = new FileInfo(output).Length;
        }
        finally { Directory.Delete(workspace, true); }
        result["ok"] = options.EnableExternalProcesses && process.ExitCode == 0;
    }
    catch (Exception e)
    {
        result["ok"] = false;
        result["error"] = e.ToString();
    }
    string report = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
    if (args.Length > 0) File.WriteAllText(Path.GetFullPath(args[0]), report);
    Console.WriteLine(report);
    return result["ok"] is true ? 0 : 1;
}).GetAwaiter().GetResult();
