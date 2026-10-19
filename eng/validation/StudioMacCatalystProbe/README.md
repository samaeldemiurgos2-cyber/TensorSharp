# Studio Mac Catalyst host probe

This small application runs the real `TensorSharp.Studio` assembly under the Mac
Catalyst runtime. An ordinary macOS xUnit process cannot catch Catalyst reporting
both `OperatingSystem.IsIOS()` and `OperatingSystem.IsMacCatalyst()` as true.

From the repository root, using a .NET SDK with the Mac Catalyst workload:

```sh
dotnet build eng/validation/StudioMacCatalystProbe/StudioMacCatalystProbe.csproj -c Debug -m:1
PATH=/usr/bin:/bin \
  'eng/validation/StudioMacCatalystProbe/bin/Debug/net10.0-maccatalyst/maccatalyst-arm64/TensorSharp Studio Mac Probe.app/Contents/MacOS/StudioMacCatalystProbe' \
  "$PWD/artifacts/studio-maccatalyst-probe.json"
```

The sparse PATH reproduces Finder launches. The production host options and media
executable locator must still find installed FFmpeg and ffprobe. `TS_FFMPEG` and
`TS_FFPROBE` may select another installation.

The probe checks process launch and exit, media capabilities, import of a one
second 64×64 clip with audio, thumbnail and waveform generation, timeline editing,
and export with the selected encoder followed by the production full decode
validation. Its temporary project is removed afterwards. The optional first
argument saves structured JSON; console output alone goes through Apple's logger.
An unavailable tool or failed stage produces `ok: false` and a nonzero exit code.

This is host and media validation. It does not load an LTX checkpoint or measure
model inference quality, speed, long clips, or high resolution exports.
