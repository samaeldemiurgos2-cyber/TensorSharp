# Apple video mux regression probe

Build and run the real `AppleMediaProvider` under Mac Catalyst without loading a
model or running inference. Use the SDK with the Mac Catalyst workload installed:

```sh
~/.dotnet/dotnet build eng/validation/AppleVideoMuxProbe/AppleVideoMuxProbe.csproj -c Debug -m:1
'eng/validation/AppleVideoMuxProbe/bin/Debug/net10.0-maccatalyst/maccatalyst-arm64/TensorSharp Apple Video Mux Probe.app/Contents/MacOS/AppleVideoMuxProbe' \
  "$PWD/artifacts/apple-video-mux"
```

The argument selects the output directory for MP4 fixtures and `report.json`.
The process exits nonzero if any fixture fails. It checks a one-second 32 kHz
stereo clip, the 145-frame/24 fps and 48 kHz stereo timing from a failed Studio
generation, a longer soundtrack than video, and a shorter soundtrack than video.
It repeats the failed generation's timing with 1024×1024 synthetic frames.
Each output must have an audio and video track, the video duration must remain
correct, and decoding its soundtrack must recover an audible tone with the
expected RMS range.

These synthetic frames exercise native writer backpressure, the reported
1024×1024 geometry, and independent track completion. They do not validate model
inference, sustained high resolution throughput, iOS devices, or other codecs. Generated outputs belong in
ignored `artifacts/` or `docs/validation/`, outside this tool's source directory.
