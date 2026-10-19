// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Text.Json;
using AVFoundation;
using Foundation;
using TensorSharp.Models.Media.Apple;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

// Run the actual Apple provider on Catalyst without loading model weights or inference libraries.
return Task.Run(() =>
{
    string directory = Path.GetFullPath(args.Length > 0 ? args[0] : "artifacts/apple-video-mux");
    Directory.CreateDirectory(directory);
    var checks = new List<object>();
    bool ok = true;
    foreach (var fixture in new[]
    {
        (Name: "short-32k", Size: 64, Frames: 24, Fps: 24, Rate: 32000, Samples: 32000),
        (Name: "ltx-48k-short-audio", Size: 64, Frames: 145, Fps: 24, Rate: 48000, Samples: 288480),
        (Name: "long-audio", Size: 64, Frames: 24, Fps: 24, Rate: 48000, Samples: 48000 * 6),
        (Name: "short-audio", Size: 64, Frames: 145, Fps: 24, Rate: 48000, Samples: 48000),
        (Name: "ltx-1024-48k", Size: 1024, Frames: 145, Fps: 24, Rate: 48000, Samples: 288480),
    })
    {
        var clock = Stopwatch.StartNew();
        string path = Path.Combine(directory, fixture.Name + ".mp4");
        try
        {
            int width = fixture.Size, height = fixture.Size;
            var frames = new RgbImage[fixture.Frames];
            for (int i = 0; i < frames.Length; i++)
            {
                var pixels = new float[width * height * 3];
                for (int pixel = 0; pixel < width * height; pixel++)
                {
                    pixels[pixel * 3] = i / (float)frames.Length;
                    pixels[pixel * 3 + 1] = pixel % width / (float)width;
                    pixels[pixel * 3 + 2] = pixel / width / (float)height;
                }
                frames[i] = new RgbImage(width, height, pixels);
            }
            var channels = new[] { new float[fixture.Samples], new float[fixture.Samples] };
            for (int channel = 0; channel < channels.Length; channel++)
                for (int i = 0; i < fixture.Samples; i++)
                    channels[channel][i] = (float)(.4 * Math.Sin(2 * Math.PI * (440 + channel * 220) * i / fixture.Rate));
            var audio = new GeneratedVideoAudio { Channels = channels, SampleRate = fixture.Rate };
            string codec = AppleMediaProvider.Instance.SaveMp4(path, frames, fixture.Fps, audio, out bool muxed);
            if (!muxed) throw new InvalidOperationException("Provider returned audioMuxed=false.");
            using var url = NSUrl.FromFilename(path);
            using var asset = new AVUrlAsset(url, new AVUrlAssetOptions { PreferPreciseDurationAndTiming = true });
            var video = asset.GetTracks(AVMediaTypes.Video);
            var sound = asset.GetTracks(AVMediaTypes.Audio);
            if (video.Length != 1 || sound.Length != 1)
                throw new InvalidOperationException($"Expected video and audio tracks; got {video.Length}/{sound.Length}.");
            double duration = fixture.Frames / (double)fixture.Fps;
            if (Math.Abs(asset.Duration.Seconds - duration) > .06)
                throw new InvalidOperationException($"Duration {asset.Duration.Seconds} differs from {duration}.");
            var decoded = AppleMediaProvider.Instance.Decode(path);
            double rms = Math.Sqrt(decoded.Channels[0].Average(v => v * (double)v));
            if (rms < .20 || rms > .36) throw new InvalidOperationException($"Decoded tone RMS {rms} is incorrect.");
            checks.Add(new { fixture.Name, ok = true, codec, seconds = clock.Elapsed.TotalSeconds,
                width, height, frames = fixture.Frames,
                videoDuration = video[0].TimeRange.Duration.Seconds, audioDuration = sound[0].TimeRange.Duration.Seconds, rms });
        }
        catch (Exception e)
        {
            ok = false;
            checks.Add(new { fixture.Name, ok = false, seconds = clock.Elapsed.TotalSeconds, error = e.ToString() });
        }
        File.WriteAllText(Path.Combine(directory, "report.json"), JsonSerializer.Serialize(new { ok, checks }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"{fixture.Name}: {clock.Elapsed.TotalSeconds:0.00}s");
    }
    return ok ? 0 : 1;
}).GetAwaiter().GetResult();
