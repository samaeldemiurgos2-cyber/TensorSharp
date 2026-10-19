// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.MLX;
using TensorSharp.Models.Ltx25;

if (args.Length != 5)
    throw new ArgumentException("Usage: Ltx25DecodeProbe checkpoint output-frames width height report.json");
string checkpoint = Path.GetFullPath(args[0]), reportPath = Path.GetFullPath(args[4]);
int frames = int.Parse(args[1]), width = int.Parse(args[2]), height = int.Parse(args[3]);
if (frames <= 0 || (frames - 1) % 8 != 0 || width <= 0 || height <= 0 || width % 32 != 0 || height % 32 != 0)
    throw new ArgumentException("Use positive dimensions on the convolutional VAE's 8n+1 frame / 32 pixel grid.");
if (File.Exists(reportPath)) throw new IOException("Use a fresh report path.");
Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
var watch = Stopwatch.StartNew();
var report = new Dictionary<string, object?>
{
    ["completed"] = false, ["checkpoint"] = checkpoint,
    ["frames"] = frames, ["width"] = width, ["height"] = height,
    ["backend"] = "MLX", ["scope"] = "Real convolutional VAE weights; deterministic synthetic latents; no denoising, audio or MP4 encoding.",
    ["limitations"] = "Functional decode validation only; synthetic latents do not assess generated video quality or complete generation performance."
};
void Save() => File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }) + "\n");
Save();
try
{
    using var allocator = new MlxAllocator();
    using var decoder = new Ltx25VideoVaeDecoder(checkpoint, allocator);
    int t = (frames - 1) / decoder.TemporalScale + 1, h = height / decoder.SpatialScale, w = width / decoder.SpatialScale;
    var latents = new float[checked(decoder.LatentChannels * t * h * w)];
    // Bounded, reproducible values exercise the actual convolutions without a denoiser checkpoint.
    for (int i = 0; i < latents.Length; i++) latents[i] = MathF.Sin(i * .001f) * .1f;
    Console.WriteLine($"Decoding {frames} frames at {width}x{height} with {latents.Length} synthetic latent values.");
    float[] rgb = decoder.Decode(latents, t, h, w);
    if (rgb.LongLength != checked((long)frames * width * height * 3)) throw new InvalidDataException("Decoded geometry differs from request.");
    float min = float.PositiveInfinity, max = float.NegativeInfinity;
    double sum = 0;
    foreach (float value in rgb)
    {
        if (!float.IsFinite(value) || value < 0 || value > 1) throw new InvalidDataException("Invalid RGB output.");
        min = Math.Min(min, value); max = Math.Max(max, value); sum += value;
    }
    report["outputElements"] = rgb.LongLength;
    report["rgbMin"] = min; report["rgbMax"] = max; report["rgbMean"] = sum / rgb.LongLength;
    report["completed"] = true;
}
catch (Exception e)
{
    report["error"] = e.ToString();
    throw;
}
finally
{
    report["elapsedSeconds"] = watch.Elapsed.TotalSeconds;
    long peakWorkingSet = Process.GetCurrentProcess().PeakWorkingSet64;
    report["peakWorkingSetBytes"] = peakWorkingSet > 0 ? peakWorkingSet : null;
    if (peakWorkingSet <= 0) report["memoryMeasurement"] = "Process.PeakWorkingSet64 unavailable; use an external process/device sampler.";
    Save();
}
Console.WriteLine(JsonSerializer.Serialize(report));
