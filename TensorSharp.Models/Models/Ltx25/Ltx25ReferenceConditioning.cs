// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Ltx25;

/// <summary>Ingredients IC-LoRA identity/appearance conditioning. A reference
/// sheet becomes a real static RGB video, is encoded by the causal video VAE,
/// and is appended as clean reference tokens on the original causal time grid.
/// It never replaces output frame zero or marks references as generated frames.</summary>
public static class Ltx25ReferenceConditioning
{
    public const int MinimumFrames = 121;
    public const int MaxImages = 16;
    public const int MaxVideos = 1;

    /// <summary>One normalized reference clip in channel-major [128,T,H,W] layout.</summary>
    public sealed record ReferenceLatent(float[] ChannelMajor, int Frames, int Height, int Width);

    /// <summary>Resolve identity sources without interpreting them as timeline
    /// anchors. Single-image fields and reference-image lists are alternative forms.</summary>
    public static IReadOnlyList<VideoImageCondition> ResolveSources(VideoGenerationParams parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (parameters.Mode != "ref") throw new ArgumentException("Ingredients identity images require mode ref.");
        if (parameters.EndImage != null || !string.IsNullOrWhiteSpace(parameters.EndImagePath))
            throw new ArgumentException("Identity references cannot be combined with a last-frame image.");
        _ = ResolveFirstFrameGuide(parameters);
        bool single = parameters.Image != null || parameters.ImageBytes is { Length: > 0 } || !string.IsNullOrWhiteSpace(parameters.ImagePath);
        if (single && parameters.ReferenceImagePaths is { Count: > 0 })
            throw new ArgumentException("Use a single identity image or a reference-image list, rather than combining both.");
        var sources = new List<VideoImageCondition>();
        if (single) sources.Add(new() { Image = parameters.Image, ImageBytes = parameters.ImageBytes, ImagePath = parameters.ImagePath });
        else if (parameters.ReferenceImagePaths != null)
            foreach (string path in parameters.ReferenceImagePaths)
            {
                if (string.IsNullOrWhiteSpace(path)) throw new ArgumentException("Identity reference paths cannot be empty.");
                sources.Add(new() { ImagePath = path });
            }
        if (sources.Count > MaxImages)
            throw new ArgumentException($"Ingredients mode accepts at most {MaxImages} identity reference images.");
        if (sources.Count == 0 && parameters.ReferenceVideoPaths is not { Count: > 0 })
            throw new ArgumentException($"Ingredients mode requires at least one identity reference image or video.");
        return sources;
    }

    /// <summary>The optional first-frame guide of an identity request: at most one image condition, at output frame 0,
    /// which pins the take's opening (the last frame of the segment a chain continues) while the sheet keeps the
    /// character. Upstream ICLoraPipeline conditions a stage on its <c>images</c> and its IC-LoRA
    /// <c>video_conditioning</c> together; a frame-0 image replaces latent frame 0 in place and adds no tokens. Last-frame
    /// and timed guides beside identity references are refused: no publisher workflow uses them.</summary>
    public static VideoImageCondition ResolveFirstFrameGuide(VideoGenerationParams parameters)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (parameters.ImageConditions is not { Count: > 0 } guides) return null;
        if (guides.Count > 1 || guides[0] is not { FrameIndex: 0 } guide)
            throw new ArgumentException("Identity references take at most one image guide, at output frame 0 (the frame a take starts on); " +
                "last-frame and timed guides cannot be combined with Ingredients references.");
        if (guide.Image == null && guide.ImageBytes is not { Length: > 0 } && string.IsNullOrWhiteSpace(guide.ImagePath))
            throw new ArgumentException("The first-frame guide requires an image.");
        if (!float.IsFinite(guide.Strength) || guide.Strength < 0 || guide.Strength > 1)
            throw new ArgumentException("Image conditioning strength must be finite and in [0,1].");
        return guide;
    }

    /// <summary>Compose an ordered contact sheet on black, fitting each complete
    /// source into a grid cell without stretching, cropping or adding text.</summary>
    public static RgbImage BuildSheet(IReadOnlyList<RgbImage> sources, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(sources);
        if (sources.Count is < 1 or > MaxImages || width < 32 || height < 32 || width % 32 != 0 || height % 32 != 0)
            throw new ArgumentException("Ingredients requires 1–16 images and positive 32-aligned sheet geometry.");
        int columns = checked((int)Math.Ceiling(Math.Sqrt(sources.Count)));
        int rows = (sources.Count + columns - 1) / columns;
        var pixels = new float[checked(width * height * 3)];
        for (int index = 0; index < sources.Count; ++index)
        {
            var source = sources[index];
            if (source == null || source.Width <= 0 || source.Height <= 0 || source.Pixels == null ||
                source.Pixels.Length != checked(source.Width * source.Height * 3) ||
                source.Pixels.Any(value => !float.IsFinite(value) || value < 0 || value > 1) ||
                source.Alpha != null && source.Alpha.Any(value => !float.IsFinite(value) || value < 0 || value > 1))
                throw new ArgumentException("Identity reference images must contain finite RGB01 pixels.");
            if (source.Alpha != null)
            {
                var opaque = new float[source.Pixels.Length];
                for (int value = 0; value < opaque.Length; ++value) opaque[value] = source.Pixels[value] * source.Alpha[value / 3];
                source = new RgbImage(source.Width, source.Height, opaque);
            }
            int left = index % columns * width / columns, right = (index % columns + 1) * width / columns;
            int top = index / columns * height / rows, bottom = (index / columns + 1) * height / rows;
            double scale = Math.Min((right - left) / (double)source.Width, (bottom - top) / (double)source.Height);
            int fittedWidth = Math.Clamp((int)Math.Round(source.Width * scale), 1, right - left);
            int fittedHeight = Math.Clamp((int)Math.Round(source.Height * scale), 1, bottom - top);
            var fitted = TensorSharp.Models.QwenImage.ImageIO.Resize(source, fittedWidth, fittedHeight);
            int x = left + (right - left - fittedWidth) / 2, y = top + (bottom - top - fittedHeight) / 2;
            for (int row = 0; row < fittedHeight; ++row)
                Array.Copy(fitted.Pixels, row * fittedWidth * 3, pixels, ((y + row) * width + x) * 3, fittedWidth * 3);
        }
        return new RgbImage(width, height, pixels);
    }

    /// <summary>Build an explicit frame-major static RGB video for the full
    /// temporal encoder and independent oracles. Production uses the encoder's
    /// equivalent static-video specialization to avoid repeated pixel storage.</summary>
    public static float[] RepeatSheet(RgbImage sheet, int frames, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(sheet);
        if (frames < MinimumFrames || (frames - 1) % 8 != 0)
            throw new ArgumentException("Ingredients static references require at least 121 RGB frames on the 8n+1 temporal grid.");
        if (sheet.Width <= 0 || sheet.Height <= 0 || sheet.Pixels == null || sheet.Pixels.Length != checked(sheet.Width * sheet.Height * 3) ||
            sheet.Pixels.Any(value => !float.IsFinite(value) || value < 0 || value > 1))
            throw new ArgumentException("The reference sheet must contain finite RGB01 pixels.");
        var result = new float[checked(sheet.Pixels.Length * frames)];
        for (int frame = 0; frame < frames; ++frame)
        {
            cancellationToken.ThrowIfCancellationRequested();
            sheet.Pixels.CopyTo(result, frame * sheet.Pixels.Length);
        }
        return result;
    }

    /// <summary>Append full normalized channel-major reference video latents.
    /// Initial noise already reserves all reference token slots. The canvas
    /// keeps its initial noise, references are clean at sigma one, and the
    /// caller's input and reference arrays remain unchanged.</summary>
    public static Ltx25DenoiserInput Apply(Ltx25DenoiserInput initial, Ltx25GenerationSettings settings,
        float[] channelMajorReference, int referenceLatentFrames, int referenceLatentHeight, int referenceLatentWidth)
        => Apply(initial, settings, [new ReferenceLatent(channelMajorReference, referenceLatentFrames, referenceLatentHeight, referenceLatentWidth)]);

    /// <summary>Append clean normalized reference-video latents after the generated canvas. Each clip keeps its own
    /// causal time positions; clips do not become one artificial continuous timeline.</summary>
    public static Ltx25DenoiserInput Apply(Ltx25DenoiserInput initial, Ltx25GenerationSettings settings,
        IReadOnlyList<ReferenceLatent> references)
    {
        ArgumentNullException.ThrowIfNull(initial); ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(references);
        if (references.Count == 0) throw new ArgumentException("Ingredients needs at least one encoded reference.", nameof(references));
        int minimumLatentFrames = (MinimumFrames - 1) / 8 + 1;
        foreach (ReferenceLatent reference in references)
            if (reference?.ChannelMajor == null || reference.Frames < minimumLatentFrames ||
                reference.Height != settings.LatentHeight || reference.Width != settings.LatentWidth)
                throw new ArgumentException("Ingredients requires at least 16 reference latent frames matching the canvas spatial geometry (downscale factor one).");
        int referenceFrames = checked(references.Sum(reference => reference.Frames));
        int referenceTokens = checked(referenceFrames * settings.LatentHeight * settings.LatentWidth);
        int tokens = checked(settings.VideoTokens + referenceTokens), canvasValues = checked(settings.VideoTokens * 128);
        if (references.Any(reference => reference.ChannelMajor.Length != checked(reference.Frames * reference.Height * reference.Width * 128) ||
                reference.ChannelMajor.Any(value => !float.IsFinite(value))) ||
            initial.VideoTokens != tokens || initial.Video == null || initial.Video.Length != checked(tokens * 128))
            throw new ArgumentException("Initial noise and normalized Ingredients reference latents have inconsistent geometry.");
        var video = (float[])initial.Video.Clone(); var clean = new float[video.Length];
        var mask = new float[tokens]; Array.Fill(mask, 1f, 0, settings.VideoTokens);
        var markers = new float[tokens]; Array.Fill(markers, 1f, 0, checked(settings.LatentHeight * settings.LatentWidth));
        int spatial = checked(settings.LatentHeight * settings.LatentWidth);
        var channelMajorReference = new float[checked(referenceTokens * 128)];
        int timeOffset = 0;
        foreach (ReferenceLatent reference in references)
        {
            int clipValues = checked(reference.Frames * spatial);
            for (int channel = 0; channel < 128; ++channel)
                Array.Copy(reference.ChannelMajor, channel * clipValues, channelMajorReference,
                    (channel * referenceFrames + timeOffset) * spatial, clipValues);
            timeOffset += reference.Frames;
        }
        float[] packed = Ltx25Geometry.PackVideo(channelMajorReference, referenceTokens, 128);
        packed.CopyTo(video, canvasValues); packed.CopyTo(clean, canvasValues);
        var positions = new float[checked(tokens * 6)];
        Ltx25Geometry.VideoPositions(settings.LatentFrames, settings.LatentHeight, settings.LatentWidth, settings.Fps).Boundaries.CopyTo(positions, 0);
        int tokenOffset = settings.VideoTokens;
        foreach (ReferenceLatent reference in references)
        {
            Ltx25Geometry.VideoPositions(reference.Frames, reference.Height, reference.Width, settings.Fps).Boundaries.CopyTo(positions, tokenOffset * 6);
            tokenOffset += checked(reference.Frames * reference.Height * reference.Width);
        }
        return new()
        {
            Video = video, VideoTokens = tokens, Audio = initial.Audio, AudioTokens = initial.AudioTokens,
            VideoContext = initial.VideoContext, VideoContextTokens = initial.VideoContextTokens,
            AudioContext = initial.AudioContext, AudioContextTokens = initial.AudioContextTokens,
            VideoPositions = new(tokens, 3, positions), AudioPositions = initial.AudioPositions,
            VideoSigma = initial.VideoSigma, AudioSigma = initial.AudioSigma, FrequencyPrecision = initial.FrequencyPrecision,
            VideoDenoiseMask = mask, CleanVideo = clean, KeyframeMask = markers, UseReferenceLora = true,
        };
    }
}
