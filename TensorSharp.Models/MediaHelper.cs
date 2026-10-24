// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.ExceptionServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models.Media;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models
{
    /// <summary>
    /// Video frame sampling for the chat vision models and the video-generation pipelines.
    ///
    /// <para>Everything here is policy and pure computation: which source frames to take,
    /// how many, and turning them into PNG files under a memory budget. The frames themselves
    /// come from <see cref="MediaCodecs.Video"/> — OpenCV on desktop, whatever the host
    /// registered elsewhere — so this file compiles for every target.</para>
    /// </summary>
    public static class MediaHelper
    {
        /// <summary>Decode the exact final frame of a video into the models' canonical
        /// RGB01 image representation. This is used when a new generation continues
        /// from a reference clip; the sparse vision-frame sampler may stop before the
        /// clip's actual end.</summary>
        public static RgbImage ExtractLastVideoFrame(string videoPath)
        {
            if (string.IsNullOrWhiteSpace(videoPath)) throw new ArgumentNullException(nameof(videoPath));
            IVideoDecoder decoder = MediaCodecs.Video;
            VideoInfo info = decoder.Probe(videoPath);
            if (info.FrameCount <= 0)
                throw new InvalidOperationException($"Invalid video: fps={info.Fps}, frames={info.FrameCount}");

            RgbImage last = null;
            decoder.ReadFrames(videoPath, new[] { info.FrameCount - 1 }, (index, bytes, width, height, stride, layout) =>
            {
                int channels = layout is PixelLayout.Bgra or PixelLayout.Rgba ? 4 : 3;
                if (width <= 0 || height <= 0 || stride < width * channels || bytes.Length < (long)stride * height)
                    throw new InvalidOperationException("The video decoder returned an invalid final-frame buffer.");
                bool bgr = layout is PixelLayout.Bgr or PixelLayout.Bgra;
                var rgb = new float[checked(width * height * 3)];
                float[] alpha = channels == 4 ? new float[checked(width * height)] : null;
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x)
                    {
                        int source = y * stride + x * channels;
                        int target = (y * width + x) * 3;
                        rgb[target] = bytes[source + (bgr ? 2 : 0)] / 255f;
                        rgb[target + 1] = bytes[source + 1] / 255f;
                        rgb[target + 2] = bytes[source + (bgr ? 0 : 2)] / 255f;
                        if (alpha != null) alpha[y * width + x] = bytes[source + 3] / 255f;
                    }
                last = new RgbImage(width, height, rgb, alpha);
            });
            return last ?? throw new InvalidOperationException($"Could not decode the final frame of '{videoPath}'.");
        }

        /// <summary>
        /// Frames sampled per second of video when no explicit rate is supplied.
        /// Overridable via the <c>VIDEO_SAMPLE_FPS</c> environment variable.
        /// </summary>
        public const double DefaultVideoSampleFps = 1.0;

        /// <summary>
        /// Default upper bound on the number of extracted frames. <c>0</c> means
        /// "no cap": extraction is purely time-based at the sampling fps. Set the
        /// <c>VIDEO_MAX_FRAMES</c> environment variable to a positive value to
        /// bound long videos.
        /// </summary>
        public const int DefaultVideoMaxFrames = 0;

        /// <summary>
        /// Resolves the sampling rate (frames per second of video) from the
        /// <c>VIDEO_SAMPLE_FPS</c> environment variable, falling back to
        /// <see cref="DefaultVideoSampleFps"/> when unset or invalid.
        /// </summary>
        public static double GetConfiguredVideoSampleFps(double fallback = DefaultVideoSampleFps)
        {
            string raw = Environment.GetEnvironmentVariable("VIDEO_SAMPLE_FPS");
            if (!string.IsNullOrWhiteSpace(raw) &&
                double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed) &&
                parsed > 0)
            {
                return parsed;
            }

            return fallback > 0 ? fallback : DefaultVideoSampleFps;
        }

        /// <summary>
        /// Resolves the optional upper bound on extracted frames from the
        /// <c>VIDEO_MAX_FRAMES</c> environment variable. Returns <c>0</c> (no cap)
        /// when unset or invalid so that extraction stays purely time-based.
        /// </summary>
        public static int GetConfiguredMaxVideoFrames(int fallback = DefaultVideoMaxFrames)
        {
            string raw = Environment.GetEnvironmentVariable("VIDEO_MAX_FRAMES");
            if (!string.IsNullOrWhiteSpace(raw) &&
                int.TryParse(raw, out int parsed) &&
                parsed > 0)
            {
                return parsed;
            }

            return fallback > 0 ? fallback : 0;
        }

        /// <summary>
        /// Extracts frames from a video using time-based sampling: one frame every
        /// <c>1 / fps</c> seconds (default 1 fps), so the frame count scales with the
        /// clip's duration. When <paramref name="maxFrames"/> resolves to a positive
        /// value (via the <c>VIDEO_MAX_FRAMES</c> env var or an explicit argument) it
        /// acts as an upper bound, evenly down-selecting; otherwise every sampled
        /// frame is kept.
        ///
        /// <para>Frames land in a fresh temp directory as <c>frame_0001.png</c>, ... —
        /// the form the CLI wants, where the returned full paths are handed straight to
        /// the image processor. A caller that must SERVE the frames (the web upload
        /// endpoint) has to place them somewhere addressable instead; see the
        /// <see cref="ExtractVideoFrames(string, string, string, int, double)"/>
        /// overload.</para>
        /// </summary>
        /// <param name="videoPath">Path to the source video file.</param>
        /// <param name="maxFrames">
        /// Optional cap on the number of frames. <c>&lt;= 0</c> resolves from
        /// <c>VIDEO_MAX_FRAMES</c> (default: no cap).
        /// </param>
        /// <param name="fps">
        /// Sampling rate in frames per second of video. <c>&lt;= 0</c> resolves from
        /// <c>VIDEO_SAMPLE_FPS</c> (default: 1 fps).
        /// </param>
        public static List<string> ExtractVideoFrames(string videoPath, int maxFrames = 0, double fps = 0.0)
        {
            string tempDir = Path.Combine(Path.GetTempPath(), $"frames_{Guid.NewGuid():N}");
            return ExtractVideoFrames(videoPath, tempDir, DefaultFramePrefix, maxFrames, fps);
        }

        /// <summary>Default base name for extracted frames, giving <c>frame_0001.png</c>.</summary>
        internal const string DefaultFramePrefix = "frame";

        /// <summary>
        /// <see cref="ExtractVideoFrames(string, int, double)"/> writing into a caller-chosen
        /// directory under a caller-chosen base name, so the frames land somewhere the caller
        /// can address them.
        ///
        /// <para>This is what the web upload endpoint needs: a chat attachment is referenced by
        /// its bare file name and resolved against the upload directory, and the same name is
        /// what <c>/uploads/&lt;name&gt;</c> serves. Frames written to a private temp directory
        /// satisfy neither — the reference does not resolve, the thumbnail 404s, and the bytes
        /// escape both the upload quota and its TTL sweep. Naming them after the upload's own
        /// GUID also keeps two clips from colliding on <c>frame_0001.png</c>.</para>
        /// </summary>
        /// <param name="videoPath">Path to the source video file.</param>
        /// <param name="outputDirectory">Directory the PNGs are written to (created if missing).</param>
        /// <param name="namePrefix">Base name for the emitted files (sanitized); frames are <c>{prefix}_0001.png</c>, ...</param>
        /// <param name="maxFrames">Optional cap on the number of frames. <c>&lt;= 0</c> resolves from <c>VIDEO_MAX_FRAMES</c>.</param>
        /// <param name="fps">Sampling rate in frames per second of video. <c>&lt;= 0</c> resolves from <c>VIDEO_SAMPLE_FPS</c>.</param>
        public static List<string> ExtractVideoFrames(
            string videoPath, string outputDirectory, string namePrefix,
            int maxFrames = 0, double fps = 0.0)
            => ExtractVideoFramesCore(videoPath, outputDirectory, namePrefix, maxFrames, fps, null);

        /// <summary>Sample video images and retain frame index / probed FPS in seconds (approximate for variable-rate clips).</summary>
        public static (List<string> Paths, List<double> Timestamps) ExtractVideoFramesWithTimestamps(
            string videoPath, string outputDirectory, string namePrefix,
            int maxFrames = 0, double fps = 0.0)
        {
            var timestamps = new List<double>();
            var paths = ExtractVideoFramesCore(videoPath, outputDirectory, namePrefix, maxFrames, fps, timestamps);
            return (paths, timestamps);
        }

        private static List<string> ExtractVideoFramesCore(
            string videoPath, string outputDirectory, string namePrefix,
            int maxFrames, double fps, List<double> timestamps)
        {
            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentNullException(nameof(videoPath));
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            if (maxFrames <= 0)
                maxFrames = GetConfiguredMaxVideoFrames();
            if (fps <= 0)
                fps = GetConfiguredVideoSampleFps();

            Directory.CreateDirectory(outputDirectory);
            string prefix = SanitizeName(namePrefix);

            IVideoDecoder decoder = MediaCodecs.Video;
            VideoInfo info = decoder.Probe(videoPath);
            double videoFps = info.Fps;
            int totalFrames = info.FrameCount;
            if (videoFps <= 0 || totalFrames <= 0)
                throw new Exception($"Invalid video: fps={videoFps}, frames={totalFrames}");

            // Time-based candidate sampling: pick one frame every (videoFps / fps)
            // source frames, i.e. one frame per (1 / fps) seconds of wall-clock time.
            int frameInterval = Math.Max(1, (int)Math.Round(videoFps / fps));
            var candidateFrames = new List<int>();
            for (int frameIdx = 0; frameIdx < totalFrames; frameIdx += frameInterval)
                candidateFrames.Add(frameIdx);

            // Keep every time-sampled frame by default. VIDEO_MAX_FRAMES (when > 0)
            // is an optional upper bound that evenly down-selects to protect against
            // runaway frame counts / context blow-up on very long clips.
            List<int> selectedPositions;
            if (maxFrames > 0 && candidateFrames.Count > maxFrames)
            {
                selectedPositions = SelectEvenlySpacedIndices(candidateFrames.Count, maxFrames);
            }
            else
            {
                selectedPositions = new List<int>(candidateFrames.Count);
                for (int i = 0; i < candidateFrames.Count; i++)
                    selectedPositions.Add(i);
            }

            var wanted = new List<int>(selectedPositions.Count);
            foreach (int pos in selectedPositions)
                wanted.Add(candidateFrames[pos]);

            return DecodeAndEncodeFrames(decoder, videoPath, wanted, outputDirectory, prefix, timestamps, videoFps);
        }

        /// <summary>
        /// Decoding is strictly sequential (the provider hands frames over one at a time on
        /// this thread), but PNG encoding is not: deflate + Adler-32 + the write cost about as
        /// much as the decode itself and depend on nothing but their own frame. Each decoded
        /// frame is therefore copied out to a private scanline buffer inside the callback and
        /// encoded on the pool while the next frame decodes, with a semaphore bounding how
        /// many raw frames are in flight so a long clip cannot balloon the heap. Blocking in
        /// the callback is the back-pressure: the decoder cannot run ahead of the encoders.
        /// </summary>
        private static List<string> DecodeAndEncodeFrames(
            IVideoDecoder decoder, string videoPath, List<int> wantedFrames, string outputDirectory, string prefix,
            List<double> timestamps = null, double sourceFps = 0)
        {
            var encodes = new List<Task>();
            var frames = new List<string>();
            SemaphoreSlim slots = null;

            try
            {
                decoder.ReadFrames(videoPath, wantedFrames, (index, pixels, width, height, stride, layout) =>
                {
                    if (slots == null)
                    {
                        // The frame size is only known once one has been decoded, and it
                        // is what the in-flight bound is priced against.
                        int limit = InFlightLimit(width, height);
                        slots = new SemaphoreSlim(limit, limit);
                    }

                    string framePath = Path.Combine(outputDirectory, $"{prefix}_{frames.Count + 1:D4}.png");
                    byte[] scanlines = BuildRgbaScanlines(pixels, width, height, stride, layout);
                    frames.Add(framePath);
                    timestamps?.Add(index / sourceFps);

                    slots.Wait();
                    encodes.Add(Task.Run(() =>
                    {
                        try { File.WriteAllBytes(framePath, PngCodec.EncodeScanlines(scanlines, width, height, 4)); }
                        finally { slots.Release(); }
                    }));
                });

                try
                {
                    Task.WaitAll(encodes.ToArray());
                }
                catch (AggregateException ex)
                {
                    // A half-written frame is worse than none: the caller would hand the model a
                    // truncated PNG. Drop everything this call produced and surface the cause.
                    foreach (string path in frames)
                    {
                        try { File.Delete(path); } catch { /* best effort */ }
                    }
                    ExceptionDispatchInfo.Capture(ex.InnerException ?? ex).Throw();
                    throw;
                }

                return frames;
            }
            finally
            {
                slots?.Dispose();
            }
        }

        /// <summary>
        /// How many frames may be awaiting encoding at once. One per core keeps the pool
        /// fed, but each in-flight frame holds a <c>width * height * 4</c> scanline buffer,
        /// and that term is the one that runs away: at 4K it is 33 MB a frame, so a
        /// core-count-only bound would put a third of a gigabyte on the heap for a clip
        /// whose frames are 8x bigger than 1080p. Cap by a memory budget as well, so the
        /// footprint stays flat in resolution and only the parallelism gives way.
        /// </summary>
        internal static int InFlightLimit(int width, int height)
        {
            long frameBytes = Math.Max(1L, (long)width * height * 4);
            long affordable = InFlightScanlineBudgetBytes / frameBytes;
            int byBudget = (int)Math.Min(int.MaxValue, Math.Max(1, affordable));
            return Math.Max(1, Math.Min(MaxParallelFrameEncodes, byBudget));
        }

        /// <summary>The budget a server gets: enough for a 4K clip to keep several frames in
        /// flight while staying well inside its working set.</summary>
        public const long DefaultInFlightScanlineBudgetBytes = 256L * 1024 * 1024;

        private static long _inFlightScanlineBudgetBytes = DefaultInFlightScanlineBudgetBytes;
        private static int _maxParallelFrameEncodes = Environment.ProcessorCount;

        /// <summary>
        /// Heap budget for scanline buffers awaiting PNG encoding. Defaults to
        /// <see cref="DefaultInFlightScanlineBudgetBytes"/>; a phone sharing an ~8 GB jetsam
        /// limit with a resident model should lower it (48-64 MB keeps a 1080p clip at two or
        /// three frames in flight). Must be positive.
        /// </summary>
        public static long InFlightScanlineBudgetBytes
        {
            get => _inFlightScanlineBudgetBytes;
            set
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException(nameof(value), "the in-flight scanline budget must be positive");
                _inFlightScanlineBudgetBytes = value;
            }
        }

        /// <summary>
        /// Upper bound on frames encoded to PNG concurrently. Defaults to
        /// <see cref="Environment.ProcessorCount"/>; the memory budget above may lower the
        /// effective number further for large frames. Must be at least 1.
        /// </summary>
        public static int MaxParallelFrameEncodes
        {
            get => _maxParallelFrameEncodes;
            set
            {
                if (value < 1)
                    throw new ArgumentOutOfRangeException(nameof(value), "at least one frame encode must be allowed");
                _maxParallelFrameEncodes = value;
            }
        }

        /// <summary>
        /// Largest step between consecutive wanted frames (counting the first from frame 0).
        /// Governs whether stepping forward or seeking is the cheaper way to reach them.
        /// </summary>
        private static int MaxGap(IReadOnlyList<int> wantedFrames)
        {
            int max = 0, previous = 0;
            foreach (int frameIdx in wantedFrames)
            {
                int gap = frameIdx - previous;
                if (gap > max) max = gap;
                previous = frameIdx;
            }
            return max;
        }

        /// <summary>
        /// Frame distance past which seeking beats stepping.
        ///
        /// <para>Measured decode-only cost per sampled frame, three clips, gaps 1..96:
        /// stepping wins by 1.9x on 768x576 H.264 and 1.8x on a 3000-frame 640x480 H.264
        /// at a gap of 12, and the crossover sits at a gap of 24-32 for both. The binding
        /// constraint is all-intra footage (MJPEG), where every frame is a keyframe and a
        /// seek has nothing to decode forward over: there the crossover falls to ~12. A
        /// gap of 12 is therefore the largest value that is still a clear win on the
        /// inter-coded clips people actually upload while costing all-intra footage at
        /// most 6%. Sampling at the default 1 fps puts the gap at the source frame rate —
        /// past the threshold, onto the seek path — which the same measurements show is
        /// within a few percent of stepping either way.</para>
        /// </summary>
        public const int SequentialStepMaxGap = 12;

        /// <summary>
        /// The access policy an <see cref="IVideoDecoder"/> should follow for
        /// <paramref name="frameIndices"/>: <c>true</c> when every frame is at most
        /// <see cref="SequentialStepMaxGap"/> past the previous one (so stepping forward is
        /// cheaper and exact), <c>false</c> when the list is sparse — or repeats an index, which
        /// stepping cannot deliver twice — so each frame should be sought directly.
        /// </summary>
        public static bool PrefersSequentialStepping(IReadOnlyList<int> frameIndices)
        {
            if (frameIndices == null)
                throw new ArgumentNullException(nameof(frameIndices));

            int previous = -1;
            foreach (int frameIdx in frameIndices)
            {
                if (frameIdx <= previous)
                    return false;
                previous = frameIdx;
            }
            return MaxGap(frameIndices) <= SequentialStepMaxGap;
        }

        private static string SanitizeName(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return DefaultFramePrefix;

            name = Path.GetFileNameWithoutExtension(name);
            var sb = new System.Text.StringBuilder(name.Length);
            foreach (char c in name)
                sb.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');

            string cleaned = sb.ToString().Trim('_');
            return cleaned.Length == 0 ? DefaultFramePrefix : cleaned;
        }

        /// <summary>Sample a video onto a fixed frame rate, returning the extracted
        /// frame files and the rate the source carried.
        ///
        /// <para>Unlike <see cref="ExtractVideoFrames"/>, which samples sparsely for a
        /// vision-language model, this reproduces a clip on a target timeline: frame
        /// <c>i</c> of the result is the source frame at <c>i * sourceFps /
        /// targetFps</c>, so a 30 fps source played onto a 24 fps grid holds frames
        /// rather than dropping content. A generative video model conditions on
        /// motion, and sparse sampling would misrepresent it.</para>
        ///
        /// <para>Also accepts a DIRECTORY of image files, which is how reference clips
        /// are usually delivered; the files are taken in sorted order and assumed to be
        /// at <paramref name="targetFps"/> already unless <paramref name="sourceFpsHint"/>
        /// says otherwise.</para></summary>
        /// <param name="maxFrames">Upper bound on the returned count. 0 = no bound.</param>
        public static (List<string> Frames, double SourceFps) ExtractFramesAtRate(
            string videoPath, double targetFps, int maxFrames = 0, double sourceFpsHint = 0)
        {
            if (string.IsNullOrWhiteSpace(videoPath))
                throw new ArgumentNullException(nameof(videoPath));
            if (targetFps <= 0) throw new ArgumentOutOfRangeException(nameof(targetFps));

            if (Directory.Exists(videoPath))
            {
                var files = new List<string>(Directory.GetFiles(videoPath));
                files.RemoveAll(f => !IsImageFile(f));
                files.Sort(StringComparer.Ordinal);
                if (files.Count == 0)
                    throw new InvalidOperationException(
                        $"'{videoPath}' contains no image files to use as video frames.");
                double dirFps = sourceFpsHint > 0 ? sourceFpsHint : targetFps;
                return (Resample(files, dirFps, targetFps, maxFrames), dirFps);
            }

            if (!File.Exists(videoPath))
                throw new FileNotFoundException($"video not found: {videoPath}", videoPath);

            IVideoDecoder decoder = MediaCodecs.Video;
            VideoInfo info = decoder.Probe(videoPath);
            double sourceFps = info.Fps;
            if (sourceFpsHint > 0) sourceFps = sourceFpsHint;
            int totalFrames = info.FrameCount;
            if (sourceFps <= 0 || totalFrames <= 0)
                throw new InvalidOperationException(
                    $"invalid video: fps={sourceFps}, frames={totalFrames}");

            int wanted = (int)Math.Round(totalFrames * targetFps / sourceFps);
            if (maxFrames > 0) wanted = Math.Min(wanted, maxFrames);
            wanted = Math.Max(1, wanted);

            // Hold-and-drop onto the target timeline; a repeated index is a held frame.
            var indices = new List<int>(wanted);
            for (int i = 0; i < wanted; i++)
                indices.Add(Math.Min(totalFrames - 1, (int)Math.Floor(i * sourceFps / targetFps)));

            string tempDir = Path.Combine(Path.GetTempPath(), $"refvid_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);
            var frames = new List<string>(wanted);
            decoder.ReadFrames(videoPath, indices, (index, pixels, width, height, stride, layout) =>
            {
                string framePath = Path.Combine(tempDir, $"frame_{frames.Count + 1:D5}.png");
                byte[] scanlines = BuildRgbaScanlines(pixels, width, height, stride, layout);
                File.WriteAllBytes(framePath, PngCodec.EncodeScanlines(scanlines, width, height, 4));
                frames.Add(framePath);
            });
            if (frames.Count == 0)
                throw new InvalidOperationException($"no frames could be read from {videoPath}");
            return (frames, sourceFps);
        }

        private static bool IsImageFile(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            return ext is ".png" or ".jpg" or ".jpeg" or ".bmp" or ".webp";
        }

        // Hold-and-drop resampling of an already-extracted frame list.
        private static List<string> Resample(List<string> files, double fromFps, double toFps, int maxFrames)
        {
            int wanted = (int)Math.Round(files.Count * toFps / fromFps);
            if (maxFrames > 0) wanted = Math.Min(wanted, maxFrames);
            wanted = Math.Max(1, wanted);
            var outp = new List<string>(wanted);
            for (int i = 0; i < wanted; i++)
                outp.Add(files[Math.Min(files.Count - 1, (int)Math.Floor(i * fromFps / toFps))]);
            return outp;
        }

        public static List<int> SelectEvenlySpacedIndices(int count, int maxCount)
        {
            var indices = new List<int>();
            if (count <= 0 || maxCount <= 0)
                return indices;

            if (count <= maxCount)
            {
                for (int i = 0; i < count; i++)
                    indices.Add(i);
                return indices;
            }

            if (maxCount == 1)
            {
                indices.Add(count / 2);
                return indices;
            }

            double step = (double)(count - 1) / (maxCount - 1);
            int previous = -1;
            for (int i = 0; i < maxCount; i++)
            {
                int idx = (int)Math.Round(i * step);
                if (idx <= previous)
                    idx = previous + 1;
                if (idx >= count)
                    idx = count - 1;

                indices.Add(idx);
                previous = idx;
            }

            return indices;
        }

        /// <summary>
        /// Copies a decoded frame out into PNG scanline form — RGBA rows each prefixed with
        /// a filter-type byte — severing every tie to the decoder's buffer, which it reuses
        /// for the next frame. Must run inside the frame callback; everything downstream of
        /// it is pure computation over this buffer.
        /// </summary>
        internal static byte[] BuildRgbaScanlines(byte[] src, int width, int height, int stride, PixelLayout layout)
        {
            int channels = layout is PixelLayout.Bgra or PixelLayout.Rgba ? 4 : 3;
            bool bgr = layout is PixelLayout.Bgr or PixelLayout.Bgra;
            if (stride < width * channels)
                throw new ArgumentOutOfRangeException(nameof(stride), $"stride {stride} is shorter than a {width}-pixel {layout} row");
            if ((long)stride * (height - 1) + (long)width * channels > src.Length)
                throw new ArgumentException("frame buffer is smaller than its declared geometry", nameof(src));

            int rowStride = 1 + width * 4;
            byte[] rawRows = new byte[height * rowStride];

            int rOff = bgr ? 2 : 0, bOff = bgr ? 0 : 2;
            for (int y = 0; y < height; y++)
            {
                int dstRowStart = y * rowStride;
                rawRows[dstRowStart] = 0; // PNG filter: None
                int rowStart = y * stride;
                for (int x = 0; x < width; x++)
                {
                    int dstOff = dstRowStart + 1 + x * 4;
                    int srcOff = rowStart + x * channels;
                    rawRows[dstOff]     = src[srcOff + rOff]; // R
                    rawRows[dstOff + 1] = src[srcOff + 1];    // G
                    rawRows[dstOff + 2] = src[srcOff + bOff]; // B
                    rawRows[dstOff + 3] = channels == 4 ? src[srcOff + 3] : (byte)255;
                }
            }

            return rawRows;
        }
    }
}
