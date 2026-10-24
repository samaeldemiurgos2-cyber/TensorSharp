// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// MiniMax-M3.1 image preprocessing: a port of transformers' MiniMaxM3VLImageProcessor
// (image_processing_minimax_m3_vl.py, the torchvision backend), which is what the
// HF oracle (eng/minimax-m3-vision-reference.py) and vLLM run.
//
//   1. smart_resize: round both edges to factor = patch * merge = 28 with PYTHON
//      round() (half-to-even: 350 -> 336, not 364), then shrink by sqrt(area/max) or
//      grow by sqrt(min/area) with floor/ceil. min = 4 tokens (3136 px), max = 576
//      tokens (451584 px). llama.cpp's mtmd uses min 8 tokens and half-away rounding;
//      the two agree except for tiny images (56x56: HF 4 tokens, llama.cpp 9).
//   2. Resize: torchvision BICUBIC with antialias=True on the uint8 tensor. On CPU
//      torchvision hands uint8 straight to ATen's AVX2 kernel
//      (UpSampleKernelAVXAntialias.h), a fixed-point port of Pillow's resampler:
//      a = -0.5 cubic, filter support widened by the downscale factor, int16 weights
//      at a per-axis precision, horizontal pass first, a uint8 intermediate, and
//      round-half-up + clamp after each pass. Reproduced here exactly so the
//      preprocessed pixels match HF's byte for byte.
//   3. Normalize: HF fuses rescale into normalize, (u - 255*mean) / (255*std) in F32.
//   4. Patchify in merge-block order (bh, bw, mh, mw), each patch [c][y][x]. HF
//      duplicates every patch along a temporal axis of 2; the encoder folds the two
//      Conv3d slices into one kernel instead, so only one copy is produced.
using System;
using System.IO;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>One preprocessed image: patches in merge-block order plus the grid.</summary>
    public sealed class MiniMaxM3ImagePatches
    {
        /// <summary>[GridH * GridW, 3 * patch * patch] row-major, merge-block order,
        /// each patch laid out [channel][y][x].</summary>
        public required float[] Patches { get; init; }
        public required int GridH { get; init; }
        public required int GridW { get; init; }
        public required int MergeSize { get; init; }
        public int ResizedHeight { get; init; }
        public int ResizedWidth { get; init; }
        public int PatchCount => GridH * GridW;
        /// <summary>Embedding rows the encoder returns (= image placeholder tokens).</summary>
        public int TokenCount => GridH / MergeSize * (GridW / MergeSize);
    }

    public sealed class MiniMaxM3ImageProcessor
    {
        public const int DefaultPatchSize = 14;
        public const int DefaultMergeSize = 2;
        /// <summary>HF <c>size.shortest_edge</c>: 4 merged tokens' worth of pixels.</summary>
        public const long DefaultMinPixels = 4L * 28 * 28;
        /// <summary>HF <c>size.longest_edge</c>: 576 merged tokens = 672 x 672.</summary>
        public const long DefaultMaxPixels = 451584;

        // OPENAI_CLIP_MEAN / OPENAI_CLIP_STD (the mmproj stores the same F32 values).
        private static readonly float[] Mean = { 0.48145466f, 0.4578275f, 0.40821073f };
        private static readonly float[] Std = { 0.26862954f, 0.26130258f, 0.27577711f };

        public int PatchSize { get; }
        public int MergeSize { get; }
        public int Factor { get; }
        public long MinPixels { get; }
        public long MaxPixels { get; }

        public MiniMaxM3ImageProcessor(int patchSize = DefaultPatchSize, int mergeSize = DefaultMergeSize,
            long minPixels = DefaultMinPixels, long maxPixels = DefaultMaxPixels)
        {
            if (patchSize <= 0 || mergeSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch and merge sizes must be positive.");
            if (minPixels <= 0 || maxPixels < minPixels)
                throw new ArgumentOutOfRangeException(nameof(minPixels), "Pixel budgets must be positive and ordered.");
            PatchSize = patchSize;
            MergeSize = mergeSize;
            Factor = patchSize * mergeSize;
            MinPixels = minPixels;
            MaxPixels = maxPixels;
        }

        /// <summary>HF <c>smart_resize</c>, operation for operation (Python float math
        /// and banker's rounding). Returns the resized (height, width).</summary>
        public (int height, int width) SmartResize(int height, int width)
        {
            if (height <= 0 || width <= 0)
                throw new ArgumentException($"Image is empty ({width}x{height}).");
            if ((double)Math.Max(height, width) / Math.Min(height, width) > 200)
                throw new ArgumentException(
                    $"Image aspect ratio {width}x{height} exceeds 200:1, which MiniMax-M3.1's processor rejects.");

            int factor = Factor;
            long hBar = (long)Math.Round((double)height / factor, MidpointRounding.ToEven) * factor;
            long wBar = (long)Math.Round((double)width / factor, MidpointRounding.ToEven) * factor;
            if (hBar * wBar > MaxPixels)
            {
                double beta = Math.Sqrt((double)((long)height * width) / MaxPixels);
                hBar = Math.Max(factor, (long)Math.Floor(height / beta / factor) * factor);
                wBar = Math.Max(factor, (long)Math.Floor(width / beta / factor) * factor);
            }
            else if (hBar * wBar < MinPixels)
            {
                double beta = Math.Sqrt(MinPixels / (double)((long)height * width));
                hBar = (long)Math.Ceiling(height * beta / factor) * factor;
                wBar = (long)Math.Ceiling(width * beta / factor) * factor;
            }
            return (checked((int)hBar), checked((int)wBar));
        }

        public int ComputeImageTokenCount(int width, int height)
        {
            var (h, w) = SmartResize(height, width);
            return h / Factor * (w / Factor);
        }

        public int ComputeImageTokenCount(string imagePath)
        {
            var (width, height) = ImageProcessorUtils.ReadImageDimensions(imagePath);
            return ComputeImageTokenCount(width, height);
        }

        public MiniMaxM3ImagePatches ProcessImage(string imagePath)
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            byte[] rgba = ImageProcessorUtils.DecodeImageToRGBA(fileBytes, out int width, out int height);
            return ProcessRgba(rgba, width, height);
        }

        /// <summary>Full pipeline on decoded RGBA8 pixels (alpha composited over white,
        /// as transformers' <c>convert_to_rgb</c> does).</summary>
        public MiniMaxM3ImagePatches ProcessRgba(byte[] rgba, int width, int height)
        {
            ArgumentNullException.ThrowIfNull(rgba);
            if (rgba.LongLength != 4L * width * height)
                throw new ArgumentException("RGBA buffer does not match the image size.", nameof(rgba));
            var (resizedH, resizedW) = SmartResize(height, width);
            byte[] planar = RgbaToPlanarRgb(rgba, width, height);
            byte[] resized = ResizeBicubicAntialiasU8(planar, width, height, resizedW, resizedH);
            return Patchify(resized, resizedH, resizedW);
        }

        /// <summary>Normalize planar uint8 RGB already at a factor-aligned size and lay it
        /// out as merge-block-ordered patches.</summary>
        public MiniMaxM3ImagePatches Patchify(byte[] planarRgb, int height, int width)
        {
            ArgumentNullException.ThrowIfNull(planarRgb);
            if (height % Factor != 0 || width % Factor != 0)
                throw new ArgumentException($"Image size {width}x{height} is not a multiple of {Factor}.");
            if (planarRgb.LongLength != 3L * height * width)
                throw new ArgumentException("Planar buffer does not match the image size.", nameof(planarRgb));

            int p = PatchSize, m = MergeSize;
            int gridH = height / p, gridW = width / p;
            int patchDim = 3 * p * p;
            long plane = (long)height * width;
            var patches = new float[(long)gridH * gridW * patchDim];

            // HF fuses rescale into normalize: mean/std are scaled by 255 in F32 first.
            var mean255 = new float[3];
            var std255 = new float[3];
            for (int c = 0; c < 3; c++)
            {
                mean255[c] = Mean[c] * 255f;
                std255[c] = Std[c] * 255f;
            }

            int blocksW = gridW / m;
            Parallel.For(0, gridH / m, bh =>
            {
                for (int bw = 0; bw < blocksW; bw++)
                    for (int mh = 0; mh < m; mh++)
                        for (int mw = 0; mw < m; mw++)
                        {
                            long index = (((long)bh * blocksW + bw) * m + mh) * m + mw;
                            long dst = index * patchDim;
                            int y0 = (bh * m + mh) * p, x0 = (bw * m + mw) * p;
                            for (int c = 0; c < 3; c++)
                            {
                                float mu = mean255[c], sd = std255[c];
                                for (int y = 0; y < p; y++)
                                {
                                    long src = c * plane + (long)(y0 + y) * width + x0;
                                    for (int x = 0; x < p; x++)
                                        patches[dst++] = (planarRgb[src + x] - mu) / sd;
                                }
                            }
                        }
            });

            return new MiniMaxM3ImagePatches
            {
                Patches = patches,
                GridH = gridH,
                GridW = gridW,
                MergeSize = m,
                ResizedHeight = height,
                ResizedWidth = width,
            };
        }

        /// <summary>RGBA8 to planar RGB, compositing transparent pixels over white.</summary>
        internal static byte[] RgbaToPlanarRgb(byte[] rgba, int width, int height)
        {
            long plane = (long)width * height;
            var planar = new byte[3 * plane];
            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    long i = (long)y * width + x;
                    int a = rgba[i * 4 + 3];
                    for (int c = 0; c < 3; c++)
                    {
                        int v = rgba[i * 4 + c];
                        if (a != 255)
                            v = (v * a + 255 * (255 - a) + 127) / 255;
                        planar[c * plane + i] = (byte)v;
                    }
                }
            });
            return planar;
        }

        // ---- torchvision / ATen uint8 antialiased bicubic ---------------------------------

        /// <summary>One axis of ATen's <c>_compute_index_ranges_int16_weights</c>: per
        /// output pixel the first source index, the tap count and int16 weights at
        /// <paramref name="precision"/> fractional bits.</summary>
        internal static void ComputeInt16Weights(int inSize, int outSize,
            out int[] starts, out int[] counts, out short[] weights, out int window, out int precision)
        {
            double scale = (double)inSize / outSize;
            double support = scale >= 1.0 ? 2.0 * scale : 2.0;
            window = (int)Math.Ceiling(support) * 2 + 1;
            double invScale = scale >= 1.0 ? 1.0 / scale : 1.0;

            starts = new int[outSize];
            counts = new int[outSize];
            var wd = new double[(long)outSize * window];
            double wtMax = 0;
            for (int i = 0; i < outSize; i++)
            {
                double center = scale * (i + 0.5);
                long xmin = Math.Max((long)(center - support + 0.5), 0L);
                long xsize = Math.Min((long)(center + support + 0.5), inSize) - xmin;
                xsize = Math.Clamp(xsize, 0, window);
                double total = 0;
                long b = (long)i * window;
                for (int j = 0; j < xsize; j++)
                {
                    double w = BicubicFilter((j + xmin - center + 0.5) * invScale);
                    wd[b + j] = w;
                    total += w;
                }
                if (total != 0)
                    for (int j = 0; j < xsize; j++)
                    {
                        wd[b + j] /= total;
                        wtMax = Math.Max(wtMax, wd[b + j]);
                    }
                starts[i] = (int)xmin;
                counts[i] = (int)xsize;
            }

            // Largest precision whose rounded peak weight still fits int16.
            for (precision = 0; precision < 22; precision++)
            {
                int next = (int)(0.5 + wtMax * (1 << (precision + 1)));
                if (next >= (1 << 15))
                    break;
            }

            weights = new short[wd.Length];
            double one = 1 << precision;
            for (long k = 0; k < wd.Length; k++)
            {
                double v = wd[k] * one;
                weights[k] = (short)(v < 0 ? (int)(-0.5 + v) : (int)(0.5 + v));
            }
        }

        private static double BicubicFilter(double x)
        {
            // ATen HelperInterpCubic::aa_filter, a = -0.5 (Pillow's constant).
            const double a = -0.5;
            x = Math.Abs(x);
            if (x < 1.0)
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
            if (x < 2.0)
                return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
            return 0.0;
        }

        /// <summary>Antialiased bicubic resize of planar uint8 RGB with torchvision's
        /// CPU uint8 semantics (an axis whose size is unchanged is skipped).</summary>
        internal static byte[] ResizeBicubicAntialiasU8(byte[] planar, int srcW, int srcH, int dstW, int dstH)
        {
            if (srcW == dstW && srcH == dstH)
                return planar;

            byte[] current = planar;
            int curW = srcW;
            if (dstW != srcW)
            {
                ComputeInt16Weights(srcW, dstW, out int[] xs, out int[] xc, out short[] xw, out int xWin, out int xp);
                var horiz = new byte[3L * srcH * dstW];
                int initial = 1 << (xp - 1);
                Parallel.For(0, 3 * srcH, row =>
                {
                    long srcRow = (long)row * srcW;
                    long dstRow = (long)row * dstW;
                    for (int x = 0; x < dstW; x++)
                    {
                        int acc = initial;
                        long wb = (long)x * xWin;
                        int s = xs[x];
                        for (int k = 0; k < xc[x]; k++)
                            acc += current[srcRow + s + k] * xw[wb + k];
                        horiz[dstRow + x] = ClampByte(acc >> xp);
                    }
                });
                current = horiz;
                curW = dstW;
            }

            if (dstH != srcH)
            {
                ComputeInt16Weights(srcH, dstH, out int[] ys, out int[] yc, out short[] yw, out int yWin, out int yp);
                var vert = new byte[3L * dstH * curW];
                int initial = 1 << (yp - 1);
                byte[] input = current;
                Parallel.For(0, 3 * dstH, row =>
                {
                    int c = row / dstH, y = row - c * dstH;
                    long planeBase = (long)c * srcH * curW;
                    long dstRow = (long)row * curW;
                    long wb = (long)y * yWin;
                    int s = ys[y], n = yc[y];
                    for (int x = 0; x < curW; x++)
                    {
                        int acc = initial;
                        for (int k = 0; k < n; k++)
                            acc += input[planeBase + (long)(s + k) * curW + x] * yw[wb + k];
                        vert[dstRow + x] = ClampByte(acc >> yp);
                    }
                });
                current = vert;
            }
            return current;
        }

        private static byte ClampByte(int v) => (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
}
