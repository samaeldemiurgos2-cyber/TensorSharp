// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3.1 image preprocessing against transformers' MiniMaxM3VLImageProcessor
// (smart_resize, torchvision uint8 antialiased bicubic, fused rescale+normalize,
// merge-block patch order). The resize vectors below were produced by
// torchvision 0.23 / torch 2.8 on an AVX2 x86 CPU:
//   tvF.resize(img_u8[3,11,13], [h, w], interpolation=BICUBIC, antialias=True)
// with img[c,y,x] = (x*37 + y*91 + c*53 + x*y*7) % 256.
using System;
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests
{
    public class MiniMaxM3VisionProcessorTests
    {
        [Theory]
        // width, height -> resized width, height, tokens (HF rules: min 4 tokens, max 576, banker's rounding)
        [InlineData(1024, 768, 756, 560, 540)]
        [InlineData(768, 1024, 560, 756, 540)]
        [InlineData(1920, 1080, 896, 504, 576)]
        [InlineData(1280, 720, 896, 504, 576)]
        [InlineData(672, 672, 672, 672, 576)]
        [InlineData(640, 480, 644, 476, 391)]
        [InlineData(512, 512, 504, 504, 324)]
        [InlineData(256, 256, 252, 252, 81)]
        [InlineData(224, 224, 224, 224, 64)]
        [InlineData(56, 56, 56, 56, 4)]
        [InlineData(50, 50, 56, 56, 4)]
        // Python round(12.5) == 12: 350 -> 336, where half-away rounding would give 364.
        [InlineData(350, 350, 336, 336, 144)]
        [InlineData(300, 900, 308, 896, 352)]
        [InlineData(500, 333, 504, 336, 216)]
        public void SmartResize_MatchesHfProcessor(int width, int height, int resizedW, int resizedH, int tokens)
        {
            var processor = new MiniMaxM3ImageProcessor();
            var (h, w) = processor.SmartResize(height, width);
            Assert.Equal((resizedH, resizedW), (h, w));
            Assert.Equal(tokens, processor.ComputeImageTokenCount(width, height));
            Assert.True((long)h * w <= MiniMaxM3ImageProcessor.DefaultMaxPixels);
        }

        [Fact]
        public void SmartResize_RejectsExtremeAspectRatio()
        {
            var processor = new MiniMaxM3ImageProcessor();
            Assert.Throws<ArgumentException>(() => processor.SmartResize(10, 2010));
            Assert.Equal((28, 812), processor.SmartResize(10, 2000));   // exactly 200:1 is allowed
        }

        private static byte[] TinyImage()
        {
            const int W = 13, H = 11;
            var img = new byte[3 * W * H];
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        img[(c * H + y) * W + x] = (byte)((x * 37 + y * 91 + c * 53 + x * y * 7) % 256);
            return img;
        }

        private static readonly byte[] Expected5x7 = Parse(
            "89,127,112,134,104,116,122,124,135,147,139,128,133,134,108,122,135,158,105,151,131,130,127,132,123,128,130,130,145,176,123,98,159,106,136," +
            "126,132,144,114,135,121,158,134,135,141,129,123,137,148,161,140,123,140,141,123,99,94,130,115,119,135,102,100,126,98,154,130,121,162,112," +
            "150,136,153,130,138,137,107,121,139,132,121,127,134,149,126,107,114,130,112,97,144,138,156,119,140,114,151,155,126,138,100,149,136,116,163");

        private static readonly byte[] Expected4x29 = Parse(
            "75,85,111,129,115,103,124,151,140,107,97,112,131,159,170,127,76,79,110,130,144,140,109,89,110,136,134,122,117,105,108,114,123,139,149,134," +
            "114,131,170,171,131,115,152,180,153,115,111,125,124,113,116,143,163,158,144,131,120,115,124,121,116,113,119,124,115,105,120,148,150,121,109," +
            "133,156,157,148,125,96,98,130,148,144,137,136,135,133,130,129,122,131,149,165,166,165,170,173,159,134,119,116,106,75,62,106,162,174,162,145," +
            "119,106,108,113,112,114,129,148,156,134,129,116,115,140,162,142,110,108,126,146,174,175,123,80,91,119,139,153,144,111,100,127,148,139,128," +
            "144,167,176,164,162,151,139,122,114,138,167,154,113,111,149,164,128,102,131,168,151,108,104,138,154,146,133,117,108,126,154,165,137,130,106," +
            "97,132,165,157,132,109,83,91,137,161,140,116,109,112,129,153,151,126,108,101,99,100,101,98,93,93,110,111,114,117,121,119,98,75,96,145,169," +
            "169,161,141,121,108,102,107,122,138,158,166,159,146,125,107,105,110,111,118,129,156,170,143,116,121,140,163,185,175,119,89,107,133,153,164," +
            "147,114,114,143,158,150,137,118,102,109,123,128,127,122,110,107,135,156,129,92,104,147,150,111,95,129,155,128,92,103,136,144,129,114,97,96," +
            "130,166,162,143,136,142,141,139,136,133,131,131,133,135,135,123,94,91,134,169,156,126,103,83,93,135,152,128,112,129,152,156,152,151,177,157," +
            "101,73,123,175,169,141,128,122,106,78,79,129,174,177,164,147,125,115,117,120,122,127,142,157,163,164,162");

        // Three windows of the 3x23x17 upscale (the full vector is 1173 bytes): the first 34,
        // [560, 600) and the last 34. It saturates at both 0 and 255, which pins the clamp
        // after each pass.
        private static readonly byte[] Expected23x17Head = Parse(
            "0,13,43,71,98,144,179,211,233,34,0,37,66,113,145,174,197,13,35,68,101,130,124,139,175,200,58,35,78,104,95,116,151,179");
        private static readonly byte[] Expected23x17Mid = Parse(
            "173,226,175,105,82,158,107,123,154,110,169,140,129,196,144,68,69,135,255,126,79,156,214,71,45,117,172,245,145,62," +
            "137,207,90,32,96,174,128,148,204,148");
        private static readonly byte[] Expected23x17Tail = Parse(
            "243,127,117,179,84,132,92,50,144,203,148,111,158,60,109,191,224,255,152,148,198,47,149,98,9,111,231,148,96,198,47,77,190,255");

        private static byte[] Parse(string csv) => csv.Split(',').Select(byte.Parse).ToArray();

        [Fact]
        public void BicubicAntialias_MatchesTorchvisionUint8_Downscale()
        {
            byte[] actual = MiniMaxM3ImageProcessor.ResizeBicubicAntialiasU8(TinyImage(), 13, 11, 7, 5);
            Assert.Equal(Expected5x7, actual);
        }

        [Fact]
        public void BicubicAntialias_MatchesTorchvisionUint8_MixedAxes()
        {
            byte[] actual = MiniMaxM3ImageProcessor.ResizeBicubicAntialiasU8(TinyImage(), 13, 11, 29, 4);
            Assert.Equal(Expected4x29, actual);
        }

        [Fact]
        public void BicubicAntialias_MatchesTorchvisionUint8_UpscaleWithClamping()
        {
            byte[] actual = MiniMaxM3ImageProcessor.ResizeBicubicAntialiasU8(TinyImage(), 13, 11, 17, 23);
            Assert.Equal(3 * 23 * 17, actual.Length);
            Assert.Equal(Expected23x17Head, actual.Take(Expected23x17Head.Length).ToArray());
            Assert.Equal(Expected23x17Mid, actual.Skip(560).Take(Expected23x17Mid.Length).ToArray());
            Assert.Equal(Expected23x17Tail, actual.Skip(actual.Length - Expected23x17Tail.Length).ToArray());
            Assert.Contains((byte)0, actual);
            Assert.Contains((byte)255, actual);
        }

        [Fact]
        public void BicubicAntialias_SameSizeIsIdentityAndConstantStaysConstant()
        {
            byte[] img = TinyImage();
            Assert.Same(img, MiniMaxM3ImageProcessor.ResizeBicubicAntialiasU8(img, 13, 11, 13, 11));

            var flat = Enumerable.Repeat((byte)173, 3 * 40 * 30).ToArray();
            byte[] down = MiniMaxM3ImageProcessor.ResizeBicubicAntialiasU8(flat, 40, 30, 9, 7);
            Assert.All(down, v => Assert.Equal(173, v));
        }

        [Fact]
        public void Int16Weights_PrecisionKeepsPeakWeightInRange()
        {
            foreach (var (inSize, outSize) in new[] { (1024, 756), (13, 7), (11, 23), (1920, 896), (56, 56) })
            {
                MiniMaxM3ImageProcessor.ComputeInt16Weights(inSize, outSize,
                    out int[] starts, out int[] counts, out short[] weights, out int window, out int precision);
                Assert.InRange(precision, 1, 22);
                for (int i = 0; i < outSize; i++)
                {
                    Assert.InRange(starts[i], 0, inSize - 1);
                    Assert.InRange(starts[i] + counts[i], 1, inSize);
                    int sum = 0;
                    for (int k = 0; k < counts[i]; k++)
                        sum += weights[(long)i * window + k];
                    // Normalized weights, each rounded half away from zero.
                    Assert.InRange(sum, (1 << precision) - window, (1 << precision) + window);
                }
            }
        }

        [Fact]
        public void Patchify_UsesMergeBlockOrderAndFusedNormalization()
        {
            var processor = new MiniMaxM3ImageProcessor();
            const int H = 56, W = 84;   // 4 x 6 patches -> 2 x 3 merge blocks
            var planar = new byte[3 * H * W];
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                        planar[(c * H + y) * W + x] = (byte)((c * 50 + y / 14 * 10 + x / 14 + y % 14 + x % 14) % 256);

            MiniMaxM3ImagePatches patches = processor.Patchify(planar, H, W);
            Assert.Equal(4, patches.GridH);
            Assert.Equal(6, patches.GridW);
            Assert.Equal(6, patches.TokenCount);
            Assert.Equal(24 * 588, patches.Patches.Length);

            float[] mean = { 0.48145466f, 0.4578275f, 0.40821073f };
            float[] std = { 0.26862954f, 0.26130258f, 0.27577711f };
            int index = 0;
            for (int bh = 0; bh < 2; bh++)
                for (int bw = 0; bw < 3; bw++)
                    for (int mh = 0; mh < 2; mh++)
                        for (int mw = 0; mw < 2; mw++, index++)
                        {
                            int py = bh * 2 + mh, px = bw * 2 + mw;
                            for (int c = 0; c < 3; c++)
                                for (int y = 0; y < 14; y++)
                                    for (int x = 0; x < 14; x++)
                                    {
                                        byte u = planar[(c * H + py * 14 + y) * W + px * 14 + x];
                                        float expected = (u - mean[c] * 255f) / (std[c] * 255f);
                                        float actual = patches.Patches[index * 588 + (c * 14 + y) * 14 + x];
                                        Assert.Equal(expected, actual);
                                    }
                        }
        }

        [Fact]
        public void ProcessRgba_CompositesAlphaOverWhite()
        {
            var processor = new MiniMaxM3ImageProcessor();
            const int W = 56, H = 56;
            var rgba = new byte[4 * W * H];
            for (int i = 0; i < W * H; i++)
            {
                rgba[i * 4] = 0; rgba[i * 4 + 1] = 0; rgba[i * 4 + 2] = 0; rgba[i * 4 + 3] = 0;   // transparent black
            }
            MiniMaxM3ImagePatches patches = processor.ProcessRgba(rgba, W, H);
            float white = (255 - 0.48145466f * 255f) / (0.26862954f * 255f);
            Assert.Equal(white, patches.Patches[0]);
        }

        [Theory]
        [InlineData(0.0, 0.0)]
        [InlineData(0.3, 0.32862675945912745)]
        [InlineData(0.5, 0.5204998778130465)]
        [InlineData(1.0, 0.8427007929497149)]
        [InlineData(2.0, 0.9953222650189527)]
        [InlineData(3.0, 0.9999779095030014)]
        [InlineData(-1.7, -0.9837904585907746)]
        [InlineData(4.5, 0.9999999998033839)]
        public void Erf_MatchesReferenceValues(double x, double expected)
        {
            Assert.Equal(expected, MiniMaxM3VisionEncoder.Erf(x), 14);
        }
    }
}
