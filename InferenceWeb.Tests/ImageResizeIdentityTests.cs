using System;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The Qwen image processor resizes to a multiple of the patch factor; a
/// screenshot that is already that size takes a direct composite-and-normalize
/// path. These tests pin it bit-for-bit to the bilinear formula it replaces
/// (every weight is exactly 0 or 1 there) and keep the general path covered.
/// </summary>
public class ImageResizeIdentityTests
{
    private static byte[] RandomRgba(int width, int height, int seed)
    {
        var rgba = new byte[width * height * 4];
        new Random(seed).NextBytes(rgba);
        // Include fully opaque and fully transparent pixels.
        for (int i = 3; i < rgba.Length; i += 16) rgba[i] = 255;
        for (int i = 7; i < rgba.Length; i += 32) rgba[i] = 0;
        return rgba;
    }

    // The original per-sample formula, kept independent of the code under test.
    private static float[] BilinearReference(byte[] rgba, int srcW, int srcH, int dstW, int dstH)
    {
        static float Channel(byte[] px, int pixelBase, int channel)
        {
            float alpha = px[pixelBase + 3] / 255f;
            float composited = px[pixelBase + channel] * alpha + 255f * (1f - alpha);
            return composited / 255f * 2f - 1f;
        }

        int pixels = dstW * dstH;
        var result = new float[3 * pixels];
        double xRatio = (double)srcW / dstW, yRatio = (double)srcH / dstH;
        for (int dy = 0; dy < dstH; dy++)
        {
            double srcY = (dy + 0.5) * yRatio - 0.5;
            int y0 = Math.Max(0, (int)srcY), y1 = Math.Min(srcH - 1, y0 + 1);
            double fy = srcY - y0;
            for (int dx = 0; dx < dstW; dx++)
            {
                double srcX = (dx + 0.5) * xRatio - 0.5;
                int x0 = Math.Max(0, (int)srcX), x1 = Math.Min(srcW - 1, x0 + 1);
                double fx = srcX - x0;
                for (int c = 0; c < 3; c++)
                {
                    float v00 = Channel(rgba, (y0 * srcW + x0) * 4, c);
                    float v01 = Channel(rgba, (y0 * srcW + x1) * 4, c);
                    float v10 = Channel(rgba, (y1 * srcW + x0) * 4, c);
                    float v11 = Channel(rgba, (y1 * srcW + x1) * 4, c);
                    double v = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) + v10 * (1 - fx) * fy + v11 * fx * fy;
                    result[c * pixels + dy * dstW + dx] = Math.Clamp((float)v, -1f, 1f);
                }
            }
        }
        return result;
    }

    [Theory]
    [InlineData(64, 32)]
    [InlineData(96, 64)]
    [InlineData(33, 17)]
    public void SameSize_MatchesBilinearReferenceExactly(int width, int height)
    {
        byte[] rgba = RandomRgba(width, height, width * 31 + height);
        float[] expected = BilinearReference(rgba, width, height, width, height);
        float[] actual = ImageProcessorUtils.ResizeRgbaToChannelFirstNormalized(rgba, width, height, width, height);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData(64, 32, 32, 32)]
    [InlineData(40, 30, 64, 48)]
    public void DifferentSize_StillMatchesBilinearReference(int srcW, int srcH, int dstW, int dstH)
    {
        byte[] rgba = RandomRgba(srcW, srcH, srcW + dstH);
        float[] expected = BilinearReference(rgba, srcW, srcH, dstW, dstH);
        float[] actual = ImageProcessorUtils.ResizeRgbaToChannelFirstNormalized(rgba, srcW, srcH, dstW, dstH);
        Assert.Equal(expected, actual);
    }
}
