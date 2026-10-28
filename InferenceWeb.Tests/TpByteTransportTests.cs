using System;
using System.Linq;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class TpByteTransportTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(1000)]
    [InlineData(1001)]
    public void DisjointOwnersSum_ToTheWholeBuffer_BitForBit(int length)
    {
        var rng = new Random(length);
        var whole = new byte[length];
        rng.NextBytes(whole);
        // Bytes that read as NaN / -0 / Inf when taken as float32 or F16.
        if (length >= 4) { whole[0] = 0xFF; whole[1] = 0xFF; whole[2] = 0xFF; whole[3] = 0x7F; }

        // Three owners with uneven, unaligned ranges, so floats straddle owners.
        int cut1 = length / 3 + (length > 1 ? 1 : 0), cut2 = Math.Min(length, cut1 + length / 4 + 1);
        var owners = new[] { (0, cut1), (cut1, cut2), (cut2, length) };
        float[] sum = new float[TpByteTransport.FloatCount(length)];
        foreach (var (from, to) in owners)
        {
            var mine = new byte[length];
            Array.Copy(whole, from, mine, from, to - from);
            float[] packed = TpByteTransport.Pack(mine);
            for (int i = 0; i < sum.Length; i++) sum[i] += packed[i];
        }

        var round = new byte[length];
        TpByteTransport.Unpack(sum, round);
        Assert.Equal(whole, round);
    }

    [Fact]
    public void OverlappingContributions_AreDetected()
    {
        var a = new byte[] { 0xFF, 0xFF, 0xFF };
        float[] sum = TpByteTransport.Pack(a).Zip(TpByteTransport.Pack(a), (x, y) => x + y).ToArray();
        Assert.Throws<InvalidOperationException>(() => TpByteTransport.Unpack(sum, new byte[3]));
    }
}
