using TensorSharp;

namespace InferenceWeb.Tests;

// Pins the vectorized F16 kernels in TensorComputePrimitives to the scalar
// reference behavior: converters must stay bit-identical (the KV-cache
// append/expand path depends on exact round-trips), dot/scale-add must match
// within summation-order drift.
public sealed class TensorComputePrimitivesF16Tests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(1000)]
    public unsafe void F16ToF32MatchesScalarBitIdentical(int count)
    {
        const int sourceOffset = 3, destinationOffset = 5, guard = 11;
        var source = Enumerable.Repeat((ushort)0x55AA, count + sourceOffset + guard).ToArray();
        var actual = Enumerable.Repeat(37.25f, count + destinationOffset + guard).ToArray();
        for (int i = 0; i < count; ++i) source[sourceOffset + i] = unchecked((ushort)(i * 17));
        ushort[] original = (ushort[])source.Clone();
        fixed (ushort* input = source)
        fixed (float* output = actual)
            TensorComputePrimitives.F16ToF32(output + destinationOffset, input + sourceOffset, count);
        // The odd multiplier visits all 65536 Half encodings, including signed
        // zeros, subnormals, infinities and every signaling/quiet NaN payload.
        // Compare F32 bits so NaN payload changes cannot hide behind equality.
        for (int i = 0; i < count; ++i)
        {
            float expected = (float)BitConverter.UInt16BitsToHalf(source[sourceOffset + i]);
            Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual[destinationOffset + i]));
        }
        Assert.All(actual.Take(destinationOffset), value => Assert.Equal(37.25f, value));
        Assert.All(actual.Skip(destinationOffset + count), value => Assert.Equal(37.25f, value));
        Assert.Equal(original, source);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(1000)]
    public unsafe void F32ToF16MatchesScalarBitIdentical(int count)
    {
        float[] values = BuildF32Corpus(count, seed: 0xF16);
        var actual = new ushort[count + 16];
        var expected = new ushort[count];
        for (int i = 0; i < count; ++i)
            expected[i] = BitConverter.HalfToUInt16Bits((System.Half)values[i]);
        fixed (float* input = values)
        fixed (ushort* output = actual)
            TensorComputePrimitives.F32ToF16(output + 3, input, count);
        for (int i = 0; i < count; ++i)
            Assert.Equal(expected[i], actual[3 + i]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(1000)]
    public unsafe void F32F16RoundTripMatchesScalar(int count)
    {
        float[] values = BuildF32Corpus(count, seed: 0xF7);
        var actual = new float[count];
        var expected = new float[count];
        var half = new ushort[Math.Max(count, 1)];
        fixed (float* input = values)
        fixed (float* outActual = actual)
        fixed (ushort* h = half)
        {
            TensorComputePrimitives.F32ToF16(h, input, count);
            TensorComputePrimitives.F16ToF32(outActual, h, count);
        }
        for (int i = 0; i < count; ++i)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((System.Half)values[i]);
            expected[i] = (float)BitConverter.UInt16BitsToHalf(bits);
            Assert.Equal(
                BitConverter.SingleToInt32Bits(expected[i]),
                BitConverter.SingleToInt32Bits(actual[i]));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(1000)]
    [InlineData(4096)]
    public unsafe void DotF32F16MatchesScalar(int count)
    {
        var rng = new Random(1234);
        float[] a = RandomFloats(rng, count, 2.0f);
        ushort[] b = RandomHalves(rng, count);
        float actual, expected = 0;
        fixed (float* ap = a)
        fixed (ushort* bp = b)
            actual = TensorComputePrimitives.DotF32F16(ap, bp, count);
        for (int i = 0; i < count; ++i)
            expected += a[i] * (float)BitConverter.UInt16BitsToHalf(b[i]);
        AssertRelEqual(expected, actual);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(128)]
    [InlineData(1000)]
    public unsafe void Dot4F32F16MatchesScalar(int count)
    {
        var rng = new Random(99);
        float[][] a = [RandomFloats(rng, count, 2.0f), RandomFloats(rng, count, 2.0f),
                       RandomFloats(rng, count, 2.0f), RandomFloats(rng, count, 2.0f)];
        ushort[] b = RandomHalves(rng, count);
        float r0, r1, r2, r3;
        fixed (float* a0 = a[0])
        fixed (float* a1 = a[1])
        fixed (float* a2 = a[2])
        fixed (float* a3 = a[3])
        fixed (ushort* bp = b)
            TensorComputePrimitives.Dot4F32F16(a0, a1, a2, a3, bp, count, out r0, out r1, out r2, out r3);
        for (int k = 0; k < 4; ++k)
        {
            float expected = 0;
            for (int i = 0; i < count; ++i)
                expected += a[k][i] * (float)BitConverter.UInt16BitsToHalf(b[i]);
            AssertRelEqual(expected, k == 0 ? r0 : k == 1 ? r1 : k == 2 ? r2 : r3);
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(128)]
    [InlineData(1000)]
    public unsafe void ScaleAddF16MatchesScalar(int count)
    {
        var rng = new Random(4242);
        float[] dst = RandomFloats(rng, count, 1.0f);
        float[] refDst = (float[])dst.Clone();
        ushort[] src = RandomHalves(rng, count);
        float weight = 0.37f;
        fixed (float* dp = dst)
        fixed (ushort* sp = src)
            TensorComputePrimitives.ScaleAddF16(dp, sp, weight, count);
        for (int i = 0; i < count; ++i)
            refDst[i] += weight * (float)BitConverter.UInt16BitsToHalf(src[i]);
        for (int i = 0; i < count; ++i)
            AssertRelEqual(refDst[i], dst[i]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(128)]
    [InlineData(1000)]
    public unsafe void ScaleAdd4F16MatchesScalar(int count)
    {
        var rng = new Random(777);
        float[][] dst = [RandomFloats(rng, count, 1.0f), RandomFloats(rng, count, 1.0f),
                         RandomFloats(rng, count, 1.0f), RandomFloats(rng, count, 1.0f)];
        float[][] refDst = [.. dst.Select(d => (float[])d.Clone())];
        ushort[] src = RandomHalves(rng, count);
        float[] w = [0.11f, -2.5f, 0.0f, 1.25f];
        fixed (float* d0 = dst[0])
        fixed (float* d1 = dst[1])
        fixed (float* d2 = dst[2])
        fixed (float* d3 = dst[3])
        fixed (ushort* sp = src)
            TensorComputePrimitives.ScaleAdd4F16(d0, d1, d2, d3, sp, w[0], w[1], w[2], w[3], count);
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < count; ++i)
                refDst[k][i] += w[k] * (float)BitConverter.UInt16BitsToHalf(src[i]);
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < count; ++i)
                AssertRelEqual(refDst[k][i], dst[k][i]);
    }

    // Widening is exact; only summation order may drift (as with the F32 Dot).
    // 5e-5 relative tolerates blocked accumulation over thousands of terms
    // while still catching any conversion bug (which would err by ~1e-3+).
    private static void AssertRelEqual(float expected, float actual)
    {
        // NaN payload propagation depends on summation order (as with the F32
        // Dot vs scalar); any-NaN == any-NaN here.
        if (float.IsNaN(expected) && float.IsNaN(actual)) return;
        float tol = 5e-5f * MathF.Max(1.0f, MathF.Abs(expected));
        Assert.True(MathF.Abs(expected - actual) <= tol, $"expected {expected:R}, got {actual:R}");
    }

    private static float[] BuildF32Corpus(int count, int seed)
    {
        var rng = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; ++i)
        {
            switch (i % 8)
            {
                case 0: values[i] = (float)BitConverter.UInt16BitsToHalf((ushort)(i * 17)); break;
                case 1: values[i] = (float)(rng.NextDouble() - 0.5) * 131072f; break;
                case 2: values[i] = BitConverter.Int32BitsToSingle(rng.Next()); break;
                case 3: values[i] = (i % 2 == 0 ? 1f : -1f) * BitConverter.Int32BitsToSingle(0x007FFFFF & rng.Next()); break;
                case 4: values[i] = float.PositiveInfinity; break;
                case 5: values[i] = float.NegativeInfinity; break;
                case 6: values[i] = BitConverter.Int32BitsToSingle(unchecked((int)0xFF800000) | (rng.Next() & 0x003FFFFF)); break;
                default: values[i] = i % 2 == 0 ? 0f : -0f; break;
            }
        }
        if (count > 0)
        {
            values[0] = float.MaxValue;
            if (count > 1) values[1] = float.MinValue;
            if (count > 2) values[2] = float.Epsilon;
        }
        return values;
    }

    private static float[] RandomFloats(Random rng, int count, float scale)
    {
        var a = new float[count];
        for (int i = 0; i < count; ++i)
            a[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return a;
    }

    private static ushort[] RandomHalves(Random rng, int count)
    {
        var b = new ushort[count];
        for (int i = 0; i < count; ++i)
            b[i] = (ushort)rng.Next(0, 65536);
        return b;
    }
}
