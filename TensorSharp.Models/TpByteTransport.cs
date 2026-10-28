using System;

namespace TensorSharp.Models
{
    /// <summary>
    /// Moves raw bytes between tensor-parallel ranks over the float AllReduce the groups
    /// already provide. Three bytes become one float holding the integer
    /// <c>b0 | b1 &lt;&lt; 8 | b2 &lt;&lt; 16</c> (below 2^24, so exact in float32). When every
    /// rank packs the bytes it owns and zeros everywhere else, the sum is the whole
    /// buffer bit for bit: owners' bytes never overlap, so even a float that straddles two
    /// owners' ranges adds disjoint bit fields, and an integer sum below 2^24 is exact.
    /// Sending the bytes as their own float values instead would not be: an F16 pair or a
    /// quantized block reinterpreted as float32 can be a NaN, which a reduce may not keep
    /// bit-exact.
    /// </summary>
    internal static class TpByteTransport
    {
        public static int FloatCount(int byteCount) => (byteCount + 2) / 3;

        /// <summary>Pack <paramref name="bytes"/>; ranges the caller does not own must be zero.</summary>
        public static float[] Pack(ReadOnlySpan<byte> bytes)
        {
            var packed = new float[FloatCount(bytes.Length)];
            for (int i = 0, f = 0; i < bytes.Length; i += 3, f++)
            {
                int v = bytes[i];
                if (i + 1 < bytes.Length) v |= bytes[i + 1] << 8;
                if (i + 2 < bytes.Length) v |= bytes[i + 2] << 16;
                packed[f] = v;
            }
            return packed;
        }

        /// <summary>Unpack a reduced buffer into <paramref name="bytes"/>.</summary>
        /// <exception cref="InvalidOperationException">A value is not a 24-bit integer, so ranks'
        /// contributions overlapped or the reduce was not exact.</exception>
        public static void Unpack(ReadOnlySpan<float> packed, Span<byte> bytes)
        {
            if (packed.Length < FloatCount(bytes.Length))
                throw new ArgumentException("The packed buffer is shorter than the byte count.", nameof(packed));
            for (int i = 0, f = 0; i < bytes.Length; i += 3, f++)
            {
                float x = packed[f];
                int v = (int)x;
                if (v != x || v < 0 || v > 0xFFFFFF)
                    throw new InvalidOperationException(
                        $"Tensor-parallel byte transport received {x} at float {f}: the ranks' byte ranges overlapped.");
                bytes[i] = (byte)v;
                if (i + 1 < bytes.Length) bytes[i + 1] = (byte)(v >> 8);
                if (i + 2 < bytes.Length) bytes[i + 2] = (byte)(v >> 16);
            }
        }
    }
}
