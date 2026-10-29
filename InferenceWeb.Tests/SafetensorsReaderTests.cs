// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests
{
    /// <summary>
    /// Tests for the native safetensors reader (<see cref="SafetensorsFile"/> / <see cref="SafetensorsModel"/>).
    /// The synthetic tests are self-contained (build a tiny file on disk) and always run. The real
    /// Qwen-Image-2.1 VAE header is read by QwenImage21CompanionValidationTests.
    /// </summary>
    public class SafetensorsReaderTests
    {
        // ---- self-contained synthetic tests (no model files needed) -----------------------------

        [Fact]
        public void Synthetic_RoundTrips_AllDtypes()
        {
            // BF16 with a non-multiple-of-16 length to exercise both the vectorised body and scalar tail.
            float[] bf16Vals = Enumerable.Range(0, 37).Select(i => (i - 18) * 0.3125f).ToArray();
            float[] f16Vals = { 0f, 1f, -2.5f, 1234.0f, -0.001953125f };
            float[] f32Vals = { float.MinValue, -1.25f, 0f, 3.4028235e38f, MathF.PI };
            int[] i32Vals = { -2147483648, -1, 0, 7, 2147483647 };
            long[] i64Vals = { long.MinValue, -1, 0, 5, long.MaxValue };

            var tensors = new (string, SafetensorDtype, long[], byte[])[]
            {
                ("w.bf16", SafetensorDtype.BF16, new long[] { 37 }, Bf16Bytes(bf16Vals)),
                ("w.f16",  SafetensorDtype.F16,  new long[] { 5 },  F16Bytes(f16Vals)),
                ("w.f32",  SafetensorDtype.F32,  new long[] { 5 },  F32Bytes(f32Vals)),
                ("w.i32",  SafetensorDtype.I32,  new long[] { 5 },  I32Bytes(i32Vals)),
                ("w.i64",  SafetensorDtype.I64,  new long[] { 5 },  I64Bytes(i64Vals)),
            };

            string path = Path.Combine(Path.GetTempPath(), $"ts_synth_{Guid.NewGuid():N}.safetensors");
            WriteSafetensors(path, tensors);
            try
            {
                using var f = new SafetensorsFile(path);
                Assert.Equal(5, f.Tensors.Count);
                Assert.True(f.HasTensor("w.bf16"));
                Assert.Equal(new long[] { 37 }, f.TensorShape("w.bf16"));
                Assert.Equal(SafetensorDtype.F16, f.Tensors["w.f16"].Dtype);

                // BF16 truncation is exact: the read value is the input float with its low 16 mantissa bits cleared.
                AssertSeqEqual(bf16Vals.Select(TruncToBf16).ToArray(), f.ReadFloat32("w.bf16"));
                AssertSeqEqual(f16Vals.Select(v => (float)(Half)v).ToArray(), f.ReadFloat32("w.f16"));
                AssertSeqEqual(f32Vals, f.ReadFloat32("w.f32"));
                AssertSeqEqual(i32Vals.Select(v => (float)v).ToArray(), f.ReadFloat32("w.i32"));
                AssertSeqEqual(i64Vals.Select(v => (float)v).ToArray(), f.ReadFloat32("w.i64"));

                // SafetensorsModel over a single file routes by name identically.
                using var m = SafetensorsModel.Open(path);
                Assert.Equal(5, m.Count);
                AssertSeqEqual(f32Vals, m.ReadFloat32("w.f32"));
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void Synthetic_RejectsCorruptOffsets()
        {
            string path = Path.Combine(Path.GetTempPath(), $"ts_bad_{Guid.NewGuid():N}.safetensors");
            // data_offsets claim 100 bytes but only 4 bytes of data follow -> must throw.
            var header = new JsonObject
            {
                ["t"] = new JsonObject
                {
                    ["dtype"] = "F32",
                    ["shape"] = new JsonArray(25),
                    ["data_offsets"] = new JsonArray(0, 100),
                }
            };
            byte[] hb = Encoding.UTF8.GetBytes(header.ToJsonString());
            using (var fs = File.Create(path))
            {
                fs.Write(BitConverter.GetBytes((ulong)hb.Length));
                fs.Write(hb);
                fs.Write(new byte[4]);
            }
            try { Assert.ThrowsAny<Exception>(() => new SafetensorsFile(path)); }
            finally { File.Delete(path); }
        }

        /// <summary>
        /// Header claims that would size an allocation are refused as malformed before anything is
        /// sized from them: a header length with the top bit set (negative as a long, so it passed
        /// the length check and failed as an OverflowException), and a shape that is negative or
        /// whose F32 byte count wraps to the few bytes it was given. Both shapes were accepted, and
        /// ReadFloat32 then sized its array from the shape.
        /// </summary>
        [Theory]
        [InlineData("header-length", "safetensors header length 18446744073709551600 is out of range")]
        [InlineData("shape-wraps", "safetensors tensor 't' has shape [1,4611686018427387905], which is negative or too large for F32")]
        [InlineData("shape-negative", "safetensors tensor 't' has shape [-1,-4], which is negative or too large for F32")]
        public void Synthetic_RejectsHeaderClaimsThatWouldSizeAnAllocation(string kind, string reason)
        {
            string path = Path.Combine(Path.GetTempPath(), $"ts_claims_{Guid.NewGuid():N}.safetensors");
            // [-1,-4] counts 4 values and [1, 2^62 + 1] wraps to 1 F32 value: both match their data.
            string shape = kind == "shape-negative" ? "[-1,-4]" : "[1,4611686018427387905]";
            int bytes = kind == "shape-negative" ? 16 : 4;
            byte[] hb = Encoding.UTF8.GetBytes($"{{\"t\":{{\"dtype\":\"F32\",\"shape\":{shape},\"data_offsets\":[0,{bytes}]}}}}");
            using (var fs = File.Create(path))
            {
                fs.Write(BitConverter.GetBytes(kind == "header-length" ? 0xFFFF_FFFF_FFFF_FFF0UL : (ulong)hb.Length));
                fs.Write(hb);
                fs.Write(new byte[bytes]);
            }
            try { Assert.Contains(reason, Assert.Throws<InvalidDataException>(() => new SafetensorsFile(path)).Message); }
            finally { File.Delete(path); }
        }

        // ---- helpers ------------------------------------------------------------------------------

        private static void AssertSeqEqual(float[] expected, float[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.True(BitConverter.SingleToUInt32Bits(expected[i]) == BitConverter.SingleToUInt32Bits(actual[i]),
                    $"index {i}: expected {expected[i]} got {actual[i]}");
        }

        private static float TruncToBf16(float v)
        {
            uint bits = BitConverter.SingleToUInt32Bits(v);
            return BitConverter.UInt32BitsToSingle(bits & 0xFFFF0000u);
        }

        private static byte[] Bf16Bytes(float[] vals)
        {
            var b = new byte[vals.Length * 2];
            for (int i = 0; i < vals.Length; i++)
            {
                ushort hi = (ushort)(BitConverter.SingleToUInt32Bits(vals[i]) >> 16);
                BitConverter.GetBytes(hi).CopyTo(b, i * 2);
            }
            return b;
        }

        private static byte[] F16Bytes(float[] vals)
        {
            var b = new byte[vals.Length * 2];
            for (int i = 0; i < vals.Length; i++)
                BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)vals[i])).CopyTo(b, i * 2);
            return b;
        }

        private static byte[] F32Bytes(float[] vals)
        {
            var b = new byte[vals.Length * 4];
            Buffer.BlockCopy(vals, 0, b, 0, b.Length);
            return b;
        }

        private static byte[] I32Bytes(int[] vals)
        {
            var b = new byte[vals.Length * 4];
            Buffer.BlockCopy(vals, 0, b, 0, b.Length);
            return b;
        }

        private static byte[] I64Bytes(long[] vals)
        {
            var b = new byte[vals.Length * 8];
            Buffer.BlockCopy(vals, 0, b, 0, b.Length);
            return b;
        }

        private static void WriteSafetensors(string path, (string name, SafetensorDtype dt, long[] shape, byte[] data)[] tensors)
        {
            var header = new JsonObject();
            long offset = 0;
            var blobs = new List<byte[]>();
            foreach (var (name, dt, shape, data) in tensors)
            {
                var shapeArr = new JsonArray();
                foreach (var d in shape) shapeArr.Add(d);
                header[name] = new JsonObject
                {
                    ["dtype"] = DtypeName(dt),
                    ["shape"] = shapeArr,
                    ["data_offsets"] = new JsonArray(offset, offset + data.Length),
                };
                offset += data.Length;
                blobs.Add(data);
            }
            byte[] hb = Encoding.UTF8.GetBytes(header.ToJsonString());
            using var fs = File.Create(path);
            fs.Write(BitConverter.GetBytes((ulong)hb.Length));
            fs.Write(hb);
            foreach (var blob in blobs) fs.Write(blob);
        }

        private static string DtypeName(SafetensorDtype dt) => dt switch
        {
            SafetensorDtype.F64 => "F64",
            SafetensorDtype.F32 => "F32",
            SafetensorDtype.F16 => "F16",
            SafetensorDtype.BF16 => "BF16",
            SafetensorDtype.I64 => "I64",
            SafetensorDtype.I32 => "I32",
            SafetensorDtype.I16 => "I16",
            SafetensorDtype.I8 => "I8",
            SafetensorDtype.U8 => "U8",
            SafetensorDtype.Bool => "BOOL",
            _ => throw new ArgumentOutOfRangeException(nameof(dt)),
        };
    }
}
