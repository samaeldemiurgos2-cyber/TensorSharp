// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3.1 vision encoder on a synthetic minimax_m3 mmproj (2 heads of 80, 2 blocks),
// against an independent double-precision reference written in the mmproj's OWN layout:
// q/k rows as llama.cpp's converter permutes them, rotated by three 26-dim NeoX blocks
// at offsets 0 (t), 26 (h) and 52 (w), dims 78-79 passing through. The encoder instead
// re-permutes q/k at load into one half-split rotation, fuses q/k/v, sums the two
// patch kernels and runs the blocks through the shared whole-encoder GGML graph, so
// agreement here is what proves those load-time rewrites exact.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxM3VisionEncoderTests
    {
        private readonly ITestOutputHelper _output;

        public MiniMaxM3VisionEncoderTests(ITestOutputHelper output) => _output = output;

        private const int Patch = 14;
        private const int PatchDim = 3 * Patch * Patch;
        private const int Hidden = 160;
        private const int Heads = 2;
        private const int HeadDim = 80;
        private const int Ffn = 64;
        private const int Blocks = 2;
        private const int Proj = 24;
        private const int Merge = 2;
        private const float Eps = 1e-5f;
        private const double Theta = 10000.0;

        // ---- the load-time rewrites in isolation ---------------------------------------

        [Fact]
        public void HeadPermutation_IsAPermutationThatKeepsAxisPairsTogether()
        {
            int[] perm = MiniMaxM3VisionEncoder.BuildHeadPermutation(HeadDim);
            Assert.Equal(26, MiniMaxM3VisionEncoder.AxisDim(HeadDim));
            Assert.Equal(Enumerable.Range(0, HeadDim), perm.OrderBy(v => v));
            for (int j = 0; j < HeadDim / 2; j++)
            {
                int lo = perm[j], hi = perm[j + HeadDim / 2];
                if (lo < 78)
                {
                    Assert.Equal(lo / 26, hi / 26);       // same axis block
                    Assert.Equal(lo % 26 + 13, hi % 26);  // NeoX partner inside it
                }
                else
                    Assert.True(hi >= 78);
            }
        }

        [Fact]
        public void HalfSplitRotation_EqualsThreeNeoxBlocks()
        {
            const int gridH = 6, gridW = 8;
            int[] perm = MiniMaxM3VisionEncoder.BuildHeadPermutation(HeadDim);
            var (cos, sin) = MiniMaxM3VisionEncoder.BuildRopeTables(gridH, gridW, Merge, HeadDim, (float)Theta);
            var rng = new Random(7);
            int half = HeadDim / 2;
            int index = 0;
            for (int bh = 0; bh < gridH / Merge; bh++)
                for (int bw = 0; bw < gridW / Merge; bw++)
                    for (int mh = 0; mh < Merge; mh++)
                        for (int mw = 0; mw < Merge; mw++, index++)
                        {
                            double[] x = Enumerable.Range(0, HeadDim).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
                            double[] expected = NeoxThreeAxis(x, 0, bh * Merge + mh, bw * Merge + mw);

                            var y = new double[HeadDim];
                            for (int d = 0; d < HeadDim; d++)
                                y[d] = x[perm[d]];
                            for (int d = 0; d < half; d++)
                            {
                                double c = cos[index * half + d], s = sin[index * half + d];
                                double y0 = y[d], y1 = y[d + half];
                                y[d] = y0 * c - y1 * s;
                                y[d + half] = y0 * s + y1 * c;
                            }
                            for (int d = 0; d < HeadDim; d++)
                                Assert.InRange(Math.Abs(expected[perm[d]] - y[d]), 0, 1e-5);
                        }
        }

        /// <summary>llama.cpp's clip_graph_minimax_m3 rope on one head in mmproj layout.</summary>
        private static double[] NeoxThreeAxis(double[] x, int t, int h, int w)
        {
            var y = (double[])x.Clone();
            int[] pos = { t, h, w };
            for (int axis = 0; axis < 3; axis++)
            {
                int o = axis * 26;
                for (int i = 0; i < 13; i++)
                {
                    double angle = pos[axis] * Math.Pow(Theta, -2.0 * i / 26);
                    double a = x[o + i], b = x[o + i + 13];
                    y[o + i] = a * Math.Cos(angle) - b * Math.Sin(angle);
                    y[o + i + 13] = a * Math.Sin(angle) + b * Math.Cos(angle);
                }
            }
            return y;
        }

        // ---- whole encoder vs the reference ---------------------------------------------

        [Fact]
        public void ManagedPath_MatchesLayoutFaithfulReference()
        {
            string path = WriteMmproj();
            try
            {
                using var encoder = new MiniMaxM3VisionEncoder(path, new CpuAllocator(BlasEnum.DotNet));
                Check(encoder, path, gridH: 4, gridW: 6, relTol: 1e-5, cosTol: 0.9999999);
                Assert.False(encoder.LastEncodeUsedWholeEncoderGraph);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void PinnedGgmlBackend_WholeEncoderGraph_MatchesLayoutFaithfulReference()
        {
            GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
            string path = WriteMmproj();
            try
            {
                var allocator = new GgmlAllocator(new GgmlContext(new[] { 0 }, backend), 0);
                using var encoder = new MiniMaxM3VisionEncoder(path, allocator);
                double tol = backend == GgmlBackendType.Cpu ? 1e-4 : 5e-3;
                Check(encoder, path, gridH: 4, gridW: 6, relTol: tol, cosTol: 1 - tol);
                Assert.True(encoder.LastEncodeUsedWholeEncoderGraph,
                    "the 32-block whole-encoder GGML graph must run, not the per-block fallback");
                Assert.True(encoder.LastEncodeUsedNativeHead,
                    "the fused projector + patch-merge graph (TSGgml_MiniMaxM3VisionHeadF32) must run, not the per-op head");
                // A second image size reuses cached device weights with a new RoPE table.
                Check(encoder, path, gridH: 2, gridW: 8, relTol: tol, cosTol: 1 - tol);
            }
            finally { File.Delete(path); }
        }

        private void Check(MiniMaxM3VisionEncoder encoder, string path, int gridH, int gridW, double relTol, double cosTol)
        {
            int n = gridH * gridW;
            var rng = new Random(gridH * 31 + gridW);
            float[] patches = Enumerable.Range(0, n * PatchDim).Select(_ => (float)(rng.NextDouble() * 4 - 2)).ToArray();

            double[] expected = Reference(Weights.Read(path), patches, gridH, gridW);
            using Tensor actualTensor = encoder.EncodePatches(patches, gridH, gridW);
            Assert.Equal(new long[] { n / 4, Proj }, actualTensor.Sizes);
            float[] actual = actualTensor.GetElementsAsFloat(n / 4 * Proj);

            double diff = 0, norm = 0, minCos = 1;
            for (int r = 0; r < n / 4; r++)
            {
                double dot = 0, na = 0, nb = 0;
                for (int j = 0; j < Proj; j++)
                {
                    double a = actual[r * Proj + j], e = expected[r * Proj + j];
                    diff += (a - e) * (a - e);
                    norm += e * e;
                    dot += a * e; na += a * a; nb += e * e;
                }
                minCos = Math.Min(minCos, dot / Math.Sqrt(na * nb));
            }
            double rel = Math.Sqrt(diff / norm);
            _output.WriteLine($"grid {gridH}x{gridW}: rel-L2 {rel:E3}, min row cosine {minCos:F9}");
            Assert.True(rel < relTol, $"rel-L2 {rel:E3} >= {relTol:E1}");
            Assert.True(minCos > cosTol, $"min cosine {minCos:F9} <= {cosTol}");
        }

        // ---- independent reference ------------------------------------------------------

        private static double[] Reference(Weights w, float[] patches, int gridH, int gridW)
        {
            int n = gridH * gridW;
            var x = new double[n, Hidden];
            for (int p = 0; p < n; p++)
                for (int o = 0; o < Hidden; o++)
                {
                    double s = 0;
                    for (int k = 0; k < PatchDim; k++)
                        s += patches[p * PatchDim + k] * ((double)w["v.patch_embd.weight"][o * PatchDim + k] + w["v.patch_embd.weight.1"][o * PatchDim + k]);
                    x[p, o] = s;
                }
            x = LayerNorm(x, w["v.pre_ln.weight"], w["v.pre_ln.bias"]);

            var hpos = new int[n];
            var wpos = new int[n];
            int idx = 0;
            for (int bh = 0; bh < gridH / Merge; bh++)
                for (int bw = 0; bw < gridW / Merge; bw++)
                    for (int mh = 0; mh < Merge; mh++)
                        for (int mw = 0; mw < Merge; mw++, idx++)
                        {
                            hpos[idx] = bh * Merge + mh;
                            wpos[idx] = bw * Merge + mw;
                        }

            for (int b = 0; b < Blocks; b++)
            {
                string pre = $"v.blk.{b}";
                double[,] h1 = LayerNorm(x, w[$"{pre}.ln1.weight"], w[$"{pre}.ln1.bias"]);
                double[,] q = Linear(h1, w[$"{pre}.attn_q.weight"], w[$"{pre}.attn_q.bias"], Hidden, Hidden);
                double[,] k = Linear(h1, w[$"{pre}.attn_k.weight"], w[$"{pre}.attn_k.bias"], Hidden, Hidden);
                double[,] v = Linear(h1, w[$"{pre}.attn_v.weight"], w[$"{pre}.attn_v.bias"], Hidden, Hidden);
                for (int p = 0; p < n; p++)
                    for (int head = 0; head < Heads; head++)
                    {
                        double[] qh = Enumerable.Range(0, HeadDim).Select(d => q[p, head * HeadDim + d]).ToArray();
                        double[] kh = Enumerable.Range(0, HeadDim).Select(d => k[p, head * HeadDim + d]).ToArray();
                        qh = NeoxThreeAxis(qh, 0, hpos[p], wpos[p]);
                        kh = NeoxThreeAxis(kh, 0, hpos[p], wpos[p]);
                        for (int d = 0; d < HeadDim; d++)
                        {
                            q[p, head * HeadDim + d] = qh[d];
                            k[p, head * HeadDim + d] = kh[d];
                        }
                    }
                var attn = new double[n, Hidden];
                double scale = 1.0 / Math.Sqrt(HeadDim);
                for (int head = 0; head < Heads; head++)
                    for (int i = 0; i < n; i++)
                    {
                        var s = new double[n];
                        double max = double.NegativeInfinity;
                        for (int j = 0; j < n; j++)
                        {
                            double dot = 0;
                            for (int d = 0; d < HeadDim; d++)
                                dot += q[i, head * HeadDim + d] * k[j, head * HeadDim + d];
                            s[j] = dot * scale;
                            max = Math.Max(max, s[j]);
                        }
                        double sum = 0;
                        for (int j = 0; j < n; j++) { s[j] = Math.Exp(s[j] - max); sum += s[j]; }
                        for (int d = 0; d < HeadDim; d++)
                        {
                            double acc = 0;
                            for (int j = 0; j < n; j++)
                                acc += s[j] / sum * v[j, head * HeadDim + d];
                            attn[i, head * HeadDim + d] = acc;
                        }
                    }
                double[,] o1 = Linear(attn, w[$"{pre}.attn_out.weight"], w[$"{pre}.attn_out.bias"], Hidden, Hidden);
                Add(x, o1);
                double[,] h2 = LayerNorm(x, w[$"{pre}.ln2.weight"], w[$"{pre}.ln2.bias"]);
                double[,] up = Gelu(Linear(h2, w[$"{pre}.ffn_up.weight"], w[$"{pre}.ffn_up.bias"], Hidden, Ffn));
                Add(x, Linear(up, w[$"{pre}.ffn_down.weight"], w[$"{pre}.ffn_down.bias"], Ffn, Hidden));
            }

            double[,] z = Linear(Gelu(Linear(x, w["mm.1.weight"], w["mm.1.bias"], Hidden, Proj)),
                w["mm.2.weight"], w["mm.2.bias"], Proj, Proj);
            var grouped = new double[n / 4, 4 * Proj];
            for (int t = 0; t < n / 4; t++)
                for (int c = 0; c < 4; c++)
                    for (int j = 0; j < Proj; j++)
                        grouped[t, c * Proj + j] = z[t * 4 + c, j];
            double[,] e = Linear(Gelu(Linear(grouped, w["mm.merger.fc1.weight"], w["mm.merger.fc1.bias"], 4 * Proj, Proj)),
                w["mm.merger.fc2.weight"], w["mm.merger.fc2.bias"], Proj, Proj);
            var flat = new double[n / 4 * Proj];
            for (int t = 0; t < n / 4; t++)
                for (int j = 0; j < Proj; j++)
                    flat[t * Proj + j] = e[t, j];
            return flat;
        }

        private static double[,] LayerNorm(double[,] x, float[] g, float[] b)
        {
            int n = x.GetLength(0), d = x.GetLength(1);
            var y = new double[n, d];
            for (int r = 0; r < n; r++)
            {
                double mean = 0, var = 0;
                for (int j = 0; j < d; j++) mean += x[r, j];
                mean /= d;
                for (int j = 0; j < d; j++) var += (x[r, j] - mean) * (x[r, j] - mean);
                var /= d;
                double inv = 1.0 / Math.Sqrt(var + Eps);
                for (int j = 0; j < d; j++) y[r, j] = (x[r, j] - mean) * inv * g[j] + b[j];
            }
            return y;
        }

        /// <summary>x @ W^T + b with W stored [out][in] (GGUF ne = [in, out]).</summary>
        private static double[,] Linear(double[,] x, float[] wt, float[] b, int inDim, int outDim)
        {
            int n = x.GetLength(0);
            var y = new double[n, outDim];
            for (int r = 0; r < n; r++)
                for (int o = 0; o < outDim; o++)
                {
                    double s = b[o];
                    for (int i = 0; i < inDim; i++) s += x[r, i] * wt[o * inDim + i];
                    y[r, o] = s;
                }
            return y;
        }

        private static double[,] Gelu(double[,] x)
        {
            for (int r = 0; r < x.GetLength(0); r++)
                for (int j = 0; j < x.GetLength(1); j++)
                    x[r, j] = MiniMaxM3VisionEncoder.GeluErf(x[r, j]);
            return x;
        }

        private static void Add(double[,] x, double[,] y)
        {
            for (int r = 0; r < x.GetLength(0); r++)
                for (int j = 0; j < x.GetLength(1); j++)
                    x[r, j] += y[r, j];
        }

        // ---- synthetic minimax_m3 mmproj ------------------------------------------------

        private sealed class Weights : Dictionary<string, float[]>
        {
            public static Weights Read(string path)
            {
                var w = new Weights();
                using var gguf = new TensorSharp.Runtime.GgufFile(path);
                foreach (var info in gguf.Tensors.Values)
                {
                    var data = new float[info.NumElements];
                    if (info.Type == TensorSharp.Runtime.GgmlTensorType.BF16)
                    {
                        byte[] raw = gguf.ReadTensorData(info);
                        for (long i = 0; i < data.Length; i++)
                            data[i] = BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(raw, (int)(2 * i)) << 16);
                    }
                    else
                        gguf.ReadTensorDataToFloat32(info, data, info.NumElements);
                    w[info.Name] = data;
                }
                return w;
            }
        }

        /// <summary>The synthetic mmproj; <paramref name="proj"/> other than <see cref="Proj"/>
        /// only changes the projector width (the reference here assumes <see cref="Proj"/>).</summary>
        internal static string WriteMmproj(int proj = Proj)
        {
            string path = Path.Combine(Path.GetTempPath(), $"ts-minimax-m3-mmproj-{Guid.NewGuid():N}.gguf");
            var tensors = new List<(string Name, int[] Dims, float[] Data)>
            {
                Gen("v.patch_embd.weight", 0.02f, 0f, Patch, Patch, 3, Hidden),
                Gen("v.patch_embd.weight.1", 0.02f, 0f, Patch, Patch, 3, Hidden),
                Gen("v.pre_ln.weight", 0.1f, 1f, Hidden),
                Gen("v.pre_ln.bias", 0.05f, 0f, Hidden),
                Gen("mm.1.weight", 0.1f, 0f, Hidden, proj),
                Gen("mm.1.bias", 0.05f, 0f, proj),
                Gen("mm.2.weight", 0.2f, 0f, proj, proj),
                Gen("mm.2.bias", 0.05f, 0f, proj),
                Gen("mm.merger.fc1.weight", 0.1f, 0f, 4 * proj, proj),
                Gen("mm.merger.fc1.bias", 0.05f, 0f, proj),
                Gen("mm.merger.fc2.weight", 0.2f, 0f, proj, proj),
                Gen("mm.merger.fc2.bias", 0.05f, 0f, proj),
            };
            for (int b = 0; b < Blocks; b++)
            {
                string p = $"v.blk.{b}";
                tensors.Add(Gen($"{p}.ln1.weight", 0.1f, 1f, Hidden));
                tensors.Add(Gen($"{p}.ln1.bias", 0.05f, 0f, Hidden));
                foreach (string t in new[] { "attn_q", "attn_k", "attn_v", "attn_out" })
                {
                    tensors.Add(Gen($"{p}.{t}.weight", t == "attn_q" || t == "attn_k" ? 0.25f : 0.08f, 0f, Hidden, Hidden));
                    tensors.Add(Gen($"{p}.{t}.bias", 0.05f, 0f, Hidden));
                }
                tensors.Add(Gen($"{p}.ln2.weight", 0.1f, 1f, Hidden));
                tensors.Add(Gen($"{p}.ln2.bias", 0.05f, 0f, Hidden));
                tensors.Add(Gen($"{p}.ffn_up.weight", 0.1f, 0f, Hidden, Ffn));
                tensors.Add(Gen($"{p}.ffn_up.bias", 0.05f, 0f, Ffn));
                tensors.Add(Gen($"{p}.ffn_down.weight", 0.1f, 0f, Ffn, Hidden));
                tensors.Add(Gen($"{p}.ffn_down.bias", 0.05f, 0f, Hidden));
            }

            using var head = new MemoryStream();
            using (var bw = new BinaryWriter(head, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(0x46554747u);
                bw.Write(3u);
                bw.Write((ulong)tensors.Count);
                var entries = new List<(string Key, uint Type, Action<BinaryWriter> Write)>
                {
                    ("general.architecture", 8, w => Str(w, "clip")),
                    ("clip.projector_type", 8, w => Str(w, "minimax_m3")),
                    ("clip.vision.patch_size", 4, w => w.Write((uint)Patch)),
                    ("clip.vision.embedding_length", 4, w => w.Write((uint)Hidden)),
                    ("clip.vision.feed_forward_length", 4, w => w.Write((uint)Ffn)),
                    ("clip.vision.attention.head_count", 4, w => w.Write((uint)Heads)),
                    ("clip.vision.block_count", 4, w => w.Write((uint)Blocks)),
                    ("clip.vision.projection_dim", 4, w => w.Write((uint)proj)),
                    ("clip.vision.spatial_merge_size", 4, w => w.Write((uint)Merge)),
                    ("clip.vision.attention.layer_norm_epsilon", 6, w => w.Write(Eps)),
                };
                bw.Write((ulong)entries.Count);
                foreach (var (key, type, write) in entries)
                {
                    Str(bw, key);
                    bw.Write(type);
                    write(bw);
                }
                ulong offset = 0;
                foreach (var (name, dims, data) in tensors)
                {
                    Str(bw, name);
                    bw.Write((uint)dims.Length);
                    foreach (int d in dims) bw.Write((ulong)d);
                    bw.Write(IsBf16(name, dims) ? 30u : 0u);   // GGML_TYPE_BF16 / F32
                    bw.Write(offset);
                    offset += ((ulong)Raw(name, dims, data).Length + 31) / 32 * 32;
                }
            }

            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            byte[] headBytes = head.ToArray();
            fs.Write(headBytes);
            fs.Write(new byte[(32 - headBytes.Length % 32) % 32]);
            foreach (var (name, dims, data) in tensors)
            {
                byte[] raw = Raw(name, dims, data);
                fs.Write(raw);
                fs.Write(new byte[(32 - raw.Length % 32) % 32]);
            }
            return path;
        }

        /// <summary>The real mmproj stores 2-D projections as BF16 and the patch kernels,
        /// norms and biases as F32; the fixture does the same.</summary>
        private static bool IsBf16(string name, int[] dims) => dims.Length == 2 && name.EndsWith(".weight");

        private static byte[] Raw(string name, int[] dims, float[] data)
        {
            if (!IsBf16(name, dims))
            {
                var f32 = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, f32, 0, f32.Length);
                return f32;
            }
            var bf16 = new byte[data.Length * 2];
            for (int i = 0; i < data.Length; i++)
            {
                uint bits = (uint)BitConverter.SingleToInt32Bits(data[i]);
                ushort h = (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);   // round to nearest even
                bf16[2 * i] = (byte)h;
                bf16[2 * i + 1] = (byte)(h >> 8);
            }
            return bf16;
        }

        private static void Str(BinaryWriter w, string s)
        {
            byte[] b = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)b.Length);
            w.Write(b);
        }

        private static (string, int[], float[]) Gen(string name, float scale, float offset, params int[] dims)
        {
            long n = dims.Aggregate(1L, (a, d) => a * d);
            uint state = 2166136261;
            foreach (char c in name) { state ^= c; state *= 16777619; }
            state |= 1;
            var data = new float[n];
            for (long i = 0; i < n; i++)
            {
                state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                data[i] = offset + scale * ((state & 0xFFFF) / 32767.5f - 1f);
            }
            return (name, dims, data);
        }
    }
}
