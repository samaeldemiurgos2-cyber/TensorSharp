// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-M3.1 vision parity with HuggingFace transformers on the real BF16 mmproj.
// The oracle directory comes from eng/minimax-m3-vision-reference.py (transformers'
// own MiniMaxM3VLImageProcessor, MiniMaxM3VLVisionModel and projector on synthetic
// PNGs). Per image this checks, on the GGML backend the process pins:
//   1. the token count and the preprocessed pixels (TS processor vs HF processor);
//   2. TS processor + TS encoder vs HF F32 compute (end to end);
//   3. HF pixels + TS encoder vs HF F32 compute (resampler excluded);
//   4. the stages apart: TS tower on HF pixels vs HF's tower rows, and the TS
//      projector + merge on HF's tower rows vs HF's embeddings;
// and reports the HF BF16-compute row for scale. The F32 HF run is the reference:
// the mmproj weights are BF16, so F32 compute on them is what both sides can agree on
// exactly; HF's own BF16 compute is itself only ~0.99 mean cosine from it.
//
// Tolerances follow the backend's GEMM precision. ggml_cpu, and ggml_cuda with TF32
// disabled (NVIDIA_TF32_OVERRIDE=0), are true F32 and land at the F32 reordering
// floor: mean cosine >= 0.99999 and rel-L2 1e-5..3e-3 (a few rows of some images are
// ill-conditioned enough that two F32 summation orders move them by ~2e-3). ggml_cuda
// runs F32 matmuls as TF32 (ggml sets CUBLAS_TF32_TENSOR_OP_MATH), which leaves mean
// cosine >= 0.9991 with a handful of sensitive rows lower: measured on an A40, at most
// 5 of 576 rows (1.1% of tall_300x900's 352) below 0.99 and one row at 0.67, all of it
// from the tower (the projector stage stays at rel-L2 ~2e-4), where HF's own BF16 run
// has 9-20% of rows below 0.99 and rel-L2 4-10x larger. The reduced bound therefore
// also requires the TS error to be below HF-BF16's on every image. Metal is held to
// the reduced bound as well. The projector stage is held to 5e-3 rel-L2 everywhere
// (ggml-cuda's small-batch F32 kernels use TF32 MMA even with the override, which is
// the ~1.6e-3 a 4-token image reads).
//
//   /root/convvenv/bin/python eng/minimax-m3-vision-reference.py ORACLE_DIR
//   export TS_TEST_MINIMAX_M3_MMPROJ=.../mmproj-MiniMax-M3.1-Flash-BF16.gguf
//   export TS_TEST_MINIMAX_M3_VISION_ORACLE=ORACLE_DIR
//   export TS_TEST_GGML_BACKEND=cuda
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    public class MiniMaxM3VisionOracleTests
    {
        private const string MmprojEnv = "TS_TEST_MINIMAX_M3_MMPROJ";
        private const string OracleEnv = "TS_TEST_MINIMAX_M3_VISION_ORACLE";
        private readonly ITestOutputHelper _output;

        public MiniMaxM3VisionOracleTests(ITestOutputHelper output) => _output = output;

        private static string OracleDir => Environment.GetEnvironmentVariable(OracleEnv);

        private static float[] ReadF32(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            var data = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            return data;
        }

        /// <summary>
        /// Per-row cosine statistics plus two norm-aware ones: <c>WorstNorm</c>, the
        /// lowest-cosine row's norm over the image's RMS row norm (a row much shorter than
        /// its neighbours loses cosine to the same absolute error), and <c>MaxRowErr</c>,
        /// the largest row error |a - e| over that RMS row norm.
        /// </summary>
        private static (double MinCos, double MeanCos, double RelL2, int Below99, double WorstNorm, double MaxRowErr)
            Compare(float[] actual, float[] expected, int dim)
        {
            Assert.Equal(expected.Length, actual.Length);
            double diff = 0, norm = 0, minCos = 1, sumCos = 0, worstNormSq = 0, maxErrSq = 0;
            int rows = expected.Length / dim, below99 = 0;
            for (int r = 0; r < rows; r++)
            {
                double dot = 0, na = 0, nb = 0, err = 0;
                for (int j = 0; j < dim; j++)
                {
                    double a = actual[(long)r * dim + j], e = expected[(long)r * dim + j];
                    err += (a - e) * (a - e);
                    dot += a * e; na += a * a; nb += e * e;
                }
                diff += err;
                norm += nb;
                double cos = dot / (Math.Sqrt(na * nb) + 1e-30);
                if (cos < minCos)
                {
                    minCos = cos;
                    worstNormSq = nb;
                }
                maxErrSq = Math.Max(maxErrSq, err);
                sumCos += cos;
                if (cos < 0.99) below99++;
            }
            double meanRowNormSq = norm / rows + 1e-30;
            return (minCos, sumCos / rows, Math.Sqrt(diff / norm), below99,
                Math.Sqrt(worstNormSq / meanRowNormSq), Math.Sqrt(maxErrSq / meanRowNormSq));
        }

        /// <summary>True F32 GEMMs: ggml_cpu, or ggml_cuda with TF32 switched off.</summary>
        private static bool ExactF32(GgmlBackendType backend) =>
            backend == GgmlBackendType.Cpu
            || (backend == GgmlBackendType.Cuda && Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE") == "0");

        /// <summary>The real tower loads, takes the shared whole-encoder graph, and turns a
        /// full-budget 672x672 image into 576 finite 6144-wide rows; reports warm latency.</summary>
        [ModelFact(MmprojEnv)]
        public void RealMmproj_Encodes672Square()
        {
            GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
            var allocator = new GgmlAllocator(new GgmlContext(new[] { 0 }, backend), 0);
            var load = System.Diagnostics.Stopwatch.StartNew();
            using var encoder = new MiniMaxM3VisionEncoder(Environment.GetEnvironmentVariable(MmprojEnv), allocator);
            _output.WriteLine($"load {load.ElapsedMilliseconds} ms");
            Assert.Equal(6144, encoder.ProjectionDim);
            Assert.Equal(32, encoder.BlockCount);

            const int W = 672, H = 672;
            var rgba = new byte[4 * W * H];
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int i = (y * W + x) * 4;
                    rgba[i] = (byte)(x * 255 / W); rgba[i + 1] = (byte)(y * 255 / H);
                    rgba[i + 2] = (byte)(((x / 48) + (y / 48)) % 2 * 200); rgba[i + 3] = 255;
                }
            var processor = new MiniMaxM3ImageProcessor(encoder.PatchSize, encoder.SpatialMergeSize);
            var prep = System.Diagnostics.Stopwatch.StartNew();
            MiniMaxM3ImagePatches image = processor.ProcessRgba(rgba, W, H);
            double prepMs = prep.Elapsed.TotalMilliseconds;
            Assert.Equal(576, image.TokenCount);

            var times = new List<double>();
            for (int i = 0; i < 6; i++)
            {
                using Tensor t = encoder.Encode(image);
                Assert.True(encoder.LastEncodeUsedWholeEncoderGraph, "whole-encoder graph declined");
                Assert.Equal(new long[] { 576, 6144 }, t.Sizes);
                if (i == 0)
                    Assert.All(t.GetElementsAsFloat(576 * 6144), v => Assert.True(float.IsFinite(v)));
                times.Add(encoder.LastEncodeMilliseconds);
            }
            double first = times[0];
            times.RemoveAt(0);
            times.Sort();
            _output.WriteLine($"672x672 -> 576 tokens on {backend}: preprocess {prepMs:F1} ms; encode first {first:F1} ms, " +
                $"warm min {times[0]:F1} / median {times[times.Count / 2]:F1} ms");
        }

        [ModelFact(OracleEnv)]
        public void Encoder_MatchesHuggingFace_OnRealMmproj()
        {
            string mmproj = Environment.GetEnvironmentVariable(MmprojEnv);
            Assert.False(string.IsNullOrEmpty(mmproj) || !File.Exists(mmproj),
                $"{OracleEnv} is set; {MmprojEnv} must name the minimax_m3 mmproj the oracle is compared against.");
            Assert.True(File.Exists(Path.Combine(OracleDir, "manifest.json")),
                $"{OracleEnv} has no manifest.json (run eng/minimax-m3-vision-reference.py).");

            using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(OracleDir, "manifest.json")));
            int embDim = manifest.RootElement.GetProperty("embedding_dim").GetInt32();
            GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
            var allocator = new GgmlAllocator(new GgmlContext(new[] { 0 }, backend), 0);
            using var encoder = new MiniMaxM3VisionEncoder(mmproj, allocator);
            var processor = new MiniMaxM3ImageProcessor(encoder.PatchSize, encoder.SpatialMergeSize);

            _output.WriteLine($"backend {backend}; transformers {manifest.RootElement.GetProperty("transformers").GetString()}, " +
                $"torchvision {manifest.RootElement.GetProperty("torchvision").GetString()}");
            bool exact = ExactF32(backend);
            double minMean = exact ? 0.99999 : 0.999, maxRel = exact ? 5e-3 : 3e-2, minRow = exact ? 0.995 : 0.5;
            double maxBelow99 = exact ? 0 : 0.02;
            const double maxProjectorRel = 5e-3;
            _output.WriteLine($"GEMM precision: {(exact ? "F32" : "reduced (TF32 / Metal)")}; bounds mean cos >= {minMean}, " +
                $"rel-L2 <= {maxRel}, min row cos >= {minRow}, rows below 0.99 <= {maxBelow99:P0}" +
                (exact ? "" : ", and below HF-BF16's rel-L2 and rows below 0.99") + $"; projector stage rel-L2 <= {maxProjectorRel}, min cos >= 0.9999");
            _output.WriteLine("image             | resized  | tok | px max|d| px != | TS e2e vs HF-f32: min / mean cos, rel-L2, rows<.99, worst-row norm, max row err | HF px -> TS vs HF-f32 | HF-bf16 vs HF-f32: min / mean, rel-L2, rows<.99, max row err | TS ms");
            var stages = new List<string>();
            // One uint8 level after normalization (the smallest std is 0.2613).
            const double oneLevel = 1.0 / (255 * 0.26130258) + 1e-6;
            var failures = new List<string>();
            foreach (JsonElement img in manifest.RootElement.GetProperty("images").EnumerateArray())
            {
                string name = img.GetProperty("name").GetString();
                int gridH = img.GetProperty("grid_h").GetInt32(), gridW = img.GetProperty("grid_w").GetInt32();
                int tokens = img.GetProperty("tokens").GetInt32();

                MiniMaxM3ImagePatches ours = processor.ProcessImage(Path.Combine(OracleDir, img.GetProperty("png").GetString()));
                if (ours.GridH != gridH || ours.GridW != gridW || ours.TokenCount != tokens)
                {
                    failures.Add($"{name}: grid {ours.GridH}x{ours.GridW} ({ours.TokenCount} tokens) vs HF {gridH}x{gridW} ({tokens})");
                    continue;
                }

                float[] hfPixels = ReadF32(Path.Combine(OracleDir, img.GetProperty("pixels").GetString()));
                double maxPx = 0;
                int pxDiff = 0;
                for (int i = 0; i < hfPixels.Length; i++)
                {
                    double d = Math.Abs(hfPixels[i] - ours.Patches[i]);
                    maxPx = Math.Max(maxPx, d);
                    if (d > 0) pxDiff++;
                }
                if (maxPx > oneLevel)
                    failures.Add($"{name}: preprocessed pixels differ by {maxPx:F4} (> one uint8 level)");

                float[] hfF32 = ReadF32(Path.Combine(OracleDir, img.GetProperty("emb_f32").GetString()));
                float[] hfBf16 = ReadF32(Path.Combine(OracleDir, img.GetProperty("emb_bf16").GetString()));

                float[] e2e;
                using (Tensor t = encoder.Encode(ours))
                    e2e = t.GetElementsAsFloat(tokens * embDim);
                double ms = encoder.LastEncodeMilliseconds;
                Assert.True(encoder.LastEncodeUsedWholeEncoderGraph, "whole-encoder graph declined");
                if (!encoder.LastEncodeUsedNativeHead)
                    _output.WriteLine($"{name}: projector ran per-op (native head unavailable)");
                float[] fromHf;
                using (Tensor t = encoder.EncodePatches(hfPixels, gridH, gridW))
                    fromHf = t.GetElementsAsFloat(tokens * embDim);

                // Stage isolation: the tower alone on HF pixels vs HF's tower rows, and the
                // projector alone on HF's tower rows vs HF's embeddings.
                int patches = gridH * gridW, towerDim = encoder.HiddenSize;
                float[] hfTower = ReadF32(Path.Combine(OracleDir, img.GetProperty("tower").GetString()));
                float[] tower, projected;
                using (Tensor t = encoder.EncodeTower(hfPixels, gridH, gridW))
                    tower = t.GetElementsAsFloat(patches * towerDim);
                using (Tensor t = encoder.ProjectTowerRows(hfTower, patches))
                    projected = t.GetElementsAsFloat(tokens * embDim);

                var a = Compare(e2e, hfF32, embDim);
                var b = Compare(fromHf, hfF32, embDim);
                var c = Compare(hfBf16, hfF32, embDim);
                var tw = Compare(tower, hfTower, towerDim);
                var pj = Compare(projected, hfF32, embDim);
                _output.WriteLine($"{name,-17} | {gridW * 14,4}x{gridH * 14,-4}| {tokens,3} | {maxPx,6:F4} | {pxDiff,5} | " +
                    $"{a.MinCos:F6} / {a.MeanCos:F6}, {a.RelL2:E2}, {a.Below99,3}, {a.WorstNorm,5:F3}, {a.MaxRowErr:E2} | " +
                    $"{b.MinCos:F6} / {b.MeanCos:F6}, {b.RelL2:E2} | " +
                    $"{c.MinCos:F6} / {c.MeanCos:F6}, {c.RelL2:E2}, {c.Below99,3}, {c.MaxRowErr:E2} | {ms,6:F1}");
                stages.Add($"{name,-17} | {tw.MinCos:F6} / {tw.MeanCos:F6}, {tw.RelL2:E2}, {tw.Below99,3}, {tw.WorstNorm,5:F3}, {tw.MaxRowErr:E2} | " +
                    $"{pj.MinCos:F6} / {pj.MeanCos:F6}, {pj.RelL2:E2}, {pj.Below99,3}, {pj.WorstNorm,5:F3}, {pj.MaxRowErr:E2}");

                foreach (var (label, r, rows) in new[] { ("end to end", a, tokens), ("HF pixels -> TS encoder", b, tokens),
                             ("tower stage", tw, patches) })
                {
                    if (r.MeanCos < minMean || r.RelL2 > maxRel || r.MinCos < minRow || r.Below99 > maxBelow99 * rows)
                        failures.Add($"{name}: {label} min cos {r.MinCos:F6}, mean {r.MeanCos:F6}, rel-L2 {r.RelL2:E2}, " +
                            $"{r.Below99} rows below 0.99");
                }
                if (!exact && (a.RelL2 >= c.RelL2 || a.Below99 > c.Below99))
                    failures.Add($"{name}: end to end rel-L2 {a.RelL2:E2} / {a.Below99} rows below 0.99 is not below " +
                        $"HF-BF16's {c.RelL2:E2} / {c.Below99}");
                if (pj.RelL2 > maxProjectorRel || pj.MinCos < 0.9999)
                    failures.Add($"{name}: projector stage min cos {pj.MinCos:F6}, rel-L2 {pj.RelL2:E2}");
            }

            _output.WriteLine("stage             | TS tower on HF px vs HF tower: min / mean cos, rel-L2, rows<.99, worst-row norm, max row err | TS projector on HF tower vs HF-f32: same");
            foreach (string line in stages)
                _output.WriteLine(line);
            Assert.True(failures.Count == 0, string.Join("\n", failures));
        }
    }
}
