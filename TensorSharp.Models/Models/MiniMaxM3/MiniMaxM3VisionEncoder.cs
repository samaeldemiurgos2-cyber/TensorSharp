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
// MiniMax-M3.1 vision tower + projector (mmproj clip.projector_type = minimax_m3),
// mirroring transformers' MiniMaxM3VLVisionModel / MiniMaxM3VLMultiModalProjector
// and llama.cpp's clip_graph_minimax_m3:
//
//   x = patches @ (W_t0 + W_t1)                 Conv3d over two identical frames, no bias
//   x = pre_ln(x)                               LayerNorm, eps 1e-5
//   32 x { x += out(SDPA(rope(q), rope(k), v)) on ln1(x);  x += fc2(gelu(fc1(ln2(x)))) }
//   z = mm.2(gelu(mm.1(x)))                     per patch, 1280 -> 6144 -> 6144
//   z = concat of the 4 patches of each 2x2 block, (mh, mw) order -> 24576
//   e = mm.merger.fc2(gelu(mm.merger.fc1(z)))   -> [gh*gw/4, 6144]
//
// All GELUs are erf; there is no post-LN, class token or learned position table.
//
// 3-D RoPE. HF rotates 78 of the 80 head dims as rotate_half pairs (j, j+39) whose
// angle is axis (j / 13) in T,H,W times 1e4^(-2(j % 13)/26). llama.cpp's converter
// permutes the q/k rows to [Ta Tb | Ha Hb | Wa Wb | pass2] so each axis is one
// 26-dim NeoX block (three ggml_rope_ext calls at offsets 0/26/52). Attention only
// sees q.k, so any permutation applied to q and k alike is exact; this encoder
// re-permutes the mmproj rows once at load to [Ta Ha Wa p0 | Tb Hb Wb p1], which
// makes the whole rotation ONE half-split rotation over a [patch, 40] cos/sin table
// (t and pass entries are cos 1, sin 0 for a still image). That is the layout the
// shared Qwen-VL whole-encoder graph (TSGgml_Qwen21VisionEncoderF32: LayerNorm with
// bias, fused qkv + bias, erf GELU MLP) consumes, so the 32 blocks run as one
// device-resident GGML graph with no MiniMax-specific native code.
//
// Load-time folds, all exact: the two Conv3d temporal slices are summed into one
// [588 -> 1280] kernel and q/k/v are fused into one [1280 -> 3840] projection (rows
// permuted as above). Weights are dequantized BF16 -> F32 and stay [out, in]; on GGML
// the patch embedding and the projector run as device-cached linears.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public sealed class MiniMaxM3VisionEncoder : IDisposable
    {
        public const string ProjectorType = "minimax_m3";

        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposed = new();
        private readonly Dictionary<long, (float[] Cos, float[] Sin)> _ropeCache = new();
        private readonly IAllocator _allocator;
        private readonly bool _ggml;

        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _spatialMergeSize;
        private readonly float _ropeTheta;
        private readonly int _axisDim;
        private readonly string[] _blockPrefixes;

        private bool _wholeEncoderUnavailable;
        private bool _wholeEncoderWarned;
        private bool _nativeHeadUnavailable;
        private ModelBase _hostModel;

        public int PatchSize => _patchSize;
        public int SpatialMergeSize => _spatialMergeSize;
        public int ProjectionDim => _projectionDim;
        public int HiddenSize => _hiddenSize;
        public int BlockCount => _blockCount;
        public int PatchDim => 3 * _patchSize * _patchSize;

        /// <summary>Wall time of the last <see cref="EncodePatches"/> call.</summary>
        public double LastEncodeMilliseconds { get; private set; }

        /// <summary>True when the last encode ran the 32 blocks as one GGML graph.</summary>
        public bool LastEncodeUsedWholeEncoderGraph { get; private set; }

        /// <summary>True when the last encode ran the projector + patch merge as one GGML graph.</summary>
        public bool LastEncodeUsedNativeHead { get; private set; }

        /// <summary>Optional: yield the host model's GPU compute lock after the tower.</summary>
        public void SetHostModel(ModelBase model) => _hostModel = model;

        public MiniMaxM3VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
            _ggml = allocator is GgmlAllocator;
            using var gguf = new GgufFile(mmProjPath);

            string projector = gguf.GetString("clip.vision.projector_type") ?? gguf.GetString("clip.projector_type");
            if (projector != ProjectorType)
                throw new InvalidDataException(
                    $"'{Path.GetFileName(mmProjPath)}' is a '{projector ?? "(none)"}' projector; MiniMax-M3.1 needs " +
                    $"clip.projector_type = {ProjectorType} (the MiniMax-M3.1 mmproj).");

            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 14);
            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 1280);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 5120);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 32);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 6144);
            _spatialMergeSize = (int)gguf.GetUint32("clip.vision.spatial_merge_size", 2);
            // Not stored by the converter; vision_config.rope_theta (llama.cpp hard-codes it too).
            _ropeTheta = gguf.GetFloat32("clip.vision.rope.freq_base", 10000f);

            if (_numHeads <= 0 || _hiddenSize % _numHeads != 0)
                throw new InvalidDataException($"MiniMax-M3 vision: hidden {_hiddenSize} is not divisible by {_numHeads} heads.");
            _headDim = _hiddenSize / _numHeads;
            _axisDim = AxisDim(_headDim);
            if (_headDim % 2 != 0 || _axisDim % 2 != 0 || (_headDim - 3 * _axisDim) % 2 != 0)
                throw new InvalidDataException($"MiniMax-M3 vision: head dim {_headDim} has no 3-axis RoPE layout.");

            _blockPrefixes = new string[_blockCount];
            for (int i = 0; i < _blockCount; i++)
                _blockPrefixes[i] = $"v.blk.{i}";

            Console.WriteLine($"MiniMax-M3 vision encoder: patch {_patchSize}, hidden {_hiddenSize}, ffn {_intermediateSize}, " +
                $"heads {_numHeads}x{_headDim}, blocks {_blockCount}, projection {_projectionDim}, merge {_spatialMergeSize}, " +
                $"rope theta {_ropeTheta} over 3x{_axisDim} dims, eps {_eps}");

            try
            {
                LoadWeights(gguf);
                BuildPatchKernel();
                for (int b = 0; b < _blockCount; b++)
                    FuseQkv(b);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>Per-axis rotary width: 2 * ((head_dim / 3) / 2), 26 for 80-wide heads.</summary>
        internal static int AxisDim(int headDim) => 2 * (headDim / 3 / 2);

        // ---- weights ---------------------------------------------------------------------

        private void LoadWeights(GgufFile gguf)
        {
            var sw = Stopwatch.StartNew();
            foreach (var info in gguf.Tensors.Values)
            {
                long n = info.NumElements;
                float[] f32 = new float[n];
                if (info.Type == GgmlTensorType.F32)
                    gguf.ReadTensorDataToFloat32(info, f32, n);
                else
                    NativeDequant.DequantizeToFloat32((int)info.Type, gguf.ReadTensorData(info), 0, f32, 0, n);

                var shape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    shape[i] = (long)info.Shape[info.Shape.Length - 1 - i];
                var tensor = new Tensor(_allocator, DType.Float32, shape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
            }

            var missing = new List<string>();
            void Need(string name) { if (!_weights.ContainsKey(name)) missing.Add(name); }
            foreach (string g in new[] { "v.patch_embd.weight", "v.patch_embd.weight.1", "v.pre_ln.weight", "v.pre_ln.bias",
                         "mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias",
                         "mm.merger.fc1.weight", "mm.merger.fc1.bias", "mm.merger.fc2.weight", "mm.merger.fc2.bias" })
                Need(g);
            foreach (string p in _blockPrefixes)
                foreach (string t in new[] { "ln1", "ln2", "attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down" })
                {
                    Need($"{p}.{t}.weight");
                    Need($"{p}.{t}.bias");
                }
            if (missing.Count > 0)
                throw new InvalidDataException(
                    $"MiniMax-M3 mmproj is missing {missing.Count} tensor(s), first '{missing[0]}'.");

            long mergedIn = (long)_projectionDim * _spatialMergeSize * _spatialMergeSize;
            if (_weights["mm.1.weight"].Sizes[1] != _hiddenSize || _weights["mm.merger.fc1.weight"].Sizes[1] != mergedIn
                || _weights["mm.merger.fc2.weight"].Sizes[0] != _projectionDim)
                throw new InvalidDataException("MiniMax-M3 mmproj projector shapes do not match its header.");
            Console.WriteLine($"  MiniMax-M3 vision weights: {_weights.Count} tensors in {sw.ElapsedMilliseconds} ms");
        }

        private const string PatchKernel = "v.patch_embd.combined";

        /// <summary>Both Conv3d temporal slices see the same still frame, so the conv is
        /// one [hidden, 3*P*P] linear (no bias) with their summed kernel.</summary>
        private void BuildPatchKernel()
        {
            Tensor w0 = _weights["v.patch_embd.weight"], w1 = _weights["v.patch_embd.weight.1"];
            var sum = new Tensor(_allocator, DType.Float32, _hiddenSize, PatchDim);
            using (var w0Flat = w0.View(_hiddenSize, PatchDim))
            using (var w1Flat = w1.View(_hiddenSize, PatchDim))
                Ops.Add(sum, w0Flat, w1Flat);
            _weights[PatchKernel] = sum;
            Release("v.patch_embd.weight");
            Release("v.patch_embd.weight.1");
        }

        /// <summary>
        /// Head-dim permutation from the mmproj's per-axis NeoX layout
        /// [Ta Tb | Ha Hb | Wa Wb | pass] to the half-split layout
        /// [Ta Ha Wa pass_lo | Tb Hb Wb pass_hi]: entry d is the mmproj dim that
        /// lands at position d. Rotating pair (d, d + headDim/2) of the result by the
        /// angle of <see cref="BuildRopeTables"/> equals the three NeoX rotations.
        /// </summary>
        internal static int[] BuildHeadPermutation(int headDim)
        {
            int axisDim = AxisDim(headDim), axisHalf = axisDim / 2;
            int half = headDim / 2, rotHalf = 3 * axisHalf, passHalf = half - rotHalf;
            var perm = new int[headDim];
            for (int j = 0; j < half; j++)
            {
                if (j < rotHalf)
                {
                    int a = j / axisHalf, i = j % axisHalf;
                    perm[j] = a * axisDim + i;
                    perm[half + j] = a * axisDim + axisHalf + i;
                }
                else
                {
                    perm[j] = 3 * axisDim + (j - rotHalf);
                    perm[half + j] = 3 * axisDim + passHalf + (j - rotHalf);
                }
            }
            return perm;
        }

        private void FuseQkv(int block)
        {
            string p = _blockPrefixes[block];
            int h = _hiddenSize;
            int[] perm = BuildHeadPermutation(_headDim);
            var w = new float[3L * h * h];
            var b = new float[3 * h];
            string[] parts = { "attn_q", "attn_k", "attn_v" };
            for (int part = 0; part < 3; part++)
            {
                float[] src = _weights[$"{p}.{parts[part]}.weight"].GetElementsAsFloat(h * h);
                float[] bias = _weights[$"{p}.{parts[part]}.bias"].GetElementsAsFloat(h);
                bool rotary = part < 2;
                for (int row = 0; row < h; row++)
                {
                    int srcRow = rotary ? row / _headDim * _headDim + perm[row % _headDim] : row;
                    Array.Copy(src, (long)srcRow * h, w, ((long)part * h + row) * h, h);
                    b[part * h + row] = bias[srcRow];
                }
                Release($"{p}.{parts[part]}.weight");
                Release($"{p}.{parts[part]}.bias");
            }
            var qkvW = new Tensor(_allocator, DType.Float32, 3L * h, h);
            qkvW.SetElementsAsFloat(w);
            var qkvB = new Tensor(_allocator, DType.Float32, 3L * h);
            qkvB.SetElementsAsFloat(b);
            _weights[$"{p}.attn_qkv.weight"] = qkvW;
            _weights[$"{p}.attn_qkv.bias"] = qkvB;
        }

        private void Release(string name)
        {
            if (_weights.Remove(name, out var t))
                t.Dispose();
        }

        // ---- RoPE ------------------------------------------------------------------------

        /// <summary>
        /// [patches, headDim/2] cos/sin tables for the half-split layout of
        /// <see cref="BuildHeadPermutation"/>, patches in merge-block order. Entry j of an
        /// axis block holds pos_axis * theta^(-2i/axisDim); t (= 0 for a still image) and
        /// the pass-through dims hold angle 0.
        /// </summary>
        internal static (float[] Cos, float[] Sin) BuildRopeTables(int gridH, int gridW, int mergeSize,
            int headDim, float theta)
        {
            int axisDim = AxisDim(headDim), axisHalf = axisDim / 2, half = headDim / 2;
            var invFreq = new float[axisHalf];
            for (int i = 0; i < axisHalf; i++)
                invFreq[i] = 1f / MathF.Pow(theta, 2f * i / axisDim);

            int n = gridH * gridW;
            var cos = new float[(long)n * half];
            var sin = new float[(long)n * half];
            int idx = 0;
            for (int bh = 0; bh < gridH / mergeSize; bh++)
                for (int bw = 0; bw < gridW / mergeSize; bw++)
                    for (int mh = 0; mh < mergeSize; mh++)
                        for (int mw = 0; mw < mergeSize; mw++, idx++)
                        {
                            int hp = bh * mergeSize + mh, wp = bw * mergeSize + mw;
                            long row = (long)idx * half;
                            for (int j = 0; j < half; j++)
                            {
                                int axis = j / axisHalf;
                                float angle = axis switch
                                {
                                    1 => hp * invFreq[j % axisHalf],
                                    2 => wp * invFreq[j % axisHalf],
                                    _ => 0f,   // t = 0 for an image; pass-through dims
                                };
                                cos[row + j] = MathF.Cos(angle);
                                sin[row + j] = MathF.Sin(angle);
                            }
                        }
            return (cos, sin);
        }

        private (float[] Cos, float[] Sin) GetRope(int gridH, int gridW)
        {
            long key = ((long)gridH << 32) | (uint)gridW;
            if (!_ropeCache.TryGetValue(key, out var tables))
            {
                tables = BuildRopeTables(gridH, gridW, _spatialMergeSize, _headDim, _ropeTheta);
                _ropeCache[key] = tables;
            }
            return tables;
        }

        // ---- encode ----------------------------------------------------------------------

        public Tensor Encode(MiniMaxM3ImagePatches image)
        {
            ArgumentNullException.ThrowIfNull(image);
            if (image.MergeSize != _spatialMergeSize)
                throw new ArgumentException($"Patches were merged by {image.MergeSize}; the tower merges by {_spatialMergeSize}.");
            return EncodePatches(image.Patches, image.GridH, image.GridW);
        }

        /// <summary>
        /// Encode merge-block-ordered patches ([gridH*gridW, 3*P*P], each [c][y][x]) into
        /// [gridH*gridW/merge^2, projectionDim] F32 embeddings, one row per image token.
        /// </summary>
        public Tensor EncodePatches(float[] patches, int gridH, int gridW)
        {
            int n = ValidateGrid(patches, gridH, gridW);
            long start = Stopwatch.GetTimestamp();
            Tensor hidden = Embed(patches, n);
            long embedded = Stopwatch.GetTimestamp();

            try
            {
                RunBlocks(hidden, n, gridH, gridW);
                _hostModel?.YieldGpuComputeLock();
                long encoded = Stopwatch.GetTimestamp();

                Tensor result = Project(hidden, n);
                long end = Stopwatch.GetTimestamp();
                double Ms(long a, long b) => (b - a) * 1000.0 / Stopwatch.Frequency;
                LastEncodeMilliseconds = Ms(start, end);
                Console.WriteLine($"  MiniMax-M3 vision encode: {LastEncodeMilliseconds:F0} ms (embed {Ms(start, embedded):F0}, " +
                    $"{_blockCount} blocks {Ms(embedded, encoded):F0} " +
                    $"[{(LastEncodeUsedWholeEncoderGraph ? "whole-encoder graph" : "per-block")}], projector {Ms(encoded, end):F0}), " +
                    $"{gridH}x{gridW} patches -> {result.Sizes[0]} tokens");
                return result;
            }
            finally
            {
                hidden.Dispose();
            }
        }

        /// <summary>The tower alone (patch embedding, pre-LN, the 32 blocks): merge-block-ordered
        /// patches -> [gridH*gridW, hidden], the rows HF's vision model returns. Lets the oracle
        /// tests compare the tower and the projector separately.</summary>
        internal Tensor EncodeTower(float[] patches, int gridH, int gridW)
        {
            int n = ValidateGrid(patches, gridH, gridW);
            Tensor hidden = Embed(patches, n);
            try
            {
                RunBlocks(hidden, n, gridH, gridW);
                return hidden;
            }
            catch
            {
                hidden.Dispose();
                throw;
            }
        }

        /// <summary>The projector + 2x2 merge alone on tower rows [patches, hidden] given in
        /// merge-block order (see <see cref="EncodeTower"/>).</summary>
        internal Tensor ProjectTowerRows(float[] towerRows, int patches)
        {
            ArgumentNullException.ThrowIfNull(towerRows);
            int cells = _spatialMergeSize * _spatialMergeSize;
            if (patches <= 0 || patches % cells != 0 || towerRows.LongLength != (long)patches * _hiddenSize)
                throw new ArgumentException($"Expected a multiple of {cells} rows of {_hiddenSize} tower values.");
            using var hidden = new Tensor(_allocator, DType.Float32, patches, _hiddenSize);
            hidden.SetElementsAsFloat(towerRows);
            return Project(hidden, patches);
        }

        private int ValidateGrid(float[] patches, int gridH, int gridW)
        {
            ArgumentNullException.ThrowIfNull(patches);
            int m = _spatialMergeSize;
            if (gridH <= 0 || gridW <= 0 || gridH % m != 0 || gridW % m != 0)
                throw new ArgumentException($"Patch grid {gridH}x{gridW} is not a whole number of {m}x{m} merge blocks.");
            int n = checked(gridH * gridW);
            if (patches.LongLength != (long)n * PatchDim)
                throw new ArgumentException($"Expected {n} x {PatchDim} patch values, got {patches.LongLength}.");
            return n;
        }

        /// <summary>Patch embedding (the folded Conv3d) and the pre-LN: [n, hidden].</summary>
        private Tensor Embed(float[] patches, int n)
        {
            using var px = new Tensor(_allocator, DType.Float32, n, PatchDim);
            px.SetElementsAsFloat(patches);
            using var embeds = Linear(px, PatchKernel, null);
            return Ops.LayerNorm(null, embeds, _weights["v.pre_ln.weight"], _weights["v.pre_ln.bias"], _eps);
        }

        /// <summary>The 32 blocks in place on <paramref name="hidden"/>.</summary>
        private void RunBlocks(Tensor hidden, int n, int gridH, int gridW)
        {
            var (cos, sin) = GetRope(gridH, gridW);
            LastEncodeUsedWholeEncoderGraph = TryWholeEncoder(hidden, n, cos, sin);
            if (!LastEncodeUsedWholeEncoderGraph)
                for (int b = 0; b < _blockCount; b++)
                    EncoderBlock(hidden, b, n, cos, sin);
        }

        /// <summary>The 32 blocks as one device-resident GGML graph (the shared Qwen-VL
        /// whole encoder with erf GELU). False without a GGML allocator; a failure is
        /// reported once and the per-block path takes over for the rest of the process.</summary>
        private bool TryWholeEncoder(Tensor hidden, int n, float[] cos, float[] sin)
        {
            if (!_ggml || _wholeEncoderUnavailable)
                return false;

            var ln1W = new Tensor[_blockCount]; var ln1B = new Tensor[_blockCount];
            var qkvW = new Tensor[_blockCount]; var qkvB = new Tensor[_blockCount];
            var outW = new Tensor[_blockCount]; var outB = new Tensor[_blockCount];
            var ln2W = new Tensor[_blockCount]; var ln2B = new Tensor[_blockCount];
            var upW = new Tensor[_blockCount]; var upB = new Tensor[_blockCount];
            var downW = new Tensor[_blockCount]; var downB = new Tensor[_blockCount];
            for (int i = 0; i < _blockCount; i++)
            {
                string p = _blockPrefixes[i];
                ln1W[i] = _weights[$"{p}.ln1.weight"]; ln1B[i] = _weights[$"{p}.ln1.bias"];
                qkvW[i] = _weights[$"{p}.attn_qkv.weight"]; qkvB[i] = _weights[$"{p}.attn_qkv.bias"];
                outW[i] = _weights[$"{p}.attn_out.weight"]; outB[i] = _weights[$"{p}.attn_out.bias"];
                ln2W[i] = _weights[$"{p}.ln2.weight"]; ln2B[i] = _weights[$"{p}.ln2.bias"];
                upW[i] = _weights[$"{p}.ffn_up.weight"]; upB[i] = _weights[$"{p}.ffn_up.bias"];
                downW[i] = _weights[$"{p}.ffn_down.weight"]; downB[i] = _weights[$"{p}.ffn_down.bias"];
            }

            try
            {
                bool ok = GgmlBasicOps.Qwen35VisionEncoder(hidden, _eps, 1f / MathF.Sqrt(_headDim),
                    n, _numHeads, _headDim, _headDim / 2, cos, sin,
                    ln1W, ln1B, qkvW, qkvB, outW, outB, ln2W, ln2B, upW, upB, downW, downB,
                    geluErf: true);
                if (!ok)
                    DeclineWholeEncoder(GgmlBasicOps.LastNativeError("graph build or compute failed"));
                return ok;
            }
            catch (Exception e) when (e is not OperationCanceledException and not ModelUnloadedException)
            {
                DeclineWholeEncoder(e.Message);
                return false;
            }
        }

        private void DeclineWholeEncoder(string reason)
        {
            _wholeEncoderUnavailable = true;
            if (_wholeEncoderWarned)
                return;
            _wholeEncoderWarned = true;
            Console.Error.WriteLine(
                $"[minimax-m3-vision] whole-encoder GGML graph failed ({reason}); running the 32 blocks " +
                "per block instead (several times slower image prefill). Reported once.");
        }

        // ---- per-block reference path ----------------------------------------------------

        private void EncoderBlock(Tensor hidden, int block, int n, float[] cos, float[] sin)
        {
            string p = _blockPrefixes[block];
            using (var ln1 = Ops.LayerNorm(null, hidden, _weights[$"{p}.ln1.weight"], _weights[$"{p}.ln1.bias"], _eps))
            using (var attn = SelfAttention(ln1, p, n, cos, sin))
                Ops.Add(hidden, hidden, attn);

            using var ln2 = Ops.LayerNorm(null, hidden, _weights[$"{p}.ln2.weight"], _weights[$"{p}.ln2.bias"], _eps);
            using var up = Linear(ln2, $"{p}.ffn_up.weight", $"{p}.ffn_up.bias");
            GeluErfInPlace(up);
            using var down = Linear(up, $"{p}.ffn_down.weight", $"{p}.ffn_down.bias");
            Ops.Add(hidden, hidden, down);
        }

        private unsafe Tensor SelfAttention(Tensor input, string p, int n, float[] cos, float[] sin)
        {
            int h = _hiddenSize, heads = _numHeads, hd = _headDim, half = hd / 2;
            using var qkv = Linear(input, $"{p}.attn_qkv.weight", $"{p}.attn_qkv.bias");
            var q = new Tensor(_allocator, DType.Float32, n, h);
            var k = new Tensor(_allocator, DType.Float32, n, h);
            var v = new Tensor(_allocator, DType.Float32, n, h);
            try
            {
                long qkvL = (long)TensorComputePrimitives.GetFloatPointer(qkv);
                long qL = (long)TensorComputePrimitives.GetFloatPointer(q);
                long kL = (long)TensorComputePrimitives.GetFloatPointer(k);
                long vL = (long)TensorComputePrimitives.GetFloatPointer(v);
                long rowBytes = (long)h * sizeof(float);
                fixed (float* cp = cos, sp = sin)
                {
                    long cL = (long)cp, sL = (long)sp;
                    Parallel.For(0, n, r =>
                    {
                        float* src = (float*)qkvL + (long)r * 3 * h;
                        float* qr = (float*)qL + (long)r * h, kr = (float*)kL + (long)r * h;
                        Buffer.MemoryCopy(src, qr, rowBytes, rowBytes);
                        Buffer.MemoryCopy(src + h, kr, rowBytes, rowBytes);
                        Buffer.MemoryCopy(src + 2 * h, (float*)vL + (long)r * h, rowBytes, rowBytes);
                        float* c = (float*)cL + (long)r * half, s = (float*)sL + (long)r * half;
                        for (int head = 0; head < heads; head++)
                        {
                            RotateHalf(qr + head * hd, c, s, half);
                            RotateHalf(kr + head * hd, c, s, half);
                        }
                    });
                }

                using var qR = q.View(n, heads, hd);
                using var kR = k.View(n, heads, hd);
                using var vR = v.View(n, heads, hd);
                using var qT = qR.Transpose(0, 1);
                using var kT0 = kR.Transpose(0, 1);
                using var vT = vR.Transpose(0, 1);
                using var qHeads = Ops.NewContiguous(qT);
                using var kHeads = Ops.NewContiguous(kT0);
                using var vHeads = Ops.NewContiguous(vT);
                using var kT = kHeads.Transpose(1, 2);

                using var scores = new Tensor(_allocator, DType.Float32, heads, n, n);
                Ops.AddmmBatch(scores, 0, scores, 1f / MathF.Sqrt(hd), qHeads, kT);
                Ops.Softmax(scores, scores);
                using var ctx = new Tensor(_allocator, DType.Float32, heads, n, hd);
                Ops.AddmmBatch(ctx, 0, ctx, 1.0f, scores, vHeads);

                using var back = ctx.Transpose(0, 1);
                using var merged = Ops.NewContiguous(back);
                using var flat = merged.View(n, h);
                return Linear(flat, $"{p}.attn_out.weight", $"{p}.attn_out.bias");
            }
            finally
            {
                q.Dispose();
                k.Dispose();
                v.Dispose();
            }
        }

        private static unsafe void RotateHalf(float* x, float* cos, float* sin, int half)
        {
            for (int d = 0; d < half; d++)
            {
                float x0 = x[d], x1 = x[d + half];
                x[d] = x0 * cos[d] - x1 * sin[d];
                x[d + half] = x0 * sin[d] + x1 * cos[d];
            }
        }

        // ---- projector -------------------------------------------------------------------

        private Tensor Project(Tensor hidden, int n)
        {
            int cells = _spatialMergeSize * _spatialMergeSize;
            LastEncodeUsedNativeHead = TryNativeHead(hidden, n, cells, out Tensor native);
            if (LastEncodeUsedNativeHead)
                return native;

            Tensor merged;
            using (var z1 = Linear(hidden, "mm.1.weight", "mm.1.bias"))
            {
                GeluErfInPlace(z1);
                merged = Linear(z1, "mm.2.weight", "mm.2.bias");
            }
            // Four consecutive patch rows are one 2x2 block in (mh, mw) order:
            // a [n, proj] row-major tensor IS the [n/4, 4*proj] concatenation.
            using (merged)
            using (var grouped = merged.View(n / cells, (long)cells * _projectionDim))
            using (var fc1 = Linear(grouped, "mm.merger.fc1.weight", "mm.merger.fc1.bias"))
            {
                GeluErfInPlace(fc1);
                return Linear(fc1, "mm.merger.fc2.weight", "mm.merger.fc2.bias");
            }
        }

        /// <summary>The projector + 2x2 merge + merge MLP as one GGML graph
        /// (TSGgml_MiniMaxM3VisionHeadF32). False without a GGML allocator; a native build
        /// without the entry point, or a failing graph, is reported once and the per-op
        /// head (identical math, ~10 intermediate round trips) takes over.</summary>
        private bool TryNativeHead(Tensor hidden, int n, int cells, out Tensor result)
        {
            result = null;
            if (!_ggml || _nativeHeadUnavailable)
                return false;
            var output = new Tensor(_allocator, DType.Float32, n / cells, _projectionDim);
            string reason;
            try
            {
                if (GgmlMiniMaxM3VisionNative.TryHead(hidden, output, cells,
                        _weights["mm.1.weight"], _weights["mm.1.bias"], _weights["mm.2.weight"], _weights["mm.2.bias"],
                        _weights["mm.merger.fc1.weight"], _weights["mm.merger.fc1.bias"],
                        _weights["mm.merger.fc2.weight"], _weights["mm.merger.fc2.bias"]))
                {
                    result = output;
                    return true;
                }
                reason = GgmlBasicOps.LastNativeError("graph build or compute failed");
            }
            catch (EntryPointNotFoundException)
            {
                reason = "the loaded GgmlOps build has no TSGgml_MiniMaxM3VisionHeadF32";
            }
            output.Dispose();
            _nativeHeadUnavailable = true;
            Console.Error.WriteLine(
                $"[minimax-m3-vision] fused projector graph unavailable ({reason}); running the projector " +
                "and patch merge as per-op GGML calls instead (~100 ms slower per full-size image). Reported once.");
            return false;
        }

        /// <summary>
        /// input @ W^T (+ bias) for a weight stored [out, in]. On GGML the weight goes
        /// through the quantized-linear entry point as an F32 "quantized" weight: that
        /// path keeps a device-resident copy keyed by the weight's host pointer, where
        /// the generic F32 addmm would re-upload (and host-transpose) the up-to-604 MB
        /// projector weights on every call.
        /// </summary>
        private unsafe Tensor Linear(Tensor input, string weightName, string biasName)
        {
            Tensor w = _weights[weightName];
            long outDim = w.Sizes[0], inDim = w.Sizes[1];
            var result = new Tensor(_allocator, DType.Float32, input.Sizes[0], outDim);
            Tensor contiguous = input.IsContiguous() ? null : Ops.NewContiguous(input);
            try
            {
                if (_ggml)
                    GgmlBasicOps.AddmmQuant(result, contiguous ?? input, (IntPtr)TensorComputePrimitives.GetFloatPointer(w),
                        0 /* GGML_TYPE_F32 */, inDim, outDim, outDim * inDim * sizeof(float));
                else
                    Ops.Addmm(result, 0, result, 1.0f, contiguous ?? input, GetTransposed(weightName));
            }
            finally
            {
                contiguous?.Dispose();
            }
            if (biasName != null)
                Ops.Add(result, result, _weights[biasName]);
            return result;
        }

        /// <summary>The [in, out] copy of a weight for the managed backends' addmm, made
        /// on first use (the GGML path multiplies the [out, in] weight directly).</summary>
        private Tensor GetTransposed(string weightName)
        {
            if (!_transposed.TryGetValue(weightName, out var t))
            {
                using var view = _weights[weightName].Transpose();
                t = Ops.NewContiguous(view);
                _transposed[weightName] = t;
            }
            return t;
        }

        private unsafe void GeluErfInPlace(Tensor t)
        {
            if (_ggml)
            {
                GgmlBasicOps.GELUErf(t, t);
                return;
            }

            long total = t.ElementCount();
            long ptrL = (long)TensorComputePrimitives.GetFloatPointer(t);
            int rows = (int)t.Sizes[0];
            long cols = total / rows;
            Parallel.For(0, rows, r =>
            {
                float* row = (float*)ptrL + r * cols;
                for (long i = 0; i < cols; i++)
                    row[i] = (float)GeluErf(row[i]);
            });
        }

        /// <summary>Exact GELU, 0.5 x (1 + erf(x / sqrt 2)), with erf from
        /// W. J. Cody's rational approximations (|error| &lt; 1e-15 relative).</summary>
        internal static double GeluErf(double x) => 0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0)));

        internal static double Erf(double x)
        {
            double ax = Math.Abs(x);
            double r;
            if (ax < 0.5)
            {
                double z = x * x;
                double num = (((0.185777706184603153 * z + 3.16112374387056560) * z + 113.864154151050156) * z
                    + 377.485237685302021) * z + 3209.37758913846947;
                double den = (((z + 23.6012909523441209) * z + 244.024637934444173) * z
                    + 1282.61652607737228) * z + 2844.23683343917062;
                return x * num / den;
            }
            if (ax < 4.0)
            {
                double num = (((((((2.15311535474403846e-8 * ax + 0.564188496988670089) * ax + 8.88314979438837594) * ax
                    + 66.1191906371416295) * ax + 298.635138197400131) * ax + 881.952221241769090) * ax
                    + 1712.04761263407058) * ax + 2051.07837782607147) * ax + 1230.33935479799725;
                double den = (((((((ax + 15.7449261107098347) * ax + 117.693950891312499) * ax
                    + 537.181101862009858) * ax + 1621.38957456669019) * ax + 3290.79923573345963) * ax
                    + 4362.61909014324716) * ax + 3439.36767414372164) * ax + 1230.33935480374942;
                r = Math.Exp(-ax * ax) * num / den;
            }
            else
            {
                double z = 1.0 / (ax * ax);
                double num = ((((0.0163153871373020978 * z + 0.305326634961232344) * z + 0.360344899949804439) * z
                    + 0.125781726111229246) * z + 0.0160837851487422766) * z + 6.58749161529837803e-4;
                double den = ((((z + 2.56852019228982242) * z + 1.87295284992346725) * z
                    + 0.527905102951428412) * z + 0.0605183413124413191) * z + 0.00233520497626869185;
                r = Math.Exp(-ax * ax) / ax * (0.564189583547756287 - z * num / den);
            }
            double erf = 1.0 - r;
            return x < 0 ? -erf : erf;
        }

        public void Dispose()
        {
            foreach (var t in _transposed.Values)
                t.Dispose();
            _transposed.Clear();
            foreach (var t in _weights.Values)
            {
                // The native graphs keep device copies keyed by these host pointers;
                // drop them before the pages can be reused.
                if (_ggml)
                    GgmlBasicOps.InvalidateTensorHostBuffer(t);
                t.Dispose();
            }
            _weights.Clear();
            _ropeCache.Clear();
        }
    }
}
