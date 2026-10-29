// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) executor for the direct-CUDA backend.
//
// The model-side half of the DSV4 direct-CUDA path: opens the split-GGUF
// shards, parses the deepseek4 hyper-parameters, dequantizes the small
// tensors (norms, gates, sinks, APE tables, the router) to F32 host arrays,
// precomputes the raw/compress RoPE cos-sin tables (same YaRN math as
// DeepSeek4CpuExecutor.RopeCacheInit), and hands everything to
// TensorSharp.Cuda.Dsv4CudaEngine, which runs the forward pass with
// driver-API kernels — fully independent of ggml.
//
// The bulk weights are never staged in host RAM: each one is described to the
// engine as (shard, file offset, length) and the engine's loader pool streams
// it through pinned chunks straight into VRAM (layer-split across the visible
// GPUs). A 150 GiB model therefore loads with a ~2 GB host footprint.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    internal sealed unsafe class DeepSeek4CudaExecutor : IDisposable, IDsv41EngramSource, IDsv4SlotExecutor
    {
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;

        // The checkpoint's shards (and a DSpark drafter's GGUF beside them).
        private GgufShardSet _gguf;

        // hparams
        private int _nLayer, _nEmbd, _nHead, _nVocab, _headDim, _nRot, _qLoraRank, _oGroups, _oLoraRank, _nSwa;
        private float _rmsEps;
        private int _nExpert, _nExpertUsed, _nFfExp, _hashLayerCount;
        private float _expertWeightsScale;
        private bool _expertWeightsNorm;
        private float[] _swigluClampExp = Array.Empty<float>();
        private float[] _swigluClampShexp = Array.Empty<float>();
        private int _idxNHead, _idxHeadSize, _idxTopK;
        private int[] _compressRatios = Array.Empty<int>();
        private float _compressRopeBase = 10000f, _ropeFreqBase = 10000f;
        private float _yarnFreqScale = 1f, _yarnExtFactor;
        private float _yarnBetaFast = 32f, _yarnBetaSlow = 1f;
        private int _nCtxOrig;
        private int _hcSinkhornIters = 20;
        private float _hcEps = 1e-6f;
        private float _compCorr0, _compCorr1;

        private Dsv4CudaEngine _engine;

        public int VocabSize => _nVocab;
        public int NPast => _engine.NPast;


        /// <summary>
        /// The engine's host-side quantized matmul for <c>--n-cpu-moe</c>
        /// layers. It lives here because the managed quantized kernels are in
        /// this assembly, which TensorSharp.Backends.Cuda cannot reference (the
        /// dependency runs the other way) — same inversion as
        /// <c>ICudaWeightSource</c>.
        /// </summary>
        internal sealed class HostMatMul : IDsv4HostMatMul
        {
            public static readonly HostMatMul Instance = new HostMatMul();

            /// <summary>Exactly the types TryAddmmQuantizedBatch has a direct kernel for; it
            /// returns false for every other one.</summary>
            public bool SupportsType(int ggmlType, int inDim)
                => ManagedQuantizedOps.TryGetActivationPlan((GgmlTensorType)ggmlType, inDim, out _);

            public bool TryMatMulBatch(int ggmlType, int inDim, int inputRowStride,
                ReadOnlySpan<IDsv4HostMatMul.Job> jobs)
            {
                if (jobs.Length == 0)
                    return true;
                var mapped = new ManagedQuantizedOps.QuantMatMulJob[jobs.Length];
                for (int i = 0; i < jobs.Length; i++)
                    mapped[i] = new ManagedQuantizedOps.QuantMatMulJob(
                        jobs[i].Weights, jobs[i].Input, jobs[i].Output,
                        jobs[i].OutDim, jobs[i].RowCount, jobs[i].OutputRowStride);
                return ManagedQuantizedOps.TryAddmmQuantizedBatch(ggmlType, inDim, inputRowStride, mapped);
            }
        }

        /// <param name="nCpuMoe">Routed-expert CPU offload policy: 0 none (the
        /// default — offload is opt-in, and a model that does not fit is refused
        /// with the number of layers that would make it fit), N the first N
        /// layers, <see cref="int.MaxValue"/> every layer, -1 auto (the fewest
        /// leading layers that make the model fit; opt-in only).</param>
        public DeepSeek4CudaExecutor(string ggufPath, int maxContext, int nUbatch, int nGpu, string dsparkPath = null,
            int nCpuMoe = 0)
        {
            var sw = Stopwatch.StartNew();
            bool stats = ParseEnvInt("TS_DSV4_LOAD_STATS", 0) != 0;
            void Mark(string phase)
            {
                if (stats)
                    Console.Error.WriteLine($"[dsv4-cuda]   +{sw.Elapsed.TotalSeconds,6:F1}s {phase}");
            }

            _gguf = new GgufShardSet(ggufPath, "dsv4-cuda", dsparkPath);
            ParseHparams();
            if (_isV41)
                LoadEngramMetadata();
            Mark("shards opened / hparams parsed");

            // Large weights are read exactly once, on their way to VRAM, so the
            // engine streams them straight from the shards through pinned
            // chunks instead of staging the whole (hundreds of GB) model in
            // host RAM first. TS_DSV4_MMAP=1 opts back into the mmap path.
            bool stream = ParseEnvInt("TS_DSV4_MMAP", 0) == 0;
            if (stream)
            {
                _gguf.BeginStreaming();
                Mark("small tensors prefetched");
            }

            int nCtx = maxContext > 0 ? maxContext : 16384;
            int ubatch = nUbatch > 0 ? nUbatch : 1024;

            var desc = BuildModelDesc(nCtx, ubatch);
            Mark("model desc built");
            _engine = new Dsv4CudaEngine(desc, nGpu, nCpuMoe);
            Mark("engine ready");

            // Everything lives in VRAM now; drop the host-side scraps.
            _gguf.EndLoad();

            Console.Error.WriteLine($"[dsv4-cuda] model ready in {sw.Elapsed.TotalSeconds:F1}s");
            StartEngramWarm();
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        public void Forward(int[] tokens, float[] logitsOut) => _engine.Forward(tokens, logitsOut);

        public void Reset() => _engine.Reset();

        // ---- DSpark speculative decoding (no-ops without a drafter) ----

        public bool HasDspark => _engine.DsparkBlockSize > 0;

        public int DsparkBlockSize => _engine.DsparkBlockSize;

        /// <summary>Row width of the target features the drafter consumes.</summary>
        public int DsparkFeatureSize => _engine.DsparkFeatureSize;

        public int UBatch => _engine.UBatch;

        public void ForwardSpec(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
            => _engine.ForwardSpec(tokens, hAllOut, logitsOut, allLogitsRows);

        public void DsparkCatchUp(float[] hRows, int rows, int firstPos)
            => _engine.DsparkCatchUp(hRows, rows, firstPos);

        public int DsparkDraft(int anchorToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            => _engine.DsparkDraft(anchorToken, hPrev, position, draftOut, confOut);

        public void Rewind(int nPast) => _engine.Rewind(nPast);

        // ---- sequence slots (the engine's; see Dsv4CudaEngine.Slots.cs) ----

        /// <summary>The multiple a truncation target must be, 0 when this model cannot truncate.</summary>
        public int TruncateAlign => _engine.TruncateAlign;

        /// <summary>The slot Forward, Reset and Truncate act on.</summary>
        public int ActiveSlot => _engine.ActiveSlot;

        public int SlotAlloc() => _engine.SlotAlloc();

        public bool SlotStatus(int slot, out int head, out int checkpoint, out bool healthy)
            => _engine.SlotStatus(slot, out head, out checkpoint, out healthy);

        public bool SlotCanReuse(int slot, int cachedHead, int target) => _engine.SlotCanReuse(slot, cachedHead, target);

        public bool SlotCanRetain(int slot, int retainedCount, ulong budgetPerDevice)
            => _engine.SlotCanRetain(slot, retainedCount, budgetPerDevice);

        public bool SlotCanAlloc() => _engine.SlotCanAlloc();

        /// <summary>The engine captures no graphs: a slot is its caches.</summary>
        public bool SlotReleaseGraphs(int slot) => _engine.SlotStatus(slot, out _, out _, out _);

        public bool SetActiveSlot(int slot) => _engine.SetActiveSlot(slot);

        public bool SlotFree(int slot) => _engine.SlotFree(slot);

        public bool ResetChecked() => _engine.ResetChecked();

        public bool Truncate(int nPast) => _engine.Truncate(nPast);

        public bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits)
            => _engine.ForwardBatchedDecode(slots, tokens, positions, logits);

        // -------------------------------------------------------------------
        // Loading (mirrors DeepSeek4CpuExecutor's split-shard resolver)
        // -------------------------------------------------------------------

        private void ParseHparams()
        {
            GgufFile g = _gguf.First;
            // V4 and V4.1 are separate architectures with separate key prefixes.
            string arch = g.GetString("general.architecture", "deepseek4");
            if (arch != "deepseek4" && arch != "deepseek41")
                throw new NotSupportedException(
                    $"The DeepSeek CUDA executor requires the deepseek4 or deepseek41 architecture, got '{arch}'.");
            _isV41 = arch == "deepseek41";
            string a = arch;
            _nLayer = (int)g.GetUint32($"{a}.block_count");
            _nEmbd = (int)g.GetUint32($"{a}.embedding_length");
            _nHead = (int)g.GetUint32($"{a}.attention.head_count");
            _headDim = (int)g.GetUint32($"{a}.attention.key_length");
            _nRot = (int)g.GetUint32($"{a}.rope.dimension_count");
            _qLoraRank = (int)g.GetUint32($"{a}.attention.q_lora_rank");
            _oGroups = (int)g.GetUint32($"{a}.attention.output_group_count");
            _oLoraRank = (int)g.GetUint32($"{a}.attention.output_lora_rank");
            _nSwa = (int)g.GetUint32($"{a}.attention.sliding_window");
            _rmsEps = g.GetFloat32($"{a}.attention.layer_norm_rms_epsilon", 1e-6f);
            _nExpert = (int)g.GetUint32($"{a}.expert_count");
            _nExpertUsed = (int)g.GetUint32($"{a}.expert_used_count");
            _nFfExp = (int)g.GetUint32($"{a}.expert_feed_forward_length");
            _expertWeightsScale = g.GetFloat32($"{a}.expert_weights_scale", 1.0f);
            _expertWeightsNorm = g.GetBool($"{a}.expert_weights_norm");
            _hashLayerCount = (int)g.GetUint32($"{a}.hash_layer_count");
            _idxNHead = (int)g.GetUint32($"{a}.attention.indexer.head_count");
            _idxHeadSize = (int)g.GetUint32($"{a}.attention.indexer.key_length");
            _idxTopK = (int)g.GetUint32($"{a}.attention.indexer.top_k");
            _compressRatios = g.GetInt32Array($"{a}.attention.compress_ratios") ?? Array.Empty<int>();
            _compressRopeBase = g.GetFloat32($"{a}.attention.compress_rope_freq_base", 10000f);
            _hcSinkhornIters = (int)g.GetUint32($"{a}.hyper_connection.sinkhorn_iterations", 20);
            _hcEps = g.GetFloat32($"{a}.hyper_connection.epsilon", 1e-6f);
            _swigluClampExp = g.GetFloatArray($"{a}.swiglu_clamp_exp") ?? Array.Empty<float>();
            _swigluClampShexp = g.GetFloatArray($"{a}.swiglu_clamp_shexp") ?? _swigluClampExp;
            _ropeFreqBase = g.GetFloat32($"{a}.rope.freq_base", 10000f);

            float yarnFactor = g.GetFloat32($"{a}.rope.scaling.factor", 0f);
            if (yarnFactor > 0f)
            {
                _yarnFreqScale = 1.0f / yarnFactor;
                _yarnExtFactor = 1.0f;
            }
            _nCtxOrig = (int)g.GetUint32($"{a}.rope.scaling.original_context_length");
            _yarnBetaFast = g.GetFloat32($"{a}.rope.scaling.yarn_beta_fast", 32f);
            _yarnBetaSlow = g.GetFloat32($"{a}.rope.scaling.yarn_beta_slow", 1f);

            int hc = (int)g.GetUint32($"{a}.hyper_connection.count", 4);
            if (hc != 4)
                throw new NotSupportedException($"DeepSeek4 CUDA executor supports hyper_connection.count == 4, got {hc}.");
            if (_nLayer <= 0 || _compressRatios.Length < _nLayer)
                throw new InvalidOperationException("Missing or invalid deepseek4 GGUF metadata.");

            _compCorr0 = MathF.Max(0f, MathF.Floor(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaFast, _compressRopeBase)));
            _compCorr1 = MathF.Min(_nRot - 1, MathF.Ceiling(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaSlow, _compressRopeBase)));
        }

        /// <summary>deepseek41 rather than deepseek4.</summary>
        private bool _isV41;
        private Dsv41EngramData _engram;
        private int[] _v41KvSource, _v41IndexSource;
        private int[] _engramHashes;
        private int _engramUbatchTokens;

        /// <summary>Reads embedded Engram metadata and derives cache-sharing
        /// topology from the checkpoint's tensor ownership across all shards.</summary>
        private void LoadEngramMetadata()
        {
            _engram = Dsv41EngramData.Load(_gguf.First, _gguf.InfoOf);

            if (_engram.Layers[^1].Id >= _nLayer ||
                _engram.KvSourceLayerIds[^1] >= _nLayer || _engram.IndexSourceLayerIds[^1] >= _nLayer)
                throw new InvalidOperationException("DeepSeek V4.1 GGUF metadata names a layer beyond the layer count");

            _v41KvSource = new int[_nLayer];
            _v41IndexSource = new int[_nLayer];
            int kv = -1, index = -1;
            for (int il = 0; il < _nLayer; il++)
            {
                if (Array.IndexOf(_engram.KvSourceLayerIds, il) >= 0) kv = il;
                if (Array.IndexOf(_engram.IndexSourceLayerIds, il) >= 0) index = il;
                int ratio = _compressRatios[il];
                if (ratio < 0 || ratio > 2)
                    throw new NotSupportedException($"Invalid DeepSeek V4.1 compression ratio {ratio} on layer {il}.");
                if (ratio != 0 && (kv < 0 || index < 0 ||
                    _compressRatios[kv] != ratio || _compressRatios[index] != ratio))
                    throw new InvalidOperationException(
                        "DeepSeek V4.1 cache-sharing topology does not match the compression ratios");
                _v41KvSource[il] = ratio != 0 ? kv : -1;
                _v41IndexSource[il] = ratio != 0 ? index : -1;
            }
        }

        // -------------------------------------------------------------------
        // IDsv41EngramSource: the host side of the Engram lookup
        //
        // An Engram table is hundreds of millions of rows and tens to hundreds
        // of GiB. It stays a host mapping and only the rows a token actually
        // selects are dequantized and handed to the engine, which uploads them.
        // -------------------------------------------------------------------

        private struct EngramTable
        {
            public byte* Base;
            public GgmlTensorType Type;
            public long RowBytes;
            public long Rows;
        }

        private EngramTable[] _engramTables;
        // Each host-mapped table's bytes in its shard, for the warm pass.
        private readonly List<(string Path, long Offset, long Bytes)> _engramRanges = new List<(string, long, long)>();
        private Task _engramWarm;
        private readonly CancellationTokenSource _engramWarmCancel = new CancellationTokenSource();

        /// <summary>
        /// A host-mapped Engram table is read a few scattered rows per token, and a row whose page is not
        /// cached is a storage round trip: about a millisecond on a network filesystem, so a cold table
        /// halves decode. Read the tables once on a background thread after the model is ready, as the
        /// native loader does (dsv4_warm_engram), when the host has room to keep them cached.
        /// </summary>
        private void StartEngramWarm()
        {
            if (_engramRanges.Count == 0 || _engine.EngramResident)
                return;
            _engramWarm = MappedTableWarm.Start(_engramRanges, "dsv4-cuda", "Engram tables", _engramWarmCancel.Token);
        }

        public int HashColumns => (int)_engram.HashColumns;

        public int HeadDim => (int)_engram.HeadDim;

        public void BeginEngramUbatch(Dsv41EngramHistory history, ReadOnlySpan<int> tokens, int startPos)
        {
            _engramHashes = _engram.HashTokens(tokens, startPos, ref history.Tokens, ref history.Length);
            _engramUbatchTokens = tokens.Length;
        }

        public void BeginEngramRows(Dsv41EngramHistory[] histories, ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
        {
            // Hash each row against its own sequence, then lay the rows out as one ubatch:
            // [table][row][column], the order GatherEngramRows reads.
            int n = tokens.Length, columns = HashColumns, tables = _engram.Layers.Length;
            var hashes = new int[tables * n * columns];
            for (int i = 0; i < n; i++)
            {
                Dsv41EngramHistory history = histories[i];
                int[] row = _engram.HashTokens(tokens.Slice(i, 1), positions[i], ref history.Tokens, ref history.Length);
                for (int t = 0; t < tables; t++)
                    Array.Copy(row, t * columns, hashes, ((long)t * n + i) * columns, columns);
            }
            _engramHashes = hashes;
            _engramUbatchTokens = n;
        }

        public void CopyEngramRowIndices(int engramIndex, int count, int* dst)
        {
            if (_engramHashes == null || count > _engramUbatchTokens)
                throw new InvalidOperationException("[dsv4-cuda] Engram rows requested before the ubatch was hashed");
            int columns = HashColumns;
            long first = (long)engramIndex * _engramUbatchTokens * columns;
            fixed (int* hashes = _engramHashes)
                Buffer.MemoryCopy(hashes + first, dst, (long)count * columns * 4, (long)count * columns * 4);
        }

        public void GatherEngramRows(int engramIndex, int count, float* dst)
        {
            if (_engramHashes == null || count > _engramUbatchTokens)
                throw new InvalidOperationException("[dsv4-cuda] Engram rows requested before the ubatch was hashed");

            EngramTable table = _engramTables[engramIndex];
            int columns = HashColumns, headDim = HeadDim;
            int rowValues = columns * headDim;
            fixed (int* hashes = _engramHashes)
            {
                int* mine = hashes + (long)engramIndex * _engramUbatchTokens * columns;
                // Each (token, column) is an independent random row: the one place
                // this executor touches the big table at all.
                Parallel.For(0, count, t =>
                {
                    float* rows = dst + (long)t * rowValues;
                    int* ids = mine + (long)t * columns;
                    for (int c = 0; c < columns; c++)
                    {
                        int row = ids[c];
                        if ((uint)row >= (uint)table.Rows)
                            throw new InvalidOperationException("[dsv4-cuda] Engram lookup is out of bounds");
                        ManagedQuantizedOps.DequantizeRowToFloat32(
                            (int)table.Type, (IntPtr)(table.Base + row * table.RowBytes),
                            rows + c * headDim, headDim);
                    }
                });
            }
        }

        /// <summary>Maps one Engram table read-only. Falls back to a full staged
        /// copy only when the platform cannot map, which for a table this size
        /// would mean RAM the box does not have -- so say so rather than
        /// silently allocating.</summary>
        private EngramTable MapEngramTable(string name)
        {
            GgufTensorInfo info = _gguf.InfoOf(name);
            var (path, offset, bytes, mapped) = _gguf.MapTensor(name);
            _engramRanges.Add((path, offset, bytes));
            return new EngramTable
            {
                Base = (byte*)mapped,
                Type = info.Type,
                RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]),
                Rows = info.Shape.Length > 1 ? (long)info.Shape[1] : 1,
            };
        }

        private static float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(freqBase));
        }

        /// <summary>
        /// Materializes a tensor in host memory. Only used for the small
        /// tensors the host actually has to look at (norms, gates, sinks, APE
        /// tables, the router, tid2eid); the bulk weights never come here —
        /// they stream from the shard directly into VRAM.
        /// </summary>
        private (IntPtr Ptr, GgufTensorInfo Info) GetRaw(string name, bool required = true) => _gguf.Raw(name, required);

        private CudaWeightDesc GetQW(string name, bool required = true) => _gguf.Weight(name, required);

        private float[] GetF32(string name, bool required = true) => _gguf.Floats(name, required);

        private int[] GetI32(string name) => _gguf.Ints(name);

        // -------------------------------------------------------------------
        // Descriptor construction
        // -------------------------------------------------------------------

        private Dsv4CudaEngine.ModelDesc BuildModelDesc(int nCtx, int nUbatch)
        {
            var tokEmbd = GetQW("token_embd.weight");
            _nVocab = tokEmbd.Ne1;
            int tokType = tokEmbd.GgmlType;
            // ts_dsv4_embed_f32 decodes these row layouts directly: Q8_0, the K-quants (Q2_K..Q6_K), F16, BF16, F32.
            if (tokType != 8 && !(tokType >= 10 && tokType <= 14) && tokType != 1 && tokType != 0 && tokType != 30)
                throw new NotSupportedException($"[dsv4-cuda] token_embd type {tokType} unsupported (Q8_0/Q2_K..Q6_K/F16/BF16/F32).");

            var m = new Dsv4CudaEngine.ModelDesc
            {
                NLayer = _nLayer,
                NEmbd = _nEmbd,
                NHead = _nHead,
                HeadDim = _headDim,
                NRot = _nRot,
                QLoraRank = _qLoraRank,
                OGroups = _oGroups,
                OLoraRank = _oLoraRank,
                NSwa = _nSwa,
                NVocab = _nVocab,
                NExpert = _nExpert,
                NExpertUsed = _nExpertUsed,
                NFfExp = _nFfExp,
                HashLayerCount = _hashLayerCount,
                IdxNHead = _idxNHead,
                IdxHeadSize = _idxHeadSize,
                IdxTopK = _idxTopK,
                HcSinkhornIters = _hcSinkhornIters,
                RmsEps = _rmsEps,
                HcEps = _hcEps,
                ExpertWeightsScale = _expertWeightsScale,
                ExpertWeightsNorm = _expertWeightsNorm,
                NCtx = nCtx,
                NUbatch = nUbatch,
                TokEmbd = tokEmbd,
                Output = GetQW("output.weight"),
                OutputNorm = GetF32("output_norm.weight"),
                // V4.1 collapses the streams for the head with the LAST layer's
                // FFN gates, so it ships no output_hc_* tensors at all.
                HcHeadFn = GetF32("output_hc_fn.weight", required: !_isV41),
                HcHeadScale = GetF32("output_hc_scale.weight", required: !_isV41),
                HcHeadBase = GetF32("output_hc_base.weight", required: !_isV41),
                V41 = _isV41,
                CandidateSource = _isV41 ? _engram.CandidateSourceLayerId : -1,
                CandidateTopk = _isV41 ? (int)_engram.CandidateTopkBlocks : 0,
                CandidateBlock = _isV41 ? (int)_engram.CandidateBlockSize : 0,
                Engram = _isV41 ? this : null,
                RopeRawTable = BuildRopeTable(nCtx, comp: false),
                RopeCompTable = BuildRopeTable(nCtx, comp: true),
                Layers = new Dsv4CudaEngine.LayerDesc[_nLayer],
                Dspark = BuildDsparkDesc(),
                HostMatMul = HostMatMul.Instance,
            };

            for (int il = 0; il < _nLayer; il++)
            {
                string p = $"blk.{il}.";
                var L = new Dsv4CudaEngine.LayerDesc
                {
                    Ratio = _compressRatios[il],
                    ClampExp = il < _swigluClampExp.Length ? _swigluClampExp[il] : 0f,
                    ClampShexp = il < _swigluClampShexp.Length ? _swigluClampShexp[il] : 0f,
                    AttnNorm = GetF32(p + "attn_norm.weight"),
                    Sinks = GetF32(p + "attn_sinks.weight"),
                    WqA = GetQW(p + "attn_q_a.weight"),
                    QANorm = GetF32(p + "attn_q_a_norm.weight"),
                    WqB = GetQW(p + "attn_q_b.weight"),
                    Wkv = GetQW(p + "attn_kv.weight"),
                    KvNorm = GetF32(p + "attn_kv_a_norm.weight"),
                    WoA = GetQW(p + "attn_output_a.weight"),
                    WoB = GetQW(p + "attn_output_b.weight"),
                    HcAttnFn = GetF32(p + "hc_attn_fn.weight"),
                    HcAttnScale = GetF32(p + "hc_attn_scale.weight"),
                    HcAttnBase = GetF32(p + "hc_attn_base.weight"),
                    HcFfnFn = GetF32(p + "hc_ffn_fn.weight"),
                    HcFfnScale = GetF32(p + "hc_ffn_scale.weight"),
                    HcFfnBase = GetF32(p + "hc_ffn_base.weight"),
                    GateInp = GetF32(p + "ffn_gate_inp.weight"),
                    FfnNorm = GetF32(p + "ffn_norm.weight"),
                    GateExps = GetQW(p + "ffn_gate_exps.weight"),
                    DownExps = GetQW(p + "ffn_down_exps.weight"),
                    UpExps = GetQW(p + "ffn_up_exps.weight"),
                    GateShexp = GetQW(p + "ffn_gate_shexp.weight"),
                    DownShexp = GetQW(p + "ffn_down_shexp.weight"),
                    UpShexp = GetQW(p + "ffn_up_shexp.weight"),
                };

                if (_isV41)
                {
                    // Only the per-ratio source layers carry compressor and
                    // indexer-query tensors; the rest of their group reads the
                    // caches those layers build.
                    L.KvSource = _v41KvSource[il];
                    L.IndexSource = _v41IndexSource[il];
                    if (L.KvSource == il)
                    {
                        L.CompWkv = GetQW(p + "attn_compressor_kv.weight");
                        L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                        if (L.Ratio > 1)
                            L.CompWgate = GetQW(p + "attn_compressor_gate.weight");
                        L.IndexerK = GetQW(p + "indexer.attn_k.weight");
                        L.IndexerKNorm = GetF32(p + "indexer.k_norm.weight");
                    }
                    if (L.IndexSource == il)
                    {
                        L.IdxProj = GetQW(p + "indexer.proj.weight");
                        L.IdxQB = GetQW(p + "indexer.attn_q_b.weight");
                    }
                    for (int t = 0; t < _engram.Layers.Length; t++)
                    {
                        if (_engram.Layers[t].Id != il)
                            continue;
                        L.EngramIndex = t;
                        L.EngramWkv = GetQW(p + "engram_wkv.weight");
                        L.EngramTable = GetQW(p + "engram_embd.weight");
                        L.EngramQ = GetF32(p + "engram_q.weight");
                        L.EngramK = GetF32(p + "engram_k.weight");
                        _engramTables ??= new EngramTable[_engram.Layers.Length];
                        _engramTables[t] = MapEngramTable(p + "engram_embd.weight");
                        break;
                    }
                }
                else if (L.Ratio != 0)
                {
                    L.CompWkv = GetQW(p + "attn_compressor_kv.weight");
                    L.CompWgate = GetQW(p + "attn_compressor_gate.weight");
                    L.CompApe = GetF32(p + "attn_compressor_ape.weight");
                    L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                    if (L.Ratio == CsaRatio)
                    {
                        L.IdxProj = GetQW(p + "indexer.proj.weight");
                        L.IdxQB = GetQW(p + "indexer.attn_q_b.weight");
                        L.IdxCompWkv = GetQW(p + "indexer_compressor_kv.weight");
                        L.IdxCompWgate = GetQW(p + "indexer_compressor_gate.weight");
                        L.IdxCompApe = GetF32(p + "indexer_compressor_ape.weight");
                        L.IdxCompNorm = GetF32(p + "indexer_compressor_norm.weight");
                    }
                }

                if (il < _hashLayerCount)
                    L.Tid2Eid = GetI32(p + "ffn_gate_tid2eid.weight");
                else
                    L.ExpProbsBias = GetF32(p + "exp_probs_b.bias");

                m.Layers[il] = L;
            }

            return m;
        }

        /// <summary>
        /// Describes the DSpark drafter (a separate GGUF: three DSV4 blocks with
        /// compress_ratio 0, plus the Markov and confidence heads) for the
        /// engine. Returns null when no drafter was supplied.
        /// </summary>
        private Dsv4CudaEngine.DsparkDesc BuildDsparkDesc()
        {
            if (_gguf.Extra == null)
                return null;

            // Published DSpark drafters carry the same weights under three
            // naming schemes (the ds4 builder's `mtp.*`, and two `dspark.*`
            // variants that differ in the metadata prefix), so resolve both the
            // keys and the tensor names by trying each spelling.
            string arch = _gguf.Extra.GetString("general.architecture") ?? string.Empty;
            if (arch != "deepseek4-dspark" && arch != "deepseek_v4_flash_dspark_draft")
            {
                throw new InvalidOperationException(
                    $"[dsv4-cuda] draft model architecture '{arch}' is not a DeepSeek V4 DSpark drafter " +
                    "(expected deepseek4-dspark or deepseek_v4_flash_dspark_draft). DSpark drafters for other " +
                    "architectures such as Gemma 4 use a different drafter design and are not supported.");
            }

            int DsUint(params string[] keys)
            {
                foreach (string k in keys)
                {
                    uint v = _gguf.Extra.GetUint32(k, 0);
                    if (v != 0)
                        return (int)v;
                }
                return 0;
            }

            int nStages = DsUint("dspark.n_layers", "dspark.stage_count", "dspark.layer_count",
                                 "deepseek4.dspark.n_layers", "deepseek4.dspark.layer_count");
            int blockSize = DsUint("dspark.block_size", "deepseek4.dspark.block_size");
            int markovRank = DsUint("dspark.markov_rank", "deepseek4.dspark.markov_rank");
            int noiseToken = DsUint("dspark.noise_token_id", "deepseek4.dspark.noise_token_id");
            int[] targetLayers = _gguf.Extra.GetInt32Array("dspark.target_layer_ids")
                ?? _gguf.Extra.GetInt32Array("dspark.target_layers")
                ?? _gguf.Extra.GetInt32Array("deepseek4.dspark.target_layer_ids")
                ?? _gguf.Extra.GetInt32Array("deepseek4.dspark.target_layers")
                ?? Array.Empty<int>();
            if (nStages <= 0 || blockSize <= 0 || markovRank <= 0 || targetLayers.Length == 0)
                throw new InvalidOperationException("[dsv4-cuda] draft model is missing dspark.* metadata");

            // The drafter has no per-layer swiglu clamp of its own; the module is
            // trained with the target's single swiglu_limit.
            float clamp = _swigluClampExp.Length > 0 ? _swigluClampExp[_swigluClampExp.Length - 1] : 0f;
            float clampSh = _swigluClampShexp.Length > 0 ? _swigluClampShexp[_swigluClampShexp.Length - 1] : clamp;

            // First spelling that exists in the drafter file wins.
            string Pick(params string[] names)
            {
                foreach (string n in names)
                    if (_gguf.Has(n))
                        return n;
                return names[0];
            }

            var stages = new Dsv4CudaEngine.LayerDesc[nStages];
            for (int s = 0; s < nStages; s++)
            {
                string p = _gguf.Has($"mtp.{s}.attn_norm.weight") ? $"mtp.{s}." : $"dspark.{s}.";
                stages[s] = new Dsv4CudaEngine.LayerDesc
                {
                    Ratio = 0,
                    ClampExp = clamp,
                    ClampShexp = clampSh,
                    AttnNorm = GetF32(p + "attn_norm.weight"),
                    Sinks = GetF32(p + "attn_sinks.weight"),
                    WqA = GetQW(p + "attn_q_a.weight"),
                    QANorm = GetF32(p + "attn_q_a_norm.weight"),
                    WqB = GetQW(p + "attn_q_b.weight"),
                    Wkv = GetQW(p + "attn_kv.weight"),
                    KvNorm = GetF32(p + "attn_kv_a_norm.weight"),
                    WoA = GetQW(p + "attn_output_a.weight"),
                    WoB = GetQW(p + "attn_output_b.weight"),
                    HcAttnFn = GetF32(p + "hc_attn_fn.weight"),
                    HcAttnScale = GetF32(p + "hc_attn_scale.weight"),
                    HcAttnBase = GetF32(p + "hc_attn_base.weight"),
                    HcFfnFn = GetF32(p + "hc_ffn_fn.weight"),
                    HcFfnScale = GetF32(p + "hc_ffn_scale.weight"),
                    HcFfnBase = GetF32(p + "hc_ffn_base.weight"),
                    GateInp = GetF32(p + "ffn_gate_inp.weight"),
                    ExpProbsBias = GetF32(p + "exp_probs_b.bias", required: false),
                    FfnNorm = GetF32(p + "ffn_norm.weight"),
                    GateExps = GetQW(p + "ffn_gate_exps.weight"),
                    DownExps = GetQW(p + "ffn_down_exps.weight"),
                    UpExps = GetQW(p + "ffn_up_exps.weight"),
                    GateShexp = GetQW(p + "ffn_gate_shexp.weight"),
                    DownShexp = GetQW(p + "ffn_down_shexp.weight"),
                    UpShexp = GetQW(p + "ffn_up_shexp.weight"),
                };
            }

            string first = "mtp.0.";
            string lastS = $"mtp.{nStages - 1}.";
            return new Dsv4CudaEngine.DsparkDesc
            {
                BlockSize = blockSize,
                NoiseTokenId = noiseToken,
                MarkovRank = markovRank,
                TargetLayerIds = targetLayers,
                Stages = stages,
                MainProj = GetQW(Pick(first + "main_proj.weight", "dspark.main_proj.weight")),
                MainNorm = GetF32(Pick(first + "main_norm.weight", "dspark.main_norm.weight")),
                Norm = GetF32(Pick(lastS + "norm.weight", "dspark.norm.weight")),
                HcHeadFn = GetF32(Pick(lastS + "hc_head_fn.weight", "dspark.hc_head_fn.weight")),
                HcHeadScale = GetF32(Pick(lastS + "hc_head_scale.weight", "dspark.hc_head_scale.weight")),
                HcHeadBase = GetF32(Pick(lastS + "hc_head_base.weight", "dspark.hc_head_base.weight")),
                MarkovW1 = GetF32(Pick(lastS + "markov_head.markov_w1.weight", "dspark.markov_w1.weight")),
                MarkovW2 = GetQW(Pick(lastS + "markov_head.markov_w2.weight", "dspark.markov_w2.weight")),
                ConfProj = GetF32(Pick(lastS + "confidence_head.proj.weight",
                                       "dspark.conf_proj.weight", "dspark.confidence_head.weight")),
            };
        }

        /// <summary>
        /// Interleaved (cos, sin) per position with the YaRN attention factor
        /// folded in — the table-driven form of DeepSeek4CpuExecutor.RopeCacheInit.
        /// </summary>
        private float[] BuildRopeTable(int nCtx, bool comp)
        {
            int nDims = _nRot;
            float freqBase = comp ? _compressRopeBase : _ropeFreqBase;
            float freqScale = comp ? _yarnFreqScale : 1f;
            float extFactor = comp ? _yarnExtFactor : 0f;
            float attnFactor = extFactor == 0f ? 1f : 1f / (1f + 0.1f * MathF.Log(1f / freqScale));
            float thetaScale = MathF.Pow(freqBase, -2f / nDims);

            var table = new float[(long)nCtx * nDims];
            Parallel.For(0, nCtx, pos =>
            {
                float theta = pos;
                long baseIdx = (long)pos * nDims;
                for (int i0 = 0; i0 < nDims; i0 += 2)
                {
                    float thetaInterp = freqScale * theta;
                    float th = thetaInterp;
                    float mscale = attnFactor;
                    if (extFactor != 0f)
                    {
                        float y = (i0 / 2 - _compCorr0) / MathF.Max(0.001f, _compCorr1 - _compCorr0);
                        float ramp = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                        th = thetaInterp * (1f - ramp) + theta * ramp;
                        mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);
                    }
                    table[baseIdx + i0 + 0] = MathF.Cos(th) * mscale;
                    table[baseIdx + i0 + 1] = MathF.Sin(th) * mscale;
                    theta *= thetaScale;
                }
            });
            return table;
        }

        public void Dispose()
        {
            _engramWarmCancel.Cancel();
            try { _engramWarm?.Wait(); }
            catch (AggregateException) { }
            _engramWarmCancel.Dispose();
            _engine?.Dispose();
            _engine = null;
            _gguf?.Dispose();
            _gguf = null;
        }
    }
}
