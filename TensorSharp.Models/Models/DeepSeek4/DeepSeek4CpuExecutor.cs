// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) whole-model executor for the pure C# CPU backend.
//
// This is a from-scratch managed port of the native ggml executor in
// TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp (itself a port of
// llama.cpp's src/models/deepseek4.cpp), restricted to a single sequence:
//
//   * 4-stream hyper-connections (pre/post mixing + Sinkhorn-normalized 4x4
//     combination matrix per token).
//   * LoRA-factored Q, single shared 512-dim K(=V) head, per-head Q RMS norm,
//     attention sinks, RoPE -> attention -> inverse RoPE on the rope slice,
//     grouped LoRA output projection.
//   * Per-layer compress ratio 0 (raw SWA-128 attention only), 4 (CSA:
//     overlap-compressed rows + lightning-indexer top-k selection) or 128
//     (HCA: block-compressed rows, all visible).
//   * MoE: sqrt(softplus) routing in F32, +bias for selection, first
//     `hash_layer_count` layers route by token id through a tid2eid LUT,
//     swiglu clamp, weight normalization, x1.5 routed scale + shared expert.
//
// Weights stay quantized in the memory-mapped GGUF shards; matmuls run
// through ManagedQuantizedOps (integer dot kernels for Q8_0/Q6_K/...,
// dequant+float dot for IQ3_S/MXFP4/BF16). Because the executor is
// imperative, the graph-shape tricks the ggml port needs (masked scratch
// rows, persist plans, zero/-inf ring rows) reduce to plain loops over
// block boundaries.
//
// KV caches are stored as F32 but every cache commit is rounded through F16
// so results track the native executor / llama.cpp (both use F16 caches).
// The Hadamard rotation is skipped for the same reason as the native port:
// it only matters for quantized caches.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cpu;

namespace TensorSharp.Models
{
    internal sealed unsafe partial class DeepSeek4CpuExecutor : IDisposable
    {
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;
        private const int HC = 4;                       // hyper-connection streams
        private const int HcMixDim = (2 + HC) * HC;     // 24

        private struct WeightRef
        {
            public byte* Ptr;
            public GgmlTensorType Type;
            public int Ne0;      // row length (input dim)
            public int Ne1;      // rows (output dim)
            public int Ne2;      // experts (3D tensors), else 1
            public long RowBytes;
            public bool IsValid => Ptr != null;

            public byte* Row(long row) => Ptr + row * RowBytes;
            public byte* Expert(long e) => Ptr + e * Ne1 * RowBytes;
        }

        private sealed class Layer
        {
            public int Ratio;

            public float[] AttnNorm, QANorm, KvNorm, Sinks;
            public WeightRef WqA, WqB, Wkv, WoA, WoB;

            public WeightRef HcAttnFn, HcFfnFn;
            public float[] HcAttnScale, HcAttnBase, HcFfnScale, HcFfnBase;

            public WeightRef CompWkv, CompWgate;
            public float[] CompApe, CompNorm;

            public WeightRef IdxProj, IdxQB, IdxCompWkv, IdxCompWgate;
            public float[] IdxCompApe, IdxCompNorm;

            /// <summary>V4.1 shares one compressed cache and one sparse selection
            /// across each run of layers with the same compression ratio.
            /// KvSource names the layer that builds the caches this layer reads;
            /// IndexSource names the layer that publishes the selection. Both are
            /// -1 on uncompressed (ratio 0) layers.</summary>
            public int KvSource = -1, IndexSource = -1;
            /// <summary>V4.1 projects the compressed latent into the indexer's key
            /// space; V4 runs a second compressor for it instead.</summary>
            public WeightRef IndexerK;
            public float[] IndexerKNorm;

            /// <summary>V4.1 Engram (layers named by the GGUF; null elsewhere).
            /// EngramEmbd stays quantized and is read one 256-value row at a time:
            /// it is ~30 GiB and only 24 rows per token are ever touched.</summary>
            public int EngramIndex = -1;
            public WeightRef EngramEmbd, EngramWkv;
            public float[] EngramQ, EngramK;   // [n_embd * hc] elementwise gains

            public float[] GateInp;          // router, dequantized to F32 [nExpert x nEmbd]
            public float[] ExpProbsBias;     // [nExpert] (null for hash layers)
            public int[] Tid2Eid;            // [nVocab x nExpertUsed] (hash layers only)
            public float[] FfnNorm;
            public WeightRef GateExps, DownExps, UpExps;
            public WeightRef GateShexp, DownShexp, UpShexp;

            // caches (native memory, F32 values rounded through F16 on commit)
            public float* RawK;              // [ringRaw x headDim]
            public float* CompK;             // CSA or HCA compressed K rows [rows x headDim]
            public float* LidK;              // lightning-indexer K rows [rows x idxHeadSize]
            public float* HistKv;            // compressor state ring [stateSize x coff*headDim]
            public float* HistScore;
            public float* LidHistKv;         // [2*CsaRatio x 2*idxHeadSize]
            public float* LidHistScore;
        }

        // hparams
        private int _nLayer, _nEmbd, _nHead, _nVocab, _headDim, _nRot, _qLoraRank, _oGroups, _oLoraRank, _nSwa;
        private float _rmsEps;
        private int _nExpert, _nExpertUsed, _nFfExp, _hashLayerCount;
        private float _expertWeightsScale;
        private bool _expertWeightsNorm;
        /// <summary>deepseek41 rather than deepseek4: adds Engram layers, a shared
        /// expert, and compress ratios of 1 and 2 where V4 uses 4 and 128.</summary>
        private bool _isV41;
        private int _nExpertShared;

        // V4.1 Engram. GGUF metadata carries the token map and bucket layout;
        // the history is per sequence and indexed by ABSOLUTE
        // position, which is what makes a chunked prompt hash the same as a
        // one-shot one.
        private Dsv41EngramData _engram;
        private int[] _engramHistory;
        private int _engramHistoryLength;
        private float* _engramLookup;   // [nt][columns][headDim]
        private float* _engramKv;       // [nt][(HC+1) * nEmbd]
        private float* _engramQn;       // [nt][nEmbd] normalized query scratch
        private float* _engramKn;       // [nt][nEmbd] normalized key scratch
        private int[] _engramHashes;    // [layer][token][column] for the ubatch

        // V4.1 shared compressed caches and candidate pruning.
        private int _candidateSource = -1, _candidateTopk, _candidateBlock;
        private int _v41MaxCompRows;    // widest compressed cache across ratio groups
        private byte* _v41CandMask;     // [nt][rows] 1 = row survived candidate pruning
        private bool _v41CandActive;    // a candidate mask was published this ubatch
        private float* _latent;         // [nt+1][headDim] compressed rows being built
        private float* _latentK;        // [nt+1][idxHeadSize] their indexer keys
        private float* _preAttn;        // [nt][HC] V4.1 delayed hyper-connection gates
        private float* _preFfn;
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
        private float _compCorr0, _compCorr1;    // YaRN correction dims (compress rope)

        // model-level weights
        private WeightRef _tokEmbd, _output, _hcHeadFn;
        private float[] _outputNorm, _hcHeadScale, _hcHeadBase;

        private Layer[] _layers;

        // geometry / state
        private int _nCtx, _nUbatch;
        private int _ringRaw, _compRowsCsa, _compRowsHca;
        private int _nPast;

        private readonly List<GgufFile> _shards = new List<GgufFile>();
        private readonly List<string> _shardPaths = new List<string>();
        private readonly Dictionary<string, (GgufFile File, GgufTensorInfo Info)> _tensorMap
            = new Dictionary<string, (GgufFile, GgufTensorInfo)>(StringComparer.Ordinal);
        private readonly List<IntPtr> _ownedBuffers = new List<IntPtr>();

        private readonly ParallelOptions _po;
        private readonly Dsv4SpinPool _pool;
        private readonly int _perf;
        private bool _useMmap = true;
        private Dictionary<string, IntPtr> _bufferedTensors;

        /// <summary>
        /// Hot-path parallel-for. Routes through the persistent spin pool (region
        /// latency in the microseconds) unless TS_DSV4_SPINPOOL=0 falls back to
        /// TPL Parallel.For.
        /// </summary>
        private void PFor(int n, Action<int> body)
        {
            if (_pool != null)
                _pool.For(n, body);
            else
                Parallel.For(0, n, _po, body);
        }

        // stage timing (ticks), printed when TS_DSV4_PERF >= 2
        private readonly long[] _stageTicks = new long[12];
        private static readonly string[] StageNames =
        {
            "embed", "hc", "attnproj", "comp", "idx", "attncore", "outproj", "router", "experts", "shexp", "lmhead", "rope"
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Tick(int stage, long t0) => _stageTicks[stage] += Stopwatch.GetTimestamp() - t0;

        public int VocabSize => _nVocab;
        public int ContextSize => _nCtx;
        public int NPast => _nPast;

        /// <summary>
        /// Tests only: every tensor TS_DSV4_MMAP=0 (or TS_DSV4_BUFFER_SHARDS) copied into RAM, by name,
        /// with its byte count; empty on the mapped path. What shows that llama.cpp's combined Engram
        /// table is buffered once, as its per-layer row views, and never whole beside them.
        /// </summary>
        internal IReadOnlyDictionary<string, long> BufferedTensorBytes()
        {
            var bytes = new Dictionary<string, long>(StringComparer.Ordinal);
            if (_bufferedTensors != null)
                foreach (string name in _bufferedTensors.Keys)
                    bytes[name] = _tensorMap[name].File.GetTensorByteCount(_tensorMap[name].Info);
            return bytes;
        }

        public DeepSeek4CpuExecutor(string ggufPath, int maxContext, int nUbatch, int nThreads, IAllocator allocator)
        {
            _alloc = allocator ?? throw new ArgumentNullException(nameof(allocator));
            var sw = Stopwatch.StartNew();
            int dop = nThreads > 0 ? nThreads : Environment.ProcessorCount;
            _po = new ParallelOptions { MaxDegreeOfParallelism = dop };
            _perf = ParseEnvInt("TS_DSV4_PERF", 0);

            // The forward pass issues thousands of short Parallel.For regions per
            // token; without a raised floor the thread pool ramps workers up too
            // slowly for any of them to reach full width.
            ThreadPool.GetMinThreads(out int minWorker, out int minIo);
            if (minWorker < dop)
                ThreadPool.SetMinThreads(dop, minIo);

            _useMmap = ParseEnvInt("TS_DSV4_MMAP", 1) != 0;
            // Persistent spinning workers only pay off when the host gives this
            // process dedicated cores; under a shared CPU quota the spinners eat
            // the budget the compute needs. Off by default, opt in via env.
            if (ParseEnvInt("TS_DSV4_SPINPOOL", 0) != 0)
                _pool = new Dsv4SpinPool(dop);

            OpenShards(ggufPath);
            ParseHparams();
            if (_isV41)
                LoadEngramMetadata();

            var shardSelected = new bool[_shards.Count];
            bool anySelected = false;
            if (!_useMmap)
            {
                for (int i = 0; i < shardSelected.Length; i++) shardSelected[i] = true;
                anySelected = true;
            }
            else
            {
                string bufShards = Environment.GetEnvironmentVariable("TS_DSV4_BUFFER_SHARDS");
                if (!string.IsNullOrWhiteSpace(bufShards))
                {
                    foreach (string part in bufShards.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
                    {
                        if (int.TryParse(part, out int idx) && idx >= 1 && idx <= shardSelected.Length)
                        {
                            shardSelected[idx - 1] = true;
                            anySelected = true;
                        }
                    }
                }
            }
            if (anySelected)
                MaterializeTensors(shardSelected);

            _nCtx = maxContext > 0 ? maxContext : 16384;
            _nUbatch = nUbatch > 0 ? nUbatch : 512;
            _ringRaw = Pad(_nSwa + _nUbatch, 256);
            _compRowsCsa = _nCtx / CsaRatio + 1;
            _compRowsHca = _nCtx / HcaRatio + 1;

            LoadWeights();
            AllocateCaches();
            AllocateScratch();

            // Pin the mapped shards in RAM (best effort). Without this, weights
            // served from a FUSE/network filesystem are re-fetched on every
            // forward pass, which caps decode at the network's throughput.
            if (ParseEnvInt("TS_DSV4_MLOCK", 1) != 0)
            {
                var lockSw = Stopwatch.StartNew();
                int locked = 0;
                Parallel.ForEach(_shards, shard => { if (shard.TryLockMappedRegion()) Interlocked.Increment(ref locked); });
                if (locked > 0 || lockSw.Elapsed.TotalSeconds > 1)
                    Console.Error.WriteLine($"[dsv4-cpu] mlock: pinned {locked}/{_shards.Count} shard(s) in {lockSw.Elapsed.TotalSeconds:F1}s");
            }

            double gib = 0;
            foreach (var s in _shards)
                foreach (var kv in s.Tensors)
                    gib += s.GetTensorByteCount(kv.Value);
            Console.Error.WriteLine($"[dsv4-cpu] mapped {gib / (1024.0 * 1024.0 * 1024.0):F1} GiB across {_shards.Count} shard(s) in {sw.Elapsed.TotalSeconds:F1}s " +
                $"(n_ctx={_nCtx}, ubatch={_nUbatch}, threads={_po.MaxDegreeOfParallelism})");
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        private static int Pad(int v, int p) => (v + p - 1) / p * p;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float F16(float v) => (float)(System.Half)v;

        // -------------------------------------------------------------------
        // Loading
        // -------------------------------------------------------------------

        private void OpenShards(string firstPath)
        {
            // Standalone, like the rest of the loop below: this executor
            // enumerates the shards itself from split.count, so letting shard 1
            // also pull in its siblings would build a second, duplicate set of
            // GgufFile objects -- a second mmap and mlock pass over the whole
            // checkpoint, and a merged tensor table this code does not use.
            var first = GgufFile.OpenWithoutSiblingShards(firstPath);
            _shards.Add(first);
            _shardPaths.Add(firstPath);

            int splitCount = (int)first.GetUint32("split.count", 1);
            if (splitCount > 1)
            {
                const string marker = "-00001-of-";
                int pos = firstPath.IndexOf(marker, StringComparison.Ordinal);
                if (pos >= 0)
                {
                    for (int i = 2; i <= splitCount; i++)
                    {
                        string path = firstPath.Substring(0, pos) + $"-{i:D5}-of-" + firstPath.Substring(pos + marker.Length);
                        // This loop IS the shard enumeration, so each shard is opened
                        // for itself. Letting it expand its siblings again would open
                        // every file N times and attribute every tensor to whichever
                        // shard happened to be opened last.
                        _shards.Add(GgufFile.OpenWithoutSiblingShards(path));
                        _shardPaths.Add(path);
                    }
                }
            }

            // A shard cut short by an interrupted download would otherwise fail
            // as a short read well into materialization, after the loader has
            // already committed its buffers.
            foreach (var shard in _shards)
                shard.ThrowIfTruncated();

            foreach (var shard in _shards)
                foreach (var kv in shard.Tensors)
                    _tensorMap[kv.Key] = (shard, kv.Value);
        }

        private void ParseHparams()
        {
            GgufFile g = _shards[0];
            // V4 and V4.1 are separate architectures with separate key prefixes,
            // and V4.1 adds a shared expert and the Engram layers on top of V4's
            // metadata. Everything the two share is read once, below.
            string arch = g.GetString("general.architecture", "deepseek4");
            if (arch != "deepseek4" && arch != "deepseek41")
                throw new NotSupportedException(
                    $"The DeepSeek CPU executor requires the deepseek4 or deepseek41 architecture, got '{arch}'.");
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

            // V4.1 runs one shared expert alongside the routed ones; V4 has none.
            _nExpertShared = (int)g.GetUint32($"{a}.expert_shared_count", 0);

            int hc = (int)g.GetUint32($"{a}.hyper_connection.count", HC);
            if (hc != HC)
                throw new NotSupportedException($"DeepSeek4 CPU executor supports hyper_connection.count == {HC}, got {hc}.");
            if (_nLayer <= 0 || _compressRatios.Length < _nLayer)
                throw new InvalidOperationException("Missing or invalid deepseek4 GGUF metadata.");

            // YaRN correction dims for the compress-rope parameter set.
            _compCorr0 = MathF.Max(0f, MathF.Floor(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaFast, _compressRopeBase)));
            _compCorr1 = MathF.Min(_nRot - 1, MathF.Ceiling(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaSlow, _compressRopeBase)));
        }

        /// <summary>Reads embedded Engram metadata and derives cache-sharing
        /// topology from the checkpoint's tensor ownership across all shards.</summary>
        private void LoadEngramMetadata()
        {
            _engram = Dsv41EngramData.Load(_shards[0],
                name => _tensorMap.TryGetValue(name, out var entry) ? entry.Info : null);
            // llama.cpp's layout: each Engram layer's table is a row range of one combined
            // tensor. Name those ranges as TensorSharp's layout names its tables, as views of
            // the same shard bytes, and drop the combined entry, which TS_DSV4_MMAP=0 would
            // otherwise buffer whole. From here on both layouts read alike.
            if (_engram.Slices != null)
                GgufShardSet.ReplaceWithRowViews(_tensorMap, Dsv41EngramData.CombinedTableName,
                    _engram.TableViews(), "dsv4-cpu");
            _v41KvSource = new int[_nLayer];
            _v41IndexSource = new int[_nLayer];

            if (_engram.Layers[^1].Id >= _nLayer)
                throw new InvalidOperationException("DeepSeek V4.1 Engram layer exceeds the layer count");
            if (_engram.KvSourceLayerIds[^1] >= _nLayer || _engram.IndexSourceLayerIds[^1] >= _nLayer)
                throw new InvalidOperationException("DeepSeek V4.1 shared-cache source exceeds the layer count");

            _candidateSource = _engram.CandidateSourceLayerId;
            _candidateTopk = (int)_engram.CandidateTopkBlocks;
            _candidateBlock = (int)_engram.CandidateBlockSize;
            if (_candidateSource >= 0 && Array.IndexOf(_engram.IndexSourceLayerIds, _candidateSource) < 0)
                throw new InvalidOperationException("DeepSeek V4.1 candidate source must own an indexer");

            // The pruning mask is addressed in compressed-row space, so every
            // indexer that consumes it has to count rows the same way.
            if (_candidateSource >= 0)
            {
                foreach (int id in _engram.IndexSourceLayerIds)
                {
                    if (id > _candidateSource && _compressRatios[id] != _compressRatios[_candidateSource])
                        throw new NotSupportedException(
                            "DeepSeek V4.1 candidate pruning across differing compression ratios is not supported.");
                }
            }

            // Each layer takes the most recent source at or before it; the two
            // lists are ascending, which Dsv41EngramData.Load already enforces.
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

        // Filled by LoadEngramMetadata, consumed by LoadWeights (which builds the
        // Layer objects afterwards).
        private int[] _v41KvSource, _v41IndexSource;

        private static float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(freqBase));
        }

        /// <summary>
        /// Copies the selected shards' tensors into anonymous RAM up front using
        /// parallel positional reads (thread-safe RandomAccess on one handle per
        /// shard, which keeps a network filesystem's readahead streaming).
        /// TS_DSV4_MMAP=0 buffers every shard; TS_DSV4_BUFFER_SHARDS=3,4 buffers
        /// only those (1-based) shards. Makes steady-state inference independent
        /// of the filesystem's page-cache behaviour.
        /// </summary>
        private void MaterializeTensors(bool[] shardSelected)
        {
            var sw = Stopwatch.StartNew();
            _bufferedTensors = new Dictionary<string, IntPtr>(StringComparer.Ordinal);

            var handles = new Microsoft.Win32.SafeHandles.SafeFileHandle[_shardPaths.Count];
            var shardIndexOf = new Dictionary<GgufFile, int>();
            for (int s = 0; s < _shards.Count; s++)
            {
                shardIndexOf[_shards[s]] = s;
                if (shardSelected[s])
                    handles[s] = File.OpenHandle(_shardPaths[s], FileMode.Open, FileAccess.Read, FileShare.Read);
            }

            var names = new List<string>();
            foreach (var kv in _tensorMap)
            {
                if (shardSelected[shardIndexOf[kv.Value.File]])
                    names.Add(kv.Key);
            }

            var slots = new IntPtr[names.Count];
            long totalBytes = 0;
            Parallel.For(0, names.Count, _po, i =>
            {
                var entry = _tensorMap[names[i]];
                int s = shardIndexOf[entry.File];
                long bytes = entry.File.GetTensorByteCount(entry.Info);
                long fileOff = entry.File.DataOffset + (long)entry.Info.Offset;
                IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
                byte* dst = (byte*)buf;
                long done = 0;
                while (done < bytes)
                {
                    int want = (int)Math.Min(32 << 20, bytes - done);
                    int got = RandomAccess.Read(handles[s], new Span<byte>(dst + done, want), fileOff + done);
                    if (got <= 0)
                        throw new IOException($"[dsv4-cpu] short read materializing {names[i]}");
                    done += got;
                }
                slots[i] = buf;
                Interlocked.Add(ref totalBytes, bytes);
            });

            foreach (var h in handles)
                h?.Dispose();

            for (int i = 0; i < names.Count; i++)
            {
                _bufferedTensors[names[i]] = slots[i];
                _ownedBuffers.Add(slots[i]);
            }
            Console.Error.WriteLine($"[dsv4-cpu] buffered {totalBytes / (1024.0 * 1024 * 1024):F1} GiB of weights into RAM in {sw.Elapsed.TotalSeconds:F1}s " +
                $"({totalBytes / (1024.0 * 1024 * 1024) / Math.Max(0.001, sw.Elapsed.TotalSeconds):F2} GiB/s)");
        }

        private WeightRef GetW(string name, bool required = true)
        {
            if (!_tensorMap.TryGetValue(name, out var entry))
            {
                if (required)
                    throw new InvalidOperationException($"[dsv4-cpu] missing tensor: {name}");
                return default;
            }

            GgufTensorInfo info = entry.Info;
            byte* ptr;
            if (_bufferedTensors != null && _bufferedTensors.TryGetValue(name, out IntPtr buffered))
            {
                ptr = (byte*)buffered;
            }
            else if (entry.File.TryGetTensorDataPointer(info, out IntPtr mapped))
            {
                ptr = (byte*)mapped;
            }
            else
            {
                long bytes = entry.File.GetTensorByteCount(info);
                IntPtr buf = Marshal.AllocHGlobal((nint)bytes);
                _ownedBuffers.Add(buf);
                entry.File.ReadTensorDataToNative(info, buf, bytes);
                ptr = (byte*)buf;
            }

            var w = new WeightRef
            {
                Ptr = ptr,
                Type = info.Type,
                Ne0 = (int)info.Shape[0],
                Ne1 = info.Shape.Length > 1 ? (int)info.Shape[1] : 1,
                Ne2 = info.Shape.Length > 2 ? (int)info.Shape[2] : 1,
            };
            w.RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, w.Ne0);
            return w;
        }

        /// <summary>Loads any-typed small tensor fully dequantized into a managed F32 array.</summary>
        private float[] GetF32(string name, bool required = true)
        {
            var w = GetW(name, required);
            if (!w.IsValid)
                return null;
            long n = (long)w.Ne0 * w.Ne1 * w.Ne2;
            var arr = new float[n];
            ManagedQuantizedOps.DequantizeToFloat32((int)w.Type, (IntPtr)w.Ptr, arr, 0, n);
            return arr;
        }

        private int[] GetI32(string name)
        {
            var w = GetW(name);
            if (w.Type != GgmlTensorType.I32)
                throw new InvalidOperationException($"[dsv4-cpu] {name}: expected I32, got {w.Type}");
            long n = (long)w.Ne0 * w.Ne1;
            var arr = new int[n];
            Marshal.Copy((IntPtr)w.Ptr, arr, 0, (int)n);
            return arr;
        }

        private void LoadWeights()
        {
            _tokEmbd = GetW("token_embd.weight");
            _nVocab = _tokEmbd.Ne1;
            _output = GetW("output.weight");
            _outputNorm = GetF32("output_norm.weight");
            // V4.1 collapses the streams for the head with the LAST layer's FFN
            // gates (the delayed pre), so it ships no output_hc_* tensors at all.
            _hcHeadFn = GetW("output_hc_fn.weight", required: !_isV41);
            _hcHeadScale = GetF32("output_hc_scale.weight", required: !_isV41);
            _hcHeadBase = GetF32("output_hc_base.weight", required: !_isV41);
            if (!_isV41 && (!_hcHeadFn.IsValid || _hcHeadScale == null || _hcHeadBase == null))
                throw new InvalidOperationException("[dsv4-cpu] missing output hyper-connection head tensors");

            _layers = new Layer[_nLayer];
            for (int il = 0; il < _nLayer; il++)
            {
                var L = new Layer { Ratio = _compressRatios[il] };
                _layers[il] = L;
                string p = $"blk.{il}.";

                L.AttnNorm = GetF32(p + "attn_norm.weight");
                L.Sinks = GetF32(p + "attn_sinks.weight");
                L.WqA = GetW(p + "attn_q_a.weight");
                L.QANorm = GetF32(p + "attn_q_a_norm.weight");
                L.WqB = GetW(p + "attn_q_b.weight");
                L.Wkv = GetW(p + "attn_kv.weight");
                L.KvNorm = GetF32(p + "attn_kv_a_norm.weight");
                L.WoA = GetW(p + "attn_output_a.weight");
                L.WoB = GetW(p + "attn_output_b.weight");

                if (_isV41 && _engram != null)
                {
                    for (int e = 0; e < _engram.Layers.Length; e++)
                    {
                        if (_engram.Layers[e].Id != il) continue;
                        L.EngramIndex = e;
                        L.EngramEmbd = GetW(p + "engram_embd.weight");
                        // llama.cpp's layout: engram.py only ever uses q_weight * k_weight, so
                        // the file stores that product once, as engram_gate, beside engram_kv
                        // (engram_wkv by another name). The product scales the query and the
                        // key keeps its plain RMS norm (a gain of ones), as in the native graph.
                        bool fused = _tensorMap.ContainsKey(p + "engram_gate.weight");
                        string q = p + (fused ? "engram_gate.weight" : "engram_q.weight");
                        L.EngramWkv = GetW(p + (fused ? "engram_kv.weight" : "engram_wkv.weight"));
                        L.EngramQ = GetF32(q);
                        if (fused)
                        {
                            L.EngramK = new float[HC * _nEmbd];
                            Array.Fill(L.EngramK, 1f);
                        }
                        else
                        {
                            L.EngramK = GetF32(p + "engram_k.weight");
                        }
                        ulong[] qShape = _tensorMap[q].Info.Shape;
                        int lookup = (int)(_engram.HashColumns * _engram.HeadDim);
                        if (L.EngramWkv.Ne0 != lookup || L.EngramWkv.Ne1 != (HC + 1) * _nEmbd ||
                            qShape.Length < 2 || qShape[0] != (ulong)_nEmbd || qShape[1] != HC ||
                            L.EngramK.Length != HC * _nEmbd)
                            throw new InvalidOperationException(
                                $"[dsv4-cpu] DeepSeek V4.1 Engram weights at layer {il} do not match {lookup} lookup " +
                                $"values, {HC} streams and n_embd {_nEmbd}");
                        break;
                    }
                }

                L.HcAttnFn = GetW(p + "hc_attn_fn.weight");
                L.HcAttnScale = GetF32(p + "hc_attn_scale.weight");
                L.HcAttnBase = GetF32(p + "hc_attn_base.weight");
                L.HcFfnFn = GetW(p + "hc_ffn_fn.weight");
                L.HcFfnScale = GetF32(p + "hc_ffn_scale.weight");
                L.HcFfnBase = GetF32(p + "hc_ffn_base.weight");

                if (_isV41)
                {
                    // Only the source layers carry compressor and indexer query
                    // tensors; every other compressed layer reads their caches.
                    L.KvSource = _v41KvSource[il];
                    L.IndexSource = _v41IndexSource[il];
                    if (L.KvSource == il)
                    {
                        L.CompWkv = GetW(p + "attn_compressor_kv.weight");
                        L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                        if (L.Ratio > 1)
                            L.CompWgate = GetW(p + "attn_compressor_gate.weight");
                        L.IndexerK = GetW(p + "indexer.attn_k.weight");
                        L.IndexerKNorm = GetF32(p + "indexer.k_norm.weight");
                    }
                    if (L.IndexSource == il)
                    {
                        L.IdxProj = GetW(p + "indexer.proj.weight");
                        L.IdxQB = GetW(p + "indexer.attn_q_b.weight");
                    }
                }
                else if (L.Ratio != 0)
                {
                    L.CompWkv = GetW(p + "attn_compressor_kv.weight");
                    L.CompWgate = GetW(p + "attn_compressor_gate.weight");
                    L.CompApe = GetF32(p + "attn_compressor_ape.weight");
                    L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                    if (L.Ratio == CsaRatio)
                    {
                        L.IdxProj = GetW(p + "indexer.proj.weight");
                        L.IdxQB = GetW(p + "indexer.attn_q_b.weight");
                        L.IdxCompWkv = GetW(p + "indexer_compressor_kv.weight");
                        L.IdxCompWgate = GetW(p + "indexer_compressor_gate.weight");
                        L.IdxCompApe = GetF32(p + "indexer_compressor_ape.weight");
                        L.IdxCompNorm = GetF32(p + "indexer_compressor_norm.weight");
                    }
                }

                L.GateInp = GetF32(p + "ffn_gate_inp.weight");
                if (il < _hashLayerCount)
                    L.Tid2Eid = GetI32(p + "ffn_gate_tid2eid.weight");
                else
                    L.ExpProbsBias = GetF32(p + "exp_probs_b.bias");
                L.FfnNorm = GetF32(p + "ffn_norm.weight");
                L.GateExps = GetW(p + "ffn_gate_exps.weight");
                L.DownExps = GetW(p + "ffn_down_exps.weight");
                L.UpExps = GetW(p + "ffn_up_exps.weight");
                L.GateShexp = GetW(p + "ffn_gate_shexp.weight");
                L.DownShexp = GetW(p + "ffn_down_shexp.weight");
                L.UpShexp = GetW(p + "ffn_up_shexp.weight");
            }
        }

        /// <summary>
        /// Every buffer this executor owns is an <see cref="IAllocator"/>-backed
        /// <see cref="Tensor"/>, so the shared Ops (RMSNorm, SiLUMulClamp, ...)
        /// can be handed the same memory the imperative kernels write through a
        /// raw pointer. <see cref="AllocF32"/> returns the pointer for the hot
        /// loops; <see cref="Buf"/> looks the tensor back up when an Op needs it.
        /// </summary>
        private readonly IAllocator _alloc;

        private readonly List<Tensor> _tensors = new List<Tensor>();
        private readonly Dictionary<IntPtr, Tensor> _byPtr = new Dictionary<IntPtr, Tensor>();

        private float* AllocF32(long count)
        {
            var tensor = new Tensor(_alloc, DType.Float32, count);
            _tensors.Add(tensor);
            float* p = (float*)CpuNativeHelpers.GetBufferStart(tensor);
            NativeMemory.Clear(p, (nuint)(count * sizeof(float)));
            _byPtr[(IntPtr)p] = tensor;
            return p;
        }

        /// <summary>The tensor that owns <paramref name="p"/> (which must be a
        /// buffer start handed out by <see cref="AllocF32(long)"/>).</summary>
        private Tensor Buf(float* p) => _byPtr[(IntPtr)p];

        /// <summary>Contiguous [rows, cols] view over the front of a buffer.</summary>
        private Tensor View(float* p, long rows, long cols)
        {
            Tensor owner = Buf(p);
            using Tensor flat = owner.View(owner.ElementCount());
            using Tensor slice = flat.Narrow(0, 0, rows * cols);
            return slice.View(rows, cols);
        }

        /// <summary>Compressed rows a V4.1 ratio group needs for the whole context.
        /// Ratio 1 keeps one row per token; ratio 2 one per pair. The extra row
        /// matches the native executor's masked scratch row.</summary>
        private int V41Rows(int ratio) => _nCtx / ratio + 1;

        private void AllocateCaches()
        {
            if (_isV41)
            {
                AllocateCachesV41();
                return;
            }
            foreach (var L in _layers)
            {
                L.RawK = AllocF32((long)_ringRaw * _headDim);
                if (L.Ratio == CsaRatio)
                {
                    L.CompK = AllocF32((long)_compRowsCsa * _headDim);
                    L.LidK = AllocF32((long)_compRowsCsa * _idxHeadSize);
                    L.HistKv = AllocF32(2L * CsaRatio * 2 * _headDim);
                    L.HistScore = AllocF32(2L * CsaRatio * 2 * _headDim);
                    L.LidHistKv = AllocF32(2L * CsaRatio * 2 * _idxHeadSize);
                    L.LidHistScore = AllocF32(2L * CsaRatio * 2 * _idxHeadSize);
                }
                else if (L.Ratio == HcaRatio)
                {
                    L.CompK = AllocF32((long)_compRowsHca * _headDim);
                    L.HistKv = AllocF32((long)HcaRatio * _headDim);
                    L.HistScore = AllocF32((long)HcaRatio * _headDim);
                }
            }
        }

        /// <summary>
        /// V4.1 caches. Every layer owns a raw sliding-window ring; only the
        /// per-ratio source layers own the compressed and indexer caches that
        /// the rest of their group reads.
        /// </summary>
        private void AllocateCachesV41()
        {
            _v41MaxCompRows = 1;
            for (int il = 0; il < _nLayer; il++)
            {
                Layer L = _layers[il];
                L.RawK = AllocF32((long)_ringRaw * _headDim);
                if (L.Ratio == 0 || L.KvSource != il)
                    continue;
                int rows = V41Rows(L.Ratio);
                _v41MaxCompRows = Math.Max(_v41MaxCompRows, rows);
                L.CompK = AllocF32((long)rows * _headDim);
                L.LidK = AllocF32((long)rows * _idxHeadSize);
                if (L.Ratio > 1)
                {
                    // One slot per window position, so a block straddling the
                    // ubatch boundary still sees its earlier half.
                    L.HistKv = AllocF32((long)L.Ratio * _headDim);
                    L.HistScore = AllocF32((long)L.Ratio * _headDim);
                }
            }
        }

        // ubatch scratch buffers
        private float* _xs;          // [nt x HC x nEmbd] hidden streams
        private float* _cur;         // [nt x nEmbd]
        private float* _pre;         // [nt x HC]
        private float* _post;        // [nt x HC]
        private float* _comb;        // [nt x HC x HC]
        private float* _mixes;       // [nt x HcMixDim]
        private float* _qr;          // [nt x qLoraRank]
        private float* _q;           // [nt x nHead x headDim]
        private float* _kvRow;       // [nt x headDim]
        private float* _stKv;        // [nt x 2*headDim]
        private float* _stScore;     // [nt x 2*headDim]
        private float* _lidStKv;     // [nt x 2*idxHeadSize]
        private float* _lidStScore;  // [nt x 2*idxHeadSize]
        private float* _iq;          // [nt x idxNHead x idxHeadSize]
        private float* _iw;          // [nt x idxNHead]
        private float* _idxScores;   // [nt x compRowsCsa]
        private int* _topK;          // [nt x idxTopK]
        private int* _topKCount;     // [nt]
        private float* _attnO;       // [nt x nHead x headDim]
        private float* _oG;          // [nt x oGroups*oLoraRank]
        private float* _attnOut;     // [nt x nEmbd]
        private float* _ffnOut;      // [nt x nEmbd]
        private float* _routerLogits;// [nt x nExpert]
        private float* _routerProbs; // [nt x nExpert]
        private int* _selExperts;    // [nt x nExpertUsed]
        private float* _selWeights;  // [nt x nExpertUsed]
        private float* _ropeRawCache;  // [nt x nRot] interleaved cos/sin (raw params)
        private float* _ropeCompCache; // [nt x nRot] interleaved cos/sin (compress params)
        private byte* _actQuant;     // quantized activations scratch [nt x maxActRowBytes]
        private float* _expertPack;  // gather buffer for MoE prefill [nt x nEmbd]
        private float* _expertGate;  // [nt x nFfExp]
        private float* _expertUp;    // [nt x nFfExp]
        private float* _expertDown;  // [nt x nEmbd]
        private float* _shGate;      // [nt x shexp ff]
        private float* _shUp;
        private float* _shDown;      // [nt x nEmbd]
        private int _maxActRowBytes;
        private int* _expCount;      // [nExpert] fused-prefill MoE grouping
        private int* _expOffset;     // [nExpert]
        private int* _expCursor;     // [nExpert]
        private int* _rowOfSlot;     // [nt x nExpertUsed]
        private int* _slotToken;     // [nt x nExpertUsed]
        private int* _slotGroupSize; // [nt x nExpertUsed]

        // Expert groups at least this large use dequantize-row-once + float dots
        // in the fused prefill MoE; smaller groups use the integer dot path.
        private const int DequantGroupMin = 4;

        private void AllocateScratch()
        {
            int nt = _nUbatch;
            _xs = AllocF32((long)nt * HC * _nEmbd);
            if (_isV41 && _engram != null)
            {
                long columns = _engram.HashColumns;
                _engramLookup = AllocF32((long)nt * columns * _engram.HeadDim);
                _engramKv = AllocF32((long)nt * (HC + 1) * _nEmbd);
                _engramQn = AllocF32((long)nt * _nEmbd);
                _engramKn = AllocF32((long)nt * _nEmbd);
            }
            _cur = AllocF32((long)nt * _nEmbd);
            _pre = AllocF32((long)nt * HC);
            _post = AllocF32((long)nt * HC);
            _comb = AllocF32((long)nt * HC * HC);
            _mixes = AllocF32((long)nt * HcMixDim);
            _qr = AllocF32((long)nt * _qLoraRank);
            _q = AllocF32((long)nt * _nHead * _headDim);
            _kvRow = AllocF32((long)nt * _headDim);
            _stKv = AllocF32((long)nt * 2 * _headDim);
            _stScore = AllocF32((long)nt * 2 * _headDim);
            _lidStKv = AllocF32((long)nt * 2 * _idxHeadSize);
            _lidStScore = AllocF32((long)nt * 2 * _idxHeadSize);
            _iq = AllocF32((long)nt * _idxNHead * _idxHeadSize);
            _iw = AllocF32((long)nt * _idxNHead);
            _idxScores = AllocF32((long)nt * (_isV41 ? _v41MaxCompRows : _compRowsCsa));
            if (_isV41)
            {
                // One extra row so a ratio-1 group can hold every token's block.
                _latent = AllocF32((long)(nt + 1) * _headDim);
                _latentK = AllocF32((long)(nt + 1) * _idxHeadSize);
                _preAttn = AllocF32((long)nt * HC);
                _preFfn = AllocF32((long)nt * HC);
                if (_candidateSource >= 0)
                {
                    var mask = new Tensor(_alloc, DType.UInt8, (long)nt * _v41MaxCompRows);
                    _tensors.Add(mask);
                    _v41CandMask = (byte*)CpuNativeHelpers.GetBufferStart(mask);
                }
            }
            _topK = (int*)AllocF32((long)nt * _idxTopK);
            _topKCount = (int*)AllocF32(nt);
            _attnO = AllocF32((long)nt * _nHead * _headDim);
            _oG = AllocF32((long)nt * _oGroups * _oLoraRank);
            _attnOut = AllocF32((long)nt * _nEmbd);
            _ffnOut = AllocF32((long)nt * _nEmbd);
            _routerLogits = AllocF32((long)nt * _nExpert);
            _routerProbs = AllocF32((long)nt * _nExpert);
            _selExperts = (int*)AllocF32((long)nt * _nExpertUsed);
            _selWeights = AllocF32((long)nt * _nExpertUsed);
            _ropeRawCache = AllocF32((long)nt * _nRot);
            _ropeCompCache = AllocF32((long)nt * _nRot);
            _hcPostTmp = AllocF32((long)nt * HC * _nEmbd);
            // sized nt * nExpertUsed: a token may select the same expert more than
            // once via the hash-router LUT, so one expert's group can exceed nt rows
            _expertPack = AllocF32((long)nt * _nExpertUsed * _nEmbd);
            _expertGate = AllocF32((long)nt * _nExpertUsed * _nFfExp);
            _expertUp = AllocF32((long)nt * _nExpertUsed * _nFfExp);
            _expertDown = AllocF32((long)nt * _nExpertUsed * _nEmbd);
            int shFf = _layers[0].UpShexp.IsValid ? _layers[0].UpShexp.Ne1 : _nFfExp;
            _shGate = AllocF32((long)nt * shFf);
            _shUp = AllocF32((long)nt * shFf);
            _shDown = AllocF32((long)nt * _nEmbd);
            _expCount = (int*)AllocF32(_nExpert);
            _expOffset = (int*)AllocF32(_nExpert);
            _expCursor = (int*)AllocF32(_nExpert);
            _rowOfSlot = (int*)AllocF32((long)nt * _nExpertUsed);
            _slotToken = (int*)AllocF32((long)nt * _nExpertUsed);
            _slotGroupSize = (int*)AllocF32((long)nt * _nExpertUsed);

            // activation quantization scratch: largest quant-weight row length is
            // oGroups*oLoraRank (wo_b input); round up generously. Capacity is
            // nt * max(8, oGroups) rows so the fused grouped-out-projection and
            // fused decode MoE can hold per-group / per-expert activations.
            int maxNe0 = Math.Max(Math.Max(_nEmbd, _oGroups * _oLoraRank), Math.Max(_qLoraRank, _nFfExp));
            _maxActRowBytes = (maxNe0 / 256 + 1) * 300;
            var actTensor = new Tensor(_alloc, DType.UInt8, (long)nt * Math.Max(8, _oGroups), _maxActRowBytes);
            _tensors.Add(actTensor);
            _actQuant = (byte*)CpuNativeHelpers.GetBufferStart(actTensor);
        }

        public void Reset()
        {
            _nPast = 0;
            // Engram lookbacks reach earlier positions, so the history has to go
            // when the sequence does.
            _engramHistoryLength = 0;
            foreach (var L in _layers)
            {
                NativeMemory.Clear(L.RawK, (nuint)((long)_ringRaw * _headDim * sizeof(float)));
                if (L.CompK != null)
                {
                    long rows = _isV41 ? V41Rows(L.Ratio) : L.Ratio == CsaRatio ? _compRowsCsa : _compRowsHca;
                    NativeMemory.Clear(L.CompK, (nuint)(rows * _headDim * sizeof(float)));
                }
                if (L.LidK != null)
                {
                    long rows = _isV41 ? V41Rows(L.Ratio) : _compRowsCsa;
                    NativeMemory.Clear(L.LidK, (nuint)(rows * _idxHeadSize * sizeof(float)));
                }
                if (L.HistKv != null)
                {
                    long n = _isV41 ? (long)L.Ratio * _headDim
                        : L.Ratio == CsaRatio ? 2L * CsaRatio * 2 * _headDim : (long)HcaRatio * _headDim;
                    NativeMemory.Clear(L.HistKv, (nuint)(n * sizeof(float)));
                    NativeMemory.Clear(L.HistScore, (nuint)(n * sizeof(float)));
                }
                if (L.LidHistKv != null)
                {
                    long n = 2L * CsaRatio * 2 * _idxHeadSize;
                    NativeMemory.Clear(L.LidHistKv, (nuint)(n * sizeof(float)));
                    NativeMemory.Clear(L.LidHistScore, (nuint)(n * sizeof(float)));
                }
            }
        }

        // -------------------------------------------------------------------
        // Forward
        // -------------------------------------------------------------------

        public void Forward(int[] tokens, float[] logitsOut)
        {
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("empty token batch", nameof(tokens));
            if (_nPast + tokens.Length > _nCtx)
                throw new InvalidOperationException($"[dsv4-cpu] context overflow: n_past={_nPast} + {tokens.Length} > n_ctx={_nCtx}");

            var sw = Stopwatch.StartNew();
            int done = 0;
            while (done < tokens.Length)
            {
                int nt = Math.Min(_nUbatch, tokens.Length - done);
                bool last = done + nt == tokens.Length;
                ForwardUbatch(tokens, done, nt, _nPast, last ? logitsOut : null);
                _nPast += nt;
                done += nt;
            }
            if (_perf > 0)
            {
                double s = sw.Elapsed.TotalSeconds;
                Console.Error.WriteLine($"[dsv4-cpu] forward {tokens.Length} tokens in {s:F3}s ({tokens.Length / s:F1} tok/s)");
                if (_perf >= 2)
                {
                    var sb = new System.Text.StringBuilder("[dsv4-cpu]   stages:");
                    for (int i = 0; i < _stageTicks.Length; i++)
                    {
                        if (_stageTicks[i] == 0) continue;
                        sb.Append($" {StageNames[i]}={_stageTicks[i] * 1000.0 / Stopwatch.Frequency:F0}ms");
                        _stageTicks[i] = 0;
                    }
                    Console.Error.WriteLine(sb.ToString());
                }
            }
        }

        // TS_DSV4_CPU_DEBUG=1: layer-0 stage prints paired with the
        // TS_DSV4_CUDA_DEBUG=1 prints in Dsv4CudaEngine for A/B debugging.
        private static readonly bool StageDebug = ParseEnvInt("TS_DSV4_CPU_DEBUG", 0) != 0;

        private static void DumpDbg(string label, float* p, int n = 6)
        {
            if (!StageDebug)
                return;
            var vals = new string[n];
            for (int i = 0; i < n; i++)
                vals[i] = p[i].ToString("G6");
            Console.Error.WriteLine($"[dbg-cpu] {label}: {string.Join(" ", vals)}");
        }

        private void ForwardUbatch(int[] tokens, int tokOff, int nt, int p0, float[] logitsOut)
        {
            int E = _nEmbd;
            long t0 = Stopwatch.GetTimestamp();

            // token embedding, replicated over the HC streams
            PFor(nt, t =>
            {
                float* dst = _xs + (long)t * HC * E;
                ManagedQuantizedOps.DequantizeRowToFloat32(
                    (int)_tokEmbd.Type, (IntPtr)_tokEmbd.Row(tokens[tokOff + t]), dst, E);
                for (int s = 1; s < HC; s++)
                    Buffer.MemoryCopy(dst, dst + (long)s * E, E * sizeof(float), E * sizeof(float));
            });
            Tick(0, t0);
            DumpDbg("embed.xs", _xs);
            _traceP0 = p0;
            _traceNt = nt;
            Trace("embedding", _xs, (long)nt * HC * E);

            // Engram row ids for the whole ubatch, once. The history is indexed by
            // absolute position and extended in place, so a prompt split across
            // ubatches hashes exactly as it would in one shot.
            if (_engram != null)
            {
                var window = new ReadOnlySpan<int>(tokens, tokOff, nt);
                _engramHashes = _engram.HashTokens(window, p0, ref _engramHistory, ref _engramHistoryLength);
            }

            // per-token rope caches for this ubatch (raw + compress parameter sets)
            t0 = Stopwatch.GetTimestamp();
            PFor(nt, t =>
            {
                RopeCacheInit(p0 + t, comp: false, _ropeRawCache + (long)t * _nRot);
                RopeCacheInit(p0 + t, comp: true, _ropeCompCache + (long)t * _nRot);
            });
            Tick(11, t0);

            // A candidate mask belongs to one ubatch: the rows it names are
            // scored against this ubatch's queries.
            _v41CandActive = false;

            for (int il = 0; il < _nLayer; il++)
            {
                Layer L = _layers[il];

                // ---- Engram (V4.1, on the layers the GGUF metadata names) ----
                // Runs before attention and rewrites the residual in place, which
                // is what build_engram does in the native graph.
                if (L.EngramIndex >= 0)
                    EngramLayer(L, nt);

                // ---- attention super-block ----
                // V4.1 delays the hyper-connection collapse by one block: each
                // block computes its own gates but collapses the streams with the
                // PREVIOUS block's, and the very first block uses the stream mean.
                bool dbg = StageDebug && il == 0;
                t0 = Stopwatch.GetTimestamp();
                HcPre(L.HcAttnFn, L.HcAttnScale, L.HcAttnBase, nt, computeComb: true,
                    delayedPre: _isV41 ? (il == 0 ? null : _preFfn) : null,
                    publishPre: _isV41 ? _preAttn : null,
                    meanCollapse: _isV41 && il == 0);
                if (dbg)
                {
                    DumpDbg("L0.attn.mixes", _mixes);
                    DumpDbg("L0.attn.pre", _pre, 4);
                    DumpDbg("L0.attn.comb", _comb, 8);
                    DumpDbg("L0.attn.cur", _cur);
                }
                RmsNormRows(_cur, L.AttnNorm, nt, E);
                if (dbg)
                    DumpDbg("L0.attn.cur_norm", _cur);
                Trace(TraceLayer(il, "attn_input"), _cur, (long)nt * E);
                Tick(1, t0);
                if (_isV41)
                    AttentionV41(il, nt, p0);
                else
                    Attention(il, nt, p0);
                if (dbg)
                    DumpDbg("L0.attn.out", _attnOut);
                Trace(TraceLayer(il, "attn_out"), _attnOut, (long)nt * E);
                t0 = Stopwatch.GetTimestamp();
                HcPost(_attnOut, nt);
                if (dbg)
                    DumpDbg("L0.attn.xs_post", _xs);

                // ---- FFN super-block ----
                HcPre(L.HcFfnFn, L.HcFfnScale, L.HcFfnBase, nt, computeComb: true,
                    delayedPre: _isV41 ? _preAttn : null,
                    publishPre: _isV41 ? _preFfn : null);
                RmsNormRows(_cur, L.FfnNorm, nt, E);
                Trace(TraceLayer(il, "ffn_input"), _cur, (long)nt * E);
                Tick(1, t0);
                MoeFfn(il, nt, tokens, tokOff);
                if (dbg)
                    DumpDbg("L0.ffn.out", _ffnOut);
                Trace(TraceLayer(il, "ffn_out"), _ffnOut, (long)nt * E);
                t0 = Stopwatch.GetTimestamp();
                HcPost(_ffnOut, nt);
                if (dbg)
                    DumpDbg("L0.ffn.xs_post", _xs);
                Trace(TraceLayer(il, "hidden"), _xs, (long)nt * HC * E);
                Tick(1, t0);
            }

            if (logitsOut != null)
            {
                t0 = Stopwatch.GetTimestamp();
                ComputeLogits(nt - 1, logitsOut);
                Tick(10, t0);
            }
        }

        // -------------------------------------------------------------------
        // Engram (V4.1)
        // -------------------------------------------------------------------

        /// <summary>
        /// Gathers this ubatch's Engram rows for one layer and folds them into the
        /// residual. Mirrors build_engram in
        /// TensorSharp.GGML.Native/ggml_ops_deepseek41.inc.
        ///
        /// <para>Per token the layer reads <c>hash_columns</c> rows of
        /// <c>head_dim</c> values out of a table with hundreds of millions of
        /// rows, projects the concatenation through <c>engram_wkv</c> into one
        /// key per hyper-connection stream plus a single shared value, scores each
        /// stream against the residual, and adds the value back through a signed
        /// square-root sigmoid gate.</para>
        ///
        /// <para>The table is never dequantized whole: only the selected rows are,
        /// which is the entire reason a 30 GiB table is affordable here.</para>
        /// </summary>
        private void EngramLayer(Layer L, int nt)
        {
            int E = _nEmbd;
            int columns = (int)_engram.HashColumns;
            int headDim = (int)_engram.HeadDim;
            int rowValues = columns * headDim;
            long tableRows = _engram.Layers[L.EngramIndex].Rows;
            var table = L.EngramEmbd;

            // Sparse gather. Each (token, column) is an independent random row, so
            // this is the one place the executor touches the big table at all.
            fixed (int* hashes = _engramHashes)
            {
                int* mine = hashes + (long)L.EngramIndex * nt * columns;
                PFor(nt, t =>
                {
                    float* dst = _engramLookup + (long)t * rowValues;
                    int* ids = mine + (long)t * columns;
                    for (int c = 0; c < columns; c++)
                    {
                        int row = ids[c];
                        if ((uint)row >= (uint)tableRows)
                            throw new InvalidOperationException("DeepSeek V4.1 Engram lookup is out of bounds");
                        ManagedQuantizedOps.DequantizeRowToFloat32(
                            (int)table.Type, (IntPtr)table.Row(row), dst + c * headDim, headDim);
                    }
                });
            }

            // One projection produces HC keys and the shared value per token.
            MatMul(L.EngramWkv, 0, L.EngramWkv.Ne1, _engramLookup, rowValues, nt, _engramKv, (HC + 1) * E);

            PFor(nt, t =>
            {
                float* kv = _engramKv + (long)t * (HC + 1) * E;
                float* value = kv + (long)HC * E;            // the stream-shared value
                float* qn = _engramQn + (long)t * E;
                float* kn = _engramKn + (long)t * E;
                var valueSpan = new ReadOnlySpan<float>(value, E);

                for (int st = 0; st < HC; st++)
                {
                    float* x = _xs + ((long)t * HC + st) * E;
                    float* key = kv + (long)st * E;

                    // Both sides are RMS-normalized with no learned gain, then
                    // scaled by their own elementwise weight row.
                    NormalizeAndScale(x, L.EngramQ, st * E, qn, E);
                    NormalizeAndScale(key, L.EngramK, st * E, kn, E);

                    float dot = TensorPrimitives.Dot(
                        new ReadOnlySpan<float>(qn, E), new ReadOnlySpan<float>(kn, E)) / MathF.Sqrt(E);

                    // Signed square root, floored away from zero so the gradient
                    // this mirrors stays finite, then a sigmoid.
                    float magnitude = MathF.Sqrt(MathF.Max(MathF.Abs(dot), 1e-6f));
                    float gate = 1.0f / (1.0f + MathF.Exp(-(dot >= 0f ? magnitude : -magnitude)));

                    TensorPrimitives.MultiplyAdd(valueSpan, gate, new ReadOnlySpan<float>(x, E),
                        new Span<float>(x, E));
                }
            });
        }

        /// <summary>RMS-normalizes <paramref name="src"/> over its whole length and
        /// multiplies by a weight row, into <paramref name="dst"/>.</summary>
        private void NormalizeAndScale(float* src, float[] weight, int weightOffset, float* dst, int n)
        {
            var srcSpan = new ReadOnlySpan<float>(src, n);
            float scale = 1.0f / MathF.Sqrt(TensorPrimitives.SumOfSquares(srcSpan) / n + _rmsEps);
            var dstSpan = new Span<float>(dst, n);
            TensorPrimitives.Multiply(srcSpan, scale, dstSpan);
            TensorPrimitives.Multiply(dstSpan, weight.AsSpan(weightOffset, n), dstSpan);
        }

        // -------------------------------------------------------------------
        // Hyper connections
        // -------------------------------------------------------------------

        /// <summary>
        /// Computes mixes = hc_fn(rms(flat(x))) per token, derives the pre/post
        /// sigmoid gates and the Sinkhorn-normalized 4x4 comb matrix, and
        /// collapses the HC streams into _cur (weighted by pre).
        /// </summary>
        /// <summary>
        /// <paramref name="delayedPre"/>, <paramref name="publishPre"/> and
        /// <paramref name="meanCollapse"/> are the V4.1 delayed-gate form: the
        /// block still derives its own gates from the current streams, but
        /// collapses them with the gates the previous block published (or, for
        /// the first block of the model, with a plain stream mean).
        /// </summary>
        private void HcPre(in WeightRef fn, float[] scale, float[] baseW, int nt, bool computeComb,
            float* delayedPre = null, float* publishPre = null, bool meanCollapse = false)
        {
            int E = _nEmbd;
            int flatDim = HC * E;
            float meanWeight = 1.0f / HC;

            // mixes = fn x rms(flat). RMS scaling is folded into the dot result.
            var fnRef = fn;
            PFor(nt, t =>
            {
                float* flat = _xs + (long)t * flatDim;
                double ss = 0;
                for (int i = 0; i < flatDim; i++) ss += (double)flat[i] * flat[i];
                float inv = 1.0f / MathF.Sqrt((float)(ss / flatDim) + _rmsEps);

                float* mixes = _mixes + (long)t * HcMixDim;
                DotRowsF32(fnRef, flat, mixes, HcMixDim);
                for (int r = 0; r < HcMixDim; r++) mixes[r] *= inv;

                float* pre = _pre + (long)t * HC;
                float* post = _post + (long)t * HC;
                for (int s = 0; s < HC; s++)
                {
                    pre[s] = Sigmoid(mixes[s] * scale[0] + baseW[s]) + _hcEps;
                    post[s] = 2.0f * Sigmoid(mixes[HC + s] * scale[1] + baseW[HC + s]);
                }

                if (computeComb)
                {
                    float* comb = _comb + (long)t * HC * HC;
                    float scaleComb = scale[2];
                    for (int isrc = 0; isrc < HC; isrc++)
                    {
                        float max = float.NegativeInfinity;
                        for (int idst = 0; idst < HC; idst++)
                        {
                            int idx = idst + HC * isrc;
                            float v = mixes[2 * HC + idx] * scaleComb + baseW[2 * HC + idx];
                            comb[idx] = v;
                            if (v > max) max = v;
                        }
                        float sum = 0f;
                        for (int idst = 0; idst < HC; idst++)
                        {
                            int idx = idst + HC * isrc;
                            float v = MathF.Exp(comb[idx] - max);
                            comb[idx] = v;
                            sum += v;
                        }
                        float invSum = 1.0f / sum;
                        for (int idst = 0; idst < HC; idst++)
                        {
                            int idx = idst + HC * isrc;
                            comb[idx] = comb[idx] * invSum + _hcEps;
                        }
                    }
                    CombNormCols(comb, _hcEps);
                    for (int i = 1; i < _hcSinkhornIters; i++)
                    {
                        CombNormRows(comb, _hcEps);
                        CombNormCols(comb, _hcEps);
                    }
                }

                // collapse streams: cur[e] = sum_s x[s][e] * w[s]
                float* w = meanCollapse ? null : delayedPre != null ? delayedPre + (long)t * HC : pre;
                float* cur = _cur + (long)t * E;
                var curSpan = new Span<float>(cur, E);
                new ReadOnlySpan<float>(flat, E).CopyTo(curSpan);
                TensorPrimitives.Multiply(curSpan, w == null ? meanWeight : w[0], curSpan);
                for (int s = 1; s < HC; s++)
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(flat + (long)s * E, E),
                        w == null ? meanWeight : w[s], curSpan, curSpan);

                // Publish AFTER the collapse: the gates this block computed are
                // what the NEXT one collapses with.
                if (publishPre != null)
                    new ReadOnlySpan<float>(pre, HC).CopyTo(new Span<float>(publishPre + (long)t * HC, HC));
            });
        }

        private static void CombNormCols(float* comb, float eps)
        {
            for (int idst = 0; idst < HC; idst++)
            {
                float sum = eps;
                for (int isrc = 0; isrc < HC; isrc++) sum += comb[idst + HC * isrc];
                float inv = 1.0f / sum;
                for (int isrc = 0; isrc < HC; isrc++) comb[idst + HC * isrc] *= inv;
            }
        }

        private static void CombNormRows(float* comb, float eps)
        {
            for (int isrc = 0; isrc < HC; isrc++)
            {
                float sum = eps;
                for (int idst = 0; idst < HC; idst++) sum += comb[idst + HC * isrc];
                float inv = 1.0f / sum;
                for (int idst = 0; idst < HC; idst++) comb[idst + HC * isrc] *= inv;
            }
        }

        /// <summary>x[e, idst] = blockOut[e]*post[idst] + sum_isrc residual[e, isrc]*comb[idst, isrc].</summary>
        private void HcPost(float* blockOut, int nt)
        {
            int E = _nEmbd;
            PFor(nt, t =>
            {
                float* x = _xs + (long)t * HC * E;
                float* o = blockOut + (long)t * E;
                float* post = _post + (long)t * HC;
                float* comb = _comb + (long)t * HC * HC;

                // The original residual streams are needed for every idst while x
                // is being overwritten, so mix into a temp buffer and copy back.
                float* outBuf = _hcPostTmp + (long)t * HC * E;
                for (int idst = 0; idst < HC; idst++)
                {
                    float* dst = outBuf + (long)idst * E;
                    var dstSpan = new Span<float>(dst, E);
                    new ReadOnlySpan<float>(o, E).CopyTo(dstSpan);
                    TensorPrimitives.Multiply(dstSpan, post[idst], dstSpan);
                    for (int isrc = 0; isrc < HC; isrc++)
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(x + (long)isrc * E, E), comb[idst + HC * isrc], dstSpan, dstSpan);
                }
                Buffer.MemoryCopy(outBuf, x, (long)HC * E * sizeof(float), (long)HC * E * sizeof(float));
            });
        }

        private float* _hcPostTmp;

        // -------------------------------------------------------------------
        // MoE FFN
        // -------------------------------------------------------------------

        private void MoeFfn(int il, int nt, int[] tokens, int tokOff)
        {
            Layer L = _layers[il];
            int E = _nEmbd;
            int nEx = _nExpert;
            int nUsed = _nExpertUsed;
            float clampExp = il < _swigluClampExp.Length ? _swigluClampExp[il] : 0f;
            float clampSh = il < _swigluClampShexp.Length ? _swigluClampShexp[il] : 0f;

            long t0 = Stopwatch.GetTimestamp();

            // router: logits -> probs = sqrt(softplus), selection, weights.
            // Parallelize over (token, expert-chunk) so decode (nt == 1) still
            // spreads the 256 x 4096 dot products across the pool.
            const int ExpertChunk = 16;
            int nExChunks = (nEx + ExpertChunk - 1) / ExpertChunk;
            PFor(nt * nExChunks, work =>
            {
                int t = work / nExChunks;
                int e0 = work % nExChunks * ExpertChunk;
                int e1 = Math.Min(e0 + ExpertChunk, nEx);
                float* cur = _cur + (long)t * E;
                float* logits = _routerLogits + (long)t * nEx;
                float* probs = _routerProbs + (long)t * nEx;
                var curSpan = new ReadOnlySpan<float>(cur, E);
                for (int e = e0; e < e1; e++)
                {
                    float v = TensorPrimitives.Dot(new ReadOnlySpan<float>(L.GateInp, e * E, E), curSpan);
                    logits[e] = v;
                    float sp = v > 20f ? v : MathF.Log(1f + MathF.Exp(v));
                    probs[e] = MathF.Sqrt(sp);
                }
            });

            PFor(nt, t =>
            {
                float* probs = _routerProbs + (long)t * nEx;
                int* sel = _selExperts + (long)t * nUsed;
                float* w = _selWeights + (long)t * nUsed;
                if (L.Tid2Eid != null)
                {
                    int tid = tokens[tokOff + t];
                    for (int j = 0; j < nUsed; j++)
                        sel[j] = L.Tid2Eid[(long)tid * nUsed + j];
                }
                else
                {
                    SelectTopK(probs, L.ExpProbsBias, nEx, nUsed, sel);
                }

                float sum = 0f;
                for (int j = 0; j < nUsed; j++)
                {
                    w[j] = probs[sel[j]];
                    sum += w[j];
                }
                if (_expertWeightsNorm)
                {
                    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
                    float inv = 1.0f / sum;
                    for (int j = 0; j < nUsed; j++) w[j] *= inv;
                }
                if (_expertWeightsScale != 0f && _expertWeightsScale != 1f)
                    for (int j = 0; j < nUsed; j++) w[j] *= _expertWeightsScale;
            });

            if (StageDebug && il == 0)
            {
                DumpDbg("L0.router", _routerLogits);
                DumpDbg("L0.selw", _selWeights, nUsed);
            }
            Tick(7, t0);
            t0 = Stopwatch.GetTimestamp();

            NativeMemory.Clear(_ffnOut, (nuint)((long)nt * E * sizeof(float)));

            if (nt == 1)
            {
                if (!TryRunExpertsDecodeFused(L, clampExp))
                {
                    // fallback: run the selected experts one by one
                    for (int j = 0; j < nUsed; j++)
                    {
                        int e = _selExperts[j];
                        float w = _selWeights[j];
                        ExpertMatmuls(L, e, clampExp, _cur, 1, _expertDown);
                        var dst = new Span<float>(_ffnOut, E);
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_expertDown, E), w, dst, dst);
                    }
                }
            }
            else if (!TryRunExpertsPrefillFused(L, nt, clampExp))
            {
                // fallback: group tokens by expert to amortize weight dequantization
                var byExpert = new Dictionary<int, List<(int Token, float Weight)>>();
                for (int t = 0; t < nt; t++)
                {
                    for (int j = 0; j < nUsed; j++)
                    {
                        int e = _selExperts[(long)t * nUsed + j];
                        if (!byExpert.TryGetValue(e, out var list))
                            byExpert[e] = list = new List<(int, float)>();
                        list.Add((t, _selWeights[(long)t * nUsed + j]));
                    }
                }

                // deterministic accumulation order (float sums depend on it)
                var expertIds = new List<int>(byExpert.Keys);
                expertIds.Sort();
                foreach (int e in expertIds)
                {
                    var list = byExpert[e];
                    int m = list.Count;
                    for (int i = 0; i < m; i++)
                    {
                        float* src = _cur + (long)list[i].Token * E;
                        Buffer.MemoryCopy(src, _expertPack + (long)i * E, E * sizeof(float), E * sizeof(float));
                    }

                    ExpertMatmuls(L, e, clampExp, _expertPack, m, _expertDown);

                    for (int i = 0; i < m; i++)
                    {
                        float w = list[i].Weight;
                        float* dst = _ffnOut + (long)list[i].Token * E;
                        var dstSpan = new Span<float>(dst, E);
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_expertDown + (long)i * E, E), w, dstSpan, dstSpan);
                    }
                }
            }

            Tick(8, t0);
            t0 = Stopwatch.GetTimestamp();

            // shared expert (dense) — add into _ffnOut
            {
                int shFf = L.UpShexp.Ne1;
                MatMul(L.UpShexp, 0, shFf, _cur, E, nt, _shUp, shFf);
                MatMul(L.GateShexp, 0, shFf, _cur, E, nt, _shGate, shFf);
                SwigluClamp(_shGate, _shUp, (long)nt * shFf, clampSh);
                MatMul(L.DownShexp, 0, E, _shGate, shFf, nt, _shDown, E);
                var all = new Span<float>(_ffnOut, nt * E);
                TensorPrimitives.Add(new ReadOnlySpan<float>(_shDown, nt * E), all, all);
            }
            Tick(9, t0);
        }

        /// <summary>
        /// Decode-path MoE: all selected experts' gate+up matvecs in one parallel
        /// region sharing a single quantized activation, one swiglu sweep, then all
        /// down matvecs in a second region. Cuts the per-layer region count from
        /// ~20 to 2 and skips redundant activation quantization — that matters
        /// under a CPU-quota cgroup where every fork/join risks a throttle stall.
        /// Returns false when the expert tensors lack a direct integer-dot plan.
        /// </summary>
        private bool TryRunExpertsDecodeFused(Layer L, float clamp)
        {
            int E = _nEmbd, ff = _nFfExp, nUsed = _nExpertUsed;
            WeightRef gateW = L.GateExps, upW = L.UpExps, downW = L.DownExps;

            if (upW.Type != gateW.Type || upW.Ne0 != gateW.Ne0)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(gateW.Type, E, out int actBytesGU) || actBytesGU > _maxActRowBytes)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(downW.Type, ff, out int actBytesDown) || actBytesDown > _maxActRowBytes)
                return false;

            // slot 0: gate/up activation; slots 1..nUsed: per-expert down activations
            byte* actGU = _actQuant;
            ManagedQuantizedOps.QuantizeActivationRow(gateW.Type, _cur, actGU, E);

            const int RowsPerTask = 128;
            int tasksPerMat = (ff + RowsPerTask - 1) / RowsPerTask;
            int totalA = nUsed * 2 * tasksPerMat;
            PFor(totalA, w =>
            {
                int slot = w / (2 * tasksPerMat);
                int rem = w % (2 * tasksPerMat);
                bool isGate = rem < tasksPerMat;
                int r0 = (isGate ? rem : rem - tasksPerMat) * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, ff);
                int e = _selExperts[slot];
                WeightRef wRef = isGate ? L.GateExps : L.UpExps;
                byte* basePtr = wRef.Expert(e);
                float* dst = (isGate ? _expertGate : _expertUp) + (long)slot * ff;
                for (int r = r0; r < r1; r++)
                    dst[r] = ManagedQuantizedOps.DotQuantizedRow(wRef.Type, basePtr + r * wRef.RowBytes, actGU, E);
            });

            SwigluClamp(_expertGate, _expertUp, (long)nUsed * ff, clamp);

            for (int slot = 0; slot < nUsed; slot++)
                ManagedQuantizedOps.QuantizeActivationRow(downW.Type, _expertGate + (long)slot * ff, _actQuant + (long)(slot + 1) * _maxActRowBytes, ff);

            int tasksPerDown = (E + RowsPerTask - 1) / RowsPerTask;
            PFor(nUsed * tasksPerDown, w =>
            {
                int slot = w / tasksPerDown;
                int r0 = w % tasksPerDown * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, E);
                int e = _selExperts[slot];
                byte* basePtr = L.DownExps.Expert(e);
                byte* act = _actQuant + (long)(slot + 1) * _maxActRowBytes;
                float* dst = _expertDown + (long)slot * E;
                for (int r = r0; r < r1; r++)
                    dst[r] = ManagedQuantizedOps.DotQuantizedRow(L.DownExps.Type, basePtr + r * L.DownExps.RowBytes, act, ff);
            });

            var outSpan = new Span<float>(_ffnOut, E);
            for (int slot = 0; slot < nUsed; slot++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(_expertDown + (long)slot * E, E), _selWeights[slot], outSpan, outSpan);

            return true;
        }

        /// <summary>
        /// Prefill-path MoE: tokens are packed into per-expert groups, then the
        /// whole layer runs as six wide parallel regions (quantize, gate+up dots,
        /// swiglu, re-quantize, down dots, weighted scatter) instead of three
        /// regions per active expert. Weight rows are read once per expert and
        /// dotted against every member token.
        /// </summary>
        private bool TryRunExpertsPrefillFused(Layer L, int nt, float clamp)
        {
            int E = _nEmbd, ff = _nFfExp, nUsed = _nExpertUsed, nEx = _nExpert;
            WeightRef gateW = L.GateExps, upW = L.UpExps, downW = L.DownExps;

            if (upW.Type != gateW.Type || upW.Ne0 != gateW.Ne0)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(gateW.Type, E, out int actBytesGU) || actBytesGU > _maxActRowBytes)
                return false;
            if (!ManagedQuantizedOps.TryGetActivationPlan(downW.Type, ff, out int actBytesDown) || actBytesDown > _maxActRowBytes)
                return false;

            // group (token, slot) pairs by expert; packed row s is the position
            // in expert-major order
            int S = nt * nUsed;
            for (int e = 0; e < nEx; e++) _expCount[e] = 0;
            for (int i = 0; i < S; i++) _expCount[_selExperts[i]]++;
            int off = 0;
            for (int e = 0; e < nEx; e++) { _expOffset[e] = off; off += _expCount[e]; _expCursor[e] = _expOffset[e]; }
            for (int t = 0; t < nt; t++)
            {
                for (int j = 0; j < nUsed; j++)
                {
                    int i = t * nUsed + j;
                    int e = _selExperts[i];
                    int row = _expCursor[e]++;
                    _rowOfSlot[i] = row;
                    _slotToken[row] = t;
                    _slotGroupSize[row] = _expCount[e];
                }
            }

            // region 1: quantize all token activations for gate/up
            byte* actGU = _actQuant;                          // rows [0, nt)
            byte* actDown = _actQuant + (long)nt * _maxActRowBytes; // rows [0, S)
            PFor(nt, t =>
                ManagedQuantizedOps.QuantizeActivationRow(gateW.Type, _cur + (long)t * E, actGU + (long)t * actBytesGU, E));

            // region 2: gate+up dots, task = (expert, mat, ff-row-block).
            // For groups with enough member tokens it is cheaper to dequantize
            // each weight row once and run float dots (this is where a batched
            // prefill amortizes the IQ3_S decode cost); small groups keep the
            // integer path.
            const int RowsPerTask = 128;
            int blocksFF = (ff + RowsPerTask - 1) / RowsPerTask;
            PFor(nEx * 2 * blocksFF, w =>
            {
                int e = w / (2 * blocksFF);
                int m = _expCount[e];
                if (m == 0) return;
                int rem = w % (2 * blocksFF);
                bool isGate = rem < blocksFF;
                int r0 = (isGate ? rem : rem - blocksFF) * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, ff);
                WeightRef wRef = isGate ? L.GateExps : L.UpExps;
                byte* basePtr = wRef.Expert(e);
                float* dstBase = (isGate ? _expertGate : _expertUp);
                int groupOff = _expOffset[e];
                if (m >= DequantGroupMin)
                {
                    float* wrow = stackalloc float[E];
                    for (int r = r0; r < r1; r++)
                    {
                        ManagedQuantizedOps.DequantizeRowToFloat32((int)wRef.Type, (IntPtr)(basePtr + r * wRef.RowBytes), wrow, E);
                        var wSpan = new ReadOnlySpan<float>(wrow, E);
                        for (int i = 0; i < m; i++)
                        {
                            int tok = _slotToken[groupOff + i];
                            dstBase[(long)(groupOff + i) * ff + r] =
                                TensorPrimitives.Dot(wSpan, new ReadOnlySpan<float>(_cur + (long)tok * E, E));
                        }
                    }
                }
                else
                {
                    for (int r = r0; r < r1; r++)
                    {
                        byte* wr = basePtr + r * wRef.RowBytes;
                        for (int i = 0; i < m; i++)
                        {
                            int tok = _slotToken[groupOff + i];
                            dstBase[(long)(groupOff + i) * ff + r] =
                                ManagedQuantizedOps.DotQuantizedRow(wRef.Type, wr, actGU + (long)tok * actBytesGU, E);
                        }
                    }
                }
            });

            // region 3: swiglu over all packed rows
            SwigluClamp(_expertGate, _expertUp, (long)S * ff, clamp);

            // region 4: quantize the packed hidden rows for the down matmul —
            // only for slots whose expert group takes the integer path
            PFor(S, s =>
            {
                if (_slotGroupSize[s] >= DequantGroupMin)
                    return;
                ManagedQuantizedOps.QuantizeActivationRow(downW.Type, _expertGate + (long)s * ff, actDown + (long)s * actBytesDown, ff);
            });

            // region 5: down dots, task = (expert, E-row-block); same hybrid
            int blocksE = (E + RowsPerTask - 1) / RowsPerTask;
            PFor(nEx * blocksE, w =>
            {
                int e = w / blocksE;
                int m = _expCount[e];
                if (m == 0) return;
                int r0 = w % blocksE * RowsPerTask;
                int r1 = Math.Min(r0 + RowsPerTask, E);
                byte* basePtr = L.DownExps.Expert(e);
                int groupOff = _expOffset[e];
                if (m >= DequantGroupMin)
                {
                    float* wrow = stackalloc float[ff];
                    for (int r = r0; r < r1; r++)
                    {
                        ManagedQuantizedOps.DequantizeRowToFloat32((int)L.DownExps.Type, (IntPtr)(basePtr + r * L.DownExps.RowBytes), wrow, ff);
                        var wSpan = new ReadOnlySpan<float>(wrow, ff);
                        for (int i = 0; i < m; i++)
                            _expertDown[(long)(groupOff + i) * E + r] =
                                TensorPrimitives.Dot(wSpan, new ReadOnlySpan<float>(_expertGate + (long)(groupOff + i) * ff, ff));
                    }
                }
                else
                {
                    for (int r = r0; r < r1; r++)
                    {
                        byte* wr = basePtr + r * L.DownExps.RowBytes;
                        for (int i = 0; i < m; i++)
                            _expertDown[(long)(groupOff + i) * E + r] =
                                ManagedQuantizedOps.DotQuantizedRow(L.DownExps.Type, wr, actDown + (long)(groupOff + i) * actBytesDown, ff);
                    }
                }
            });

            // region 6: weighted scatter back to tokens (deterministic per token)
            PFor(nt, t =>
            {
                var dst = new Span<float>(_ffnOut + (long)t * E, E);
                for (int j = 0; j < nUsed; j++)
                {
                    int i = t * nUsed + j;
                    TensorPrimitives.MultiplyAdd(
                        new ReadOnlySpan<float>(_expertDown + (long)_rowOfSlot[i] * E, E),
                        _selWeights[i], dst, dst);
                }
            });

            return true;
        }

        /// <summary>gate/up/swiglu/down for one expert over m packed rows; result in downOut [m x E].</summary>
        private void ExpertMatmuls(Layer L, int e, float clamp, float* packedIn, int m, float* downOut)
        {
            int E = _nEmbd;
            int ff = _nFfExp;
            MatMulExpert(L.UpExps, e, packedIn, E, m, _expertUp, ff);
            MatMulExpert(L.GateExps, e, packedIn, E, m, _expertGate, ff);
            SwigluClamp(_expertGate, _expertUp, (long)m * ff, clamp);
            MatMulExpert(L.DownExps, e, _expertGate, ff, m, downOut, E);
        }

        /// <summary>up = clamp(up, ±limit); gate = min(gate, limit); gate = silu(gate) * up
        /// (written into gate), through the shared Ops.SiLUMulClamp.</summary>
        private void SwigluClamp(float* gate, float* up, long n, float clamp)
        {
            using Tensor g = View(gate, 1, n);
            using Tensor u = View(up, 1, n);
            Ops.SiLUMulClamp(g, g, u, clamp > 1e-6f ? clamp : 0.0f);
        }

        private static void SelectTopK(float* probs, float[] bias, int n, int k, int* outIdx)
        {
            // simple partial selection (k is small: 6)
            float* best = stackalloc float[k];
            for (int j = 0; j < k; j++) { best[j] = float.NegativeInfinity; outIdx[j] = 0; }
            for (int e = 0; e < n; e++)
            {
                float v = probs[e] + bias[e];
                if (v <= best[k - 1]) continue;
                int j = k - 1;
                while (j > 0 && best[j - 1] < v)
                {
                    best[j] = best[j - 1];
                    outIdx[j] = outIdx[j - 1];
                    j--;
                }
                best[j] = v;
                outIdx[j] = e;
            }
        }

        // -------------------------------------------------------------------
        // Output head
        // -------------------------------------------------------------------

        private void ComputeLogits(int t, float[] logitsOut)
        {
            int E = _nEmbd;
            int flatDim = HC * E;
            float* x = _xs + (long)t * flatDim;

            float* headBuf = stackalloc float[HC];
            if (_isV41)
            {
                // V4.1 has no head mixer: the last layer's FFN block already
                // published the gates the head collapses with.
                new ReadOnlySpan<float>(_preFfn + (long)t * HC, HC).CopyTo(new Span<float>(headBuf, HC));
            }
            else
            {
                // hc_head: mixes = output_hc_fn(rms(flat)); pre = sigmoid(m*scale+base)+eps; collapse
                float* mixes = stackalloc float[HC];
                double ss = 0;
                for (int i = 0; i < flatDim; i++) ss += (double)x[i] * x[i];
                float inv = 1.0f / MathF.Sqrt((float)(ss / flatDim) + _rmsEps);
                DotRowsF32(_hcHeadFn, x, mixes, HC);

                for (int s = 0; s < HC; s++)
                {
                    float scale = _hcHeadScale.Length >= HC ? _hcHeadScale[s] : _hcHeadScale[0];
                    float b = _hcHeadBase.Length >= HC ? _hcHeadBase[s] : _hcHeadBase[0];
                    headBuf[s] = Sigmoid(mixes[s] * inv * scale + b) + _hcEps;
                }
            }

            float* cur = _cur; // reuse token-0 slot
            var curSpan = new Span<float>(cur, E);
            new ReadOnlySpan<float>(x, E).CopyTo(curSpan);
            TensorPrimitives.Multiply(curSpan, headBuf[0], curSpan);
            for (int s = 1; s < HC; s++)
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(x + (long)s * E, E), headBuf[s], curSpan, curSpan);

            DumpDbg("head.cur", cur);
            RmsNormRows(cur, _outputNorm, 1, E);

            fixed (float* lo = logitsOut)
            {
                MatMul(_output, 0, _nVocab, cur, E, 1, lo, _nVocab);
                DumpDbg("head.logits", lo, 8);
                Trace("logits_last", lo, _nVocab);
            }
        }

        // -------------------------------------------------------------------
        // Primitives
        // -------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));

        /// <summary>
        /// Small dense weights (norms) as Tensors, built once and cached by array
        /// identity so the shared Ops can take them without a per-call copy.
        /// </summary>
        // float[] has no value equality, so the default comparer is reference identity.
        private readonly Dictionary<float[], Tensor> _weightTensors = new Dictionary<float[], Tensor>();

        private Tensor WeightTensor(float[] w)
        {
            if (w == null)
                return null;
            if (_weightTensors.TryGetValue(w, out Tensor t))
                return t;
            t = new Tensor(_alloc, DType.Float32, w.Length);
            float* dst = (float*)CpuNativeHelpers.GetBufferStart(t);
            new ReadOnlySpan<float>(w).CopyTo(new Span<float>(dst, w.Length));
            _weightTensors[w] = t;
            return t;
        }

        /// <summary>RMS-norm each of nt rows of len n in place, through the shared
        /// Ops.RMSNorm (the executor no longer carries its own kernel).</summary>
        private void RmsNormRows(float* data, float[] weight, int nt, int n)
        {
            using Tensor rows = View(data, nt, n);
            Ops.RMSNorm(rows, rows, WeightTensor(weight), null, _rmsEps);
        }

        /// <summary>Dot input (len = w.Ne0) against the first nRows rows of an F32/any-typed small weight, serially.</summary>
        private static void DotRowsF32(in WeightRef w, float* input, float* output, int nRows)
        {
            if (w.Type == GgmlTensorType.F32)
            {
                var inSpan = new ReadOnlySpan<float>(input, w.Ne0);
                for (int r = 0; r < nRows; r++)
                    output[r] = TensorPrimitives.Dot(new ReadOnlySpan<float>((float*)w.Row(r), w.Ne0), inSpan);
            }
            else
            {
                ManagedQuantizedOps.DotRowBatchToFloat32((int)w.Type, (IntPtr)w.Ptr, input, w.Ne0, 1, w.Ne0, output);
                for (int r = 1; r < nRows; r++)
                    ManagedQuantizedOps.DotRowBatchToFloat32((int)w.Type, (IntPtr)w.Row(r), input, w.Ne0, 1, w.Ne0, output + r);
            }
        }

        /// <summary>
        /// output[t, r] = dot(weight row (rowBase + r), input row t) for r in [0, nRows), t in [0, nt).
        /// Parallelizes over blocked output rows; uses integer dot kernels when the
        /// weight type supports them (activations quantized once per call).
        /// </summary>
        private void MatMul(in WeightRef w, long rowBase, int nRows, float* input, int inStride, int nt, float* output, int outStride)
        {
            // One managed quantized-linear implementation for the whole library
            // (see ManagedQuantizedOps.AddmmQuantizedToFloat32): integer dot
            // kernels for the types that have them, dequant-once + register-
            // blocked float dots for the rest. rowBase/nRows select the weight
            // row block (grouped LoRA out-projection, expert slices), and the
            // strides let the operands sit inside wider scratch buffers.
            ManagedQuantizedOps.AddmmQuantizedToFloat32(
                (int)w.Type,
                (IntPtr)w.Row(rowBase),
                w.Ne0,
                nRows,
                input,
                inStride,
                nt,
                output,
                outStride,
                _po);
        }

        /// <summary>MatMul against expert e of a stacked 3D expert tensor.</summary>
        private void MatMulExpert(in WeightRef w, int e, float* input, int inStride, int nt, float* output, int outStride)
        {
            var view = w;
            view.Ptr = w.Expert(e);
            MatMul(view, 0, w.Ne1, input, inStride, nt, output, outStride);
        }

        public void Dispose()
        {
            foreach (var t in _tensors)
                t.Dispose();
            _tensors.Clear();
            _byPtr.Clear();
            foreach (var t in _weightTensors.Values)
                t.Dispose();
            _weightTensors.Clear();
            foreach (var p in _ownedBuffers)
                Marshal.FreeHGlobal(p);
            _ownedBuffers.Clear();
            foreach (var s in _shards)
                s.Dispose();
            _shards.Clear();
        }
    }
}
