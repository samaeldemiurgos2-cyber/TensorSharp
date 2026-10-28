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
// ============================================================================
// NemotronModel.TensorParallelGgmlMoE.cs
//
// Expert-parallel MoE for the Nemotron tensor-parallel path on the GGML backends.
//
// The Megatron split this replaces slices INSIDE every expert (up column-parallel,
// down row-parallel). That leaves each rank holding a piece of all 128 experts, so
// a layer can no longer be one batched dispatch and degenerates into a per-expert
// loop: at 24 MoE layers x 2 ranks x up-to-128 active experts x 2 projections that
// is ~12k tiny matmuls per prefill chunk, each wrapped in a host gather and a host
// scatter-add of every routed row.
//
// Whole experts partition cleanly instead. The expert index is the stacked tensor's
// OUTER dimension, so rank r's share is a contiguous byte range - a zero-copy view -
// and each rank runs the same batched ggml_mul_mat_id kernel the single-GPU path
// uses. Each rank sums only the experts it owns, so the AllReduce the block already
// performs is exactly the right recombination.
//
// Nemotron's routed experts are ReLU-squared single-projection (up -> relu(x)^2 ->
// down, no gate), which GgmlBasicOps.MoEFFNPrefill expresses by passing the single
// projection as gateData and leaving upData null.
//
// The shared expert is deliberately NOT touched here: it stays replicated on rank 0
// and is added by NemotronMoEBlockTP after the AllReduce, exactly as before.
//
// Nemotron 3.5 Super VL and Bolt run their experts in a 1024-wide latent space
// (ffn_latent_in -> experts -> ffn_latent_out): the experts then read and write the
// latent vector, which NemotronMoEBlockTP projects once and hands every rank. Bolt
// also gives each MoE block its own expert count and top-k, so the partition is per
// layer. Super VL's NVFP4 experts carry a per-expert up scale2 that has to be applied
// before ReLU^2; it goes to the kernel as one multiplier per route.
// ============================================================================
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class NemotronModel
    {
        // Per-rank slices of the stacked expert weights, [layer][local rank].
        private StackedExpertWeights[][] _tpStackedUp;
        private StackedExpertWeights[][] _tpStackedDown;
        // Experts each rank owns on a layer (a block's expert count / global TP degree).
        private int[] _tpExpertsPerRank;

        /// <summary>
        /// True when the routed experts are partitioned by whole expert rather than
        /// sliced inside each expert.
        /// </summary>
        private bool UsesExpertParallelMoE => _tpStackedUp != null;

        /// <summary>
        /// Whether this model/backend combination can take the expert-parallel path.
        /// Deliberately shape-based rather than reading <c>_moeLayerInfo</c>: this
        /// runs during sharding, which happens BEFORE InitLayerInfo populates that.
        /// </summary>
        private bool CanUseNemotronExpertParallelMoE() => DescribeNemotronExpertParallelRefusal() == null;

        /// <summary>
        /// Why the routed experts cannot be partitioned by whole expert, or null when they
        /// can. Per layer, because Bolt gives each MoE block its own expert count and top-k:
        /// every rank must own a whole share of each layer's experts, and at least as many
        /// as that layer routes a token to (an absent route is pointed at an unused LOCAL
        /// expert with weight 0, so a spare must exist). The experts read and write either
        /// the hidden vector or, for a latent MoE, the latent one.
        /// </summary>
        private string DescribeNemotronExpertParallelRefusal()
        {
            int tp = GlobalTpDegree;
            if (!IsGgmlBackend) return "expert parallelism runs on the GGML backends";
            if (_numExperts <= 0 || tp <= 1) return "the model has no routed experts to partition";
            if (_layerTypes == null || _stackedExpertWeights == null || _numExpertsPerLayer == null)
                return "the stacked expert weights are not available";

            int hiddenSize = Config.HiddenSize;
            bool sawLayer = false;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.FFN)
                    continue;
                int experts = _numExpertsPerLayer[l];
                int used = _numExpertsUsedPerLayer[l];
                if (experts <= 0 || used <= 0)
                    continue;
                if (experts % tp != 0)
                    return $"layer {l} has {experts} experts, not divisible by {tp} ranks";
                if (experts / tp < used)
                    return $"layer {l} routes to {used} experts but a rank would own only {experts / tp}";
                if (!_stackedExpertWeights.TryGetValue($"blk.{l}.ffn_up_exps.weight", out var up) || up == null
                    || !_stackedExpertWeights.TryGetValue($"blk.{l}.ffn_down_exps.weight", out var down) || down == null)
                    return $"layer {l} has no stacked ffn_up_exps/ffn_down_exps";
                if (up.NumExperts != experts || down.NumExperts != experts)
                    return $"layer {l} stacks {up.NumExperts}/{down.NumExperts} experts, expected {experts}";
                // The experts' input and output width: hidden, or the latent width for a
                // latent MoE. Named by weight rather than inferred from the expert shape,
                // so a checkpoint whose latent width happens to equal hidden still reads
                // the latent vector.
                int expertDim = NemotronExpertDimForSharding(l, hiddenSize);
                if (up.PerExpertNe0 != expertDim || down.PerExpertNe1 != expertDim)
                    return $"layer {l} experts are [{up.PerExpertNe0} -> {down.PerExpertNe1}], expected {expertDim}";
                if (up.PerExpertNe1 != down.PerExpertNe0)
                    return $"layer {l} expert widths disagree ({up.PerExpertNe1} vs {down.PerExpertNe0})";
                sawLayer = true;
            }
            return sawLayer ? null : "no MoE layer has stacked experts";
        }

        /// <summary>
        /// Width the routed experts of <paramref name="layer"/> read and write: the latent
        /// width when the layer has an ffn_latent_in projection, else hidden. Runs during
        /// sharding, before InitLayerInfo fills <c>_moeLayerInfo</c>, so it reads the
        /// weights directly.
        /// </summary>
        private int NemotronExpertDimForSharding(int layer, int hiddenSize)
        {
            string latentIn = $"blk.{layer}.ffn_latent_in.weight";
            if (_quantWeights.TryGetValue(latentIn, out var q) && q != null)
                return (int)q.Ne1;
            if (_weights.TryGetValue(latentIn, out var w) && w != null)
                return (int)w.Sizes[0];
            return hiddenSize;
        }

        /// <summary>
        /// Build the per-rank whole-expert slices. Returns false when the model did
        /// not expose usable stacked expert weights, leaving the caller on the
        /// per-expert sharding path.
        ///
        /// All-or-nothing on purpose: a per-layer bail-out would leave some layers
        /// with no per-expert shards AND no stacked shard, and the per-expert
        /// fallback would then throw looking up weights that were never created.
        /// </summary>
        private bool BuildNemotronExpertParallelShards()
        {
            string refusal = DescribeNemotronExpertParallelRefusal();
            if (refusal != null)
            {
                if (IsGgmlBackend && _numExperts > 0)
                    Console.WriteLine($"  Nemotron MoE: per-expert tensor parallelism ({refusal}).");
                return false;
            }

            int globalTp = GlobalTpDegree;
            int localTp = TpDegree;
            int rankOffset = TpRankOffset;
            int n = Config.NumLayers;

            var up = new StackedExpertWeights[n][];
            var down = new StackedExpertWeights[n][];
            var perRankByLayer = new int[n];
            int minPerRank = int.MaxValue, maxPerRank = 0;

            for (int layer = 0; layer < n; layer++)
            {
                if (_layerTypes[layer] != LayerType.FFN || _numExpertsPerLayer[layer] <= 0)
                    continue;
                int perRank = _numExpertsPerLayer[layer] / globalTp;
                perRankByLayer[layer] = perRank;
                minPerRank = Math.Min(minPerRank, perRank);
                maxPerRank = Math.Max(maxPerRank, perRank);

                var srcUp = _stackedExpertWeights[$"blk.{layer}.ffn_up_exps.weight"];
                var srcDown = _stackedExpertWeights[$"blk.{layer}.ffn_down_exps.weight"];

                up[layer] = new StackedExpertWeights[localTp];
                down[layer] = new StackedExpertWeights[localTp];
                for (int lr = 0; lr < localTp; lr++)
                {
                    int firstExpert = (rankOffset + lr) * perRank;
                    up[layer][lr] = SliceNemotronStackedExperts(srcUp, firstExpert, perRank);
                    down[layer][lr] = SliceNemotronStackedExperts(srcDown, firstExpert, perRank);
                }
            }

            _tpStackedUp = up;
            _tpStackedDown = down;
            _tpExpertsPerRank = perRankByLayer;

            string share = minPerRank == maxPerRank ? $"{minPerRank}" : $"{minPerRank}-{maxPerRank}";
            Console.WriteLine(
                $"  Nemotron MoE: expert-parallel across {globalTp} GPU(s), {share} experts per GPU per layer "
                + "(one batched ggml_mul_mat_id dispatch per projection per layer).");
            return true;
        }

        /// <summary>
        /// Zero-copy view of <paramref name="count"/> consecutive experts. The expert
        /// index is the stacked tensor's outer dimension, so this is a byte offset -
        /// no copy, and each rank's device cache holds only its own slice.
        /// </summary>
        private static StackedExpertWeights SliceNemotronStackedExperts(
            StackedExpertWeights src, int firstExpert, int count)
        {
            long perExpertBytes = src.PerExpertRawBytes;
            return new StackedExpertWeights(
                new IntPtr(src.Data.ToInt64() + firstExpert * perExpertBytes),
                src.GgmlType,
                src.PerExpertNe0,
                src.PerExpertNe1,
                count,
                perExpertBytes * count,
                isExternalView: true,
                ownerToken: src,
                ownedBuffer: IntPtr.Zero);
        }

        /// <summary>
        /// Make each rank's expert slice device-resident at load time. Without this
        /// the first forward pays the whole upload, which looks like a hang.
        /// </summary>
        protected override void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
        {
            // Runs inside the per-rank preload fan-out: the calling thread is already
            // pinned to this rank's GPU and every rank uploads concurrently.
            if (!UsesExpertParallelMoE)
                return;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_layerTypes[layer] != LayerType.FFN || _tpStackedUp[layer] == null)
                    continue;
                // --n-cpu-moe: this layer's experts are multiplied on the host out of
                // the GGUF mmap. Not uploading them IS the VRAM saving.
                if (MoeCpuOffloadConfig.IsLayerOnCpu(layer))
                    continue;
                PreloadNemotronStackedShard(_tpStackedUp[layer][rank], bytesPerRank, countPerRank, rank);
                PreloadNemotronStackedShard(_tpStackedDown[layer][rank], bytesPerRank, countPerRank, rank);
            }
        }

        private static void PreloadNemotronStackedShard(
            StackedExpertWeights w, long[] bytesPerRank, int[] countPerRank, int rank)
        {
            if (w == null) return;
            // The MoE kernel looks the buffer up by its data pointer, so the cache
            // key must be that same pointer.
            if (GgmlBasicOps.PreloadQuantizedWeight(
                    w.Data, w.Data, w.GgmlType, w.PerExpertNe0, w.PerExpertNe1 * w.NumExperts, w.TotalRawBytes))
            {
                bytesPerRank[rank] += w.TotalRawBytes;
                countPerRank[rank]++;
            }
        }

        /// <summary>
        /// Under expert parallelism the per-expert weights are only ever read through
        /// the stacked per-rank slices above. They are still present in
        /// <c>_quantWeights</c> (sharding skipped them), and the replicated-weight
        /// preload would otherwise push a full unsharded copy of every expert - the
        /// whole ~30 GB of them - onto rank 0.
        /// </summary>
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !(UsesExpertParallelMoE && _stackedExpertMemberNames.Contains(weightName))
               && base.ShouldPreloadCudaQuantWeightToDevice(weightName);

        /// <summary>
        /// The same weights, but for the load-time ACCOUNTING rather than the upload.
        /// Without this the per-expert entries left in <c>_quantWeights</c> are tallied
        /// as "replicated on rank 0" - ~30 GB of tensors that are not resident there at
        /// all - and startup then warns that TP sharded almost nothing and the run is
        /// probably slower than one GPU. Both statements would be false, and that
        /// warning is worth keeping honest so it still means something when a real
        /// regression puts it back.
        /// </summary>
        protected override bool IsSupersededByTpShard(string weightName)
            => (UsesExpertParallelMoE && _stackedExpertMemberNames.Contains(weightName))
               || base.IsSupersededByTpShard(weightName);

        /// <summary>
        /// Expert parallelism gives every rank ONE batched dispatch per projection per
        /// layer, so the MoE block is no longer the slow per-expert loop that the
        /// lightweight startup warmup exists to dodge.
        /// </summary>
        protected override bool MoEUnderTpIsSlow => IsTensorParallel && !UsesExpertParallelMoE;

        /// <summary>
        /// The routed experts are the bulk of this model and they are split by whole
        /// expert, not through <c>_tpQuantWeights</c>. Report those bytes so the
        /// "TP sharded almost nothing" warning reflects what was actually split.
        /// </summary>
        protected override long AdditionalTpShardedBytes
        {
            get
            {
                if (!UsesExpertParallelMoE)
                    return 0;
                long total = 0;
                for (int layer = 0; layer < Config.NumLayers; layer++)
                {
                    if (_tpStackedUp[layer] == null)
                        continue;
                    for (int r = 0; r < TpDegree; r++)
                    {
                        total += _tpStackedUp[layer][r]?.TotalRawBytes ?? 0;
                        total += _tpStackedDown[layer][r]?.TotalRawBytes ?? 0;
                    }
                }
                return total;
            }
        }

        /// <summary>
        /// One rank's dense route table for the expert-parallel kernel, which takes a
        /// fixed <paramref name="nUsed"/> routes per token. A route to one of this rank's
        /// experts (<paramref name="first"/> .. first + perRank) keeps its weight, as a
        /// local id. A route owned by another rank still occupies its slot: it gets weight
        /// 0 and a DISTINCT unused local id. Real top-k routing never repeats an expert
        /// within a token and the batched kernel's per-expert gather/scatter relies on
        /// that - the id list is collapsed per token while the slot count is not, so a
        /// duplicate MISPLACES rows rather than being harmlessly ignored. perRank &gt;=
        /// nUsed (checked at load) guarantees a free id exists.
        ///
        /// With <paramref name="upScale"/> (per global expert), every route - owned or
        /// neutralised - takes the up scale of the expert it actually runs: a weight-0 route
        /// left unscaled would square an un-normalised NVFP4 projection, which can overflow,
        /// and Inf * 0 is NaN.
        /// </summary>
        internal static void BuildExpertParallelRoutes(
            int[] selectedExperts, float[] routeWeights, int seqLen, int nUsed, int first, int perRank,
            float[] upScale, out int[] ids, out float[] weights, out float[] upScales)
        {
            int totalRoutes = seqLen * nUsed;
            int last = first + perRank;
            ids = new int[totalRoutes];
            weights = new float[totalRoutes];
            var taken = new bool[perRank];

            for (int s = 0; s < seqLen; s++)
            {
                int baseIdx = s * nUsed;
                Array.Clear(taken, 0, perRank);

                // Own routes first, so the filler pass can see which local experts
                // this token already uses.
                for (int k = 0; k < nUsed; k++)
                {
                    int i = baseIdx + k;
                    int e = selectedExperts[i];
                    if (e >= first && e < last)
                    {
                        int local = e - first;
                        ids[i] = local;
                        taken[local] = true;
                        weights[i] = routeWeights[i];
                    }
                    else
                    {
                        ids[i] = -1;
                        weights[i] = 0f;
                    }
                }

                int probe = 0;
                for (int k = 0; k < nUsed; k++)
                {
                    int i = baseIdx + k;
                    if (ids[i] >= 0) continue;
                    while (probe < perRank && taken[probe]) probe++;
                    if (probe < perRank) { taken[probe] = true; ids[i] = probe; }
                    else ids[i] = 0;
                }
            }

            upScales = null;
            if (upScale != null)
            {
                upScales = new float[totalRoutes];
                for (int i = 0; i < totalRoutes; i++)
                    upScales[i] = upScale[first + ids[i]];
            }
        }

        /// <summary>
        /// Run one MoE layer expert-parallel: each rank dispatches the batched
        /// <c>ggml_mul_mat_id</c> kernel over the experts it owns. Every rank's result
        /// is a PARTIAL that the caller's AllReduce completes. Returns null when this
        /// model is not on the expert-parallel path.
        /// </summary>
        /// <param name="expertInputs">Per rank: the [seqLen, expertDim] vector the experts read
        /// (the normed hidden state, or the latent projection of it).</param>
        private Tensor[] TryNemotronMoEExpertParallel(
            int layer, int seqLen, int expertDim, Tensor[] expertInputs,
            int[] selectedExperts, float[] routeWeightsAll)
        {
            if (!UsesExpertParallelMoE)
                return null;

            var upShards = _tpStackedUp[layer];
            var downShards = _tpStackedDown[layer];
            if (upShards == null || downShards == null)
                return null;

            int tp = TpDegree;
            int rankOffset = TpRankOffset;
            int nUsed = _numExpertsUsedPerLayer[layer];
            int perRank = _tpExpertsPerRank[layer];
            int intermediate = (int)upShards[0].PerExpertNe1;

            // BuildNemotronExpertParallelShards already proved these at load time, and
            // it removed the per-expert shards the fallback would need. So a mismatch
            // here is a bug, not a shape this model can serve another way - say so
            // instead of failing later with a confusing missing-weight lookup.
            if (expertDim != upShards[0].PerExpertNe0 || expertDim != downShards[0].PerExpertNe1)
            {
                throw new InvalidOperationException(
                    $"Nemotron expert-parallel MoE layer {layer}: the experts read and write "
                    + $"{upShards[0].PerExpertNe0}/{downShards[0].PerExpertNe1}-wide vectors, got {expertDim}. "
                    + "The per-expert shards were not built, so there is no fallback.");
            }

            // NVFP4 experts (Super VL): the up scale2 must multiply the up projection
            // BEFORE ReLU^2 (see InitExpertScales), so it goes to the kernel per route.
            float[] layerUpScale = HasExpertUpScale(layer) ? _layerExpertUpScale[layer] : null;

            // Per-rank dense route table, built on the calling thread: the rank
            // workers below must not race on shared scratch.
            var localExperts = new int[tp][];
            var localWeights = new float[tp][];
            var localUpScales = layerUpScale != null ? new float[tp][] : null;
            for (int r = 0; r < tp; r++)
            {
                BuildExpertParallelRoutes(selectedExperts, routeWeightsAll, seqLen, nUsed,
                    (rankOffset + r) * perRank, perRank, layerUpScale,
                    out localExperts[r], out localWeights[r], out float[] scales);
                if (localUpScales != null)
                    localUpScales[r] = scales;
            }

            var results = new Tensor[tp];
            Exception failure = null;

            long t0 = Stopwatch.GetTimestamp();
            _tpGroup.RunPerRank(r =>
            {
                var output = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, seqLen, expertDim);
                var u = upShards[r];
                var d = downShards[r];
                try
                {
                    GgmlBasicOps.MoEFFNPrefill(
                        expertInputs[r], output,
                        seqLen, expertDim, intermediate, perRank, nUsed,
                        localExperts[r], localWeights[r],
                        // ReLU^2 is a SINGLE projection: it goes in the gate slot and
                        // the up slot stays null.
                        u.Data, u.GgmlType, u.PerExpertNe0, u.PerExpertNe1, u.TotalRawBytes,
                        IntPtr.Zero, 0, 0, 0, 0,
                        d.Data, d.GgmlType, d.PerExpertNe0, d.PerExpertNe1, d.TotalRawBytes,
                        gateBias: null, upBias: null, downBias: null,
                        activation: GgmlBasicOps.MoEActivation.ReluSquared,
                        runOnCpu: MoeCpuOffloadConfig.IsLayerOnCpu(layer),
                        routeUpScales: localUpScales?[r]);
                    InvalidateTensorDeviceCache(output);
                    results[r] = output;
                }
                catch (Exception ex)
                {
                    output.Dispose();
                    failure ??= ex;
                }
            });
            _linearTicks += Stopwatch.GetTimestamp() - t0;

            if (failure != null)
            {
                for (int r = 0; r < tp; r++)
                {
                    results[r]?.Dispose();
                    results[r] = null;
                }
                throw new InvalidOperationException(
                    "Nemotron expert-parallel MoE dispatch failed and the per-expert shards "
                    + "were not built, so there is no fallback: " + failure.Message, failure);
            }

            return results;
        }
    }
}
