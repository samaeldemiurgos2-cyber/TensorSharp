// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public partial class MuseSparkModel
{
    private SparkExpertPlacement _sparkTpPlacement;
    private readonly long[] _sparkTpPlanningFreeBytes;

    internal IReadOnlyList<int> TensorParallelExpertCounts => _sparkTpPlacement?.Counts;
    internal IReadOnlyList<long> TensorParallelFixedWeightBytes => _sparkTpPlacement?.FixedBytes;
    internal long TensorParallelBytesPerExpert => _sparkTpPlacement?.BytesPerExpert ?? 0;

    internal sealed class SparkExpertPlacement
    {
        internal IReadOnlyList<int> Counts { get; }
        internal IReadOnlyList<long> WeightBytes { get; }
        internal IReadOnlyList<long> FixedBytes { get; }
        internal long BytesPerExpert { get; }
        internal string Policy { get; }

        internal SparkExpertPlacement(int[] counts, long[] fixedBytes, long bytesPerExpert, string policy)
        {
            var weights = new long[counts.Length];
            for (int rank = 0; rank < counts.Length; rank++)
                weights[rank] = checked(fixedBytes[rank] + bytesPerExpert * counts[rank]);
            Counts = Array.AsReadOnly((int[])counts.Clone());
            WeightBytes = Array.AsReadOnly(weights);
            FixedBytes = Array.AsReadOnly((long[])fixedBytes.Clone());
            BytesPerExpert = bytesPerExpert;
            Policy = policy;
        }
    }

    // Every expert ID has the same byte cost within a layer. Sum that cost
    // across GPU expert layers; CPU-offloaded layers never change ownership.
    // The returned immutable plan is consumed by preflight AND materialization.
    internal static SparkExpertPlacement PlanSparkExperts(int experts, long[] fixedBytes,
        long bytesPerExpert, long[] freeBytes, long reserve)
    {
        if (fixedBytes == null || freeBytes == null || fixedBytes.Length == 0 ||
            fixedBytes.Length != freeBytes.Length || experts < fixedBytes.Length ||
            bytesPerExpert < 0 || reserve < 0)
            throw new ArgumentException("Invalid Muse Spark expert placement inputs.");
        int degree = fixedBytes.Length;
        var uniform = new int[degree];
        bool unknown = false;
        for (int rank = 0; rank < degree; rank++)
        {
            if (fixedBytes[rank] < 0 || freeBytes[rank] < -1)
                throw new ArgumentException("Invalid Muse Spark expert placement bytes.");
            uniform[rank] = SparkPartition(experts, degree, rank).Count;
            unknown |= freeBytes[rank] < 0;
        }
        if (unknown || bytesPerExpert == 0)
            return new SparkExpertPlacement(uniform, fixedBytes, bytesPerExpert,
                unknown ? "uniform (device memory unavailable)" : "uniform (no partitioned expert bytes)");

        var capacity = new int[degree];
        bool uniformFits = true;
        long totalCapacity = 0;
        for (int rank = 0; rank < degree; rank++)
        {
            long available = freeBytes[rank] - fixedBytes[rank];
            if (available >= reserve)
                capacity[rank] = (int)Math.Min(experts, (available - reserve) / bytesPerExpert);
            totalCapacity += capacity[rank];
            uniformFits &= capacity[rank] >= uniform[rank];
        }
        if (uniformFits)
            return new SparkExpertPlacement(uniform, fixedBytes, bytesPerExpert, "uniform (fits measured capacity)");
        if (Array.Exists(capacity, value => value == 0) || totalCapacity < experts)
            throw new ModelLoadRefusedException(
                $"Muse Spark TP cannot fit {experts} whole experts per GPU layer with the requested CPU offload: " +
                $"per-rank capacities [{string.Join(',', capacity)}] after fixed weights and {reserve / MiB} MiB reserve. " +
                "Every rank requires at least one expert. Increase --n-cpu-moe, use more/larger GPUs, or choose a smaller quantization. " +
                "The requested CPU offload count was not changed.");

        // Uniform fits were retained above, including tiny models. Otherwise
        // give each next expert to the rank with the most remaining bytes,
        // maximizing the smallest remaining margin for equally sized items.
        // This avoids filling rank 0 to its capacity while leaving other ranks
        // empty, which would discard useful headroom for the vision encoder.
        // Stable rank order resolves exact ties.
        var counts = new int[degree];
        Array.Fill(counts, 1);
        for (int remaining = experts - degree; remaining > 0; remaining--)
        {
            int best = -1;
            long bestHeadroom = long.MinValue;
            for (int rank = 0; rank < degree; rank++)
            {
                if (counts[rank] >= capacity[rank]) continue;
                long headroom = freeBytes[rank] - fixedBytes[rank] - reserve - bytesPerExpert * counts[rank];
                if (best < 0 || headroom > bestHeadroom)
                { best = rank; bestHeadroom = headroom; }
            }
            if (best < 0) throw new InvalidOperationException("Expert capacity changed during immutable planning.");
            counts[best]++;
        }
        return new SparkExpertPlacement(counts, fixedBytes, bytesPerExpert, "capacity-balanced");
    }
}
