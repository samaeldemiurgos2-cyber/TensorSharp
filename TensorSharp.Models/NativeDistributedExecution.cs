// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>
/// Shared managed bridge for native executors that own their local device queues.
/// Native callbacks never let exceptions cross the ABI. The model must keep its
/// callback delegates alive until its native handle has been freed.
/// </summary>
internal sealed class NativeDistributedExecution
{
    private readonly ITensorParallelGroup _group;
    private readonly object _callbackLock = new();
    private float[] _scratch = Array.Empty<float>();
    private Exception _failure;
    private bool _driverRunsLastStage;

    /// <summary>Native pipelines address peers by stage. With a rotated order the
    /// driver runs the last stage; translate stages to node ids before sending.</summary>
    internal void UseStageOrder(bool driverRunsLastStage) => _driverRunsLastStage = driverRunsLastStage;

    internal NativeDistributedExecution(ITensorParallelGroup group)
    {
        ArgumentNullException.ThrowIfNull(group);
        if (group.NodeCount < 2 || group is not INestedTensorParallelGroup)
            throw new ArgumentException("Native distributed execution requires a multi-node nested group.", nameof(group));
        _group = group;
    }

    internal bool IsAborted => Volatile.Read(ref _failure) != null;
    internal Exception Failure => Volatile.Read(ref _failure);

    internal void ThrowIfAborted()
    {
        if (Failure is { } failure)
            throw new InvalidOperationException("Distributed execution failed; restart every node before continuing.", failure);
    }

    internal void Abort(Exception failure = null)
    {
        failure ??= new InvalidOperationException("The native executor aborted distributed execution.");
        if (Interlocked.CompareExchange(ref _failure, failure, null) != null) return;
        // Live weights and caches still belong to the executor. Close the
        // network now, and leave allocator disposal to normal model cleanup.
        try
        {
            if (_group is IDistributedExecutionGroup execution) execution.Abort(failure);
            else _group.Dispose();
        }
        catch { /* Retain the original failure, even when transport cleanup fails. */ }
    }

    internal void ValidateConfiguration(string configuration)
    {
        ThrowIfAborted();
        try { ValidateConfiguration(_group, configuration); }
        catch (Exception error) { Abort(error); throw; }
    }

    /// <summary>Agree on metadata and resolved numerical settings before allocating weights.</summary>
    internal static void ValidateConfiguration(ITensorParallelGroup group, string configuration)
    {
        if (group == null || group.NodeCount <= 1) return;
        ArgumentNullException.ThrowIfNull(configuration);
        if (group is not INestedTensorParallelGroup nested)
            throw new NotSupportedException("Distributed execution requires cross-node configuration agreement.");
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(configuration));
        var votes = new float[digest.Length * 8];
        for (int bit = 0; bit < votes.Length; bit++) votes[bit] = (digest[bit / 8] >> (bit % 8)) & 1;
        nested.CrossNodeAllReduce(votes, votes.Length);
        for (int bit = 0; bit < votes.Length; bit++)
            if (votes[bit] != (((digest[bit / 8] >> (bit % 8)) & 1) * group.NodeCount))
                throw new InvalidOperationException("Distributed nodes have different model metadata or execution settings. Use the same checkpoint and numerical settings on every node.");
    }

    /// <summary>A bounded startup check; deployment validation must separately hash every shard.</summary>
    internal static string HeaderIdentity(string path)
    {
        using var stream = File.OpenRead(path);
        var header = new byte[checked((int)Math.Min(stream.Length, 64 * 1024))];
        stream.ReadExactly(header);
        return Convert.ToHexString(SHA256.HashData(header));
    }

    /// <summary>
    /// Full metadata and tensor layout, including sibling shards. A large
    /// tokenizer can push layout records beyond the bounded file header check.
    /// Weight contents still require the separate deployment manifest hashes.
    /// </summary>
    internal static string MetadataIdentity(GgufFile file)
    {
        ArgumentNullException.ThrowIfNull(file);
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        void Append(byte[] value)
        {
            Span<byte> length = stackalloc byte[4];
            BinaryPrimitives.WriteInt32LittleEndian(length, value.Length);
            hash.AppendData(length);
            hash.AppendData(value);
        }
        void Text(string value) => Append(Encoding.UTF8.GetBytes(value));
        Text("TensorSharp.GGUF.metadata.v1");
        Append(JsonSerializer.SerializeToUtf8Bytes(file.Version));
        foreach (var pair in file.Metadata.OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            Text("metadata");
            Text(pair.Key);
            Text(pair.Value?.GetType().FullName ?? "null");
            Append(JsonSerializer.SerializeToUtf8Bytes(pair.Value));
        }
        foreach (var pair in file.Tensors.OrderBy(x => x.Key, StringComparer.Ordinal))
        {
            Text("tensor");
            Text(pair.Key);
            Append(JsonSerializer.SerializeToUtf8Bytes(new
            {
                pair.Value.Name, Type = (uint)pair.Value.Type, pair.Value.Shape, pair.Value.Offset
            }));
        }
        return Convert.ToHexString(hash.GetHashAndReset());
    }

    internal static string EnvironmentIdentity(IEnumerable<string> names)
    {
        var identity = new StringBuilder();
        foreach (string name in names)
        {
            string value = Environment.GetEnvironmentVariable(name);
            // Length-prefix values so separators in user-provided values cannot
            // turn distinct configurations into the same identity string.
            identity.Append('|').Append(name.Length).Append(':').Append(name)
                .Append('=').Append(value?.Length ?? -1).Append(':').Append(value);
        }
        return identity.ToString();
    }

    /// <summary>Cross-node F32 reduction after the native executor reduces its local ranks.</summary>
    internal bool AllReduce(IntPtr user, IntPtr data, int count)
    {
        lock (_callbackLock)
        {
            if (IsAborted) return false;
            try
            {
                if (count < 0) throw new InvalidOperationException("Native distributed reduction aborted.");
                if (count == 0) return true;
                if (data == IntPtr.Zero) throw new ArgumentException("Native reduction supplied a null data buffer.");
                if (_scratch.Length < count) _scratch = new float[count];
                Marshal.Copy(data, _scratch, 0, count);
                ((INestedTensorParallelGroup)_group).CrossNodeAllReduce(_scratch, count);
                Marshal.Copy(_scratch, 0, data, count);
                return true;
            }
            catch (Exception error) { Abort(error); return false; }
        }
    }

    /// <summary>
    /// Lossless opaque stage payload. Float words are a transport container, not
    /// a numeric conversion: NaN bit patterns, integer indices and tail bytes survive.
    /// A negative send argument is the native executor's explicit abort signal.
    /// </summary>
    internal unsafe bool Transfer(IntPtr user, int peer, int send, IntPtr data, int bytes)
    {
        lock (_callbackLock)
        {
            if (IsAborted) return false;
            try
            {
                if (send < 0 || bytes < 0) throw new InvalidOperationException("Native layer transfer aborted.");
                if (_group is not ILayerSplitGroup layer)
                    throw new InvalidOperationException("Native stage transfers require a layer-split group.");
                if (peer < 0 || peer >= layer.NodeCount)
                    throw new ArgumentOutOfRangeException(nameof(peer), "A stage transfer requires another valid stage.");
                peer = LayerSplitPlan.NodeOfStage(peer, layer.NodeCount, _driverRunsLastStage);
                if (peer == layer.NodeId)
                    throw new ArgumentOutOfRangeException(nameof(peer), "A stage transfer requires another valid node.");
                if (bytes != 0 && data == IntPtr.Zero) throw new ArgumentException("Native transfer supplied a null data buffer.");
                int words = checked((int)((bytes + 3L) / 4));
                if (_scratch.Length < words) _scratch = new float[words];
                fixed (float* buffer = _scratch)
                {
                    if (send != 0)
                    {
                        if (words != 0)
                        {
                            _scratch[words - 1] = 0;
                            Buffer.MemoryCopy(data.ToPointer(), buffer, (long)words * 4, bytes);
                        }
                        layer.Send(_scratch, words, peer);
                    }
                    else
                    {
                        layer.Receive(_scratch, words, peer);
                        if (bytes != 0) Buffer.MemoryCopy(buffer, data.ToPointer(), bytes, bytes);
                    }
                }
                return true;
            }
            catch (Exception error) { Abort(error); return false; }
        }
    }
}
