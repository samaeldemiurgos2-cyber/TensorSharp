// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Super VL (ggml-org) and Nano Omni mmproj files store the vision attention as one fused
// v.blk.N.attn_qkv; Bolt's (mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf) stores attn_q,
// attn_k and attn_v with their biases. The encoder read only attn_qkv, so every Bolt image
// request threw NullReferenceException (HTTP 500) and the multimodal warm-up was skipped. On a
// tiny synthetic RADIO tower these tests pin that a split file encodes exactly as its fused
// twin, and that a file missing a tensor the encoder reads fails at load, naming it.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class NemotronVisionSplitQkvTests : IDisposable
{
    private const int Patch = 4;
    private const int Image = 16;
    private const int Hidden = 8;
    private const int Heads = 2;
    private const int Ffn = 16;
    private const int Scale = 2;
    private const int ProjHidden = 12;
    private const int Projection = 6;
    private const int Blocks = 2;

    private readonly string _dir = Directory.CreateTempSubdirectory("ts-nemotron-mmproj-").FullName;
    private readonly IAllocator _allocator = new CpuAllocator(BlasEnum.DotNet);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private enum Layout { Fused, Split, SplitWithoutKBias, FusedZeroKBias }

    /// <summary>The same tower under either layout: Q, K and V (and their biases) are generated
    /// per projection and either written separately or concatenated as [Q | K | V].</summary>
    private string WriteProjector(Layout layout, string fileName, Func<string, bool> omit = null)
    {
        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>();
        void Add(string name, float scale, float offset, params int[] dims)
        {
            if (omit != null && omit(name)) return;
            tensors.Add((name, dims.Select(d => (ulong)d).ToArray(), Gen(name, dims.Aggregate(1, (a, b) => a * b), scale, offset)));
        }

        Add("v.patch_embd.weight", 0.2f, 0f, Patch, Patch, 3, Hidden);
        Add("v.patch_embd.bias", 0.1f, 0f, Hidden);
        for (int b = 0; b < Blocks; b++)
        {
            string p = $"v.blk.{b}.";
            Add(p + "ln1.weight", 0.2f, 1f, Hidden);
            Add(p + "ln1.bias", 0.1f, 0f, Hidden);
            var parts = new[] { "attn_q", "attn_k", "attn_v" };
            var weights = parts.Select(n => Gen(p + n + ".weight", Hidden * Hidden, 0.3f, 0f)).ToArray();
            var biases = parts.Select(n => layout is Layout.SplitWithoutKBias or Layout.FusedZeroKBias && n == "attn_k"
                ? new float[Hidden] : Gen(p + n + ".bias", Hidden, 0.1f, 0f)).ToArray();
            if (layout is Layout.Fused or Layout.FusedZeroKBias)
            {
                // GGUF [in, out] with in fastest: concatenating along out is concatenating the arrays.
                tensors.Add((p + "attn_qkv.weight", new ulong[] { Hidden, 3 * Hidden }, weights.SelectMany(w => w).ToArray()));
                tensors.Add((p + "attn_qkv.bias", new ulong[] { 3 * Hidden }, biases.SelectMany(w => w).ToArray()));
            }
            else
            {
                for (int i = 0; i < 3; i++)
                {
                    if (omit != null && omit(p + parts[i] + ".weight")) continue;
                    tensors.Add((p + parts[i] + ".weight", new ulong[] { Hidden, Hidden }, weights[i]));
                    if (!(layout == Layout.SplitWithoutKBias && parts[i] == "attn_k"))
                        tensors.Add((p + parts[i] + ".bias", new ulong[] { Hidden }, biases[i]));
                }
            }
            Add(p + "attn_out.weight", 0.3f, 0f, Hidden, Hidden);
            Add(p + "attn_out.bias", 0.1f, 0f, Hidden);
            Add(p + "ln2.weight", 0.2f, 1f, Hidden);
            Add(p + "ln2.bias", 0.1f, 0f, Hidden);
            Add(p + "ffn_up.weight", 0.3f, 0f, Hidden, Ffn);
            Add(p + "ffn_up.bias", 0.1f, 0f, Ffn);
            Add(p + "ffn_down.weight", 0.3f, 0f, Ffn, Hidden);
            Add(p + "ffn_down.bias", 0.1f, 0f, Hidden);
        }
        Add("mm.model.mlp.0.weight", 0.2f, 1f, Hidden * Scale * Scale);
        Add("mm.model.mlp.1.weight", 0.3f, 0f, Hidden * Scale * Scale, ProjHidden);
        Add("mm.model.mlp.3.weight", 0.3f, 0f, ProjHidden, Projection);

        var kv = new List<(string, Action<BinaryWriter>)>
        {
            ("general.architecture", Str("clip")),
            ("clip.vision.image_size", U32(Image)),
            ("clip.vision.patch_size", U32(Patch)),
            ("clip.vision.embedding_length", U32(Hidden)),
            ("clip.vision.feed_forward_length", U32(Ffn)),
            ("clip.vision.block_count", U32(Blocks)),
            ("clip.vision.attention.head_count", U32(Heads)),
            ("clip.vision.attention.layer_norm_epsilon", F32(1e-6f)),
            ("clip.vision.projection_dim", U32(Projection)),
            ("clip.vision.projector.scale_factor", U32(Scale)),
            ("clip.use_gelu", Bool(true)),
        };
        string path = Path.Combine(_dir, fileName);
        using var fs = File.Create(path);
        WriteGguf(fs, kv, tensors);
        return path;
    }

    private float[] Encode(string path)
    {
        using var encoder = new NemotronVisionEncoder(path, _allocator);
        var pixels = new float[3 * Image * Image];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = MathF.Sin(i * 0.37f) * 0.8f;
        using var output = encoder.Encode(pixels, Image, Image);
        Assert.Equal(new long[] { (Image / Patch / Scale) * (Image / Patch / Scale), Projection }, output.Sizes.ToArray());
        return output.GetElementsAsFloat((int)output.ElementCount());
    }

    [Fact]
    public void SplitAttentionProjections_EncodeExactlyAsTheFusedLayout()
    {
        float[] fused = Encode(WriteProjector(Layout.Fused, "fused.gguf"));
        float[] split = Encode(WriteProjector(Layout.Split, "split.gguf"));
        Assert.Equal(fused.Length, split.Length);
        for (int i = 0; i < fused.Length; i++)
            Assert.Equal(fused[i], split[i], 5);
        Assert.Contains(fused, v => MathF.Abs(v) > 1e-3f);
    }

    [Fact]
    public void ASplitBiasThatIsMissing_IsZero()
    {
        // The fused twin carries an explicit zero K bias in its rows.
        float[] fused = Encode(WriteProjector(Layout.FusedZeroKBias, "fused-zero-k.gguf"));
        float[] split = Encode(WriteProjector(Layout.SplitWithoutKBias, "split-no-k.gguf"));
        for (int i = 0; i < fused.Length; i++)
            Assert.Equal(fused[i], split[i], 5);
    }

    [Fact]
    public void AMissingTensor_FailsTheLoadNamingIt()
    {
        string noV = WriteProjector(Layout.Split, "no-v.gguf", omit: n => n == "v.blk.1.attn_v.weight");
        var ex = Assert.Throws<InvalidDataException>(() => new NemotronVisionEncoder(noV, _allocator));
        Assert.Contains("v.blk.1.attn_qkv.weight (or v.blk.1.attn_q/attn_k/attn_v.weight)", ex.Message);
        Assert.Contains("no-v.gguf", ex.Message);

        string noOut = WriteProjector(Layout.Fused, "no-out.gguf", omit: n => n == "v.blk.0.attn_out.weight");
        ex = Assert.Throws<InvalidDataException>(() => new NemotronVisionEncoder(noOut, _allocator));
        Assert.Contains("v.blk.0.attn_out.weight", ex.Message);
    }

    /// <summary>A real fused mmproj (Super VL, Nano Omni) still passes the load-time check.
    /// Set TS_NEMOTRON_MMPROJ to its path.</summary>
    [ModelFact("TS_NEMOTRON_MMPROJ")]
    public void ARealFusedProjector_StillLoads()
    {
        string path = Environment.GetEnvironmentVariable("TS_NEMOTRON_MMPROJ");
        using var encoder = new NemotronVisionEncoder(path, _allocator);
        Assert.True(encoder.OutputDim > 0);
    }

    // ---- GGUF writer (F32 tensors only) ----------------------------------------

    private static float[] Gen(string name, int n, float scale, float offset)
    {
        uint state = 0x811c9dc5;
        foreach (byte b in Encoding.UTF8.GetBytes(name)) { state ^= b; state = unchecked(state * 0x01000193); }
        if (state == 0) state = 1;
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            data[i] = offset + scale * ((state & 0xFFFF) / 32768f - 1f);
        }
        return data;
    }

    private static Action<BinaryWriter> U32(uint v) => w => { w.Write(4u); w.Write(v); };
    private static Action<BinaryWriter> F32(float v) => w => { w.Write(6u); w.Write(v); };
    private static Action<BinaryWriter> Bool(bool v) => w => { w.Write(7u); w.Write(v); };
    private static Action<BinaryWriter> Str(string v) => w => { w.Write(8u); WriteStr(w, v); };

    private static void WriteStr(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private const int Alignment = 32;

    private static void WriteGguf(Stream fs, List<(string Key, Action<BinaryWriter> Write)> kv,
        List<(string Name, ulong[] Dims, float[] Data)> tensors)
    {
        using var head = new MemoryStream();
        using (var w = new BinaryWriter(head, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);
            w.Write(3u);
            w.Write((ulong)tensors.Count);
            w.Write((ulong)kv.Count);
            foreach (var (key, write) in kv)
            {
                WriteStr(w, key);
                write(w);
            }
            ulong offset = 0;
            foreach (var (name, dims, data) in tensors)
            {
                WriteStr(w, name);
                w.Write((uint)dims.Length);
                foreach (ulong dim in dims) w.Write(dim);
                w.Write(0u); // F32
                w.Write(offset);
                ulong bytes = (ulong)data.Length * sizeof(float);
                offset += (bytes + Alignment - 1) / Alignment * Alignment;
            }
        }

        byte[] headBytes = head.ToArray();
        fs.Write(headBytes, 0, headBytes.Length);
        int pad = (Alignment - headBytes.Length % Alignment) % Alignment;
        fs.Write(new byte[pad], 0, pad);
        foreach (var (_, _, data) in tensors)
        {
            byte[] raw = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
            fs.Write(raw, 0, raw.Length);
            int tail = (Alignment - raw.Length % Alignment) % Alignment;
            fs.Write(new byte[tail], 0, tail);
        }
    }
}
