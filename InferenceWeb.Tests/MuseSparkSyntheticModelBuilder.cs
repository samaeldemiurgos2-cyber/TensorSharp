// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;

namespace InferenceWeb.Tests;

/// <summary>Tiny deterministic checkpoint and an independent scalar decoder oracle.</summary>
internal sealed class MuseSparkSyntheticModelBuilder
{
    internal const int Hidden = 32, HeadDim = 8, Heads = 4, KvHeads = 2;
    internal const int Latent = 16, Ffn = 24, Experts = 4, Used = 2, Layers = 2, Vocab = 256, Window = 3;
    internal const float Eps = 1e-5f, PostEps = 2e-6f, RopeBase = 7777, LogitScale = 0.7f, Softcap = 1.3f;
    internal readonly Dictionary<string, (ulong[] Shape, float[] Data)> Weights = new();

    private readonly int _heads, _kvHeads, _headDim, _experts, _used, _latent, _ffn, _contextLength, _slidingLayer;

    /// <param name="contextLength">The GGUF's advertised context.</param>
    /// <param name="slidingFirst">Layer 0 is the sliding-window layer and layer 1 the global one
    /// (the default), or the other way round.</param>
    internal MuseSparkSyntheticModelBuilder(int heads = Heads, int kvHeads = KvHeads, int experts = Experts,
        int used = Used, int headDim = HeadDim, int latent = Latent, int ffn = Ffn, int contextLength = 64,
        bool slidingFirst = true)
    {
        _heads = heads; _kvHeads = kvHeads; _experts = experts; _used = used;
        _headDim = headDim; _latent = latent; _ffn = ffn;
        _contextLength = contextLength; _slidingLayer = slidingFirst ? 0 : 1;
        Add("token_embd.weight", 0.8f, 0, Hidden, Vocab);
        Add("output_norm.weight", 0.2f, 1, Hidden);
        Add("output.weight", 0.1f, 0, Hidden, Vocab);
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            foreach (string n in new[] { "attn_norm", "ffn_norm", "post_ffw_norm" }) Add(p + n + ".weight", 0.4f, 1, Hidden);
            foreach (string n in new[] { "attn_resid_alpha", "ffn_resid_alpha" }) Add(p + n + ".weight", 0.2f, 0.7f, Hidden);
            foreach (string n in new[] { "attn_resid_beta", "ffn_resid_beta" }) Add(p + n + ".weight", 0.2f, 0.5f, Hidden);
            Add(p + "attn_q.weight", 0.12f, 0, Hidden, _heads * _headDim);
            Add(p + "attn_k.weight", 0.12f, 0, Hidden, _kvHeads * _headDim);
            Add(p + "attn_v.weight", 0.12f, 0, Hidden, _kvHeads * _headDim);
            Add(p + "attn_gate.weight", 0.12f, 0, Hidden, _heads * _headDim);
            Add(p + "attn_output.weight", 0.12f, 0, _heads * _headDim, Hidden);
            Add(p + "attn_q_norm.weight", 0.2f, 0.7f, _headDim);
            Add(p + "attn_k_norm.weight", 0.2f, 1, _headDim);
            Add(p + "ffn_latent_down.weight", 0.12f, 0, Hidden, _latent);
            Add(p + "ffn_latent_up.weight", 0.12f, 0, _latent, Hidden);
            Add(p + "ffn_pre_exp_norm.weight", 0.4f, 1, _latent);
            Add(p + "ffn_post_exp_norm.weight", 0.8f, 1, _latent);
            Add(p + "ffn_gate_inp.weight", 0.2f, 0, Hidden, _experts);
            Add(p + "exp_probs_b.bias", 0.2f, 0, _experts);
            Add(p + "ffn_gate_exps.weight", 0.2f, 0, _latent, _ffn, _experts);
            Add(p + "ffn_up_exps.weight", 0.2f, 0, _latent, _ffn, _experts);
            Add(p + "ffn_down_exps.weight", 0.2f, 0, _ffn, _latent, _experts);
        }
    }

    private void Add(string name, float scale, float offset, params int[] shape)
    {
        uint state = 2166136261;
        foreach (char ch in name) state = unchecked((state ^ ch) * 16777619);
        var data = new float[shape.Aggregate(1, (n, x) => n * x)];
        for (int i = 0; i < data.Length; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            data[i] = offset + scale * ((state & 65535) / 32768f - 1);
        }
        Weights.Add(name, (shape.Select(x => (ulong)x).ToArray(), data));
    }

    internal string Write(string path, bool halfMatrices = false)
        => WriteFile(path, halfMatrices, CreateMetadata(), Weights);

    internal string WriteMixedHalf(string path, Func<string, bool> halfWeight)
        => WriteFile(path, false, CreateMetadata(), Weights, halfWeight);

    internal string[] WriteSplit(string directory, bool halfMatrices = false)
    {
        string[] paths = [Path.Combine(directory, "spark-00001-of-00002.gguf"),
            Path.Combine(directory, "spark-00002-of-00002.gguf")];
        var firstMetadata = CreateMetadata();
        firstMetadata["split.no"] = (ushort)0;
        firstMetadata["split.count"] = (ushort)2;
        firstMetadata["split.tensors.count"] = Weights.Count;
        var lastMetadata = new Dictionary<string, object>
        {
            ["split.no"] = (ushort)1,
            ["split.count"] = (ushort)2,
            ["split.tensors.count"] = Weights.Count,
        };
        // Published Muse Spark splits put all architecture/tokenizer metadata in
        // the tensor-free first shard and only split metadata in later shards.
        WriteFile(paths[0], halfMatrices, firstMetadata, new Dictionary<string, (ulong[] Shape, float[] Data)>());
        WriteFile(paths[1], halfMatrices, lastMetadata, Weights);
        return paths;
    }

    internal Dictionary<string, object> CreateMetadata()
    {
        return new Dictionary<string, object>
        {
            ["general.architecture"] = "muse-spark",
            ["muse-spark.block_count"] = (uint)Layers,
            ["muse-spark.context_length"] = (uint)_contextLength,
            ["muse-spark.embedding_length"] = (uint)Hidden,
            ["muse-spark.feed_forward_length"] = (uint)_ffn,
            ["muse-spark.attention.head_count"] = (uint)_heads,
            ["muse-spark.attention.head_count_kv"] = (uint)_kvHeads,
            ["muse-spark.attention.key_length"] = (uint)_headDim,
            ["muse-spark.attention.value_length"] = (uint)_headDim,
            ["muse-spark.attention.layer_norm_rms_epsilon"] = Eps,
            ["muse-spark.post_norm_epsilon"] = PostEps,
            ["muse-spark.rope.freq_base_swa"] = RopeBase,
            ["muse-spark.rope.freq_base"] = RopeBase,
            ["muse-spark.rope.dimension_count"] = (uint)_headDim,
            ["muse-spark.attention.sliding_window"] = (uint)Window,
            ["muse-spark.attention.sliding_window_pattern"] = new[] { _slidingLayer == 0, _slidingLayer == 1 },
            ["muse-spark.moe_latent_size"] = (uint)_latent,
            ["muse-spark.expert_feed_forward_length"] = (uint)_ffn,
            ["muse-spark.expert_count"] = (uint)_experts,
            ["muse-spark.expert_used_count"] = (uint)_used,
            ["muse-spark.expert_gating_func"] = 2u,
            ["muse-spark.expert_weights_norm"] = true,
            ["muse-spark.expert_weights_scale"] = 1.7f,
            ["muse-spark.logit_scale"] = LogitScale,
            ["muse-spark.final_logit_softcapping"] = Softcap,
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.pre"] = "llama4",
            ["tokenizer.ggml.tokens"] = Enumerable.Range(0, Vocab).Select(i => char.ConvertFromUtf32(256 + i)).ToArray(),
            ["tokenizer.ggml.token_type"] = Enumerable.Repeat(1, Vocab).ToArray(),
            ["tokenizer.ggml.merges"] = Array.Empty<string>(),
            ["tokenizer.ggml.bos_token_id"] = 1u,
            ["tokenizer.ggml.eos_token_id"] = 0u,
            ["tokenizer.ggml.add_bos_token"] = false,
        };
    }

    private static string WriteFile(string path, bool halfMatrices, IReadOnlyDictionary<string, object> metadata,
        IReadOnlyDictionary<string, (ulong[] Shape, float[] Data)> weights, Func<string, bool> halfWeight = null)
    {
        using var file = File.Create(path);
        using var w = new BinaryWriter(file, Encoding.UTF8);
        w.Write(0x46554747u); w.Write(3u); w.Write((ulong)weights.Count); w.Write((ulong)metadata.Count);
        foreach (var (key, value) in metadata) { WriteString(w, key); WriteValue(w, value); }
        ulong offset = 0;
        foreach (var (key, (shape, data)) in weights)
        {
            WriteString(w, key); w.Write((uint)shape.Length);
            foreach (ulong d in shape) w.Write(d);
            bool half = shape.Length > 1 && (halfMatrices || halfWeight?.Invoke(key) == true);
            w.Write(half ? 1u : 0u); w.Write(offset);
            offset += ((ulong)data.Length * (half ? 2u : 4u) + 31) / 32 * 32;
        }
        if (weights.Count == 0) return path; // Metadata-only shards have no tensor alignment padding.
        while (file.Position % 32 != 0) w.Write((byte)0);
        foreach (var (name, (shape, data)) in weights)
        {
            bool half = shape.Length > 1 && (halfMatrices || halfWeight?.Invoke(name) == true);
            foreach (float x in data)
                if (half) w.Write(BitConverter.HalfToUInt16Bits((System.Half)x)); else w.Write(x);
            while (file.Position % 32 != 0) w.Write((byte)0);
        }
        return path;
    }

    private static void WriteString(BinaryWriter w, string value) { byte[] b = Encoding.UTF8.GetBytes(value); w.Write((ulong)b.Length); w.Write(b); }
    private static void WriteValue(BinaryWriter w, object value)
    {
        switch (value)
        {
            case ushort x: w.Write(2u); w.Write(x); break;
            case uint x: w.Write(4u); w.Write(x); break;
            case int x: w.Write(5u); w.Write(x); break;
            case float x: w.Write(6u); w.Write(x); break;
            case bool x: w.Write(7u); w.Write(x); break;
            case string x: w.Write(8u); WriteString(w, x); break;
            case bool[] x: w.Write(9u); w.Write(7u); w.Write((ulong)x.Length); foreach (bool a in x) w.Write(a); break;
            case int[] x: w.Write(9u); w.Write(5u); w.Write((ulong)x.Length); foreach (int a in x) w.Write(a); break;
            case string[] x: w.Write(9u); w.Write(8u); w.Write((ulong)x.Length); foreach (string a in x) WriteString(w, a); break;
            default: throw new ArgumentException(nameof(value));
        }
    }

    // This oracle uses double intermediates, dense arrays and no TensorSharp ops.
    // It recomputes the complete sequence and never shares the runtime's KV cache.
    internal float[] Reference(int[] tokens, bool halfMatrices = false, Dictionary<int, float[]> replacements = null)
    {
        double[] W(string key) => Weights[key].Data.Select(x => halfMatrices && Weights[key].Shape.Length > 1
            ? (double)(System.Half)x : x).ToArray();
        double[] Linear(double[] x, string key, int expert = 0)
        {
            var (shape, _) = Weights[key];
            int input = (int)shape[0], output = (int)shape[1];
            double[] weight = W(key), result = new double[output];
            for (int o = 0; o < output; o++)
            for (int i = 0; i < input; i++) result[o] += x[i] * weight[(expert * output + o) * input + i];
            return result;
        }
        double[] Norm(double[] x, float eps, string key = null)
        {
            double inverse = 1 / Math.Sqrt(x.Select(v => v * v).Average() + eps);
            double[] weight = key == null ? Enumerable.Repeat(1d, x.Length).ToArray() : W(key);
            return x.Select((v, i) => v * inverse * weight[i]).ToArray();
        }
        double[] Multiply(double[] x, string key) { var w = W(key); return x.Select((v, i) => v * w[i]).ToArray(); }
        double[] Residual(double[] x, double[] y, string p)
        { var a = W(p + "alpha.weight"); var b = W(p + "beta.weight"); return x.Select((v, i) => v * a[i] + y[i] * b[i]).ToArray(); }
        static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));
        var embedding = W("token_embd.weight");
        var hidden = tokens.Select((token, index) => Norm(replacements != null && replacements.TryGetValue(index, out var replacement)
            ? replacement.Select(x => (double)x).ToArray() : embedding.AsSpan(token * Hidden, Hidden).ToArray(), Eps)).ToArray();
        for (int layer = 0; layer < Layers; layer++)
        {
            string p = $"blk.{layer}.";
            var norms = hidden.Select(x => Norm(x, Eps, p + "attn_norm.weight")).ToArray();
            var queries = norms.Select(x => Linear(x, p + "attn_q.weight")).ToArray();
            var keys = norms.Select(x => Linear(x, p + "attn_k.weight")).ToArray();
            var values = norms.Select(x => Linear(x, p + "attn_v.weight")).ToArray();
            void NormalizeAndRotate(double[][] tensor, int heads, string norm)
            {
                for (int t = 0; t < tokens.Length; t++)
                for (int h = 0; h < heads; h++)
                {
                    double[] head = Norm(tensor[t].AsSpan(h * _headDim, _headDim).ToArray(), Eps, norm);
                    if (layer == _slidingLayer)
                        for (int d = 0; d < _headDim; d += 2)
                        {
                            double angle = t / Math.Pow(RopeBase, (double)d / _headDim), a = head[d], b = head[d + 1];
                            head[d] = a * Math.Cos(angle) - b * Math.Sin(angle);
                            head[d + 1] = a * Math.Sin(angle) + b * Math.Cos(angle);
                        }
                    head.CopyTo(tensor[t], h * _headDim);
                }
            }
            NormalizeAndRotate(queries, _heads, p + "attn_q_norm.weight");
            NormalizeAndRotate(keys, _kvHeads, p + "attn_k_norm.weight");
            for (int t = 0; t < tokens.Length; t++)
            {
                var attention = new double[_heads * _headDim];
                for (int h = 0; h < _heads; h++)
                {
                    int kv = h / (_heads / _kvHeads), first = layer == _slidingLayer ? Math.Max(0, t - Window + 1) : 0;
                    var scores = new double[t + 1];
                    for (int s = first; s <= t; s++)
                    for (int d = 0; d < _headDim; d++) scores[s] += queries[t][h * _headDim + d] * keys[s][kv * _headDim + d] / Math.Sqrt(_headDim);
                    double max = scores.Skip(first).Max();
                    for (int s = first; s <= t; s++) scores[s] = Math.Exp(scores[s] - max);
                    double sum = scores.Skip(first).Sum();
                    var head = new double[_headDim];
                    for (int s = first; s <= t; s++)
                    for (int d = 0; d < _headDim; d++) head[d] += scores[s] / sum * values[s][kv * _headDim + d];
                    Norm(head, Eps).CopyTo(attention, h * _headDim);
                }
                var gate = Linear(norms[t], p + "attn_gate.weight");
                attention = attention.Select((v, i) => v * Sigmoid(gate[i])).ToArray();
                var branch = Norm(Linear(attention, p + "attn_output.weight"), PostEps);
                var ffn = Residual(hidden[t], branch, p + "attn_resid_");
                var full = Norm(ffn, Eps, p + "ffn_norm.weight");
                var latent = Norm(Linear(full, p + "ffn_latent_down.weight"), Eps, p + "ffn_pre_exp_norm.weight");
                var probs = Linear(full, p + "ffn_gate_inp.weight").Select(Sigmoid).ToArray();
                var bias = W(p + "exp_probs_b.bias");
                var selected = Enumerable.Range(0, _experts).OrderByDescending(e => probs[e] + bias[e]).Take(_used).ToArray();
                double denominator = Math.Max(selected.Sum(e => probs[e]), 6.103515625e-5);
                var mixture = new double[_latent];
                foreach (int expert in selected)
                {
                    var g = Linear(latent, p + "ffn_gate_exps.weight", expert);
                    var u = Linear(latent, p + "ffn_up_exps.weight", expert);
                    var activated = g.Select((v, i) => v * Sigmoid(v) * u[i]).ToArray();
                    var down = Linear(activated, p + "ffn_down_exps.weight", expert);
                    var normalized = Norm(Multiply(down, p + "ffn_post_exp_norm.weight"), PostEps);
                    for (int d = 0; d < _latent; d++) mixture[d] += normalized[d] * probs[expert] / denominator * 1.7;
                }
                var projected = Linear(mixture, p + "ffn_latent_up.weight");
                var post = Norm(Multiply(projected, p + "post_ffw_norm.weight"), PostEps);
                hidden[t] = Residual(ffn, post, p + "ffn_resid_");
            }
        }
        return Linear(Norm(hidden[^1], Eps, "output_norm.weight"), "output.weight")
            .Select(x => (float)(Softcap * Math.Tanh(x * LogitScale / Softcap))).ToArray();
    }
}
