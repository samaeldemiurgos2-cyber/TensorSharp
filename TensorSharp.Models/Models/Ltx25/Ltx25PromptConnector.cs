using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.Cuda;
using TensorSharp.Models.Direct;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>The quantized checkpoint's noncausal, register-filled embedding connector.</summary>
internal sealed class Ltx25PromptConnector : IDisposable
{
    private readonly GgufFile _file;
    private readonly DirectContext _context;
    private readonly string _prefix;
    private readonly Dictionary<string, DirectLinear> _linears = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Tensor> _weights = new(StringComparer.Ordinal);
    private readonly CudaLtx25ProjectionSession _projection;
    private readonly CudaLtx25AttentionSession _attention;
    private readonly float[] _registers;
    private readonly int _heads, _layers;
    private float[] _cosine, _sine;   // RoPE per (token, frequency), computed once per token count
    private bool _warm;              // weights read since construction or the last release
    internal int Dimension { get; }
    internal string CudaModulePath => _projection?.ModulePath;
    internal long CudaProjections => _projection?.ExecutedF16Projections ?? 0;
    internal long CudaAttentions => _attention?.ExecutedAttentions ?? 0;

    internal Ltx25PromptConnector(string path, IAllocator allocator, string prefix)
    {
        _file = new(path); _context = new(allocator); _prefix = prefix;
        try
        {
            var register = _file.Tensors[prefix + ".learnable_registers"];
            Dimension = checked((int)register.Shape[0]);
            _registers = DirectOps.DequantTensor(_file, register.Name);
            _heads = checked((int)_file.Tensors[prefix + ".transformer_1d_blocks.0.attn1.to_gate_logits.bias"].NumElements);
            while (_file.Tensors.ContainsKey(prefix + ".transformer_1d_blocks." + _layers + ".attn1.to_q.weight")) ++_layers;
            if (Dimension < 1 || _heads < 1 || Dimension % (2 * _heads) != 0 || _layers < 1 || _registers.Length % Dimension != 0)
                throw new InvalidDataException("Invalid LTX embedding connector geometry.");
            if (allocator is CudaAllocator cuda) { _projection = new(cuda); _attention = new(cuda); }
        }
        catch { Dispose(); throw; }
    }
    private DirectLinear Linear(string name)
    {
        if (_linears.TryGetValue(name, out var linear)) return linear;
        string full = _prefix + "." + name;
        linear = DirectLinear.FromGguf(_context, _file, full + ".weight", full + ".bias", ltxProjection: _projection);
        _linears.Add(name, linear); return linear;
    }
    private Tensor Weight(string name)
    {
        if (_weights.TryGetValue(name, out var weight)) return weight;
        weight = _context.Own(_context.FromFloats(DirectOps.DequantTensor(_file, _prefix + "." + name), Dimension));
        _weights.Add(name, weight); return weight;
    }
    internal float[] Forward(float[] valid, int validTokens, int tokens, CancellationToken cancellationToken)
    {
        int registers = _registers.Length / Dimension;
        if (validTokens < 1 || validTokens > tokens || valid.Length != checked(validTokens * Dimension) || tokens % registers != 0)
            throw new ArgumentException("Invalid LTX connector token/register layout.");
        var input = new float[checked(tokens * Dimension)]; Array.Copy(valid, input, valid.Length);
        for (int token = validTokens; token < tokens; ++token) Array.Copy(_registers, (token % registers) * Dimension, input, token * Dimension, Dimension);
        using var readAhead = _warm ? null : StartReadAhead();
        Tensor x = _context.FromFloats(input, tokens, Dimension);
        try
        {
            for (int layer = 0; layer < _layers; ++layer)
            {
                cancellationToken.ThrowIfCancellationRequested();
                readAhead?.Enter(layer + 1);
                string p = "transformer_1d_blocks." + layer;
                using var normalized = Ltx25PromptMath.Norm(_context, x, null, 1e-6f, false);
                using var rawq = Linear(p + ".attn1.to_q").Forward(normalized);
                using var rawk = Linear(p + ".attn1.to_k").Forward(normalized);
                using var v = Linear(p + ".attn1.to_v").Forward(normalized);
                using var q = Ltx25PromptMath.Norm(_context, rawq, Weight(p + ".attn1.q_norm.weight"), 1e-6f, false);
                using var k = Ltx25PromptMath.Norm(_context, rawk, Weight(p + ".attn1.k_norm.weight"), 1e-6f, false);
                Rope(q); Rope(k);
                Tensor attended = _context.NewF32(tokens, Dimension);
                if (_attention == null || !_attention.TryAttention(attended, q, k, v, _heads, Dimension / _heads, 1f / MathF.Sqrt(Dimension / _heads)))
                { attended.Dispose(); attended = DirectOps.Attention(_context, q, k, v, _heads, Dimension / _heads, 1f / MathF.Sqrt(Dimension / _heads)); }
                using (attended)
                {
                    using var logits = Linear(p + ".attn1.to_gate_logits").Forward(normalized);
                    var gates = logits.GetElementsAsFloat(checked(tokens * _heads));
                    var values = attended.GetElementsAsFloat(checked(tokens * Dimension));
                    int head = Dimension / _heads;
                    for (int row = 0; row < gates.Length; ++row)
                    { float gate = 2 / (1 + MathF.Exp(-gates[row])); for (int d = 0; d < head; ++d) values[row * head + d] *= gate; }
                    attended.SetElementsAsFloat(values);
                    using var projected = Linear(p + ".attn1.to_out.0").Forward(attended);
                    Ops.Add(x, x, projected);
                }
                using var ffInput = Ltx25PromptMath.Norm(_context, x, null, 1e-6f, false);
                using var up = Linear(p + ".ff.net.0.proj").Forward(ffInput);
                using var activated = DirectOps.Gelu(_context, up);
                using var down = Linear(p + ".ff.net.2").Forward(activated);
                Ops.Add(x, x, down);
            }
            using var output = Ltx25PromptMath.Norm(_context, x, null, 1e-6f, false);
            _warm = true;
            return output.GetElementsAsFloat(checked((int)output.ElementCount()));
        }
        finally { x.Dispose(); }
    }

    // A new prompt reads this connector's weights from the denoiser checkpoint
    // before the denoiser's own upload read-ahead starts; cold, through page faults
    // (on a network volume most of 12 s of connector time). Stage 0 holds the tensors
    // outside the blocks, then one stage per block in forward order.
    private MappedFileReadAhead StartReadAhead()
    {
        if (_file.IsSplit) return null;
        var stages = new List<(long Offset, long Length)>[_layers + 1];
        for (int i = 0; i < stages.Length; ++i) stages[i] = new();
        string blocks = _prefix + ".transformer_1d_blocks.";
        foreach (var (name, info) in _file.Tensors)
        {
            if (!name.StartsWith(_prefix + ".", StringComparison.Ordinal)) continue;
            int stage = 0;
            if (name.StartsWith(blocks, StringComparison.Ordinal))
            {
                int end = name.IndexOf('.', blocks.Length);
                if (end < 0 || !int.TryParse(name.AsSpan(blocks.Length, end - blocks.Length), out int layer) ||
                    (uint)layer >= (uint)_layers) continue;
                stage = layer + 1;
            }
            stages[stage].Add((checked(_file.DataOffset + (long)info.Offset), _file.GetTensorByteCount(info)));
        }
        return MappedFileReadAhead.TryStart(_file.FilePaths[0], stages.Select(stage => stage.OrderBy(range => range.Offset)));
    }

    // The rotation depends only on the token and the frequency index, so every layer
    // and both Q and K share one table, built with the same expressions as before.
    private void Rope(Tensor input)
    {
        var values = input.GetElementsAsFloat(checked((int)input.ElementCount()));
        int half = Dimension / _heads / 2, count = Dimension / 2, tokens = checked((int)input.Sizes[0]);
        if (_cosine == null || _cosine.Length != checked(tokens * count))
        {
            var frequencies = new float[count];
            for (int i = 0; i < count; ++i) frequencies[i] = (float)(Math.Pow(10000, (double)i / (count - 1)) * (Math.PI / 2));
            var cosineTable = new float[tokens * count];
            var sineTable = new float[tokens * count];
            for (int token = 0; token < tokens; ++token)
            for (int i = 0; i < count; ++i)
            {
                float angle = (token / 4096f * 2 - 1) * frequencies[i];
                (sineTable[token * count + i], cosineTable[token * count + i]) = MathF.SinCos(angle);
            }
            _cosine = cosineTable; _sine = sineTable;
        }
        float[] cosineValues = _cosine, sineValues = _sine;
        for (int token = 0; token < tokens; ++token)
        for (int h = 0; h < _heads; ++h)
        for (int d = 0; d < half; ++d)
        {
            int table = token * count + h * half + d;
            float cosine = cosineValues[table], sine = sineValues[table];
            int a = token * Dimension + h * half * 2 + d, b = a + half;
            float first = values[a], second = values[b];
            values[a] = first * cosine - second * sine; values[b] = second * cosine + first * sine;
        }
        input.SetElementsAsFloat(values);
    }
    internal void ReleaseDeviceResidency() { _context.ReleaseDeviceResidency(); _projection?.ReleaseWorkspace(); _attention?.ReleaseWorkspace(); _warm = false; }
    public void Dispose() { _attention?.Dispose(); _projection?.Dispose(); _context.Dispose(); _file.Dispose(); }
}
