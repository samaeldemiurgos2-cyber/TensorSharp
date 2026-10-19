// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Shared Float32, device-resident convolution and layout operations for the LTX video VAE.</summary>
internal sealed class Ltx25VideoVaeOps
{
    private readonly DirectContext _context;
    private readonly SafetensorsFile _file;
    private readonly string _prefix;
    private const long ColumnBudgetBytes = 32L << 20;
    internal Ltx25VideoVaeOps(DirectContext context, SafetensorsFile file, string prefix)
    { _context = context; _file = file; _prefix = prefix; }

    internal Tensor NormSilu(Tensor input)
    {
        long channels = input.Sizes[3];
        using var rows = input.View(input.ElementCount() / channels, channels);
        using var normalized = DirectOps.RmsNorm(_context, rows, _context.Ones(channels), 1e-8f);
        using var activated = DirectOps.Silu(_context, normalized);
        return activated.View(input.Sizes);
    }

    /// <summary>3×3×3 convolution over channels-last [T,H,W,C] with zero spatial
    /// padding. The VAE repeats temporal edge frames; <paramref name="zeroTemporalPadding"/>
    /// selects torch Conv3d(padding=1) instead, as the latent upsampler uses. A
    /// [out,in,3,3] weight runs the same 2D convolution on every frame.</summary>
    internal Tensor Convolve(Tensor input, string name, CancellationToken cancellationToken, bool causal = false,
        bool zeroTemporalPadding = false)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string weightName = _prefix + name + ".weight";
        var shape = _file.TensorShape(weightName);
        int outputChannels = checked((int)shape[0]), channels = checked((int)shape[1]);
        int t = checked((int)input.Sizes[0]), h = checked((int)input.Sizes[1]), w = checked((int)input.Sizes[2]);
        if (input.Sizes[3] != channels) throw new InvalidOperationException("LTX VAE convolution channel mismatch.");
        if (causal && zeroTemporalPadding) throw new ArgumentException("Causal LTX convolutions repeat their first frame.");
        bool spatialOnly = shape.Length == 4;
        int inner = checked(channels * 9);
        // A still image repeats the same frame for all causal temporal taps.
        // Fold them before projection: one im2col, upload and GEMM instead of
        // three. Video and decoder paths preserve their original tap schedule.
        bool foldTemporal = causal && t == 1 && !spatialOnly;
        var taps = new Tensor[foldTemporal || spatialOnly ? 1 : 3];
        Tensor output = null;
        // MLX updates a view by creating a new parent array. Updating a band in
        // a whole clip therefore copies all T frames on every projection. Keep
        // independent frame parents, evaluate them, then concatenate once.
        var frameOutputs = _context.IsMlx && t > 1 ? new Tensor[t] : null;
        try
        {
            for (int j = 0; j < taps.Length; ++j) taps[j] = ReadTemporalWeight(weightName, foldTemporal ? -1 : j, spatialOnly ? 1 : 3);
            using var bias = _context.FromFloats(_file.ReadFloat32(_prefix + name + ".bias"), outputChannels);
            if (frameOutputs == null) output = _context.NewF32(t, h, w, outputChannels);
            int bandHeight = checked((int)Math.Max(1, Math.Min(h, ColumnBudgetBytes / (sizeof(float) * (long)w * inner))));
            for (int frame = 0; frame < t; ++frame)
            {
                Tensor parent = output;
                if (frameOutputs != null) parent = frameOutputs[frame] = _context.NewF32(1, h, w, outputChannels);
                using var targetFrame = parent.Select(0, frameOutputs == null ? frame : 0);
                using var rows = targetFrame.View((long)h * w, outputChannels);
                for (int start = 0; start < h; start += bandHeight)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int band = Math.Min(bandHeight, h - start);
                    using var target = rows.Narrow(0, (long)start * w, (long)band * w);
                    using var columns = _context.NewF32((long)band * w, inner);
                    bool accumulate = false;
                    for (int j = 0; j < taps.Length; ++j)
                    {
                        // Both VAE paths repeat temporal edges; the encoder only reads the current/past frames.
                        int sourceFrame = spatialOnly ? frame : frame + j - (causal ? 2 : 1);
                        if (zeroTemporalPadding && (sourceFrame < 0 || sourceFrame >= t)) continue;
                        using var source = input.Select(0, Math.Clamp(sourceFrame, 0, t - 1));
                        Im2Col(columns, source, channels, h, w, start, band);
                        if (_context.IsCuda)
                        {
                            if (!CudaMiniMaxH3Ops.TryMatmul(target, columns, taps[j], accumulate ? 1 : 0))
                                throw new NotSupportedException("CUDA Float32 LTX video VAE projection is unavailable.");
                        }
                        else DirectOps.CpuGemmABt(columns, taps[j], target, 1f, accumulate ? 1f : 0f);
                        accumulate = true;
                    }
                }
                DirectOps.AddBiasRows(_context, rows, bias);
                if (_context.IsMlx) MlxLtx25VaeOps.Materialize(parent);
            }
            if (frameOutputs != null)
            {
                cancellationToken.ThrowIfCancellationRequested();
                output = _context.NewF32(t, h, w, outputChannels);
                if (!MlxLtx25VaeOps.TryConcatenateFrames(output, frameOutputs))
                    throw new NotSupportedException("MLX LTX video VAE frame concatenation is unavailable.");
                MlxLtx25VaeOps.Materialize(output);
            }
            var result = output; output = null; return result;
        }
        finally
        {
            output?.Dispose();
            if (frameOutputs != null) foreach (var frame in frameOutputs) frame?.Dispose();
            foreach (var tap in taps) tap?.Dispose();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private unsafe Tensor ReadTemporalWeight(string name, int temporalTap, int kernelDepth = 3)
    {
        var info = _file.GetInfo(name);
        if (!_file.TryGetTensorDataPointer(info, out var pointer))
            throw new InvalidOperationException("Cannot map LTX video VAE convolution weights.");
        if (kernelDepth is not (1 or 3) || (kernelDepth == 1 && temporalTap != 0))
            throw new ArgumentException("LTX convolution weights have one (2D) or three temporal taps.");
        int output = checked((int)info.Shape[0]), input = checked((int)info.Shape[1]);
        int stride = 9 * kernelDepth;
        var tensor = _context.NewF32(output, checked(input * 9));
        try
        {
            // Fill the backend's owned host weight buffer directly. This avoids
            // large temporary managed arrays, a second per-element copy and
            // first-call tier-zero conversion loops during every convolution.
            // Every element is written, so a CUDA storage skips reading its
            // never-written device buffer back first (3.1 s of a 1024x1024 CUDA run).
            // Rows are independent: the conversion splits across cores, element for
            // element the same (11.1 s single-threaded on an A40 pod's 8 cores).
            if (info.Dtype is not (SafetensorDtype.BF16 or SafetensorDtype.F16 or SafetensorDtype.F32))
                throw new NotSupportedException("Unsupported LTX VAE weight dtype.");
            float* targetBase = (float*)tensor.Storage.PtrAtElementForOverwrite(0);
            long rows = (long)output * input;
            const long RowsPerBlock = 4096;
            long blocks = (rows + RowsPerBlock - 1) / RowsPerBlock;
            var dtype = info.Dtype;
            Parallel.For(0, blocks, block =>
            {
                long begin = block * RowsPerBlock, end = Math.Min(rows, begin + RowsPerBlock);
                float* target = targetBase + begin * 9;
                switch (dtype)
                {
                    case SafetensorDtype.BF16:
                    {
                        ushort* source = (ushort*)pointer + Math.Max(0, temporalTap) * 9 + begin * stride;
                        for (long row = begin; row < end; ++row, source += stride, target += 9)
                            for (int tap = 0; tap < 9; ++tap)
                                target[tap] = temporalTap < 0
                                    ? BitConverter.Int32BitsToSingle(source[tap] << 16) + BitConverter.Int32BitsToSingle(source[tap + 9] << 16) + BitConverter.Int32BitsToSingle(source[tap + 18] << 16)
                                    : BitConverter.Int32BitsToSingle(source[tap] << 16);
                        break;
                    }
                    case SafetensorDtype.F16:
                    {
                        ushort* source = (ushort*)pointer + Math.Max(0, temporalTap) * 9 + begin * stride;
                        for (long row = begin; row < end; ++row, source += stride, target += 9)
                            for (int tap = 0; tap < 9; ++tap)
                                target[tap] = temporalTap < 0
                                    ? (float)BitConverter.UInt16BitsToHalf(source[tap]) + (float)BitConverter.UInt16BitsToHalf(source[tap + 9]) + (float)BitConverter.UInt16BitsToHalf(source[tap + 18])
                                    : (float)BitConverter.UInt16BitsToHalf(source[tap]);
                        break;
                    }
                    default:
                    {
                        float* source = (float*)pointer + Math.Max(0, temporalTap) * 9 + begin * stride;
                        for (long row = begin; row < end; ++row, source += stride, target += 9)
                            if (temporalTap < 0)
                                for (int tap = 0; tap < 9; ++tap) target[tap] = source[tap] + source[tap + 9] + source[tap + 18];
                            else Buffer.MemoryCopy(source, target, 9 * sizeof(float), 9 * sizeof(float));
                        break;
                    }
                }
            });
            return tensor;
        }
        catch { tensor.Dispose(); throw; }
    }

    private unsafe void Im2Col(Tensor output, Tensor source, int channels, int height, int width, int start, int rows)
    {
        if (_context.IsCuda)
        {
            if (!CudaWanOps.TryIm2Col(output, source, channels, height, width, 3, 3, height, width, 1, 1, 1, 1, start, rows))
                throw new NotSupportedException("CUDA LTX video VAE im2col is unavailable.");
        }
        else if (_context.IsMlx)
        {
            if (!MlxLtx25VaeOps.TryIm2Col(output, source, channels, height, width, 3, start, rows))
                throw new NotSupportedException("MLX LTX video VAE im2col is unavailable.");
        }
        else
        {
            float* input = (float*)CpuNativeHelpers.GetBufferStart(source);
            float* target = (float*)CpuNativeHelpers.GetBufferStart(output);
            int inner = checked(channels * 9);
            Parallel.For(0, checked(rows * width), row =>
            {
                int y = start + row / width, x = row % width;
                for (int c = 0; c < channels; ++c)
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx)
                        {
                            int sy = y - 1 + ky, sx = x - 1 + kx;
                            target[(long)row * inner + c * 9 + ky * 3 + kx] =
                                sy >= 0 && sy < height && sx >= 0 && sx < width
                                    ? input[((long)sy * width + sx) * channels + c] : 0;
                        }
            });
        }
    }

    internal Tensor Contiguous(Tensor source)
    {
        if (!_context.IsCuda || source.IsContiguous()) return Ops.NewContiguous(source);
        var result = _context.NewF32(source.Sizes);
        try
        {
            if (!CudaMiniMaxH3Ops.TryCopyStrided(result, source))
                throw new NotSupportedException("LTX CUDA VAE requires the native Float32 four-axis strided copy kernel; CPU layout fallback is disabled.");
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    internal float[] Planar(Tensor tensor)
    {
        using var permuted = tensor.Permute(3, 0, 1, 2);
        using var contiguous = Contiguous(permuted);
        return contiguous.GetElementsAsFloat(checked((int)contiguous.ElementCount()));
    }
}
