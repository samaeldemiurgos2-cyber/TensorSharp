// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// CpuDecodeAttentionBench: per-call latency of ModelBase's pure-C# GQA decode attention, the
// host path every model without a fused device attention falls back to (the cpu backend, the
// direct cuda backend's Nemotron decode, --tp host attention).
//
//   CpuDecodeAttentionBench [--shape heads:kvHeads:headDim:keys:f16|f32]... [--reps N] [--passes N]
//                           [--warmup-seconds S] [--label NAME] [--scratch-modes default,stack,rent]
//
// Every shape is warmed up, then timed call by call; a pass walks all shapes, and the reported
// numbers are over every pass. Each line also prints an FNV-1a hash of the output bits, so two
// builds that should compute the same thing can be checked for bit-identical results.
//
// The protected entry points are reached through a stub subclass of ModelBase that is never
// constructed (GetUninitializedObject): the attention kernels touch no instance state, and that
// keeps this source buildable against any tree. For an A/B, build it twice -
// -p:TensorSharpRoot=<baseline tree>/ for the baseline - and alternate the two binaries (ab.py).
//
// --scratch-modes times each shape once per mode, interleaved within every pass of one process:
// "default" is the build as it is, "stack" forces every score scratch onto the stack and "rent"
// forces every one into the pool (CpuScratch.OverrideStackFloatLimit, reached by reflection, so
// those two need a build that has it). Rows of a non-default mode print as shape@mode.
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal static unsafe class Program
{
    private static int Main(string[] args)
    {
        // Bind Core's CpuParallel to the Models worker pool exactly as a model load would.
        RuntimeHelpers.RunModuleConstructor(typeof(ModelBase).Module.ModuleHandle);

        var shapes = new List<Shape>();
        int reps = 400, passes = 3;
        double warmupSeconds = 0.5;
        string label = "run";
        string[] modes = { "default" };
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--shape": shapes.Add(Shape.Parse(args[++i])); break;
                case "--reps": reps = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--passes": passes = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--warmup-seconds": warmupSeconds = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--label": label = args[++i]; break;
                case "--scratch-modes": modes = args[++i].Split(','); break;
                default: Console.Error.WriteLine($"unknown argument {args[i]}"); return 2;
            }
        }
        if (shapes.Count == 0)
        {
            foreach (string s in new[]
            {
                "32:2:128:512:f16", "32:2:128:2048:f16", "32:2:128:512:f32", "32:2:128:2048:f32",
                "16:4:128:512:f16", "16:4:128:2048:f16", "16:4:128:512:f32", "16:4:128:2048:f32",
            })
                shapes.Add(Shape.Parse(s));
        }

        var alloc = new CpuAllocator(BlasEnum.DotNet);
        var host = AttentionHost.Create();
        var cases = shapes.Select(s => new Case(s, alloc)).ToList();
        // One sample list per (shape, mode); a shape's tensors are shared by its modes.
        var samples = cases.ToDictionary(c => c, c => modes.ToDictionary(m => m, m => new List<double>()));
        var hashes = cases.ToDictionary(c => c, c => modes.ToDictionary(m => m, m => 0UL));
        try
        {
            foreach (var c in cases)
            {
                foreach (string mode in modes)
                {
                    using var scope = EnterScratchMode(mode);
                    var warm = Stopwatch.StartNew();
                    do c.Run(host); while (warm.Elapsed.TotalSeconds < warmupSeconds);
                }
            }

            for (int p = 0; p < passes; p++)
            {
                foreach (var c in cases)
                {
                    foreach (string mode in modes)
                    {
                        using var scope = EnterScratchMode(mode);
                        var list = samples[c][mode];
                        for (int r = 0; r < reps; r++)
                        {
                            long t0 = Stopwatch.GetTimestamp();
                            c.Run(host);
                            list.Add(Stopwatch.GetElapsedTime(t0).TotalMicroseconds);
                        }
                        hashes[c][mode] = c.OutputHash();
                    }
                }
            }

            foreach (var c in cases)
            {
                foreach (string mode in modes)
                {
                    var list = samples[c][mode];
                    list.Sort();
                    double Pct(double q) => list[(int)Math.Min(list.Count - 1, Math.Round(q * (list.Count - 1)))];
                    string name = mode == "default" ? c.Shape.ToString() : $"{c.Shape}@{mode}";
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"{label}\t{name}\tmedian_us={Pct(0.5):F2}\tp10_us={Pct(0.1):F2}\tp90_us={Pct(0.9):F2}\tmin_us={list[0]:F2}\tn={list.Count}\thash={hashes[c][mode]:x16}"));
                }
            }
        }
        finally
        {
            foreach (var c in cases) c.Dispose();
        }
        return 0;
    }

    /// <summary>Forces the score scratch onto the stack ("stack") or into the pool ("rent") for the
    /// calls this thread makes until disposed; null for "default".</summary>
    private static IDisposable EnterScratchMode(string mode)
    {
        if (mode == "default") return null;
        int limit = mode switch
        {
            "stack" => int.MaxValue,
            "rent" => 0,
            _ => throw new ArgumentException($"scratch mode '{mode}' is not default, stack or rent"),
        };
        MethodInfo set = typeof(ModelBase).Assembly.GetType("TensorSharp.Models.CpuScratch")
            ?.GetMethod("OverrideStackFloatLimit", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("this build has no CpuScratch.OverrideStackFloatLimit; only 'default' applies");
        return (IDisposable)set.Invoke(null, new object[] { limit });
    }

    private sealed record Shape(int Heads, int KvHeads, int HeadDim, int Keys, bool F16)
    {
        public static Shape Parse(string s)
        {
            string[] p = s.Split(':');
            if (p.Length != 5) throw new ArgumentException($"shape '{s}' is not heads:kvHeads:headDim:keys:f16|f32");
            int I(int i) => int.Parse(p[i], CultureInfo.InvariantCulture);
            bool f16 = p[4] switch { "f16" => true, "f32" => false, _ => throw new ArgumentException($"cache dtype '{p[4]}'") };
            return new Shape(I(0), I(1), I(2), I(3), f16);
        }

        public override string ToString() => $"{Heads}:{KvHeads}:{HeadDim}:{Keys}:{(F16 ? "f16" : "f32")}";
    }

    private sealed class Case : IDisposable
    {
        public readonly Shape Shape;
        private readonly Tensor _q, _k, _v, _r;
        private readonly float _scale;

        public Case(Shape s, IAllocator alloc)
        {
            Shape = s;
            _scale = 1f / MathF.Sqrt(s.HeadDim);
            // A fixed seed per shape (HashCode is randomized per process), so every run and both
            // builds of an A/B see the same inputs and their output hashes are comparable.
            var rnd = new Random(s.Heads * 1_000_003 + s.KvHeads * 10_007 + s.HeadDim * 101 + s.Keys);
            _q = new Tensor(alloc, DType.Float32, 1, (long)s.Heads * s.HeadDim);
            _r = new Tensor(alloc, DType.Float32, 1, (long)s.Heads * s.HeadDim);
            DType cacheType = s.F16 ? DType.Float16 : DType.Float32;
            // The cache is the model's (num_kv_heads, max_seq_len, head_dim) layout with headroom
            // past the attended keys, as a live KV cache has.
            long maxSeq = s.Keys + 64;
            _k = new Tensor(alloc, cacheType, s.KvHeads, maxSeq, s.HeadDim);
            _v = new Tensor(alloc, cacheType, s.KvHeads, maxSeq, s.HeadDim);
            Fill(_q, rnd, 1f);
            Fill(_k, rnd, 1f);
            Fill(_v, rnd, 1f);
        }

        public void Run(AttentionHost host)
        {
            if (Shape.F16) host.F16(_q, _k, _v, _r, Shape.Heads, Shape.KvHeads, Shape.HeadDim, Shape.Keys, _scale);
            else host.F32(_q, _k, _v, _r, Shape.Heads, Shape.KvHeads, Shape.HeadDim, Shape.Keys, _scale);
        }

        public ulong OutputHash()
        {
            float* p = TensorComputePrimitives.GetFloatPointer(_r);
            ulong h = 14695981039346656037UL;
            long n = _r.ElementCount();
            for (long i = 0; i < n; i++)
            {
                uint bits = BitConverter.SingleToUInt32Bits(p[i]);
                for (int b = 0; b < 4; b++) { h ^= (bits >> (8 * b)) & 0xFF; h *= 1099511628211UL; }
            }
            return h;
        }

        private static void Fill(Tensor t, Random rnd, float amplitude)
        {
            long n = t.ElementCount();
            if (t.ElementType == DType.Float16)
            {
                ushort* p = TensorComputePrimitives.GetHalfPointer(t);
                for (long i = 0; i < n; i++)
                    p[i] = BitConverter.HalfToUInt16Bits((System.Half)(((float)rnd.NextDouble() * 2f - 1f) * amplitude));
            }
            else
            {
                float* p = TensorComputePrimitives.GetFloatPointer(t);
                for (long i = 0; i < n; i++) p[i] = ((float)rnd.NextDouble() * 2f - 1f) * amplitude;
            }
        }

        public void Dispose()
        {
            _q.Dispose(); _k.Dispose(); _v.Dispose(); _r.Dispose();
        }
    }

    /// <summary>Reaches ModelBase's protected decode-attention entry points. Never constructed:
    /// <see cref="Create"/> returns an uninitialized instance (no field, no finalizer is involved).</summary>
    private sealed class AttentionHost : ModelBase
    {
        private AttentionHost() : base(null, BackendType.Cpu) => throw new NotSupportedException();

        public static AttentionHost Create() => (AttentionHost)RuntimeHelpers.GetUninitializedObject(typeof(AttentionHost));

        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();

        protected override void ResetKVCacheCore() { }

        public void F32(Tensor q, Tensor k, Tensor v, Tensor r, int heads, int kvHeads, int headDim, int keys, float scale)
            => AttentionDecodePureCS(q, k, v, r, heads, kvHeads, headDim, keys, scale);

        public void F16(Tensor q, Tensor k, Tensor v, Tensor r, int heads, int kvHeads, int headDim, int keys, float scale)
            => AttentionDecodePureCSF16(q, k, v, r, heads, kvHeads, headDim, keys, scale);
    }
}
