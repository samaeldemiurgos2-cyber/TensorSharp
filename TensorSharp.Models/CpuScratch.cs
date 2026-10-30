// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Buffers;

namespace TensorSharp.Models
{
    /// <summary>
    /// Where a managed CPU kernel keeps a scratch buffer whose length follows its input - the
    /// attention scores over the context, mostly: on the stack while it is small, rented from
    /// <see cref="ArrayPool{T}.Shared"/> once it is longer than <see cref="StackFloatLimit"/>.
    ///
    /// These kernels run on <see cref="CpuWorkerPool"/> workers, whose stacks are only
    /// <see cref="CpuWorkerPool.WorkerStackBytes"/>, and on ThreadPool and host threads. Running
    /// out of stack is not an exception: the runtime aborts the process. A stackalloc sized by
    /// the context therefore killed the process once a prompt was long enough - Nemotron-3.5
    /// Lightning (16 query heads per KV head) decoding a 13,844-token prompt asked a worker for
    /// 16 x 13,844 x 4 B = 865 KiB and the process died with "Stack overflow" (exit 134).
    ///
    /// The limit is a quarter of a worker's stack, so short contexts keep exactly the stackalloc
    /// they had (a group of 4 query heads per KV head stays on the stack up to 8,192 keys, a group
    /// of 16 up to 2,048), and what a kernel puts on top of it cannot reach the guard page. The
    /// limit bounds each buffer, not a thread's total, so a kernel holds at most one stack scratch
    /// at a time: it must not enter another CpuScratch kernel while its own scratch is live
    /// (<see cref="CpuWorkerPool"/> runs a nested <c>CpuParallel.For</c> inline, on the same stack).
    ///
    /// The rented path does not allocate per call in steady state: the shared pool keeps one array
    /// per size class per thread. That holds under normal memory load only - after a Gen2
    /// collection that finds memory load high (or an array unused for about a minute) the pool
    /// drops its cached arrays, and the next call on each thread allocates its scratch again.
    ///
    /// Call sites follow one shape (stackalloc has to happen in the kernel's own frame):
    /// <code>
    /// float[] rented = null;
    /// Span&lt;float&gt; scratch = n &lt;= limit ? stackalloc float[n] : (rented = CpuScratch.RentFloats(n)).AsSpan(0, n);
    /// try { ... } finally { CpuScratch.Return(rented); }
    /// </code>
    /// with <c>limit</c> read from <see cref="StackFloatLimit"/> on the thread that entered the
    /// kernel and handed to its workers.
    /// </summary>
    internal static class CpuScratch
    {
        /// <summary>Largest length-scaled buffer, in bytes, a kernel puts on the stack.</summary>
        internal const int MaxStackBytes = CpuWorkerPool.WorkerStackBytes / 4;

        /// <summary><see cref="MaxStackBytes"/> in floats (32,768).</summary>
        internal const int MaxStackFloats = MaxStackBytes / sizeof(float);

        // Per entering thread, +1 so the zero default means "no override". Kernels read the
        // limit on the thread that called them and pass the value to their workers, so an
        // override changes only the kernels that thread runs.
        [ThreadStatic] private static int t_limitOverridePlusOne;

        /// <summary>
        /// Floats a kernel entered on this thread may stackalloc for a length-scaled buffer:
        /// <see cref="MaxStackFloats"/> unless <see cref="OverrideStackFloatLimit"/> is in effect.
        /// </summary>
        internal static int StackFloatLimit
        {
            get
            {
                int o = t_limitOverridePlusOne;
                return o == 0 ? MaxStackFloats : o - 1;
            }
        }

        /// <summary>
        /// Replaces <see cref="StackFloatLimit"/> for kernels entered on this thread until the
        /// returned scope is disposed: 0 rents every buffer, a large value stackallocs every one
        /// (the behaviour before the limit existed). For tests and A/B benchmarks.
        /// </summary>
        internal static LimitScope OverrideStackFloatLimit(int floats)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(floats);
            int previous = t_limitOverridePlusOne;
            t_limitOverridePlusOne = Math.Min(floats, int.MaxValue - 1) + 1;
            return new LimitScope(previous);
        }

        /// <summary>
        /// Tests only: fill every rented buffer with NaN before handing it out. Stack scratch is
        /// zeroed (locals are initialised) while a pooled array holds whatever its last renter left,
        /// so a kernel that read its scratch before writing it could still pass a stack-vs-rented
        /// comparison by luck; with NaN in every unwritten slot it cannot.
        /// </summary>
        internal static volatile bool PoisonRentedBuffers;

        /// <summary>A buffer of at least <paramref name="floats"/> floats; contents are undefined.</summary>
        internal static float[] RentFloats(int floats)
        {
            float[] rented = ArrayPool<float>.Shared.Rent(floats);
            if (PoisonRentedBuffers)
                rented.AsSpan().Fill(float.NaN);
            return rented;
        }

        /// <summary>Returns a buffer from <see cref="RentFloats"/>; null (the stack path) is a no-op.</summary>
        internal static void Return(float[] rented)
        {
            if (rented != null)
                ArrayPool<float>.Shared.Return(rented);
        }

        internal readonly struct LimitScope : IDisposable
        {
            private readonly int _previous;
            internal LimitScope(int previous) => _previous = previous;
            public void Dispose() => t_limitOverridePlusOne = _previous;
        }
    }
}
