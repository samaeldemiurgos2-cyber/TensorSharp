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
using System.Threading.Tasks;

namespace TensorSharp
{
    /// <summary>
    /// Bulk host-memory copy and clear for cache-sized buffers. A destination that was
    /// never touched (a fresh anonymous mapping) costs a page fault plus kernel zeroing
    /// per 4 KiB on first write, which caps a single thread near 1-2 GB/s: a 150 MB
    /// recurrent-state copy into a new holder took ~75 ms. Splitting large ranges
    /// across threads faults and copies the pages in parallel. Small ranges stay on
    /// the calling thread.
    /// </summary>
    public static unsafe class HostMemory
    {
        private const long ParallelThresholdBytes = 1L << 20;
        private const long ChunkBytes = 512L << 10;

        public static void Copy(void* source, void* destination, long bytes)
        {
            if (bytes <= 0) return;
            if (bytes < ParallelThresholdBytes || Environment.ProcessorCount < 2)
            {
                Buffer.MemoryCopy(source, destination, bytes, bytes);
                return;
            }
            long src = (long)source, dst = (long)destination;
            long chunks = (bytes + ChunkBytes - 1) / ChunkBytes;
            Parallel.For(0L, chunks, i =>
            {
                long offset = i * ChunkBytes;
                long length = Math.Min(ChunkBytes, bytes - offset);
                Buffer.MemoryCopy((void*)(src + offset), (void*)(dst + offset), length, length);
            });
        }

        public static void Clear(void* destination, long bytes)
        {
            if (bytes <= 0) return;
            if (bytes < ParallelThresholdBytes || Environment.ProcessorCount < 2)
            {
                ClearSerial((byte*)destination, bytes);
                return;
            }
            long dst = (long)destination;
            long chunks = (bytes + ChunkBytes - 1) / ChunkBytes;
            Parallel.For(0L, chunks, i =>
            {
                long offset = i * ChunkBytes;
                ClearSerial((byte*)(dst + offset), Math.Min(ChunkBytes, bytes - offset));
            });
        }

        /// <summary>Clear several ranges as one parallel job (e.g. every head plane of a
        /// holder's K/V before a download lands in them).</summary>
        public static void Clear((IntPtr Pointer, long Bytes)[] ranges)
        {
            if (ranges == null || ranges.Length == 0) return;
            var chunks = new System.Collections.Generic.List<(long Address, long Bytes)>();
            foreach (var (pointer, bytes) in ranges)
                for (long offset = 0; offset < bytes; offset += ChunkBytes)
                    chunks.Add(((long)pointer + offset, Math.Min(ChunkBytes, bytes - offset)));
            if (chunks.Count == 1 || Environment.ProcessorCount < 2)
            {
                foreach (var (address, bytes) in chunks) ClearSerial((byte*)address, bytes);
                return;
            }
            Parallel.For(0, chunks.Count, i => ClearSerial((byte*)chunks[i].Address, chunks[i].Bytes));
        }

        private static void ClearSerial(byte* destination, long bytes)
        {
            while (bytes > 0)
            {
                int slice = (int)Math.Min(bytes, int.MaxValue);
                new Span<byte>(destination, slice).Clear();
                destination += slice;
                bytes -= slice;
            }
        }
    }
}
