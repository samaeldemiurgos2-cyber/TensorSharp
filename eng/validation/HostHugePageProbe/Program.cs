using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.GGML;

// Exercise the actual managed pool without creating a context or loading CUDA.
// The internal pool is accessed in the same way as its unit tests; production
// allocation/free/advice code is not reproduced in this probe.
internal static class Program
{
    private static readonly Type PoolType = typeof(GgmlContext).Assembly.GetType("TensorSharp.GGML.GgmlMemoryPool", true)!;
    private static object Call(object pool, string name, params object[] args) => PoolType.GetMethod(name,
        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(pool, args)!;
    private static GgmlMemoryPoolStatistics Stats(object pool) => (GgmlMemoryPoolStatistics)Call(pool, "GetStatistics");

    public static unsafe int Main(string[] args)
    {
        int length = 40 << 20;
        string? output = null;
        bool requireAdvice = false, exerciseFailure = false;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--bytes" when i + 1 < args.Length:
                    length = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--out" when i + 1 < args.Length:
                    output = Path.GetFullPath(args[++i]); break;
                case "--require-advice": requireAdvice = true; break;
                case "--exercise-failure": exerciseFailure = true; break;
                default: throw new ArgumentException("Usage: --out artifacts/.../result.json [--bytes 41943040] [--require-advice] [--exercise-failure]");
            }
        }
        if (output is null || length < 4096 || length > (128 << 20))
            throw new ArgumentException("--out is required; --bytes must be between 4096 and 134217728.");
        // Generated evidence is kept out of tracked source, including on a VM.
        string[] components = output.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        if (!components.Contains("artifacts") && !output.Contains($"{Path.DirectorySeparatorChar}docs{Path.DirectorySeparatorChar}validation{Path.DirectorySeparatorChar}", StringComparison.Ordinal))
            throw new ArgumentException("Output must be beneath artifacts/ or docs/validation/.");

        var evidence = new Dictionary<string, object?>
        {
            ["complete"] = false, ["passed"] = false, ["bytes"] = length,
            ["timestamp_utc"] = DateTimeOffset.UtcNow,
            ["scope"] = "Managed Linux host allocation, byte preservation, hint counters and release only; no CUDA or inference.",
            ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["os"] = RuntimeInformation.OSDescription,
            ["setting"] = Environment.GetEnvironmentVariable("TS_GGML_HOST_HUGEPAGES"),
            ["require_advice"] = requireAdvice, ["exercise_failure"] = exerciseFailure,
        };
        object? pool = null;
        IntPtr pointer = IntPtr.Zero;
        bool passed = false;
        try
        {
            if (!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("Linux mmap/madvise is required; unavailable is not a passing run.");
            evidence["thp_enabled"] = Read("/sys/kernel/mm/transparent_hugepage/enabled");
            evidence["thp_defrag"] = Read("/sys/kernel/mm/transparent_hugepage/defrag");
            pool = Activator.CreateInstance(PoolType, [GgmlBackendType.Cuda])!;
            evidence["before"] = Stats(pool);
            pointer = (IntPtr)Call(pool, "Allocate", (long)length);
            if (pointer == IntPtr.Zero) throw new InvalidOperationException("Allocation returned null.");
            var bytes = new Span<byte>((void*)pointer, length);
            for (int i = 0; i < length; i++) bytes[i] = (byte)(i ^ (i >> 8) ^ 0xa5);
            Verify(bytes);
            evidence["all_bytes_verified"] = true;
            GgmlMemoryPoolStatistics allocated = Stats(pool);
            evidence["allocated"] = allocated;
            evidence["smaps_mapping"] = Mapping(pointer, length);
            if (allocated.OutstandingBlocks != 1 || allocated.FreshAllocations != 1)
                throw new InvalidOperationException("Allocation counters disagree.");
            long mappedBytes = ((long)length + Environment.SystemPageSize - 1) / Environment.SystemPageSize * Environment.SystemPageSize;
            evidence["mapped_bytes"] = mappedBytes;
            bool shouldAdvise = allocated.HugePageAdviceEnabled && mappedBytes >= (32 << 20);
            if (allocated.HugePageAdviceRequests != (shouldAdvise ? 1 : 0) ||
                allocated.HugePageAdviceRequests != allocated.HugePageAdviceAccepted + allocated.HugePageAdviceFailures ||
                allocated.HugePageAdvisedBytes != allocated.HugePageAdviceAccepted * mappedBytes)
                throw new InvalidOperationException("Advice counters disagree with policy.");
            if (requireAdvice && (!shouldAdvise || allocated.HugePageAdviceAccepted != 1 || allocated.HugePageAdviceFailures != 0))
                throw new InvalidOperationException("An accepted Linux advice call was required but not observed.");
            if (exerciseFailure)
            {
                // Linux rejects a non-page-aligned start with EINVAL. A failed
                // hint must neither alter the data nor replace its owned extent.
                Call(pool, "AdviseHostHugePages", IntPtr.Add(pointer, 1), (nuint)(length - 1));
                var rejected = Stats(pool);
                evidence["after_rejected_hint"] = rejected;
                if (rejected.HugePageAdviceRequests != allocated.HugePageAdviceRequests + 1 ||
                    rejected.HugePageAdviceFailures != allocated.HugePageAdviceFailures + 1 ||
                    rejected.HugePageAdviceAccepted != allocated.HugePageAdviceAccepted)
                    throw new InvalidOperationException("The deliberately invalid hint did not fail safely.");
                Verify(bytes);
                evidence["bytes_verified_after_rejected_hint"] = true;
            }
            Call(pool, "Free", pointer, (long)length);
            pointer = IntPtr.Zero;
            Call(pool, "Close");
            var released = Stats(pool);
            evidence["released"] = released;
            if (released.OutstandingBlocks != 0 || released.RetainedBlocks != 0 ||
                released.SystemFrees != 1 || released.SystemFreeFailures != 0 || !released.IsClosed)
                throw new InvalidOperationException("The original owned allocation was not fully released.");
            passed = true;
        }
        catch (Exception error)
        {
            evidence["error"] = error.ToString();
        }
        finally
        {
            try
            {
                if (pool is not null)
                {
                    if (pointer != IntPtr.Zero) Call(pool, "Free", pointer, (long)length);
                    Call(pool, "Close");
                    evidence["final"] = Stats(pool);
                }
            }
            catch (Exception error)
            {
                evidence["cleanup_error"] = error.ToString();
                passed = false;
            }
            evidence["complete"] = true;
            evidence["passed"] = passed;
            Directory.CreateDirectory(Path.GetDirectoryName(output)!);
            File.WriteAllText(output, JsonSerializer.Serialize(evidence, new JsonSerializerOptions { WriteIndented = true }));
        }
        Console.WriteLine($"passed={passed}; evidence={output}");
        return passed ? 0 : 1;
    }

    private static void Verify(Span<byte> bytes)
    {
        for (int i = 0; i < bytes.Length; i++)
            if (bytes[i] != (byte)(i ^ (i >> 8) ^ 0xa5)) throw new InvalidOperationException($"Byte mismatch at {i}.");
    }

    private static string Read(string path)
    {
        try { return File.ReadAllText(path).Trim(); }
        catch (IOException error) { return "unavailable: " + error.Message; }
    }

    private static Dictionary<string, string>? Mapping(IntPtr pointer, int bytes)
    {
        ulong address = (ulong)pointer.ToInt64();
        Dictionary<string, string>? result = null;
        foreach (string line in File.ReadLines("/proc/self/smaps"))
        {
            string first = line.Split(' ', 2)[0];
            string[] range = first.Split('-');
            if (range.Length == 2 && ulong.TryParse(range[0], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out ulong start) &&
                ulong.TryParse(range[1], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out ulong end))
            {
                if (result is not null) break;
                if (start <= address && address < end)
                    result = new() { ["range"] = first, ["whole_allocation_contained"] = (address + (ulong)bytes <= end).ToString(),
                        ["note"] = "Values cover the containing VMA, which may include adjacent mappings; accepted advice alone does not establish physical huge-page backing." };
            }
            else if (result is not null && line.IndexOf(':') is int colon && colon >= 0)
                result[line[..colon]] = line[(colon + 1)..].Trim();
        }
        return result;
    }
}
