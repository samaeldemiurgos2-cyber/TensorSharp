using System.Diagnostics;
using System.Runtime.InteropServices;

internal static class Native
{
    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, (name, assembly, search) => name switch
        {
            "cuda_driver" => Load("libcuda.so.1", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
            "cublas_probe" => Load("libcublas.so.13", "/usr/local/cuda/lib64/libcublas.so.13"),
            _ => IntPtr.Zero
        });
    }

    private static IntPtr Load(params string[] candidates)
    {
        foreach(string candidate in candidates)if(NativeLibrary.TryLoad(candidate,out var library))return library;
        throw new DllNotFoundException($"No usable library among {string.Join(",",candidates)}.");
    }

    internal static void Check(int status, string call)
    { if (status != 0) throw new InvalidOperationException($"{call} returned {status}."); }

    [DllImport("cuda_driver")] internal static extern int cuInit(uint flags);
    [DllImport("cuda_driver")] internal static extern int cuDriverGetVersion(out int version);
    [DllImport("cuda_driver")] internal static extern int cuDeviceGet(out int device, int ordinal);
    [DllImport("cuda_driver")] internal static extern int cuDeviceGetName(byte[] name, int length, int device);
    [DllImport("cuda_driver")] internal static extern int cuDeviceGetAttribute(out int value, int attribute, int device);
    [DllImport("cuda_driver", EntryPoint = "cuDeviceTotalMem_v2")] internal static extern int cuDeviceTotalMem(out nuint bytes, int device);
    [DllImport("cuda_driver", EntryPoint = "cuCtxCreate_v2")] internal static extern int cuCtxCreate(out IntPtr context, uint flags, int device);
    [DllImport("cuda_driver", EntryPoint = "cuCtxDestroy_v2")] internal static extern int cuCtxDestroy(IntPtr context);
    [DllImport("cuda_driver", EntryPoint = "cuMemGetInfo_v2")] internal static extern int cuMemGetInfo(out nuint free, out nuint total);
    [DllImport("cuda_driver", EntryPoint = "cuMemAlloc_v2")] internal static extern int cuMemAlloc(out IntPtr pointer, nuint bytes);
    [DllImport("cuda_driver", EntryPoint = "cuMemFree_v2")] internal static extern int cuMemFree(IntPtr pointer);
    [DllImport("cuda_driver", EntryPoint = "cuMemcpyHtoD_v2")] internal static extern int cuMemcpyHtoD(IntPtr target, IntPtr source, nuint bytes);
    [DllImport("cuda_driver", EntryPoint = "cuMemcpyDtoH_v2")] internal static extern int cuMemcpyDtoH(IntPtr target, IntPtr source, nuint bytes);
    [DllImport("cuda_driver")] internal static extern int cuStreamCreate(out IntPtr stream, uint flags);
    [DllImport("cuda_driver", EntryPoint = "cuStreamDestroy_v2")] internal static extern int cuStreamDestroy(IntPtr stream);
    [DllImport("cuda_driver")] internal static extern int cuStreamSynchronize(IntPtr stream);
    [DllImport("cuda_driver")] internal static extern int cuEventCreate(out IntPtr evt, uint flags);
    [DllImport("cuda_driver", EntryPoint = "cuEventDestroy_v2")] internal static extern int cuEventDestroy(IntPtr evt);
    [DllImport("cuda_driver")] internal static extern int cuEventRecord(IntPtr evt, IntPtr stream);
    [DllImport("cuda_driver")] internal static extern int cuEventSynchronize(IntPtr evt);
    [DllImport("cuda_driver")] internal static extern int cuEventElapsedTime(out float milliseconds, IntPtr begin, IntPtr end);
    [DllImport("cublas_probe", EntryPoint = "cublasCreate_v2")] internal static extern int cublasCreate(out IntPtr handle);
    [DllImport("cublas_probe", EntryPoint = "cublasDestroy_v2")] internal static extern int cublasDestroy(IntPtr handle);
    [DllImport("cublas_probe", EntryPoint = "cublasSetStream_v2")] internal static extern int cublasSetStream(IntPtr handle, IntPtr stream);
    [DllImport("cublas_probe", EntryPoint = "cublasGetVersion_v2")] internal static extern int cublasGetVersion(IntPtr handle, out int version);
    [DllImport("cublas_probe")] internal static extern int cublasGemmEx(IntPtr handle, int transA, int transB,
        int m, int n, int k, ref float alpha, IntPtr a, int typeA, int lda, IntPtr b, int typeB, int ldb,
        ref float beta, IntPtr c, int typeC, int ldc, int computeType, int algorithm);
}

internal sealed class DeviceBuffer : IDisposable
{
    internal IntPtr Pointer { get; private set; }
    internal int Count { get; }
    internal DeviceBuffer(int count)
    {
        Count = count;
        Native.Check(Native.cuMemAlloc(out var pointer, checked((nuint)count * 4)), "cuMemAlloc");
        Pointer = pointer;
    }
    internal unsafe void Upload(float[] values)
    {
        if (values.Length != Count) throw new ArgumentException("Wrong upload length.");
        fixed (float* source = values)
            Native.Check(Native.cuMemcpyHtoD(Pointer, (IntPtr)source, checked((nuint)Count * 4)), "cuMemcpyHtoD");
    }
    internal unsafe float[] Download()
    {
        var values = new float[Count];
        fixed (float* target = values)
            Native.Check(Native.cuMemcpyDtoH((IntPtr)target, Pointer, checked((nuint)Count * 4)), "cuMemcpyDtoH");
        return values;
    }
    public void Dispose()
    {
        if (Pointer == IntPtr.Zero) return;
        Native.Check(Native.cuMemFree(Pointer), "cuMemFree"); Pointer = IntPtr.Zero;
    }
}

internal sealed class Gpu : IDisposable
{
    private IntPtr _context, _begin, _end;
    internal IntPtr Stream { get; private set; }
    internal IntPtr Blas { get; private set; }
    internal int DriverVersion { get; }
    internal int BlasVersion { get; }
    internal string Name { get; }
    internal string ComputeCapability { get; }
    internal ulong TotalMemory { get; }
    internal Gpu(int ordinal)
    {
        Native.Check(Native.cuInit(0), "cuInit");
        Native.Check(Native.cuDriverGetVersion(out int driver), "cuDriverGetVersion"); DriverVersion = driver;
        Native.Check(Native.cuDeviceGet(out int device, ordinal), "cuDeviceGet");
        byte[] name = new byte[256]; Native.Check(Native.cuDeviceGetName(name, name.Length, device), "cuDeviceGetName");
        Name = System.Text.Encoding.UTF8.GetString(name).TrimEnd('\0');
        Native.Check(Native.cuDeviceGetAttribute(out int major, 75, device), "computeMajor");
        Native.Check(Native.cuDeviceGetAttribute(out int minor, 76, device), "computeMinor");
        ComputeCapability = $"{major}.{minor}";
        if (major < 8) throw new NotSupportedException("This probe requires TF32-capable compute capability >=8.");
        Native.Check(Native.cuDeviceTotalMem(out nuint total, device), "cuDeviceTotalMem"); TotalMemory = (ulong)total;
        Native.Check(Native.cuCtxCreate(out _context, 0, device), "cuCtxCreate");
        try
        {
            Native.Check(Native.cuStreamCreate(out var stream, 1), "cuStreamCreate"); Stream = stream;
            Native.Check(Native.cublasCreate(out var blas), "cublasCreate"); Blas = blas;
            Native.Check(Native.cublasSetStream(Blas, Stream), "cublasSetStream");
            Native.Check(Native.cublasGetVersion(Blas, out int version), "cublasGetVersion"); BlasVersion = version;
            Native.Check(Native.cuEventCreate(out _begin, 0), "cuEventCreate");
            Native.Check(Native.cuEventCreate(out _end, 0), "cuEventCreate");
        }
        catch { Dispose(); throw; }
    }
    internal void Gemm(bool rowMajorWeight, IntPtr input, IntPtr weight, IntPtr output, int rows, int inner, int columns, int compute)
    {
        float alpha = 1, beta = 0;
        Native.Check(Native.cublasGemmEx(Blas, rowMajorWeight ? 1 : 0, 0,
            columns, rows, inner, ref alpha, weight, 0, rowMajorWeight ? inner : columns,
            input, 0, inner, ref beta, output, 0, columns, compute, -1), "cublasGemmEx");
    }
    internal (double GpuMs, double HostUs, double CompletedWallMs) Measure(Action call, int calls)
    {
        Native.Check(Native.cuEventRecord(_begin, Stream), "cuEventRecord(begin)");
        long wall = Stopwatch.GetTimestamp(), host = Stopwatch.GetTimestamp();
        for (int i = 0; i < calls; i++) call();
        double hostUs = Stopwatch.GetElapsedTime(host).TotalMicroseconds / calls;
        Native.Check(Native.cuEventRecord(_end, Stream), "cuEventRecord(end)");
        Native.Check(Native.cuEventSynchronize(_end), "cuEventSynchronize");
        double completedWall = Stopwatch.GetElapsedTime(wall).TotalMilliseconds / calls;
        Native.Check(Native.cuEventElapsedTime(out float ms, _begin, _end), "cuEventElapsedTime");
        return (ms / calls, hostUs, completedWall);
    }
    internal void Synchronize() => Native.Check(Native.cuStreamSynchronize(Stream), "cuStreamSynchronize");
    internal ulong FreeMemory()
    {
        Native.Check(Native.cuMemGetInfo(out nuint free, out _), "cuMemGetInfo"); return (ulong)free;
    }
    public void Dispose()
    {
        if (Stream != IntPtr.Zero) Synchronize();
        if (_begin != IntPtr.Zero) { Native.Check(Native.cuEventDestroy(_begin), "cuEventDestroy"); _begin = IntPtr.Zero; }
        if (_end != IntPtr.Zero) { Native.Check(Native.cuEventDestroy(_end), "cuEventDestroy"); _end = IntPtr.Zero; }
        if (Blas != IntPtr.Zero) { Native.Check(Native.cublasDestroy(Blas), "cublasDestroy"); Blas = IntPtr.Zero; }
        if (Stream != IntPtr.Zero) { Native.Check(Native.cuStreamDestroy(Stream), "cuStreamDestroy"); Stream = IntPtr.Zero; }
        if (_context != IntPtr.Zero) { Native.Check(Native.cuCtxDestroy(_context), "cuCtxDestroy"); _context = IntPtr.Zero; }
    }
}
