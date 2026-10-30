using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaDevice
    {
        private CudaDevice(int ordinal, string name, long totalMemoryBytes, int ccMajor, int ccMinor, int multiprocessorCount)
        {
            Ordinal = ordinal;
            Name = name;
            TotalMemoryBytes = totalMemoryBytes;
            ComputeCapabilityMajor = ccMajor;
            ComputeCapabilityMinor = ccMinor;
            MultiprocessorCount = multiprocessorCount;
        }

        public int Ordinal { get; }
        public string Name { get; }
        public long TotalMemoryBytes { get; }
        public int ComputeCapabilityMajor { get; }
        public int ComputeCapabilityMinor { get; }
        public int MultiprocessorCount { get; }

        public string TotalMemoryFormatted => $"{TotalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB";
        public string ComputeCapability => $"{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

        /// <summary>
        /// True when the driver is CUDA 13 or newer and at least one GPU meets
        /// <see cref="CudaRequirements"/>. When it is false because of those requirements,
        /// <see cref="UnavailableReason"/> says why.
        /// </summary>
        public static bool IsAvailable()
        {
            try
            {
                string driverName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
                if (!NativeLibrary.TryLoad(driverName, out IntPtr handle))
                    return false;

                NativeLibrary.Free(handle);
                return ProbeSupportedDeviceCount() > 0;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Why the last <see cref="IsAvailable"/> probe found no usable GPU although the driver
        /// loaded (a CUDA 12 driver, or only GPUs below compute capability 7.5); null otherwise.
        /// </summary>
        public static string UnavailableReason { get; private set; }

        public static int GetDeviceCount()
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            return count;
        }

        public static CudaDevice GetDevice(int ordinal)
        {
            CudaLibraryResolver.Register();
            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGet(out int device, ordinal).ThrowOnError();

            byte[] nameBuffer = new byte[256];
            CudaDriverApi.cuDeviceGetName(nameBuffer, nameBuffer.Length, device).ThrowOnError();
            int nullIndex = Array.IndexOf(nameBuffer, (byte)0);
            string name = Encoding.ASCII.GetString(nameBuffer, 0, nullIndex >= 0 ? nullIndex : nameBuffer.Length).Trim();

            CudaDriverApi.cuDeviceTotalMem(out UIntPtr totalMemory, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMinor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int smCount, CudaDriverApi.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).ThrowOnError();

            return new CudaDevice(ordinal, name, checked((long)totalMemory.ToUInt64()), ccMajor, ccMinor, smCount);
        }

        public override string ToString()
        {
            return $"GPU {Ordinal}: {Name} ({TotalMemoryFormatted}, sm_{ComputeCapabilityMajor}{ComputeCapabilityMinor}, {MultiprocessorCount} SMs)";
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static int ProbeSupportedDeviceCount()
        {
            CudaLibraryResolver.Register();
            UnavailableReason = null;
            CudaDriverApi.cuDriverGetVersion(out int driverVersion).ThrowOnError();
            string driverReason = CudaRequirements.DescribeUnsupportedDriver(driverVersion);
            if (driverReason != null)
            {
                UnavailableReason = driverReason;
                return 0;
            }

            CudaDriverApi.cuInit(0).ThrowOnError();
            CudaDriverApi.cuDeviceGetCount(out int count).ThrowOnError();
            int supported = 0;
            string firstReason = null;
            for (int ordinal = 0; ordinal < count; ordinal++)
            {
                CudaDevice device = GetDevice(ordinal);
                string reason = CudaRequirements.DescribeUnsupportedDevice(
                    device.ComputeCapabilityMajor, device.ComputeCapabilityMinor, device.Name);
                if (reason == null)
                    supported++;
                else
                    firstReason ??= reason;
            }

            if (supported == 0 && firstReason != null)
                UnavailableReason = firstReason;
            return supported;
        }
    }
}
