using System;
using System.Text;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// What the direct CUDA backend runs on: a CUDA 13 driver (R580 or newer) and a GPU of
    /// compute capability 7.5 (Turing) or newer, the oldest architecture CUDA 13 compiles.
    /// CUDA 12 drivers and Maxwell/Pascal/Volta GPUs are not supported. Checked when a context
    /// is created, and used to explain a kernel module the driver will not load, so an
    /// unsupported setup fails with a cause instead of a bare CUDA error 209, 218 or 222.
    /// The decisions are pure functions of the version numbers, so they are unit-tested
    /// without a GPU.
    /// </summary>
    public static class CudaRequirements
    {
        /// <summary>cuDriverGetVersion's value (1000 * major + 10 * minor) for CUDA 13.0.</summary>
        public const int MinimumDriverVersion = 13000;

        /// <summary>Compute capability 7.5 (Turing) as major * 10 + minor.</summary>
        public const int MinimumComputeCapability = 75;

        private const int CudaErrorNoBinaryForGpu = 209;
        private const int CudaErrorInvalidPtx = 218;
        private const int CudaErrorJitCompilerNotFound = 221;
        private const int CudaErrorUnsupportedPtxVersion = 222;
        private const int CudaErrorJitCompilationDisabled = 223;

        /// <summary>"13.5" for cuDriverGetVersion's 13050.</summary>
        public static string FormatDriverVersion(int driverVersion) =>
            $"{driverVersion / 1000}.{driverVersion % 1000 / 10}";

        /// <summary>
        /// Why a driver of <paramref name="driverVersion"/> (cuDriverGetVersion) and a GPU of
        /// compute capability <paramref name="ccMajor"/>.<paramref name="ccMinor"/> cannot run
        /// the cuda backend, or null when they can.
        /// </summary>
        public static string DescribeUnsupported(int driverVersion, int ccMajor, int ccMinor, string deviceName = null) =>
            DescribeUnsupportedDriver(driverVersion) ?? DescribeUnsupportedDevice(ccMajor, ccMinor, deviceName);

        /// <summary>Why a driver of <paramref name="driverVersion"/> (cuDriverGetVersion) is too old, or null.</summary>
        public static string DescribeUnsupportedDriver(int driverVersion)
        {
            if (driverVersion >= MinimumDriverVersion)
                return null;
            return "the cuda backend needs a CUDA 13 driver (R580 or newer); this driver is CUDA " +
                   FormatDriverVersion(driverVersion) + ". CUDA 12 is not supported: update the NVIDIA driver.";
        }

        /// <summary>Why a GPU of compute capability <paramref name="ccMajor"/>.<paramref name="ccMinor"/> is too old, or null.</summary>
        public static string DescribeUnsupportedDevice(int ccMajor, int ccMinor, string deviceName = null)
        {
            if (ccMajor * 10 + ccMinor >= MinimumComputeCapability)
                return null;
            string device = string.IsNullOrWhiteSpace(deviceName) ? "this GPU" : deviceName.Trim();
            return $"the cuda backend needs compute capability 7.5+ (Turing or newer); {device} is " +
                   $"{ccMajor}.{ccMinor}. CUDA 13 dropped Maxwell, Pascal and Volta.";
        }

        /// <summary>
        /// Throws <see cref="NotSupportedException"/> when the installed driver or the GPU behind
        /// <paramref name="device"/> (a CUdevice handle) is below the floor above.
        /// </summary>
        internal static void EnsureSupported(int device)
        {
            CudaDriverApi.cuDriverGetVersion(out int driverVersion).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMajor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int ccMinor, CudaDriverApi.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).ThrowOnError();
            string reason = DescribeUnsupported(driverVersion, ccMajor, ccMinor, TryGetDeviceName(device));
            if (reason != null)
                throw new NotSupportedException(char.ToUpperInvariant(reason[0]) + reason.Substring(1));
        }

        /// <summary>
        /// Explains why cuModuleLoadData rejected a PTX image, from the image's header (the
        /// toolkit release and PTX ISA nvcc wrote, and its .target) and the driver and GPU it
        /// was loaded on. Null when the error is not a compatibility failure.
        /// </summary>
        public static string DescribeModuleLoadFailure(
            int errorCode, string ptxToolkit, string ptxIsa, int ptxTargetArch,
            int driverVersion, int ccMajor, int ccMinor)
        {
            int cc = ccMajor * 10 + ccMinor;
            string driver = driverVersion > 0 ? "CUDA " + FormatDriverVersion(driverVersion) : "unknown";
            switch (errorCode)
            {
                case CudaErrorUnsupportedPtxVersion:
                {
                    string isa = string.IsNullOrEmpty(ptxIsa) ? "of this module" : ptxIsa;
                    string needs = string.IsNullOrEmpty(ptxToolkit)
                        ? "a newer driver"
                        : $"a CUDA {ptxToolkit}+ driver";
                    string built = string.IsNullOrEmpty(ptxToolkit) ? "" : $" (built by CUDA {ptxToolkit})";
                    return $"PTX ISA {isa}{built} needs {needs}; this driver is {driver}. A driver JITs only " +
                           "PTX no newer than itself: update the driver, or build with an nvcc no newer than it " +
                           "(a fatbin with SASS for this GPU loads on any CUDA 13 driver).";
                }

                case CudaErrorNoBinaryForGpu:
                case CudaErrorInvalidPtx:
                {
                    if (cc > 0 && cc < MinimumComputeCapability)
                        return $"no code for sm_{cc}: compute capability {ccMajor}.{ccMinor} is older than CUDA 13 supports (7.5+, Turing).";
                    if (cc > 0 && ptxTargetArch > cc)
                        return $"no code for sm_{cc}: the PTX targets sm_{ptxTargetArch}, newer than this GPU (compute capability {ccMajor}.{ccMinor}).";
                    string target = ptxTargetArch > 0 ? $"sm_{ptxTargetArch}" : "its target";
                    string gpu = cc > 0 ? $"sm_{cc}" : "this GPU";
                    return $"no code for {gpu}: the driver could not compile this PTX ({target}) for it; " +
                           "rebuild the kernels with a CUDA 13 nvcc for this GPU.";
                }

                case CudaErrorJitCompilerNotFound:
                case CudaErrorJitCompilationDisabled:
                {
                    string gpu = cc > 0 ? $"sm_{cc}" : "this GPU";
                    return "the driver cannot JIT-compile PTX here (JIT compiler missing or disabled); build " +
                           $"TensorSharp.Backends.Cuda with a CUDA 13 nvcc so a fatbin with SASS for {gpu} ships beside the PTX.";
                }

                default:
                    return null;
            }
        }

        /// <summary>
        /// The toolkit release ("13.0" from "Cuda compilation tools, release 13.0, V13.0.88") and
        /// PTX ISA (".version 9.0") nvcc wrote at the top of a PTX image; null when absent.
        /// </summary>
        internal static (string Toolkit, string Isa) ParsePtxHeader(byte[] ptx)
        {
            if (ptx == null)
                return (null, null);
            // Both lines sit in the first few lines; never scan a whole module.
            string head = Encoding.ASCII.GetString(ptx, 0, Math.Min(ptx.Length, 64 * 1024));
            var toolkit = System.Text.RegularExpressions.Regex.Match(head, @"(?m)^//\s*Cuda compilation tools, release (\d+\.\d+),");
            var isa = System.Text.RegularExpressions.Regex.Match(head, @"(?m)^\.version (\d+\.\d+)\r?$");
            return (toolkit.Success ? toolkit.Groups[1].Value : null, isa.Success ? isa.Groups[1].Value : null);
        }

        private static string TryGetDeviceName(int device)
        {
            try
            {
                byte[] buffer = new byte[256];
                if (CudaDriverApi.cuDeviceGetName(buffer, buffer.Length, device) != 0)
                    return null;
                int end = Array.IndexOf(buffer, (byte)0);
                return Encoding.ASCII.GetString(buffer, 0, end >= 0 ? end : buffer.Length).Trim();
            }
            catch
            {
                return null;
            }
        }
    }
}
