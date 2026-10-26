using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed class CudaModule : IDisposable
    {
        private readonly Dictionary<string, IntPtr> functions = new Dictionary<string, IntPtr>(StringComparer.Ordinal);
        private IntPtr module;

        private CudaModule(IntPtr module)
        {
            this.module = module;
        }

        /// <summary>
        /// Load a kernel module. For a <c>.ptx</c> path, the <c>.fatbin</c> the same build
        /// produced beside it is preferred: its SASS loads on any driver of the toolkit's
        /// major version, while the driver can only JIT PTX from a toolkit no newer than
        /// itself (CUDA 13.4 PTX fails on a 13.2 driver). The fatbin is used only when its
        /// <c>.fatbin.sha256</c> sidecar names this PTX, so one left behind by an earlier
        /// build never shadows newer kernels; a fatbin the driver rejects (no SASS for this
        /// GPU) falls back to the PTX.
        /// </summary>
        public static CudaModule LoadFromFile(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            byte[] bytes = File.ReadAllBytes(path);
            byte[] fatbin = TryReadMatchingFatbin(path, bytes);
            if (fatbin != null)
            {
                try
                {
                    return LoadImage(fatbin);
                }
                catch (CudaException)
                {
                    // No SASS for this GPU and PTX the driver cannot JIT; try the PTX file.
                }
            }
            return LoadFromBytes(bytes);
        }

        internal static byte[] TryReadMatchingFatbin(string ptxPath, byte[] ptxBytes)
        {
            if (!ptxPath.EndsWith(".ptx", StringComparison.OrdinalIgnoreCase))
                return null;
            string fatbinPath = Path.ChangeExtension(ptxPath, ".fatbin");
            string sidecarPath = fatbinPath + ".sha256";
            if (!File.Exists(fatbinPath) || !File.Exists(sidecarPath))
                return null;
            try
            {
                string recorded = File.ReadAllText(sidecarPath).Trim();
                string actual = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(ptxBytes));
                return string.Equals(recorded, actual, StringComparison.OrdinalIgnoreCase)
                    ? File.ReadAllBytes(fatbinPath)
                    : null;
            }
            catch (IOException)
            {
                return null;
            }
        }

        private static unsafe CudaModule LoadImage(byte[] image)
        {
            fixed (byte* data = image)
            {
                CudaDriverApi.cuModuleLoadData(out IntPtr module, (IntPtr)data).ThrowOnError();
                return new CudaModule(module);
            }
        }

        public static unsafe CudaModule LoadFromBytes(byte[] ptxBytes)
        {
            if (ptxBytes == null)
                throw new ArgumentNullException(nameof(ptxBytes));

            byte[] terminated = ptxBytes;
            if (terminated.Length == 0 || terminated[terminated.Length - 1] != 0)
            {
                terminated = new byte[ptxBytes.Length + 1];
                Buffer.BlockCopy(ptxBytes, 0, terminated, 0, ptxBytes.Length);
            }

            fixed (byte* ptx = terminated)
            {
                CudaDriverApi.cuModuleLoadData(out IntPtr module, (IntPtr)ptx).ThrowOnError();
                return new CudaModule(module);
            }
        }

        public IntPtr GetFunction(string name)
        {
            if (!functions.TryGetValue(name, out IntPtr function))
            {
                CudaDriverApi.cuModuleGetFunction(out function, module, name).ThrowOnError();
                functions.Add(name, function);
            }

            return function;
        }

        public void Dispose()
        {
            IntPtr current = module;
            if (current != IntPtr.Zero)
            {
                module = IntPtr.Zero;
                functions.Clear();
                CudaDriverApi.cuModuleUnload(current);
            }
        }
    }
}
