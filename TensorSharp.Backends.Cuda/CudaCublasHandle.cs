using System;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed class CudaCublasHandle : IDisposable
    {
        private IntPtr handle;

        private CudaCublasHandle(IntPtr handle)
        {
            this.handle = handle;
        }

        public IntPtr Handle => handle;

        /// <summary>
        /// The backend's shared handle (one per allocator; every cuBLAS call in this assembly uses it).
        /// </summary>
        /// <remarks>
        /// F32 data keeps F32 operands, so the handle stays in <c>CUBLAS_DEFAULT_MATH</c>. It used to be
        /// <c>CUBLAS_TENSOR_OP_MATH</c>, which cuBLAS 13 defines as <c>CUBLAS_COMPUTE_32F_FAST_16F</c>
        /// (cublas_api.h:174). Under it, every F32 GEMM that does not name its compute type
        /// (<c>cublasSgemm</c>, <c>cublasSgemmStridedBatched</c>, <c>cublasGemmEx</c> with
        /// <c>CUBLAS_COMPUTE_32F</c>) rounded both operands to IEEE half on the shapes cuBLAS sent to
        /// tensor cores: a 10-bit mantissa, and inf past 65504. Which shapes those were depended on the
        /// row count, so a projection's numerics changed with the batch size. It moved DeepSeek V4's
        /// router logits and hyper-connection mixes enough to pick other experts; prefill routers,
        /// Qwen3.8's GDN gates and indexer inputs, prefill attention scores and vision encoders were
        /// rounded the same way.
        /// <para>
        /// The default mode keeps every F32-operand GEMM exact at every shape. 16-bit-operand GEMMs
        /// (F16/BF16 weights, the dequantized prefill route) still run on tensor cores with F32
        /// accumulation, bit-for-bit and speed-for-speed as before. Not <c>CUBLAS_TF32_TENSOR_OP_MATH</c>:
        /// TF32 keeps the exponent range but the same 10-bit mantissa. A call that accepts an
        /// approximation names it per call and says why there: <c>CUBLAS_COMPUTE_32F_FAST_TF32</c>
        /// (<see cref="CudaEmbeddingOps.TryLinearTf32"/>, the embedding attention plan's TF32 policy,
        /// MiniMax-H3's large mapped dense GEMMs), or half operands through
        /// <see cref="CudaHalfOperandGemm"/> (<c>CUBLAS_COMPUTE_32F_FAST_16F</c>, the old handle's
        /// behaviour, kept by name for the bulk attention, vision-encoder and VAE GEMMs that ggml also runs
        /// on half operands and that cost 1.3-3.7x as exact F32). A call that must stay exact whatever the
        /// handle says names <c>CUBLAS_COMPUTE_32F_PEDANTIC</c> (CudaQuantizedOps.RunGemm for F32 weights).
        /// Regression tests: InferenceWeb.Tests/CudaF32GemmPrecisionTests.cs.
        /// </para>
        /// </remarks>
        public static CudaCublasHandle Create()
        {
            CublasApi.cublasCreate(out IntPtr handle).ThrowOnCublasError();
            try
            {
                CublasApi.cublasSetMathMode(handle, CublasApi.CUBLAS_DEFAULT_MATH).ThrowOnCublasError();
                return new CudaCublasHandle(handle);
            }
            catch
            {
                CublasApi.cublasDestroy(handle);
                throw;
            }
        }

        /// <summary>The handle's current <c>cublasMath_t</c>.</summary>
        internal int GetMathMode()
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaCublasHandle));

            CublasApi.cublasGetMathMode(handle, out int mode).ThrowOnCublasError();
            return mode;
        }

        public void SetStream(IntPtr stream)
        {
            if (handle == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaCublasHandle));

            CublasApi.cublasSetStream(handle, stream).ThrowOnCublasError();
        }

        public void Dispose()
        {
            IntPtr nativeHandle = Interlocked.Exchange(ref handle, IntPtr.Zero);
            if (nativeHandle != IntPtr.Zero)
                CublasApi.cublasDestroy(nativeHandle);
        }
    }
}
