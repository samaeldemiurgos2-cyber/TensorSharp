using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cuda.Interop
{
    internal static partial class CublasApi
    {
        private const string LibName = "cublas";

        [LibraryImport(LibName, EntryPoint = "cublasCreate_v2")]
        public static partial int cublasCreate(out IntPtr handle);

        [LibraryImport(LibName, EntryPoint = "cublasDestroy_v2")]
        public static partial int cublasDestroy(IntPtr handle);

        [LibraryImport(LibName, EntryPoint = "cublasSetStream_v2")]
        public static partial int cublasSetStream(IntPtr handle, IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cublasSetMathMode(IntPtr handle, int mode);

        [LibraryImport(LibName)]
        public static partial int cublasGetMathMode(IntPtr handle, out int mode);

        [LibraryImport(LibName, EntryPoint = "cublasSgemm_v2")]
        public static partial int cublasSgemm(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            IntPtr b,
            int ldb,
            ref float beta,
            IntPtr c,
            int ldc);

        [LibraryImport(LibName)]
        public static partial int cublasSgemmStridedBatched(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            long strideA,
            IntPtr b,
            int ldb,
            long strideB,
            ref float beta,
            IntPtr c,
            int ldc,
            long strideC,
            int batchCount);

        [LibraryImport(LibName)]
        public static partial int cublasGemmEx(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int aType,
            int lda,
            IntPtr b,
            int bType,
            int ldb,
            ref float beta,
            IntPtr c,
            int cType,
            int ldc,
            int computeType,
            int algo);

        [LibraryImport(LibName)]
        public static partial int cublasGemmStridedBatchedEx(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int aType,
            int lda,
            long strideA,
            IntPtr b,
            int bType,
            int ldb,
            long strideB,
            ref float beta,
            IntPtr c,
            int cType,
            int ldc,
            long strideC,
            int batchCount,
            int computeType,
            int algo);

        public const int CUBLAS_OP_N = 0;
        public const int CUBLAS_OP_T = 1;
        // cublasMath_t and cublasComputeType_t, CUDA 13.4 cublas_api.h:172-212. CUBLAS_TENSOR_OP_MATH (1)
        // is left out on purpose: cuBLAS 13 defines it as CUBLAS_COMPUTE_32F_FAST_16F, which rounds the
        // operands of every F32 GEMM that does not name its compute type to IEEE half (see
        // CudaCublasHandle.Create).
        public const int CUBLAS_DEFAULT_MATH = 0;
        public const int CUDA_R_16F = 2;
        public const int CUDA_R_32F = 0;
        public const int CUDA_R_16BF = 14;
        /// <summary>F32 accumulation; for F32 operands the handle's math mode decides their precision.</summary>
        public const int CUBLAS_COMPUTE_32F = 68;
        /// <summary>F32 operands and accumulation whatever the handle's math mode or the environment.</summary>
        public const int CUBLAS_COMPUTE_32F_PEDANTIC = 69;
        /// <summary>Explicit approximation: F32 operands may be rounded to IEEE half (10-bit mantissa, inf past
        /// 65504) on the shapes cuBLAS sends to tensor cores; F32 accumulation and output. What the old
        /// handle's CUBLAS_TENSOR_OP_MATH gave every F32 GEMM; now only by name (CudaHalfOperandGemm).</summary>
        public const int CUBLAS_COMPUTE_32F_FAST_16F = 74;
        /// <summary>Explicit approximation: F32 operands may be rounded to TF32 (F32 exponent range kept).</summary>
        public const int CUBLAS_COMPUTE_32F_FAST_TF32 = 77;
        public const int CUBLAS_GEMM_DEFAULT = -1;
    }
}
