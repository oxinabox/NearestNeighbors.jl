using CUBLAS

typealias BlasChar Char
import CUBLAS: cublasStatus_t, cublasHandle_t, cublasOperation_t, cublashandle, statuscheck, cublasop,
               gemm!, ger!

const libcublas = Libdl.find_library(["libcublas"], ["/usr/local/cuda"])

typealias CudaVMPitch{T} Union{CudaVecOrMat{T}, CudaPitchedArray{T,1}, CudaPitchedArray{T,2}}

for (fname, elty) in
        ((:cublasDgemm_v2,:Float64),
         (:cublasSgemm_v2,:Float32),
         (:cublasZgemm_v2,:Complex128),
         (:cublasCgemm_v2,:Complex64))
    @eval begin
        # cublasStatus_t cublasDgemm(
        #   cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        #   int m, int n, int k,
        #   const double *alpha, const double *A, int lda,
        #   const double *B, int ldb, const double *beta,
        #   double *C, int ldc)
        function gemm!(transA::BlasChar,
                       transB::BlasChar,
                       alpha::($elty),
                       A::CudaVMPitch{$elty},
                       B::CudaVMPitch{$elty},
                       beta::($elty),
                       C::CudaVMPitch{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            cutransA = cublasop(transA)
            cutransB = cublasop(transB)
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            statuscheck(ccall(($(string(fname)),libcublas), cublasStatus_t,
                              (cublasHandle_t, cublasOperation_t,
                              cublasOperation_t, Cint, Cint, Cint, Ptr{$elty},
                              Ptr{$elty}, Cint, Ptr{$elty}, Cint, Ptr{$elty},
                              Ptr{$elty}, Cint), cublashandle[1], cutransA,
                              cutransB, m, n, k, [alpha], A, lda, B, ldb, [beta],
                              C, ldc))
            C
        end
        function gemm(transA::BlasChar,
                      transB::BlasChar,
                      alpha::($elty),
                      A::CudaMatrix{$elty},
                      B::CudaMatrix{$elty})
            gemm!(transA, transB, alpha, A, B, zero($elty),
                  similar(B, $elty, (size(A, transA == 'N' ? 1 : 2),
                                     size(B, transB == 'N' ? 2 : 1))))
        end
        function gemm(transA::BlasChar,
                      transB::BlasChar,
                      A::CudaMatrix{$elty},
                      B::CudaMatrix{$elty})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end

### ger
for (fname, elty) in ((:cublasDger_v2,:Float64),
                      (:cublasSger_v2,:Float32),
                      (:cublasZgerc_v2,:Complex128),
                      (:cublasCgerc_v2,:Complex64))
    @eval begin
        # cublasStatus_t cublasDger(
        #   cublasHandle_t handle, int m, int n, const double *alpha,
        #   const double *x, int incx,
        #   const double *y, int incy,
        #   double *A, int lda)
        function ger!(alpha::$elty,
                      x::CudaVector{$elty},
                      y::CudaVector{$elty},
                      A::CudaVMPitch{$elty})
            m, n = size(A)
            m == length(x) || throw(DimensionMismatch(""))
            n == length(y) || throw(DimensionMismatch(""))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            statuscheck(ccall(($(string(fname)),libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Cint, Ptr{$elty},
                              Ptr{$elty}, Cint, Ptr{$elty}, Cint, Ptr{$elty},
                              Cint), cublashandle[1], m, n, [alpha], x, incx, y,
                              incy, A, lda))
            A
        end
    end
end

