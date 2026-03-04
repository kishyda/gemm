#include <nanobind/nanobind.h>
#include <cublas_v2.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(cublas, m) {
    m.def("launch_gemm_float", [](size_t m, size_t n, size_t k, float alpha, nb::ndarray<>& A, size_t lda, nb::ndarray<>& B, size_t ldb, float beta, nb::ndarray<>& C, size_t ldc){
        cublasHandle_t handle;
        cublasCreate(&handle); //
        cublasStatus_t status = cublasSgemm(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_N, 
            m, n, k, 
            &alpha, 
            (float*) A.data(), lda, 
            (float*) B.data(), ldb, 
            &beta, 
            (float*) C.data(), ldc
        ); 
    }, "m"_a, "n"_a, "k"_a, "alpha"_a, "A"_a, "lda"_a, "B"_a, "ldb"_a, "beta"_a, "C"_a, "ldc"_a);
}
