#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
__global__ void gemm_02(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    // Each thread compute
    // C[C_row_idx, C_col_idx] = alpha * A[C_row_idx, :] * B[:, C_col_idx] +
    // beta * C[C_row_idx, C_col_idx].
    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_02(size_t m, size_t n, size_t k, T const alpha,
                    T const* A, size_t lda, T const* B, size_t ldb,
                    T const beta, T* C, size_t ldc)
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_02<T><<<grid_dim, block_dim, 0U>>>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}



NB_MODULE(gemm_02, m) {
    m.def("launch_gemm_float", [](size_t m, size_t n, size_t k, float alpha, nb::ndarray<>& A, size_t lda, nb::ndarray<>& B, size_t ldb, float beta, nb::ndarray<>& C, size_t ldc){
        launch_gemm_02<float>(m, n, k, alpha, (float*) A.data(), lda, (float*) B.data(), ldb, beta, (float*) C.data(), ldc);
    }, "m"_a, "n"_a, "k"_a, "alpha"_a, "A"_a, "lda"_a, "B"_a, "ldb"_a, "beta"_a, "C"_a, "ldc"_a);
}
