#include "nanobind/nb_defs.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
__global__ void gemm_01(size_t m, size_t n, size_t k, T alpha, T const* A,
                         size_t lda, T const* B, size_t ldb, T beta, T* C,
                         size_t ldc)
{
    // Compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

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
void launch_gemm_01(size_t m, size_t n, size_t k, T const alpha,
                            T const* A, size_t lda, T const* B, size_t ldb,
                            T const beta, T* C, size_t ldc
                            )
{
    dim3 const block_dim{32U, 32U, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm_01<T><<<grid_dim, block_dim, 0U>>>(m, n, k, alpha, A, lda, B,
                                                     ldb, beta, C, ldc);
}

void function() {
    // Define matrix dimensions
    size_t m = 100, n = 100, k = 50;

    // Allocate and initialize host memory for matrices A, B, and C
    float *h_A = new float[m * k];
    float *h_B = new float[k * n];
    float *h_C = new float[m * n];

    for (size_t i = 0; i < m * k; ++i) h_A[i] = 1.0f; // Example initialization
    for (size_t i = 0; i < k * n; ++i) h_B[i] = 1.0f;
    for (size_t i = 0; i < m * n; ++i) h_C[i] = 0.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define alpha and beta scalars
    float alpha = 1.0f, beta = 0.0f;

    // Launch the kernel
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_gemm_01(m, n, k, alpha, d_A, k, d_B, n, beta, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t m_idx{0}; m_idx < m; m_idx++) {
        for (size_t n_idx{0}; n_idx < n; n_idx++) {
            // std::cout << h_C[m_idx * m + n_idx];
        }
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaStreamDestroy(stream);

    // return 0;
}


// Explicit instantiation.
template void launch_gemm_01<float>(size_t m, size_t n, size_t k,
                                            float const alpha, float const* A,
                                            size_t lda, float const* B,
                                            size_t ldb, float const beta,
                                            float* C, size_t ldc);
template void launch_gemm_01<double>(size_t m, size_t n, size_t k,
                                             double const alpha,
                                             double const* A, size_t lda,
                                             double const* B, size_t ldb,
                                             double const beta, double* C,
                                             size_t ldc);

NB_MODULE(gemm_01, m) {
    m.def("launch_gemm_float", [](size_t m, size_t n, size_t k, float alpha, nb::ndarray<>& A, size_t lda, nb::ndarray<>& B, size_t ldb, float beta, nb::ndarray<>& C, size_t ldc){
        launch_gemm_01<float>(m, n, k, alpha, (float*) A.data(), lda, (float*) B.data(), ldb, beta, (float*) C.data(), ldc);
    }, "m"_a, "n"_a, "k"_a, "alpha"_a, "A"_a, "lda"_a, "B"_a, "ldb"_a, "beta"_a, "C"_a, "ldc"_a);
}