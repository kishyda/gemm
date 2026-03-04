#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <tuple>

namespace nb = nanobind;

__global__ void vectorAdd(float *A, float *B, float *C, int N)
{
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform addition if within bounds
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int foo(int &in) { in *= 2; return std::sqrt(in); }

NB_MODULE(test, m) {
    m.def("inspect", [](const nb::ndarray<>& a) {
        printf("Array data pointer : %p\n", a.data());
        printf("Array dimension : %zu\n", a.ndim());
        for (size_t i = 0; i < a.ndim(); ++i) {
            printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
            printf("Array stride    [%zu] : %zd\n", i, a.stride(i));
        }
        printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
            int(a.device_type() == nb::device::cpu::value),
            int(a.device_type() == nb::device::cuda::value)
        );
        // printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
        //     a.dtype() == nb::dtype<int16_t>(),
        //     a.dtype() == nb::dtype<uint32_t>(),
        //     a.dtype() == nb::dtype<float>()
        // );
    });
    m.def("vectorAdd", [](const nb::ndarray<>& A, const nb::ndarray<>& B, nb::ndarray<>& C, size_t size) {
        float* a = (float* ) A.data();
        float* b = (float* ) B.data();
        float* c = (float* ) C.data();
        vectorAdd<<<size, 1>>>(a, b, c, size);
    });
}