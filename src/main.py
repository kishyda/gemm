import time
import CudaKernels.build.gemm_02 as gemm_02 
import CudaKernels.build.test as test 
import CudaKernels.build.cublas as cublas

import numpy as np
import torch 

m, n, k = 2048, 2048, 2048

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

a = torch.arange(m * k, dtype=torch.float, device="cuda").reshape(m,k)
b = torch.arange(k * n, dtype=torch.float, device="cuda").reshape(k,n)
c = torch.arange(m * n, dtype=torch.float, device="cuda").reshape(m,n)
# a = torch.arange(4_194_304, dtype=torch.float, device="cuda").reshape(2048,2048)
# b = torch.arange(4_194_304, dtype=torch.float, device="cuda").reshape(2048,2048)
# c = torch.arange(4_194_304, dtype=torch.float, device="cuda").reshape(2048,2048)

# test.vectorAdd(a, b, c, 256)

def benchmark(name, func, args):
    # 1. WARMUP: Run 10 times to let the GPU clocks ramp up
    for _ in range(10):
        func(*args)
    torch.cuda.synchronize()

    # 2. ACTUAL TIMING: Use CUDA Events (not perf_counter)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    func(*args)
    end_event.record()
    
    # 3. SYNC: Wait for the specific end event
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event)

t_custom = benchmark("custom", gemm_02.launch_gemm_float, (m,n,k, 1, a, m, b, n, 0, c, k))
t_torch  = benchmark("torch", torch.matmul, (a, b))
t_cublas = benchmark("cublas", cublas.launch_gemm_float, (m,n,k, 1, a, m, b, n, 0, c, k))

print(f"Custom: {t_custom:.4f} ms")
print(f"Torch:  {t_torch:.4f} ms")
print(f"cuBLAS: {t_cublas:.4f} ms")

print(f"{t_cublas / t_custom:.4f}")

# begin = time.perf_counter()
# gemm_02.launch_gemm_float(m,n,k, 1, a, m, b, n, 0, c, k)
# torch.cuda.synchronize()
# end = time.perf_counter() 
# time_for_custom_gemm = end - begin

# begin = time.perf_counter()
# torch.matmul(a, b)
# torch.cuda.synchronize()
# end = time.perf_counter()

# time_for_torch_gem = end - begin

# begin = time.perf_counter()
# cublas.launch_gemm_float(m,n,k, 1, a, m, b, n, 0, c, k)
# torch.cuda.synchronize()
# end = time.perf_counter()
# time_for_cublas_gem = end - begin


# print(f"custom: {time_for_custom_gemm}\ntorch: {time_for_torch_gem}\ncublas: {time_for_cublas_gem}")
# print(f"Within {time_for_cublas_gem / time_for_custom_gemm}")