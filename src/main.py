import CudaKernels
import CudaKernels.build.cublas as cublas
from benchmark import measure_execution_time_ms
from optimize import optimize 
import torch 

from pprint import pprint

def old():
    m, n, k = 2048, 2048, 2048

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    a = torch.arange(m * k, dtype=torch.float, device="cuda").reshape(m,k)
    b = torch.arange(k * n, dtype=torch.float, device="cuda").reshape(k,n)
    c = torch.arange(m * n, dtype=torch.float, device="cuda").reshape(m,n)

    t_torch  = measure_execution_time_ms("torch", torch.matmul, (a, b))
    t_cublas = measure_execution_time_ms("cublas", cublas.launch_gemm_float, (m,n,k, 1, a, m, b, n, 0, c, k))
    params = CudaKernels.GemmParams[float](m=m,n=n,k=k, alpha=1, A=a, lda=m, B=b, ldb=n, beta=0, C=c, ldc=k)
    print(params)
    t_custom = measure_execution_time_ms("custom", CudaKernels.launch_gemm_02, params)

    CudaKernels.launch_gemm_02(CudaKernels.GemmParams[float](m=m, n=n, k=k, 
                                                            alpha=1, A=a, lda=m, 
                                                            B=b, ldb=n, beta=0, 
                                                            C=c, ldc=k))

    print(f"Custom: {t_custom:.4f} ms")
    print(f"Torch:  {t_torch:.4f} ms")
    print(f"cuBLAS: {t_cublas:.4f} ms")

    print(f"{t_cublas / t_custom * 100 :.4f}% of cublas performance")

kernels = [getattr(CudaKernels, name) for name in CudaKernels.__all__ if name.startswith("launch_")]
optimizer_results = [optimize(kernel) for kernel in kernels]

pprint(optimizer_results)