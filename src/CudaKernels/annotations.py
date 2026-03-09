import torch
from CudaKernels.build.gemm_01 import launch_gemm_float as launch_gemm_01
from CudaKernels.build.gemm_02 import launch_gemm_float as launch_gemm_02
from CudaKernels.build.gemm_03 import launch_gemm_float as launch_gemm_03
from CudaKernels.build.gemm_04 import launch_gemm_float as launch_gemm_04
from CudaKernels.build.gemm_05 import launch_gemm_float as launch_gemm_05
from CudaKernels.build.gemm_06 import launch_gemm_float as launch_gemm_06

def _annotate_gemm(fn):
    fn.__annotations__ = {
        "m": int, "n": int, "k": int,
        "alpha": float,
        "A": torch.Tensor, "lda": int,
        "B": torch.Tensor, "ldb": int,
        "beta": float,
        "C": torch.Tensor, "ldc": int,
        "return": None,
    }
    return fn

launch_gemm_01 = _annotate_gemm(launch_gemm_01)
launch_gemm_02 = _annotate_gemm(launch_gemm_02)
launch_gemm_03 = _annotate_gemm(launch_gemm_03)
launch_gemm_04 = _annotate_gemm(launch_gemm_04)
launch_gemm_05 = _annotate_gemm(launch_gemm_05)
launch_gemm_06 = _annotate_gemm(launch_gemm_06)