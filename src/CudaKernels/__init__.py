import torch
from dataclasses import dataclass
from typing import NamedTuple
from CudaKernels.build.gemm_01 import launch_gemm_float as _launch_gemm_01
from CudaKernels.build.gemm_02 import launch_gemm_float as _launch_gemm_02
from CudaKernels.build.gemm_03 import launch_gemm_float as _launch_gemm_03
from CudaKernels.build.gemm_04 import launch_gemm_float as _launch_gemm_04
from CudaKernels.build.gemm_05 import launch_gemm_float as _launch_gemm_05
from CudaKernels.build.gemm_06 import launch_gemm_float as _launch_gemm_06
from CudaKernels.build.gemm_07 import launch_gemm_float as _launch_gemm_07

@dataclass
class GemmParams[T]:
    m: int
    n: int
    k: int
    alpha: T
    A: torch.Tensor
    lda: int
    B: torch.Tensor
    ldb: int
    beta: T
    C: torch.Tensor
    ldc: int

def launch_gemm_01(p: GemmParams) -> None:
    return _launch_gemm_01(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

def launch_gemm_02(p: GemmParams) -> None:
    return _launch_gemm_02(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

def launch_gemm_03(p: GemmParams) -> None:
    return _launch_gemm_03(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

def launch_gemm_04(p: GemmParams) -> None:
    return _launch_gemm_04(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

def launch_gemm_05(p: GemmParams) -> None:
    return _launch_gemm_05(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

def launch_gemm_06(p: GemmParams) -> None:
    return _launch_gemm_06(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

def launch_gemm_07(p: GemmParams) -> None:
    return _launch_gemm_07(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

# def launch_gemm_08(p: GemmParams) -> None:
#     return _launch_gemm_06(p.m, p.n, p.k, p.alpha, p.A, p.lda, p.B, p.ldb, p.beta, p.C, p.ldc) #type: ignore

__all__ = [
    "GemmParams",
    # "launch_gemm_01",
    # "launch_gemm_02",
    "launch_gemm_03",
    "launch_gemm_04",
    "launch_gemm_05",
    "launch_gemm_06",
    "launch_gemm_07",
]