import torch
from torch.utils.cpp_extension import load
from pathlib import Path
from typing import TYPE_CHECKING
import CudaKernels.build.gemm_01 as gemm_01 
import CudaKernels.build.gemm_02 as gemm_02 

print("Hello")