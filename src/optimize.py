from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from typing import Callable, Any
import torch

from CudaKernels import GemmParams
from benchmark import measure_execution_time_ms, calculate_tflops

# Define the search space
space = [
    Integer(256, 4096, name="m"),
    Integer(256, 4096, name="n"),
    Integer(256, 4096, name="k"),
]

def optimize(kernel: Callable[[], None]):
    @use_named_args(space)
    def objective(m: int, n: int, k: int):
        m, n, k = [x - x % 4 or 4 for x in [m, n, k]]
        p = GemmParams(
            m=m, n=n, k=k,
            alpha=1.0,
            A=torch.randn(m, k, dtype=torch.float32, device="cuda"),
            lda=k,
            B=torch.randn(k, n, dtype=torch.float32, device="cuda"),
            ldb=n,
            beta=0.0,
            C=torch.zeros(m, n, dtype=torch.float32, device="cuda"),
            ldc=n,
        )
        execution_time_ms = measure_execution_time_ms("", kernel, p)  # your benchmark function
        tflops = calculate_tflops(m=m, n=n, k=k, elapsed_ms=execution_time_ms)
        return -tflops

    result = gp_minimize(
        objective,
        space,
        n_calls=100,       # total evaluations
        n_initial_points=10,  # random exploration before fitting GP
        verbose=True,
    )

    print(result.x)
    return {"Kernel: ": kernel.__name__}, {"TFLOPS: ": float(-result.fun)}, {"Parameters: ": (*map(int, result.x),)}, {"Data: ": result }