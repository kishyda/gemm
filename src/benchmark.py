import torch
import dataclasses

def measure_execution_time_ms(name, func, args):
    def call():
        if dataclasses.is_dataclass(args):
            return func(args)
        return func(*args)

    # 1. WARMUP
    # for _ in range(10):
    #     call()  # was func(*args)
    # torch.cuda.synchronize()

    # 2. TIMING
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    call()
    end_event.record()

    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event)

def calculate_tflops(*, m, n, k, elapsed_ms):
    total_operations = 2 * m * n * k
    tflops = total_operations / (elapsed_ms * 1e-3) / 1e12
    return tflops
