import time

import torch


@torch.no_grad()
def measure_inference_time(model, input_tensor, warmup=20, repeat=100):
    """Measure average single forward time in seconds."""
    model.eval()
    for _ in range(warmup):
        _ = model(input_tensor)
    if input_tensor.device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(input_tensor)
    if input_tensor.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat

