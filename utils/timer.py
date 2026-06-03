import time

import torch


@torch.no_grad()
def measure_inference_time(model, input_tensor, warmup=20, repeat=100):
    """Measure average and std single forward time in seconds."""
    model.eval()
    for _ in range(warmup):
        _ = model(input_tensor)
    if input_tensor.device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        if input_tensor.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(input_tensor)
        if input_tensor.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    values = torch.tensor(times)
    return float(values.mean().item()), float(values.std(unbiased=False).item())
