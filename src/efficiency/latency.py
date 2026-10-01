import time

import torch


@torch.no_grad()
def measure_latency(model, input_shape, device="cpu", warmup: int = 10, runs: int = 50) -> dict:
    """Mean/std forward latency in ms for a fixed input shape; documents device and batch."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    x = torch.randn(*input_shape, device=dev)

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    for _ in range(warmup):
        model(x)
    sync()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        model(x)
        sync()
        ts.append((time.perf_counter() - t) * 1e3)
    t = torch.tensor(ts)
    return {"device": str(dev), "input_shape": list(input_shape), "warmup": warmup, "runs": runs,
            "latency_ms_mean": float(t.mean()), "latency_ms_std": float(t.std()),
            "throughput_samples_per_s": float(input_shape[0] / (t.mean() / 1e3))}
