import torch
from torch.utils.flop_counter import FlopCounterMode


def count_flops(model: torch.nn.Module, input_shape) -> dict:
    """FLOPs / MACs of one forward pass (batch given in input_shape) on CPU, counted by
    torch.utils.flop_counter (convolutions and matmuls; 1 MAC = 2 FLOPs). Reported per sample."""
    model = model.to("cpu").eval()
    x = torch.randn(*input_shape)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    flops = fc.get_total_flops() / input_shape[0]
    return {"flops_per_sample": float(flops), "macs_per_sample": float(flops / 2), "input_shape": list(input_shape)}
