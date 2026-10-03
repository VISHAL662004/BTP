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


def effective_flops(model: torch.nn.Module, input_shape) -> dict:
    """Dense FLOPs vs. *effective* FLOPs when zero weights of conv/linear layers are skipped
    (what a perfect sparse kernel would do; NOT what dense CPU/MPS kernels do). Per sample.
    effective = dense_total - sum_layers(dense_layer_flops * (1 - nonzero_fraction_of_layer_weights))."""
    model = model.to("cpu").eval()
    layer = []

    def hook(m, inp, out):
        if isinstance(m, torch.nn.Conv2d):
            macs = m.weight.numel() * out.shape[2] * out.shape[3]
        else:
            macs = m.weight.numel() * (inp[0].numel() // inp[0].shape[-1])
        layer.append((2 * macs / input_shape[0], float((m.weight != 0).float().mean())))

    dense = count_flops(model, input_shape)["flops_per_sample"]      # before hooks: avoid double counting
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    with torch.no_grad():
        model(torch.randn(*input_shape))
    for h in hs:
        h.remove()
    saved = sum(f * (1 - d) for f, d in layer)
    return {"flops_dense": dense, "flops_effective": dense - saved}
