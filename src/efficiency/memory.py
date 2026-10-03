import torch


@torch.no_grad()
def activation_memory(model: torch.nn.Module, input_shape) -> dict:
    """Device-independent memory estimate for one inference forward pass.

    * parameters_mb : parameter + buffer storage.
    * activations_mb: sum of all leaf-module output tensors for ONE sample (upper bound on the
      activation footprint if nothing is freed; the real peak is lower). Used for relative comparison.
    * peak_live_mb  : largest (input + output) pair of any leaf module for one sample (lower bound
      on the working set when buffers are freed eagerly).
    Batch size in input_shape[0] is normalised to 1 sample."""
    model = model.to("cpu").eval()
    outs, peaks = [], []

    def hook(_, inp, out):
        o = out if isinstance(out, torch.Tensor) else out[0]
        i = inp[0] if isinstance(inp[0], torch.Tensor) else None
        n = input_shape[0]
        outs.append(o.numel() * o.element_size() / n)
        peaks.append((o.numel() * o.element_size() + (i.numel() * i.element_size() if i is not None else 0)) / n)

    hs = [m.register_forward_hook(hook) for m in model.modules() if not list(m.children())]
    model(torch.randn(*input_shape))
    for h in hs:
        h.remove()
    pb = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(b.numel() * b.element_size() for b in model.buffers())
    return {"parameters_mb": pb / 1e6, "activations_mb_per_sample": sum(outs) / 1e6, "peak_live_mb_per_sample": max(peaks) / 1e6}
