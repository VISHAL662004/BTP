import torch


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {"parameters": total, "trainable": trainable, "nonzero": nonzero,
            "sparsity": 1 - nonzero / total, "model_size_mb": size_mb}
