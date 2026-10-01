"""Device selection: MPS where supported, CPU fallback. Never assumes CUDA."""
import torch


def get_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
