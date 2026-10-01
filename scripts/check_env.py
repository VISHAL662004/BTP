"""Phase 1 environment check: libraries, tensor creation, small model on CPU and MPS."""
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LIBS = ["torch", "numpy", "scipy", "pandas", "sklearn", "SimpleITK", "nibabel", "yaml", "matplotlib", "pytest", "jupyter_core"]


def main() -> int:
    print(f"Python {platform.python_version()} on {platform.platform()}")
    failed = []
    for name in LIBS:
        try:
            m = importlib.import_module(name)
            print(f"  OK   {name:12s} {getattr(m, '__version__', '')}")
        except Exception as e:  # report every missing lib, then exit non-zero
            print(f"  FAIL {name:12s} {e}")
            failed.append(name)
    if failed:
        return 1

    import torch
    from torch import nn

    from src.utils.config import load_config

    cfg = load_config("configs/base.yaml")
    print("Config loaded: seed =", cfg["experiment"]["seed"])

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    else:
        print("MPS not available (CPU only)")
    for dev in devices:
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 1)).to(dev)
        x = torch.randn(2, 3, 64, 64, device=dev)
        out = model(x)
        out.sum().backward()
        print(f"  OK   forward/backward on {dev}: output {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
