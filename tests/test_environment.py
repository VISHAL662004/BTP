import torch

from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def test_config_loads():
    cfg = load_config("configs/base.yaml")
    assert cfg["data"]["input_slices"] == 3


def test_seed_reproducible():
    set_seed(1)
    a = torch.randn(3)
    set_seed(1)
    assert torch.equal(a, torch.randn(3))


def test_device_falls_back_to_cpu():
    assert get_device("cpu").type == "cpu"
    assert get_device("mps").type in {"mps", "cpu"}
