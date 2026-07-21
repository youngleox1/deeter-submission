import torch

from src.train import resolve_device


def test_auto_resolves_to_cuda_if_available_else_cpu():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def test_explicit_device_passes_through_unchanged():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
