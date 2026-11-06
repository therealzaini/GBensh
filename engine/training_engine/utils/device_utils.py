"""Resolve the execution device at the boundary to PyTorch."""


def resolve_device(name: str):
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("training_device must be 'auto', 'cpu', or 'cuda'.")

    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    elif name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")

    return torch.device(name)
