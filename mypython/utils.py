"""Device selection: Apple GPU (MPS) -> CUDA -> CPU."""
import torch


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():          # Apple silicon GPU
        return torch.device("mps")
    if torch.cuda.is_available():                  # NVIDIA GPU
        return torch.device("cuda")
    return torch.device("cpu")
