"""Seeding and device selection, so every run is reproducible.

In plain words: Two small helpers: make random numbers repeatable, and pick the device (CPU or GPU)
to compute on.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fix every random number generator so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """Choose where to compute. "auto" picks an NVIDIA GPU, else the Apple GPU (mps), else the CPU.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
