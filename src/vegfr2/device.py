"""CUDA device enforcement for training and screening."""

import torch


def get_device() -> torch.device:
    """Return the CUDA device; raise RuntimeError when CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU required: training/screening is CUDA-only "
            "(torch.cuda.is_available()=False). Run on a CUDA machine."
        )
    return torch.device('cuda')
