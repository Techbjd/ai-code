"""Shared utilities for all PyG GNN models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


def save_checkpoint(model: nn.Module, path: str | Path) -> Path:
    """Save model checkpoint to disk.

    Args:
        model: Trained model with ``init_kwargs`` attribute.
        path: Destination file path.

    Returns:
        Resolved path of the saved checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "init_kwargs": model.init_kwargs},
        path,
    )
    return path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Load model checkpoint. Requires ``build_model`` to be called separately.

    This is a generic loader; each model file provides its own typed wrapper.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt
