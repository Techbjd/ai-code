"""Tests for vegfr2.device module (GPU-only enforcement)."""

from __future__ import annotations

import pytest
import torch

from vegfr2.device import get_device


def test_device_raises_on_non_cuda_machine():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError) as exc_info:
            get_device()
        assert "GPU required" in str(exc_info.value)
    else:
        d = get_device()
        assert d.type == "cuda"
