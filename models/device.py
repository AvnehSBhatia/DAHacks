"""
Torch device selection: CUDA (if present) → MPS (Apple Silicon) → CPU.

Override with env ``TORCH_DEVICE=cuda|mps|cpu``.
"""

from __future__ import annotations

import os

import torch


def autodetect_device_str() -> str:
    forced = os.environ.get("TORCH_DEVICE", "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_torch_device(explicit: str | None = None) -> torch.device:
    if explicit is not None:
        return torch.device(explicit)
    return torch.device(autodetect_device_str())


def qr_reduced(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``torch.linalg.qr(t, mode='reduced')`` with a CPU fallback on MPS, where QR is
    not implemented (see PyTorch MPS coverage).
    """
    if t.device.type == "mps":
        q, r = torch.linalg.qr(t.cpu(), mode="reduced")
        return q.to(t.device), r.to(t.device)
    return torch.linalg.qr(t, mode="reduced")
