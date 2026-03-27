"""Repository paths for checkpoints (relative to repo root)."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = _REPO_ROOT / "models" / "checkpoints"


def ensure_checkpoints_dir() -> Path:
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINTS_DIR


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to repo root, or pass through absolute paths."""
    p = Path(path)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p
