"""Validation helpers for configuration and input data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import ValidationError


def require_positive_int(name: str, value: Any) -> int:
    """Ensure *value* is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value!r}")
    return value


def require_in_range(name: str, value: float, lo: float, hi: float) -> float:
    """Ensure *value* is within [lo, hi]."""
    if not (lo <= value <= hi):
        raise ValidationError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def require_file_exists(path: str | Path) -> Path:
    """Ensure *path* points to an existing file and return a resolved Path."""
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f"File not found: {p}")
    return p.resolve()


def validate_render_config(cfg: Any) -> None:
    """Validate a :class:`RenderConfig` instance."""
    require_positive_int("width", cfg.width)
    require_positive_int("height", cfg.height)
    require_in_range("opacity", cfg.opacity, 0.0, 1.0)
    require_in_range("edge_opacity", cfg.edge_opacity, 0.0, 1.0)
    require_in_range("connection_density", cfg.connection_density, 0.0, 1.0)
