"""Validation helpers for configuration and input data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import ValidationError

_LABEL_MODES     = frozenset({"auto", "name", "compact", "shape", "full"})
_DETAIL_LEVELS   = frozenset({"auto", "summary", "full"})
_TRANSFORMER_MODES = frozenset({"block_summary", "unsupported"})
_APPROXIMATE_MODES = frozenset({"warn", "error", "allow"})


def require_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer, got {value!r}")
    return value


def require_in_range(name: str, value: float, lo: float, hi: float) -> float:
    if not (lo <= value <= hi):
        raise ValidationError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def require_file_exists(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise ValidationError(f"File not found: {p}")
    return p.resolve()


def _require_choice(name: str, value: str, choices: frozenset[str]) -> str:
    if value not in choices:
        valid = ", ".join(f"{c!r}" for c in sorted(choices))
        raise ValidationError(
            f"{name} must be one of {valid}, got {value!r}"
        )
    return value


def validate_render_config(cfg: Any) -> None:
    """Validate a :class:`RenderConfig` instance."""
    require_positive_int("width", cfg.width)
    require_positive_int("height", cfg.height)
    require_in_range("opacity", cfg.opacity, 0.0, 1.0)
    require_in_range("edge_opacity", cfg.edge_opacity, 0.0, 1.0)
    require_in_range("connection_density", cfg.connection_density, 0.0, 1.0)
    _require_choice("label_mode",        cfg.label_mode,        _LABEL_MODES)
    _require_choice("detail_level",      cfg.detail_level,      _DETAIL_LEVELS)
    _require_choice("transformer_mode",  cfg.transformer_mode,  _TRANSFORMER_MODES)
    _require_choice("approximate_mode",  cfg.approximate_mode,  _APPROXIMATE_MODES)
