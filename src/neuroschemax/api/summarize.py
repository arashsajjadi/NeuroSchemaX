"""Public summarisation API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ir.semantic_ir import SemanticArchitecture
from ..visualization.text import markdown_summary, text_summary


def _resolve(arch_or_source: SemanticArchitecture | str | Path | Any) -> SemanticArchitecture:
    if isinstance(arch_or_source, SemanticArchitecture):
        return arch_or_source
    from .parse import parse_model
    return parse_model(arch_or_source)


def summarize_model(
    source: SemanticArchitecture | str | Path | Any,
    format: str = "text",
) -> str:
    """Return a human-readable summary of *source*.

    Args:
        source: A model source or an already-parsed ``SemanticArchitecture``.
        format: ``"text"`` (default) or ``"markdown"``.
    """
    arch = _resolve(source)
    if format == "markdown":
        return markdown_summary(arch)
    if format == "text":
        return text_summary(arch)
    raise ValueError(f"Unknown summary format: {format!r}")
