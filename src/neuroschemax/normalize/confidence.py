"""Aggregate confidence information across the normalisation pipeline."""

from __future__ import annotations

from ..core.enums import ConfidenceLevel
from ..ir.semantic_ir import SemanticArchitecture


def overall_confidence(arch: SemanticArchitecture) -> ConfidenceLevel:
    """Compute a single aggregate confidence for the whole architecture."""
    if not arch.layers:
        return ConfidenceLevel.UNKNOWN

    levels = [lay.confidence for lay in arch.layers]
    unknown = sum(1 for lv in levels if lv == ConfidenceLevel.UNKNOWN)
    low = sum(1 for lv in levels if lv == ConfidenceLevel.LOW)
    total = len(levels)

    if unknown > total * 0.5:
        return ConfidenceLevel.UNKNOWN
    if (unknown + low) > total * 0.3:
        return ConfidenceLevel.LOW
    if low > 0 or unknown > 0:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.HIGH
