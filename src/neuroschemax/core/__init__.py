"""Core types, enums, configuration, and validation for NeuroSchemaX."""

from .aliases import resolve_alias
from .config import RenderConfig
from .enums import (
    ConfidenceLevel,
    InputFormat,
    LabelDensity,
    LayerKind,
    LayoutMode,
    LineStyle,
    Orientation,
    OutputFormat,
    RenderFamily,
    Theme,
)
from .validation import validate_render_config

__all__ = [
    "ConfidenceLevel",
    "InputFormat",
    "LabelDensity",
    "LayoutMode",
    "LayerKind",
    "LineStyle",
    "Orientation",
    "OutputFormat",
    "RenderConfig",
    "RenderFamily",
    "Theme",
    "resolve_alias",
    "validate_render_config",
]
