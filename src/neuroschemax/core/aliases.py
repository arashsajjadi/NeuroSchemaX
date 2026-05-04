"""Alias resolution for flexible configuration values.

Allows users to specify configuration values as strings, integers, or enum
members and have them resolved consistently to canonical enum values.
"""

from __future__ import annotations

from typing import Any, TypeVar

from .enums import (
    LabelDensity,
    LayoutMode,
    LineStyle,
    Orientation,
    OutputFormat,
    RenderFamily,
    Theme,
)

T = TypeVar("T")

# ── Alias tables ────────────────────────────────────────────────────────────

_RENDER_FAMILY_ALIASES: dict[str | int, RenderFamily] = {
    "fcnn": RenderFamily.FCNN,
    "fc": RenderFamily.FCNN,
    "mlp": RenderFamily.FCNN,
    "dense": RenderFamily.FCNN,
    0: RenderFamily.FCNN,
    "lenet": RenderFamily.LENET,
    "cnn": RenderFamily.LENET,
    "small_cnn": RenderFamily.LENET,
    1: RenderFamily.LENET,
    "alexnet": RenderFamily.ALEXNET,
    "deep_cnn": RenderFamily.ALEXNET,
    "vgg": RenderFamily.ALEXNET,
    2: RenderFamily.ALEXNET,
}

_THEME_ALIASES: dict[str | int, Theme] = {
    "paper": Theme.PAPER,
    "publication": Theme.PAPER,
    0: Theme.PAPER,
    "thesis": Theme.THESIS,
    "academic": Theme.THESIS,
    1: Theme.THESIS,
    "debug": Theme.DEBUG,
    "dev": Theme.DEBUG,
    2: Theme.DEBUG,
    "readme": Theme.README,
    "docs": Theme.README,
    3: Theme.README,
}

_LINE_STYLE_ALIASES: dict[str | int, LineStyle] = {
    "solid": LineStyle.SOLID,
    0: LineStyle.SOLID,
    "dashed": LineStyle.DASHED,
    1: LineStyle.DASHED,
    "dotted": LineStyle.DOTTED,
    2: LineStyle.DOTTED,
}

_ORIENTATION_ALIASES: dict[str | int, Orientation] = {
    "horizontal": Orientation.HORIZONTAL,
    "h": Orientation.HORIZONTAL,
    0: Orientation.HORIZONTAL,
    "vertical": Orientation.VERTICAL,
    "v": Orientation.VERTICAL,
    1: Orientation.VERTICAL,
}

_LAYOUT_MODE_ALIASES: dict[str | int, LayoutMode] = {
    "compact": LayoutMode.COMPACT,
    0: LayoutMode.COMPACT,
    "presentation": LayoutMode.PRESENTATION,
    "full": LayoutMode.PRESENTATION,
    1: LayoutMode.PRESENTATION,
}

_LABEL_DENSITY_ALIASES: dict[str | int, LabelDensity] = {
    "none": LabelDensity.NONE,
    0: LabelDensity.NONE,
    "minimal": LabelDensity.MINIMAL,
    "min": LabelDensity.MINIMAL,
    1: LabelDensity.MINIMAL,
    "normal": LabelDensity.NORMAL,
    2: LabelDensity.NORMAL,
    "verbose": LabelDensity.VERBOSE,
    "all": LabelDensity.VERBOSE,
    3: LabelDensity.VERBOSE,
}

_OUTPUT_FORMAT_ALIASES: dict[str | int, OutputFormat] = {
    "html": OutputFormat.HTML,
    0: OutputFormat.HTML,
    "svg": OutputFormat.SVG,
    1: OutputFormat.SVG,
    "png": OutputFormat.PNG,
    2: OutputFormat.PNG,
    "nnsvg_json": OutputFormat.NNSVG_JSON,
    "nnsvg": OutputFormat.NNSVG_JSON,
    3: OutputFormat.NNSVG_JSON,
    "paper_json": OutputFormat.PAPER_JSON,
    "paper": OutputFormat.PAPER_JSON,
    4: OutputFormat.PAPER_JSON,
    "debug_json": OutputFormat.DEBUG_JSON,
    5: OutputFormat.DEBUG_JSON,
    "text": OutputFormat.TEXT,
    "txt": OutputFormat.TEXT,
    6: OutputFormat.TEXT,
    "markdown": OutputFormat.MARKDOWN,
    "md": OutputFormat.MARKDOWN,
    7: OutputFormat.MARKDOWN,
}

# ── Dispatch table ──────────────────────────────────────────────────────────

_ALIAS_TABLES: dict[type, dict[str | int, Any]] = {
    RenderFamily: _RENDER_FAMILY_ALIASES,
    Theme: _THEME_ALIASES,
    LineStyle: _LINE_STYLE_ALIASES,
    Orientation: _ORIENTATION_ALIASES,
    LayoutMode: _LAYOUT_MODE_ALIASES,
    LabelDensity: _LABEL_DENSITY_ALIASES,
    OutputFormat: _OUTPUT_FORMAT_ALIASES,
}

# ── Public resolver ─────────────────────────────────────────────────────────


def resolve_alias(enum_type: type[T], value: Any) -> T:
    """Resolve a string, int, or enum value to the canonical enum member.

    Args:
        enum_type: Target enum class.
        value: Raw value — may be a string alias, integer alias, or enum member.

    Returns:
        The resolved enum member.

    Raises:
        ValueError: If the value cannot be resolved.
    """
    if isinstance(value, enum_type):
        return value  # type: ignore[return-value]

    table = _ALIAS_TABLES.get(enum_type)
    if table is None:
        raise ValueError(f"No alias table registered for {enum_type.__name__}")

    key = value.lower().strip() if isinstance(value, str) else value
    try:
        return table[key]
    except KeyError:
        valid = sorted({k for k in table if isinstance(k, str)})
        raise ValueError(
            f"Cannot resolve {value!r} to {enum_type.__name__}. "
            f"Valid aliases: {', '.join(valid)}"
        ) from None
