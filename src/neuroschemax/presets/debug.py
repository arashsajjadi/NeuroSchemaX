"""Debug preset: emphasises every label and attribute for diagnosis."""

from __future__ import annotations

from ..core.config import RenderConfig
from ..core.enums import LabelDensity, LayoutMode, LineStyle, Theme


def debug_preset() -> RenderConfig:
    return RenderConfig(
        theme=Theme.DEBUG,
        layout_mode=LayoutMode.COMPACT,
        width=1800,
        height=800,
        show_labels=True,
        show_shapes=True,
        label_density=LabelDensity.VERBOSE,
        line_style=LineStyle.DASHED,
        line_width=1.0,
        edge_opacity=0.5,
        font_family="monospace",
        font_size=10,
        show_legend=True,
        margin=20.0,
    )
