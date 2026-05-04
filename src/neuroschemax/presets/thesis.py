"""Thesis preset: larger canvas and verbose labels for a printed page."""

from __future__ import annotations

from ..core.config import RenderConfig
from ..core.enums import LabelDensity, LayoutMode, LineStyle, Theme


def thesis_preset() -> RenderConfig:
    return RenderConfig(
        theme=Theme.THESIS,
        layout_mode=LayoutMode.PRESENTATION,
        width=1600,
        height=900,
        show_labels=True,
        show_shapes=True,
        label_density=LabelDensity.VERBOSE,
        line_style=LineStyle.SOLID,
        line_width=1.2,
        edge_opacity=0.4,
        font_family="serif",
        font_size=13,
        show_legend=True,
        margin=50.0,
    )
