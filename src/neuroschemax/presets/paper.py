"""Paper preset: clean, monochrome-friendly, publication-ready."""

from __future__ import annotations

from ..core.config import RenderConfig
from ..core.enums import LabelDensity, LayoutMode, LineStyle, Theme


def paper_preset() -> RenderConfig:
    return RenderConfig(
        theme=Theme.PAPER,
        layout_mode=LayoutMode.PRESENTATION,
        width=1400,
        height=700,
        show_labels=True,
        show_shapes=True,
        label_density=LabelDensity.MINIMAL,
        line_style=LineStyle.SOLID,
        line_width=1.0,
        edge_opacity=0.35,
        font_family="Helvetica, Arial, sans-serif",
        font_size=12,
        margin=40.0,
    )
