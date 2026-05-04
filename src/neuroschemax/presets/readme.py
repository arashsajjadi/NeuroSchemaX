"""README preset: compact, friendly dimensions for embedding in docs."""

from __future__ import annotations

from ..core.config import RenderConfig
from ..core.enums import LabelDensity, LayoutMode, LineStyle, Theme


def readme_preset() -> RenderConfig:
    return RenderConfig(
        theme=Theme.README,
        layout_mode=LayoutMode.COMPACT,
        width=900,
        height=400,
        show_labels=True,
        show_shapes=False,
        label_density=LabelDensity.MINIMAL,
        line_style=LineStyle.SOLID,
        line_width=1.0,
        edge_opacity=0.4,
        font_family="sans-serif",
        font_size=11,
        margin=25.0,
    )
