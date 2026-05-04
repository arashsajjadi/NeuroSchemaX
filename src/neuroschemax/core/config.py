"""Render configuration for NeuroSchemaX visualisations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import (
    LabelDensity,
    LayoutMode,
    LineStyle,
    Orientation,
    RenderFamily,
    Theme,
)


@dataclass
class RenderConfig:
    """Full configuration surface for a single rendering pass.

    All fields have sensible defaults so callers can create a config with
    ``RenderConfig()`` and override only what they need.
    """

    # ── diagram family & theme ──────────────────────────────────────────
    style: RenderFamily | None = None  # None = auto-detect
    theme: Theme = Theme.PAPER
    layout_mode: LayoutMode = LayoutMode.PRESENTATION

    # ── canvas ──────────────────────────────────────────────────────────
    width: int = 1200
    height: int = 700
    orientation: Orientation = Orientation.HORIZONTAL

    # ── labels & text ───────────────────────────────────────────────────
    show_labels: bool = True
    show_shapes: bool = True
    label_density: LabelDensity = LabelDensity.NORMAL
    title: str | None = None
    subtitle: str | None = None
    show_legend: bool = False
    font_family: str = "sans-serif"
    font_size: int = 12

    # ── line / connection style ─────────────────────────────────────────
    line_style: LineStyle = LineStyle.SOLID
    line_width: float = 1.0
    arrow_style: str = "default"
    connection_density: float = 1.0

    # ── node geometry ───────────────────────────────────────────────────
    node_size: float = 1.0
    spacing: float = 1.0
    margin: float = 30.0
    border_radius: float = 4.0
    between_layers_spacing: float = 1.0

    # ── colour ──────────────────────────────────────────────────────────
    color_fill: str | None = None
    color_stroke: str | None = None
    opacity: float = 1.0
    edge_opacity: float = 0.4

    # ── 3D / depth ──────────────────────────────────────────────────────
    pseudo_3d: bool = False
    depth_scale: float = 1.0

    # ── highlighting ────────────────────────────────────────────────────
    highlight_layers: list[str] = field(default_factory=list)

    # ── determinism ─────────────────────────────────────────────────────
    seed: int | None = None

    # ── extra pass-through ──────────────────────────────────────────────
    options: dict[str, Any] = field(default_factory=dict)

    # ── helpers ─────────────────────────────────────────────────────────

    def merge(self, overrides: dict[str, Any]) -> RenderConfig:
        """Return a new config with *overrides* applied on top of self."""
        data = {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}
        data.update(overrides)
        return RenderConfig(**data)
