"""Data structures describing an NN-SVG rendering specification.

An :class:`NNSVGSpec` is the final configuration object that gets serialised
to JSON and embedded in the generated HTML for the NN-SVG JavaScript to
consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.enums import RenderFamily


@dataclass
class NNSVGLayerSpec:
    """Specification for a single layer in the NN-SVG diagram."""

    # Common to all families
    layer_type: str = ""       # "input", "conv", "pool", "dense", "output", etc.
    label: str = ""
    units: int = 0             # FCNN: neuron count
    channels: int = 0          # LeNet/AlexNet: feature-map channels
    kernel_size: int = 0
    stride: int = 1
    feature_map_width: int = 0
    feature_map_height: int = 0
    color: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NNSVGSpec:
    """Complete NN-SVG rendering specification.

    This is the JSON-serialisable configuration that the NN-SVG JavaScript
    runtime consumes to draw the diagram.
    """

    family: RenderFamily = RenderFamily.FCNN
    layers: list[NNSVGLayerSpec] = field(default_factory=list)

    # Canvas
    width: int = 1200
    height: int = 700

    # Display options
    show_labels: bool = True
    show_shapes: bool = True
    title: str = ""
    edge_opacity: float = 0.4
    node_size: float = 1.0
    spacing: float = 1.0
    between_layers_spacing: float = 1.0
    font_family: str = "sans-serif"
    font_size: int = 12
    color_fill: str | None = None
    color_stroke: str | None = None

    # Optional subtitle (rendered below the title, subtle metadata line)
    subtitle: str = ""

    # Metadata (not rendered, but preserved in export)
    model_name: str = ""
    warnings: list[str] = field(default_factory=list)

    # Diagnostic mode: when True, the HTML renderer produces a structured
    # "this cannot be rendered exactly" card instead of running an NN-SVG
    # family.  Set by the mapper for transformer_mode="unsupported".
    diagnostic: dict[str, Any] | None = None

    # Legend: when True, a colour-key and fidelity note appears in HTML output.
    show_legend: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "family": self.family.value,
            "layers": [_layer_to_dict(lay) for lay in self.layers],
            "width": self.width,
            "height": self.height,
            "showLabels": self.show_labels,
            "showShapes": self.show_shapes,
            "title": self.title,
            "edgeOpacity": self.edge_opacity,
            "nodeSize": self.node_size,
            "spacing": self.spacing,
            "betweenLayersSpacing": self.between_layers_spacing,
            "fontFamily": self.font_family,
            "fontSize": self.font_size,
            "colorFill": self.color_fill,
            "colorStroke": self.color_stroke,
            "subtitle": self.subtitle,
            "modelName": self.model_name,
            "warnings": self.warnings,
            "diagnostic": self.diagnostic,
            "showLegend": self.show_legend,
        }


def _layer_to_dict(layer: NNSVGLayerSpec) -> dict[str, Any]:
    d: dict[str, Any] = {
        "layerType": layer.layer_type,
        "label": layer.label,
        "units": layer.units,
        "channels": layer.channels,
        "kernelSize": layer.kernel_size,
        "stride": layer.stride,
        "featureMapWidth": layer.feature_map_width,
        "featureMapHeight": layer.feature_map_height,
    }
    if layer.color:
        d["color"] = layer.color
    if layer.extra:
        d["extra"] = layer.extra
    return d
