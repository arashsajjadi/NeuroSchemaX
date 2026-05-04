"""Semantic intermediate representation.

The semantic IR enriches the raw graph with architectural understanding:
layer kinds, block groupings, skip connections, and family hints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.enums import ConfidenceLevel, LayerKind, RenderFamily


@dataclass
class SemanticLayer:
    """A single semantically-understood layer."""

    id: str = ""
    name: str = ""
    kind: LayerKind = LayerKind.UNKNOWN
    units: int | None = None
    channels: int | None = None
    kernel_size: list[int] | None = None
    stride: list[int] | None = None
    padding: list[int] | None = None
    activation: str | None = None
    input_shape: list[int | str] = field(default_factory=list)
    output_shape: list[int | str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH


@dataclass
class SkipConnection:
    """A residual / skip link between two layers."""

    source_id: str = ""
    target_id: str = ""
    kind: str = "add"  # add, concat, multiply
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


@dataclass
class SemanticBlock:
    """A logical block / stage grouping layers together."""

    id: str = ""
    name: str = ""
    layer_ids: list[str] = field(default_factory=list)
    role: str = ""  # e.g. "encoder", "decoder", "backbone", "head"


@dataclass
class SemanticArchitecture:
    """The fully-normalised semantic view of a neural network.

    This is the central data structure that the visualisation mapper consumes.
    """

    model_name: str = ""
    framework: str = ""
    layers: list[SemanticLayer] = field(default_factory=list)
    blocks: list[SemanticBlock] = field(default_factory=list)
    skip_connections: list[SkipConnection] = field(default_factory=list)

    # Family recommendation (may be overridden by the user)
    recommended_family: RenderFamily | None = None
    family_confidence: ConfidenceLevel = ConfidenceLevel.UNKNOWN

    # Global shape info
    input_shapes: list[list[int | str]] = field(default_factory=list)
    output_shapes: list[list[int | str]] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def layer_by_id(self, layer_id: str) -> SemanticLayer | None:
        for lay in self.layers:
            if lay.id == layer_id:
                return lay
        return None

    @property
    def has_skip_connections(self) -> bool:
        return len(self.skip_connections) > 0

    @property
    def has_convolutions(self) -> bool:
        return any(
            lay.kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV)
            for lay in self.layers
        )

    @property
    def has_dense_only(self) -> bool:
        significant = [
            lay for lay in self.layers
            if lay.kind not in (
                LayerKind.INPUT, LayerKind.OUTPUT, LayerKind.ACTIVATION,
                LayerKind.DROPOUT, LayerKind.FLATTEN, LayerKind.RESHAPE,
                LayerKind.BATCH_NORM, LayerKind.LAYER_NORM, LayerKind.UNKNOWN,
            )
        ]
        return all(lay.kind == LayerKind.DENSE for lay in significant) if significant else False
