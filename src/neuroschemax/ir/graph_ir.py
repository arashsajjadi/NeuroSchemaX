"""Low-level graph intermediate representation.

This is the first stage of the pipeline: raw nodes, edges, and metadata
produced by ingestion adapters before any semantic interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TensorInfo:
    """Shape and dtype metadata for a single tensor."""

    name: str = ""
    shape: list[int | str] = field(default_factory=list)
    dtype: str = ""


@dataclass
class GraphNode:
    """A single operation / layer in the raw graph."""

    id: str = ""
    op_type: str = ""
    name: str = ""
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    input_shapes: list[list[int | str]] = field(default_factory=list)
    output_shapes: list[list[int | str]] = field(default_factory=list)


@dataclass
class GraphEdge:
    """A directed connection between two graph nodes."""

    source_id: str = ""
    target_id: str = ""
    tensor_name: str = ""
    shape: list[int | str] = field(default_factory=list)


@dataclass
class GraphIR:
    """Complete low-level graph representation of a neural network.

    Produced by an ingestion adapter and consumed by the normalisation pipeline.
    """

    model_name: str = ""
    framework: str = ""
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    inputs: list[TensorInfo] = field(default_factory=list)
    outputs: list[TensorInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def node_by_id(self, node_id: str) -> GraphNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None
