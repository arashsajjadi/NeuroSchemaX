"""Debug JSON exporter.

Emits a verbose structural dump including raw op attributes, all
shapes, confidence, and metadata — intended for debugging and
regression testing, not for publication.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ir.graph_ir import GraphIR
from ..ir.semantic_ir import SemanticArchitecture


def build_debug_dict(
    arch: SemanticArchitecture,
    graph: GraphIR | None = None,
) -> dict[str, Any]:
    """Build a verbose debug dict for *arch* (and optionally the raw graph)."""
    data: dict[str, Any] = {
        "model_name": arch.model_name,
        "framework": arch.framework,
        "recommended_family": (
            arch.recommended_family.value if arch.recommended_family else None
        ),
        "family_confidence": arch.family_confidence.value,
        "metadata": arch.metadata,
        "warnings": list(arch.warnings),
        "input_shapes": arch.input_shapes,
        "output_shapes": arch.output_shapes,
        "layers": [
            {
                "id": lay.id,
                "name": lay.name,
                "kind": lay.kind.name.lower(),
                "units": lay.units,
                "channels": lay.channels,
                "kernel_size": lay.kernel_size,
                "stride": lay.stride,
                "padding": lay.padding,
                "activation": lay.activation,
                "input_shape": lay.input_shape,
                "output_shape": lay.output_shape,
                "attributes": _sanitise(lay.attributes),
                "confidence": lay.confidence.value,
            }
            for lay in arch.layers
        ],
        "blocks": [
            {"id": b.id, "name": b.name, "layer_ids": b.layer_ids, "role": b.role}
            for b in arch.blocks
        ],
        "skip_connections": [
            {
                "source": s.source_id,
                "target": s.target_id,
                "kind": s.kind,
                "confidence": s.confidence.value,
            }
            for s in arch.skip_connections
        ],
    }
    if graph is not None:
        data["raw_graph"] = {
            "node_count": graph.node_count,
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "tensor": e.tensor_name,
                    "shape": e.shape,
                }
                for e in graph.edges
            ],
        }
    return data


def _sanitise(value: Any) -> Any:
    """Convert non-JSON-serialisable values to strings."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_sanitise(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items()}
    return str(value)


def export_debug_json(
    arch: SemanticArchitecture,
    graph: GraphIR | None = None,
) -> str:
    return json.dumps(build_debug_dict(arch, graph), indent=2, ensure_ascii=False)


def save_debug_json(
    arch: SemanticArchitecture,
    path: str | Path,
    graph: GraphIR | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(export_debug_json(arch, graph), encoding="utf-8")
    return out
