"""Paper-oriented semantic JSON exporter.

Produces a compact, human-friendly JSON representation suitable for
inclusion in publications or dataset releases: layer kinds, shapes,
block groupings, and family recommendation — without low-level op
attributes or raw framework details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ir.semantic_ir import SemanticArchitecture


def build_paper_dict(arch: SemanticArchitecture) -> dict[str, Any]:
    """Build the paper-JSON dict for *arch*."""
    return {
        "model_name": arch.model_name,
        "framework": arch.framework,
        "recommended_family": (
            arch.recommended_family.value if arch.recommended_family else None
        ),
        "family_confidence": arch.family_confidence.value,
        "input_shapes": arch.input_shapes,
        "output_shapes": arch.output_shapes,
        "layer_count": arch.layer_count,
        "has_skip_connections": arch.has_skip_connections,
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
                "output_shape": lay.output_shape,
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
        "warnings": list(arch.warnings),
    }


def export_paper_json(arch: SemanticArchitecture) -> str:
    """Return the paper-JSON string for *arch*."""
    return json.dumps(build_paper_dict(arch), indent=2, ensure_ascii=False)


def save_paper_json(arch: SemanticArchitecture, path: str | Path) -> Path:
    """Write the paper-JSON for *arch* to *path*."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(export_paper_json(arch), encoding="utf-8")
    return out
