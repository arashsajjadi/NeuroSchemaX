"""Export an NN-SVG rendering spec to JSON.

This is useful for reproducibility: the exact spec fed to the NN-SVG
JavaScript runtime can be saved and replayed without re-running the
whole pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..visualization.nnsvg_schema import NNSVGSpec


def export_nnsvg_spec(spec: NNSVGSpec) -> str:
    """Return the NN-SVG spec as a JSON string."""
    return json.dumps(spec.to_dict(), indent=2, ensure_ascii=False)


def save_nnsvg_spec(spec: NNSVGSpec, path: str | Path) -> Path:
    """Save an NN-SVG spec to *path*."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(export_nnsvg_spec(spec), encoding="utf-8")
    return out
