"""Public export API: paper JSON, debug JSON, NN-SVG spec JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exporters.debug_json import export_debug_json, save_debug_json
from ..exporters.nnsvg import export_nnsvg_spec as _export_spec
from ..exporters.nnsvg import save_nnsvg_spec as _save_spec
from ..exporters.paper_json import export_paper_json, save_paper_json
from ..ir.semantic_ir import SemanticArchitecture
from ..visualization.nnsvg_schema import NNSVGSpec


def _resolve(arch_or_source: SemanticArchitecture | str | Path | Any) -> SemanticArchitecture:
    if isinstance(arch_or_source, SemanticArchitecture):
        return arch_or_source
    from .parse import parse_model
    return parse_model(arch_or_source)


def export_paper_json_from(
    source: SemanticArchitecture | str | Path | Any,
) -> str:
    """Return the paper-JSON string for *source*."""
    return export_paper_json(_resolve(source))


def save_paper_json_from(
    path: str | Path,
    source: SemanticArchitecture | str | Path | Any,
) -> Path:
    """Save the paper-JSON for *source* to *path*."""
    return save_paper_json(_resolve(source), path)


def export_debug_json_from(
    source: SemanticArchitecture | str | Path | Any,
) -> str:
    """Return the debug-JSON string for *source*."""
    return export_debug_json(_resolve(source))


def save_debug_json_from(
    path: str | Path,
    source: SemanticArchitecture | str | Path | Any,
) -> Path:
    """Save the debug-JSON for *source* to *path*."""
    return save_debug_json(_resolve(source), path)


def export_nnsvg_spec(spec: NNSVGSpec) -> str:
    """Serialise *spec* to a JSON string."""
    return _export_spec(spec)


def save_nnsvg_spec(path: str | Path, spec: NNSVGSpec) -> Path:
    """Save *spec* to *path* as JSON."""
    return _save_spec(spec, path)
