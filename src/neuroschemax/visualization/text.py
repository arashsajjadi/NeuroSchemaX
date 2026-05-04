"""Text and Markdown summaries of a :class:`SemanticArchitecture`."""

from __future__ import annotations

from ..core.enums import LayerKind
from ..ir.semantic_ir import SemanticArchitecture


def _shape_str(shape: list[int | str]) -> str:
    if not shape:
        return "?"
    return "x".join(str(d) for d in shape)


def _fmt_size(val: list[int] | int | None) -> str:
    """Format a kernel/stride size as NxM (never as a raw Python list)."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return "x".join(str(v) for v in val)
    return str(val)


def _layer_params(lay: SemanticArchitecture) -> str:
    """Build a compact params string for a single layer."""
    parts: list[str] = []
    if lay.kind == LayerKind.DENSE and lay.units is not None:
        parts.append(f"units={lay.units}")
    elif lay.kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV):
        if lay.channels:
            parts.append(f"ch={lay.channels}")
        if lay.kernel_size:
            parts.append(f"k={_fmt_size(lay.kernel_size)}")
        if lay.stride and any(s != 1 for s in (lay.stride if isinstance(lay.stride, list) else [lay.stride])):
            parts.append(f"s={_fmt_size(lay.stride)}")
    elif lay.kind in (LayerKind.POOL_MAX, LayerKind.POOL_AVG):
        if lay.kernel_size:
            parts.append(f"k={_fmt_size(lay.kernel_size)}")
        if lay.stride:
            parts.append(f"s={_fmt_size(lay.stride)}")
    return " ".join(parts)


def text_summary(arch: SemanticArchitecture) -> str:
    """Render a plain-text summary of the architecture."""
    lines: list[str] = []
    lines.append(f"Model: {arch.model_name}")
    lines.append(f"Framework: {arch.framework}")
    lines.append(f"Layers: {arch.layer_count}")
    if arch.recommended_family:
        lines.append(
            f"Recommended render: {arch.recommended_family.value} "
            f"(confidence: {arch.family_confidence.value})"
        )
    lines.append("")

    # Column widths
    w_name   = 24
    w_kind   = 16
    w_output = 20
    w_params = 20

    header = (
        f"{'#':>3}  "
        f"{'Name':<{w_name}} "
        f"{'Kind':<{w_kind}} "
        f"{'Output':<{w_output}} "
        f"{'Params':<{w_params}}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for i, lay in enumerate(arch.layers):
        params = _layer_params(lay)
        lines.append(
            f"{i:>3}  "
            f"{lay.name[:w_name]:<{w_name}} "
            f"{lay.kind.name.lower()[:w_kind]:<{w_kind}} "
            f"{_shape_str(lay.output_shape)[:w_output]:<{w_output}} "
            f"{params[:w_params]:<{w_params}}"
        )

    if arch.skip_connections:
        lines.append("")
        lines.append(f"Skip connections: {len(arch.skip_connections)}")
        for skip in arch.skip_connections:
            lines.append(f"  {skip.source_id} -> {skip.target_id} ({skip.kind})")

    if arch.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in arch.warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


def markdown_summary(arch: SemanticArchitecture) -> str:
    """Render a Markdown summary of the architecture."""
    lines: list[str] = []
    lines.append(f"# {arch.model_name}")
    lines.append("")
    lines.append(f"- **Framework:** `{arch.framework}`")
    lines.append(f"- **Layers:** {arch.layer_count}")
    if arch.recommended_family:
        lines.append(
            f"- **Recommended render:** `{arch.recommended_family.value}` "
            f"(confidence: {arch.family_confidence.value})"
        )
    if arch.has_skip_connections:
        lines.append(f"- **Skip connections:** {len(arch.skip_connections)}")
    lines.append("")
    lines.append("## Layers")
    lines.append("")
    lines.append("| # | Name | Kind | Output shape | Params |")
    lines.append("|---|------|------|--------------|--------|")
    for i, lay in enumerate(arch.layers):
        params = _layer_params(lay)
        lines.append(
            f"| {i} | `{lay.name}` | {lay.kind.name.lower()} | "
            f"`{_shape_str(lay.output_shape)}` | {params} |"
        )

    if arch.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for w in arch.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines) + "\n"
