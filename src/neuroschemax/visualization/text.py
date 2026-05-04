"""Text and Markdown summaries of a :class:`SemanticArchitecture`."""

from __future__ import annotations

from ..core.enums import LayerKind
from ..ir.semantic_ir import SemanticArchitecture


def _shape_str(shape: list[int | str]) -> str:
    if not shape:
        return "?"
    return "x".join(str(d) for d in shape)


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

    header = f"{'#':>3}  {'Name':<24} {'Kind':<16} {'Output':<20} {'Params':<12}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, lay in enumerate(arch.layers):
        params = ""
        if lay.kind == LayerKind.DENSE and lay.units is not None:
            params = f"units={lay.units}"
        elif lay.kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV):
            parts = []
            if lay.channels:
                parts.append(f"ch={lay.channels}")
            if lay.kernel_size:
                parts.append(f"k={lay.kernel_size}")
            if lay.stride:
                parts.append(f"s={lay.stride}")
            params = " ".join(parts)
        elif lay.kind in (LayerKind.POOL_MAX, LayerKind.POOL_AVG) and lay.kernel_size:
            params = f"k={lay.kernel_size}"
        lines.append(
            f"{i:>3}  {lay.name[:24]:<24} {lay.kind.name.lower()[:16]:<16} "
            f"{_shape_str(lay.output_shape)[:20]:<20} {params[:12]:<12}"
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
    lines.append("| # | Name | Kind | Output shape |")
    lines.append("|---|------|------|--------------|")
    for i, lay in enumerate(arch.layers):
        lines.append(
            f"| {i} | `{lay.name}` | {lay.kind.name.lower()} | "
            f"`{_shape_str(lay.output_shape)}` |"
        )

    if arch.warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.append("")
        for w in arch.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines) + "\n"
