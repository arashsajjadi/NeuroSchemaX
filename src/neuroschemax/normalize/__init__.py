"""Normalisation pipeline: GraphIR → SemanticArchitecture."""

from __future__ import annotations

from ..ir.graph_ir import GraphIR
from ..ir.semantic_ir import SemanticArchitecture, SemanticLayer
from .attr_extractor import (
    extract_activation,
    extract_channels,
    extract_kernel_size,
    extract_padding,
    extract_stride,
    extract_units,
)
from .block_fuser import fuse_blocks
from .branch_analyzer import compute_complexity_hint
from .family_recognizer import recommend_family
from .op_normalizer import normalize_op
from .shape_normalizer import (
    infer_channels_from_shape,
    infer_units_from_shape,
    propagate_shapes,
)
from .skip_detector import detect_skip_connections


def normalize(graph: GraphIR) -> SemanticArchitecture:
    """Run the full normalisation pipeline on a :class:`GraphIR`.

    Returns a :class:`SemanticArchitecture` with layers, blocks, skip
    connections, a recommended rendering family, and diagnostics.
    """
    # 1. Convert each graph node into a SemanticLayer
    layers: list[SemanticLayer] = []
    for node in graph.nodes:
        kind, confidence = normalize_op(node.op_type)
        layer = SemanticLayer(
            id=node.id,
            name=node.name,
            kind=kind,
            units=extract_units(node.attributes, kind),
            channels=extract_channels(node.attributes, kind),
            kernel_size=extract_kernel_size(node.attributes),
            stride=extract_stride(node.attributes),
            padding=extract_padding(node.attributes),
            activation=extract_activation(node.attributes),
            input_shape=node.input_shapes[0] if node.input_shapes else [],
            output_shape=node.output_shapes[0] if node.output_shapes else [],
            attributes=node.attributes,
            confidence=confidence,
        )
        layers.append(layer)

    # 2. Shape propagation and inference
    propagate_shapes(layers)
    for layer in layers:
        infer_units_from_shape(layer)
        infer_channels_from_shape(layer)

    # 3. Block fusion
    blocks = fuse_blocks(layers)

    # 4. Skip-connection detection
    skips = detect_skip_connections(graph)

    # 5. Family recommendation
    arch = SemanticArchitecture(
        model_name=graph.model_name,
        framework=graph.framework,
        layers=layers,
        blocks=blocks,
        skip_connections=skips,
        input_shapes=[inp.shape for inp in graph.inputs],
        output_shapes=[out.shape for out in graph.outputs],
        metadata=dict(graph.metadata),
    )

    family, fam_conf, warnings = recommend_family(arch)
    arch.recommended_family = family
    arch.family_confidence = fam_conf
    arch.warnings.extend(warnings)

    complexity = compute_complexity_hint(graph)
    arch.metadata["complexity_hint"] = complexity

    return arch


__all__ = ["normalize"]
