"""Recommend an NN-SVG rendering family based on architecture characteristics."""

from __future__ import annotations

from ..core.enums import ConfidenceLevel, LayerKind, RenderFamily
from ..ir.semantic_ir import SemanticArchitecture

_CONV_KINDS = frozenset({
    LayerKind.CONV,
    LayerKind.DEPTHWISE_CONV,
    LayerKind.TRANSPOSED_CONV,
})

_MERGE_KINDS = frozenset({
    LayerKind.ADD,
    LayerKind.CONCAT,
    LayerKind.MULTIPLY,
})

_RECURRENT_KINDS = frozenset({
    LayerKind.LSTM,
    LayerKind.GRU,
    LayerKind.RECURRENT,
})


def recommend_family(
    arch: SemanticArchitecture,
) -> tuple[RenderFamily, ConfidenceLevel, list[str]]:
    """Choose the best NN-SVG family and explain the reasoning.

    Returns:
        (family, confidence, warnings)

    NN-SVG supports three families: FCNN, LeNet, and AlexNet.  Models that do
    not fit cleanly (Transformers, ResNets, U-Nets, RNNs) are mapped to the
    nearest family with a warning and a reduced confidence score.  The exact
    graph is always preserved in the debug-JSON export.
    """
    warnings: list[str] = []

    conv_count = sum(1 for lay in arch.layers if lay.kind in _CONV_KINDS)
    dense_count = sum(1 for lay in arch.layers if lay.kind == LayerKind.DENSE)

    has_attention = any(lay.kind == LayerKind.ATTENTION for lay in arch.layers)
    has_recurrent = any(lay.kind in _RECURRENT_KINDS for lay in arch.layers)

    # Merge ops in the *layer list* (proxy for skip/branch in manual specs where
    # the graph adapter does not create edges and skip_detector finds nothing).
    merge_layers = [lay for lay in arch.layers if lay.kind in _MERGE_KINDS]
    has_merge = len(merge_layers) > 0

    # Combine skip_detector output with layer-level merge detection.
    has_skip = arch.has_skip_connections or has_merge

    # ── Priority 1: Transformer / attention ─────────────────────────────────
    if has_attention:
        warnings.append(
            "Transformer-like architecture detected.  "
            "Q/K/V projections, individual attention heads, exact residual paths, "
            "and tensor flow are NOT drawn.  "
            "Use transformer_mode='block_summary' for a conceptual overview.  "
            "Full layer metadata preserved in debug JSON."
        )
        if conv_count > 0:
            if has_skip:
                warnings.append(
                    f"Additional skip/merge operations detected "
                    f"({len(merge_layers)} merge layer(s)) — collapsed in visual output."
                )
            return RenderFamily.ALEXNET, ConfidenceLevel.LOW, warnings
        return RenderFamily.FCNN, ConfidenceLevel.LOW, warnings

    # ── Priority 2: Recurrent (LSTM / GRU / RNN) ────────────────────────────
    if has_recurrent:
        warnings.append(
            "Recurrent layers (LSTM/GRU/RNN) detected.  "
            "Recurrent connections are not drawn — rendered as a linear sequence.  "
            "Full layer metadata preserved in debug JSON."
        )
        return RenderFamily.FCNN, ConfidenceLevel.LOW, warnings

    # ── Priority 3: Pure dense / MLP ────────────────────────────────────────
    if conv_count == 0 and dense_count > 0:
        if has_merge:
            warnings.append(
                f"Model has {len(merge_layers)} merge operation(s) (Add/Concat/Multiply).  "
                "Branches are not drawn — layers rendered sequentially.  "
                "Full graph preserved in debug JSON."
            )
            return RenderFamily.FCNN, ConfidenceLevel.MEDIUM, warnings
        return RenderFamily.FCNN, ConfidenceLevel.HIGH, warnings

    # ── Priority 4: Small CNN (1–3 convolutions) ────────────────────────────
    if 1 <= conv_count <= 3:
        if has_skip:
            merge_count = len(merge_layers)
            warnings.append(
                f"Model has {merge_count} residual/skip operation(s).  "
                "Skip arcs are not drawn — sequential backbone only.  "
                "Full graph preserved in debug JSON."
            )
            return RenderFamily.LENET, ConfidenceLevel.MEDIUM, warnings
        return RenderFamily.LENET, ConfidenceLevel.HIGH, warnings

    # ── Priority 5: Deeper CNN (4+ convolutions) ────────────────────────────
    if conv_count > 3:
        if has_skip:
            concat_in_merges = any(lay.kind == LayerKind.CONCAT for lay in merge_layers)
            if concat_in_merges:
                warnings.append(
                    f"U-Net/encoder-decoder style ({len(merge_layers)} Concat operation(s)).  "
                    "Concat skip paths are not drawn — encoder-decoder summary rendered.  "
                    "Full graph preserved in debug JSON."
                )
            else:
                warnings.append(
                    f"ResNet-like architecture ({len(merge_layers)} residual Add operation(s)).  "
                    "Residual skip arcs are not drawn — residual block summary rendered.  "
                    "Full graph preserved in debug JSON."
                )
            return RenderFamily.ALEXNET, ConfidenceLevel.MEDIUM, warnings
        return RenderFamily.ALEXNET, ConfidenceLevel.HIGH, warnings

    # ── Fallback: nothing standard found ────────────────────────────────────
    if dense_count == 0 and conv_count == 0:
        warnings.append(
            "No standard convolutional or dense layers detected.  "
            "Architecture type is unknown; using FCNN view as a best-effort fallback.  "
            "The full graph is preserved in the debug-JSON export."
        )
        return RenderFamily.FCNN, ConfidenceLevel.UNKNOWN, warnings

    warnings.append(
        "Architecture does not map cleanly to any NN-SVG family.  "
        "Using FCNN as a best-effort approximation.  "
        "The full graph is preserved in the debug-JSON export."
    )
    return RenderFamily.FCNN, ConfidenceLevel.LOW, warnings
