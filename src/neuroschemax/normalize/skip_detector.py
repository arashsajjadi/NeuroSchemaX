"""Detect skip / residual connections in the graph."""

from __future__ import annotations

from ..core.enums import ConfidenceLevel, LayerKind
from ..ir.graph_ir import GraphIR
from ..ir.semantic_ir import SkipConnection
from .op_normalizer import normalize_op


def detect_skip_connections(graph: GraphIR) -> list[SkipConnection]:
    """Identify likely skip / residual connections.

    Heuristic: look for Add or Concat nodes whose inputs originate from
    nodes that are not direct predecessors (i.e. one input skips at least
    one layer).
    """
    skips: list[SkipConnection] = []

    # Build direct-predecessor map
    direct_pred: dict[str, set[str]] = {}
    for edge in graph.edges:
        direct_pred.setdefault(edge.target_id, set()).add(edge.source_id)

    for node in graph.nodes:
        kind, _ = normalize_op(node.op_type)
        if kind not in (LayerKind.ADD, LayerKind.CONCAT, LayerKind.MULTIPLY):
            continue

        preds = direct_pred.get(node.id, set())
        if len(preds) < 2:
            continue

        # The merge node itself is the target; the skip comes from the
        # predecessor that is further away in topological order.
        pred_list = sorted(preds)
        for src in pred_list[:-1]:
            merge_kind = "add" if kind == LayerKind.ADD else (
                "concat" if kind == LayerKind.CONCAT else "multiply"
            )
            skips.append(SkipConnection(
                source_id=src,
                target_id=node.id,
                kind=merge_kind,
                confidence=ConfidenceLevel.MEDIUM,
            ))

    return skips
