"""Detect branching and merge points in the raw graph."""

from __future__ import annotations

from ..ir.graph_ir import GraphIR


def find_branch_points(graph: GraphIR) -> list[str]:
    """Return node IDs that have more than one outgoing edge (fan-out)."""
    out_count: dict[str, int] = {}
    for edge in graph.edges:
        out_count[edge.source_id] = out_count.get(edge.source_id, 0) + 1
    return [nid for nid, count in out_count.items() if count > 1]


def find_merge_points(graph: GraphIR) -> list[str]:
    """Return node IDs that have more than one incoming edge (fan-in)."""
    in_count: dict[str, int] = {}
    for edge in graph.edges:
        in_count[edge.target_id] = in_count.get(edge.target_id, 0) + 1
    return [nid for nid, count in in_count.items() if count > 1]


def has_branching(graph: GraphIR) -> bool:
    """Return True if the graph contains any branching structure."""
    return len(find_branch_points(graph)) > 0


def compute_complexity_hint(graph: GraphIR) -> str:
    """Return a rough complexity classification for the graph.

    Returns one of ``"sequential"``, ``"skip"``, ``"multi_branch"``, ``"dag"``.
    """
    branches = len(find_branch_points(graph))
    merges = len(find_merge_points(graph))
    if branches == 0 and merges == 0:
        return "sequential"
    if branches <= 2 and merges <= 2:
        return "skip"
    if branches <= 6:
        return "multi_branch"
    return "dag"
