"""PyTorch model ingestion adapter — Tier 2.

Requires ``torch`` to be installed.  Falls back gracefully when missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import AdapterImportError
from ..ir.graph_ir import GraphIR, GraphNode
from .base import BaseAdapter


def _import_torch() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise AdapterImportError("pytorch", "torch") from exc


class PyTorchAdapter(BaseAdapter):
    """Parse a ``torch.nn.Module`` into :class:`GraphIR`.

    This adapter walks the module tree to extract layer information.
    For deeper graph-level analysis, exporting to ONNX first is recommended.
    """

    name = "pytorch"

    def can_handle(self, source: str | Path | Any) -> bool:
        try:
            torch = _import_torch()
            return isinstance(source, torch.nn.Module)
        except AdapterImportError:
            return False

    def parse(self, source: str | Path | Any) -> GraphIR:
        module = source
        nodes: list[GraphNode] = []

        for idx, (name, child) in enumerate(module.named_modules()):
            if name == "":
                continue
            op_type = type(child).__name__
            attrs: dict[str, Any] = {}

            if hasattr(child, "in_features"):
                attrs["in_features"] = child.in_features
            if hasattr(child, "out_features"):
                attrs["out_features"] = child.out_features
            if hasattr(child, "in_channels"):
                attrs["in_channels"] = child.in_channels
            if hasattr(child, "out_channels"):
                attrs["out_channels"] = child.out_channels
            if hasattr(child, "kernel_size"):
                ks = child.kernel_size
                attrs["kernel_size"] = list(ks) if isinstance(ks, tuple) else [ks]
            if hasattr(child, "stride"):
                s = child.stride
                attrs["stride"] = list(s) if isinstance(s, tuple) else [s]
            if hasattr(child, "padding"):
                p = child.padding
                if isinstance(p, tuple):
                    attrs["padding"] = list(p)
                elif isinstance(p, int):
                    attrs["padding"] = [p]

            nodes.append(GraphNode(
                id=f"node_{idx}",
                op_type=op_type,
                name=name,
                attributes=attrs,
            ))

        return GraphIR(
            model_name=type(module).__name__,
            framework="pytorch",
            nodes=nodes,
        )
