"""Base adapter and adapter registry for model ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import ParseError, UnsupportedFormatError
from ..ir.graph_ir import GraphIR


class BaseAdapter:
    """Abstract base for all ingestion adapters."""

    name: str = "base"

    def can_handle(self, source: str | Path | Any) -> bool:
        raise NotImplementedError

    def parse(self, source: str | Path | Any) -> GraphIR:
        raise NotImplementedError


class AdapterRegistry:
    """Ordered collection of adapters tried during :func:`auto_parse`."""

    def __init__(self) -> None:
        self._adapters: list[BaseAdapter] = []

    def register(self, adapter: BaseAdapter) -> None:
        self._adapters.append(adapter)

    def parse(self, source: str | Path | Any) -> GraphIR:
        """Try each registered adapter in order and return the first success."""
        for adapter in self._adapters:
            if adapter.can_handle(source):
                try:
                    return adapter.parse(source)
                except Exception as exc:
                    raise ParseError(
                        f"Adapter '{adapter.name}' accepted source but failed: {exc}"
                    ) from exc
        raise UnsupportedFormatError(
            f"No adapter can handle source: {source!r}. "
            f"Registered: {[a.name for a in self._adapters]}"
        )


def build_default_registry() -> AdapterRegistry:
    """Build the default adapter registry with all available adapters."""
    from .manual_spec_adapter import ManualSpecAdapter
    from .onnx_adapter import OnnxAdapter
    from .pytorch_adapter import PyTorchAdapter
    from .tensorflow_adapter import TensorFlowAdapter

    registry = AdapterRegistry()
    registry.register(OnnxAdapter())
    registry.register(ManualSpecAdapter())
    registry.register(PyTorchAdapter())
    registry.register(TensorFlowAdapter())
    return registry
