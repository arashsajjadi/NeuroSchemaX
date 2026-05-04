"""Protocol / interface definitions for adapter and pipeline contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..ir.graph_ir import GraphIR


@runtime_checkable
class IngestAdapter(Protocol):
    """Protocol that every model-ingestion adapter must satisfy."""

    name: str

    def can_handle(self, source: str | Path | Any) -> bool:
        """Return *True* if this adapter can parse *source*."""
        ...

    def parse(self, source: str | Path | Any) -> GraphIR:
        """Parse *source* into a :class:`GraphIR`."""
        ...
