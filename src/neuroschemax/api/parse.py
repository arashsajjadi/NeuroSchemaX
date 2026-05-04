"""Public parsing API: ``parse_model`` and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ingest.base import build_default_registry
from ..ir.graph_ir import GraphIR
from ..ir.semantic_ir import SemanticArchitecture
from ..normalize import normalize


def parse_graph(source: str | Path | Any) -> GraphIR:
    """Parse *source* into a low-level :class:`GraphIR`.

    *source* may be:
    - a path to an ``.onnx``, ``.json``, ``.yaml``, or ``.yml`` file
    - a ``dict`` holding a manual architecture spec
    - a ``torch.nn.Module`` instance
    - a ``tf.keras.Model`` instance
    - an already-loaded ``onnx.ModelProto``
    """
    registry = build_default_registry()
    return registry.parse(source)


def parse_model(source: str | Path | Any) -> SemanticArchitecture:
    """Parse *source* and return a normalised :class:`SemanticArchitecture`.

    This is the primary entry point for most users. It runs both the
    ingestion and semantic-normalisation stages.
    """
    graph = parse_graph(source)
    return normalize(graph)
