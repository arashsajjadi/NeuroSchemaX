"""TensorFlow / Keras model ingestion adapter — Tier 2.

Requires ``tensorflow`` to be installed.  Falls back gracefully when missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import AdapterImportError
from ..ir.graph_ir import GraphIR, GraphNode
from .base import BaseAdapter


def _import_tf() -> Any:
    try:
        import tensorflow as tf
        return tf
    except ImportError as exc:
        raise AdapterImportError("tensorflow", "tensorflow") from exc


class TensorFlowAdapter(BaseAdapter):
    """Parse a ``tf.keras.Model`` into :class:`GraphIR`."""

    name = "tensorflow"

    def can_handle(self, source: str | Path | Any) -> bool:
        try:
            tf = _import_tf()
            return isinstance(source, tf.keras.Model)
        except (AdapterImportError, Exception):
            return False

    def parse(self, source: str | Path | Any) -> GraphIR:
        model = source
        nodes: list[GraphNode] = []

        for idx, layer in enumerate(model.layers):
            op_type = type(layer).__name__
            config = layer.get_config()
            attrs: dict[str, Any] = {}

            if "units" in config:
                attrs["units"] = config["units"]
            if "filters" in config:
                attrs["out_channels"] = config["filters"]
            if "kernel_size" in config:
                ks = config["kernel_size"]
                attrs["kernel_size"] = list(ks) if isinstance(ks, (tuple, list)) else [ks]
            if "strides" in config:
                s = config["strides"]
                attrs["stride"] = list(s) if isinstance(s, (tuple, list)) else [s]
            if "padding" in config:
                attrs["padding_mode"] = config["padding"]
            if "activation" in config and config["activation"]:
                attrs["activation"] = config["activation"]
            if "pool_size" in config:
                ps = config["pool_size"]
                attrs["kernel_size"] = list(ps) if isinstance(ps, (tuple, list)) else [ps]
            if "rate" in config:
                attrs["rate"] = config["rate"]

            out_shape: list[list[int | str]] = []
            try:
                shape = layer.output_shape
                out_shape = [list(s) for s in shape] if isinstance(shape, list) else [list(shape)]
            except (AttributeError, RuntimeError):
                pass

            nodes.append(GraphNode(
                id=f"node_{idx}",
                op_type=op_type,
                name=layer.name,
                attributes=attrs,
                output_shapes=out_shape,
            ))

        model_name = model.name if hasattr(model, "name") else "keras_model"
        return GraphIR(
            model_name=model_name,
            framework="tensorflow",
            nodes=nodes,
        )
