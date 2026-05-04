"""Manual JSON / YAML architecture specification adapter — Tier 1.

Accepts a dict, a JSON file, or a YAML file describing layers directly.

Expected format::

    {
        "model_name": "my_model",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 28, 28]},
            {"name": "conv1", "kind": "conv", "channels": 32, "kernel_size": [3, 3]},
            {"name": "pool1", "kind": "pool_max", "kernel_size": [2, 2]},
            {"name": "fc1",   "kind": "dense", "units": 128},
            {"name": "output","kind": "dense", "units": 10}
        ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..exceptions import ParseError
from ..ir.graph_ir import GraphIR, GraphNode, TensorInfo
from .base import BaseAdapter


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ParseError(
            "YAML spec files require PyYAML. Install with: pip install pyyaml"
        ) from exc
    with open(path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _load_source(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path) as f:
            return json.load(f)  # type: ignore[no-any-return]
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    raise ParseError(f"Unsupported spec file format: {suffix}")


class ManualSpecAdapter(BaseAdapter):
    """Parse a hand-written architecture specification."""

    name = "manual_spec"

    def can_handle(self, source: str | Path | Any) -> bool:
        if isinstance(source, dict):
            return "layers" in source
        if isinstance(source, (str, Path)):
            suffix = Path(source).suffix.lower()
            return suffix in (".json", ".yaml", ".yml")
        return False

    def parse(self, source: str | Path | Any) -> GraphIR:
        data = _load_source(source)  # type: ignore[arg-type]
        if "layers" not in data:
            raise ParseError("Manual spec must contain a 'layers' key")

        nodes: list[GraphNode] = []
        for idx, layer in enumerate(data["layers"]):
            if not isinstance(layer, dict):
                raise ParseError(f"Layer {idx} must be a dict, got {type(layer).__name__}")
            attrs: dict[str, Any] = {
                k: v for k, v in layer.items()
                if k not in ("name", "kind", "shape", "input_shape", "output_shape")
            }
            in_shapes: list[list[int | str]] = []
            out_shapes: list[list[int | str]] = []
            if "input_shape" in layer:
                in_shapes = [layer["input_shape"]]
            if "output_shape" in layer:
                out_shapes = [layer["output_shape"]]
            if "shape" in layer:
                out_shapes = [layer["shape"]]

            nodes.append(GraphNode(
                id=f"node_{idx}",
                op_type=layer.get("kind", "unknown"),
                name=layer.get("name", f"layer_{idx}"),
                attributes=attrs,
                input_shapes=in_shapes,
                output_shapes=out_shapes,
            ))

        inputs: list[TensorInfo] = []
        if nodes and data["layers"][0].get("kind") == "input":
            shape = data["layers"][0].get("shape", [])
            inputs = [TensorInfo(name="input", shape=shape)]

        return GraphIR(
            model_name=data.get("model_name", "manual_model"),
            framework="manual_spec",
            nodes=nodes,
            inputs=inputs,
            metadata=data.get("metadata", {}),
        )
