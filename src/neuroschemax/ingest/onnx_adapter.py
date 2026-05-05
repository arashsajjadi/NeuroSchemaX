"""ONNX model ingestion adapter — Tier 1 (best supported)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import AdapterImportError
from ..ir.graph_ir import GraphEdge, GraphIR, GraphNode, TensorInfo
from .base import BaseAdapter

# ONNX element-type id → human-readable dtype
_ONNX_DTYPE_MAP: dict[int, str] = {
    0: "undefined", 1: "float32", 2: "uint8", 3: "int8", 4: "uint16",
    5: "int16", 6: "int32", 7: "int64", 8: "string", 9: "bool",
    10: "float16", 11: "float64", 12: "uint32", 13: "uint64",
    14: "complex64", 15: "complex128", 16: "bfloat16",
}


def _import_onnx() -> Any:
    try:
        import onnx
        return onnx
    except ImportError as exc:
        raise AdapterImportError("onnx", "onnx") from exc


def _type_proto(vi: Any) -> Any:
    """Return the TypeProto from either a ValueInfoProto or a bare TypeProto."""
    # ValueInfoProto has a .type field; TypeProto does not.
    if hasattr(vi, "type"):
        return vi.type
    return vi


def _extract_shape(vi: Any) -> list[int | str]:
    """Extract shape dimensions from a ValueInfoProto or TypeProto.

    Returns an empty list for non-tensor types or unknown shapes.
    Never raises.
    """
    try:
        tp = _type_proto(vi)
        # Tensor type only
        if not tp.HasField("tensor_type"):
            return []
        tt = tp.tensor_type
        if not tt.HasField("shape"):
            return []
        shape: list[int | str] = []
        for dim in tt.shape.dim:
            if dim.dim_param:            # symbolic dim, e.g. "batch_size"
                shape.append(dim.dim_param)
            elif dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append("?")        # unknown (dim_value == 0)
        return shape
    except Exception:  # noqa: BLE001
        return []


def _extract_dtype(vi: Any) -> str:
    """Extract dtype string from a ValueInfoProto or TypeProto.  Never raises."""
    try:
        tp = _type_proto(vi)
        if not tp.HasField("tensor_type"):
            return "unknown"
        elem = tp.tensor_type.elem_type
        return _ONNX_DTYPE_MAP.get(elem, f"type_{elem}")
    except Exception:  # noqa: BLE001
        return "unknown"


def _attribute_to_python(attr: Any) -> Any:
    """Convert an ONNX AttributeProto to a plain Python value."""
    onnx = _import_onnx()
    atype = attr.type
    if atype == onnx.AttributeProto.INT:
        return attr.i
    if atype == onnx.AttributeProto.FLOAT:
        return attr.f
    if atype == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8", errors="replace")
    if atype == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if atype == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if atype == onnx.AttributeProto.TENSOR:
        return "<tensor>"
    return str(attr)


def _try_infer_shapes(model: Any) -> Any:
    """Run ONNX shape inference and return the enriched model.

    Falls back to the original model if inference fails.
    """
    try:
        import onnx
        return onnx.shape_inference.infer_shapes(model)
    except Exception:  # noqa: BLE001
        return model


class OnnxAdapter(BaseAdapter):
    """Parse ONNX model files into :class:`GraphIR`."""

    name = "onnx"

    def can_handle(self, source: str | Path | Any) -> bool:
        if isinstance(source, (str, Path)):
            return Path(source).suffix.lower() == ".onnx"
        try:
            onnx = _import_onnx()
            return isinstance(source, onnx.ModelProto)
        except Exception:  # noqa: BLE001
            return False

    def parse(self, source: str | Path | Any) -> GraphIR:
        onnx = _import_onnx()

        model = onnx.load(str(source)) if isinstance(source, (str, Path)) else source

        # Run shape inference to populate intermediate value_info shapes.
        model = _try_infer_shapes(model)
        graph = model.graph

        # Build value-info shape map from graph inputs, outputs, and
        # intermediate tensors.  Uses defensive extraction so missing or
        # non-tensor types silently produce empty lists.
        shape_map: dict[str, list[int | str]] = {}
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            shape_map[vi.name] = _extract_shape(vi)

        # Inputs / outputs — skip weight initializers (they appear in graph.input
        # too but have entries in graph.initializer).
        init_names = {init.name for init in graph.initializer}
        ir_inputs = [
            TensorInfo(
                name=inp.name,
                shape=_extract_shape(inp),
                dtype=_extract_dtype(inp),
            )
            for inp in graph.input if inp.name not in init_names
        ]
        ir_outputs = [
            TensorInfo(
                name=out.name,
                shape=_extract_shape(out),
                dtype=_extract_dtype(out),
            )
            for out in graph.output
        ]

        # Nodes & edges
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        output_to_node: dict[str, str] = {}

        for idx, node in enumerate(graph.node):
            node_id = node.name or f"node_{idx}"
            attrs: dict[str, Any] = {}
            import contextlib
            with contextlib.suppress(Exception):
                attrs = {a.name: _attribute_to_python(a) for a in node.attribute}

            in_shapes = [shape_map.get(i, []) for i in node.input if i]
            out_shapes = [shape_map.get(o, []) for o in node.output if o]

            nodes.append(GraphNode(
                id=node_id,
                op_type=node.op_type,
                name=node.name or node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attrs,
                input_shapes=in_shapes,
                output_shapes=out_shapes,
            ))

            for o in node.output:
                if o:
                    output_to_node[o] = node_id

        # Build edges from tensor connections
        for node in nodes:
            for inp in node.inputs:
                if inp and inp in output_to_node:
                    src = output_to_node[inp]
                    edges.append(GraphEdge(
                        source_id=src,
                        target_id=node.id,
                        tensor_name=inp,
                        shape=shape_map.get(inp, []),
                    ))

        model_name = graph.name or "onnx_model"
        metadata: dict[str, Any] = {"framework": "onnx"}
        if model.producer_name:
            metadata["producer"] = model.producer_name
        if model.ir_version:
            metadata["ir_version"] = model.ir_version
        for prop in model.metadata_props:
            metadata[prop.key] = prop.value

        return GraphIR(
            model_name=model_name,
            framework="onnx",
            nodes=nodes,
            edges=edges,
            inputs=ir_inputs,
            outputs=ir_outputs,
            metadata=metadata,
        )
