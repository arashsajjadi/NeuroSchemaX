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


def _extract_shape(type_proto: Any) -> list[int | str]:
    """Extract shape dimensions from an ONNX TypeProto."""
    shape: list[int | str] = []
    tensor_type = type_proto.tensor_type
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append(dim.dim_value)
    return shape


def _extract_dtype(type_proto: Any) -> str:
    elem = type_proto.tensor_type.elem_type
    return _ONNX_DTYPE_MAP.get(elem, f"type_{elem}")


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


class OnnxAdapter(BaseAdapter):
    """Parse ONNX model files into :class:`GraphIR`."""

    name = "onnx"

    def can_handle(self, source: str | Path | Any) -> bool:
        if isinstance(source, (str, Path)):
            return Path(source).suffix.lower() == ".onnx"
        try:
            onnx = _import_onnx()
            return isinstance(source, onnx.ModelProto)
        except Exception:
            return False

    def parse(self, source: str | Path | Any) -> GraphIR:
        onnx = _import_onnx()

        model = onnx.load(str(source)) if isinstance(source, (str, Path)) else source

        graph = model.graph

        # Build value-info shape map
        shape_map: dict[str, list[int | str]] = {}
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            shape_map[vi.name] = _extract_shape(vi)

        # Inputs / outputs
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
        metadata: dict[str, Any] = {}
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
