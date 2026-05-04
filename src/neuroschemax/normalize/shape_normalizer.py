"""Normalise and propagate tensor shapes through the layer sequence."""

from __future__ import annotations

from ..ir.semantic_ir import SemanticLayer


def propagate_shapes(layers: list[SemanticLayer]) -> None:
    """Forward-propagate shapes where missing.

    Modifies layers in-place: if a layer has no ``input_shape`` but the
    preceding layer has an ``output_shape``, carry it forward.
    """
    for i in range(1, len(layers)):
        prev = layers[i - 1]
        curr = layers[i]
        if not curr.input_shape and prev.output_shape:
            curr.input_shape = list(prev.output_shape)


def infer_units_from_shape(layer: SemanticLayer) -> None:
    """If *units* is not set but *output_shape* is available, try to infer it."""
    if layer.units is not None:
        return
    from ..core.enums import LayerKind
    if layer.kind == LayerKind.DENSE and layer.output_shape:
        # Last dimension of a dense layer's output is the unit count
        last = layer.output_shape[-1]
        if isinstance(last, int) and last > 0:
            layer.units = last


def infer_channels_from_shape(layer: SemanticLayer) -> None:
    """If *channels* is not set but *output_shape* is available, try to infer it."""
    if layer.channels is not None:
        return
    from ..core.enums import LayerKind
    if (
        layer.kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV)
        and layer.output_shape
        and len(layer.output_shape) >= 3
    ):
        # Assume NCHW or CHW layout → channels is dim 1 (or 0 if no batch)
        ch_dim = layer.output_shape[1] if len(layer.output_shape) >= 4 else layer.output_shape[0]
        if isinstance(ch_dim, int) and ch_dim > 0:
            layer.channels = ch_dim
