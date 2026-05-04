"""Normalise and propagate tensor shapes through the layer sequence."""

from __future__ import annotations

from ..core.enums import LayerKind
from ..ir.semantic_ir import SemanticLayer

# Layer kinds that pass their input shape through unchanged.
_PASSTHROUGH_KINDS = frozenset({
    LayerKind.RELU,
    LayerKind.SIGMOID,
    LayerKind.TANH,
    LayerKind.SOFTMAX,
    LayerKind.GELU,
    LayerKind.ACTIVATION,
    LayerKind.DROPOUT,
    LayerKind.BATCH_NORM,
    LayerKind.LAYER_NORM,
    LayerKind.GROUP_NORM,
    LayerKind.INSTANCE_NORM,
    LayerKind.ADD,
    LayerKind.MULTIPLY,
    LayerKind.PAD,
})


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
    if layer.kind == LayerKind.DENSE and layer.output_shape:
        last = layer.output_shape[-1]
        if isinstance(last, int) and last > 0:
            layer.units = last


def infer_channels_from_shape(layer: SemanticLayer) -> None:
    """If *channels* is not set but *output_shape* is available, try to infer it."""
    if layer.channels is not None:
        return
    if (
        layer.kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV)
        and layer.output_shape
        and len(layer.output_shape) >= 3
    ):
        ch_dim = layer.output_shape[1] if len(layer.output_shape) >= 4 else layer.output_shape[0]
        if isinstance(ch_dim, int) and ch_dim > 0:
            layer.channels = ch_dim


def _parse_nchw(
    shape: list[int | str],
) -> tuple[int | None, int | None, int | None, int | None]:
    """Extract (N, C, H, W) from a shape; return None for unknown/missing dims."""
    ints: list[int | None] = [d if isinstance(d, int) else None for d in shape]
    if len(ints) == 4:
        return ints[0], ints[1], ints[2], ints[3]
    if len(ints) == 3:
        return None, ints[0], ints[1], ints[2]
    return None, None, None, None


def _to_pair(val: list[int] | None, default: int) -> tuple[int, int]:
    """Convert a kernel/stride list to an (h, w) pair with a fallback default."""
    if not val:
        return default, default
    if len(val) >= 2:
        return int(val[0]), int(val[1])
    return int(val[0]), int(val[0])


def infer_sequential_shapes(layers: list[SemanticLayer]) -> None:
    """Lightweight sequential shape inference for manual specs.

    Infers output shapes for ops whose semantics are unambiguous in a
    sequential (non-branching) context.  Unknown or dynamic shapes are
    left as-is.  Modifies layers in-place.
    """
    prev_shape: list[int | str] = []

    for layer in layers:
        # Already has a shape — use it and move on.
        if layer.output_shape:
            prev_shape = layer.output_shape
            continue

        if not prev_shape:
            continue

        # If any previous dim is symbolic / dynamic, skip inference.
        if any(isinstance(d, str) for d in prev_shape):
            continue

        inferred: list[int | str] | None = _infer_one(layer, prev_shape)
        if inferred is not None:
            layer.output_shape = inferred
            prev_shape = inferred


def _infer_one(
    layer: SemanticLayer,
    prev: list[int | str],
) -> list[int | str] | None:
    """Return the inferred output shape for *layer* given *prev* input shape."""
    kind = layer.kind

    # ── Passthrough ops ─────────────────────────────────────────────────────
    if kind in _PASSTHROUGH_KINDS:
        return list(prev)

    # ── Convolution ─────────────────────────────────────────────────────────
    if kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV):
        N, C, H, W = _parse_nchw(prev)
        if H is None or W is None:
            return None
        kh, kw = _to_pair(layer.kernel_size, 1)
        sh, sw = _to_pair(layer.stride, 1)
        ph, pw = _to_pair(layer.padding, 0)
        c_out = layer.channels or C or 1
        h_out = (H + 2 * ph - kh) // sh + 1
        w_out = (W + 2 * pw - kw) // sw + 1
        if N is not None:
            return [N, c_out, h_out, w_out]
        return [c_out, h_out, w_out]

    # ── Max / Avg pool ───────────────────────────────────────────────────────
    if kind in (LayerKind.POOL_MAX, LayerKind.POOL_AVG):
        N, C, H, W = _parse_nchw(prev)
        if H is None or W is None:
            return None
        kh, kw = _to_pair(layer.kernel_size, 2)
        # Pool stride defaults to kernel_size when not specified.
        sh, sw = _to_pair(layer.stride, 0)
        if sh == 0:
            sh, sw = kh, kw
        h_out = (H - kh) // sh + 1
        w_out = (W - kw) // sw + 1
        c = C or 1
        if N is not None:
            return [N, c, h_out, w_out]
        return [c, h_out, w_out]

    # ── Global pool ─────────────────────────────────────────────────────────
    if kind == LayerKind.POOL_GLOBAL:
        N, C, _H, _W = _parse_nchw(prev)
        c = C or 1
        if N is not None:
            return [N, c]
        return [c]

    # ── Flatten ─────────────────────────────────────────────────────────────
    if kind in (LayerKind.FLATTEN, LayerKind.RESHAPE):
        ints = [d for d in prev if isinstance(d, int)]
        if not ints:
            return None
        if len(ints) == 1:
            return list(prev)  # already flat
        # Flatten everything except the batch dim.
        flat = 1
        for d in ints[1:]:
            flat *= d
        return [ints[0], flat]

    # ── Dense / Linear ──────────────────────────────────────────────────────
    if kind == LayerKind.DENSE and layer.units:
        ints = [d for d in prev if isinstance(d, int)]
        if ints:
            return [ints[0], layer.units]
        return [layer.units]

    # ── Upsample / Resize ───────────────────────────────────────────────────
    if kind == LayerKind.UPSAMPLE:
        scale = layer.attributes.get("scale_factor", layer.attributes.get("scales"))
        if scale is not None:
            N, C, H, W = _parse_nchw(prev)
            if H is not None and W is not None:
                if isinstance(scale, (int, float)):
                    h_out, w_out = int(H * scale), int(W * scale)
                elif isinstance(scale, (list, tuple)) and len(scale) >= 2:
                    h_out = int(H * float(scale[-2]))
                    w_out = int(W * float(scale[-1]))
                else:
                    return None
                c = C or 1
                if N is not None:
                    return [N, c, h_out, w_out]
                return [c, h_out, w_out]
        # No scale info: output shape unknown.
        return None

    # ── Attention (self-attention output = same shape as input) ─────────────
    if kind == LayerKind.ATTENTION:
        return list(prev)

    return None
