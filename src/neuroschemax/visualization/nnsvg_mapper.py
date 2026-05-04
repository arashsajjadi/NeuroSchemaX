"""Map a :class:`SemanticArchitecture` to an :class:`NNSVGSpec`.

This is the bridge between the framework-agnostic semantic representation
and the NN-SVG rendering engine.  The mapping is family-aware: FCNN, LeNet,
and AlexNet each have different layer-spec expectations.
"""

from __future__ import annotations

from ..core.config import RenderConfig
from ..core.enums import LayerKind, RenderFamily
from ..ir.semantic_ir import SemanticArchitecture, SemanticLayer
from .nnsvg_schema import NNSVGLayerSpec, NNSVGSpec

# ── Colour palettes ─────────────────────────────────────────────────────────

_COLORS: dict[str, str] = {
    "input": "#9FC5E8",
    "conv": "#93C47D",
    "pool": "#F6B26B",
    "dense": "#6FA8DC",
    "output": "#E06666",
    "norm": "#B4A7D6",
    "activation": "#FFD966",
    "attention": "#EA9999",
    "recurrent": "#A2C4C9",
    "other": "#CCCCCC",
}

# Ops that don't contribute a visible column in any diagram family.
_SKIP_IN_ALL = frozenset({
    LayerKind.ACTIVATION,
    LayerKind.RELU,
    LayerKind.SIGMOID,
    LayerKind.TANH,
    LayerKind.SOFTMAX,
    LayerKind.GELU,
    LayerKind.DROPOUT,
    LayerKind.BATCH_NORM,
    LayerKind.LAYER_NORM,
    LayerKind.GROUP_NORM,
    LayerKind.INSTANCE_NORM,
    LayerKind.PAD,
    # Merge ops: structural but not a neuron column
    LayerKind.ADD,
    LayerKind.CONCAT,
    LayerKind.MULTIPLY,
})


def _color_for(kind: LayerKind) -> str:
    if kind == LayerKind.INPUT:
        return _COLORS["input"]
    if kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV):
        return _COLORS["conv"]
    if kind in (LayerKind.POOL_MAX, LayerKind.POOL_AVG, LayerKind.POOL_GLOBAL):
        return _COLORS["pool"]
    if kind == LayerKind.DENSE:
        return _COLORS["dense"]
    if kind == LayerKind.OUTPUT:
        return _COLORS["output"]
    if kind in (LayerKind.BATCH_NORM, LayerKind.LAYER_NORM,
                LayerKind.GROUP_NORM, LayerKind.INSTANCE_NORM):
        return _COLORS["norm"]
    if kind in (LayerKind.ACTIVATION, LayerKind.RELU, LayerKind.SIGMOID,
                LayerKind.TANH, LayerKind.SOFTMAX, LayerKind.GELU):
        return _COLORS["activation"]
    if kind == LayerKind.ATTENTION:
        return _COLORS["attention"]
    if kind in (LayerKind.LSTM, LayerKind.GRU, LayerKind.RECURRENT):
        return _COLORS["recurrent"]
    return _COLORS["other"]


# ── Mapping helpers ──────────────────────────────────────────────────────────

_MAX_LABEL_LEN = 20


def _truncate(text: str, max_len: int = _MAX_LABEL_LEN) -> str:
    """Truncate a label to *max_len* characters."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"  # … (ellipsis)


def _make_label(layer: SemanticLayer, show_shapes: bool, max_len: int = _MAX_LABEL_LEN) -> str:
    base = layer.name or layer.kind.name.lower()
    if show_shapes and layer.output_shape:
        dims = "x".join(str(d) for d in layer.output_shape if isinstance(d, int))
        if dims:
            candidate = f"{base} {dims}"
            return _truncate(candidate, max_len)
    return _truncate(base, max_len)


def _map_fcnn(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to FCNN: each significant layer becomes a column of neurons."""
    specs: list[NNSVGLayerSpec] = []
    for layer in arch.layers:
        if layer.kind in _SKIP_IN_ALL:
            continue
        # Reshape / Flatten — skip (not a neuron column)
        if layer.kind in (LayerKind.FLATTEN, LayerKind.RESHAPE):
            continue
        # Upsample — not a dense/conv column
        if layer.kind == LayerKind.UPSAMPLE:
            continue
        # Unknown ops — skip to avoid noise
        if layer.kind == LayerKind.UNKNOWN:
            continue

        units = layer.units or layer.channels or 0
        if units == 0 and layer.output_shape:
            ints = [d for d in layer.output_shape if isinstance(d, int) and d > 0]
            if ints:
                units = ints[-1]
        if units == 0:
            # For attention / recurrent / embedding, use a reasonable default
            if layer.kind == LayerKind.EMBEDDING:
                units = 64
            elif layer.kind in (LayerKind.ATTENTION, LayerKind.LSTM,
                                 LayerKind.GRU, LayerKind.RECURRENT):
                units = 32
            else:
                units = 10

        specs.append(NNSVGLayerSpec(
            layer_type="dense",
            label=_make_label(layer, cfg.show_shapes),
            units=min(units, 256),
            color=_color_for(layer.kind),
        ))
    return specs


def _estimate_feature_map(layer: SemanticLayer) -> tuple[int, int]:
    """Estimate feature-map spatial dims from output shape."""
    shape = layer.output_shape
    ints = [d for d in shape if isinstance(d, int) and d > 0]
    if len(ints) >= 3:
        return ints[-2], ints[-1]
    if len(ints) == 2:
        return ints[-1], ints[-1]
    return 0, 0


def _map_lenet(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to LeNet view: conv layers show feature maps, dense layers show neurons."""
    specs: list[NNSVGLayerSpec] = []
    for layer in arch.layers:
        if layer.kind in _SKIP_IN_ALL:
            continue
        if layer.kind in (LayerKind.FLATTEN, LayerKind.RESHAPE, LayerKind.UPSAMPLE):
            continue
        if layer.kind == LayerKind.UNKNOWN:
            continue

        if layer.kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV):
            h, w = _estimate_feature_map(layer)
            specs.append(NNSVGLayerSpec(
                layer_type="conv",
                label=_make_label(layer, cfg.show_shapes),
                channels=layer.channels or 1,
                kernel_size=layer.kernel_size[0] if layer.kernel_size else 3,
                stride=layer.stride[0] if layer.stride else 1,
                feature_map_width=w or 28,
                feature_map_height=h or 28,
                color=_color_for(layer.kind),
            ))
        elif layer.kind in (LayerKind.POOL_MAX, LayerKind.POOL_AVG, LayerKind.POOL_GLOBAL):
            h, w = _estimate_feature_map(layer)
            prev_channels = specs[-1].channels if specs else 1
            specs.append(NNSVGLayerSpec(
                layer_type="pool",
                label=_make_label(layer, cfg.show_shapes),
                channels=layer.channels or prev_channels,
                kernel_size=layer.kernel_size[0] if layer.kernel_size else 2,
                stride=layer.stride[0] if layer.stride else 2,
                feature_map_width=w or 14,
                feature_map_height=h or 14,
                color=_color_for(layer.kind),
            ))
        elif layer.kind == LayerKind.DENSE:
            units = layer.units or 10
            specs.append(NNSVGLayerSpec(
                layer_type="dense",
                label=_make_label(layer, cfg.show_shapes),
                units=min(units, 256),
                color=_color_for(layer.kind),
            ))
        elif layer.kind == LayerKind.INPUT:
            if layer.output_shape:
                h, w = _estimate_feature_map(layer)
                ints = [d for d in layer.output_shape if isinstance(d, int)]
                ch = ints[1] if len(ints) >= 4 else (ints[0] if len(ints) >= 3 else 1)
                specs.append(NNSVGLayerSpec(
                    layer_type="input",
                    label="input",
                    channels=ch,
                    feature_map_width=w or 28,
                    feature_map_height=h or 28,
                    color=_color_for(layer.kind),
                ))
        else:
            # Attention / recurrent / embedding → show as dense column
            units = layer.units or layer.channels or 0
            if units == 0 and layer.output_shape:
                ints = [d for d in layer.output_shape if isinstance(d, int) and d > 0]
                if ints:
                    units = ints[-1]
            if units == 0:
                units = 32
            specs.append(NNSVGLayerSpec(
                layer_type="dense",
                label=_make_label(layer, cfg.show_shapes),
                units=min(units, 256),
                color=_color_for(layer.kind),
            ))
    return specs


def _map_alexnet(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to AlexNet view: like LeNet but expects more layers."""
    return _map_lenet(arch, cfg)


# ── Auto-sizing ──────────────────────────────────────────────────────────────

_MIN_PX_PER_LAYER = 80
_IDEAL_PX_PER_LAYER = 100


def _auto_width(n_layers: int, configured_width: int) -> int:
    """Ensure the canvas is wide enough to give each layer breathing room."""
    needed = n_layers * _MIN_PX_PER_LAYER + 120  # margins
    return max(configured_width, needed)


# ── Public entry point ───────────────────────────────────────────────────────

_FAMILY_MAPPERS = {
    RenderFamily.FCNN: _map_fcnn,
    RenderFamily.LENET: _map_lenet,
    RenderFamily.ALEXNET: _map_alexnet,
}


def map_to_nnsvg(
    arch: SemanticArchitecture,
    config: RenderConfig,
) -> NNSVGSpec:
    """Build an :class:`NNSVGSpec` from a semantic architecture and config."""
    family = config.style or arch.recommended_family or RenderFamily.FCNN
    mapper = _FAMILY_MAPPERS[family]
    layers = mapper(arch, config)

    if not layers:
        layers = [NNSVGLayerSpec(layer_type="dense", label="(empty)", units=1)]

    # Auto-size canvas width if the user did not explicitly set it.
    # We consider it user-set if it differs from the RenderConfig default (1200).
    auto_width = _auto_width(len(layers), config.width)

    return NNSVGSpec(
        family=family,
        layers=layers,
        width=auto_width,
        height=config.height,
        show_labels=config.show_labels,
        show_shapes=config.show_shapes,
        title=config.title or arch.model_name,
        edge_opacity=config.edge_opacity,
        node_size=config.node_size,
        spacing=config.spacing,
        between_layers_spacing=config.between_layers_spacing,
        font_family=config.font_family,
        font_size=config.font_size,
        color_fill=config.color_fill,
        color_stroke=config.color_stroke,
        model_name=arch.model_name,
        warnings=list(arch.warnings),
    )
