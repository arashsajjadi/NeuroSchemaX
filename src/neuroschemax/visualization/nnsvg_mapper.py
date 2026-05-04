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
    # Merge ops: structural, not a neuron column
    LayerKind.ADD,
    LayerKind.CONCAT,
    LayerKind.MULTIPLY,
})

# Ops absorbed into a preceding Transformer block during block-level mapping.
_TRANSFORMER_ABSORB = frozenset({
    LayerKind.LAYER_NORM,
    LayerKind.BATCH_NORM,
    LayerKind.GROUP_NORM,
    LayerKind.ADD,
    LayerKind.MULTIPLY,
    LayerKind.DROPOUT,
})

_ACTIVATION_KINDS = frozenset({
    LayerKind.RELU, LayerKind.GELU, LayerKind.SIGMOID,
    LayerKind.TANH, LayerKind.SOFTMAX, LayerKind.ACTIVATION,
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
    if kind in _ACTIVATION_KINDS:
        return _COLORS["activation"]
    if kind == LayerKind.ATTENTION:
        return _COLORS["attention"]
    if kind in (LayerKind.LSTM, LayerKind.GRU, LayerKind.RECURRENT):
        return _COLORS["recurrent"]
    return _COLORS["other"]


# ── Mapping helpers ──────────────────────────────────────────────────────────

_MAX_LABEL_LEN = 20


def _truncate(text: str, max_len: int = _MAX_LABEL_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _make_label(layer: SemanticLayer, show_shapes: bool, max_len: int = _MAX_LABEL_LEN) -> str:
    base = layer.name or layer.kind.name.lower()
    if show_shapes and layer.output_shape:
        dims = "x".join(str(d) for d in layer.output_shape if isinstance(d, int))
        if dims:
            candidate = f"{base} {dims}"
            return _truncate(candidate, max_len)
    return _truncate(base, max_len)


# ── Transformer block-level mapper ───────────────────────────────────────────

def _map_transformer_blocks(
    arch: SemanticArchitecture,
    cfg: RenderConfig,
) -> list[NNSVGLayerSpec]:
    """Block-level approximation for Transformer/attention architectures.

    Instead of rendering attention as neuron columns (misleading), groups
    operations into labeled computation blocks.  Each block is rendered as
    a small fixed-size column with a meaningful label and distinct colour.

    This is explicitly an approximation: NN-SVG has no native Transformer
    renderer.  The block sequence is correct; spatial attention relationships
    are not drawn.
    """
    layers = arch.layers
    specs: list[NNSVGLayerSpec] = []
    i = 0

    while i < len(layers):
        layer = layers[i]
        kind = layer.kind

        if kind == LayerKind.INPUT:
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label="Input", units=3, color=_COLORS["input"],
            ))
            i += 1

        elif kind == LayerKind.EMBEDDING:
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label="Embedding", units=3, color=_COLORS["dense"],
            ))
            i += 1

        elif kind == LayerKind.ATTENTION:
            # Absorb any immediately following Norm/Add/residual ops.
            j = i + 1
            while j < len(layers) and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label="[Attention]", units=3, color=_COLORS["attention"],
            ))
            i = j

        elif kind == LayerKind.DENSE:
            # Absorb a complete feed-forward block:
            # Dense + optional activation(s) + optional Dense + optional Norm/Add.
            j = i + 1
            while j < len(layers) and layers[j].kind in (
                _ACTIVATION_KINDS | frozenset({LayerKind.DENSE, LayerKind.DROPOUT})
            ):
                j += 1
            while j < len(layers) and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1

            # Decide label: last dense block → Classifier, otherwise FeedFwd.
            remaining = [layers[k].kind for k in range(j, len(layers))]
            is_classifier = all(
                k in (_ACTIVATION_KINDS | frozenset({LayerKind.OUTPUT, LayerKind.SOFTMAX,
                                                      LayerKind.FLATTEN, LayerKind.RESHAPE,
                                                      LayerKind.UNKNOWN}))
                for k in remaining
            )
            if is_classifier:
                specs.append(NNSVGLayerSpec(
                    layer_type="dense", label="Classifier", units=3, color=_COLORS["output"],
                ))
            else:
                specs.append(NNSVGLayerSpec(
                    layer_type="dense", label="FeedFwd", units=3, color=_COLORS["dense"],
                ))
            i = j

        elif kind in (LayerKind.LSTM, LayerKind.GRU, LayerKind.RECURRENT):
            j = i + 1
            while j < len(layers) and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            specs.append(NNSVGLayerSpec(
                layer_type="dense",
                label=f"[{kind.name}]",
                units=3,
                color=_COLORS["recurrent"],
            ))
            i = j

        else:
            # Skip structural / passthrough ops not absorbed above.
            i += 1

    return specs


# ── FCNN mapper ──────────────────────────────────────────────────────────────

def _map_fcnn(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to FCNN: columns of neurons.

    For Transformer/attention architectures, delegates to the block-level
    mapper which avoids rendering attention as misleading neuron columns.
    """
    has_attention = any(lay.kind == LayerKind.ATTENTION for lay in arch.layers)
    has_recurrent = any(
        lay.kind in (LayerKind.LSTM, LayerKind.GRU, LayerKind.RECURRENT)
        for lay in arch.layers
    )
    if has_attention or has_recurrent:
        return _map_transformer_blocks(arch, cfg)

    specs: list[NNSVGLayerSpec] = []
    for layer in arch.layers:
        if layer.kind in _SKIP_IN_ALL:
            continue
        if layer.kind in (LayerKind.FLATTEN, LayerKind.RESHAPE, LayerKind.UPSAMPLE):
            continue
        if layer.kind == LayerKind.UNKNOWN:
            continue

        units = layer.units or layer.channels or 0
        if units == 0 and layer.output_shape:
            ints = [d for d in layer.output_shape if isinstance(d, int) and d > 0]
            if ints:
                units = ints[-1]
        if units == 0:
            units = 10

        specs.append(NNSVGLayerSpec(
            layer_type="dense",
            label=_make_label(layer, cfg.show_shapes),
            units=min(units, 256),
            color=_color_for(layer.kind),
        ))
    return specs


# ── CNN mappers ──────────────────────────────────────────────────────────────

def _estimate_feature_map(layer: SemanticLayer) -> tuple[int, int]:
    shape = layer.output_shape
    ints = [d for d in shape if isinstance(d, int) and d > 0]
    if len(ints) >= 3:
        return ints[-2], ints[-1]
    if len(ints) == 2:
        return ints[-1], ints[-1]
    return 0, 0


# Dense (fc/classifier) units capped to prevent visual domination in CNN views.
_MAX_DENSE_UNITS_LENET   = 10
_MAX_DENSE_UNITS_ALEXNET =  8


def _map_lenet(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to LeNet view: conv/pool layers as feature maps, dense as neurons."""
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
            units = min(layer.units or 10, _MAX_DENSE_UNITS_LENET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense",
                label=_make_label(layer, cfg.show_shapes),
                units=units,
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
            # Attention / recurrent / embedding → small dense block
            units = min(layer.units or layer.channels or 8, _MAX_DENSE_UNITS_LENET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense",
                label=_make_label(layer, cfg.show_shapes),
                units=units,
                color=_color_for(layer.kind),
            ))
    return specs


def _map_alexnet(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to AlexNet view: like LeNet but with tighter dense caps for deep nets."""
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
                feature_map_width=w or 20,
                feature_map_height=h or 20,
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
                feature_map_width=w or 10,
                feature_map_height=h or 10,
                color=_color_for(layer.kind),
            ))
        elif layer.kind == LayerKind.DENSE:
            # Cap classifier columns so they don't dominate deep CNN diagrams.
            units = min(layer.units or 8, _MAX_DENSE_UNITS_ALEXNET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense",
                label=_make_label(layer, cfg.show_shapes),
                units=units,
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
                    feature_map_width=w or 20,
                    feature_map_height=h or 20,
                    color=_color_for(layer.kind),
                ))
        else:
            units = min(layer.units or layer.channels or 6, _MAX_DENSE_UNITS_ALEXNET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense",
                label=_make_label(layer, cfg.show_shapes),
                units=units,
                color=_color_for(layer.kind),
            ))
    return specs


# ── Auto-sizing ──────────────────────────────────────────────────────────────

_MIN_PX_PER_LAYER = 80


def _auto_width(n_layers: int, configured_width: int) -> int:
    needed = n_layers * _MIN_PX_PER_LAYER + 120
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
