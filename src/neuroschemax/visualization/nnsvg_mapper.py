"""Map a :class:`SemanticArchitecture` to an :class:`NNSVGSpec`.

This is the bridge between the framework-agnostic semantic representation
and the NN-SVG rendering engine.  The mapping is family-aware: FCNN, LeNet,
and AlexNet each have different layer-spec expectations.

For Transformer/attention architectures, a block-level approximation is
produced using single-rectangle blocks via the LeNet renderer.  This is
explicitly NOT exact Transformer rendering — NN-SVG has no native Transformer
family.  The block sequence is correct; attention relationships, residual
paths, and repeated-block groupings are approximated visually and preserved
in full in the debug-JSON export.
"""

from __future__ import annotations

from ..core.config import RenderConfig
from ..core.enums import LayerKind, RenderFamily
from ..ir.semantic_ir import SemanticArchitecture, SemanticLayer
from .nnsvg_schema import NNSVGLayerSpec, NNSVGSpec

# ── Colour palettes ─────────────────────────────────────────────────────────

_COLORS: dict[str, str] = {
    "input":      "#9FC5E8",
    "conv":       "#93C47D",
    "pool":       "#F6B26B",
    "dense":      "#6FA8DC",
    "output":     "#E06666",
    "norm":       "#B4A7D6",
    "activation": "#FFD966",
    "attention":  "#EA9999",
    "recurrent":  "#A2C4C9",
    "ffn":        "#A8D1F7",
    "other":      "#CCCCCC",
}

# ── Kind sets ────────────────────────────────────────────────────────────────

_CONV_KINDS = frozenset({
    LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV,
})

_POOL_KINDS = frozenset({
    LayerKind.POOL_MAX, LayerKind.POOL_AVG, LayerKind.POOL_GLOBAL,
})

_NORM_KINDS = frozenset({
    LayerKind.BATCH_NORM, LayerKind.LAYER_NORM,
    LayerKind.GROUP_NORM, LayerKind.INSTANCE_NORM,
})

_ACTIVATION_KINDS = frozenset({
    LayerKind.ACTIVATION, LayerKind.RELU, LayerKind.SIGMOID,
    LayerKind.TANH, LayerKind.SOFTMAX, LayerKind.GELU,
})

_RECURRENT_KINDS = frozenset({
    LayerKind.LSTM, LayerKind.GRU, LayerKind.RECURRENT,
})

# Ops that have no visual column of their own in standard NN-SVG families.
_SKIP_IN_ALL = frozenset({
    *_ACTIVATION_KINDS,
    *_NORM_KINDS,
    LayerKind.DROPOUT,
    LayerKind.PAD,
    LayerKind.ADD,
    LayerKind.CONCAT,
    LayerKind.MULTIPLY,
})

# Ops absorbed into a preceding block during Transformer block-grouping.
_TRANSFORMER_ABSORB = frozenset({
    *_NORM_KINDS,
    LayerKind.ADD,
    LayerKind.MULTIPLY,
    LayerKind.DROPOUT,
})

# ── Label helpers ────────────────────────────────────────────────────────────

_MAX_LABEL_LEN = 20

# Common ML acronyms that should be all-caps or use CamelCase.
_WORD_OVERRIDES: dict[str, str] = {
    "mlp": "MLP", "cnn": "CNN", "rnn": "RNN", "lstm": "LSTM", "gru": "GRU",
    "vgg": "VGG", "gan": "GAN", "vae": "VAE", "gpt": "GPT", "bert": "BERT",
    "resnet": "ResNet", "unet": "U-Net", "fcnn": "FCNN",
}


def _format_title(raw_name: str) -> str:
    """Convert a raw model name to a clean display title.

    Examples::

        'tiny_mlp'         -> 'Tiny MLP'
        'resnet_like'      -> 'ResNet Like'
        'transformer_like' -> 'Transformer Like'
    """
    if not raw_name:
        return "Neural Network"
    words = raw_name.replace("-", "_").split("_")
    out = [_WORD_OVERRIDES.get(w.lower(), w.capitalize()) for w in words]
    return " ".join(out)


def _truncate(text: str, max_len: int = _MAX_LABEL_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _act_name(kind: LayerKind) -> str:
    return {
        LayerKind.RELU:    "ReLU",
        LayerKind.GELU:    "GeLU",
        LayerKind.SIGMOID: "Sigmoid",
        LayerKind.TANH:    "Tanh",
        LayerKind.SOFTMAX: "Softmax",
    }.get(kind, "Act")


def _make_label(layer: SemanticLayer, show_shapes: bool) -> str:
    base = layer.name or layer.kind.name.lower()
    if show_shapes and layer.output_shape:
        dims = "x".join(str(d) for d in layer.output_shape if isinstance(d, int))
        if dims:
            return _truncate(f"{base} {dims}")
    return _truncate(base)


def _color_for(kind: LayerKind) -> str:
    if kind == LayerKind.INPUT:
        return _COLORS["input"]
    if kind in _CONV_KINDS:
        return _COLORS["conv"]
    if kind in _POOL_KINDS:
        return _COLORS["pool"]
    if kind == LayerKind.DENSE:
        return _COLORS["dense"]
    if kind == LayerKind.OUTPUT:
        return _COLORS["output"]
    if kind in _NORM_KINDS:
        return _COLORS["norm"]
    if kind in _ACTIVATION_KINDS:
        return _COLORS["activation"]
    if kind == LayerKind.ATTENTION:
        return _COLORS["attention"]
    if kind in _RECURRENT_KINDS:
        return _COLORS["recurrent"]
    return _COLORS["other"]


# ── Transformer block-level view ─────────────────────────────────────────────

def _map_transformer_blocks(
    arch: SemanticArchitecture,
    cfg: RenderConfig,
) -> list[NNSVGLayerSpec]:
    """Block-level rectangle view for Transformer/attention/recurrent architectures.

    Produces single-rectangle blocks (channels=1) rendered via the LeNet
    renderer.  Each major computation stage (attention, feed-forward, norm)
    becomes one labeled block.  Adjacent norm/add/residual ops are absorbed
    into the preceding block.

    This is a block-level approximation of the computation sequence.  It is
    NOT exact Transformer rendering — NN-SVG has no native Transformer family.
    The label inside each block identifies the operation; the full layer list
    is preserved in the debug-JSON export.
    """
    layers = arch.layers
    n = len(layers)
    specs: list[NNSVGLayerSpec] = []
    i = 0

    while i < n:
        layer = layers[i]
        kind = layer.kind

        if kind == LayerKind.INPUT:
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Input",
                channels=1, feature_map_width=42, feature_map_height=60,
                color=_COLORS["input"],
            ))
            i += 1

        elif kind == LayerKind.EMBEDDING:
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Embedding",
                channels=1, feature_map_width=48, feature_map_height=70,
                color=_COLORS["dense"],
            ))
            i += 1

        elif kind == LayerKind.ATTENTION:
            # Absorb any immediately following Norm / Add / residual ops.
            j = i + 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="[Attention]",
                channels=1, feature_map_width=54, feature_map_height=100,
                color=_COLORS["attention"],
            ))
            i = j

        elif kind == LayerKind.DENSE:
            # Greedily absorb a feed-forward block:
            # Dense + activations + optional Dense + Norm/Add.
            j = i + 1
            while j < n and layers[j].kind in (
                _ACTIVATION_KINDS | frozenset({LayerKind.DENSE, LayerKind.DROPOUT})
            ):
                j += 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1

            # Determine if this is the final classifier block.
            remaining_kinds = {layers[k].kind for k in range(j, n)}
            non_trivial = remaining_kinds - (
                _ACTIVATION_KINDS
                | frozenset({LayerKind.OUTPUT, LayerKind.SOFTMAX,
                             LayerKind.FLATTEN, LayerKind.RESHAPE, LayerKind.UNKNOWN})
            )
            is_classifier = (j >= n) or (not non_trivial)

            if is_classifier:
                specs.append(NNSVGLayerSpec(
                    layer_type="conv", label="Classifier",
                    channels=1, feature_map_width=44, feature_map_height=65,
                    color=_COLORS["output"],
                ))
            else:
                specs.append(NNSVGLayerSpec(
                    layer_type="conv", label="FeedFwd",
                    channels=1, feature_map_width=54, feature_map_height=88,
                    color=_COLORS["ffn"],
                ))
            i = j

        elif kind in _RECURRENT_KINDS:
            j = i + 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label=f"[{kind.name}]",
                channels=1, feature_map_width=54, feature_map_height=90,
                color=_COLORS["recurrent"],
            ))
            i = j

        elif kind in _NORM_KINDS:
            # Standalone norm (not absorbed by a preceding block).
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Norm",
                channels=1, feature_map_width=30, feature_map_height=48,
                color=_COLORS["norm"],
            ))
            i += 1

        else:
            i += 1  # skip structural / passthrough ops not absorbed above

    return specs


# ── FCNN mapper ──────────────────────────────────────────────────────────────

def _map_fcnn(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to FCNN: each significant layer becomes a column of neurons.

    Activations immediately following a dense/embedding layer are fused into
    the label (e.g. ``fc1 +ReLU``) rather than shown as a separate column.
    """
    layers = arch.layers
    n = len(layers)
    specs: list[NNSVGLayerSpec] = []
    i = 0

    while i < n:
        layer = layers[i]
        kind = layer.kind

        if kind in _SKIP_IN_ALL:
            i += 1
            continue
        if kind in (LayerKind.FLATTEN, LayerKind.RESHAPE,
                    LayerKind.UPSAMPLE, LayerKind.UNKNOWN):
            i += 1
            continue

        # Fuse immediately following activation into label.
        fused_act = ""
        if i + 1 < n and layers[i + 1].kind in _ACTIVATION_KINDS:
            fused_act = _act_name(layers[i + 1].kind)

        units = layer.units or layer.channels or 0
        if units == 0 and layer.output_shape:
            ints = [d for d in layer.output_shape if isinstance(d, int) and d > 0]
            if ints:
                units = ints[-1]
        if units == 0:
            units = 10

        base = _make_label(layer, cfg.show_shapes)
        label = _truncate(f"{base} +{fused_act}" if fused_act else base)

        specs.append(NNSVGLayerSpec(
            layer_type="dense",
            label=label,
            units=min(units, 256),
            color=_color_for(kind),
        ))

        i += 1
        if fused_act:
            i += 1  # skip the absorbed activation layer

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


_MAX_DENSE_UNITS_LENET   = 10
_MAX_DENSE_UNITS_ALEXNET =  8


def _map_lenet(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to LeNet view: conv/pool as feature maps, dense as neuron columns.

    Activations immediately after conv/pool/dense are fused into the label.
    """
    layers = arch.layers
    n = len(layers)
    specs: list[NNSVGLayerSpec] = []
    i = 0

    while i < n:
        layer = layers[i]
        kind = layer.kind

        if kind in _SKIP_IN_ALL:
            i += 1
            continue
        if kind in (LayerKind.FLATTEN, LayerKind.RESHAPE,
                    LayerKind.UPSAMPLE, LayerKind.UNKNOWN):
            i += 1
            continue

        fused_act = ""
        if i + 1 < n and layers[i + 1].kind in _ACTIVATION_KINDS:
            fused_act = _act_name(layers[i + 1].kind)

        base = _make_label(layer, cfg.show_shapes)
        label = _truncate(f"{base} +{fused_act}" if fused_act else base)
        absorb = bool(fused_act)

        if kind in _CONV_KINDS:
            h, w = _estimate_feature_map(layer)
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label=label,
                channels=layer.channels or 1,
                kernel_size=layer.kernel_size[0] if layer.kernel_size else 3,
                stride=layer.stride[0] if layer.stride else 1,
                feature_map_width=w or 28,
                feature_map_height=h or 28,
                color=_color_for(kind),
            ))

        elif kind in _POOL_KINDS:
            h, w = _estimate_feature_map(layer)
            prev_ch = specs[-1].channels if specs else 1
            specs.append(NNSVGLayerSpec(
                layer_type="pool", label=label,
                channels=layer.channels or prev_ch,
                kernel_size=layer.kernel_size[0] if layer.kernel_size else 2,
                stride=layer.stride[0] if layer.stride else 2,
                feature_map_width=w or 14,
                feature_map_height=h or 14,
                color=_color_for(kind),
            ))

        elif kind == LayerKind.DENSE:
            units = min(layer.units or 10, _MAX_DENSE_UNITS_LENET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=units, color=_color_for(kind),
            ))

        elif kind == LayerKind.INPUT:
            if layer.output_shape:
                h, w = _estimate_feature_map(layer)
                ints = [d for d in layer.output_shape if isinstance(d, int)]
                ch = ints[1] if len(ints) >= 4 else (ints[0] if len(ints) >= 3 else 1)
                specs.append(NNSVGLayerSpec(
                    layer_type="input", label="input",
                    channels=ch,
                    feature_map_width=w or 28,
                    feature_map_height=h or 28,
                    color=_color_for(kind),
                ))
                absorb = False  # INPUT has no trailing activation to absorb

        else:
            # Other ops → small dense block
            units = min(layer.units or layer.channels or 8, _MAX_DENSE_UNITS_LENET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=units, color=_color_for(kind),
            ))

        i += 1 + (1 if absorb else 0)

    return specs


def _map_alexnet(arch: SemanticArchitecture, cfg: RenderConfig) -> list[NNSVGLayerSpec]:
    """Map to AlexNet view: deep CNN with tightly capped dense/classifier columns."""
    layers = arch.layers
    n = len(layers)
    specs: list[NNSVGLayerSpec] = []
    i = 0

    while i < n:
        layer = layers[i]
        kind = layer.kind

        if kind in _SKIP_IN_ALL:
            i += 1
            continue
        if kind in (LayerKind.FLATTEN, LayerKind.RESHAPE,
                    LayerKind.UPSAMPLE, LayerKind.UNKNOWN):
            i += 1
            continue

        fused_act = ""
        if i + 1 < n and layers[i + 1].kind in _ACTIVATION_KINDS:
            fused_act = _act_name(layers[i + 1].kind)

        base = _make_label(layer, cfg.show_shapes)
        label = _truncate(f"{base} +{fused_act}" if fused_act else base)
        absorb = bool(fused_act)

        if kind in _CONV_KINDS:
            h, w = _estimate_feature_map(layer)
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label=label,
                channels=layer.channels or 1,
                kernel_size=layer.kernel_size[0] if layer.kernel_size else 3,
                stride=layer.stride[0] if layer.stride else 1,
                feature_map_width=w or 20,
                feature_map_height=h or 20,
                color=_color_for(kind),
            ))

        elif kind in _POOL_KINDS:
            h, w = _estimate_feature_map(layer)
            prev_ch = specs[-1].channels if specs else 1
            specs.append(NNSVGLayerSpec(
                layer_type="pool", label=label,
                channels=layer.channels or prev_ch,
                kernel_size=layer.kernel_size[0] if layer.kernel_size else 2,
                stride=layer.stride[0] if layer.stride else 2,
                feature_map_width=w or 10,
                feature_map_height=h or 10,
                color=_color_for(kind),
            ))

        elif kind == LayerKind.DENSE:
            # Tightly cap classifier to prevent visual domination.
            units = min(layer.units or 8, _MAX_DENSE_UNITS_ALEXNET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=units, color=_color_for(kind),
            ))

        elif kind == LayerKind.INPUT:
            if layer.output_shape:
                h, w = _estimate_feature_map(layer)
                ints = [d for d in layer.output_shape if isinstance(d, int)]
                ch = ints[1] if len(ints) >= 4 else (ints[0] if len(ints) >= 3 else 1)
                specs.append(NNSVGLayerSpec(
                    layer_type="input", label="input",
                    channels=ch,
                    feature_map_width=w or 20,
                    feature_map_height=h or 20,
                    color=_color_for(kind),
                ))
                absorb = False

        else:
            units = min(layer.units or layer.channels or 6, _MAX_DENSE_UNITS_ALEXNET)
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=units, color=_color_for(kind),
            ))

        i += 1 + (1 if absorb else 0)

    return specs


# ── Auto-sizing ──────────────────────────────────────────────────────────────

_MIN_PX_PER_LAYER = 80


def _auto_width(n_layers: int, configured_width: int) -> int:
    needed = n_layers * _MIN_PX_PER_LAYER + 120
    return max(configured_width, needed)


# ── Public entry point ───────────────────────────────────────────────────────

_FAMILY_MAPPERS = {
    RenderFamily.FCNN:    _map_fcnn,
    RenderFamily.LENET:   _map_lenet,
    RenderFamily.ALEXNET: _map_alexnet,
}


def map_to_nnsvg(
    arch: SemanticArchitecture,
    config: RenderConfig,
) -> NNSVGSpec:
    """Build an :class:`NNSVGSpec` from a semantic architecture and config."""
    family = config.style or arch.recommended_family or RenderFamily.FCNN

    # Detect sequential-op architectures (attention / recurrent).
    # Route them to the block-level rect view (LeNet renderer) unless the
    # user explicitly overrode the style, in which case respect their choice.
    has_sequential_op = any(
        lay.kind in ({LayerKind.ATTENTION} | _RECURRENT_KINDS)
        for lay in arch.layers
    )
    if has_sequential_op and config.style is None:
        layers = _map_transformer_blocks(arch, config)
        family = RenderFamily.LENET   # LeNet renderer draws single-rect blocks
    else:
        layers = _FAMILY_MAPPERS[family](arch, config)

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
        title=config.title or _format_title(arch.model_name),
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
