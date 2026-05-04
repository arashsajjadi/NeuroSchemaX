"""Map a :class:`SemanticArchitecture` to an :class:`NNSVGSpec`.

This is the bridge between the framework-agnostic semantic representation
and the NN-SVG rendering engine.  The mapping is family-aware: FCNN, LeNet,
and AlexNet each have different layer-spec expectations.

For Transformer/attention and recurrent architectures, a block-level
approximation is produced using single-rectangle blocks via the LeNet renderer.
This is NOT exact Transformer rendering — NN-SVG has no native Transformer
family.  The block sequence is correct; Q/K/V flows, residual paths, and
repeated-block groupings are approximated visually and preserved in full in
the debug-JSON export.

Rendering controls (all accepted as RenderConfig fields or **kwargs):
  label_mode:       "auto" | "name" | "compact" | "shape" | "full"
  detail_level:     "auto" | "summary" | "full"
  show_activations: True | False
  transformer_mode: "block_summary" | "unsupported"
  approximate_mode: "warn" | "error" | "allow"
"""

from __future__ import annotations

from ..core.config import RenderConfig
from ..core.enums import LayerKind, RenderFamily
from ..exceptions import RenderError
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

# ── Kind groups ──────────────────────────────────────────────────────────────

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
_SKIP_IN_ALL = frozenset({
    *_ACTIVATION_KINDS, *_NORM_KINDS,
    LayerKind.DROPOUT, LayerKind.PAD,
    LayerKind.ADD, LayerKind.CONCAT, LayerKind.MULTIPLY,
})
_TRANSFORMER_ABSORB = frozenset({
    *_NORM_KINDS,
    LayerKind.ADD, LayerKind.MULTIPLY, LayerKind.DROPOUT,
})

# ── Title/subtitle helpers ───────────────────────────────────────────────────

_WORD_OVERRIDES: dict[str, str] = {
    "mlp": "MLP", "cnn": "CNN", "rnn": "RNN", "lstm": "LSTM", "gru": "GRU",
    "vgg": "VGG", "gan": "GAN", "vae": "VAE", "gpt": "GPT", "bert": "BERT",
    "resnet": "ResNet", "unet": "U-Net", "fcnn": "FCNN",
}
_FAMILY_DISPLAY: dict[RenderFamily, str] = {
    RenderFamily.FCNN:    "FCNN",
    RenderFamily.LENET:   "LeNet-style",
    RenderFamily.ALEXNET: "AlexNet-style",
}


def _format_title(raw_name: str) -> str:
    """'tiny_cnn' → 'Tiny CNN',  'resnet_like' → 'ResNet Like'."""
    if not raw_name:
        return "Neural Network"
    words = raw_name.replace("-", "_").split("_")
    return " ".join(_WORD_OVERRIDES.get(w.lower(), w.capitalize()) for w in words)


def _make_subtitle(
    family: RenderFamily,
    arch: SemanticArchitecture,
    has_approx: bool,
    is_transformer_block: bool = False,
) -> str:
    """Build a one-line metadata subtitle for the diagram header."""
    parts: list[str] = []
    if is_transformer_block:
        parts.append("Transformer block summary")
    else:
        parts.append(_FAMILY_DISPLAY.get(family, family.value))
    parts.append("approximate" if has_approx else "exact")
    parts.append(f"{arch.layer_count} layers")
    return " · ".join(parts)


# ── Label helpers ────────────────────────────────────────────────────────────

_MAX_LABEL = 20


def _truncate(text: str, max_len: int = _MAX_LABEL) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _act_name(kind: LayerKind) -> str:
    return {
        LayerKind.RELU:    "ReLU",
        LayerKind.GELU:    "GeLU",
        LayerKind.SIGMOID: "Sigmoid",
        LayerKind.TANH:    "Tanh",
        LayerKind.SOFTMAX: "Softmax",
    }.get(kind, "Act")


def _make_label(layer: SemanticLayer, label_mode: str) -> str:
    """Generate a display label for *layer* according to *label_mode*.

    Modes:
      name    — layer name only; never overlaps
      shape   — most-relevant dimension only (HxW or units)
      compact — short name + most-relevant dimension
      full    — name + complete shape string
      auto    — compact (resolved before calling this function)
    """
    name = layer.name or layer.kind.name.lower()
    shape = layer.output_shape
    ints = [d for d in shape if isinstance(d, int) and d > 0] if shape else []

    if label_mode == "name":
        return _truncate(name)

    if label_mode == "shape":
        if len(ints) >= 3:
            return _truncate("x".join(str(d) for d in ints[-2:]))
        if ints:
            return _truncate(str(ints[-1]))
        return _truncate(name)

    if label_mode == "full":
        if ints:
            return _truncate(f"{name} " + "x".join(str(d) for d in ints))
        return _truncate(name)

    # "compact" (and "auto" after resolution)
    if len(ints) >= 3:
        hw = "x".join(str(d) for d in ints[-2:])
        return _truncate(f"{name} {hw}")
    if ints:
        return _truncate(f"{name} {ints[-1]}")
    return _truncate(name)


def _effective_label_mode(cfg: RenderConfig, arch_layer_count: int) -> str:
    """Resolve 'auto' label mode to a concrete mode based on model size.

    Threshold is conservative: compact labels (name + shape) are only used for
    small models where the canvas gives enough horizontal space per layer.
    For 10+ arch layers we switch to name-only labels to avoid overlap.
    """
    if cfg.label_mode != "auto":
        return cfg.label_mode
    return "compact" if arch_layer_count <= 9 else "name"


_AVG_CHAR_PX = 0.62   # fraction of font_size per character (conservative estimate)
_LABEL_SLOT_FRAC = 0.78  # use ≤ 78 % of each slot's width for text


def _safe_label_policy(
    spec_layers: list[NNSVGLayerSpec],
    canvas_width: int,
    font_size: int,
    allow_thinning: bool = True,
) -> list[NNSVGLayerSpec]:
    """Guarantee that labels fit within their canvas slots without overlap.

    Algorithm:
    1. Estimate the available pixel width per layer slot.
    2. Derive a safe character budget from that width and the font size.
    3. If any label exceeds the budget:
       - Truncate every label to the budget.
       - If the budget is still very tight (< 5 chars) AND thinning is allowed,
         also thin labels (show every other one) to further reduce clutter.

    This is a post-processing safety net.  The primary defence is choosing the
    right label_mode; this catches edge cases that slip through.

    Labels for transformer block layers (ch == 1 rects) are skipped because the
    LeNet.js renderer auto-scales font size to fit inside the rectangle.
    """
    n = len(spec_layers)
    if n <= 1:
        return spec_layers

    slot_px = max(1.0, (canvas_width - 120) / n)
    safe_label_px = slot_px * _LABEL_SLOT_FRAC
    char_px = font_size * _AVG_CHAR_PX
    safe_chars = max(3, int(safe_label_px / char_px))

    # Skip layers whose labels are already inside boxes (ch == 1 blocks).
    # Those are handled by JS auto-scaling.
    def _is_box_layer(l: NNSVGLayerSpec) -> bool:
        return l.layer_type != "dense" and l.channels == 1

    needs_action = any(
        len(l.label) > safe_chars
        for l in spec_layers
        if l.label and not _is_box_layer(l)
    )
    if not needs_action:
        return spec_layers

    # Step 1: truncate long labels.
    for l in spec_layers:
        if l.label and not _is_box_layer(l) and len(l.label) > safe_chars:
            l.label = _truncate(l.label, safe_chars)

    # Step 2: thin if budget is very tight and thinning is permitted.
    if allow_thinning and safe_chars < 5:
        for i, l in enumerate(spec_layers):
            if i % 2 != 0 and l.layer_type not in ("input",) and not _is_box_layer(l):
                l.label = ""

    return spec_layers


def _color_for(kind: LayerKind) -> str:
    if kind == LayerKind.INPUT:      return _COLORS["input"]
    if kind in _CONV_KINDS:          return _COLORS["conv"]
    if kind in _POOL_KINDS:          return _COLORS["pool"]
    if kind == LayerKind.DENSE:      return _COLORS["dense"]
    if kind == LayerKind.OUTPUT:     return _COLORS["output"]
    if kind in _NORM_KINDS:          return _COLORS["norm"]
    if kind in _ACTIVATION_KINDS:    return _COLORS["activation"]
    if kind == LayerKind.ATTENTION:  return _COLORS["attention"]
    if kind in _RECURRENT_KINDS:     return _COLORS["recurrent"]
    return _COLORS["other"]


# ── Detail-level summary grouping ────────────────────────────────────────────

def _summarize_sequential(spec_layers: list[NNSVGLayerSpec]) -> list[NNSVGLayerSpec]:
    """Group consecutive conv/pool layers into labeled 'Block N' entries.

    Dense/classifier layers are collapsed into a single 'Classifier' block.
    Input layers are preserved as-is.
    """
    result: list[NNSVGLayerSpec] = []
    i = 0
    n = len(spec_layers)
    block_num = 0

    while i < n:
        l = spec_layers[i]

        if l.layer_type == "input":
            result.append(l)
            i += 1

        elif l.layer_type in ("conv", "pool"):
            j = i + 1
            while j < n and spec_layers[j].layer_type in ("conv", "pool"):
                j += 1
            block_num += 1
            # Use the deepest conv in this group for visual properties.
            last_conv = next(
                (s for s in reversed(spec_layers[i:j]) if s.layer_type == "conv"),
                spec_layers[i],
            )
            result.append(NNSVGLayerSpec(
                layer_type="conv",
                label=f"Block {block_num}",
                channels=min(last_conv.channels, 5),
                feature_map_width=max(last_conv.feature_map_width, 14),
                feature_map_height=max(last_conv.feature_map_height, 14),
                color=last_conv.color,
            ))
            i = j

        elif l.layer_type == "dense":
            # All remaining dense layers → single Classifier block.
            result.append(NNSVGLayerSpec(
                layer_type="dense",
                label="Classifier",
                units=min(l.units, _MAX_DENSE_UNITS_ALEXNET),
                color=_COLORS["output"],
            ))
            break  # consume the rest

        else:
            result.append(l)
            i += 1

    return result


def _summarize_residual(
    spec_layers: list[NNSVGLayerSpec],
    arch: SemanticArchitecture,
) -> list[NNSVGLayerSpec]:
    """Block summary for ResNet/U-Net style architectures with merge ops.

    Produces:
      ResNet: Input → Stem → Res Block 1 → ... → Res Block N → Head
      U-Net:  Input → Encoder → Bottleneck → Decoder → Output
    """
    has_concat = any(lay.kind == LayerKind.CONCAT for lay in arch.layers)
    result: list[NNSVGLayerSpec] = []

    # Preserve input block if present.
    if spec_layers and spec_layers[0].layer_type == "input":
        result.append(spec_layers[0])

    if has_concat:
        # U-Net style encoder-decoder
        result += [
            NNSVGLayerSpec(
                layer_type="conv", label="Encoder",
                channels=4, feature_map_width=44, feature_map_height=72,
                color=_COLORS["conv"],
            ),
            NNSVGLayerSpec(
                layer_type="conv", label="Bottleneck",
                channels=5, feature_map_width=24, feature_map_height=38,
                color=_COLORS["pool"],
            ),
            NNSVGLayerSpec(
                layer_type="conv", label="Decoder",
                channels=4, feature_map_width=44, feature_map_height=72,
                color=_COLORS["conv"],
            ),
        ]
        has_dense = any(l.layer_type == "dense" for l in spec_layers)
        result.append(NNSVGLayerSpec(
            layer_type="dense" if has_dense else "conv",
            label="Output",
            units=6, channels=1, feature_map_width=28, feature_map_height=44,
            color=_COLORS["output"],
        ))
    else:
        # ResNet style
        add_count = sum(1 for lay in arch.layers if lay.kind == LayerKind.ADD)
        result.append(NNSVGLayerSpec(
            layer_type="conv", label="Stem",
            channels=2, feature_map_width=30, feature_map_height=50,
            color=_COLORS["conv"],
        ))
        for b in range(max(1, add_count)):
            result.append(NNSVGLayerSpec(
                layer_type="conv", label=f"Res Block {b + 1}",
                channels=3, feature_map_width=28, feature_map_height=46,
                color=_COLORS["norm"],  # purple-ish to distinguish from plain conv
            ))
        has_dense = any(l.layer_type == "dense" for l in spec_layers)
        result.append(NNSVGLayerSpec(
            layer_type="dense" if has_dense else "conv",
            label="Head",
            units=6, channels=1, feature_map_width=20, feature_map_height=32,
            color=_COLORS["output"],
        ))

    return result


def _apply_detail_level(
    spec_layers: list[NNSVGLayerSpec],
    arch: SemanticArchitecture,
    detail_level: str,
) -> list[NNSVGLayerSpec]:
    """Apply summary grouping according to *detail_level*."""
    effective = detail_level
    if detail_level == "auto" and len(spec_layers) > 12:
        effective = "summary"

    if effective != "summary":
        return spec_layers

    # Check for merge operations in the original arch.
    has_merge = any(
        lay.kind in (LayerKind.ADD, LayerKind.CONCAT, LayerKind.MULTIPLY)
        for lay in arch.layers
    )

    if has_merge:
        return _summarize_residual(spec_layers, arch)
    return _summarize_sequential(spec_layers)


# ── Transformer block-level view ─────────────────────────────────────────────

def _map_transformer_blocks(
    arch: SemanticArchitecture,
    cfg: RenderConfig,
) -> list[NNSVGLayerSpec]:
    """Block-level rectangle approximation for Transformer/attention/recurrent.

    Uses single-channel rectangles (channels=1) via the LeNet renderer.
    Each stage (Attention, FeedFwd, Norm, Classifier) → one labeled block.
    Labels are rendered centered inside the block by LeNet.js.

    NOT exact Transformer rendering.  Full layer list in debug-JSON export.
    """
    layers = arch.layers
    n = len(layers)
    specs: list[NNSVGLayerSpec] = []
    i = 0

    while i < n:
        kind = layers[i].kind

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
                channels=1, feature_map_width=72, feature_map_height=72,
                color=_COLORS["dense"],
            ))
            i += 1

        elif kind == LayerKind.ATTENTION:
            j = i + 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            # fmW=80 ensures "[Attention]" (11 chars) renders at ≥9pt inside the box.
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="[Attention]",
                channels=1, feature_map_width=80, feature_map_height=104,
                color=_COLORS["attention"],
            ))
            i = j

        elif kind == LayerKind.DENSE:
            # Absorb feed-forward block: Dense + activations + Dense + Norm/Add.
            j = i + 1
            while j < n and layers[j].kind in (
                _ACTIVATION_KINDS | frozenset({LayerKind.DENSE, LayerKind.DROPOUT})
            ):
                j += 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            remaining = {layers[k].kind for k in range(j, n)}
            non_trivial = remaining - (
                _ACTIVATION_KINDS
                | frozenset({LayerKind.OUTPUT, LayerKind.SOFTMAX,
                             LayerKind.FLATTEN, LayerKind.RESHAPE, LayerKind.UNKNOWN})
            )
            is_classifier = (j >= n) or not non_trivial
            # fmW=72 gives enough room for "FeedFwd" (7) and "Classifier" (10).
            specs.append(NNSVGLayerSpec(
                layer_type="conv",
                label="Classifier" if is_classifier else "FeedFwd",
                channels=1,
                feature_map_width=72,
                feature_map_height=70 if is_classifier else 90,
                color=_COLORS["output"] if is_classifier else _COLORS["ffn"],
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
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Norm",
                channels=1, feature_map_width=30, feature_map_height=48,
                color=_COLORS["norm"],
            ))
            i += 1

        else:
            i += 1

    return specs


# ── FCNN mapper ──────────────────────────────────────────────────────────────

def _map_fcnn(
    arch: SemanticArchitecture,
    cfg: RenderConfig,
    lm: str,
) -> list[NNSVGLayerSpec]:
    """Each significant layer → a column of neurons.

    Activations are fused into the preceding label when show_activations=True.
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
        if (
            cfg.show_activations
            and i + 1 < n
            and layers[i + 1].kind in _ACTIVATION_KINDS
        ):
            fused_act = _act_name(layers[i + 1].kind)

        units = layer.units or layer.channels or 0
        if units == 0 and layer.output_shape:
            ints = [d for d in layer.output_shape if isinstance(d, int) and d > 0]
            if ints:
                units = ints[-1]
        if units == 0:
            units = 10

        base = _make_label(layer, lm)
        label = _truncate(f"{base} +{fused_act}" if fused_act else base)

        specs.append(NNSVGLayerSpec(
            layer_type="dense",
            label=label,
            units=min(units, 256),
            color=_color_for(kind),
        ))

        i += 1 + (1 if fused_act else 0)

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


def _map_lenet(
    arch: SemanticArchitecture,
    cfg: RenderConfig,
    lm: str,
) -> list[NNSVGLayerSpec]:
    """Conv/pool → feature-map stacks; dense → neuron columns."""
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
        if (
            cfg.show_activations
            and i + 1 < n
            and layers[i + 1].kind in _ACTIVATION_KINDS
        ):
            fused_act = _act_name(layers[i + 1].kind)

        base = _make_label(layer, lm)
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
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(layer.units or 10, _MAX_DENSE_UNITS_LENET),
                color=_color_for(kind),
            ))

        elif kind == LayerKind.INPUT:
            if layer.output_shape:
                h, w = _estimate_feature_map(layer)
                ints = [d for d in layer.output_shape if isinstance(d, int)]
                ch = ints[1] if len(ints) >= 4 else (ints[0] if len(ints) >= 3 else 1)
                specs.append(NNSVGLayerSpec(
                    layer_type="input", label="input",
                    channels=ch, feature_map_width=w or 28,
                    feature_map_height=h or 28, color=_color_for(kind),
                ))
                absorb = False

        else:
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(layer.units or layer.channels or 8, _MAX_DENSE_UNITS_LENET),
                color=_color_for(kind),
            ))

        i += 1 + (1 if absorb else 0)

    return specs


def _map_alexnet(
    arch: SemanticArchitecture,
    cfg: RenderConfig,
    lm: str,
) -> list[NNSVGLayerSpec]:
    """Deep CNN: like LeNet but tighter dense cap for classifier sections."""
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
        if (
            cfg.show_activations
            and i + 1 < n
            and layers[i + 1].kind in _ACTIVATION_KINDS
        ):
            fused_act = _act_name(layers[i + 1].kind)

        base = _make_label(layer, lm)
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
            # Tightly cap dense to prevent classifier domination.
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(layer.units or 8, _MAX_DENSE_UNITS_ALEXNET),
                color=_color_for(kind),
            ))

        elif kind == LayerKind.INPUT:
            if layer.output_shape:
                h, w = _estimate_feature_map(layer)
                ints = [d for d in layer.output_shape if isinstance(d, int)]
                ch = ints[1] if len(ints) >= 4 else (ints[0] if len(ints) >= 3 else 1)
                specs.append(NNSVGLayerSpec(
                    layer_type="input", label="input",
                    channels=ch, feature_map_width=w or 20,
                    feature_map_height=h or 20, color=_color_for(kind),
                ))
                absorb = False

        else:
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(layer.units or layer.channels or 6, _MAX_DENSE_UNITS_ALEXNET),
                color=_color_for(kind),
            ))

        i += 1 + (1 if absorb else 0)

    return specs


# ── Auto-sizing ──────────────────────────────────────────────────────────────

_MIN_PX_PER_LAYER = 80


def _auto_width(n_layers: int, configured_width: int) -> int:
    return max(configured_width, n_layers * _MIN_PX_PER_LAYER + 120)


# ── Public entry point ───────────────────────────────────────────────────────

def map_to_nnsvg(
    arch: SemanticArchitecture,
    config: RenderConfig,
) -> NNSVGSpec:
    """Build an :class:`NNSVGSpec` from a semantic architecture and config."""

    # ── approximate_mode: "error" raises before rendering ────────────────
    if arch.warnings and config.approximate_mode == "error":
        raise RenderError(
            f"Architecture {arch.model_name!r} requires approximate rendering "
            f"(confidence: {arch.family_confidence.value}).  "
            "Set approximate_mode='warn' or 'allow' to proceed."
        )

    family = config.style or arch.recommended_family or RenderFamily.FCNN

    # ── Resolve effective label mode ─────────────────────────────────────
    lm = _effective_label_mode(config, len(arch.layers))

    # ── Detect sequential-op architectures (attention / recurrent) ───────
    has_seq_op = any(
        lay.kind in ({LayerKind.ATTENTION} | _RECURRENT_KINDS)
        for lay in arch.layers
    )

    is_transformer_block = False
    if has_seq_op and config.style is None:
        if config.transformer_mode == "unsupported":
            layers = [NNSVGLayerSpec(
                layer_type="conv",
                label="(not supported)",
                channels=1,
                feature_map_width=80,
                feature_map_height=40,
                color=_COLORS["other"],
            )]
            family = RenderFamily.LENET
        else:
            layers = _map_transformer_blocks(arch, config)
            family = RenderFamily.LENET
            is_transformer_block = True
    else:
        dispatch = {
            RenderFamily.FCNN:    _map_fcnn,
            RenderFamily.LENET:   _map_lenet,
            RenderFamily.ALEXNET: _map_alexnet,
        }
        layers = dispatch[family](arch, config, lm)

    if not layers:
        layers = [NNSVGLayerSpec(layer_type="dense", label="(empty)", units=1)]

    # ── Apply detail-level grouping ───────────────────────────────────────
    if not is_transformer_block:
        layers = _apply_detail_level(layers, arch, config.detail_level)

    # ── Auto-size canvas before label safety ──────────────────────────────
    auto_width = _auto_width(len(layers), config.width)

    # ── Label safety: prevent overlap and clipping ────────────────────────
    # Skip for transformer blocks — JS auto-scales font inside rectangles.
    # Skip thinning when the user explicitly requested full detail.
    if not is_transformer_block:
        allow_thin = config.detail_level != "full"
        layers = _safe_label_policy(layers, auto_width, config.font_size, allow_thin)

    # ── Subtitle ──────────────────────────────────────────────────────────
    has_approx = bool(arch.warnings) and config.approximate_mode != "allow"
    subtitle = _make_subtitle(family, arch, has_approx, is_transformer_block)

    # ── Suppress warning badges when approximate_mode="allow" ────────────
    warnings_for_html = list(arch.warnings) if config.approximate_mode != "allow" else []

    return NNSVGSpec(
        family=family,
        layers=layers,
        width=auto_width,
        height=config.height,
        show_labels=config.show_labels,
        show_shapes=config.show_shapes,
        title=config.title or _format_title(arch.model_name),
        subtitle=subtitle,
        edge_opacity=config.edge_opacity,
        node_size=config.node_size,
        spacing=config.spacing,
        between_layers_spacing=config.between_layers_spacing,
        font_family=config.font_family,
        font_size=config.font_size,
        color_fill=config.color_fill,
        color_stroke=config.color_stroke,
        model_name=arch.model_name,
        warnings=warnings_for_html,
    )
