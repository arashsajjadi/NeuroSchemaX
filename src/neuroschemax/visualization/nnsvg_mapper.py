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


def _fmt_kern(k: list[int] | None) -> str:
    """Format kernel_size as 'k3' (square) or 'k3x5' (non-square)."""
    if not k:
        return ""
    if len(k) == 1 or k[0] == k[1]:
        return f"k{k[0]}"
    return "k" + "x".join(str(v) for v in k)


def _fmt_stride(s: list[int] | None) -> str:
    """Format stride as 's2' only when stride is non-trivial (> 1)."""
    if not s or all(v == 1 for v in s):
        return ""
    if len(s) == 1 or s[0] == s[1]:
        return f"s{s[0]}"
    return "s" + "x".join(str(v) for v in s)


def _compact_op_label(layer: SemanticLayer) -> str:
    """Operation-aware compact label used by label_mode='compact'/'auto'.

    Produces professional labels like ``Conv 64 k3``, ``MaxP k2 s2``,
    ``Dense 128``, ``GAP``, ``Input 28x28``, ``[MH-Attn]`` — the operation
    type and key parameters, not the arbitrary layer name.
    """
    kind = layer.kind

    if kind in _CONV_KINDS:
        parts = ["Conv"]
        if layer.channels:
            parts.append(str(layer.channels))
        kern = _fmt_kern(layer.kernel_size)
        if kern:
            parts.append(kern)
        strd = _fmt_stride(layer.stride)
        if strd:
            parts.append(strd)
        return " ".join(parts)

    if kind == LayerKind.POOL_MAX:
        parts = ["MaxP"]
        kern = _fmt_kern(layer.kernel_size)
        if kern:
            parts.append(kern)
        strd = _fmt_stride(layer.stride)
        if strd:
            parts.append(strd)
        return " ".join(parts)

    if kind == LayerKind.POOL_AVG:
        parts = ["AvgP"]
        kern = _fmt_kern(layer.kernel_size)
        if kern:
            parts.append(kern)
        strd = _fmt_stride(layer.stride)
        if strd:
            parts.append(strd)
        return " ".join(parts)

    if kind == LayerKind.POOL_GLOBAL:
        return "GAP"

    if kind == LayerKind.DENSE:
        return f"Dense {layer.units}" if layer.units else "Dense"

    if kind == LayerKind.INPUT:
        shape = layer.output_shape
        ints = [d for d in shape if isinstance(d, int) and d > 0]
        if len(ints) >= 3:
            hw = "x".join(str(d) for d in ints[-2:])
            return f"Input {hw}"
        if ints:
            return f"Input {ints[-1]}"
        return "Input"

    if kind == LayerKind.ATTENTION:
        return "[MH-Attn]"

    if kind in _RECURRENT_KINDS:
        return f"[{kind.name}]"

    if kind == LayerKind.EMBEDDING:
        return "Emb"

    if kind == LayerKind.UPSAMPLE:
        scale = layer.attributes.get("scale_factor", layer.attributes.get("scales"))
        if scale is not None:
            if isinstance(scale, (int, float)):
                return f"Up x{int(scale)}"
            if isinstance(scale, (list, tuple)) and len(scale) >= 2:
                return f"Up x{int(scale[-1])}"
        return "Upsample"

    # Fallback: use the layer name for unrecognised op types.
    return layer.name or kind.name.lower()


def _make_label(layer: SemanticLayer, label_mode: str) -> str:
    """Generate a display label for *layer* according to *label_mode*.

    Modes:
      name    — layer name only; shortest, never overlaps
      shape   — most-relevant dimension (HxW or units), no name
      compact — operation-aware: ``Conv 64 k3``, ``Dense 128``, ``GAP``
      full    — layer name + complete shape string
      auto    — resolved to compact or name before this function is called
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

    # "compact" (and "auto" after resolution) → operation-aware format
    return _truncate(_compact_op_label(layer))


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
    """Group consecutive conv/pool layers into informative 'Block N' ch=1 boxes.

    Uses ch=1 (single-rectangle) blocks so LeNet.js can render multi-line
    labels inside the box.  Each block label shows the conv count, output
    channels, and a ↓ marker when a pool layer is present.

    Example block label:   ``Block 2\\n4cv 128ch ↓``
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

        elif l.layer_type == "conv":
            # Absorb consecutive convs, then include one pool if present.
            # Splitting at pool boundaries gives meaningful per-stage blocks.
            j = i + 1
            while j < n and spec_layers[j].layer_type == "conv":
                j += 1
            has_pool = j < n and spec_layers[j].layer_type == "pool"
            if has_pool:
                j += 1  # include the pool in this block

            block_num += 1
            group = spec_layers[i: j - (1 if has_pool else 0)]
            conv_layers = [s for s in group if s.layer_type == "conv"]
            last_conv = conv_layers[-1] if conv_layers else group[-1]
            n_conv = len(conv_layers)

            line2_parts: list[str] = []
            if n_conv:
                line2_parts.append(f"{n_conv}cv")
            if last_conv.channels:
                line2_parts.append(f"{last_conv.channels}ch")
            if has_pool:
                line2_parts.append("↓")
            label = f"Block {block_num}"
            if line2_parts:
                label += "\n" + " ".join(line2_parts)

            result.append(NNSVGLayerSpec(
                layer_type="conv",
                label=label,
                channels=1,          # ch=1 = box rendering with inside label
                feature_map_width=80,
                feature_map_height=72,
                color=last_conv.color,
            ))
            i = j

        elif l.layer_type == "pool":
            # Standalone pool not absorbed by a preceding conv group — skip it.
            i += 1

        elif l.layer_type == "dense":
            # All remaining dense layers → single Classifier box.
            result.append(NNSVGLayerSpec(
                layer_type="conv",
                label="Classifier",
                channels=1,
                feature_map_width=72,
                feature_map_height=60,
                color=_COLORS["output"],
            ))
            break

        else:
            result.append(l)
            i += 1

    return result


def _summarize_residual(
    spec_layers: list[NNSVGLayerSpec],
    arch: SemanticArchitecture,
) -> list[NNSVGLayerSpec]:
    """Block summary for ResNet/U-Net architectures with merge ops.

    Produces ch=1 box blocks with multi-line informative labels:
      ResNet: Input → Stem → Res Block 1 → ... → Res Block N → Head
      U-Net:  Input → Encoder\\n↓conv → Bottleneck → Decoder\\n↑conv → Output
    """
    has_concat = any(lay.kind == LayerKind.CONCAT for lay in arch.layers)
    result: list[NNSVGLayerSpec] = []

    # Preserve input block if present.
    if spec_layers and spec_layers[0].layer_type == "input":
        result.append(spec_layers[0])

    if has_concat:
        # U-Net style encoder-decoder with multi-line box blocks.
        result += [
            NNSVGLayerSpec(
                layer_type="conv", label="Encoder\n↓conv",
                channels=1, feature_map_width=80, feature_map_height=80,
                color=_COLORS["conv"],
            ),
            NNSVGLayerSpec(
                layer_type="conv", label="Bottleneck",
                channels=1, feature_map_width=72, feature_map_height=60,
                color=_COLORS["pool"],
            ),
            NNSVGLayerSpec(
                layer_type="conv", label="Decoder\n↑conv",
                channels=1, feature_map_width=80, feature_map_height=80,
                color=_COLORS["conv"],
            ),
            NNSVGLayerSpec(
                layer_type="conv", label="Output",
                channels=1, feature_map_width=64, feature_map_height=56,
                color=_COLORS["output"],
            ),
        ]
    else:
        # ResNet style — group conv layers between Add ops.
        add_count = sum(1 for lay in arch.layers if lay.kind == LayerKind.ADD)
        # Stem: ch=1 box so multi-line label renders inside
        result.append(NNSVGLayerSpec(
            layer_type="conv", label="Stem\nconv",
            channels=1, feature_map_width=72, feature_map_height=64,
            color=_COLORS["conv"],
        ))
        for b in range(max(1, add_count)):
            result.append(NNSVGLayerSpec(
                layer_type="conv", label=f"Res Block {b + 1}\n2×conv",
                channels=1, feature_map_width=80, feature_map_height=80,
                color=_COLORS["norm"],   # purple-ish marks residual blocks
            ))
        result.append(NNSVGLayerSpec(
            layer_type="conv", label="Head",
            channels=1, feature_map_width=64, feature_map_height=56,
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

    Each stage becomes a ch=1 rectangle with a multi-line label centered inside.
    LeNet.js renders these labels with auto font-scaling.

    Stages shown:
      Input → Embedding → [PosEnc] → [MH-Attn]\\nAdd & Norm →
      [FFN]\\nAdd & Norm → ... → [Head]

    NOT exact Transformer rendering.  Q/K/V projections, individual attention
    heads, exact residual paths, and tensor flow are NOT drawn.
    Full layer list is preserved in the debug-JSON export.
    """
    layers = arch.layers
    n = len(layers)
    specs: list[NNSVGLayerSpec] = []
    i = 0
    first_attn_seen = False

    while i < n:
        kind = layers[i].kind

        if kind == LayerKind.INPUT:
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Tokens\nInput",
                channels=1, feature_map_width=72, feature_map_height=64,
                color=_COLORS["input"],
            ))
            i += 1

        elif kind == LayerKind.EMBEDDING:
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Embedding",
                channels=1, feature_map_width=80, feature_map_height=72,
                color=_COLORS["dense"],
            ))
            i += 1

        elif kind == LayerKind.ADD and not first_attn_seen:
            # ADD before the first attention → likely positional encoding.
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="PosEnc",
                channels=1, feature_map_width=72, feature_map_height=60,
                color=_COLORS["norm"],
            ))
            i += 1

        elif kind == LayerKind.ATTENTION:
            first_attn_seen = True
            # Absorb following Add/Norm (residual sub-layer).
            j = i + 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            # Multi-line label: first line is the main op, second is the residual.
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="[MH-Attn]\nAdd & Norm",
                channels=1, feature_map_width=88, feature_map_height=108,
                color=_COLORS["attention"],
            ))
            i = j

        elif kind == LayerKind.DENSE:
            # Absorb FFN block non-greedily:
            #   Dense-up → optional (Act + Dense-down) → optional Add/Norm
            # A second Dense is only absorbed when an activation separated the two,
            # which is the standard FFN up-project → activate → down-project pattern.
            # This prevents the classifier (a separate trailing Dense) from being
            # swallowed into the same block as the FFN.
            j = i + 1
            activation_seen = False
            while j < n and layers[j].kind in (_ACTIVATION_KINDS | frozenset({LayerKind.DROPOUT})):
                j += 1
                activation_seen = True
            if activation_seen and j < n and layers[j].kind == LayerKind.DENSE:
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
            if is_classifier:
                specs.append(NNSVGLayerSpec(
                    layer_type="conv", label="[Head]\nClassifier",
                    channels=1, feature_map_width=80, feature_map_height=72,
                    color=_COLORS["output"],
                ))
            else:
                specs.append(NNSVGLayerSpec(
                    layer_type="conv", label="[FFN]\nAdd & Norm",
                    channels=1, feature_map_width=80, feature_map_height=92,
                    color=_COLORS["ffn"],
                ))
            i = j

        elif kind in _RECURRENT_KINDS:
            j = i + 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label=f"[{kind.name}]\nAdd & Norm",
                channels=1, feature_map_width=80, feature_map_height=92,
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
