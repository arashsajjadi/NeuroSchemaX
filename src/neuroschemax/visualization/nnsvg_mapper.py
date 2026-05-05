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
    # Activations and Norm/Dropout are absorbed by _peek_fused_decoration when
    # adjacent to a major op.  This frozenset is the safety net for stray
    # supporting ops that have no preceding host op (e.g. BN as the first layer).
    *_ACTIVATION_KINDS, *_NORM_KINDS,
    LayerKind.DROPOUT, LayerKind.PAD,
    LayerKind.ADD, LayerKind.CONCAT, LayerKind.MULTIPLY,
})
_TRANSFORMER_ABSORB = frozenset({
    *_NORM_KINDS,
    LayerKind.ADD, LayerKind.MULTIPLY, LayerKind.DROPOUT,
})

# Supporting ops that should be surfaced as inline "+badges" rather than
# drawn as separate visual stages.  These are absorbed by the preceding
# major op (Conv / Dense / Pool) when the user has not opted out.
_BADGE_KINDS = frozenset({
    LayerKind.BATCH_NORM, LayerKind.LAYER_NORM,
    LayerKind.GROUP_NORM, LayerKind.INSTANCE_NORM,
    LayerKind.DROPOUT,
})


def _badge_text(layer: SemanticLayer) -> str:
    """Short badge string for a supporting op (BN, LN, Drop 0.5)."""
    kind = layer.kind
    if kind == LayerKind.BATCH_NORM:
        return "BN"
    if kind == LayerKind.LAYER_NORM:
        return "LN"
    if kind == LayerKind.GROUP_NORM:
        return "GN"
    if kind == LayerKind.INSTANCE_NORM:
        return "IN"
    if kind == LayerKind.DROPOUT:
        rate = layer.attributes.get("rate") or layer.attributes.get("p")
        if isinstance(rate, (int, float)) and 0 < rate < 1:
            return f"Drop {rate:g}"
        return "Drop"
    return ""

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
    n_spec_layers: int = 0,
) -> str:
    """Build a metadata subtitle for the diagram header.

    When the number of visual stages (spec layers) differs from the original
    layer count, both are shown so users understand how much was grouped.
    """
    parts: list[str] = []
    if is_transformer_block:
        parts.append("Transformer block summary")
    else:
        parts.append(_FAMILY_DISPLAY.get(family, family.value))
    parts.append("approximate" if has_approx else "exact")
    parts.append(f"{arch.layer_count} layers")
    if n_spec_layers > 0 and n_spec_layers != arch.layer_count:
        parts.append(f"{n_spec_layers} visual stages")
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
_LABEL_SLOT_FRAC = 0.92  # use ≤ 92 % of each slot's width for text (multi-line is safe)


def _wrap_label(text: str, max_chars: int) -> str:
    """Wrap *text* onto multiple lines so no line exceeds *max_chars*.

    Splits on whitespace and on '+' boundaries (so badge-decorated labels
    like ``Dense 256 +ReLU +Drop 0.5`` break before each badge instead of
    being cut mid-token).  Important tokens — ``ReLU``, ``+Drop 0.5``,
    ``k3``, ``s2``, ``Pool ↓2``, ``1000 classes``, ``12 heads``, ``d=768``
    — are preserved whole.

    Pre-existing newlines are honoured as hard line breaks.
    """
    if not text or max_chars <= 0:
        return text

    out_lines: list[str] = []
    for hard_line in text.split("\n"):
        if len(hard_line) <= max_chars:
            out_lines.append(hard_line)
            continue
        # Break before each "+" badge first (keeps "ReLU"/"Drop 0.5" intact).
        chunks = _split_keeping_badges(hard_line)
        # Then greedily pack chunks into lines of ≤ max_chars.
        cur = ""
        for chunk in chunks:
            cand = chunk if not cur else f"{cur} {chunk}"
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    out_lines.append(cur)
                # Chunk itself may still be too long (no internal whitespace);
                # in that rare case allow it to stand alone — a slightly wide
                # line is still better than corrupting an important token.
                cur = chunk
        if cur:
            out_lines.append(cur)
    return "\n".join(out_lines)


def _split_keeping_badges(line: str) -> list[str]:
    """Split *line* on whitespace, but keep ``+badge`` tokens whole.

    Example: ``Dense 256 +ReLU +Drop 0.5`` →
        ``["Dense 256", "+ReLU", "+Drop 0.5"]``
    """
    badge_starts = [i for i, ch in enumerate(line) if ch == "+" and (i == 0 or line[i - 1] == " ")]
    if not badge_starts:
        return line.split()
    parts: list[str] = []
    head = line[: badge_starts[0]].strip()
    if head:
        parts.append(head)
    for k, start in enumerate(badge_starts):
        end = badge_starts[k + 1] if k + 1 < len(badge_starts) else len(line)
        chunk = line[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def _safe_label_policy(
    spec_layers: list[NNSVGLayerSpec],
    canvas_width: int,
    font_size: int,
    allow_thinning: bool = True,
) -> list[NNSVGLayerSpec]:
    """Guarantee that labels fit within their canvas slots without overlap.

    Strategy (in order):
    1. Estimate the available pixel width per layer slot.
    2. Derive a safe character budget per line.
    3. **Wrap** any oversized label onto multiple lines on whitespace and
       ``+badge`` boundaries — never cut mid-token with ``…``.
    4. If wrapping cannot fit and the slot is extremely tight, hide every
       other label (only when ``allow_thinning`` is True).

    Labels for ch=1 block layers (Transformer / ResNet / U-Net summaries) are
    skipped because the LeNet.js renderer renders multi-line text inside the
    rectangle with auto font scaling.
    """
    n = len(spec_layers)
    if n <= 1:
        return spec_layers

    slot_px = max(1.0, (canvas_width - 120) / n)
    safe_label_px = slot_px * _LABEL_SLOT_FRAC
    char_px = font_size * _AVG_CHAR_PX
    safe_chars = max(4, int(safe_label_px / char_px))

    def _is_box_layer(layer: NNSVGLayerSpec) -> bool:
        return layer.layer_type != "dense" and layer.channels == 1

    # Step 1: wrap any label whose longest line exceeds the safe budget.
    # Preserve the original label in extra["full_label"] before wrapping so
    # debug JSON and paper JSON always carry the complete human-readable text.
    for layer in spec_layers:
        if not layer.label or _is_box_layer(layer):
            continue
        longest = max((len(line) for line in layer.label.split("\n")), default=0)
        if longest > safe_chars:
            if "full_label" not in layer.extra:
                layer.extra["full_label"] = layer.label
            layer.label = _wrap_label(layer.label, safe_chars)

    # Step 2: extreme-density fallback — thin every other label.
    if allow_thinning and safe_chars < 5:
        for i, layer in enumerate(spec_layers):
            if i % 2 != 0 and layer.layer_type not in ("input",) and not _is_box_layer(layer):
                layer.label = ""

    return spec_layers


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


# ── Detail-level summary grouping ────────────────────────────────────────────

def _summarize_sequential(spec_layers: list[NNSVGLayerSpec]) -> list[NNSVGLayerSpec]:
    """Group consecutive conv/pool layers into informative 'Block N' ch=1 boxes.

    Uses ch=1 (single-rectangle) blocks so LeNet.js can render multi-line
    labels inside the box.  Each block label includes:
    - block index
    - conv count with kernel size: ``4×Conv k3``
    - output channel count: ``, 64ch``
    - pool/downsampling marker: ``Pool ↓2``

    Example:   ``Block 2\\n4×Conv k3, 128ch\\nPool ↓2``
    """
    result: list[NNSVGLayerSpec] = []
    i = 0
    n = len(spec_layers)
    block_num = 0

    while i < n:
        cur = spec_layers[i]

        if cur.layer_type == "input":
            result.append(cur)
            i += 1

        elif cur.layer_type == "conv":
            # Absorb consecutive convs then include one pool if present.
            j = i + 1
            while j < n and spec_layers[j].layer_type == "conv":
                j += 1
            has_pool = j < n and spec_layers[j].layer_type == "pool"
            pool_spec = spec_layers[j] if has_pool else None
            if has_pool:
                j += 1

            block_num += 1
            group = spec_layers[i: j - (1 if has_pool else 0)]
            conv_layers = [s for s in group if s.layer_type == "conv"]
            last_conv = conv_layers[-1] if conv_layers else group[-1]
            n_conv = len(conv_layers)

            # Kernel info: if all convs share the same kernel, show it once.
            kernels = {s.kernel_size for s in conv_layers if s.kernel_size}
            kern_str = f" k{kernels.pop()}" if len(kernels) == 1 else ""

            ch_str = f", {last_conv.channels}ch" if last_conv.channels else ""
            line2 = f"{n_conv}×Conv{kern_str}{ch_str}"

            label = f"Block {block_num}\n{line2}"
            if has_pool and pool_spec is not None:
                pool_k = pool_spec.kernel_size
                pool_s = pool_spec.stride
                pool_str = f"Pool ↓{pool_s}" if pool_s and pool_s > 1 else (
                    f"Pool k{pool_k}" if pool_k else "Pool ↓"
                )
                label += f"\n{pool_str}"

            result.append(NNSVGLayerSpec(
                layer_type="conv",
                label=label,
                channels=1,          # ch=1 = box rendering with multi-line inside label
                feature_map_width=88,
                feature_map_height=80,
                color=last_conv.color,
            ))
            i = j

        elif cur.layer_type == "pool":
            i += 1

        elif cur.layer_type == "dense":
            # All remaining dense layers → Classifier box.  Use the original
            # semantic unit count (preserved on .extra) rather than the
            # capped visual-node count when known.
            true_u = cur.extra.get("true_units") if cur.extra else None
            units = true_u or cur.units
            units_str = f"\n{units} classes" if units and units > 1 else ""
            result.append(NNSVGLayerSpec(
                layer_type="conv",
                label=f"Classifier{units_str}",
                channels=1,
                feature_map_width=96,
                feature_map_height=72,
                color=_COLORS["output"],
            ))
            break

        else:
            result.append(cur)
            i += 1

    return result


def _summarize_residual(
    spec_layers: list[NNSVGLayerSpec],
    arch: SemanticArchitecture,
) -> list[NNSVGLayerSpec]:
    """Block summary for ResNet/U-Net architectures with merge ops.

    Produces ch=1 box blocks with informative labels that explicitly note
    when skip connections or concat links have been collapsed.

    ResNet: Input → Stem → Residual Block 1 → ... → Residual Block N → Head
    U-Net:  Input → Encoder → Bottleneck → Decoder → Segmentation Head
    """
    has_concat = any(lay.kind == LayerKind.CONCAT for lay in arch.layers)
    has_upsample = any(lay.kind == LayerKind.UPSAMPLE for lay in arch.layers)
    conv_count = sum(1 for lay in arch.layers if lay.kind in _CONV_KINDS)
    last_conv_ch = next(
        (lay.channels for lay in reversed(arch.layers)
         if lay.kind in _CONV_KINDS and lay.channels),
        None,
    )

    result: list[NNSVGLayerSpec] = []
    if spec_layers and spec_layers[0].layer_type == "input":
        result.append(spec_layers[0])

    if has_concat:
        # U-Net style encoder-decoder.  Concat/skip links are collapsed;
        # the label makes this explicit.
        upsample_str = "Upsample ×2\n" if has_upsample else ""
        out_label = (
            f"Segmentation Head\n{last_conv_ch}ch out"
            if last_conv_ch else "Segmentation Head"
        )
        result += [
            NNSVGLayerSpec(
                layer_type="conv",
                label="Encoder\nconv stages\nPool ↓2\nskip→debug JSON",
                channels=1, feature_map_width=120, feature_map_height=110,
                color=_COLORS["conv"],
            ),
            NNSVGLayerSpec(
                layer_type="conv",
                label="Bottleneck\nconv",
                channels=1, feature_map_width=92, feature_map_height=80,
                color=_COLORS["pool"],
            ),
            NNSVGLayerSpec(
                layer_type="conv",
                label=f"Decoder\n{upsample_str}conv stages\nconcat collapsed",
                channels=1, feature_map_width=120, feature_map_height=110,
                color=_COLORS["conv"],
            ),
            NNSVGLayerSpec(
                layer_type="conv", label=out_label,
                channels=1, feature_map_width=96, feature_map_height=72,
                color=_COLORS["output"],
            ),
        ]
    else:
        # ResNet style — skip links are collapsed; label notes this explicitly.
        add_count = sum(1 for lay in arch.layers if lay.kind == LayerKind.ADD)
        n_blocks = max(1, add_count)
        # Distribute conv layers across blocks for an honest convs-per-block hint.
        conv_per_block = max(1, conv_count // max(1, n_blocks + 1))

        result.append(NNSVGLayerSpec(
            layer_type="conv", label="Stem\nconv",
            channels=1, feature_map_width=88, feature_map_height=72,
            color=_COLORS["conv"],
        ))
        for b in range(n_blocks):
            ch_str = f"\n{last_conv_ch}ch" if last_conv_ch else ""
            result.append(NNSVGLayerSpec(
                layer_type="conv",
                label=(
                    f"Residual Block {b + 1}\n"
                    f"{conv_per_block}×Conv k3{ch_str}\n"
                    "+skip collapsed"
                ),
                channels=1, feature_map_width=120, feature_map_height=108,
                color=_COLORS["norm"],
            ))
        result.append(NNSVGLayerSpec(
            layer_type="conv", label="Head\nClassifier",
            channels=1, feature_map_width=88, feature_map_height=68,
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

def _transformer_metadata(arch: SemanticArchitecture) -> tuple[str, str, str]:
    """Extract heads / d_model / FFN-dim hints when present in attributes.

    Returns (heads_str, dmodel_str, ffn_str).  Empty strings when unknown.
    Non-cryptic: "12 heads", "d=768", "FFN 3072".
    """
    heads = ""
    dmodel = ""
    ffn_dim = ""

    # First attention layer's attributes
    for lay in arch.layers:
        if lay.kind == LayerKind.ATTENTION:
            attrs = lay.attributes or {}
            h = attrs.get("num_heads") or attrs.get("heads")
            d = attrs.get("d_model") or attrs.get("embed_dim") or attrs.get("hidden_size")
            if isinstance(h, int) and h > 0:
                heads = f"{h} heads"
            if isinstance(d, int) and d > 0:
                dmodel = f"d={d}"
            elif lay.output_shape:
                ints = [i for i in lay.output_shape if isinstance(i, int) and i > 0]
                if ints:
                    dmodel = f"d={ints[-1]}"
            break

    # FFN width: the largest dense-units between the first attention and a
    # smaller "down-project" dense, if discernible.
    dense_units = [
        lay.units for lay in arch.layers
        if lay.kind == LayerKind.DENSE and isinstance(lay.units, int) and lay.units > 0
    ]
    if len(dense_units) >= 2:
        # Heuristic: the FFN expansion sits between embedding and classifier.
        # Take the max non-classifier dense width.
        candidates = sorted(dense_units[:-1], reverse=True) if len(dense_units) > 1 else []
        if candidates:
            ffn_dim = f"FFN {candidates[0]}"

    return heads, dmodel, ffn_dim


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

    Heads / d_model / FFN-dim are surfaced when present in layer attributes.

    NOT exact Transformer rendering.  Q/K/V projections, individual attention
    heads, exact residual paths, and tensor flow are NOT drawn.
    Full layer list is preserved in the debug-JSON export.
    """
    heads_str, dmodel_str, ffn_str = _transformer_metadata(arch)
    attn_meta_lines = [s for s in (heads_str, dmodel_str) if s]
    ffn_meta_lines = [ffn_str] if ffn_str else []

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
                channels=1, feature_map_width=80, feature_map_height=68,
                color=_COLORS["input"],
            ))
            i += 1

        elif kind == LayerKind.EMBEDDING:
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Embedding",
                channels=1, feature_map_width=88, feature_map_height=72,
                color=_COLORS["dense"],
            ))
            i += 1

        elif kind == LayerKind.ADD and not first_attn_seen:
            # ADD before the first attention → likely positional encoding.
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Positional\nEncoding",
                channels=1, feature_map_width=80, feature_map_height=68,
                color=_COLORS["norm"],
            ))
            i += 1

        elif kind == LayerKind.ATTENTION:
            first_attn_seen = True
            j = i + 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            attn_lines = ["[MH-Attn]"]
            attn_lines.extend(attn_meta_lines)
            attn_lines.append("Add & Norm")
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="\n".join(attn_lines),
                channels=1, feature_map_width=104, feature_map_height=124,
                color=_COLORS["attention"],
            ))
            i = j

        elif kind == LayerKind.DENSE:
            # Absorb FFN block non-greedily:
            #   Dense-up → optional (Act + Dense-down) → optional Add/Norm
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
                # Use the trailing dense's true unit count when known.
                trailing_units = layers[i].units if isinstance(layers[i].units, int) else None
                head_lines = ["[Head]"]
                if trailing_units and trailing_units > 1:
                    head_lines.append(f"{trailing_units} classes")
                else:
                    head_lines.append("Classifier")
                specs.append(NNSVGLayerSpec(
                    layer_type="conv", label="\n".join(head_lines),
                    channels=1, feature_map_width=92, feature_map_height=80,
                    color=_COLORS["output"],
                ))
            else:
                ffn_lines = ["[FFN]"]
                ffn_lines.extend(ffn_meta_lines)
                ffn_lines.append("Add & Norm")
                specs.append(NNSVGLayerSpec(
                    layer_type="conv", label="\n".join(ffn_lines),
                    channels=1, feature_map_width=96, feature_map_height=108,
                    color=_COLORS["ffn"],
                ))
            i = j

        elif kind in _RECURRENT_KINDS:
            j = i + 1
            while j < n and layers[j].kind in _TRANSFORMER_ABSORB:
                j += 1
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label=f"[{kind.name}]\nAdd & Norm",
                channels=1, feature_map_width=88, feature_map_height=96,
                color=_COLORS["recurrent"],
            ))
            i = j

        elif kind in _NORM_KINDS:
            specs.append(NNSVGLayerSpec(
                layer_type="conv", label="Norm",
                channels=1, feature_map_width=44, feature_map_height=60,
                color=_COLORS["norm"],
            ))
            i += 1

        else:
            i += 1

    return specs


# ── FCNN mapper ──────────────────────────────────────────────────────────────

def _peek_fused_decoration(
    layers: list[SemanticLayer],
    start: int,
    show_activations: bool,
) -> tuple[str, list[str], int]:
    """Look ahead from *start* and return (act_name, badges, n_consumed).

    Activations are fused only when ``show_activations`` is True.
    BatchNorm/LayerNorm/Dropout are always fused as ``+BN``/``+LN``/``+Drop``
    badges so that supporting ops are visible without taking their own slot.
    """
    n = len(layers)
    i = start
    fused_act = ""
    badges: list[str] = []
    while i < n:
        k = layers[i].kind
        if k in _ACTIVATION_KINDS:
            if not fused_act and show_activations:
                fused_act = _act_name(k)
                i += 1
                continue
            break
        if k in _BADGE_KINDS:
            text = _badge_text(layers[i])
            if text and text not in badges:
                badges.append(text)
            i += 1
            continue
        break
    return fused_act, badges, i - start


def _decorate_label(base: str, fused_act: str, badges: list[str]) -> str:
    """Append +act and badge tokens to *base*.

    No truncation here: the per-slot safe-label policy further down the
    pipeline knows the canvas width and chooses a safe character budget.
    Truncating here would corrupt important tokens like ``+Drop 0.5``.
    """
    parts = [base]
    if fused_act:
        parts.append(f"+{fused_act}")
    parts.extend(f"+{b}" for b in badges)
    return " ".join(parts)


def _map_fcnn(
    arch: SemanticArchitecture,
    cfg: RenderConfig,
    lm: str,
) -> list[NNSVGLayerSpec]:
    """Each significant layer → a column of neurons.

    Activations are fused into the preceding label when show_activations=True.
    Norms/Dropout are fused as +BN, +LN, +Drop badges.
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

        fused_act, badges, n_consumed = _peek_fused_decoration(
            layers, i + 1, cfg.show_activations
        )

        units = layer.units or layer.channels or 0
        if units == 0 and layer.output_shape:
            ints = [d for d in layer.output_shape if isinstance(d, int) and d > 0]
            if ints:
                units = ints[-1]
        if units == 0:
            units = 10

        base = _make_label(layer, lm)
        label = _decorate_label(base, fused_act, badges)

        specs.append(NNSVGLayerSpec(
            layer_type="dense",
            label=label,
            units=min(units, 256),
            color=_color_for(kind),
            extra={"true_units": units} if units else {},
        ))

        i += 1 + n_consumed

    # Relabel the last dense layer as "Output N" using true semantic units
    # rather than the visual cap.  Falls back to display units when unknown.
    dense_specs = [s for s in specs if s.layer_type == "dense"]
    if dense_specs:
        last = dense_specs[-1]
        u = last.extra.get("true_units") if last.extra else None
        if not u:
            u = last.units or 0
        last.label = _truncate(f"Output {u}" if u else "Output")

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

        fused_act, badges, n_consumed = _peek_fused_decoration(
            layers, i + 1, cfg.show_activations
        )
        # INPUT layers do not absorb decorations
        if kind == LayerKind.INPUT:
            fused_act, badges, n_consumed = "", [], 0

        base = _make_label(layer, lm)
        label = _decorate_label(base, fused_act, badges)

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
            true_units = layer.units or 10
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(true_units, _MAX_DENSE_UNITS_LENET),
                color=_color_for(kind),
                extra={"true_units": true_units},
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

        else:
            true_units = layer.units or layer.channels or 8
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(true_units, _MAX_DENSE_UNITS_LENET),
                color=_color_for(kind),
                extra={"true_units": true_units},
            ))

        i += 1 + n_consumed

    # Relabel the last dense column as "Classifier N" using semantic units
    # (visual node count is capped; the *meaning* should match the architecture).
    dense_specs = [s for s in specs if s.layer_type == "dense"]
    if dense_specs:
        last = dense_specs[-1]
        true_u = last.extra.get("true_units") if last.extra else None
        if true_u and true_u > 1:
            last.label = _truncate(f"Classifier {true_u}")
        else:
            last.label = "Classifier"

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

        fused_act, badges, n_consumed = _peek_fused_decoration(
            layers, i + 1, cfg.show_activations
        )
        if kind == LayerKind.INPUT:
            fused_act, badges, n_consumed = "", [], 0

        base = _make_label(layer, lm)
        label = _decorate_label(base, fused_act, badges)

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
            true_units = layer.units or 8
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(true_units, _MAX_DENSE_UNITS_ALEXNET),
                color=_color_for(kind),
                extra={"true_units": true_units},
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

        else:
            true_units = layer.units or layer.channels or 6
            specs.append(NNSVGLayerSpec(
                layer_type="dense", label=label,
                units=min(true_units, _MAX_DENSE_UNITS_ALEXNET),
                color=_color_for(kind),
                extra={"true_units": true_units},
            ))

        i += 1 + n_consumed

    # Relabel the last dense column as "Classifier N" with the semantic units.
    dense_specs = [s for s in specs if s.layer_type == "dense"]
    if dense_specs:
        last = dense_specs[-1]
        true_u = last.extra.get("true_units") if last.extra else None
        if true_u and true_u > 1:
            last.label = _truncate(f"Classifier {true_u}")
        else:
            last.label = "Classifier"

    return specs


# ── Auto-sizing ──────────────────────────────────────────────────────────────

_MIN_PX_PER_LAYER = 80
# Default screenshot-friendly width for very small models so the diagram does
# not look lost in a large empty canvas.  Compact mode shrinks further.
_SMALL_MODEL_WIDTH = 760
_COMPACT_PX_PER_LAYER = 64


def _auto_width(
    n_layers: int,
    configured_width: int,
    compact: bool = False,
    spec_layers: list[NNSVGLayerSpec] | None = None,
    font_size: int = 12,
) -> int:
    """Choose a canvas width that fits both the layer count and the label widths.

    - Compact mode uses a tighter per-layer budget so moderately deep
      summaries do not require horizontal scrolling.
    - We never shrink the user/theme-configured width; we only grow.
    - If any below-block label (multi-line, longest line) is wider than
      its slot, expand the canvas so it fits without overlap.
    """
    px_per = _COMPACT_PX_PER_LAYER if compact else _MIN_PX_PER_LAYER
    width = max(configured_width, n_layers * px_per + 120)

    if not spec_layers:
        return width

    # Estimate the longest label line (multi-line wrapping reduces this).
    char_px = font_size * _AVG_CHAR_PX
    longest_line_px = 0.0
    for layer in spec_layers:
        # Inside-box labels (ch=1) are auto-scaled by the JS renderer; skip.
        if layer.layer_type != "dense" and layer.channels == 1:
            continue
        if not layer.label:
            continue
        for line in layer.label.split("\n"):
            longest_line_px = max(longest_line_px, len(line) * char_px)
    if longest_line_px <= 0:
        return width

    # Each slot needs to fit the longest line + a small gutter.
    needed_per_slot = longest_line_px / _LABEL_SLOT_FRAC + 8
    needed_total = int(n_layers * needed_per_slot + 120)
    # Cap label-driven growth at ~1.5× the configured width.  Beyond that
    # the safe-label policy wraps the label onto multiple lines instead of
    # producing an absurdly wide canvas for a single oversized label.
    growth_cap = int(max(configured_width, 600) * 1.5)
    return min(growth_cap, max(width, needed_total))


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
    _unsupported_subtitle: str = ""
    diagnostic_payload: dict | None = None
    if has_seq_op and config.style is None:
        if config.transformer_mode == "unsupported":
            # Build a structured diagnostic payload.  The HTML generator
            # renders this as a styled card so layout never overflows and
            # text stays readable at any canvas width.
            detected: list[str] = []
            if any(lay.kind == LayerKind.EMBEDDING for lay in arch.layers):
                detected.append("Embedding")
            if any(lay.kind == LayerKind.ATTENTION for lay in arch.layers):
                detected.append("Attention")
            if any(lay.kind in _RECURRENT_KINDS for lay in arch.layers):
                detected.append("Recurrent")
            if any(lay.kind == LayerKind.DENSE for lay in arch.layers):
                detected.append("Dense / FFN")
            if any(lay.kind in _NORM_KINDS for lay in arch.layers):
                detected.append("Norm")
            detected_str = ", ".join(detected) if detected else "—"
            diagnostic_payload = {
                "kind": "transformer_unsupported",
                "headline": "Transformer exact rendering is not supported",
                "body": (
                    "NN-SVG has no native Transformer renderer. "
                    "Q/K/V projections, individual heads, exact residual "
                    "paths, and tensor flow cannot be drawn as a graph."
                ),
                "actions": [
                    'Use transformer_mode="block_summary" for an '
                    "approximate block-level overview.",
                    "Inspect export-debug-json for full layer metadata "
                    "(heads, d_model, dropout rate, norms).",
                ],
                "detected": detected_str,
            }
            # Provide a small fallback ch=1 box so any non-HTML consumer
            # (e.g. NN-SVG JSON spec) still sees a labelled block.
            diag_label = (
                "Transformer exact rendering\nis not supported\n\n"
                'Use transformer_mode="block_summary"\n'
                "for an approximate overview.\n\n"
                f"Detected: {detected_str}\n"
                "Full metadata in debug JSON."
            )
            layers = [NNSVGLayerSpec(
                layer_type="conv",
                label=diag_label,
                channels=1,
                feature_map_width=420,
                feature_map_height=260,
                color="#E8EAF6",
            )]
            family = RenderFamily.LENET
            _unsupported_subtitle = (
                "Transformer exact rendering not supported  ·  "
                'set transformer_mode="block_summary" for an approximate view'
            )
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
    from ..core.enums import LayoutMode
    is_compact = config.layout_mode == LayoutMode.COMPACT
    auto_width = _auto_width(
        len(layers), config.width,
        compact=is_compact,
        spec_layers=layers,
        font_size=config.font_size,
    )

    # ── Label safety: wrap or thin labels to prevent overlap ──────────────
    # Skip for transformer blocks — JS auto-scales font inside rectangles.
    # Skip thinning when the user explicitly requested full detail.
    if not is_transformer_block:
        allow_thin = config.detail_level != "full"
        layers = _safe_label_policy(layers, auto_width, config.font_size, allow_thin)
        # Re-evaluate width after wrapping: each line is shorter now, so the
        # canvas can stay smaller (relevant for the explicit-width path).
        auto_width = _auto_width(
            len(layers), config.width,
            compact=is_compact,
            spec_layers=layers,
            font_size=config.font_size,
        )

    # ── Subtitle ──────────────────────────────────────────────────────────
    has_approx = bool(arch.warnings) and config.approximate_mode != "allow"
    if _unsupported_subtitle:
        subtitle = _unsupported_subtitle
    else:
        subtitle = _make_subtitle(
            family, arch, has_approx, is_transformer_block, n_spec_layers=len(layers)
        )

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
        diagnostic=diagnostic_payload,
    )
