"""Public rendering API: build NN-SVG specs, HTML, and SVG output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.aliases import resolve_alias
from ..core.config import RenderConfig
from ..core.enums import RenderFamily, Theme
from ..core.validation import validate_render_config
from ..ir.semantic_ir import SemanticArchitecture
from ..presets import get_preset
from ..visualization.nnsvg_html import generate_html
from ..visualization.nnsvg_mapper import map_to_nnsvg
from ..visualization.nnsvg_runtime import save_svg_from_html
from ..visualization.nnsvg_schema import NNSVGSpec


def _resolve_figsize(
    width: int | None,
    height: int | None,
    extras: dict[str, Any],
) -> tuple[int | None, int | None]:
    """Pop ``figsize``/``dpi`` from *extras* and convert to pixel dimensions.

    Explicit *width*/*height* take priority over *figsize* on their respective
    axes.  *dpi* alone (without *figsize*) is validated and ignored.
    """
    figsize = extras.pop("figsize", None)
    dpi = extras.pop("dpi", None)

    # Validate dpi regardless of whether figsize is present.
    resolved_dpi: float = 100.0
    if dpi is not None:
        try:
            resolved_dpi = float(dpi)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dpi must be a positive number, got {dpi!r}") from exc
        if resolved_dpi <= 0:
            raise ValueError(f"dpi must be positive, got {dpi!r}")

    if figsize is None:
        return width, height

    # Validate figsize.
    try:
        fw, fh = figsize
        fw, fh = float(fw), float(fh)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"figsize must be a 2-element sequence of positive numbers, got {figsize!r}"
        ) from exc
    if fw <= 0 or fh <= 0:
        raise ValueError(
            f"figsize values must be positive, got {figsize!r}"
        )

    # Apply figsize only where explicit pixel dimensions were not given.
    if width is None:
        width = round(fw * resolved_dpi)
    if height is None:
        height = round(fh * resolved_dpi)

    return width, height


def _resolve_architecture(arch_or_source: SemanticArchitecture | str | Path | Any) -> SemanticArchitecture:
    if isinstance(arch_or_source, SemanticArchitecture):
        return arch_or_source
    from .parse import parse_model
    return parse_model(arch_or_source)


def _build_config(
    theme: str | Theme | None,
    style: str | int | RenderFamily | None,
    width: int | None,
    height: int | None,
    show_shapes: bool | None,
    show_labels: bool | None,
    title: str | None,
    compact: bool | None,
    options: dict[str, Any] | None,
    extras: dict[str, Any],
) -> RenderConfig:
    """Build a :class:`RenderConfig` from keyword arguments."""
    width, height = _resolve_figsize(width, height, extras)
    if theme is not None:
        t = resolve_alias(Theme, theme)
        cfg = get_preset(t)
    else:
        cfg = RenderConfig()

    overrides: dict[str, Any] = {}
    if style is not None:
        overrides["style"] = resolve_alias(RenderFamily, style)
    if width is not None:
        overrides["width"] = width
    if height is not None:
        overrides["height"] = height
    if show_shapes is not None:
        overrides["show_shapes"] = show_shapes
    if show_labels is not None:
        overrides["show_labels"] = show_labels
    if title is not None:
        overrides["title"] = title
    if compact is True:
        from ..core.enums import LayoutMode
        overrides["layout_mode"] = LayoutMode.COMPACT
    elif compact is False:
        from ..core.enums import LayoutMode
        overrides["layout_mode"] = LayoutMode.PRESENTATION
    if options:
        overrides["options"] = options
    overrides.update(extras)

    cfg = cfg.merge(overrides)
    validate_render_config(cfg)
    return cfg


def build_nnsvg_spec(
    architecture: SemanticArchitecture | str | Path | Any,
    theme: str | Theme | None = None,
    style: str | int | RenderFamily | None = None,
    width: int | None = None,
    height: int | None = None,
    show_shapes: bool | None = None,
    show_labels: bool | None = None,
    title: str | None = None,
    compact: bool | None = None,
    options: dict[str, Any] | None = None,
    **extras: Any,
) -> NNSVGSpec:
    """Build an :class:`NNSVGSpec` from a model / architecture."""
    arch = _resolve_architecture(architecture)
    cfg = _build_config(theme, style, width, height, show_shapes, show_labels,
                        title, compact, options, extras)
    return map_to_nnsvg(arch, cfg)


def render_network_html(
    architecture: SemanticArchitecture | str | Path | Any,
    **kwargs: Any,
) -> str:
    """Render *architecture* to a standalone HTML document string."""
    spec = build_nnsvg_spec(architecture, **kwargs)
    return generate_html(spec)


def render_network_svg(
    architecture: SemanticArchitecture | str | Path | Any,
    **kwargs: Any,
) -> str:
    """Render *architecture* to an SVG string via headless browser.

    Raises:
        BrowserNotAvailableError: If Playwright is not installed.
    """
    from ..visualization.nnsvg_runtime import extract_svg_from_html
    html = render_network_html(architecture, **kwargs)
    return extract_svg_from_html(html)


def save_network_html(
    path: str | Path,
    architecture: SemanticArchitecture | str | Path | Any,
    **kwargs: Any,
) -> Path:
    """Render *architecture* to HTML and write it to *path*."""
    html = render_network_html(architecture, **kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def save_network_svg(
    path: str | Path,
    architecture: SemanticArchitecture | str | Path | Any,
    **kwargs: Any,
) -> Path:
    """Render *architecture* to SVG (via headless browser) and save to *path*."""
    html = render_network_html(architecture, **kwargs)
    return save_svg_from_html(html, path)


def _build_reason(arch: SemanticArchitecture) -> str:
    """Build a human-readable reason string for the family recommendation."""
    from ..core.enums import LayerKind

    conv_count = sum(
        1 for lay in arch.layers
        if lay.kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV)
    )
    dense_count = sum(1 for lay in arch.layers if lay.kind == LayerKind.DENSE)
    # Include both skip_detector output AND merge ops in layer list (manual spec proxy)
    has_merge_layers = any(
        lay.kind in (LayerKind.ADD, LayerKind.CONCAT, LayerKind.MULTIPLY)
        for lay in arch.layers
    )
    has_skip = arch.has_skip_connections or has_merge_layers
    has_attention = any(lay.kind == LayerKind.ATTENTION for lay in arch.layers)
    has_recurrent = any(
        lay.kind in (LayerKind.RECURRENT, LayerKind.LSTM, LayerKind.GRU)
        for lay in arch.layers
    )

    if has_attention:
        return (
            "Attention layers detected (Transformer-like); "
            "no direct NN-SVG mapping — approximated as FCNN/AlexNet backbone"
        )
    if has_recurrent:
        return (
            "Recurrent layers (LSTM/GRU/RNN) detected; "
            "no direct NN-SVG mapping — approximated as sequential FCNN"
        )
    if conv_count == 0 and dense_count > 0:
        if has_skip:
            return "Dense-only (MLP) with merge operations; rendered as FCNN (branches collapsed)"
        return "Only dense layers detected (MLP/FCNN)"
    if 1 <= conv_count <= 3:
        if has_skip:
            return (
                f"Small CNN ({conv_count} conv layer(s)) with skip/merge operations; "
                "mapped to LeNet — skip links collapsed"
            )
        return f"Small CNN with {conv_count} conv layer(s); mapped to LeNet"
    if conv_count > 3:
        if has_skip:
            merge_layers = [
                l for l in arch.layers
                if l.kind in (LayerKind.ADD, LayerKind.CONCAT, LayerKind.MULTIPLY)
            ]
            if any(l.kind == LayerKind.CONCAT for l in merge_layers):
                return (
                    f"Deep CNN ({conv_count} convs) with Concat operations "
                    "(U-Net/encoder-decoder); mapped to AlexNet — branches collapsed"
                )
            return (
                f"Deep CNN ({conv_count} convs) with residual connections (Add); "
                "mapped to AlexNet — skip links collapsed"
            )
        return f"Deep CNN with {conv_count} conv layers; mapped to AlexNet"
    if conv_count == 0 and dense_count == 0:
        return "No standard conv/dense layers detected; best-effort FCNN fallback"
    return "Mixed architecture; best-effort FCNN approximation"


def recommend_view(
    architecture: SemanticArchitecture | str | Path | Any,
) -> dict[str, Any]:
    """Return the recommended rendering family and reasoning for *architecture*.

    Returns a dict with keys:

    - ``family``: ``"fcnn"``, ``"lenet"``, or ``"alexnet"``
    - ``confidence``: ``"high"``, ``"medium"``, ``"low"``, or ``"unknown"``
    - ``is_approximate``: ``True`` when the diagram is a lossy approximation
    - ``reason``: short human-readable explanation of the choice
    - ``warnings``: list of warning strings about unsupported features
    - ``complexity_hint``: ``"sequential"``, ``"skip"``, ``"multi_branch"``, ``"dag"``
    """
    from ..core.enums import ConfidenceLevel

    arch = _resolve_architecture(architecture)
    family = arch.recommended_family.value if arch.recommended_family else None
    is_approximate = arch.family_confidence in (
        ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.UNKNOWN
    ) or bool(arch.warnings)
    return {
        "family": family,
        "confidence": arch.family_confidence.value,
        "is_approximate": is_approximate,
        "reason": _build_reason(arch),
        "warnings": list(arch.warnings),
        "complexity_hint": arch.metadata.get("complexity_hint", ""),
    }
