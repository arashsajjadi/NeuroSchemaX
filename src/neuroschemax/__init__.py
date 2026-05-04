"""NeuroSchemaX — neural network visualisation & export, powered by NN-SVG.

Public API::

    from neuroschemax import (
        parse_model, summarize_model, recommend_view,
        build_nnsvg_spec, render_network_html, render_network_svg,
        save_network_html, save_network_svg,
        export_paper_json, export_debug_json,
        save_paper_json, save_debug_json, save_nnsvg_spec,
        RenderConfig, RenderFamily, Theme,
        # Simplified stateful API
        draw, savefig, show, save_html, save_svg,
        figure, Figure, doctor,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .api.export import (
    export_debug_json_from as export_debug_json,
)
from .api.export import (
    export_nnsvg_spec,
    save_nnsvg_spec,
)
from .api.export import (
    export_paper_json_from as export_paper_json,
)
from .api.export import (
    save_debug_json_from as save_debug_json,
)
from .api.export import (
    save_paper_json_from as save_paper_json,
)
from .api.figure import Figure
from .api.parse import parse_graph, parse_model
from .api.render import (
    build_nnsvg_spec,
    recommend_view,
    render_network_html,
    render_network_svg,
    save_network_html,
    save_network_svg,
)
from .api.summarize import summarize_model
from .core.config import RenderConfig
from .core.enums import (
    ConfidenceLevel,
    InputFormat,
    LabelDensity,
    LayerKind,
    LayoutMode,
    LineStyle,
    Orientation,
    OutputFormat,
    RenderFamily,
    Theme,
)
from .exceptions import (
    BrowserNotAvailableError,
    NeuroSchemaXError,
    ParseError,
    RenderError,
    UnsupportedFormatError,
)
from .ir.graph_ir import GraphIR
from .ir.semantic_ir import SemanticArchitecture
from .version import __version__
from .visualization.nnsvg_schema import NNSVGSpec

# ── Module-level stateful API ────────────────────────────────────────────────
#
# State is stored in a plain dict so we avoid the ``global`` keyword and so
# that the dict object itself never needs to be rebound (only its contents).

_STATE: dict[str, Any] = {"arch": None, "kwargs": {}}


def draw(source: Any, **kwargs: Any) -> SemanticArchitecture:
    """Parse *source*, stash the result, and return the architecture.

    This is the entry point for the simplified stateful API::

        arch = nsx.draw("model.onnx")
        nsx.savefig("diagram.html")

    The arch and kwargs are stored in the module-level ``_STATE`` dict so that
    :func:`savefig`, :func:`save_html`, :func:`save_svg`, and :func:`show` can
    use them without repeating the source.
    """
    arch = parse_model(source)
    _STATE["arch"] = arch
    _STATE["kwargs"] = kwargs
    return arch


def savefig(path: str | Path, **kwargs: Any) -> Path:
    """Save the current architecture (from the last :func:`draw` call) to *path*.

    Format is inferred from the file extension (``.html`` or ``.svg``).

    Raises:
        RuntimeError: If :func:`draw` has not been called yet.
    """
    arch = _STATE["arch"]
    if arch is None:
        raise RuntimeError(
            "No architecture loaded. Call nsx.draw() before nsx.savefig()."
        )
    merged: dict[str, Any] = dict(_STATE["kwargs"])
    merged.update(kwargs)
    out = Path(path)
    suffix = out.suffix.lower()
    if suffix == ".html":
        return save_network_html(out, arch, **merged)
    elif suffix == ".svg":
        return save_network_svg(out, arch, **merged)
    else:
        raise ValueError(
            f"Cannot infer format from extension {suffix!r}. "
            "Use .html or .svg."
        )


def show(source: Any = None, **kwargs: Any) -> None:
    """Render *source* (or the stashed arch) and display it.

    - Inside a Jupyter / IPython kernel the diagram is shown **inline**.
    - Outside a notebook it opens in the **default web browser**.

    If *source* is ``None`` the last architecture loaded by :func:`draw` is used.
    """
    if source is not None:
        arch = parse_model(source)
    else:
        arch = _STATE["arch"]
        if arch is None:
            raise RuntimeError(
                "No architecture loaded. Call nsx.draw() or pass a source to nsx.show()."
            )
    merged: dict[str, Any] = dict(_STATE["kwargs"])
    merged.update(kwargs)
    html = render_network_html(arch, **merged)
    from ._display import show_html
    show_html(html)


def save_html(path: str | Path, source: Any = None, **kwargs: Any) -> Path:
    """Save architecture to HTML.

    If *source* is ``None`` the stashed architecture from :func:`draw` is used.
    """
    if source is not None:
        return save_network_html(path, source, **kwargs)
    arch = _STATE["arch"]
    if arch is None:
        raise RuntimeError(
            "No architecture loaded. Call nsx.draw() or pass a source to nsx.save_html()."
        )
    merged: dict[str, Any] = dict(_STATE["kwargs"])
    merged.update(kwargs)
    return save_network_html(path, arch, **merged)


def save_svg(path: str | Path, source: Any = None, **kwargs: Any) -> Path:
    """Save architecture to SVG.

    If *source* is ``None`` the stashed architecture from :func:`draw` is used.
    """
    if source is not None:
        return save_network_svg(path, source, **kwargs)
    arch = _STATE["arch"]
    if arch is None:
        raise RuntimeError(
            "No architecture loaded. Call nsx.draw() or pass a source to nsx.save_svg()."
        )
    merged: dict[str, Any] = dict(_STATE["kwargs"])
    merged.update(kwargs)
    return save_network_svg(path, arch, **merged)


def doctor() -> dict[str, Any]:
    """Return a structured environment/dependency status summary.

    See :func:`neuroschemax.visualization.compat.environment_summary` for the
    full dict shape.
    """
    from .visualization.compat import environment_summary
    return environment_summary()


def figure(
    width: int | None = None,
    height: int | None = None,
    theme: str | None = "paper",
    **kwargs: Any,
) -> Figure:
    """Create and return a :class:`Figure` instance for object-style usage::

        fig = nsx.figure(width=1400, height=700, theme="paper")
        fig.draw("model.onnx")
        fig.savefig("diagram.svg")

        # Matplotlib-style sizing: figsize=(inches_wide, inches_tall), dpi=pixels_per_inch
        fig = nsx.figure(figsize=(12, 6), dpi=120, theme="paper")
        fig.draw("model.onnx")
        fig.savefig("diagram.html")
    """
    return Figure(width=width, height=height, theme=theme, **kwargs)


# ── __all__ ──────────────────────────────────────────────────────────────────

__all__ = [
    # Meta
    "__version__",
    # Parsing
    "parse_graph",
    "parse_model",
    # Summarising
    "summarize_model",
    "recommend_view",
    # Rendering
    "build_nnsvg_spec",
    "render_network_html",
    "render_network_svg",
    "save_network_html",
    "save_network_svg",
    # Exporting
    "export_paper_json",
    "export_debug_json",
    "export_nnsvg_spec",
    "save_paper_json",
    "save_debug_json",
    "save_nnsvg_spec",
    # Config & enums
    "RenderConfig",
    "RenderFamily",
    "Theme",
    "LineStyle",
    "Orientation",
    "LayoutMode",
    "LabelDensity",
    "LayerKind",
    "OutputFormat",
    "InputFormat",
    "ConfidenceLevel",
    # IR
    "GraphIR",
    "SemanticArchitecture",
    "NNSVGSpec",
    # Exceptions
    "NeuroSchemaXError",
    "ParseError",
    "UnsupportedFormatError",
    "RenderError",
    "BrowserNotAvailableError",
    # Simplified stateful API
    "draw",
    "savefig",
    "show",
    "save_html",
    "save_svg",
    "doctor",
    "figure",
    "Figure",
]
