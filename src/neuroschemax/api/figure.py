"""Object-oriented Figure API for NeuroSchemaX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ir.semantic_ir import SemanticArchitecture


class Figure:
    """A figure object that holds a parsed architecture and rendering options.

    Typical usage::

        fig = nsx.figure(width=1400, height=700, theme="paper")
        fig.draw("model.onnx")
        fig.savefig("diagram.svg")
        fig.save_html("diagram.html")
        fig.export_debug_json("debug.json")

    All rendering keyword arguments (``width``, ``height``, ``theme``, ``style``,
    ``show_labels``, ``show_shapes``, ``title``, ``compact``) can be passed at
    construction time and overridden per-method call.
    """

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        theme: str | None = "paper",
        **kwargs: Any,
    ) -> None:
        self._arch: SemanticArchitecture | None = None
        self._render_kwargs: dict[str, Any] = {}
        if width is not None:
            self._render_kwargs["width"] = width
        if height is not None:
            self._render_kwargs["height"] = height
        if theme is not None:
            self._render_kwargs["theme"] = theme
        self._render_kwargs.update(kwargs)

    # ------------------------------------------------------------------
    # Drawing / parsing
    # ------------------------------------------------------------------

    def draw(self, source: Any, **kwargs: Any) -> Figure:
        """Parse *source* and store the resulting architecture.

        Returns ``self`` so calls can be chained::

            fig.draw("model.onnx").savefig("out.svg")
        """
        from .parse import parse_model
        self._arch = parse_model(source)
        # Per-draw kwargs override construction-time kwargs
        if kwargs:
            merged = dict(self._render_kwargs)
            merged.update(kwargs)
            self._render_kwargs = merged
        return self

    def _require_arch(self) -> SemanticArchitecture:
        if self._arch is None:
            raise RuntimeError(
                "No architecture loaded. Call draw() before calling this method."
            )
        return self._arch

    def _merged_kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self._render_kwargs)
        merged.update(overrides)
        return merged

    # ------------------------------------------------------------------
    # Saving / exporting
    # ------------------------------------------------------------------

    def savefig(self, path: str | Path, **kwargs: Any) -> Path:
        """Save to *path*.

        The format is inferred from the file extension:
        - ``.html`` -> HTML
        - ``.svg`` -> SVG (requires Playwright)
        """
        self._require_arch()
        out = Path(path)
        suffix = out.suffix.lower()
        self._merged_kwargs(kwargs)
        if suffix == ".html":
            return self.save_html(path, **kwargs)
        elif suffix == ".svg":
            return self.save_svg(path, **kwargs)
        else:
            raise ValueError(
                f"Cannot infer format from extension {suffix!r}. "
                "Use .html or .svg, or call save_html() / save_svg() directly."
            )

    def save_html(self, path: str | Path, **kwargs: Any) -> Path:
        """Render the architecture to HTML and save to *path*."""
        arch = self._require_arch()
        from .render import save_network_html
        merged = self._merged_kwargs(kwargs)
        return save_network_html(path, arch, **merged)

    def save_svg(self, path: str | Path, **kwargs: Any) -> Path:
        """Render the architecture to SVG and save to *path*."""
        arch = self._require_arch()
        from .render import save_network_svg
        merged = self._merged_kwargs(kwargs)
        return save_network_svg(path, arch, **merged)

    def show(self, **kwargs: Any) -> None:
        """Display the rendered diagram.

        - Inside a Jupyter / IPython kernel the diagram is shown **inline**.
        - Outside a notebook it opens in the **default web browser**.
        """
        arch = self._require_arch()
        from .._display import show_html
        from .render import render_network_html
        merged = self._merged_kwargs(kwargs)
        html = render_network_html(arch, **merged)
        show_html(html)

    def to_html(self, **kwargs: Any) -> str:
        """Return the full self-contained HTML string for this diagram.

        This produces the same output as :meth:`save_html` — a complete
        standalone document with embedded CSS and JavaScript suitable for
        writing to a file or opening in a browser.

        .. note::
            Passing this directly to ``IPython.display.HTML()`` in Colab will
            print raw CSS/JS as visible text.  Use :meth:`show` or
            :meth:`to_notebook_html` for notebook display.
        """
        arch = self._require_arch()
        from .render import render_network_html
        merged = self._merged_kwargs(kwargs)
        return render_network_html(arch, **merged)

    def to_notebook_html(self, width: int = 1200, height: int = 520, **kwargs: Any) -> str:
        """Return notebook-safe HTML for inline display.

        Wraps the standalone HTML in an ``<iframe srcdoc="...">`` so that CSS
        and JavaScript execute inside the browser's iframe sandbox without
        leaking raw source as visible notebook output.

        This is what :meth:`show` uses internally.  Pass the result to
        ``IPython.display.HTML()`` for manual control::

            from IPython.display import HTML, display
            display(HTML(fig.to_notebook_html()))
        """
        standalone = self.to_html(**kwargs)
        from .._display import to_notebook_html
        return to_notebook_html(standalone, width=width, height=height)

    def _repr_html_(self) -> str:
        """Jupyter/IPython rich display hook.

        When a :class:`Figure` is the last expression in a cell, Jupyter
        automatically calls this and renders the diagram inline via a sandboxed
        iframe — no raw CSS/JS leaks into the notebook output.
        """
        if self._arch is None:
            return "<pre>Figure: no architecture loaded. Call draw() first.</pre>"
        try:
            return self.to_notebook_html()
        except Exception as exc:  # noqa: BLE001
            return f"<pre>Figure: render error — {exc}</pre>"

    # ------------------------------------------------------------------
    # JSON export helpers
    # ------------------------------------------------------------------

    def export_debug_json(self, path: str | Path) -> Path:
        """Save verbose debug JSON for the current architecture to *path*."""
        arch = self._require_arch()
        from ..exporters.debug_json import save_debug_json
        return save_debug_json(arch, path)

    def export_paper_json(self, path: str | Path) -> Path:
        """Save paper-oriented JSON for the current architecture to *path*."""
        arch = self._require_arch()
        from ..exporters.paper_json import save_paper_json
        return save_paper_json(arch, path)

    def export_nnsvg_json(self, path: str | Path, **kwargs: Any) -> Path:
        """Save the NN-SVG spec JSON for the current architecture to *path*.

        This is an alias for :meth:`export_nnsvg_spec` using file-path output.
        """
        arch = self._require_arch()
        from ..exporters.nnsvg import save_nnsvg_spec
        from .render import build_nnsvg_spec
        merged = self._merged_kwargs(kwargs)
        spec = build_nnsvg_spec(arch, **merged)
        return save_nnsvg_spec(spec, path)

    # Convenience alias
    export_nnsvg_spec = export_nnsvg_json

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        arch_info = (
            f"arch={self._arch.model_name!r}"
            if self._arch is not None
            else "no arch loaded"
        )
        return f"Figure({arch_info}, kwargs={self._render_kwargs!r})"
