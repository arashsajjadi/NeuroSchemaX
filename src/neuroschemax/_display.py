"""HTML display helper: Jupyter/Colab inline or browser fallback."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path


def _can_display_html() -> bool:
    """Return True when IPython.display.HTML / display are importable.

    Tests the actual display API, not whether the *ipython* package (lowercase)
    can be imported as a module.  Colab and Jupyter both satisfy this even when
    ``import ipython`` (lowercase) would fail.
    """
    try:
        from IPython.display import HTML, display  # type: ignore[import]  # noqa: F401
        return True
    except ImportError:
        return False


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter / IPython kernel.

    Works in local Jupyter, JupyterLab, VS Code notebooks, and Google Colab.
    """
    try:
        from IPython import get_ipython  # type: ignore[import]
        shell = get_ipython()
        return shell is not None and hasattr(shell, "kernel")
    except ImportError:
        return False


def _in_colab() -> bool:
    """Return True when running inside Google Colab specifically."""
    try:
        import google.colab  # type: ignore[import]  # noqa: F401
        return True
    except ImportError:
        return False


def _make_notebook_iframe(standalone_html: str, width: int = 1200, height: int = 520) -> str:
    """Wrap *standalone_html* in an ``<iframe srcdoc="...">`` element.

    This is the notebook-safe display method.  Passing a full ``<html>``
    document directly to ``IPython.display.HTML()`` causes Colab to render raw
    ``<style>`` and ``<script>`` blocks as visible text rather than executing
    them.  The ``srcdoc`` iframe sandboxes the document so the browser (not the
    notebook's DOM) handles rendering — CSS and JS execute correctly inside the
    frame, and no raw source leaks into the visible notebook output.

    The HTML content is escaped for use inside an HTML attribute value:
      - ``&`` → ``&amp;``  (must come first)
      - ``<`` → ``&lt;``
      - ``>`` → ``&gt;``
      - ``"`` → ``&quot;``

    Escaping ``<`` and ``>`` ensures that no raw ``<style>``, ``<script>``,
    or ``<!DOCTYPE>`` strings appear in the outer notebook HTML, which prevents
    Colab from mis-parsing the attribute and leaking CSS/JS as visible text.
    The browser decodes these entities back to the original characters when it
    loads the srcdoc iframe content.
    """
    escaped = (
        standalone_html
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        f'<iframe srcdoc="{escaped}" '
        f'width="{width}" height="{height}" '
        f'style="border:none;width:100%;overflow:hidden;" '
        f'frameborder="0"></iframe>'
    )


def to_notebook_html(standalone_html: str, width: int = 1200, height: int = 520) -> str:
    """Return notebook-safe display HTML for *standalone_html*.

    The result is a minimal ``<iframe srcdoc="...">`` wrapper that embeds
    the full standalone HTML inside a sandboxed frame.  Callers can pass this
    to ``IPython.display.HTML(...)`` without leaking raw CSS/JS as visible text.
    """
    return _make_notebook_iframe(standalone_html, width=width, height=height)


def show_html(html: str) -> None:
    """Display *html*:

    - **Jupyter / JupyterLab / VS Code notebooks**: renders inline via an
      ``iframe srcdoc`` wrapper so CSS and JS execute correctly without
      leaking raw source as visible notebook output.
    - **Google Colab**: same srcdoc iframe approach.  In Colab a brief message
      notes that ``fig.save_html()`` gives the fully interactive version.
    - **Script / terminal**: saves to a temp file and opens in the browser.

    The generated HTML is self-contained — ``save_html()`` / ``savefig()`` are
    unchanged and always produce the full standalone document.
    """
    if (_in_jupyter() or _in_colab()) and _can_display_html():
        _show_inline(html)
        return

    _show_browser(html)


def _show_inline(html: str) -> None:
    """Render *html* inside a sandboxed iframe in the notebook.

    Uses ``<iframe srcdoc="...">`` so the full standalone HTML (including
    ``<style>`` and ``<script>`` blocks) is passed to the browser for rendering
    inside the frame.  This prevents Colab from printing raw CSS/JS as visible
    text while still executing the NN-SVG JavaScript renderer.
    """
    from IPython.display import HTML, display  # type: ignore[import]

    if _in_colab():
        print(
            "NeuroSchemaX: inline preview below.  "
            "For full interactivity run:\n"
            "  fig.save_html('diagram.html')  # then download via Files panel"
        )

    notebook_html = _make_notebook_iframe(html)
    display(HTML(notebook_html))


def _show_browser(html: str) -> None:
    """Save to a temp file and open in the default web browser."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(html)
        tmp_path = fh.name
    print(f"Opening in browser: {tmp_path}")
    webbrowser.open(Path(tmp_path).as_uri())
