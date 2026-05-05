"""HTML display helper: Jupyter/Colab inline or browser fallback."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path


def _can_display_html() -> bool:
    """Return True when IPython.display.HTML / display are importable.

    This is the correct test for notebook-display capability — not whether the
    *ipython* package (lowercase) can be imported as a module, but whether the
    actual display API is available.  Colab and Jupyter both satisfy this even
    when ``import ipython`` (lowercase) would fail.
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


def show_html(html: str) -> None:
    """Display *html*:

    - **Jupyter / JupyterLab / VS Code notebooks**: renders inline via
      ``IPython.display.HTML``.
    - **Google Colab**: renders inline via ``IPython.display.HTML``.  Full
      JavaScript interactivity may be restricted by Colab's content-security
      policy; call ``fig.save_html("diagram.html")`` and download the file
      for the fully interactive version.
    - **Script / terminal**: saves to a temp file and opens in the browser.

    The generated HTML is self-contained in all cases — saving to a file
    always works regardless of display environment.
    """
    if (_in_jupyter() or _in_colab()) and _can_display_html():
        _show_inline(html)
        return

    _show_browser(html)


def _show_inline(html: str) -> None:
    """Render *html* inline using ``IPython.display.HTML``.

    In Colab, the browser's content-security policy may suppress JavaScript
    inside ``display(HTML(...))`` blocks.  We print a brief note when running
    in Colab so the user knows to download the file for the fully interactive
    diagram.  We do NOT fall back to a local-file IFrame because Colab's
    browser sandbox cannot access ``/tmp/`` paths on the VM.
    """
    from IPython.display import HTML, display  # type: ignore[import]

    if _in_colab():
        print(
            "NeuroSchemaX: showing inline preview.  "
            "For full JavaScript interactivity, run:\n"
            "  fig.save_html('diagram.html')  # then download via Files panel"
        )

    display(HTML(html))


def _show_browser(html: str) -> None:
    """Save to a temp file and open in the default web browser."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(html)
        tmp_path = fh.name
    print(f"Opening in browser: {tmp_path}")
    webbrowser.open(Path(tmp_path).as_uri())
