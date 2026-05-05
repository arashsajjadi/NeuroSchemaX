"""HTML display helper: Jupyter/Colab inline or browser fallback."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter / IPython kernel.

    Works in local Jupyter, JupyterLab, VS Code notebooks, and Google Colab.
    """
    try:
        from IPython import get_ipython  # type: ignore[import]
        shell = get_ipython()
        # ZMQInteractiveShell (Jupyter/Colab) has a .kernel attribute;
        # plain terminal InteractiveShell does not.
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

    - **Colab**: saves a temp file, shows an IFrame preview, and prints a
      note about downloading for full interactivity.
    - **Jupyter / JupyterLab**: renders inline via ``IPython.display.HTML``.
    - **Other environments**: saves to a temp file and opens in the browser.

    In all cases the HTML is valid and self-contained — save it to a file for
    the most reliable rendering.
    """
    if _in_colab():
        _show_colab(html)
        return

    if _in_jupyter():
        try:
            from IPython.display import HTML, display  # type: ignore[import]
            display(HTML(html))
            return
        except ImportError:
            pass

    _show_browser(html)


def _show_colab(html: str) -> None:
    """Colab display: save + IFrame + informational message."""
    try:
        from IPython.display import IFrame, display  # type: ignore[import]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(html)
            tmp = fh.name
        print(
            "NeuroSchemaX: diagram saved to temp file.\n"
            "For full interactivity, download the HTML file and open it in Chrome/Firefox.\n"
            f"File: {tmp}"
        )
        display(IFrame(src=tmp, width=900, height=500))
        return
    except Exception:  # noqa: BLE001
        pass

    # Fallback: raw HTML inline (may strip scripts in some Colab versions)
    try:
        from IPython.display import HTML, display  # type: ignore[import]
        display(HTML(html))
    except Exception:  # noqa: BLE001
        print(
            "NeuroSchemaX: cannot display inline in this environment.\n"
            "Use fig.save_html('diagram.html') and download the file."
        )


def _show_browser(html: str) -> None:
    """Save to a temp file and open in the default web browser."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(html)
        tmp_path = fh.name
    print(f"Opening in browser: {tmp_path}")
    webbrowser.open(Path(tmp_path).as_uri())
