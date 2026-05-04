"""HTML display helper: Jupyter-inline or browser fallback."""

from __future__ import annotations


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter / IPython kernel."""
    try:
        from IPython import get_ipython  # type: ignore[import]
        shell = get_ipython()
        # ZMQInteractiveShell has a .kernel attribute; plain terminal shells do not.
        return shell is not None and hasattr(shell, "kernel")
    except ImportError:
        return False


def show_html(html: str) -> None:
    """Display *html*:
    - inline via ``IPython.display`` when inside a Jupyter/IPython kernel, or
    - in the default web browser otherwise.
    """
    if _in_jupyter():
        try:
            from IPython.display import HTML, display  # type: ignore[import]
            display(HTML(html))
            return
        except ImportError:
            pass

    import tempfile
    import webbrowser
    from pathlib import Path

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(html)
        tmp_path = fh.name
    print(f"Opening in browser: {tmp_path}")
    webbrowser.open(Path(tmp_path).as_uri())
