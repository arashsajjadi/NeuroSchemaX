"""Headless browser runtime for extracting SVG from generated HTML.

Uses Playwright to load the standalone HTML, wait for the NN-SVG
JavaScript to finish rendering, and extract the resulting SVG.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

from ..exceptions import BrowserNotAvailableError, RenderError


def is_playwright_available() -> bool:
    """Check if Playwright is importable."""
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


def extract_svg_from_html(
    html: str,
    timeout_ms: int = 15000,
    wait_for_ready: bool = True,
) -> str:
    """Load *html* in a headless browser and return the rendered SVG string.

    Args:
        html: The full HTML document.
        timeout_ms: How long to wait for ``window.__nnsvg_ready`` to flip true.
        wait_for_ready: If False, capture the SVG immediately after load.

    Raises:
        BrowserNotAvailableError: If Playwright is not installed.
        RenderError: If the browser fails to produce an SVG.
    """
    if not is_playwright_available():
        raise BrowserNotAvailableError()

    try:
        from playwright.sync_api import TimeoutError as PWTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserNotAvailableError() from exc

    # Write HTML to a temp file so the browser can load it with a file:// URL.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        tmp_path = Path(f.name)

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except Exception as exc:
                raise BrowserNotAvailableError() from exc

            try:
                page = browser.new_page()
                page.goto(tmp_path.as_uri(), timeout=timeout_ms)
                if wait_for_ready:
                    with contextlib.suppress(PWTimeoutError):
                        page.wait_for_function(
                            "window.__nnsvg_ready === true",
                            timeout=timeout_ms,
                        )

                svg = page.evaluate(
                    "() => window.__nnsvg_export_svg ? window.__nnsvg_export_svg() : ''"
                )
                if not svg:
                    el = page.query_selector("#diagram svg")
                    if el:
                        svg = page.evaluate(
                            "(e) => new XMLSerializer().serializeToString(e)", el
                        )
                if not svg:
                    raise RenderError("Browser produced no SVG output")
                return svg
            finally:
                browser.close()
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()


def save_svg_from_html(html: str, output_path: str | Path, **kwargs: int) -> Path:
    """Extract SVG from *html* and save to *output_path*."""
    svg = extract_svg_from_html(html, **kwargs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not svg.lstrip().startswith("<?xml"):
        svg = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + svg
    out.write_text(svg, encoding="utf-8")
    return out
