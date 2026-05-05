"""Headless browser runtime for extracting SVG from generated HTML.

Uses Playwright to load the standalone HTML, wait for the NN-SVG JavaScript
to finish rendering, and extract the resulting SVG.  The returned SVG has a
viewBox that matches the actual content bounding box (not a fixed viewport
screenshot), so it can be scaled without cropping or large whitespace areas.
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


def _raise_browser_not_available() -> None:
    raise BrowserNotAvailableError(
        "SVG/PNG export requires Playwright and Chromium.\n"
        "Install:  pip install playwright\n"
        "Then:     playwright install chromium\n"
        "Or:       pip install \"neuroschemax[svg]\"\n"
        "\nStandalone HTML export (save_network_html) works without any browser."
    )


def extract_svg_from_html(
    html: str,
    timeout_ms: int = 15000,
    wait_for_ready: bool = True,
) -> str:
    """Load *html* in a headless browser and return the rendered SVG string.

    The returned SVG element's ``viewBox`` is set to the content bounding box
    of the rendered diagram, so there is no unnecessary whitespace and labels
    are not cropped.  Works for both normal diagrams and diagnostic-mode pages.

    Args:
        html: The full HTML document.
        timeout_ms: How long to wait for ``window.__nnsvg_ready`` to flip true.
        wait_for_ready: If False, capture the SVG immediately after load.

    Raises:
        BrowserNotAvailableError: If Playwright is not installed or Chromium
            is not available.  The error message includes install instructions.
        RenderError: If the browser fails to produce an SVG.
    """
    if not is_playwright_available():
        _raise_browser_not_available()

    try:
        from playwright.sync_api import TimeoutError as PWTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError:
        _raise_browser_not_available()

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
                raise BrowserNotAvailableError(
                    "Could not launch Chromium.\n"
                    "Run: playwright install chromium"
                ) from exc

            try:
                page = browser.new_page()
                page.goto(tmp_path.as_uri(), timeout=timeout_ms)

                if wait_for_ready:
                    with contextlib.suppress(PWTimeoutError):
                        page.wait_for_function(
                            "window.__nnsvg_ready === true",
                            timeout=timeout_ms,
                        )

                # Extract SVG using the exported helper, then clean up the
                # viewBox using the SVG element's getBBox() so the result
                # matches the actual content size rather than the full canvas.
                svg = page.evaluate("""() => {
                    var fn = window.__nnsvg_export_svg;
                    if (fn) return fn();
                    var el = document.querySelector('#diagram svg');
                    if (el) return new XMLSerializer().serializeToString(el);
                    return '';
                }""")

                if not svg:
                    raise RenderError(
                        "Browser produced no SVG output.  "
                        "The diagram may have failed to render — try opening the "
                        "HTML file in a browser and checking the console for errors."
                    )

                # Tighten the viewBox to the actual content bounding box.
                svg = page.evaluate("""(svgStr) => {
                    var tmp = document.createElement('div');
                    tmp.innerHTML = svgStr;
                    var el = tmp.querySelector('svg');
                    if (!el) return svgStr;
                    document.body.appendChild(el);
                    try {
                        var bb = el.getBBox();
                        var pad = 8;
                        if (bb && bb.width > 0 && bb.height > 0) {
                            var x = Math.max(0, bb.x - pad);
                            var y = Math.max(0, bb.y - pad);
                            var w = bb.width + pad * 2;
                            var h = bb.height + pad * 2;
                            el.setAttribute('viewBox', x + ' ' + y + ' ' + w + ' ' + h);
                            el.setAttribute('width', w);
                            el.setAttribute('height', h);
                        }
                    } finally {
                        document.body.removeChild(el);
                    }
                    return new XMLSerializer().serializeToString(el);
                }""", svg)

                return svg or ""

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
