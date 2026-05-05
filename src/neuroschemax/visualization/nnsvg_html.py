"""Generate standalone HTML files that render NN-SVG diagrams.

The HTML embeds the NeuroSchemaX vanilla-SVG renderer scripts so that it
works offline with no external dependencies.  D3 and Three.js are NOT
required — the bundled renderers (FCNN.js, LeNet.js, AlexNet.js) use
only native SVG DOM APIs.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.enums import RenderFamily
from .nnsvg_schema import NNSVGSpec

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _read_asset(filename: str) -> str:
    """Read a bundled JS asset file."""
    path = _ASSETS_DIR / filename
    if not path.exists():
        return f"// Asset not found: {filename}\n// Run: neuroschemax doctor\n"
    return path.read_text(encoding="utf-8", errors="replace")


def _family_js_file(family: RenderFamily) -> str:
    return {
        RenderFamily.FCNN: "FCNN.js",
        RenderFamily.LENET: "LeNet.js",
        RenderFamily.ALEXNET: "AlexNet.js",
    }[family]


def generate_html(spec: NNSVGSpec) -> str:
    """Generate a standalone HTML document that renders the NN-SVG diagram.

    The document:
    - Embeds util.js and the family-specific renderer inline
    - Passes the spec as a JSON object
    - Exposes ``window.__nnsvg_ready`` and ``window.__nnsvg_export_svg()``
      for headless SVG extraction

    When ``spec.diagnostic`` is set the body renders a structured
    diagnostic card instead of the SVG diagram — used by
    ``transformer_mode="unsupported"`` so the page is professional and
    readable instead of a tiny placeholder box.
    """
    spec_json = json.dumps(spec.to_dict(), indent=2)
    family_file = _family_js_file(spec.family)

    util_js = _read_asset("util.js")
    family_js = _read_asset(family_file)

    title = spec.title or spec.model_name or "NN-SVG Diagram"
    diagnostic_html = _render_diagnostic_card(spec)
    legend_html = _render_legend(spec)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape_html(title)}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #fff; font-family: {spec.font_family}, sans-serif; color: #333; }}
  #diagram {{ width: {spec.width}px; height: {spec.height}px; margin: 16px auto 0; display: block; }}
  .nnsvg-header {{ max-width: {spec.width}px; margin: 20px auto 0; padding: 0 8px; }}
  .nnsvg-title {{ font-size: 17px; font-weight: 600; color: #222; margin-bottom: 6px; }}
  .nnsvg-warnings {{ margin-top: 8px; }}
  .nnsvg-warning-item {{
    display: inline-block;
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 11px;
    color: #795548;
    margin: 2px 0;
    line-height: 1.5;
  }}
  .nnsvg-warning-label {{
    font-weight: 600;
    color: #e65100;
    margin-right: 4px;
  }}
  .nnsvg-subtitle {{
    font-size: 11px;
    color: #888;
    margin-top: 2px;
    letter-spacing: 0.3px;
  }}
  .nnsvg-diagnostic {{
    max-width: 720px;
    margin: 24px auto;
    padding: 24px 28px;
    background: #fafbff;
    border: 1px solid #c5cae9;
    border-left: 4px solid #5c6bc0;
    border-radius: 8px;
    color: #2c3038;
    line-height: 1.55;
  }}
  .nnsvg-diagnostic-title {{
    font-size: 18px;
    font-weight: 600;
    color: #1a237e;
    margin-bottom: 10px;
  }}
  .nnsvg-diagnostic-body {{
    font-size: 14px;
    margin-bottom: 14px;
  }}
  .nnsvg-diagnostic-actions {{
    margin: 0 0 16px 22px;
    padding: 0;
    font-size: 14px;
  }}
  .nnsvg-diagnostic-actions li {{
    margin-bottom: 6px;
  }}
  .nnsvg-diagnostic-meta {{
    font-size: 12px;
    color: #555;
    border-top: 1px dashed #c5cae9;
    padding-top: 10px;
    margin-top: 6px;
  }}
  .nnsvg-diagnostic-meta code {{
    background: #eef0fb;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 12px;
  }}
  .nnsvg-legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    margin-top: 8px;
    padding: 6px 0;
    border-top: 1px solid #f0f0f0;
    font-size: 10px;
    color: #999;
    align-items: center;
  }}
  .nnsvg-legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
  }}
  .nnsvg-legend-swatch {{
    width: 10px;
    height: 10px;
    border-radius: 2px;
    border: 1px solid rgba(0,0,0,0.15);
    flex-shrink: 0;
  }}
  .nnsvg-legend-approx {{
    font-style: italic;
    color: #aaa;
  }}
</style>
</head>
<body>
<div class="nnsvg-header">
{"<div class='nnsvg-title'>" + _escape_html(title) + "</div>" if title else ""}
{"<div class='nnsvg-subtitle'>" + _escape_html(spec.subtitle) + "</div>" if spec.subtitle else ""}
{_render_warnings_html(spec.warnings)}
{legend_html}
</div>
{diagnostic_html}
<div id="diagram"{' style="display:none"' if diagnostic_html else ""}></div>

<script>
// ── NN-SVG shared utilities ──────────────────────────────────────────────────
{util_js}
</script>
<script>
// ── NN-SVG family renderer ───────────────────────────────────────────────────
{family_js}
</script>
<script>
// ── NeuroSchemaX runtime bootstrap ──────────────────────────────────────────
(function() {{
  "use strict";

  var SPEC = {spec_json};

  window.__nnsvg_ready = false;

  function render() {{
    try {{
      if (typeof window.renderNNSVG === "function") {{
        window.renderNNSVG(SPEC, document.getElementById("diagram"));
      }} else {{
        renderFallback(SPEC, document.getElementById("diagram"));
      }}
    }} catch(e) {{
      console.error("NN-SVG render error:", e);
      renderFallback(SPEC, document.getElementById("diagram"));
    }}
    window.__nnsvg_ready = true;
  }}

  function renderFallback(spec, container) {{
    var w = spec.width || 1200;
    var h = spec.height || 700;
    var layers = spec.layers || [];
    var n = layers.length;
    if (n === 0) return;

    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    svg.setAttribute("width", w);
    svg.setAttribute("height", h);
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);

    var family = spec.family || "fcnn";
    if (family === "fcnn") {{
      renderFCNNFallback(svg, spec, layers, w, h);
    }} else {{
      renderCNNFallback(svg, spec, layers, w, h);
    }}

    container.innerHTML = "";
    container.appendChild(svg);
  }}

  function renderFCNNFallback(svg, spec, layers, w, h) {{
    var n = layers.length;
    var layerSpacing = w / (n + 1);
    var edgeOpacity = spec.edgeOpacity || 0.4;
    var nodeScale = spec.nodeSize || 1.0;

    for (var i = 0; i < n - 1; i++) {{
      var x1 = layerSpacing * (i + 1);
      var x2 = layerSpacing * (i + 2);
      var u1 = Math.min(layers[i].units || 1, 20);
      var u2 = Math.min(layers[i+1].units || 1, 20);
      for (var a = 0; a < u1; a++) {{
        for (var b = 0; b < u2; b++) {{
          var y1 = (h / (u1 + 1)) * (a + 1);
          var y2 = (h / (u2 + 1)) * (b + 1);
          var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
          line.setAttribute("x1", x1); line.setAttribute("y1", y1);
          line.setAttribute("x2", x2); line.setAttribute("y2", y2);
          line.setAttribute("stroke", "#aaa"); line.setAttribute("stroke-width", "0.5");
          line.setAttribute("opacity", edgeOpacity);
          svg.appendChild(line);
        }}
      }}
    }}

    for (var i = 0; i < n; i++) {{
      var x = layerSpacing * (i + 1);
      var units = Math.min(layers[i].units || 1, 20);
      var color = layers[i].color || "#6FA8DC";
      var r = Math.max(4, Math.min(14, (h / (units + 1)) * 0.3)) * nodeScale;
      for (var j = 0; j < units; j++) {{
        var y = (h / (units + 1)) * (j + 1);
        var circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", x); circle.setAttribute("cy", y); circle.setAttribute("r", r);
        circle.setAttribute("fill", color); circle.setAttribute("stroke", "#333");
        circle.setAttribute("stroke-width", "1");
        svg.appendChild(circle);
      }}
      if (spec.showLabels && layers[i].label) {{
        var text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", x); text.setAttribute("y", h - 10);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", (spec.fontSize || 11) + "px");
        text.setAttribute("font-family", spec.fontFamily || "sans-serif");
        text.setAttribute("fill", "#333");
        text.textContent = layers[i].label;
        svg.appendChild(text);
      }}
    }}
  }}

  function renderCNNFallback(svg, spec, layers, w, h) {{
    var n = layers.length;
    var layerSpacing = w / (n + 1);
    var edgeOpacity = spec.edgeOpacity || 0.4;
    var nodeScale = spec.nodeSize || 1.0;

    for (var i = 0; i < n - 1; i++) {{
      var line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", layerSpacing * (i + 1)); line.setAttribute("y1", h / 2);
      line.setAttribute("x2", layerSpacing * (i + 2)); line.setAttribute("y2", h / 2);
      line.setAttribute("stroke", "#aaa"); line.setAttribute("stroke-width", "2");
      line.setAttribute("opacity", edgeOpacity);
      svg.appendChild(line);
    }}

    for (var i = 0; i < n; i++) {{
      var x = layerSpacing * (i + 1);
      var layer = layers[i];
      var color = layer.color || "#93C47D";
      if (layer.layerType === "dense") {{
        var units = Math.min(layer.units || 5, 12);
        var r = Math.max(3, Math.min(10, (h / (units + 1)) * 0.25)) * nodeScale;
        for (var j = 0; j < units; j++) {{
          var y = (h / (units + 1)) * (j + 1);
          var circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", x); circle.setAttribute("cy", y); circle.setAttribute("r", r);
          circle.setAttribute("fill", color); circle.setAttribute("stroke", "#333");
          svg.appendChild(circle);
        }}
      }} else {{
        var ch = Math.min(layer.channels || 1, 8);
        var fmH = Math.min(layer.featureMapHeight || 28, 80) * nodeScale;
        var fmW = Math.min(layer.featureMapWidth || 28, 80) * nodeScale;
        var baseY = (h - fmH) / 2;
        for (var c = 0; c < ch; c++) {{
          var offset = c * 3;
          var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
          rect.setAttribute("x", x - fmW / 2 + offset); rect.setAttribute("y", baseY - offset);
          rect.setAttribute("width", fmW); rect.setAttribute("height", fmH);
          rect.setAttribute("fill", color); rect.setAttribute("fill-opacity", 0.6);
          rect.setAttribute("stroke", "#333");
          svg.appendChild(rect);
        }}
      }}
      if (spec.showLabels && layer.label) {{
        var text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", x); text.setAttribute("y", h - 10);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("font-size", (spec.fontSize || 11) + "px");
        text.setAttribute("fill", "#333");
        text.textContent = layer.label;
        svg.appendChild(text);
      }}
    }}
  }}

  window.__nnsvg_export_svg = function() {{
    var el = document.querySelector("#diagram svg");
    if (!el) return "";
    return new XMLSerializer().serializeToString(el);
  }};

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", render);
  }} else {{
    render();
  }}
}})();
</script>
</body>
</html>"""


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_warnings_html(warnings: list[str]) -> str:
    """Render a list of warnings as styled HTML badges."""
    if not warnings:
        return ""
    items = "".join(
        f"<div class='nnsvg-warning-item'>"
        f"<span class='nnsvg-warning-label'>Approximate:</span>"
        f"{_escape_html(w)}"
        f"</div>"
        for w in warnings
    )
    return f"<div class='nnsvg-warnings'>{items}</div>"


_LEGEND_ENTRIES: list[tuple[str, str]] = [
    ("#9FC5E8", "Input"),
    ("#93C47D", "Conv"),
    ("#F6B26B", "Pool"),
    ("#6FA8DC", "Dense"),
    ("#E06666", "Classifier"),
    ("#B4A7D6", "Norm"),
    ("#EA9999", "Attention"),
    ("#E8EAF6", "Unsupported"),
]


def _render_legend(spec: NNSVGSpec) -> str:
    """Render a compact colour-key legend strip.

    Only rendered when ``spec.show_legend`` is True and the spec is not in
    full diagnostic mode (which has its own explanatory card).
    """
    if not spec.show_legend or spec.diagnostic:
        return ""

    is_approx = bool(spec.warnings)
    items = "".join(
        f"<span class='nnsvg-legend-item'>"
        f"<span class='nnsvg-legend-swatch' style='background:{color}'></span>"
        f"{_escape_html(label)}"
        f"</span>"
        for color, label in _LEGEND_ENTRIES
    )
    approx_note = (
        "<span class='nnsvg-legend-approx'>· approximate — see warnings</span>"
        if is_approx else
        "<span class='nnsvg-legend-approx'>· exact</span>"
    )
    return f"<div class='nnsvg-legend'>{items}{approx_note}</div>"


def _render_diagnostic_card(spec: NNSVGSpec) -> str:
    """Render a structured diagnostic card when ``spec.diagnostic`` is set.

    The card replaces the SVG diagram for cases where exact rendering is not
    possible (e.g. ``transformer_mode="unsupported"``).  It is wide,
    properly typeset, and never overflows — content scales to the container
    rather than to a fixed-width SVG box.
    """
    info = spec.diagnostic
    if not info:
        return ""
    headline = _escape_html(info.get("headline", "Cannot render exactly"))
    body = _escape_html(info.get("body", ""))
    actions = info.get("actions") or []
    detected = _escape_html(info.get("detected", ""))
    actions_html = "".join(
        f"<li>{_escape_html(a)}</li>" for a in actions
    )
    actions_block = (
        f"<ul class='nnsvg-diagnostic-actions'>{actions_html}</ul>"
        if actions_html else ""
    )
    detected_block = (
        f"<div class='nnsvg-diagnostic-meta'>"
        f"Detected components: <code>{detected}</code>"
        f"</div>"
        if detected else ""
    )
    return (
        "<div class='nnsvg-diagnostic' role='note'>"
        f"<div class='nnsvg-diagnostic-title'>{headline}</div>"
        f"<div class='nnsvg-diagnostic-body'>{body}</div>"
        f"{actions_block}"
        f"{detected_block}"
        "</div>"
    )
