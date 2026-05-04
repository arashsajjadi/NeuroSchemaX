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
    """
    spec_json = json.dumps(spec.to_dict(), indent=2)
    family_file = _family_js_file(spec.family)

    util_js = _read_asset("util.js")
    family_js = _read_asset(family_file)

    title = spec.title or spec.model_name or "NN-SVG Diagram"

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
</style>
</head>
<body>
<div class="nnsvg-header">
{"<div class='nnsvg-title'>" + _escape_html(title) + "</div>" if title else ""}
{_render_warnings_html(spec.warnings)}
</div>
<div id="diagram"></div>

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
