// NN-SVG compatible LeNet-style CNN renderer.
// Conv/pool layers are drawn as stacked feature-map rectangles (pseudo-3D);
// dense layers are drawn as columns of circles.
//
// Special case: when channels === 1, the layer is treated as an opaque
// block (used for Transformer/block-level approximations).  The label is
// rendered centered *inside* the rectangle instead of below it.
(function() {
  "use strict";

  function renderLeNet(spec, container) {
    var U = window.NNSVGUtil;
    U.clear(container);

    var w = spec.width || 1200;
    var h = spec.height || 700;
    var svg = U.createSVG(w, h);
    var layers = spec.layers || [];
    var n = layers.length;
    if (n === 0) {
      container.appendChild(svg);
      return;
    }

    var marginX = 60;
    var marginY = 60;
    var availW = w - 2 * marginX;
    var availH = h - 2 * marginY;
    var layerSpacing = availW / Math.max(1, n);
    var edgeOpacity = spec.edgeOpacity != null ? spec.edgeOpacity : 0.35;
    var nodeScale = spec.nodeSize || 1.0;

    var centres = [];
    for (var i = 0; i < n; i++) {
      centres.push(marginX + layerSpacing * (i + 0.5));
    }

    // Centre-to-centre connections
    for (var i = 0; i < n - 1; i++) {
      U.el("line", {
        x1: centres[i], y1: h / 2,
        x2: centres[i + 1], y2: h / 2,
        stroke: "#888", "stroke-width": "1.5",
        opacity: edgeOpacity
      }, svg);
    }

    var MAX_DENSE = 10;
    var fontSize = spec.fontSize || 11;

    for (var i = 0; i < n; i++) {
      var x = centres[i];
      var layer = layers[i];
      var color = layer.color || "#93C47D";

      if (layer.layerType === "dense") {
        var units = U.clamp(layer.units || 5, 1, MAX_DENSE);
        var colH = availH * 0.7;
        var colTop = (h - colH) / 2;
        var r = U.clamp((colH / (units + 1)) * 0.28, 4, 12) * nodeScale;
        for (var j = 0; j < units; j++) {
          var y = colTop + (colH / (units + 1)) * (j + 1);
          U.el("circle", {
            cx: x, cy: y, r: r,
            fill: color, stroke: "#333", "stroke-width": "1"
          }, svg);
        }
        if (spec.showLabels && layer.label) {
          U.label(svg, x, h - 20, layer.label, spec);
        }

      } else {
        // Conv, pool, input — stacked pseudo-3D rectangles
        var ch = U.clamp(layer.channels || 1, 1, 8);
        var fmH = U.clamp(layer.featureMapHeight || 28, 10, 120) * nodeScale;
        var fmW = U.clamp(layer.featureMapWidth  || 28, 10, 120) * nodeScale;
        var baseY = (h - fmH) / 2;

        for (var c = ch - 1; c >= 0; c--) {
          var offset = c * 4;
          U.el("rect", {
            x: x - fmW / 2 + offset,
            y: baseY - offset,
            width: fmW, height: fmH,
            fill: color, "fill-opacity": 0.75,
            stroke: "#333", "stroke-width": "1"
          }, svg);
        }

        if (spec.showLabels && layer.label) {
          if (ch === 1) {
            // Single-channel block (e.g. Transformer stage): label(s) centered
            // inside the rectangle.  Labels may embed '\n' for multi-line output.
            // Font size is auto-scaled so the longest line fits within the box.
            var rawLabel = layer.label || "";
            var lines = rawLabel.split('\n').filter(function(s) { return s.length > 0; });
            if (lines.length === 0) lines = [rawLabel];

            var maxLineLen = 0;
            for (var li = 0; li < lines.length; li++) {
              if (lines[li].length > maxLineLen) maxLineLen = lines[li].length;
            }

            var availW = fmW - 8;  // 4 px padding each side
            var textFontSize = fontSize;
            if (maxLineLen > 0) {
              var estW = maxLineLen * fontSize * 0.62;
              if (estW > availW) {
                textFontSize = Math.max(8, Math.floor(availW / (maxLineLen * 0.62)));
              }
            }

            var lineH = textFontSize * 1.35;
            var totalTextH = lineH * lines.length;
            var startY = baseY + (fmH - totalTextH) / 2 + textFontSize;

            for (var li = 0; li < lines.length; li++) {
              U.el("text", {
                x: x,
                y: startY + lineH * li,
                "text-anchor": "middle",
                "font-size": textFontSize,
                "font-family": spec.fontFamily || "sans-serif",
                fill: "#222",
                "font-weight": "600"
              }, svg).textContent = lines[li];
            }
          } else {
            // Multi-channel stack: label below the diagram.
            U.label(svg, x, h - 20, layer.label, spec);
          }
        }
      }
    }

    container.appendChild(svg);
  }

  window.renderLeNet = renderLeNet;
})();
