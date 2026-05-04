// NN-SVG compatible AlexNet-style deep CNN renderer.
// Like LeNet but optimised for denser, deeper architectures: smaller
// feature-map rectangles, tighter spacing, and capped classifier columns.
(function() {
  "use strict";

  function renderAlexNet(spec, container) {
    var U = window.NNSVGUtil;
    U.clear(container);

    var w = spec.width || 1400;
    var h = spec.height || 700;
    var svg = U.createSVG(w, h);
    var layers = spec.layers || [];
    var n = layers.length;
    if (n === 0) {
      container.appendChild(svg);
      return;
    }

    var marginX = 40;
    var marginY = 40;
    var availW = w - 2 * marginX;
    var availH = h - 2 * marginY;
    var layerSpacing = availW / Math.max(1, n);
    var edgeOpacity = spec.edgeOpacity != null ? spec.edgeOpacity : 0.3;
    var nodeScale = (spec.nodeSize || 1.0) * 0.85;

    var centres = [];
    for (var i = 0; i < n; i++) {
      centres.push(marginX + layerSpacing * (i + 0.5));
    }

    for (var i = 0; i < n - 1; i++) {
      U.el("line", {
        x1: centres[i], y1: h / 2,
        x2: centres[i + 1], y2: h / 2,
        stroke: "#888", "stroke-width": "1.2",
        opacity: edgeOpacity
      }, svg);
    }

    // Dense (classifier) columns capped tightly to avoid visual dominance.
    var MAX_DENSE = 8;
    // Dense columns are rendered at 60 % of canvas height, vertically centred.
    var DENSE_FRAC = 0.6;

    for (var i = 0; i < n; i++) {
      var x = centres[i];
      var layer = layers[i];
      var color = layer.color || "#93C47D";

      if (layer.layerType === "dense") {
        var units = U.clamp(layer.units || 5, 1, MAX_DENSE);
        var colH = availH * DENSE_FRAC;
        var colTop = (h - colH) / 2;
        var r = U.clamp((colH / (units + 1)) * 0.22, 3, 9) * nodeScale;
        for (var j = 0; j < units; j++) {
          var y = colTop + (colH / (units + 1)) * (j + 1);
          U.el("circle", {
            cx: x, cy: y, r: r,
            fill: color, stroke: "#333", "stroke-width": "0.8"
          }, svg);
        }
      } else {
        var ch = U.clamp(layer.channels || 1, 1, 10);
        var fmH = U.clamp(layer.featureMapHeight || 20, 8, 90) * nodeScale;
        var fmW = U.clamp(layer.featureMapWidth  || 20, 8, 90) * nodeScale;
        var baseY = (h - fmH) / 2;
        for (var c = ch - 1; c >= 0; c--) {
          var offset = c * 3;
          U.el("rect", {
            x: x - fmW / 2 + offset,
            y: baseY - offset,
            width: fmW, height: fmH,
            fill: color, "fill-opacity": 0.7,
            stroke: "#333", "stroke-width": "0.8"
          }, svg);
        }
      }

      if (spec.showLabels && layer.label) {
        // Multi-line aware label.  Anchor near the bottom but lift if there
        // are several stacked lines so the last line stays inside the canvas.
        var fs = spec.fontSize || 10;
        var nLines = layer.label.split("\n").length;
        var lh = fs * 1.2;
        var labelY = Math.min(h - 6 - (nLines - 1) * lh, h - 15);
        var t = U.el("text", {
          x: x, y: labelY,
          "text-anchor": "middle",
          "font-size": fs,
          "font-family": spec.fontFamily || "sans-serif",
          fill: "#333"
        }, svg);
        var ll = layer.label.split("\n");
        for (var li = 0; li < ll.length; li++) {
          var ts = U.el("tspan", {
            x: x,
            dy: li === 0 ? 0 : lh
          }, t);
          ts.textContent = ll[li];
        }
      }
    }

    container.appendChild(svg);
  }

  window.renderAlexNet = renderAlexNet;
})();
