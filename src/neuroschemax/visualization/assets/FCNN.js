// NN-SVG compatible Fully-Connected Neural Network renderer.
// Draws a classic MLP diagram: columns of neurons with fully-connected edges.
(function() {
  "use strict";

  function renderFCNN(spec, container) {
    var U = window.NNSVGUtil;
    U.clear(container);

    var w = spec.width || 1200;
    var h = spec.height || 700;
    var svg = U.createSVG(w, h);
    var layers = (spec.layers || []).filter(function(l) { return (l.units || 0) > 0; });
    var n = layers.length;
    if (n === 0) {
      container.appendChild(svg);
      return;
    }

    var marginX = 80;
    var marginY = 60;
    var availW = w - 2 * marginX;
    var availH = h - 2 * marginY;
    var layerSpacing = availW / Math.max(1, n - 1);

    // Reduced default opacity for readability in dense networks.
    var edgeOpacity = spec.edgeOpacity != null ? spec.edgeOpacity : 0.18;
    var nodeScale = spec.nodeSize || 1.0;

    // Cap display units at 12 (was 20) to keep edge count manageable.
    // For a 12x12 pair that is 144 edges — already dense but still readable.
    var MAX_DISPLAY = 12;
    var displayUnits = layers.map(function(l) {
      return U.clamp(l.units || 1, 1, MAX_DISPLAY);
    });

    // Edge sampling: for adjacent layer pairs whose product exceeds the
    // SAMPLE_THRESHOLD, only draw edges from every other source neuron.
    var SAMPLE_THRESHOLD = 60;

    // Edges first (nodes drawn on top)
    for (var i = 0; i < n - 1; i++) {
      var x1 = marginX + layerSpacing * i;
      var x2 = marginX + layerSpacing * (i + 1);
      var u1 = displayUnits[i];
      var u2 = displayUnits[i + 1];
      var step = (u1 * u2 > SAMPLE_THRESHOLD) ? 2 : 1;
      for (var a = 0; a < u1; a += step) {
        var y1 = marginY + (availH / (u1 + 1)) * (a + 1);
        for (var b = 0; b < u2; b++) {
          var y2 = marginY + (availH / (u2 + 1)) * (b + 1);
          U.el("line", {
            x1: x1, y1: y1, x2: x2, y2: y2,
            stroke: "#888", "stroke-width": "0.5",
            opacity: edgeOpacity
          }, svg);
        }
      }
    }

    // Nodes
    for (var i = 0; i < n; i++) {
      var x = marginX + layerSpacing * i;
      var u = displayUnits[i];
      var color = layers[i].color || "#6FA8DC";
      var r = U.clamp((availH / (u + 1)) * 0.3, 4, 16) * nodeScale;
      for (var j = 0; j < u; j++) {
        var y = marginY + (availH / (u + 1)) * (j + 1);
        U.el("circle", {
          cx: x, cy: y, r: r,
          fill: color, stroke: "#333", "stroke-width": "1"
        }, svg);
      }
      if (spec.showLabels && layers[i].label) {
        U.label(svg, x, h - 20, layers[i].label, spec);
      }
    }

    container.appendChild(svg);
  }

  window.renderFCNN = renderFCNN;
})();
