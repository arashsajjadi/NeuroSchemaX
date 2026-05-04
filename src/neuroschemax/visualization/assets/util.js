// NeuroSchemaX / NN-SVG shared utilities.
// Exposes a small helper set on window.NNSVGUtil and registers a dispatch
// function window.renderNNSVG(spec, container) that forwards to the
// family-specific renderer based on spec.family.
(function() {
  "use strict";

  var Util = {
    svgNS: "http://www.w3.org/2000/svg",

    clear: function(container) {
      while (container.firstChild) container.removeChild(container.firstChild);
    },

    createSVG: function(width, height) {
      var svg = document.createElementNS(Util.svgNS, "svg");
      svg.setAttribute("xmlns", Util.svgNS);
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
      svg.setAttribute("viewBox", "0 0 " + width + " " + height);
      return svg;
    },

    el: function(tag, attrs, parent) {
      var n = document.createElementNS(Util.svgNS, tag);
      if (attrs) {
        for (var k in attrs) {
          if (Object.prototype.hasOwnProperty.call(attrs, k)) {
            n.setAttribute(k, attrs[k]);
          }
        }
      }
      if (parent) parent.appendChild(n);
      return n;
    },

    clamp: function(x, lo, hi) {
      return Math.max(lo, Math.min(hi, x));
    },

    label: function(svg, x, y, text, spec) {
      // Render *text* as a multi-line label centred horizontally at (x, y).
      // *y* is the baseline of the FIRST line; subsequent lines stack DOWN.
      // Multi-line input is split on '\n'.  Each line uses a <tspan> so SVG
      // serialisation preserves the layout when exported.
      if (!spec.showLabels || !text) return;
      var fs = spec.fontSize || 11;
      var lines = String(text).split("\n");
      var t = Util.el("text", {
        x: x, y: y,
        "text-anchor": "middle",
        "font-size": fs,
        "font-family": spec.fontFamily || "sans-serif",
        fill: "#333"
      }, svg);
      for (var i = 0; i < lines.length; i++) {
        var ts = Util.el("tspan", {
          x: x,
          dy: i === 0 ? 0 : (fs * 1.2)
        }, t);
        ts.textContent = lines[i];
      }
    }
  };

  window.NNSVGUtil = Util;

  window.renderNNSVG = function(spec, container) {
    var family = (spec && spec.family) || "fcnn";
    if (family === "fcnn" && window.renderFCNN) {
      window.renderFCNN(spec, container);
    } else if (family === "lenet" && window.renderLeNet) {
      window.renderLeNet(spec, container);
    } else if (family === "alexnet" && window.renderAlexNet) {
      window.renderAlexNet(spec, container);
    } else {
      throw new Error("Unknown family or missing renderer: " + family);
    }
  };
})();
