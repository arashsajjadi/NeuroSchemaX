"""Render a tiny CNN as an SVG file (requires Playwright).

Install Playwright first::

    pip install playwright
    playwright install chromium

Then run::

    python examples/tiny_cnn_to_svg.py
"""

from __future__ import annotations

from pathlib import Path

import neuroschemax as nsx


def main() -> None:
    spec = {
        "model_name": "tiny_cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16,
             "kernel_size": [3, 3], "stride": [1, 1]},
            {"name": "relu1", "kind": "relu"},
            {"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "conv2", "kind": "conv", "out_channels": 32,
             "kernel_size": [3, 3]},
            {"name": "relu2", "kind": "relu"},
            {"name": "pool2", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "flatten", "kind": "flatten"},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    }

    output = Path("tiny_cnn.svg")
    try:
        nsx.save_network_svg(output, spec, theme="paper", title="Tiny CNN")
    except nsx.BrowserNotAvailableError as exc:
        print(f"SVG export unavailable: {exc}")
        print("Falling back to HTML output…")
        nsx.save_network_html(output.with_suffix(".html"), spec, theme="paper")
        return

    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
