"""Render a manually-written JSON spec to both HTML and SVG.

Usage::

    python examples/manual_spec_to_svg.py
"""

from __future__ import annotations

import json
from pathlib import Path

import neuroschemax as nsx


SPEC = {
    "model_name": "example_arch",
    "layers": [
        {"name": "input",  "kind": "input",  "shape": [1, 3, 32, 32]},
        {"name": "conv1",  "kind": "conv",   "out_channels": 32, "kernel_size": [3, 3]},
        {"name": "bn1",    "kind": "batchnormalization"},
        {"name": "relu1",  "kind": "relu"},
        {"name": "pool1",  "kind": "maxpool", "kernel_size": [2, 2]},
        {"name": "conv2",  "kind": "conv",   "out_channels": 64, "kernel_size": [3, 3]},
        {"name": "bn2",    "kind": "batchnormalization"},
        {"name": "relu2",  "kind": "relu"},
        {"name": "pool2",  "kind": "maxpool", "kernel_size": [2, 2]},
        {"name": "flatten", "kind": "flatten"},
        {"name": "fc1",    "kind": "dense",  "units": 128},
        {"name": "out",    "kind": "dense",  "units": 10},
    ],
}


def main() -> None:
    spec_file = Path("example_arch.json")
    spec_file.write_text(json.dumps(SPEC, indent=2))
    print(f"Wrote spec: {spec_file}")

    html_file = Path("example_arch.html")
    nsx.save_network_html(html_file, spec_file, theme="paper")
    print(f"Wrote HTML: {html_file}")

    svg_file = Path("example_arch.svg")
    try:
        nsx.save_network_svg(svg_file, spec_file, theme="paper")
        print(f"Wrote SVG:  {svg_file}")
    except nsx.BrowserNotAvailableError as exc:
        print(f"Skipping SVG (Playwright not installed): {exc}")


if __name__ == "__main__":
    main()
