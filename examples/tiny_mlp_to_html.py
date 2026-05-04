"""Render a tiny MLP as a standalone HTML file.

Usage::

    python examples/tiny_mlp_to_html.py
"""

from __future__ import annotations

from pathlib import Path

import neuroschemax as nsx


def main() -> None:
    spec = {
        "model_name": "tiny_mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2", "kind": "dense", "units": 64},
            {"name": "relu2", "kind": "relu"},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    }

    output = Path("tiny_mlp.html")
    nsx.save_network_html(output, spec, theme="paper", title="Tiny MLP")
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
