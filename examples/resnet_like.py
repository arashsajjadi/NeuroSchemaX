"""Example: ResNet-like model with skip connections.

This example shows how NeuroSchemaX handles residual/skip connections.
Because NN-SVG does not natively represent skip links, the diagram approximates
the model as a sequential backbone.  Skip connection metadata is preserved in
the exported debug JSON.

Run this script from the project root::

    python examples/resnet_like.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import neuroschemax as nsx

# A simplified ResNet-like spec with Add nodes that represent skip connections.
RESNET_LIKE_SPEC = {
    "model_name": "resnet_like",
    "layers": [
        {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
        # Block 1
        {"name": "conv1_1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
        {"name": "relu1_1", "kind": "relu"},
        {"name": "conv1_2", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
        {"name": "relu1_2", "kind": "relu"},
        {"name": "add1", "kind": "add"},          # skip connection
        # Block 2
        {"name": "conv2_1", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
        {"name": "relu2_1", "kind": "relu"},
        {"name": "conv2_2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
        {"name": "relu2_2", "kind": "relu"},
        {"name": "add2", "kind": "add"},          # skip connection
        # Block 3
        {"name": "conv3_1", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
        {"name": "relu3_1", "kind": "relu"},
        {"name": "conv3_2", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
        {"name": "relu3_2", "kind": "relu"},
        {"name": "add3", "kind": "add"},          # skip connection
        {"name": "pool", "kind": "globalaveragepool"},
        {"name": "fc", "kind": "dense", "units": 1000},
    ],
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)

        # Parse and inspect
        arch = nsx.parse_model(RESNET_LIKE_SPEC)
        print(f"Model       : {arch.model_name}")
        print(f"Family      : {arch.recommended_family.value}")
        print(f"Confidence  : {arch.family_confidence.value}")
        print(f"Skip conns  : {len(arch.skip_connections)}")

        if arch.warnings:
            print("\nLimitations:")
            for w in arch.warnings:
                print(f"  Note: {w}")

        # Render to HTML
        html_path = out_dir / "resnet_like.html"
        nsx.save_network_html(html_path, RESNET_LIKE_SPEC, theme="paper")
        print(f"\nHTML saved to: {html_path}")

        # Export debug JSON (preserves skip connection info)
        debug_path = out_dir / "resnet_like_debug.json"
        nsx.save_debug_json(debug_path, RESNET_LIKE_SPEC)
        data = json.loads(debug_path.read_text())
        print(f"Debug JSON saved to: {debug_path}")
        print(f"  Layers in JSON: {len(data['layers'])}")

        print("\nNote: The HTML diagram renders the sequential backbone only.")
        print("Skip connections are captured in the debug JSON but are not")
        print("displayed visually because NN-SVG uses a linear topology.")


if __name__ == "__main__":
    main()
