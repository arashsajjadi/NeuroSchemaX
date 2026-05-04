"""Example: Transformer-like model.

This example shows how NeuroSchemaX handles a Transformer-like architecture that
contains attention layers.  Because NN-SVG has no native attention-block renderer,
the tool approximates the model using the FCNN family and emits a warning.
The debug JSON export still captures the full layer information.

Run this script from the project root::

    python examples/transformer_like.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import neuroschemax as nsx

TRANSFORMER_LIKE_SPEC = {
    "model_name": "transformer_like",
    "layers": [
        {"name": "input", "kind": "input", "shape": [1, 512]},
        {"name": "embed", "kind": "embedding"},
        # Encoder block 1
        {"name": "attn1", "kind": "attention"},
        {"name": "norm1_1", "kind": "layer_norm"},
        {"name": "ff1_1", "kind": "dense", "units": 2048},
        {"name": "gelu1", "kind": "gelu"},
        {"name": "ff1_2", "kind": "dense", "units": 512},
        {"name": "norm1_2", "kind": "layer_norm"},
        # Encoder block 2
        {"name": "attn2", "kind": "attention"},
        {"name": "norm2_1", "kind": "layer_norm"},
        {"name": "ff2_1", "kind": "dense", "units": 2048},
        {"name": "gelu2", "kind": "gelu"},
        {"name": "ff2_2", "kind": "dense", "units": 512},
        {"name": "norm2_2", "kind": "layer_norm"},
        # Output head
        {"name": "out", "kind": "dense", "units": 10},
    ],
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)

        # Parse and inspect
        arch = nsx.parse_model(TRANSFORMER_LIKE_SPEC)
        print(f"Model       : {arch.model_name}")
        print(f"Family      : {arch.recommended_family.value}")
        print(f"Confidence  : {arch.family_confidence.value}")

        if arch.warnings:
            print("\nLimitations / Warnings:")
            for w in arch.warnings:
                print(f"  Note: {w}")

        # Recommend view
        view_info = nsx.recommend_view(TRANSFORMER_LIKE_SPEC)
        print(f"\nReason: {view_info['reason']}")

        # Render to HTML
        html_path = out_dir / "transformer_like.html"
        nsx.save_network_html(html_path, TRANSFORMER_LIKE_SPEC, theme="paper")
        print(f"\nHTML saved to: {html_path}")

        # Export debug JSON
        debug_path = out_dir / "transformer_like_debug.json"
        nsx.save_debug_json(debug_path, TRANSFORMER_LIKE_SPEC)
        data = json.loads(debug_path.read_text())
        print(f"Debug JSON saved to: {debug_path}")
        print(f"  Layers in JSON: {len(data['layers'])}")

        print("\nNote: Attention layers are approximated as dense/FCNN nodes.")
        print("There is no native NN-SVG rendering for Transformer architectures.")
        print("Consider the debug JSON for full structural analysis.")


if __name__ == "__main__":
    main()
