"""Ask NeuroSchemaX which NN-SVG family best fits a given model."""

from __future__ import annotations

import json

import neuroschemax as nsx


MODELS = {
    "mlp": {
        "model_name": "mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 100]},
            {"name": "fc1", "kind": "dense", "units": 64},
            {"name": "fc2", "kind": "dense", "units": 32},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    },
    "small_cnn": {
        "model_name": "small_cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "fc", "kind": "dense", "units": 10},
        ],
    },
    "deep_cnn": {
        "model_name": "deep_cnn",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [
                {"name": f"conv{i}", "kind": "conv",
                 "out_channels": 64 * (i + 1), "kernel_size": [3, 3]}
                for i in range(6)
            ]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    },
}


def main() -> None:
    for name, spec in MODELS.items():
        info = nsx.recommend_view(spec)
        print(f"{name:>12}  →  {json.dumps(info)}")


if __name__ == "__main__":
    main()
