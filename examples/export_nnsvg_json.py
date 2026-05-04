"""Export the NN-SVG JSON spec for a tiny model."""

from __future__ import annotations

from pathlib import Path

import neuroschemax as nsx


def main() -> None:
    spec = {
        "model_name": "tiny_mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 100]},
            {"name": "fc1", "kind": "dense", "units": 32},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2", "kind": "dense", "units": 8},
        ],
    }

    nnsvg_spec = nsx.build_nnsvg_spec(spec, theme="paper")
    out = Path("tiny_mlp.nnsvg.json")
    nsx.save_nnsvg_spec(out, nnsvg_spec)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
