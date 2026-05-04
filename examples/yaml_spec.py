"""Example: defining a model with a YAML spec file.

NeuroSchemaX accepts YAML files as an alternative to JSON or Python dicts.
This example creates a temporary YAML file, parses it, and renders it to HTML.

Run this script from the project root::

    python examples/yaml_spec.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import neuroschemax as nsx

YAML_CONTENT = """\
model_name: yaml_mlp
layers:
  - name: input
    kind: input
    shape: [1, 784]
  - name: fc1
    kind: dense
    units: 256
  - name: relu1
    kind: relu
  - name: fc2
    kind: dense
    units: 128
  - name: relu2
    kind: relu
  - name: out
    kind: dense
    units: 10
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)

        # Write the YAML spec to a temp file
        yaml_path = out_dir / "model.yaml"
        yaml_path.write_text(YAML_CONTENT, encoding="utf-8")
        print(f"Wrote YAML spec to: {yaml_path}")

        # Parse directly from the YAML file path
        arch = nsx.parse_model(str(yaml_path))
        print(f"Model name  : {arch.model_name}")
        print(f"Layer count : {arch.layer_count}")
        print(f"Family      : {arch.recommended_family.value}")

        # Render to HTML
        html_path = out_dir / "yaml_mlp.html"
        nsx.save_network_html(html_path, str(yaml_path), theme="paper")
        print(f"HTML saved to: {html_path}")
        print("Done.")


if __name__ == "__main__":
    main()
