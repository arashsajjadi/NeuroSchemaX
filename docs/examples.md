# Examples

All example scripts are in the `examples/` directory and can be run directly.

## Example outputs at a glance

| Script | Diagram family | Rendering fidelity |
|---|---|---|
| `tiny_mlp_to_html.py` | FCNN | **Exact** — all layers rendered faithfully |
| `tiny_cnn_to_svg.py` | LeNet | **Exact** — requires `playwright install chromium` |
| `resnet_like.py` | AlexNet | **Approximate** — backbone only; skip links in debug JSON |
| `transformer_like.py` | FCNN | **Approximate** — attention collapsed; layers in debug JSON |
| `yaml_spec.py` | FCNN | **Exact** — manual YAML spec |
| `manual_spec_to_svg.py` | LeNet | **Exact** — requires Playwright |
| `recommend_view.py` | — | Shows `recommend_view()` output |
| `export_nnsvg_json.py` | — | Exports NN-SVG spec JSON |
| `summarize_model.py` | — | Prints text and Markdown summaries |

"Exact" means the diagram represents the model faithfully.
"Approximate" means the diagram shows the sequential backbone; complex
topology (skip connections, branches, attention) is noted in a warning badge
and preserved in the debug JSON export.

---

## Tiny MLP to HTML

```bash
python examples/tiny_mlp_to_html.py
```

Parses a small dense network from a Python dict and saves `tiny_mlp.html`.
No dependencies beyond the base install.  Open the file in any browser.

## Tiny CNN to SVG

```bash
python examples/tiny_cnn_to_svg.py
```

Renders a small CNN to SVG via headless Chromium.
Requires `pip install "neuroschemax[svg]"` and `playwright install chromium`.

## ResNet-like model with skip connections

```bash
python examples/resnet_like.py
```

Demonstrates honest approximate rendering for skip-connection architectures.
The sequential conv backbone is drawn as an AlexNet diagram.  The Add
(residual merge) nodes are excluded from the visual diagram but preserved
in the debug JSON export.  An amber warning badge in the HTML explains the
approximation.

## Transformer-like model

```bash
python examples/transformer_like.py
```

Shows how attention layers and feed-forward blocks are handled.  NeuroSchemaX
does not claim to draw attention graphs — the dense layers are shown as FCNN
columns and attention blocks are documented with a confidence warning.  Run
this example to see `recommend_view()` return `"confidence": "low"` and
`"is_approximate": true`.

## YAML spec

```bash
python examples/yaml_spec.py
```

Creates a temporary YAML architecture spec file and renders it to HTML.
Demonstrates the manual-spec format for users who do not have a model file.

## Manual spec to SVG

```bash
python examples/manual_spec_to_svg.py
```

Builds a model from a Python dict and exports to SVG (requires Playwright).

## Recommend view

```bash
python examples/recommend_view.py
```

Prints the `recommend_view()` output — family, confidence, `is_approximate`,
reason, and warnings — for a sample model.

## Export NN-SVG JSON

```bash
python examples/export_nnsvg_json.py
```

Exports the NN-SVG spec JSON, which can be loaded directly into the
[NN-SVG web application](https://github.com/alexlenail/NN-SVG) for further
manual customisation.

## Summarize model

```bash
python examples/summarize_model.py
```

Prints a plain-text layer table and a Markdown summary for a sample model.
