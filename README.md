# NeuroSchemaX

**Neural network architecture visualization and export, powered by [NN-SVG][nnsvg].**

NeuroSchemaX parses neural-network models (ONNX, PyTorch, TensorFlow, or
hand-written JSON/YAML specs), normalises them into a semantic representation,
and renders them using the NN-SVG JavaScript engine — producing standalone
HTML and SVG diagrams suitable for papers, theses, READMEs, and documentation.

**Best-supported targets:** MLP/FCNN and sequential CNN-style architectures
(LeNet-style, AlexNet/VGG-style).  ResNet, U-Net, Transformer, and other
complex graph structures are rendered as honest approximate summaries; exact
topology cannot be drawn for architectures that fall outside the three
sequential NN-SVG families.

[nnsvg]: https://github.com/alexlenail/NN-SVG

---

## What it does

1. **Parse** — reads ONNX, PyTorch, Keras, JSON, or YAML and understands the layer structure.
2. **Analyse** — detects layer types, skip connections, and block groupings, then recommends the best diagram style.
3. **Render** — produces standalone offline HTML or SVG (via headless Chromium), plus JSON export formats.

---

## Installation

Install from PyPI:

```bash
pip install neuroschemax
```

Install from GitHub:

```bash
pip install git+https://github.com/arashsajjadi/NeuroSchemaX.git
```

Optional extras:

```bash
pip install "neuroschemax[svg]"    # SVG export (requires headless Chromium)
playwright install chromium

pip install "neuroschemax[torch]"  # PyTorch model input
pip install "neuroschemax[tf]"     # TensorFlow / Keras model input
pip install "neuroschemax[dev]"    # tests and linter
pip install "neuroschemax[all]"    # everything
```

---

## Quickstart

```python
import neuroschemax as nsx

nsx.draw("model.onnx")
nsx.savefig("architecture.html")
```

```bash
neuroschemax draw model.onnx
```

Open the generated HTML in any browser — no internet connection required.

---

## Python API

### Simplified stateful API

```python
import neuroschemax as nsx

nsx.draw("model.onnx")        # parse and stash
nsx.savefig("diagram.html")   # use stashed arch — format inferred from extension
nsx.save_html("out.html")     # also HTML
nsx.show()                    # open in browser (inline in Jupyter)

# Or pass source directly
nsx.save_html("out.html", "model.onnx")
```

### Figure object API

```python
fig = nsx.figure(width=1400, height=700, theme="paper")
fig.draw("model.onnx")
fig.savefig("diagram.html")
fig.save_html("diagram.html")
fig.save_svg("diagram.svg")       # needs Playwright
fig.show()
fig.export_debug_json("debug.json")
fig.export_paper_json("paper.json")
fig.export_nnsvg_json("spec.json")

# Matplotlib-style sizing
fig = nsx.figure(figsize=(12, 6), dpi=120, theme="paper")

# Chaining
nsx.figure().draw("model.onnx").save_html("out.html")
```

### Explicit functional API

```python
nsx.parse_model(source)               # SemanticArchitecture
nsx.summarize_model(source)           # str
nsx.recommend_view(source)            # dict: family/confidence/is_approximate/reason/warnings

nsx.build_nnsvg_spec(source, **kw)    # NNSVGSpec
nsx.render_network_html(source)       # HTML string
nsx.render_network_svg(source)        # SVG string (needs Playwright)
nsx.save_network_html(path, source)   # Path
nsx.save_network_svg(path, source)    # Path (needs Playwright)

nsx.export_paper_json(source)         # JSON string
nsx.export_debug_json(source)         # JSON string
nsx.save_paper_json(path, source)     # Path
nsx.save_debug_json(path, source)     # Path
nsx.save_nnsvg_spec(path, spec)       # Path

nsx.doctor()                          # dict with status/version/assets/deps/messages
```

### Rendering keyword arguments

These are accepted by every rendering function and `nsx.figure()`:

| Argument | Type | Default | Description |
|---|---|---|---|
| `theme` | `str` | `"paper"` | `"paper"`, `"thesis"`, `"debug"`, `"readme"` |
| `style` | `str` | auto | Force `"fcnn"`, `"lenet"`, or `"alexnet"` |
| `figsize` | `(float, float)` | — | Matplotlib-style: `width = round(w * dpi)`, `height = round(h * dpi)` |
| `dpi` | `float` | `100` | Pixels per inch used with `figsize` |
| `width` | `int` | `1200` | Canvas width in pixels (overrides `figsize`) |
| `height` | `int` | `700` | Canvas height in pixels (overrides `figsize`) |
| `title` | `str` | model name | Diagram title shown above the diagram |
| `show_labels` | `bool` | `True` | Show layer labels |
| `show_shapes` | `bool` | `True` | Show shape dimensions in labels |
| `compact` | `bool` | `False` | Use compact layout (tighter per-layer budget) |
| `label_mode` | `str` | `"auto"` | `"auto"`, `"name"`, `"compact"`, `"shape"`, `"full"` |
| `detail_level` | `str` | `"auto"` | `"auto"`, `"summary"`, `"full"` |
| `show_activations` | `bool` | `True` | Fuse activation names into preceding layer labels |
| `transformer_mode` | `str` | `"block_summary"` | `"block_summary"` or `"unsupported"` |
| `approximate_mode` | `str` | `"warn"` | `"warn"`, `"error"`, or `"allow"` |

**`label_mode`:**
- `auto` — compact labels for small models; name-only for large models
- `name` — layer name only (shortest; never overlaps)
- `compact` — operation type + key parameters: `Conv 64 k3`, `MaxP k2 s2`, `Dense 128`, `GAP`
- `shape` — shape only, no name
- `full` — layer name + complete shape string; may be wide for large models

Activations following a layer are fused into the label when `show_activations=True`:
`Conv 64 k3 +ReLU`, `Dense 128 +GeLU`.  BatchNorm, LayerNorm, and Dropout appear as
inline badges: `+BN`, `+LN`, `+Drop 0.5`.

**`detail_level`:**
- `auto` — all layers for small models; grouped blocks for models with more than 12 spec layers
- `summary` — groups conv/pool sequences into named blocks with metadata, e.g. `Block 2 / 4×Conv k3, 128ch / Pool ↓2`
- `full` — every individual layer shown; intended for inspection and debugging, not screenshots

**`transformer_mode`:**
- `block_summary` — approximate conceptual block sequence: `Tokens/Input → Embedding → [MH-Attn] / Add & Norm → [FFN] / Add & Norm → [Head]`.  This is **not** exact Transformer rendering.  Q/K/V projections, individual heads, exact residual paths, and tensor flow are not drawn.
- `unsupported` — renders a structured HTML diagnostic card instead of a diagram, listing detected components and suggesting `block_summary` or debug JSON

**`approximate_mode`:**
- `warn` — amber warning badge shown in HTML (default)
- `error` — raises `RenderError` before rendering an approximate diagram
- `allow` — suppresses warning badges

---

## CLI

```bash
# Draw (output defaults to <model>.html)
neuroschemax draw model.onnx
neuroschemax draw model.onnx -o diagram.html --theme paper

# Render with full options
neuroschemax render model.onnx -o diagram.html --theme thesis --width 1600
neuroschemax render model.onnx -o diagram.svg

# Inspect and summarise
neuroschemax inspect        model.onnx
neuroschemax summarize      model.onnx
neuroschemax summarize      model.onnx --format markdown
neuroschemax recommend-view model.onnx

# JSON exports
neuroschemax export-paper-json model.onnx -o model.paper.json
neuroschemax export-debug-json model.onnx -o model.debug.json
neuroschemax export-nnsvg      model.onnx -o model.nnsvg.json

# Environment diagnostics
neuroschemax doctor
```

---

## Supported inputs

| Source | Notes |
|---|---|
| `.onnx` file | Standard ONNX format |
| `.json` file | Manual spec JSON |
| `.yaml` / `.yml` file | Manual spec YAML |
| Python `dict` | Manual spec as a Python dict |
| `torch.nn.Module` | Requires `pip install torch` |
| `tf.keras.Model` | Requires `pip install tensorflow` |
| `onnx.ModelProto` | Pre-loaded ONNX object |

## Supported outputs

| Format | API | CLI |
|---|---|---|
| Standalone HTML | `save_network_html` | `render -o .html` |
| SVG (via Playwright) | `save_network_svg` | `render -o .svg` |
| Paper JSON | `save_paper_json` | `export-paper-json` |
| Debug JSON | `save_debug_json` | `export-debug-json` |
| NN-SVG JSON spec | `save_nnsvg_spec` | `export-nnsvg` |
| Text summary | `summarize_model` | `summarize` |
| Markdown summary | `summarize_model(..., "markdown")` | `summarize --format markdown` |

---

## Diagram families and fidelity

NN-SVG supports three sequential diagram families.  NeuroSchemaX selects the
best fit automatically based on the model structure.

| Architecture | Rendered as | Fidelity | What is preserved |
|---|---|---|---|
| MLP / dense network | FCNN neuron columns | **Exact** | — |
| Small CNN (≤ 3 convs) | LeNet feature maps | **Exact** | — |
| VGG-style deep CNN | AlexNet feature maps | **Exact** | — |
| ResNet / residual blocks | Block summary (skip collapsed) | **Approximate** | Skip links in debug JSON |
| U-Net / encoder-decoder | Block summary (concat collapsed) | **Approximate** | Decoder branches in debug JSON |
| Transformer / attention | Conceptual block sequence | **Approximate** | All layers in debug JSON |
| LSTM / GRU / RNN | Block sequence | **Approximate** | All layers in debug JSON |
| Arbitrary DAG | Not supported | — | Full graph in debug JSON |

NeuroSchemaX does not claim to render arbitrary DAGs, exact ResNet skip edges,
exact U-Net encoder-decoder skip topology, or exact Transformer attention flow.
Models that fall outside the three sequential families are shown as honest
approximate summaries with clear on-diagram markers (`+skip collapsed`,
`concat collapsed`) and amber warning badges in the HTML.

ResNet summary: `Stem → Residual Block N (n×Conv k3, Cch, +skip collapsed) → Head / Classifier`

U-Net summary: `Encoder → Bottleneck → Decoder (concat collapsed) → Segmentation Head`

Transformer block summary: `Tokens/Input → Embedding → [MH-Attn] / Add & Norm → [FFN] / Add & Norm → [Head]`

The complete layer graph (every Add/Concat/Upsample with attributes) is
preserved in `export-debug-json` regardless of what is shown visually.

---

## How architecture recommendation works

```python
info = nsx.recommend_view("model.onnx")
# {
#   "family": "alexnet",
#   "confidence": "high",
#   "is_approximate": False,
#   "reason": "Deep CNN with 8 convolutional layers, mapped to AlexNet",
#   "warnings": [],
#   "complexity_hint": "sequential"
# }
```

Selection rules (in priority order):
- Attention layers → **block-level LeNet view** (LOW confidence, `is_approximate=True`)
- Recurrent layers (LSTM/GRU) → **block-level LeNet view** (LOW confidence)
- No convolutions, at least one dense layer → **FCNN** (HIGH confidence)
- 1–3 conv layers → **LeNet** (HIGH; MEDIUM with skip/merge warning)
- 4+ conv layers → **AlexNet** (HIGH; MEDIUM if skip connections present)

---

## Troubleshooting

Run `neuroschemax doctor` first.

**SVG export fails:**

```bash
pip install playwright
playwright install chromium
```

**Missing assets:**

```bash
pip install --force-reinstall neuroschemax
```

**HTML is blank:** Open in a modern browser (Chrome/Firefox/Edge).  Check the
browser console for JS errors.  Try `--theme debug` for extra output.

See [docs/troubleshooting.md](docs/troubleshooting.md) for more.

---

## FAQ

**Q: Does it work offline?**
Yes. Generated HTML files are fully self-contained — all JS is embedded.

**Q: Can I use it in a Jupyter notebook?**
Yes. `nsx.show()` and `fig.show()` detect the runtime automatically.  Inside
Jupyter the diagram is displayed inline; outside, it opens in the browser.

**Q: Does it support ONNX opset X?**
It reads graph topology and layer attributes regardless of opset version.
Unsupported op types are normalised to `LayerKind.UNKNOWN`.

**Q: Can I customise the diagram further?**
Override any rendering option via kwargs.  For pixel-level control, export the
NN-SVG JSON spec and load it in the [NN-SVG web app][nnsvg].

**Q: Why is confidence MEDIUM/LOW?**
The model has features that don't map perfectly to the chosen NN-SVG family.
The diagram is still generated; read the `warnings` for specifics.

**Q: Why does the diagram show fewer layers than my model has?**
Large models are grouped into conv/pool blocks by default (`detail_level="auto"`).
Pass `detail_level="full"` to see every individual layer.  Note that full mode
is intended for inspection; for screenshots, `detail_level="summary"` or the
default produces more readable output.

---

## Development

```bash
git clone https://github.com/arashsajjadi/NeuroSchemaX.git
cd NeuroSchemaX
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Author

**Arash Sajjadi** — maintainer and author.

GitHub: [arashsajjadi/NeuroSchemaX](https://github.com/arashsajjadi/NeuroSchemaX)

---

## License

MIT — see [LICENSE](LICENSE).

NN-SVG is by Alex Lenail and also MIT-licensed.
NeuroSchemaX integrates it as an embedded rendering engine.
