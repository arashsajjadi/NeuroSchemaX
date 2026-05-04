# NeuroSchemaX

**Neural network architecture visualization and export, powered by [NN-SVG][nnsvg].**

NeuroSchemaX parses neural-network models (ONNX, PyTorch, TensorFlow, or
hand-written JSON/YAML specs), normalises them into a semantic representation,
and renders them using the real NN-SVG JavaScript engine — producing standalone
HTML and clean SVG diagrams suitable for papers, theses, READMEs, and
documentation.

[nnsvg]: https://github.com/alexlenail/NN-SVG

---

## What it is

NeuroSchemaX is a Python toolkit that turns neural-network model files into
publication-quality architecture diagrams.  It does three things:

1. **Parse** — reads ONNX, PyTorch, Keras, JSON, or YAML and understands
   the layer structure.
2. **Analyse** — detects layer types, skip connections, and block groupings,
   then recommends the best diagram style automatically.
3. **Render** — produces standalone HTML (self-contained, works offline) or
   SVG (via headless Chromium), plus several JSON export formats.

## Why it exists

NN-SVG is the standard tool used in hundreds of deep-learning papers, but it
requires manual data entry through a web UI.  NeuroSchemaX automates that:
point it at a model file and get a diagram in one command.

## Who should use it

- Researchers who need architecture diagrams for papers or theses
- Engineers documenting models in READMEs
- Anyone who wants consistent, reproducible diagrams from real model files

---

## Good defaults and output quality

NeuroSchemaX is designed to produce useful, readable diagrams without manual
tuning.  Point it at a model file and get a clean result:

```python
import neuroschemax as nsx

nsx.draw("model.onnx")
nsx.savefig("architecture.html")
```

```bash
neuroschemax draw model.onnx
```

What you get without any options:

- **Automatic style selection** — NeuroSchemaX inspects the model and picks
  the best NN-SVG diagram family (FCNN for MLPs, LeNet for small CNNs,
  AlexNet for deeper CNNs).
- **Sensible canvas size** — the canvas expands automatically so layers are
  never crowded, even for deep networks.
- **Readable labels** — layer names are shown by default, truncated cleanly
  when space is tight.
- **Clean spacing** — layers are evenly distributed with comfortable margins.
- **Professional themes** — `paper` (clean, minimal), `thesis` (print-ready),
  `readme` (docs-friendly), `debug` (verbose, all annotations).
- **Compact mode** — pass `--compact` or `compact=True` for very deep models
  to reduce node size and spacing.
- **Honest warnings** — when a model cannot be rendered exactly (ResNet skip
  connections, Transformer attention blocks, U-Net branches), an amber badge
  appears in the HTML explaining what was approximated and where the full
  graph is preserved.
- **Offline HTML** — generated HTML files are fully self-contained.  All
  JavaScript is embedded.  No internet connection or server is needed.
- **Clean SVG export** — when Playwright/Chromium is installed, SVG output
  is a proper vector file suitable for papers and presentations.
- **Debug and metadata exports** — for complex models, `export-debug-json`
  preserves the complete layer graph, skip connections, and warnings.

### Example outputs

| Model | Diagram family | Fidelity |
|---|---|---|
| `examples/tiny_mlp_to_html.py` | FCNN | Exact |
| `examples/tiny_cnn_to_svg.py` | LeNet | Exact |
| `examples/resnet_like.py` | AlexNet (approximate) | Sequential backbone; skip links in debug JSON |
| `examples/transformer_like.py` | FCNN (approximate) | Attention blocks collapsed; full layers in debug JSON |

---

## Installation

Base (HTML output, ONNX and YAML input):

```bash
pip install neuroschemax
```

With SVG export (requires headless Chromium):

```bash
pip install "neuroschemax[svg]"
playwright install chromium
```

With PyTorch model input:

```bash
pip install "neuroschemax[torch]"
```

With TensorFlow/Keras model input:

```bash
pip install "neuroschemax[tf]"
```

Development (tests + linter):

```bash
pip install "neuroschemax[dev]"
```

Everything at once:

```bash
pip install "neuroschemax[all]"
playwright install chromium
```

---

## Quickstart (30 seconds)

### Python — simplified API

```python
import neuroschemax as nsx

nsx.draw("model.onnx")
nsx.savefig("diagram.html")     # format inferred from extension
```

Open `diagram.html` in any browser.

### Python — figure object

```python
fig = nsx.figure(width=1400, height=700, theme="paper")
fig.draw("model.onnx")
fig.savefig("diagram.svg")
fig.export_debug_json("debug.json")
```

### Python — explicit functional API

```python
import neuroschemax as nsx

arch = nsx.parse_model("model.onnx")
print(nsx.summarize_model(arch))

nsx.save_network_html("model.html", arch, theme="paper")
nsx.save_network_svg("model.svg",  arch, theme="paper")   # needs Playwright

spec = nsx.build_nnsvg_spec(arch, style="lenet", width=1400, height=600)
nsx.save_nnsvg_spec("model.nnsvg.json", spec)
```

### CLI

```bash
# User-friendly draw command (output defaults to <model>.html)
neuroschemax draw model.onnx
neuroschemax draw model.onnx -o diagram.html --theme paper

# Full control
neuroschemax render model.onnx -o diagram.html --theme thesis --width 1600
neuroschemax render model.onnx -o diagram.svg  --theme paper

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

## Python API

### Simplified stateful API

Stores the last parsed architecture internally so you don't repeat the source.

```python
import neuroschemax as nsx

nsx.draw("model.onnx")            # parse and stash
nsx.savefig("diagram.html")       # use stashed arch
nsx.save_html("out.html")         # also HTML
nsx.show()                        # open in browser

# Or pass source directly (no stash needed)
nsx.save_html("out.html", "model.onnx")
```

### Figure object API

```python
fig = nsx.figure(width=1400, height=700, theme="paper")

# Matplotlib-style sizing: figsize=(inches_wide, inches_tall), dpi=pixels_per_inch
# width = round(figsize[0] * dpi), height = round(figsize[1] * dpi)
fig = nsx.figure(figsize=(12, 6), dpi=120, theme="paper")
fig.draw("model.onnx")
fig.savefig("diagram.html")

fig = nsx.figure(width=1400, height=700, theme="paper")
fig.draw("model.onnx")            # returns self for chaining
fig.savefig("diagram.html")
fig.save_html("diagram.html")
fig.save_svg("diagram.svg")
fig.show()
fig.export_debug_json("debug.json")
fig.export_paper_json("paper.json")
fig.export_nnsvg_json("spec.json")

# Chaining
nsx.figure().draw("model.onnx").save_html("out.html")
```

### Explicit functional API

```python
nsx.parse_model(source)               # SemanticArchitecture
nsx.parse_graph(source)               # GraphIR
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

| Argument | Type | Description |
|---|---|---|
| `theme` | `str` | `"paper"`, `"thesis"`, `"debug"`, `"readme"` |
| `style` | `str` | Force `"fcnn"`, `"lenet"`, or `"alexnet"` |
| `width` | `int` | Canvas width in pixels |
| `height` | `int` | Canvas height in pixels |
| `title` | `str` | Diagram title |
| `show_labels` | `bool` | Show layer name labels |
| `show_shapes` | `bool` | Show shape annotations |
| `compact` | `bool` | Use compact layout |

---

## CLI examples with all commands

```bash
# Minimal
neuroschemax draw model.onnx

# Full render with all options
neuroschemax render model.onnx -o out.html \
    --theme paper --style lenet \
    --width 1400 --height 700 \
    --title "My CNN" \
    --no-labels --no-shapes --compact

# From a YAML spec
neuroschemax draw spec.yaml -o diagram.html

# From a JSON spec
neuroschemax draw spec.json -o diagram.html

# Environment check
neuroschemax doctor
```

---

## Local folder usage

Working with a local model file:

```bash
# Current directory
neuroschemax draw ./my_model.onnx

# Another directory
neuroschemax draw /path/to/models/resnet.onnx -o /path/to/output/resnet.html
```

Using a YAML spec (no model file needed):

```yaml
# model.yaml
model_name: my_mlp
layers:
  - name: input
    kind: input
    shape: [1, 784]
  - name: fc1
    kind: dense
    units: 256
  - name: out
    kind: dense
    units: 10
```

```bash
neuroschemax draw model.yaml
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

---

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

## Themes and styles

Themes set the overall look; styles set the NN-SVG diagram family.

```python
# Themes
nsx.save_network_html("out.html", model, theme="paper")    # clean, minimal
nsx.save_network_html("out.html", model, theme="thesis")   # print-ready
nsx.save_network_html("out.html", model, theme="debug")    # verbose, all labels
nsx.save_network_html("out.html", model, theme="readme")   # dark background

# Styles (override auto-detection)
nsx.save_network_html("out.html", model, style="fcnn")     # MLP / dense
nsx.save_network_html("out.html", model, style="lenet")    # small CNN
nsx.save_network_html("out.html", model, style="alexnet")  # deep CNN
```

---

## How architecture recommendation works

`recommend_view()` analyses the parsed architecture and returns the best
NN-SVG family with a `confidence` level and a human-readable `reason`:

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

Rules (in priority order):
- Attention layers detected → **block-level LeNet view** (LOW confidence, `is_approximate=True`, warning)
- Recurrent layers (LSTM/GRU) detected → **block-level LeNet view** (LOW confidence, warning)
- No conv layers, at least one dense layer → **FCNN** (HIGH confidence)
- 1–3 conv layers → **LeNet** (HIGH; MEDIUM with skip/merge warning if Add/Concat present)
- 4+ conv layers → **AlexNet** (HIGH; MEDIUM if Add/Concat skip connections present)

`is_approximate` is `True` whenever `confidence` is not `"high"` or there are warnings.
Skip connections are detected from both ONNX graph edges and explicit Add/Concat layers
in manual specs.

Note: the `family` field in `recommend_view()` reflects the semantic architecture
family (e.g. `"fcnn"` for Transformers), not the rendering family used in the diagram.
Transformers are rendered as `"lenet"` (rect blocks) regardless of the semantic family.

---

## Limitations

NN-SVG supports three diagram families: FCNN, LeNet, and AlexNet.  All are
sequential left-to-right layouts.  NeuroSchemaX does not claim to render
arbitrary DAGs — models that fall outside these three families are shown as
an honest approximation of the sequential backbone, with a clear on-diagram
warning.

| Architecture | How it appears | Visual fidelity | What is preserved |
|---|---|---|---|
| MLP / dense network | FCNN neuron columns | **Exact** | — |
| Small CNN (≤ 3 convs) | LeNet feature maps | **Exact** | — |
| VGG-style deep CNN | AlexNet feature maps | **Exact** | — |
| ResNet / residual blocks | AlexNet backbone | **Approximate** | Skip links in debug JSON |
| U-Net / encoder-decoder | AlexNet backbone | **Approximate** | Decoder branches in debug JSON |
| Transformer / attention | Block-level sequence* | **Approximate** | All layers in debug JSON |
| LSTM / GRU / RNN | Block-level sequence* | **Approximate** | All layers in debug JSON |
| Object-detection head | AlexNet backbone | **Approximate** | Detection branches in debug JSON |

*Transformer and recurrent architectures are shown as a sequence of labeled
rectangular blocks (Input → Embedding → [Attention] → FeedFwd → … → Classifier),
rendered via the LeNet rectangle renderer.  This is a block-level approximation
of the computation sequence — it is **not** an exact Transformer diagram.
NN-SVG has no native Transformer, U-Net, or ResNet renderer.  The label inside
each block identifies the operation type; residual connections and attention
relationships are not drawn.

Every approximate rendering:
- shows an amber warning badge in the HTML explaining what was simplified
- reports `confidence: "medium"` or `"low"` and `is_approximate: true` in `recommend_view()`
- preserves the complete layer and edge information in `export-debug-json`

Use `export-debug-json` when you need the exact graph structure.

---

## Troubleshooting

Run `neuroschemax doctor` first.  It prints a pass/fail summary.

**SVG export fails:** Install Playwright and Chromium:

```bash
pip install playwright
playwright install chromium
```

**Missing assets:** Reinstall:

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
Yes. `nsx.show()` and `fig.show()` detect whether they are running inside a
Jupyter or IPython kernel automatically.  Inside a notebook the diagram is
displayed **inline** using `IPython.display`.  Outside a notebook the same
call opens the rendered HTML in your default web browser.  IPython is not a
required dependency — it is imported lazily and the browser fallback is used
if it is unavailable.

**Q: Does it support ONNX opset X?**
It reads the graph topology and layer attributes regardless of opset version.
Unsupported op types are normalised to `LayerKind.UNKNOWN` and included with
LOW confidence.

**Q: Can I customise the diagram further?**
You can override any rendering option via kwargs.  For pixel-level control,
export the NN-SVG JSON spec and load it in the [NN-SVG web app][nnsvg].

**Q: Why is confidence MEDIUM/LOW?**
A model has features that don't map perfectly to the chosen NN-SVG family.
The diagram is still generated; read the `warnings` for specifics.

---

## Development setup

```bash
git clone <repo-url>
cd NeuroSchemaX
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Publishing checklist

- [ ] Bump version in `src/neuroschemax/version.py`
- [ ] Update `CHANGELOG.md`
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linter errors: `ruff check src/ tests/`
- [ ] `python -m build` succeeds
- [ ] Test on TestPyPI, then publish

See [docs/packaging.md](docs/packaging.md).

---

## License

MIT — see [LICENSE](LICENSE).

NN-SVG is by Alex Lenail and also MIT-licensed.
NeuroSchemaX integrates it as an embedded rendering engine.
