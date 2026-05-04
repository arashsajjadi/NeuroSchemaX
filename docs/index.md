# NeuroSchemaX Documentation

**NeuroSchemaX** turns neural-network model files into clean, readable
architecture diagrams — with good defaults, no manual tuning required.

Point it at an ONNX file, a PyTorch module, a Keras model, or a hand-written
JSON/YAML spec and get a standalone HTML diagram in one command:

```bash
neuroschemax draw model.onnx
```

```python
import neuroschemax as nsx

nsx.draw("model.onnx")
nsx.savefig("architecture.html")
```

The generated HTML is fully self-contained and works offline.  For vector
SVG output, install Playwright/Chromium (`playwright install chromium`).

NeuroSchemaX uses [NN-SVG][nnsvg] as the rendering engine, which supports
three diagram families: FCNN (MLP), LeNet (small CNN), and AlexNet (deep CNN).
The style is selected automatically based on the model structure.  For
architectures that do not fit a single sequential layout (ResNets, U-Nets,
Transformers), the sequential backbone is rendered honestly with a clear
approximation warning and complete metadata preserved in the debug-JSON export.

[nnsvg]: https://github.com/alexlenail/NN-SVG

## Where to start

- **[Quickstart](quickstart.md)** — install and get your first diagram in 30 seconds.
- **[Examples](examples.md)** — runnable scripts for MLP, CNN, ResNet-like, Transformer-like, and YAML specs.
- **[Python API](api.md)** — all public functions and the Figure object API.
- **[CLI reference](cli.md)** — all commands and flags.
- **[Limitations](limitations.md)** — what NN-SVG can and cannot render, and how approximations work.
- **[Model support](model-support.md)** — ONNX, PyTorch, TensorFlow, and manual specs.
- **[Troubleshooting](troubleshooting.md)** — SVG export, missing assets, blank HTML.

## Supported inputs

| Format | Notes |
|---|---|
| ONNX (`.onnx`) | Best-tested; richest shape and edge information |
| Manual JSON/YAML | Explicit control over every layer |
| PyTorch `nn.Module` | Module-tree walk; use ONNX export for richer graphs |
| TensorFlow / Keras | Layer-list walk |

## Supported outputs

| Format | Notes |
|---|---|
| Standalone HTML | Offline, self-contained, all JS embedded |
| SVG | Requires `playwright install chromium` |
| Paper JSON | Semantic layer view for publications |
| Debug JSON | Full graph, skip connections, warnings, metadata |
| NN-SVG JSON spec | Reproducible spec for the NN-SVG web app |
| Text / Markdown summary | Human-readable layer table |
