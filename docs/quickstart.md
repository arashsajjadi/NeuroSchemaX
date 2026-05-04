# Quickstart

Get a useful diagram from a model file in under 30 seconds.

## Install

```bash
pip install neuroschemax
```

That is all you need for HTML output, ONNX input, and manual JSON/YAML specs.

## One command

```bash
neuroschemax draw model.onnx
# Writes model.html to the current directory.
# Open in any browser — no internet connection required.
```

Or with an explicit output path:

```bash
neuroschemax draw model.onnx -o architecture.html
neuroschemax draw model.onnx -o architecture.html --theme paper
```

## One Python snippet

```python
import neuroschemax as nsx

nsx.draw("model.onnx")
nsx.savefig("architecture.html")  # format inferred from .html extension
```

Open `architecture.html` in Chrome, Firefox, or Edge.

## HTML vs SVG

**HTML export** (default) works offline with no additional dependencies.
The generated file is fully self-contained — all rendering scripts are
embedded inline.

**SVG export** produces a vector file suitable for papers and presentations,
but requires a headless browser:

```bash
pip install "neuroschemax[svg]"
playwright install chromium
```

```python
nsx.draw("model.onnx")
nsx.savefig("architecture.svg")  # triggers SVG export via Chromium
```

If Playwright or Chromium is not installed, SVG export raises a clear error
with the exact install commands needed.  HTML export is never affected.

## What you get without options

- Automatic diagram style (FCNN, LeNet, or AlexNet) chosen from the model structure
- Canvas sized automatically — layers are never crowded
- Labels and shape annotations shown by default
- Paper theme applied by default — clean and minimal

## Jupyter notebooks

`nsx.show()` and `fig.show()` detect the runtime environment automatically:

- **Inside Jupyter / IPython** — the diagram is displayed **inline** in the
  cell output using `IPython.display`.
- **Outside a notebook** — the rendered HTML is saved to a temporary file
  and opened in the default web browser.

IPython is not a required dependency.  If it is not installed, the browser
fallback is used silently.

```python
# Works in both notebooks and scripts without any change
nsx.draw("model.onnx")
nsx.show()
```

## Next steps

- [Python API](api.md) — all functions, the Figure object, and keyword arguments
- [CLI reference](cli.md) — all commands and flags
- [Examples](examples.md) — MLP, CNN, ResNet-like, Transformer-like, YAML spec
- [Limitations](limitations.md) — what happens with ResNets, U-Nets, Transformers
