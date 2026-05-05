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

## Jupyter notebooks and Google Colab

```bash
# Colab install
!pip install neuroschemax
# For inline display in notebooks:
!pip install "neuroschemax[colab]"
```

`fig._repr_html_()` is called automatically by Jupyter when a `Figure` is the
last expression in a cell — no manual `display()` needed:

```python
import neuroschemax as nsx
fig = nsx.figure(theme="paper")
fig.draw({"model_name": "mlp", "layers": [
    {"name": "input", "kind": "input", "shape": [1, 784]},
    {"name": "fc1",   "kind": "dense", "units": 128},
    {"name": "out",   "kind": "dense", "units": 10},
]})
fig  # renders inline in Jupyter; opens browser outside
```

`nsx.show()` and `fig.show()` detect the runtime environment:

- **Jupyter / JupyterLab** — renders inline via `IPython.display.HTML`.
  Full JavaScript interactivity works.
- **Google Colab** — renders inline via `IPython.display.HTML`.  Colab's
  content-security policy may limit JavaScript, so diagrams display but
  interactions may be restricted.  For the fully interactive version:

```python
fig.save_html("diagram.html")
# Files panel → right-click diagram.html → Download → open in Chrome/Firefox
```

- **Script / terminal** — saves to a temp file and opens in the browser.

`fig.to_html()` returns the self-contained HTML string for manual display:

```python
from IPython.display import HTML, display
display(HTML(fig.to_html()))
```

## Next steps

- [Python API](api.md) — all functions, the Figure object, and keyword arguments
- [CLI reference](cli.md) — all commands and flags
- [Examples](examples.md) — MLP, CNN, ResNet-like, Transformer-like, YAML spec
- [Limitations](limitations.md) — what happens with ResNets, U-Nets, Transformers
