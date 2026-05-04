# Python API Reference

## Simplified stateful API

The stateful API stores the last parsed architecture in a module-level dict, so
you do not need to pass the source to every call.

```python
import neuroschemax as nsx

nsx.draw("model.onnx")          # parse and stash
nsx.savefig("diagram.html")     # save using stashed arch
nsx.save_html("out.html")       # save HTML using stashed arch
nsx.show()                      # open in default browser
```

### `nsx.draw(source, **kwargs)`

Parse *source* and stash the result.  Returns the `SemanticArchitecture`.

### `nsx.savefig(path, **kwargs)`

Save the stashed architecture to *path*.  Format inferred from extension
(`.html` or `.svg`).  Raises `RuntimeError` if `draw()` has not been called.

### `nsx.show(source=None, **kwargs)`

Open the rendered HTML in the default web browser.  If *source* is `None` the
stashed architecture is used.

### `nsx.save_html(path, source=None, **kwargs)`

Save HTML to *path*.  If *source* is provided it is parsed first; otherwise the
stashed architecture is used.

### `nsx.save_svg(path, source=None, **kwargs)`

Save SVG to *path*.  Requires Playwright and Chromium.

### `nsx.doctor()`

Return a structured dict describing the environment: installed dependencies,
available assets, and actionable messages.

---

## Figure object API

Use `Figure` when you want to keep rendering options bundled with the diagram.

```python
fig = nsx.figure(width=1400, height=700, theme="paper")
fig.draw("model.onnx")
fig.savefig("diagram.svg")
fig.save_html("diagram.html")
fig.export_debug_json("debug.json")
```

`draw()` returns `self` so calls can be chained:

```python
nsx.figure().draw("model.onnx").save_html("out.html")
```

### `Figure.draw(source, **kwargs)`

Parse *source* and store internally.  Returns `self`.

### `Figure.savefig(path, **kwargs)`

Save to *path* (format from extension).

### `Figure.save_html(path, **kwargs)`

Save HTML.

### `Figure.save_svg(path, **kwargs)`

Save SVG (requires Playwright).

### `Figure.show(**kwargs)`

Open in browser.

### `Figure.export_debug_json(path)`

Save verbose debug JSON.

### `Figure.export_paper_json(path)`

Save paper-oriented JSON.

### `Figure.export_nnsvg_json(path, **kwargs)`

Save NN-SVG spec JSON.

---

## Explicit functional API

All functions accept a source (path, dict, or `SemanticArchitecture`).

```python
nsx.parse_model(source)             # -> SemanticArchitecture
nsx.summarize_model(source)         # -> str
nsx.recommend_view(source)          # -> dict (family, confidence, is_approximate, reason, warnings)

nsx.build_nnsvg_spec(source, **kw)  # -> NNSVGSpec
nsx.render_network_html(source)     # -> str (HTML)
nsx.save_network_html(path, source) # -> Path
nsx.save_network_svg(path, source)  # -> Path (needs Playwright)

nsx.export_paper_json(source)       # -> str (JSON)
nsx.export_debug_json(source)       # -> str (JSON)
nsx.save_paper_json(path, source)   # -> Path
nsx.save_debug_json(path, source)   # -> Path
nsx.save_nnsvg_spec(path, spec)     # -> Path
```

### Keyword arguments for rendering

| Argument | Type | Description |
|---|---|---|
| `theme` | `str` | `"paper"`, `"thesis"`, `"debug"`, `"readme"` |
| `style` | `str` | Force `"fcnn"`, `"lenet"`, or `"alexnet"` |
| `width` | `int` | Canvas width in pixels |
| `height` | `int` | Canvas height in pixels |
| `figsize` | `(float, float)` | Matplotlib-style size in inches: `width = round(w * dpi)`, `height = round(h * dpi)` |
| `dpi` | `float` | Pixels per inch for `figsize` conversion (default `100`) |
| `title` | `str` | Diagram title |
| `show_labels` | `bool` | Show layer labels |
| `show_shapes` | `bool` | Show shape annotations |
| `compact` | `bool` | Use compact layout |

`figsize` and `dpi` are accepted anywhere `width`/`height` are accepted.
Explicit `width` or `height` values take priority over the corresponding
`figsize` axis.  Both values must be positive; `figsize` must have exactly
two elements.
