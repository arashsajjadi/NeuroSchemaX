# CLI Reference

NeuroSchemaX installs a `neuroschemax` command.

```
usage: neuroschemax [-h] [--version] <command> ...
```

## Commands

### `draw` — draw a diagram (friendly alias for `render`)

```bash
neuroschemax draw model.onnx
# Saves model.html in the current directory

neuroschemax draw model.onnx -o out.html --theme paper --width 1400
neuroschemax draw model.onnx -o out.svg  --style lenet
```

| Flag | Description |
|---|---|
| `model` | Path to model file |
| `-o` / `--output` | Output path (default: `<model>.html`) |
| `--theme` | `paper`, `thesis`, `debug`, `readme` |
| `--style` | `fcnn`, `lenet`, `alexnet` |
| `--width` | Canvas width |
| `--height` | Canvas height |
| `--title` | Diagram title |
| `--no-labels` | Hide layer labels |
| `--no-shapes` | Hide shape annotations |
| `--compact` | Compact layout |

### `render` — render to HTML or SVG

Same flags as `draw` but `-o` / `--output` is required.

```bash
neuroschemax render model.onnx -o diagram.html
neuroschemax render model.onnx -o diagram.svg --theme thesis
```

### `inspect` — print structural overview

```bash
neuroschemax inspect model.onnx
```

### `summarize` — print layer summary

```bash
neuroschemax summarize model.onnx
neuroschemax summarize model.onnx --format markdown
```

### `recommend-view` — print recommended rendering style

```bash
neuroschemax recommend-view model.onnx
```

Output is JSON with `family`, `confidence`, `reason`, and `warnings`.

### `export-paper-json` — export paper-oriented JSON

```bash
neuroschemax export-paper-json model.onnx -o paper.json
```

### `export-debug-json` — export verbose debug JSON

```bash
neuroschemax export-debug-json model.onnx -o debug.json
```

### `export-nnsvg` — export NN-SVG spec JSON

```bash
neuroschemax export-nnsvg model.onnx -o spec.json
```

### `doctor` — environment diagnostics

```bash
neuroschemax doctor
```

Prints a human-readable pass/fail summary of assets, dependencies, and
any actionable messages (e.g. "Run: playwright install chromium").
