# Architecture

NeuroSchemaX is organised as a linear pipeline from source model to
final diagram. Each stage has a single responsibility and a well-defined
data contract.

## Pipeline overview

```
   source model
       │
       ▼
┌──────────────┐    ingest/
│   adapter    │    framework-specific parsers
└──────┬───────┘
       │  GraphIR (raw nodes, edges, shapes)
       ▼
┌──────────────┐    normalize/
│  normaliser  │    op mapping, shape inference, block fusion,
└──────┬───────┘    skip detection, family recognition
       │  SemanticArchitecture
       ▼
┌──────────────┐    visualization/nnsvg_mapper.py
│    mapper    │    semantic → NN-SVG spec (family-aware)
└──────┬───────┘
       │  NNSVGSpec
       ▼
┌──────────────┐    visualization/nnsvg_html.py
│ HTML render  │    inline assets + JSON spec + runtime bootstrap
└──────┬───────┘
       │  standalone HTML
       ▼
┌──────────────┐    visualization/nnsvg_runtime.py
│ SVG extract  │    headless Chromium → window.__nnsvg_export_svg()
└──────┬───────┘
       │
       ▼
     SVG / HTML / JSON on disk
```

## Module map

| Package                     | Responsibility                               |
| --------------------------- | -------------------------------------------- |
| `neuroschemax.core`         | Enums, aliases, `RenderConfig`, validation   |
| `neuroschemax.ir`           | `GraphIR`, `SemanticArchitecture` dataclasses |
| `neuroschemax.ingest`       | ONNX / PyTorch / TF / manual-spec adapters   |
| `neuroschemax.normalize`    | Op mapping, shape inference, family picker   |
| `neuroschemax.visualization`| NN-SVG spec, HTML gen, SVG extraction, JS assets |
| `neuroschemax.exporters`    | Paper JSON, debug JSON, NN-SVG JSON          |
| `neuroschemax.presets`      | Theme presets (paper / thesis / debug / readme) |
| `neuroschemax.api`          | Public, user-facing functions                |
| `neuroschemax.cli`          | Argparse-based CLI                           |

## Why NN-SVG?

NN-SVG by Alex Lenail is an established, well-understood rendering tool
that produces diagrams in three recognisable families — FCNN, LeNet,
and AlexNet. Rather than reinvent a renderer, NeuroSchemaX integrates
NN-SVG directly and focuses its effort on the parts that are genuinely
hard for Python: understanding what a model *is* and choosing the right
family to represent it.

## Data contracts

- **GraphIR**: low-level, framework-agnostic, structure-preserving.
  Nodes hold raw op types and raw attributes. No semantics.
- **SemanticArchitecture**: normalised. Layers have `LayerKind`,
  inferred shapes, block groupings, skip links, a recommended family,
  and confidence/warning fields.
- **NNSVGSpec**: final render config. Family-specific; JSON-serialisable.

Each stage is pure: given the same input, it produces the same output.
