# Changelog

All notable changes to NeuroSchemaX are documented here.

The format is loosely based on [Keep a Changelog][kac], and this project
follows [Semantic Versioning][semver].

[kac]: https://keepachangelog.com/en/1.1.0/
[semver]: https://semver.org/spec/v2.0.0.html

## [0.1.1] — 2026-05-04

### Fixed

- **ONNX adapter**: `_extract_shape` and `_extract_dtype` now accept both
  `ValueInfoProto` and bare `TypeProto` inputs.  The previous code crashed
  with `AttributeError: tensor_type` when parsing real PyTorch-exported
  ONNX models.  Both helpers are now fully defensive and never raise.
- **ONNX shape inference**: `onnx.shape_inference.infer_shapes` is run
  before building the shape map so intermediate tensor shapes are populated
  for sequential models (Conv, ReLU, MaxPool, Gemm, …).
- **Figure.to_html / _repr_html_**: `Figure` now exposes `to_html()` for
  explicit HTML-string access and `_repr_html_()` for automatic Jupyter/
  JupyterLab inline rendering.  Calling `fig` as the last expression in a
  cell renders the diagram inline without any manual `display()` call.
- **Colab display**: `show_html()` detects Google Colab and uses an IFrame
  with a helpful message; falls back gracefully to `IPython.display.HTML`.
- **doctor()**: added `capabilities` dict separating HTML export (always
  available) from SVG export (requires Playwright + Chromium); added
  `ipython` to `dependencies`; messages now include `[colab]` extra hint.

### Changed

- `pyproject.toml`: author/maintainer set to Arash Sajjadi; project URLs
  updated to `github.com/arashsajjadi/NeuroSchemaX`; added `onnx`, `export`,
  and `colab` optional extras; `ipython` added to `colab` and `dev` extras.

### Added

- `examples/colab_quickstart.py` — runnable Colab/Jupyter quickstart covering
  manual specs, PyTorch→ONNX roundtrip, and Transformer block summary.
- Regression tests in `tests/test_onnx_parsing.py` covering defensive ONNX
  parsing, shape inference, Gemm units, dynamic shapes, and file-based loading.
- Tests for `Figure.to_html`, `_repr_html_`, `doctor()` capabilities key,
  and version string.

## [0.1.0] — 2026-05-01

### Added

- Initial public release.
- ONNX, manual JSON/YAML, PyTorch, and TensorFlow ingestion adapters.
- Graph IR and Semantic IR layers with shape inference, block fusion,
  skip-connection detection, and family recommendation.
- NN-SVG integration: `NNSVGSpec` schema, family-aware mapper, standalone
  HTML generator, headless-Chromium SVG extraction via Playwright.
- Four theme presets (`paper`, `thesis`, `debug`, `readme`).
- Python API: `parse_model`, `summarize_model`, `recommend_view`,
  `build_nnsvg_spec`, `render_network_html`, `render_network_svg`,
  `save_network_html`, `save_network_svg`, `export_paper_json`,
  `export_debug_json`, `export_nnsvg_spec`, `save_paper_json`,
  `save_debug_json`, `save_nnsvg_spec`.
- CLI subcommands: `inspect`, `summarize`, `recommend-view`, `render`,
  `export-paper-json`, `export-debug-json`, `export-nnsvg`, `doctor`.
- Example scripts and sample YAML configs.
- Test suite covering imports, CLI smoke paths, and pipeline structure.
- CI workflow (GitHub Actions) for Python 3.11 and 3.12.
