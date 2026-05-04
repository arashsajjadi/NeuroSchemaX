# Changelog

All notable changes to NeuroSchemaX are documented here.

The format is loosely based on [Keep a Changelog][kac], and this project
follows [Semantic Versioning][semver].

[kac]: https://keepachangelog.com/en/1.1.0/
[semver]: https://semver.org/spec/v2.0.0.html

## [0.1.0] — Unreleased

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
