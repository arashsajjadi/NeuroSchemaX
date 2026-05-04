# Contributing to NeuroSchemaX

Thanks for your interest! NeuroSchemaX is a young project and
contributions of all sizes are welcome.

## Development setup

```bash
git clone <repo-url>
cd NeuroSchemaX
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For SVG-export tests, also install Playwright:

```bash
pip install playwright
playwright install chromium
```

## Layout

The codebase is strictly layered. When adding code, put it in the layer
that owns the concern:

| Concern                            | Go to                           |
| ---------------------------------- | ------------------------------- |
| A new input framework              | `neuroschemax/ingest/`          |
| A new op / layer kind              | `neuroschemax/normalize/op_normalizer.py` + `core/enums.py` |
| A new semantic heuristic           | `neuroschemax/normalize/`       |
| A new family mapper (NN-SVG)       | `neuroschemax/visualization/nnsvg_mapper.py` |
| A new render family (NN-SVG asset) | `neuroschemax/visualization/assets/` + `nnsvg_schema.py` |
| A new export format                | `neuroschemax/exporters/` + `api/export.py` |
| A new preset / theme               | `neuroschemax/presets/`         |
| A new CLI subcommand               | `neuroschemax/cli.py`           |

Keep the public API (`neuroschemax/__init__.py`) honest: every name in
`__all__` must be importable and callable.

## Style

- Python ≥ 3.11 syntax (`X | Y`, `from __future__ import annotations`).
- Type hints on every public function.
- Docstrings on every public function and class.
- `ruff check .` must pass.
- No silent `except:` blocks; raise a specific `NeuroSchemaXError`.

## Tests

- Add tests under `tests/`. Keep each test focused.
- If you add a public symbol, add it to `tests/test_imports.py` too.
- If you add a CLI flag, cover it in `tests/test_cli_smoke.py`.
- Run `pytest -q` before opening a PR.

## Pull requests

1. Fork, branch from `main`.
2. Keep PRs small and focused; one logical change per PR.
3. Describe the *why* in the PR body, not just the *what*.
4. Reference any issue it fixes.

## Reporting bugs

Please include:

- A minimal reproducer (ideally a manual JSON spec).
- `neuroschemax doctor` output.
- The full traceback.
