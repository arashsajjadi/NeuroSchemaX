# Packaging and Publishing

## Development install

```bash
git clone https://github.com/neuroschemax/neuroschemax
cd neuroschemax
pip install -e ".[dev]"
```

## Run tests

```bash
pytest tests/ -v
```

## Run linter

```bash
ruff check src/ tests/
```

## Build a distribution

```bash
pip install build
python -m build
```

This produces `dist/neuroschemax-*.whl` and `dist/neuroschemax-*.tar.gz`.

## Publishing checklist

Before publishing to PyPI:

- [ ] Bump version in `src/neuroschemax/version.py` and `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linter errors: `ruff check src/ tests/`
- [ ] `MANIFEST.in` includes all required assets
- [ ] `pyproject.toml` `[tool.setuptools.package-data]` includes all JS and font files
- [ ] Build succeeds: `python -m build`
- [ ] Test on TestPyPI first: `twine upload --repository testpypi dist/*`
- [ ] Verify install from TestPyPI and run `neuroschemax doctor`
- [ ] Publish: `twine upload dist/*`

## Optional dependencies

Install extras for optional features:

```bash
pip install neuroschemax[svg]    # SVG export via Playwright
pip install neuroschemax[torch]  # PyTorch model input
pip install neuroschemax[tf]     # TensorFlow/Keras model input
pip install neuroschemax[dev]    # Development tools
pip install neuroschemax[all]    # All optional deps
```
