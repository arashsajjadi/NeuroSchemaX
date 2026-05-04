"""Tests for the simplified / stateful module-level API and Figure class."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import neuroschemax as nsx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mlp_spec() -> dict:
    return {
        "model_name": "tiny_mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2", "kind": "dense", "units": 10},
        ],
    }


def _reset_state() -> None:
    """Clear module-level stash between tests."""
    nsx._STATE["arch"] = None
    nsx._STATE["kwargs"] = {}


@pytest.fixture(autouse=True)
def clean_state():
    _reset_state()
    yield
    _reset_state()


# ---------------------------------------------------------------------------
# Module-level stateful API
# ---------------------------------------------------------------------------

def test_draw_and_savefig_html(tmp_path: Path):
    nsx.draw(_mlp_spec())
    out = tmp_path / "diagram.html"
    result = nsx.savefig(str(out))
    assert out.exists()
    assert result == out
    html = out.read_text()
    assert "__nnsvg_ready" in html


def test_draw_and_save_html_explicit(tmp_path: Path):
    out = tmp_path / "explicit.html"
    nsx.save_html(str(out), _mlp_spec())
    assert out.exists()
    html = out.read_text()
    assert "__nnsvg_ready" in html


def test_draw_and_save_html_stash(tmp_path: Path):
    nsx.draw(_mlp_spec())
    out = tmp_path / "stash.html"
    nsx.save_html(str(out))
    assert out.exists()
    html = out.read_text()
    assert "__nnsvg_ready" in html


def test_save_html_shorthand(tmp_path: Path):
    """nsx.save_html(path, spec) should work as a thin one-liner."""
    out = tmp_path / "shorthand.html"
    nsx.save_html(str(out), _mlp_spec())
    assert out.exists()


def test_draw_no_draw_raises():
    """savefig without draw must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="draw"):
        nsx.savefig("should_fail.html")


def test_save_html_no_draw_raises():
    """save_html without source and without draw must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="draw"):
        nsx.save_html("should_fail.html")


# ---------------------------------------------------------------------------
# Figure object
# ---------------------------------------------------------------------------

def test_figure_object(tmp_path: Path):
    fig = nsx.figure(width=800, height=400, theme="paper")
    fig.draw(_mlp_spec())
    out = tmp_path / "fig.html"
    result = fig.savefig(str(out))
    assert out.exists()
    assert result == out


def test_figure_chaining(tmp_path: Path):
    """draw() must return self for chaining."""
    out = tmp_path / "chain.html"
    fig = nsx.figure()
    returned = fig.draw(_mlp_spec())
    assert returned is fig
    fig.save_html(str(out))
    assert out.exists()


def test_figure_export_debug_json(tmp_path: Path):
    out = tmp_path / "debug.json"
    fig = nsx.figure()
    fig.draw(_mlp_spec())
    fig.export_debug_json(str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert "layers" in data


def test_figure_export_paper_json(tmp_path: Path):
    out = tmp_path / "paper.json"
    fig = nsx.figure()
    fig.draw(_mlp_spec())
    fig.export_paper_json(str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["model_name"] == "tiny_mlp"


def test_figure_export_nnsvg_json(tmp_path: Path):
    out = tmp_path / "nnsvg.json"
    fig = nsx.figure()
    fig.draw(_mlp_spec())
    fig.export_nnsvg_json(str(out))
    assert out.exists()
    data = json.loads(out.read_text())
    assert "family" in data


def test_figure_show_is_callable():
    """show() must be callable; mock webbrowser to avoid opening a browser."""
    fig = nsx.figure()
    fig.draw(_mlp_spec())
    with patch("webbrowser.open") as mock_open:
        fig.show()
        assert mock_open.called


# ---------------------------------------------------------------------------
# show() / _display — Jupyter and browser paths
# ---------------------------------------------------------------------------

def test_show_html_opens_browser_outside_jupyter():
    """show_html() calls webbrowser.open when not in Jupyter."""
    from neuroschemax._display import show_html
    with patch("neuroschemax._display._in_jupyter", return_value=False), \
         patch("webbrowser.open") as mock_open:
        show_html("<html><body>test</body></html>")
        assert mock_open.called


def test_show_html_uses_ipython_display_in_jupyter():
    """show_html() calls IPython.display.display when _in_jupyter() is True."""
    from neuroschemax._display import show_html
    mock_display = patch("neuroschemax._display._in_jupyter", return_value=True)
    mock_ipython_display = patch("IPython.display.display")
    mock_ipython_html = patch("IPython.display.HTML", side_effect=lambda h: h)

    try:
        import IPython.display  # noqa: F401
    except ImportError:
        pytest.skip("IPython not installed")

    with mock_display, \
         patch("neuroschemax._display._in_jupyter", return_value=True), \
         patch("IPython.display.display") as mock_disp, \
         patch("IPython.display.HTML", side_effect=lambda h: h):
        show_html("<html><body>test</body></html>")
        assert mock_disp.called


def test_show_html_falls_back_to_browser_if_ipython_missing():
    """If IPython is unavailable even inside 'Jupyter', browser fallback is used."""
    from neuroschemax._display import show_html
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "IPython.display":
            raise ImportError("no IPython")
        return real_import(name, *args, **kwargs)

    with patch("neuroschemax._display._in_jupyter", return_value=True), \
         patch("builtins.__import__", side_effect=mock_import), \
         patch("webbrowser.open") as mock_open:
        show_html("<html><body>test</body></html>")
        assert mock_open.called


def test_nsx_show_uses_browser_outside_jupyter():
    """nsx.show() calls webbrowser.open when not in a Jupyter kernel."""
    nsx.draw(_mlp_spec())
    with patch("neuroschemax._display._in_jupyter", return_value=False), \
         patch("webbrowser.open") as mock_open:
        nsx.show()
        assert mock_open.called


def test_in_jupyter_returns_false_in_test_environment():
    """_in_jupyter() must be False in the standard pytest environment."""
    from neuroschemax._display import _in_jupyter
    assert _in_jupyter() is False


def test_figure_no_draw_raises():
    """Calling savefig without draw should raise RuntimeError."""
    fig = nsx.Figure()
    with pytest.raises(RuntimeError, match="draw"):
        fig.savefig("nope.html")


# ---------------------------------------------------------------------------
# doctor()
# ---------------------------------------------------------------------------

def test_doctor_returns_dict():
    result = nsx.doctor()
    assert isinstance(result, dict)


def test_doctor_has_status():
    result = nsx.doctor()
    assert "status" in result
    assert result["status"] in ("ok", "partial", "error")


def test_doctor_has_expected_keys():
    result = nsx.doctor()
    for key in ("status", "version", "python", "assets", "dependencies", "messages"):
        assert key in result, f"Missing key: {key}"


def test_doctor_assets_is_dict():
    result = nsx.doctor()
    assert isinstance(result["assets"], dict)
    for k in ("util.js", "FCNN.js", "LeNet.js", "AlexNet.js"):
        assert k in result["assets"]


def test_doctor_dependencies_has_keys():
    result = nsx.doctor()
    deps = result["dependencies"]
    for k in ("onnx", "playwright", "chromium", "torch", "tensorflow", "yaml"):
        assert k in deps, f"Missing dependency key: {k}"


# ---------------------------------------------------------------------------
# SVG export error clarity
# ---------------------------------------------------------------------------

def test_savefig_svg_requires_browser(tmp_path: Path):
    """savefig('.svg') must raise BrowserNotAvailableError if Playwright missing."""
    from neuroschemax.visualization.nnsvg_runtime import is_playwright_available
    if is_playwright_available():
        pytest.skip("Playwright is installed; skipping negative test")
    nsx.draw(_mlp_spec())
    with pytest.raises(nsx.BrowserNotAvailableError):
        nsx.savefig(str(tmp_path / "out.svg"))


def test_save_svg_shorthand_requires_browser(tmp_path: Path):
    """save_svg must raise BrowserNotAvailableError if Playwright missing."""
    from neuroschemax.visualization.nnsvg_runtime import is_playwright_available
    if is_playwright_available():
        pytest.skip("Playwright is installed")
    with pytest.raises(nsx.BrowserNotAvailableError):
        nsx.save_svg(str(tmp_path / "out.svg"), _mlp_spec())


# ---------------------------------------------------------------------------
# HTML warning display
# ---------------------------------------------------------------------------

def test_html_contains_warning_badge_for_approximate_model(tmp_path: Path):
    """ResNet-like model (with Add nodes) should have warning badges in HTML."""
    resnet_spec = {
        "model_name": "resnet_like",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            {"name": "c1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c2", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c3", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c4", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "add1", "kind": "add"},
            {"name": "fc", "kind": "dense", "units": 10},
        ],
    }
    out = tmp_path / "resnet.html"
    nsx.save_html(str(out), resnet_spec)
    html = out.read_text()
    # Warnings are present in the HTML as styled badges
    assert "nnsvg-warning-item" in html or "Approximate:" in html


def test_html_no_warning_badge_for_clean_model(tmp_path: Path):
    """Clean MLP should have no warning div content."""
    out = tmp_path / "mlp.html"
    nsx.save_html(str(out), _mlp_spec())
    html = out.read_text()
    # The warning div should be absent (only the CSS class may appear in <style>)
    assert "<div class='nnsvg-warnings'>" not in html


# ---------------------------------------------------------------------------
# CLI external-directory usage
# ---------------------------------------------------------------------------

def test_cli_draw_default_output(tmp_path: Path):
    """draw without -o writes <model_stem>.html to the CWD."""
    import os
    from neuroschemax.cli import main

    spec_path = tmp_path / "my_model.json"
    spec_path.write_text(json.dumps(_mlp_spec()))

    # Run from tmp_path so the default output goes there
    orig = os.getcwd()
    try:
        os.chdir(tmp_path)
        rc = main(["draw", str(spec_path)])
    finally:
        os.chdir(orig)

    assert rc == 0
    default_out = tmp_path / "my_model.html"
    assert default_out.exists(), f"Default output not found: {default_out}"
    assert "__nnsvg_ready" in default_out.read_text()


def test_cli_draw_explicit_output(tmp_path: Path):
    """draw with -o writes to the specified path."""
    from neuroschemax.cli import main

    spec_path = tmp_path / "model.json"
    spec_path.write_text(json.dumps(_mlp_spec()))
    out_path = tmp_path / "explicit_out.html"

    rc = main(["draw", str(spec_path), "-o", str(out_path)])
    assert rc == 0
    assert out_path.exists()


def test_cli_draw_nonexistent_file_returns_error():
    """draw with a missing file must return exit code 1."""
    from neuroschemax.cli import main
    rc = main(["draw", "/definitely/does/not/exist.onnx"])
    assert rc == 1


# ---------------------------------------------------------------------------
# figsize / dpi
# ---------------------------------------------------------------------------

def test_figsize_converts_to_width_height():
    spec = nsx.build_nnsvg_spec(_mlp_spec(), figsize=(14, 7))
    assert spec.width == 1400   # 14 * 100
    assert spec.height == 700   # 7  * 100


def test_figsize_with_custom_dpi():
    spec = nsx.build_nnsvg_spec(_mlp_spec(), figsize=(10, 5), dpi=120)
    assert spec.width == 1200   # 10 * 120
    assert spec.height == 600   # 5  * 120


def test_explicit_width_overrides_figsize():
    """Explicit width wins; figsize still sets height."""
    spec = nsx.build_nnsvg_spec(_mlp_spec(), figsize=(14, 7), width=800)
    assert spec.width == 800    # explicit
    assert spec.height == 700   # from figsize


def test_explicit_height_overrides_figsize():
    """Explicit height wins; figsize still sets width."""
    spec = nsx.build_nnsvg_spec(_mlp_spec(), figsize=(14, 7), height=400)
    assert spec.width == 1400   # from figsize
    assert spec.height == 400   # explicit


def test_figsize_via_figure_object(tmp_path):
    fig = nsx.figure(figsize=(12, 6), dpi=100)
    fig.draw(_mlp_spec())
    out = tmp_path / "fig.html"
    fig.save_html(str(out))
    # Just confirm it produced a file without error
    assert out.exists()


def test_figsize_via_nsx_draw(tmp_path):
    nsx.draw(_mlp_spec())
    out = tmp_path / "sized.html"
    nsx.savefig(str(out), figsize=(10, 5), dpi=80)
    assert out.exists()


def test_invalid_figsize_raises():
    with pytest.raises(ValueError, match="figsize"):
        nsx.build_nnsvg_spec(_mlp_spec(), figsize=(0, 5))

    with pytest.raises(ValueError, match="figsize"):
        nsx.build_nnsvg_spec(_mlp_spec(), figsize=(-1, 5))

    with pytest.raises(ValueError, match="figsize"):
        nsx.build_nnsvg_spec(_mlp_spec(), figsize=(10,))  # wrong length


def test_invalid_dpi_raises():
    with pytest.raises(ValueError, match="dpi"):
        nsx.build_nnsvg_spec(_mlp_spec(), figsize=(10, 5), dpi=0)

    with pytest.raises(ValueError, match="dpi"):
        nsx.build_nnsvg_spec(_mlp_spec(), figsize=(10, 5), dpi=-50)
