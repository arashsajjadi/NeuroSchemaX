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
    patch("IPython.display.display")
    patch("IPython.display.HTML", side_effect=lambda h: h)

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
    import builtins

    from neuroschemax._display import show_html
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


# ---------------------------------------------------------------------------
# Figure.to_html / _repr_html_ (notebook / Colab API)
# ---------------------------------------------------------------------------

def test_figure_to_html_returns_string():
    """Figure.to_html() returns a non-empty HTML string."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec())
    html = fig.to_html()
    assert isinstance(html, str)
    assert len(html) > 100
    assert "<!DOCTYPE html>" in html


def test_figure_to_html_is_self_contained():
    """to_html() HTML is offline-capable (no external URLs required)."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec())
    html = fig.to_html()
    # Self-contained: no src="http..." or href="http..." for critical assets
    import re
    external_srcs = re.findall(r'src="https?://', html)
    assert not external_srcs, f"External src links found: {external_srcs}"


def test_repr_html_returns_string():
    """_repr_html_() is called by Jupyter rich display."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec())
    html = fig._repr_html_()
    assert isinstance(html, str)
    # _repr_html_ now returns a notebook-safe iframe wrapper
    assert "<iframe" in html
    assert len(html) > 100


def test_repr_html_before_draw_returns_placeholder():
    """_repr_html_() before draw() returns a safe placeholder, not an error."""
    fig = nsx.Figure()
    html = fig._repr_html_()
    assert isinstance(html, str)
    assert len(html) > 0
    # Should mention draw()
    assert "draw" in html.lower() or "loaded" in html.lower()


def test_figure_to_html_matches_save_html(tmp_path: Path):
    """to_html() and save_html() produce the same content."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec())
    html_str = fig.to_html()
    path = tmp_path / "mlp.html"
    fig.save_html(path)
    html_file = path.read_text()
    assert html_str == html_file


# ---------------------------------------------------------------------------
# doctor() capabilities key
# ---------------------------------------------------------------------------

def test_doctor_has_capabilities_key():
    result = nsx.doctor()
    assert "capabilities" in result, "doctor() must include 'capabilities' dict"


def test_doctor_capabilities_has_html_export():
    result = nsx.doctor()
    caps = result["capabilities"]
    assert "html_export" in caps
    # HTML export should be True when assets are present (they always are in dev)
    assert caps["html_export"] is True


def test_doctor_capabilities_has_svg_export():
    result = nsx.doctor()
    caps = result["capabilities"]
    assert "svg_export" in caps
    assert isinstance(caps["svg_export"], bool)


def test_doctor_dependencies_has_ipython_key():
    result = nsx.doctor()
    deps = result["dependencies"]
    assert "ipython" in deps, "doctor() dependencies must include 'ipython'"


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def test_version_is_at_least_0_1_1():
    from neuroschemax.version import __version__
    assert __version__ >= "0.1.1"


# ---------------------------------------------------------------------------
# Notebook/IPython detection fixes
# ---------------------------------------------------------------------------

def test_can_display_html_uses_ipython_display_api():
    """_can_display_html() returns True when IPython.display.HTML/display import works."""
    from neuroschemax._display import _can_display_html
    # In this test environment IPython is installed, so this must be True.
    try:
        from IPython.display import HTML, display  # type: ignore[import]  # noqa: F401
        expected = True
    except ImportError:
        expected = False
    assert _can_display_html() == expected


def test_doctor_notebook_display_matches_can_display_html():
    """doctor() notebook_display must agree with _can_display_html()."""
    from neuroschemax._display import _can_display_html
    result = nsx.doctor()
    assert result["capabilities"]["notebook_display"] == _can_display_html()


def test_doctor_ipython_key_matches_can_display_html():
    """doctor() dependencies['ipython'] must agree with _can_display_html()."""
    from neuroschemax._display import _can_display_html
    result = nsx.doctor()
    assert result["dependencies"]["ipython"] == _can_display_html()


def test_can_display_html_does_not_use_lowercase_ipython_import():
    """_can_display_html() must not rely on 'import ipython' (lowercase)."""
    # Simulate the case where the lowercase 'ipython' package cannot be imported
    # but IPython.display IS available — the canonical Colab/Jupyter situation.
    import builtins
    real_import = builtins.__import__

    def block_lowercase(name: str, *args, **kwargs):
        if name == "ipython":            # block only the lowercase bare name
            raise ImportError("no lowercase ipython")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=block_lowercase):
        from neuroschemax._display import _can_display_html
        # Must still return True because IPython.display is reachable
        try:
            from IPython.display import HTML, display  # type: ignore  # noqa: F401
            ipython_available = True
        except ImportError:
            ipython_available = False

        result = _can_display_html()
        assert result == ipython_available, (
            "_can_display_html() should not fail just because 'import ipython' (lowercase) fails"
        )


# ---------------------------------------------------------------------------
# show() / to_html() — notebook paths
# ---------------------------------------------------------------------------

def test_show_html_in_jupyter_calls_display_not_browser():
    """In Jupyter mode, show_html must call IPython.display.display, not webbrowser."""
    try:
        from IPython.display import HTML, display  # type: ignore[import]  # noqa: F401
    except ImportError:
        pytest.skip("IPython not installed")

    from neuroschemax._display import show_html
    with patch("neuroschemax._display._in_jupyter", return_value=True), \
         patch("neuroschemax._display._in_colab", return_value=False), \
         patch("neuroschemax._display._can_display_html", return_value=True), \
         patch("IPython.display.display") as mock_disp, \
         patch("IPython.display.HTML", side_effect=lambda h: h), \
         patch("webbrowser.open") as mock_browser:
        show_html("<html><body>hello</body></html>")
        assert mock_disp.called, "display() must be called in Jupyter mode"
        assert not mock_browser.called, "webbrowser must NOT be opened in Jupyter mode"


def test_show_html_in_colab_calls_display_not_iframe_file(tmp_path: Path):
    """In Colab mode, show_html must not produce a local-file IFrame."""
    try:
        from IPython.display import HTML, display  # type: ignore[import]  # noqa: F401
    except ImportError:
        pytest.skip("IPython not installed")

    from neuroschemax._display import show_html
    displayed_objects = []

    def capture_display(obj):
        displayed_objects.append(obj)

    with patch("neuroschemax._display._in_jupyter", return_value=True), \
         patch("neuroschemax._display._in_colab", return_value=True), \
         patch("neuroschemax._display._can_display_html", return_value=True), \
         patch("IPython.display.display", side_effect=capture_display), \
         patch("IPython.display.HTML", side_effect=lambda h: ("HTML", h)), \
         patch("webbrowser.open") as mock_browser:
        show_html("<html><body>colab test</body></html>")
        assert not mock_browser.called, "webbrowser must not open in Colab mode"
        # Must have called display() with the HTML content
        assert len(displayed_objects) > 0, "display() must be called in Colab mode"
        # Must NOT have passed a local /tmp/ file path to display()
        for obj in displayed_objects:
            obj_str = str(obj)
            assert "/tmp/" not in obj_str, (
                "Colab display must not use a local /tmp/ file path IFrame"
            )


def test_figure_show_manual_spec_no_tmp_file_in_notebook():
    """fig.show() with manual spec must not use local temp-file IFrame in notebook mode."""
    try:
        from IPython.display import HTML, display  # type: ignore[import]  # noqa: F401
    except ImportError:
        pytest.skip("IPython not installed")

    spec = {
        "model_name": "test_mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 10},
        ],
    }
    fig = nsx.Figure()
    fig.draw(spec)
    displayed: list = []

    with patch("neuroschemax._display._in_jupyter", return_value=True), \
         patch("neuroschemax._display._in_colab", return_value=True), \
         patch("neuroschemax._display._can_display_html", return_value=True), \
         patch("IPython.display.display", side_effect=displayed.append), \
         patch("IPython.display.HTML", side_effect=lambda h: ("HTML", h)):
        fig.show()

    assert displayed, "display() must have been called"
    for obj in displayed:
        assert "/tmp/" not in str(obj), "Must not display a /tmp/ path"


def test_figure_to_html_manual_spec_contains_nnsvg():
    """fig.to_html() for a manual spec must contain NN-SVG rendering code."""
    spec = {
        "model_name": "test",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 10},
        ],
    }
    fig = nsx.Figure()
    fig.draw(spec)
    html = fig.to_html()
    assert "<!DOCTYPE html>" in html
    assert "renderNNSVG" in html or "renderFCNN" in html
    assert "test" in html  # model name in title


def test_figure_to_html_onnx_model_contains_nnsvg():
    """fig.to_html() for an ONNX model must contain NN-SVG rendering code."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx not available")

    relu = helper.make_node("Relu", ["x"], ["out"])
    x    = helper.make_tensor_value_info("x",   TensorProto.FLOAT, [1, 64])
    out  = helper.make_tensor_value_info("out",  TensorProto.FLOAT, [1, 64])
    graph = helper.make_graph([relu], "onnx_model", [x], [out])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8

    fig = nsx.Figure()
    fig.draw(model)
    html = fig.to_html()
    assert "<!DOCTYPE html>" in html
    assert len(html) > 200


def test_show_html_outside_notebook_opens_browser():
    """show_html() outside any notebook environment opens the browser."""
    from neuroschemax._display import show_html
    with patch("neuroschemax._display._in_jupyter", return_value=False), \
         patch("neuroschemax._display._in_colab", return_value=False), \
         patch("webbrowser.open") as mock_browser:
        show_html("<html><body>browser test</body></html>")
        assert mock_browser.called


# ---------------------------------------------------------------------------
# Notebook-safe display: srcdoc iframe, no raw CSS/JS leakage
# ---------------------------------------------------------------------------

def _mlp_spec_simple() -> dict:
    return {
        "model_name": "nb_mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 64},
            {"name": "out",   "kind": "dense", "units": 10},
        ],
    }


def test_to_notebook_html_is_iframe_not_full_document():
    """to_notebook_html() must return an iframe wrapper, not a full HTML document.

    The outer notebook HTML must be just an <iframe> element.  The full
    standalone HTML lives inside the srcdoc attribute value, where `<`, `>`,
    `&`, and `"` are all HTML-entity-escaped.  This means `<!DOCTYPE html>`
    does NOT appear as a raw string in the outer wrapper.
    """
    fig = nsx.Figure()
    fig.draw(_mlp_spec_simple())
    nb_html = fig.to_notebook_html()
    assert "<iframe" in nb_html, "Notebook HTML must use an iframe"
    assert "srcdoc=" in nb_html, "iframe must use srcdoc attribute"
    # The outer wrapper must not start with a full HTML document preamble
    assert nb_html.strip().startswith("<iframe"), (
        "to_notebook_html() must start with <iframe, not a full HTML document"
    )
    # Raw (unescaped) DOCTYPE string must not appear in the outer wrapper
    assert "<!DOCTYPE html>" not in nb_html, (
        "Raw <!DOCTYPE html> must not appear in the outer notebook wrapper "
        "(it should be &lt;!DOCTYPE html&gt; inside the srcdoc value)"
    )


def test_to_notebook_html_no_raw_css_visible():
    """to_notebook_html() must not expose raw CSS rules as visible text.

    With full escaping (`<` → `&lt;`, `>` → `&gt;`) the `<style>` tag is
    encoded to `&lt;style&gt;` inside the srcdoc value, so no raw CSS tag or
    CSS content appears in the outer notebook HTML.
    """
    fig = nsx.Figure()
    fig.draw(_mlp_spec_simple())
    nb_html = fig.to_notebook_html()
    # Raw <style> must not appear — it will be &lt;style&gt; in the srcdoc value
    assert "<style>" not in nb_html, (
        "Raw <style> tag must not appear in the outer notebook HTML"
    )


def test_to_notebook_html_no_raw_js_visible():
    """to_notebook_html() must not expose executable <script> tags in the outer wrapper.

    The root cause of Colab's raw-JS-as-text problem is that ``<script>`` tags
    in the content passed to ``IPython.display.HTML()`` are not executed —
    Colab renders them as literal text.  The srcdoc iframe approach encodes
    ``<script>`` as ``&lt;script&gt;``, so no raw ``<script>`` tag appears in
    the outer notebook document.  JS source text (comments like ``// NN-SVG``)
    may still appear inside the attribute value, but that does not leak as
    visible text because it is enclosed within the ``srcdoc="..."`` attribute.
    """
    fig = nsx.Figure()
    fig.draw(_mlp_spec_simple())
    nb_html = fig.to_notebook_html()
    # Raw <script> tag must not appear — it must be encoded as &lt;script&gt;
    assert "<script>" not in nb_html, (
        "Raw <script> tag must not appear in the outer notebook HTML"
    )
    # Raw closing tag similarly must not appear
    assert "</script>" not in nb_html


def test_to_notebook_html_no_tmp_file_path():
    """to_notebook_html() must not reference a /tmp/*.html file path."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec_simple())
    nb_html = fig.to_notebook_html()
    assert "/tmp/" not in nb_html


def test_to_html_still_returns_full_standalone():
    """to_html() must still return a complete standalone HTML document."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec_simple())
    html = fig.to_html()
    assert "<!DOCTYPE html>" in html
    assert "<style>" in html
    assert "<script>" in html
    assert "renderNNSVG" in html or "renderFCNN" in html


def test_save_html_unchanged(tmp_path: Path):
    """save_html() must still write a full standalone HTML document."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec_simple())
    out = tmp_path / "mlp.html"
    fig.save_html(out)
    content = out.read_text()
    assert "<!DOCTYPE html>" in content
    assert "<style>" in content
    assert "<script>" in content


def test_repr_html_uses_iframe_not_full_document():
    """_repr_html_() must return an iframe notebook wrapper, not a full document."""
    fig = nsx.Figure()
    fig.draw(_mlp_spec_simple())
    repr_html = fig._repr_html_()
    assert "<iframe" in repr_html
    assert "srcdoc=" in repr_html
    assert repr_html.strip().startswith("<iframe"), (
        "_repr_html_() must produce an <iframe>, not a full HTML document"
    )
    # Raw DOCTYPE and style/script must not appear in the outer wrapper
    assert "<!DOCTYPE html>" not in repr_html
    assert "<style>" not in repr_html
    assert "<script>" not in repr_html


def test_make_notebook_iframe_escapes_quotes():
    """_make_notebook_iframe escapes double-quotes so the srcdoc attribute is valid."""
    from neuroschemax._display import _make_notebook_iframe
    html_with_quotes = '<html><body class="x">hello &amp; world</body></html>'
    result = _make_notebook_iframe(html_with_quotes)
    assert 'srcdoc="' in result
    # Double quotes inside the srcdoc value must be &quot;
    # The original " in class="x" should have been escaped
    assert 'class=&quot;x&quot;' in result or '"x"' not in result


def test_make_notebook_iframe_escapes_ampersands():
    """_make_notebook_iframe must escape & as &amp; inside the srcdoc value."""
    from neuroschemax._display import _make_notebook_iframe
    html_with_amp = "<html><body>a &amp; b</body></html>"
    result = _make_notebook_iframe(html_with_amp)
    # & → &amp; first pass, then the result &amp; → &amp;amp; in srcdoc
    # Verify no unescaped bare & remains in the attribute value
    srcdoc_start = result.index('srcdoc="') + len('srcdoc="')
    srcdoc_end = result.rindex('"', srcdoc_start)
    srcdoc_val = result[srcdoc_start:srcdoc_end]
    import re
    bare_amp = re.findall(r"&(?!amp;|quot;|lt;|gt;|#)", srcdoc_val)
    assert not bare_amp, f"Unescaped & found in srcdoc: {bare_amp}"


def test_show_html_in_notebook_uses_iframe(monkeypatch: pytest.MonkeyPatch):
    """show_html() in notebook mode must pass an iframe to display(), not full HTML."""
    try:
        from IPython.display import HTML, display  # type: ignore  # noqa: F401
    except ImportError:
        pytest.skip("IPython not installed")

    from neuroschemax._display import show_html

    displayed_content: list[str] = []

    def capture(obj: object) -> None:
        # obj is whatever was passed to display() — may be wrapped by HTML()
        displayed_content.append(str(obj))

    monkeypatch.setattr("neuroschemax._display._in_jupyter", lambda: True)
    monkeypatch.setattr("neuroschemax._display._in_colab", lambda: False)
    monkeypatch.setattr("neuroschemax._display._can_display_html", lambda: True)

    import neuroschemax._display as _disp_mod
    monkeypatch.setattr(_disp_mod, "_can_display_html", lambda: True)

    with patch("IPython.display.display", side_effect=capture), \
         patch("IPython.display.HTML", side_effect=lambda h: h):
        show_html("<html><head><style>body{margin:0}</style></head>"
                  "<body><script>// NN-SVG utils</script></body></html>")

    assert displayed_content, "display() must have been called"
    combined = " ".join(displayed_content)
    assert "<iframe" in combined, "display() must receive iframe content"
    assert "<style>" not in combined, "Raw <style> must not appear in notebook output"
    assert "<script>" not in combined, "Raw <script> must not appear in notebook output"


def test_onnx_figure_to_notebook_html_no_raw_css():
    """ONNX-backed figure to_notebook_html() must not expose raw CSS."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx not available")
    relu = helper.make_node("Relu", ["x"], ["out"])
    x   = helper.make_tensor_value_info("x",   TensorProto.FLOAT, [1, 64])
    out = helper.make_tensor_value_info("out",  TensorProto.FLOAT, [1, 64])
    graph = helper.make_graph([relu], "onnx_nb", [x], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    fig = nsx.Figure()
    fig.draw(model)
    nb_html = fig.to_notebook_html()
    assert "<iframe" in nb_html
    assert "<style>" not in nb_html
    assert "<script>" not in nb_html
