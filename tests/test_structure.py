"""Integration tests covering the parse → normalise → render pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import neuroschemax as nsx
from neuroschemax.core.enums import LayerKind, RenderFamily


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


def _small_cnn_spec() -> dict:
    return {
        "model_name": "tiny_cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16,
             "kernel_size": [3, 3], "stride": [1, 1]},
            {"name": "relu1", "kind": "relu"},
            {"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "conv2", "kind": "conv", "out_channels": 32,
             "kernel_size": [3, 3], "stride": [1, 1]},
            {"name": "relu2", "kind": "relu"},
            {"name": "pool2", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "flatten", "kind": "flatten"},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "fc2", "kind": "dense", "units": 10},
        ],
    }


def _deep_cnn_spec() -> dict:
    layers = [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
    for i in range(6):
        layers.append({
            "name": f"conv{i}", "kind": "conv",
            "out_channels": 64 * (i + 1), "kernel_size": [3, 3],
        })
        layers.append({"name": f"relu{i}", "kind": "relu"})
    layers.append({"name": "pool", "kind": "globalaveragepool"})
    layers.append({"name": "fc", "kind": "dense", "units": 1000})
    return {"model_name": "deep_cnn", "layers": layers}


def test_parse_mlp():
    arch = nsx.parse_model(_mlp_spec())
    assert arch.model_name == "tiny_mlp"
    assert arch.recommended_family == RenderFamily.FCNN
    assert any(lay.kind == LayerKind.DENSE for lay in arch.layers)


def test_parse_small_cnn():
    arch = nsx.parse_model(_small_cnn_spec())
    assert arch.recommended_family == RenderFamily.LENET
    assert arch.has_convolutions


def test_parse_deep_cnn():
    arch = nsx.parse_model(_deep_cnn_spec())
    assert arch.recommended_family == RenderFamily.ALEXNET


def test_build_nnsvg_spec_mlp():
    spec = nsx.build_nnsvg_spec(_mlp_spec())
    assert spec.family == RenderFamily.FCNN
    assert len(spec.layers) > 0


def test_build_nnsvg_spec_cnn():
    spec = nsx.build_nnsvg_spec(_small_cnn_spec())
    assert spec.family == RenderFamily.LENET
    # Should contain both conv-type and dense-type layers
    types = {lay.layer_type for lay in spec.layers}
    assert "conv" in types
    assert "dense" in types


def test_build_nnsvg_spec_force_style():
    spec = nsx.build_nnsvg_spec(_mlp_spec(), style="lenet")
    assert spec.family == RenderFamily.LENET


def test_build_nnsvg_spec_theme_preset():
    spec = nsx.build_nnsvg_spec(_mlp_spec(), theme="paper")
    assert spec.width == 1400  # paper preset


def test_render_html_contains_runtime_hooks(tmp_path: Path):
    out = tmp_path / "diagram.html"
    nsx.save_network_html(out, _mlp_spec())
    html = out.read_text()
    assert "__nnsvg_ready" in html
    assert "__nnsvg_export_svg" in html
    assert "renderNNSVG" in html or "renderFCNN" in html


def test_render_html_with_cnn(tmp_path: Path):
    out = tmp_path / "cnn.html"
    nsx.save_network_html(out, _small_cnn_spec())
    html = out.read_text()
    assert "renderNNSVG" in html or "renderLeNet" in html


def test_export_paper_json_roundtrip(tmp_path: Path):
    out = tmp_path / "paper.json"
    nsx.save_paper_json(out, _small_cnn_spec())
    data = json.loads(out.read_text())
    assert data["model_name"] == "tiny_cnn"
    assert data["recommended_family"] == "lenet"
    assert data["has_skip_connections"] is False


def test_export_debug_json_roundtrip(tmp_path: Path):
    out = tmp_path / "debug.json"
    nsx.save_debug_json(out, _small_cnn_spec())
    data = json.loads(out.read_text())
    assert "layers" in data
    assert all("attributes" in lay for lay in data["layers"])


def test_export_nnsvg_spec_roundtrip(tmp_path: Path):
    spec = nsx.build_nnsvg_spec(_mlp_spec())
    out = tmp_path / "spec.json"
    nsx.save_nnsvg_spec(out, spec)
    data = json.loads(out.read_text())
    assert data["family"] == "fcnn"


def test_summarize_text():
    s = nsx.summarize_model(_mlp_spec())
    assert "tiny_mlp" in s
    assert "fc1" in s


def test_summarize_markdown():
    s = nsx.summarize_model(_mlp_spec(), format="markdown")
    assert s.startswith("# tiny_mlp")


def test_recommend_view():
    info = nsx.recommend_view(_mlp_spec())
    assert info["family"] == "fcnn"
    assert "confidence" in info


def test_manual_spec_from_file(tmp_path: Path):
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(_mlp_spec()))
    arch = nsx.parse_model(str(p))
    assert arch.model_name == "tiny_mlp"


def test_svg_export_needs_browser(tmp_path: Path):
    """If Playwright is not available, SVG export must raise a helpful error."""
    from neuroschemax.visualization.nnsvg_runtime import is_playwright_available
    if is_playwright_available():
        pytest.skip("Playwright is installed; skipping negative test")
    with pytest.raises(nsx.BrowserNotAvailableError):
        nsx.save_network_svg(tmp_path / "out.svg", _mlp_spec())
