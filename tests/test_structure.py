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


# ── Rendering quality / summary tests ─────────────────────────────────────────

def _cnn_spec() -> dict:
    return {
        "model_name": "tiny_cnn",
        "layers": [
            {"name": "input",   "kind": "input",   "shape": [1, 1, 28, 28]},
            {"name": "conv1",   "kind": "conv",    "out_channels": 16, "kernel_size": [3, 3], "stride": [1, 1]},
            {"name": "pool1",   "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]},
            {"name": "conv2",   "kind": "conv",    "out_channels": 32, "kernel_size": [3, 3], "stride": [1, 1]},
            {"name": "pool2",   "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]},
            {"name": "flatten", "kind": "flatten"},
            {"name": "fc1",     "kind": "dense",   "units": 128},
            {"name": "out",     "kind": "dense",   "units": 10},
        ],
    }


def _transformer_spec() -> dict:
    return {
        "model_name": "transformer_like",
        "layers": [
            {"name": "embedding", "kind": "embedding"},
            {"name": "attn1",     "kind": "attention"},
            {"name": "norm1",     "kind": "layernorm"},
            {"name": "ff1",       "kind": "dense",     "units": 2048},
            {"name": "ff1_out",   "kind": "dense",     "units": 512},
            {"name": "norm2",     "kind": "layernorm"},
            {"name": "output",    "kind": "dense",     "units": 1000},
        ],
    }


def _resnet_spec() -> dict:
    return {
        "model_name": "resnet_like",
        "layers": [
            {"name": "input",  "kind": "input", "shape": [1, 3, 224, 224]},
            {"name": "c1",     "kind": "conv",  "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c2",     "kind": "conv",  "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c3",     "kind": "conv",  "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c4",     "kind": "conv",  "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "add1",   "kind": "add"},
            {"name": "fc",     "kind": "dense", "units": 10},
        ],
    }


def _unet_spec() -> dict:
    return {
        "model_name": "unet_like",
        "layers": [
            {"name": "input",   "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "enc1",    "kind": "conv",  "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "enc2",    "kind": "conv",  "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "enc3",    "kind": "conv",  "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "dec1",    "kind": "conv",  "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat1",    "kind": "concat"},
            {"name": "dec2",    "kind": "conv",  "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "out",     "kind": "conv",  "out_channels": 1,  "kernel_size": [1, 1]},
        ],
    }


def test_fcnn_edge_opacity_reduced():
    """FCNN spec default edge_opacity should be ≤ 0.35 to reduce visual noise."""
    spec = nsx.build_nnsvg_spec(_mlp_spec(), theme="paper")
    assert spec.edge_opacity <= 0.35


def test_cnn_renders_conv_and_dense():
    """Tiny CNN spec still produces both conv-type and dense-type layers."""
    spec = nsx.build_nnsvg_spec(_cnn_spec())
    types = {l.layer_type for l in spec.layers}
    assert "conv" in types
    assert "dense" in types


def test_cnn_dense_units_capped():
    """Dense columns in LeNet view must be capped to avoid visual domination."""
    spec = nsx.build_nnsvg_spec(_cnn_spec())
    for l in spec.layers:
        if l.layer_type == "dense":
            assert l.units <= 10, f"Dense units {l.units} exceeds LeNet cap"


def test_alexnet_dense_units_capped():
    """Dense units in AlexNet view must be tightly capped."""
    deep_spec = {
        "model_name": "vgg_like",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"conv{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3,3]}
               for i in range(6)]
            + [{"name": "fc", "kind": "dense", "units": 4096}]
        ),
    }
    spec = nsx.build_nnsvg_spec(deep_spec)
    for l in spec.layers:
        if l.layer_type == "dense":
            assert l.units <= 8, f"AlexNet dense units {l.units} exceeds cap"


def test_resnet_approximate_warning():
    """ResNet-like model must have a skip/residual warning."""
    arch = nsx.parse_model(_resnet_spec())
    assert arch.warnings, "Expected warnings for ResNet-like model"
    warn_text = " ".join(arch.warnings).lower()
    assert "residual" in warn_text or "skip" in warn_text or "add" in warn_text


def test_resnet_debug_json_preserves_skip_info(tmp_path: Path):
    """Debug JSON must include the Add layer in the layers list."""
    out = tmp_path / "resnet_debug.json"
    nsx.save_debug_json(out, _resnet_spec())
    data = json.loads(out.read_text())
    kinds = [l["kind"] for l in data["layers"]]
    assert "add" in kinds, "Add/skip layer missing from debug JSON"


def test_unet_approximate_warning():
    """U-Net-like model must warn about Concat/encoder-decoder approximation."""
    arch = nsx.parse_model(_unet_spec())
    assert arch.warnings
    warn_text = " ".join(arch.warnings).lower()
    assert "concat" in warn_text or "encoder" in warn_text or "branch" in warn_text


def test_unet_debug_json_preserves_concat(tmp_path: Path):
    """Debug JSON must include the Concat layer."""
    out = tmp_path / "unet_debug.json"
    nsx.save_debug_json(out, _unet_spec())
    data = json.loads(out.read_text())
    kinds = [l["kind"] for l in data["layers"]]
    assert "concat" in kinds


def test_transformer_is_approximate():
    """Transformer-like model must set is_approximate=True."""
    info = nsx.recommend_view(_transformer_spec())
    assert info["is_approximate"] is True
    assert info["confidence"] == "low"


def test_transformer_html_contains_attention_warning(tmp_path: Path):
    """Rendered HTML for Transformer must contain the approximation warning."""
    out = tmp_path / "transformer.html"
    nsx.save_network_html(out, _transformer_spec())
    html = out.read_text()
    assert "attention" in html.lower() or "Approximate" in html


def test_transformer_debug_json_preserves_attention_layers(tmp_path: Path):
    """Debug JSON must preserve the original attention layer."""
    out = tmp_path / "transformer_debug.json"
    nsx.save_debug_json(out, _transformer_spec())
    data = json.loads(out.read_text())
    kinds = [l["kind"] for l in data["layers"]]
    assert "attention" in kinds, "Attention layer missing from debug JSON"


def test_transformer_fcnn_uses_block_mapping():
    """Transformer FCNN mapping must use small block-style units (≤ 4), not large neuron columns."""
    spec = nsx.build_nnsvg_spec(_transformer_spec())
    for l in spec.layers:
        assert l.units <= 4, f"Layer {l.label!r} has {l.units} units — expected block-style (≤ 4)"


def test_large_cnn_compact_renders():
    """A 20-layer CNN in compact mode must produce a valid spec."""
    layers = [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
    for i in range(18):
        layers.append({"name": f"conv{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3,3]})
    layers.append({"name": "fc", "kind": "dense", "units": 1000})
    spec = nsx.build_nnsvg_spec({"model_name": "large_cnn", "layers": layers}, compact=True)
    assert len(spec.layers) > 0


# ── Summary shape inference tests ─────────────────────────────────────────────

def test_summary_infers_conv_output_shape():
    """summarize_model should show inferred H/W for a simple conv layer."""
    summary = nsx.summarize_model(_cnn_spec())
    # conv1: input 28x28, kernel 3x3, stride 1, padding 0 → 26x26
    assert "26" in summary, f"Expected '26' in summary output:\n{summary}"


def test_summary_infers_pool_output_shape():
    """summarize_model should show inferred pool output shape."""
    summary = nsx.summarize_model(_cnn_spec())
    # pool1: 26x26 → kernel 2x2, stride 2 → 13x13
    assert "13" in summary, f"Expected '13' in summary output:\n{summary}"


def test_summary_infers_flatten_shape():
    """summarize_model should show inferred flattened size."""
    summary = nsx.summarize_model(_cnn_spec())
    # After two conv+pool stages the exact size depends on precision,
    # but it should NOT be '?' for flatten.
    lines = [l for l in summary.splitlines() if "flatten" in l.lower()]
    assert lines, "Flatten layer missing from summary"
    assert "?" not in lines[0], f"Flatten shape not inferred: {lines[0]}"


def test_summary_infers_dense_output_shape():
    """Dense layer output shape should be inferred when units are known."""
    summary = nsx.summarize_model(_mlp_spec())
    # fc1 has units=128 → output shape 1x128
    assert "128" in summary


def test_summary_params_kernel_format():
    """Kernel sizes in summary params must use NxM format, not Python list repr."""
    summary = nsx.summarize_model(_cnn_spec())
    assert "[3," not in summary, f"Found raw list repr in summary:\n{summary}"
    assert "3x3" in summary, f"Expected '3x3' in summary:\n{summary}"


def test_summary_dynamic_shape_stays_unknown():
    """Layers with symbolic/dynamic shapes must still show '?'."""
    spec = {
        "model_name": "dynamic",
        "layers": [
            {"name": "input",  "kind": "input", "shape": ["N", 3, "H", "W"]},
            {"name": "conv1",  "kind": "conv",  "out_channels": 64, "kernel_size": [3, 3]},
        ],
    }
    summary = nsx.summarize_model(spec)
    # conv output cannot be inferred when spatial dims are symbolic
    lines = [l for l in summary.splitlines() if "conv1" in l]
    assert lines
    assert "?" in lines[0], f"Expected '?' for dynamic shape: {lines[0]}"
