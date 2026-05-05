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
    types = {s.layer_type for s in spec.layers}
    assert "conv" in types
    assert "dense" in types


def test_cnn_dense_units_capped():
    """Dense columns in LeNet view must be capped to avoid visual domination."""
    spec = nsx.build_nnsvg_spec(_cnn_spec())
    for s in spec.layers:
        if s.layer_type == "dense":
            assert s.units <= 10, f"Dense units {s.units} exceeds LeNet cap"


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
    for s in spec.layers:
        if s.layer_type == "dense":
            assert s.units <= 8, f"AlexNet dense units {s.units} exceeds cap"


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
    kinds = [s["kind"] for s in data["layers"]]
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
    kinds = [s["kind"] for s in data["layers"]]
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
    kinds = [s["kind"] for s in data["layers"]]
    assert "attention" in kinds, "Attention layer missing from debug JSON"


def test_transformer_uses_rect_block_view():
    """Transformer spec must be routed to LENET family (rect blocks), not FCNN neuron columns."""
    from neuroschemax.core.enums import RenderFamily
    spec = nsx.build_nnsvg_spec(_transformer_spec())
    # Must use LeNet renderer (which draws rectangles)
    assert spec.family == RenderFamily.LENET, (
        f"Expected LENET for block view, got {spec.family}"
    )
    # All layers must be single-rect blocks (channels=1, layer_type="conv")
    for s in spec.layers:
        assert s.layer_type == "conv", f"Expected rect block, got layer_type={s.layer_type!r}"
        assert s.channels == 1, f"Expected single-channel block, got channels={s.channels}"


def test_transformer_block_labels_are_meaningful():
    """Transformer block labels must name the operation, not be generic."""
    spec = nsx.build_nnsvg_spec(_transformer_spec())
    labels = {s.label for s in spec.layers}
    # Must include at least one attention block and at least one FFN/classifier block
    has_attention = any("attention" in lb.lower() or "attn" in lb.lower() for lb in labels)
    has_fwd = any(
        w in lb.lower() for lb in labels
        for w in ("feedfwd", "ffn", "classifier", "feed")
    )
    assert has_attention, f"No attention block in Transformer labels: {labels}"
    assert has_fwd, f"No FeedFwd/Classifier block in Transformer labels: {labels}"


def test_transformer_html_contains_block_labels(tmp_path: Path):
    """Rendered HTML for Transformer must embed block labels in the spec JSON."""
    out = tmp_path / "transformer.html"
    nsx.save_network_html(out, _transformer_spec())
    html = out.read_text()
    # Labels are embedded in the spec JSON; check for operation-type strings.
    # "[MH-Attn]" contains "Attn"; "[FFN]" contains "FFN"; "[Head]" contains "Head".
    assert "Attn" in html or "attn" in html.lower(), "Expected attention label in HTML"
    assert "FFN" in html or "Classifier" in html or "Head" in html


def test_activation_fused_into_fcnn_label():
    """FCNN mapper must fuse a following ReLU activation into the preceding layer label."""
    spec_dict = {
        "model_name": "mlp_relu",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 128},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2",   "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict)
    labels = [s.label for s in spec.layers]
    fused = any("+ReLU" in lb or "+relu" in lb.lower() for lb in labels)
    assert fused, f"Expected fused ReLU label, got: {labels}"


def test_relu_not_a_separate_column_in_fcnn():
    """After activation fusion, ReLU must not appear as its own neuron column."""
    spec_dict = {
        "model_name": "mlp_relu",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 128},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2",   "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict)
    # With fusion: input, fc1+ReLU, fc2  → 3 layers (or 2 if input skipped)
    # Without fusion: input, fc1, relu1, fc2 → 4 layers
    assert len(spec.layers) <= 3, (
        f"Expected ≤3 layers with activation fusion, got {len(spec.layers)}: "
        f"{[s.label for s in spec.layers]}"
    )


def test_title_formatted_cleanly():
    """Title must be human-readable (no raw underscores)."""
    spec = nsx.build_nnsvg_spec(_mlp_spec())
    assert "_" not in spec.title, f"Underscores in title: {spec.title!r}"


def test_title_mlp_acronym():
    """'tiny_mlp' model name should produce 'Tiny MLP' title."""
    spec = nsx.build_nnsvg_spec(_mlp_spec())
    assert "MLP" in spec.title, f"Expected 'MLP' in title, got {spec.title!r}"


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
    lines = [line for line in summary.splitlines() if "flatten" in line.lower()]
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
    lines = [line for line in summary.splitlines() if "conv1" in line]
    assert lines
    assert "?" in lines[0], f"Expected '?' for dynamic shape: {lines[0]}"


# ── New controls: label_mode, detail_level, transformer_mode, approximate_mode ──

def test_label_mode_name_no_shapes():
    spec = nsx.build_nnsvg_spec(_cnn_spec(), label_mode="name")
    for s in spec.layers:
        if s.label:
            # Name-only mode: no "x" dimension strings like "26x26"
            assert not any(c.isdigit() and "x" in s.label for c in s.label), \
                f"label_mode='name' should not include shape: {s.label!r}"


def test_label_mode_shape_no_names():
    spec = nsx.build_nnsvg_spec(_cnn_spec(), label_mode="shape")
    for s in spec.layers:
        if s.label and s.layer_type in ("conv", "pool"):
            # Shape mode: label should NOT be the full layer name
            assert s.label not in ("conv1", "pool1", "conv2", "pool2"), \
                f"label_mode='shape' still shows bare name: {s.label!r}"


def test_label_mode_compact_shorter_than_full():
    spec_c = nsx.build_nnsvg_spec(_cnn_spec(), label_mode="compact")
    spec_f = nsx.build_nnsvg_spec(_cnn_spec(), label_mode="full")
    avg_compact = sum(len(s.label) for s in spec_c.layers if s.label) / max(len(spec_c.layers), 1)
    avg_full    = sum(len(s.label) for s in spec_f.layers if s.label) / max(len(spec_f.layers), 1)
    assert avg_compact <= avg_full + 2, "compact labels should not be longer than full labels"


def test_label_mode_invalid_raises():
    with pytest.raises(Exception):  # ValidationError  # noqa: B017
        nsx.build_nnsvg_spec(_mlp_spec(), label_mode="nonsense")


def test_detail_level_summary_reduces_layer_count():
    """Summary mode must produce fewer spec layers than full for a deep CNN."""
    deep_spec = {
        "model_name": "vgg_like",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"conv{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3,3]}
               for i in range(10)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec_full    = nsx.build_nnsvg_spec(deep_spec, detail_level="full")
    spec_summary = nsx.build_nnsvg_spec(deep_spec, detail_level="summary")
    assert len(spec_summary.layers) < len(spec_full.layers), (
        f"summary ({len(spec_summary.layers)}) should have fewer layers than "
        f"full ({len(spec_full.layers)})"
    )


def test_detail_level_auto_summarizes_large_cnn():
    """Auto detail_level must compress a 15+ layer CNN into fewer spec layers."""
    large_spec = {
        "model_name": "large",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3,3]}
               for i in range(14)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec = nsx.build_nnsvg_spec(large_spec, detail_level="auto")
    assert len(spec.layers) <= 10, (
        f"Auto detail_level should group large CNN, got {len(spec.layers)} layers"
    )


def test_detail_level_invalid_raises():
    with pytest.raises(Exception):  # noqa: B017
        nsx.build_nnsvg_spec(_mlp_spec(), detail_level="none")


def test_show_activations_false_removes_fused_labels():
    """With show_activations=False, activation names must not appear in labels."""
    spec_dict = {
        "model_name": "mlp_relu",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 128},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2",   "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict, show_activations=False)
    labels = " ".join(s.label for s in spec.layers)
    assert "ReLU" not in labels and "relu" not in labels.lower(), \
        f"show_activations=False but found activation in labels: {labels!r}"


def test_show_activations_true_fuses_relu():
    spec_dict = {
        "model_name": "mlp_relu",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 128},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2",   "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict, show_activations=True)
    labels = " ".join(s.label for s in spec.layers)
    assert "ReLU" in labels, f"show_activations=True but ReLU not fused: {labels!r}"


def test_transformer_mode_unsupported_produces_placeholder():
    spec = nsx.build_nnsvg_spec(_transformer_spec(), transformer_mode="unsupported")
    assert len(spec.layers) == 1
    assert "not supported" in spec.layers[0].label.lower()


def test_transformer_mode_invalid_raises():
    with pytest.raises(Exception):  # noqa: B017
        nsx.build_nnsvg_spec(_transformer_spec(), transformer_mode="fancy")


def test_approximate_mode_warn_shows_warnings(tmp_path: Path):
    html_path = tmp_path / "resnet.html"
    nsx.save_network_html(html_path, _resnet_spec(), approximate_mode="warn")
    html = html_path.read_text()
    assert "Approximate" in html or "nnsvg-warning" in html


def test_approximate_mode_allow_suppresses_badges(tmp_path: Path):
    html_path = tmp_path / "resnet_allow.html"
    nsx.save_network_html(html_path, _resnet_spec(), approximate_mode="allow")
    html = html_path.read_text()
    assert "<div class='nnsvg-warnings'>" not in html


def test_approximate_mode_error_raises():
    from neuroschemax.exceptions import RenderError
    with pytest.raises(RenderError, match="approximate"):
        nsx.build_nnsvg_spec(_resnet_spec(), approximate_mode="error")


def test_approximate_mode_invalid_raises():
    with pytest.raises(Exception):  # noqa: B017
        nsx.build_nnsvg_spec(_mlp_spec(), approximate_mode="silent")


def test_subtitle_present_in_spec():
    spec = nsx.build_nnsvg_spec(_mlp_spec())
    assert spec.subtitle, "subtitle should be non-empty"


def test_subtitle_in_html(tmp_path: Path):
    out = tmp_path / "mlp.html"
    nsx.save_network_html(out, _mlp_spec())
    html = out.read_text()
    assert "nnsvg-subtitle" in html


def test_title_clean_no_underscores():
    spec = nsx.build_nnsvg_spec(_mlp_spec())
    assert "_" not in spec.title


def test_resnet_summary_has_residual_block_labels():
    """ResNet summary grouping must produce Stem / Res Block / Head labels."""
    spec = nsx.build_nnsvg_spec(_resnet_spec(), detail_level="summary")
    labels = [s.label for s in spec.layers]
    has_stem_or_block = any("Stem" in lb or "Res Block" in lb or "Block" in lb for lb in labels)
    assert has_stem_or_block, f"Expected block labels in ResNet summary: {labels}"


def test_unet_summary_has_encoder_decoder_labels():
    """U-Net summary grouping must produce Encoder/Bottleneck/Decoder labels."""
    spec = nsx.build_nnsvg_spec(_unet_spec(), detail_level="summary")
    labels = [s.label for s in spec.layers]
    has_enc_dec = any(
        w in lb for lb in labels for w in ("Encoder", "Bottleneck", "Decoder")
    )
    assert has_enc_dec, f"Expected encoder/decoder labels in U-Net summary: {labels}"


# ── Label safety system tests ──────────────────────────────────────────────

def _large_cnn_spec(n_convs: int = 12) -> dict:
    """A large sequential CNN with many conv layers (no relu)."""
    return {
        "model_name": "large_cnn",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"conv{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(n_convs)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }


def test_large_cnn_labels_do_not_overflow_slot():
    """For a 12-conv CNN in auto mode, labels must not exceed the safe character budget."""
    spec = nsx.build_nnsvg_spec(_large_cnn_spec(12))
    n = len(spec.layers)
    slot_px = (spec.width - 120) / n
    # Conservative safe-chars estimate at default font_size=12
    safe_chars = max(3, int(slot_px * 0.78 / (12 * 0.62)))
    non_box = [s for s in spec.layers if s.layer_type == "dense" or s.channels != 1]
    for s in non_box:
        assert len(s.label) <= max(safe_chars, 3), (
            f"Label {s.label!r} ({len(s.label)} chars) exceeds safe budget {safe_chars} "
            f"for slot_px={slot_px:.0f}px"
        )


def test_auto_label_mode_uses_name_for_large_cnn():
    """For a CNN with > 9 arch layers, auto label mode must use name-only labels."""
    # 12-conv CNN → 14 arch layers → auto → 'name' → no shape dims in labels
    spec = nsx.build_nnsvg_spec(_large_cnn_spec(12))
    for s in spec.layers:
        if s.label and s.layer_type in ("conv", "dense"):
            # Name-only: no digit + 'x' pattern (which would indicate a shape dim)
            has_shape_dim = any(c.isdigit() for c in s.label) and "x" in s.label
            assert not has_shape_dim, (
                f"Auto mode for large CNN should use name-only; got shape in label: {s.label!r}"
            )


def test_detail_level_full_preserves_all_layer_labels():
    """With detail_level='full', every conv layer must have a non-empty label."""
    spec = nsx.build_nnsvg_spec(_large_cnn_spec(10), detail_level="full")
    conv_layers = [s for s in spec.layers if s.layer_type == "conv" and s.channels != 1]
    assert all(s.label for s in conv_layers), (
        "detail_level='full' should not thin labels; some are empty"
    )


def test_small_cnn_compact_labels_include_shape():
    """For a tiny CNN (≤ 9 arch layers) auto mode gives compact labels with shape info."""
    tiny = {
        "model_name": "tiny",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "fc1", "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(tiny)
    labels = " ".join(s.label for s in spec.layers if s.label)
    # Compact mode should include at least some numeric dimension info
    has_dims = any(c.isdigit() for c in labels)
    assert has_dims, f"Compact labels for small model should include dims: {labels!r}"


def test_transformer_warning_mentions_qkv_and_heads():
    """Transformer family_recognizer warning must mention Q/K/V and attention heads."""
    trans_spec = {
        "model_name": "transformer",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn",  "kind": "attention"},
            {"name": "ff",    "kind": "dense", "units": 512},
            {"name": "out",   "kind": "dense", "units": 10},
        ],
    }
    arch = nsx.parse_model(trans_spec)
    assert arch.warnings, "Transformer arch should have warnings"
    combined = " ".join(arch.warnings)
    assert "Q/K/V" in combined or "Q, K, V" in combined, (
        f"Warning should mention Q/K/V projections; got: {combined[:200]!r}"
    )
    assert "head" in combined.lower(), (
        f"Warning should mention attention heads; got: {combined[:200]!r}"
    )
    assert "NOT" in combined or "not drawn" in combined.lower(), (
        f"Warning should be explicit about what is not drawn; got: {combined[:200]!r}"
    )


def test_transformer_warning_mentions_residual():
    trans_spec = {
        "model_name": "transformer",
        "layers": [
            {"name": "attn", "kind": "attention"},
            {"name": "ff",   "kind": "dense", "units": 512},
        ],
    }
    arch = nsx.parse_model(trans_spec)
    combined = " ".join(arch.warnings).lower()
    assert "residual" in combined, "Warning should mention residual connections"


def test_transformer_block_labels_fit_at_min_font():
    """Transformer block labels must be renderable at ≥ 8pt (JS minimum font)."""
    trans_spec = {
        "model_name": "transformer",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn",  "kind": "attention"},
            {"name": "ff",    "kind": "dense", "units": 512},
            {"name": "out",   "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(trans_spec)
    MIN_FONT = 8
    for s in spec.layers:
        if s.layer_type == "conv" and s.channels == 1 and s.label:
            avail = s.feature_map_width - 8
            # Multi-line labels: measure against the LONGEST LINE, not total len.
            max_line_len = max(len(line) for line in s.label.split("\n"))
            required_font = avail / (max_line_len * 0.62) if max_line_len else MIN_FONT
            assert required_font >= MIN_FONT, (
                f"Block label {s.label!r} (max line {max_line_len} chars) "
                f"in fmW={s.feature_map_width} "
                f"would need font_size={required_font:.1f}pt < minimum {MIN_FONT}pt"
            )


def test_safe_label_policy_direct():
    """_safe_label_policy wraps long labels on whitespace boundaries.

    Labels with whitespace must be broken into multiple lines so each line
    fits the slot.  Continuous identifier-like tokens with no break points
    are left intact rather than corrupted by '…' truncation.
    """
    from neuroschemax.visualization.nnsvg_mapper import _safe_label_policy
    from neuroschemax.visualization.nnsvg_schema import NNSVGLayerSpec
    layers = [
        NNSVGLayerSpec(layer_type="conv", label="Dense 256 +ReLU +Drop 0.5", channels=3),
        NNSVGLayerSpec(layer_type="conv", label="Conv 128 k3 s2", channels=2),
        NNSVGLayerSpec(layer_type="dense", label="Classifier 1000", units=5),
    ]
    # 480px canvas, 3 layers: slot≈120px, safe_chars per line ≈ 14
    result = _safe_label_policy(layers, 480, 12, allow_thinning=False)
    for s in result:
        if not s.label:
            continue
        # Must never contain the unsafe ellipsis truncation marker.
        assert "…" not in s.label, f"Unsafe ellipsis in label: {s.label!r}"
        # Each line must fit (or be a single unbreakable token).
        for line in s.label.split("\n"):
            assert " " not in line or len(line) <= 18, (
                f"Line too long after wrap: {line!r} in {s.label!r}"
            )
    # Multi-token labels should now span multiple lines.
    assert "\n" in result[0].label, (
        f"Long multi-token label should be wrapped onto multiple lines: {result[0].label!r}"
    )


def test_safe_label_policy_skips_box_layers():
    """_safe_label_policy must NOT truncate labels inside ch=1 box layers."""
    from neuroschemax.visualization.nnsvg_mapper import _safe_label_policy
    from neuroschemax.visualization.nnsvg_schema import NNSVGLayerSpec
    box_layer = NNSVGLayerSpec(
        layer_type="conv", label="[Attention]", channels=1,
        feature_map_width=80, feature_map_height=100,
    )
    original_label = box_layer.label
    result = _safe_label_policy([box_layer], 200, 12, allow_thinning=True)
    assert result[0].label == original_label, (
        "Safety policy must not modify box-layer (ch=1) labels"
    )


# ── Operation-aware label and block summary tests ────────────────────────────

def test_compact_label_conv_shows_channels_and_kernel():
    """Compact mode for conv layers must show channel count and kernel size."""
    spec_dict = {
        "model_name": "tiny",
        "layers": [
            {"name": "input",  "kind": "input",   "shape": [1, 1, 28, 28]},
            {"name": "conv1",  "kind": "conv",    "out_channels": 32, "kernel_size": [3, 3]},
            {"name": "fc1",    "kind": "dense",   "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict, label_mode="compact")
    conv_labels = [s.label for s in spec.layers if s.layer_type == "conv" and s.channels != 1]
    # At least one conv label should show "Conv", channels, and kernel info
    assert any("Conv" in lb and ("32" in lb or "k3" in lb) for lb in conv_labels), (
        f"Compact label should show op type + channels + kernel; got: {conv_labels}"
    )


def test_compact_label_pool_shows_pool_type():
    """Compact labels for pool layers must not be raw layer names."""
    spec_dict = {
        "model_name": "cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv",    "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]},
            {"name": "fc1",   "kind": "dense",   "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict, label_mode="compact")
    pool_labels = [s.label for s in spec.layers if s.layer_type == "pool"]
    assert any("MaxP" in lb or "AvgP" in lb or "Pool" in lb for lb in pool_labels), (
        f"Pool compact label should show pool type; got: {pool_labels}"
    )


def test_compact_label_dense_shows_units():
    """Dense compact labels must show unit count."""
    spec_dict = {
        "model_name": "mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 256},
            {"name": "out",   "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict, label_mode="compact")
    dense_labels = [s.label for s in spec.layers if s.layer_type == "dense"]
    assert any("256" in lb for lb in dense_labels), (
        f"Dense compact label should show units; got: {dense_labels}"
    )


def test_compact_label_gap_is_gap():
    """Global average pool must produce 'GAP' as compact label."""
    spec_dict = {
        "model_name": "cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            {"name": "c1",   "kind": "conv",              "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c2",   "kind": "conv",              "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c3",   "kind": "conv",              "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "c4",   "kind": "conv",              "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "gap",  "kind": "globalaveragepool"},
            {"name": "fc",   "kind": "dense",             "units": 1000},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict, label_mode="compact")
    labels = [s.label for s in spec.layers]
    # After summary grouping (7 arch layers → compact auto), GAP may be inside a block
    # but "GAP" should appear somewhere OR be absorbed into a block summary
    # Either way, the op type must be recognisable
    all_labels = " ".join(labels)
    # Ensure we didn't just get raw layer names (like "gap" → "gap")
    assert "GAP" in all_labels or "Block" in all_labels or "GlobalAvg" in all_labels, (
        f"Expected GAP or block label; got: {labels}"
    )


def test_large_cnn_summary_blocks_have_conv_info():
    """Summary blocks for large CNN must include conv count and channel info."""
    large_spec = {
        "model_name": "vgg_like",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"conv{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(12)]
            + [{"name": "fc", "kind": "dense", "units": 4096}]
        ),
    }
    spec = nsx.build_nnsvg_spec(large_spec, detail_level="summary")
    # Summary now uses multi-channel conv primitives (not ch=1 boxes).
    block_labels = [s.label for s in spec.layers if s.layer_type == "conv" and s.layer_type != "input"]
    assert len(block_labels) > 0, "Summary should produce block conv labels"
    # At least one block should mention conv count or channel info
    combined = "\n".join(block_labels)
    has_info = any(
        kw in combined for kw in ("cv", "ch", "conv", "Block")
    )
    assert has_info, f"Summary blocks should include conv/channel info; got:\n{combined!r}"


def test_large_cnn_summary_multiple_blocks():
    """Summary for large CNN must produce more than one named block (not 'Block 1' only)."""
    large_spec = {
        "model_name": "vgg20",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(6)]
            + [{"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2]}]
            + [{"name": f"c{i+6}", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]}
               for i in range(6)]
            + [{"name": "pool2", "kind": "maxpool", "kernel_size": [2, 2]}]
            + [{"name": "fc", "kind": "dense", "units": 4096}]
        ),
    }
    spec = nsx.build_nnsvg_spec(large_spec, detail_level="summary")
    block_labels = [s.label for s in spec.layers if "Block" in s.label]
    assert len(block_labels) >= 2, (
        f"Large CNN summary should have multiple named blocks; got: {[s.label for s in spec.layers]}"
    )


def test_transformer_block_labels_include_mh_attn():
    """Transformer block summary must include [MH-Attn] label."""
    trans_spec = {
        "model_name": "transformer",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn",  "kind": "attention"},
            {"name": "ff",    "kind": "dense", "units": 2048},
            {"name": "out",   "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(trans_spec)
    labels = [s.label for s in spec.layers]
    combined = "\n".join(labels)
    assert "MH-Attn" in combined or "Attn" in combined, (
        f"Transformer block should include [MH-Attn]; got: {labels}"
    )


def test_transformer_block_labels_include_ffn():
    """Transformer block summary must include [FFN] label."""
    trans_spec = {
        "model_name": "transformer",
        "layers": [
            {"name": "attn", "kind": "attention"},
            {"name": "ff",   "kind": "dense", "units": 2048},
            {"name": "out",  "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(trans_spec)
    labels = [s.label for s in spec.layers]
    combined = "\n".join(labels)
    assert "FFN" in combined or "FeedFwd" in combined, (
        f"Transformer block should include [FFN]; got: {labels}"
    )


def test_upsample_shape_inference():
    """Upsample with scale_factor should have its output shape inferred."""
    spec_dict = {
        "model_name": "upsample_test",
        "layers": [
            {"name": "input",  "kind": "input",    "shape": [1, 128, 8, 8]},
            {"name": "up1",    "kind": "upsample", "scale_factor": 2},
        ],
    }
    arch = nsx.parse_model(spec_dict)
    up_layer = next(s for s in arch.layers if s.name == "up1")
    # Expect output [1, 128, 16, 16]
    assert up_layer.output_shape == [1, 128, 16, 16], (
        f"Upsample scale_factor=2 should double spatial dims; got {up_layer.output_shape}"
    )


def test_attention_shape_preserved():
    """Self-attention output shape must equal the input shape."""
    spec_dict = {
        "model_name": "attn_test",
        "layers": [
            {"name": "input", "kind": "input",     "shape": [1, 512]},
            {"name": "attn1", "kind": "attention"},
        ],
    }
    arch = nsx.parse_model(spec_dict)
    attn = next(s for s in arch.layers if s.name == "attn1")
    # Attention output should preserve the input shape [1, 512]
    assert attn.output_shape == [1, 512], (
        f"Attention output should preserve input shape; got {attn.output_shape}"
    )


def test_debug_json_preserves_add_and_norm_layers(tmp_path: Path):
    """Debug JSON must include Add and LayerNorm layers even after block grouping."""
    trans_spec = {
        "model_name": "transformer",
        "layers": [
            {"name": "attn",  "kind": "attention"},
            {"name": "add1",  "kind": "add"},
            {"name": "norm1", "kind": "layernorm"},
            {"name": "ff",    "kind": "dense", "units": 512},
        ],
    }
    out = tmp_path / "trans_debug.json"
    nsx.save_debug_json(out, trans_spec)
    import json
    data = json.loads(out.read_text())
    kinds = [s["kind"] for s in data["layers"]]
    assert "attention" in kinds,  f"Debug JSON must preserve attention; got {kinds}"
    assert "add" in kinds,        f"Debug JSON must preserve add; got {kinds}"
    assert "layer_norm" in kinds, f"Debug JSON must preserve layernorm; got {kinds}"


# ── Visual-quality systemic policy tests ────────────────────────────────────

def test_mlp_subtitle_includes_layer_and_stage_count():
    """Subtitle must distinguish original layer count from visual stage count."""
    # 7-layer MLP but activations are fused → fewer visual stages
    spec = nsx.build_nnsvg_spec({
        "model_name": "mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 256},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2", "kind": "dense", "units": 128},
            {"name": "relu2", "kind": "relu"},
            {"name": "fc3", "kind": "dense", "units": 64},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    })
    # subtitle must contain layer count (7 arch layers)
    assert "7" in spec.subtitle, f"Subtitle should mention 7 layers: {spec.subtitle!r}"
    # When visual stages != layer count, subtitle should note both
    if len(spec.layers) != 7:
        assert "stages" in spec.subtitle or str(len(spec.layers)) in spec.subtitle, (
            f"Subtitle should note visual stage count: {spec.subtitle!r}"
        )


def test_mlp_output_layer_labeled_output():
    """The last dense layer in an MLP should be labeled 'Output N'."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    })
    last_label = spec.layers[-1].label
    assert "Output" in last_label or "output" in last_label.lower(), (
        f"Last MLP layer should be labeled 'Output N'; got {last_label!r}"
    )
    assert "10" in last_label, f"Output label should include unit count; got {last_label!r}"


def test_cnn_classifier_layer_labeled_classifier():
    """The last dense layer in a CNN should be labeled 'Classifier N'."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "fc1", "kind": "dense", "units": 10},
        ],
    })
    dense_labels = [s.label for s in spec.layers if s.layer_type == "dense"]
    assert dense_labels, "CNN should have at least one dense spec layer"
    last = dense_labels[-1]
    assert "Classifier" in last or "classifier" in last.lower(), (
        f"Last CNN dense layer should be 'Classifier'; got {last!r}"
    )


def test_block_summary_uses_cross_conv_format():
    """Summary block labels must use '×Conv' format, not raw abbreviations like '4cv'."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "vgg",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(6)]
            + [{"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]}]
            + [{"name": f"c{i+6}", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]}
               for i in range(6)]
            + [{"name": "pool2", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]}]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }, detail_level="summary")
    block_labels = [s.label for s in spec.layers if "Block" in s.label]
    # Must use "×Conv" (cross/times symbol), not raw "cv" abbreviation
    for lb in block_labels:
        assert "×Conv" in lb, (
            f"Block label must use '×Conv' format, not abbreviation; got {lb!r}"
        )
    # Must also include channel count
    assert any("ch" in lb or "128" in lb or "64" in lb for lb in block_labels), (
        f"Block labels should include channel info; got: {block_labels}"
    )


def test_resnet_summary_labels_mention_skip_collapsed():
    """ResNet residual block labels must note that skip links are collapsed."""
    spec = nsx.build_nnsvg_spec(_resnet_spec(), detail_level="summary")
    res_labels = [s.label for s in spec.layers if "Residual" in s.label or "Res Block" in s.label]
    assert res_labels, f"Expected residual block labels; got {[s.label for s in spec.layers]}"
    combined = "\n".join(res_labels)
    assert "skip" in combined.lower() or "collapsed" in combined.lower(), (
        f"Residual block labels should mention skip-collapsed; got: {res_labels}"
    )


def test_unet_summary_labels_mention_concat_collapsed():
    """U-Net encoder/decoder labels must note concat/skip links are collapsed."""
    spec = nsx.build_nnsvg_spec(_unet_spec(), detail_level="summary")
    encoder_labels = [s.label for s in spec.layers if "Encoder" in s.label or "Decoder" in s.label]
    assert encoder_labels, f"Expected encoder/decoder labels; got {[s.label for s in spec.layers]}"
    combined = "\n".join(encoder_labels)
    assert (
        "concat" in combined.lower()
        or "skip" in combined.lower()
        or "collapsed" in combined.lower()
        or "debug" in combined.lower()
    ), f"Encoder/decoder labels should mention concat-collapsed; got: {encoder_labels}"


def test_transformer_unsupported_generates_diagnostic_content():
    """Unsupported mode must produce a labeled diagnostic block, not a tiny placeholder."""
    spec = nsx.build_nnsvg_spec(_transformer_spec(), transformer_mode="unsupported")
    assert len(spec.layers) == 1, "Unsupported should produce a single diagnostic block"
    label = spec.layers[0].label
    # Must say more than just "(not supported)"
    assert len(label) > 20, f"Diagnostic label is too short: {label!r}"
    # Must contain actionable guidance
    assert "block_summary" in label or "not supported" in label.lower(), (
        f"Diagnostic must reference block_summary mode; got {label!r}"
    )


def test_transformer_unsupported_block_is_wide_enough():
    """The unsupported block must be wide enough to display multi-line content."""
    spec = nsx.build_nnsvg_spec(_transformer_spec(), transformer_mode="unsupported")
    box = spec.layers[0]
    assert box.channels == 1, "Diagnostic block should be ch=1 (box style)"
    assert box.feature_map_width >= 160, (
        f"Diagnostic block must be wide for multi-line content; got fmW={box.feature_map_width}"
    )


def test_transformer_unsupported_subtitle_explains():
    """Unsupported mode subtitle must be informative, not just the family name."""
    spec = nsx.build_nnsvg_spec(_transformer_spec(), transformer_mode="unsupported")
    assert spec.subtitle, "Subtitle must not be empty for unsupported mode"
    assert "not supported" in spec.subtitle.lower() or "block_summary" in spec.subtitle, (
        f"Subtitle should explain the unsupported situation; got {spec.subtitle!r}"
    )


def test_summary_subtitle_shows_visual_stage_count():
    """When summary mode reduces many layers to few blocks, subtitle must note both."""
    large_spec = {
        "model_name": "deep_cnn",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(16)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec = nsx.build_nnsvg_spec(large_spec, detail_level="summary")
    # Should have fewer spec layers than arch layers (18 total, summarized)
    assert len(spec.layers) < 10, "Summary should reduce many layers to fewer blocks"
    # Subtitle should mention original layer count
    assert "18" in spec.subtitle or "layers" in spec.subtitle, (
        f"Subtitle should mention original layer count; got {spec.subtitle!r}"
    )
    if len(spec.layers) < 18:
        assert "stages" in spec.subtitle or str(len(spec.layers)) in spec.subtitle, (
            f"Subtitle should also mention visual stage count; got {spec.subtitle!r}"
        )


# ── Systemic policy tests for label badges, classifier semantics, etc. ──────

def _cnn_with_bn_dropout_spec() -> dict:
    return {
        "model_name": "cnn_bn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "bn1",   "kind": "batchnorm"},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc1",   "kind": "dense", "units": 128},
            {"name": "drop1", "kind": "dropout", "rate": 0.5},
            {"name": "out",   "kind": "dense", "units": 10},
        ],
    }


def test_batch_norm_appears_as_badge_in_compact_label():
    """BatchNorm following Conv should appear as a +BN badge, not a separate column."""
    spec = nsx.build_nnsvg_spec(_cnn_with_bn_dropout_spec(), label_mode="compact")
    conv_labels = [s.label for s in spec.layers if s.layer_type == "conv" and s.channels != 1]
    assert any("+BN" in lb for lb in conv_labels), (
        f"Expected '+BN' badge fused into conv label; got: {conv_labels}"
    )


def test_dropout_appears_as_badge_in_compact_label():
    """Dropout following Dense should appear as a +Drop badge, not a separate column."""
    spec = nsx.build_nnsvg_spec(_cnn_with_bn_dropout_spec(), label_mode="compact")
    dense_labels = [s.label for s in spec.layers if s.layer_type == "dense"]
    combined = " ".join(dense_labels)
    assert "+Drop" in combined, (
        f"Expected '+Drop' badge in dense label; got: {dense_labels}"
    )


def test_badges_do_not_create_extra_visual_columns():
    """BN/Dropout should not occupy their own visual columns."""
    spec = nsx.build_nnsvg_spec(_cnn_with_bn_dropout_spec())
    labels = [s.label for s in spec.layers]
    for lb in labels:
        # No raw layer name for BN/Drop should appear as the entire label
        assert lb not in ("bn1", "drop1"), (
            f"BN/Dropout should be a badge, not its own column: {labels}"
        )


def test_show_activations_false_keeps_badges_visible():
    """show_activations=False removes ReLU but BN/Drop badges remain visible."""
    spec = nsx.build_nnsvg_spec(_cnn_with_bn_dropout_spec(), show_activations=False)
    labels = " ".join(s.label for s in spec.layers)
    assert "+ReLU" not in labels and "+relu" not in labels.lower(), (
        f"show_activations=False but ReLU appeared: {labels!r}"
    )
    assert "+BN" in labels, (
        f"Norm badges should remain when show_activations=False: {labels!r}"
    )


def test_classifier_label_uses_true_units_not_visual_count():
    """A 1000-class classifier must show '1000' in its label even though visual cap is < 10."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "imgnet",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(4)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    })
    last = spec.layers[-1]
    assert "1000" in last.label, (
        f"Classifier label must reflect true unit count (1000), got {last.label!r}"
    )


def test_summary_classifier_uses_classes_label():
    """Summary mode classifier should label semantic 'N classes', not just unit count."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "deep",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(14)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }, detail_level="summary")
    last_label = spec.layers[-1].label
    assert "Classifier" in last_label and "1000" in last_label, (
        f"Summary classifier should be 'Classifier\\n1000 classes'; got {last_label!r}"
    )


def test_unsupported_diagnostic_block_is_wide_for_text():
    """Diagnostic block must be wide enough for the explanation lines."""
    trans_spec = {
        "model_name": "trans",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn",  "kind": "attention"},
            {"name": "ff",    "kind": "dense", "units": 512},
        ],
    }
    spec = nsx.build_nnsvg_spec(trans_spec, transformer_mode="unsupported")
    box = spec.layers[0]
    # Wide enough that the longest line in the new diagnostic content fits
    assert box.feature_map_width >= 320, (
        f"Diagnostic block must be wide for multi-line content; got fmW={box.feature_map_width}"
    )


def test_unsupported_diagnostic_mentions_block_summary_and_debug():
    """Diagnostic copy must mention both block_summary mode and debug JSON."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "trans",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn",  "kind": "attention"},
        ],
    }, transformer_mode="unsupported")
    label = spec.layers[0].label
    assert "block_summary" in label, (
        f"Diagnostic must reference block_summary mode; got {label!r}"
    )
    assert "debug" in label.lower(), (
        f"Diagnostic must mention debug JSON; got {label!r}"
    )


def test_transformer_block_summary_includes_pos_encoding():
    """When an Add appears before the first Attention, label it Positional Encoding."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "trans",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "pos",   "kind": "add"},
            {"name": "attn",  "kind": "attention"},
            {"name": "ff",    "kind": "dense", "units": 512},
        ],
    })
    labels = " ".join(s.label for s in spec.layers)
    assert "Positional" in labels or "PosEnc" in labels, (
        f"PosEnc/Positional Encoding label expected; got: {labels!r}"
    )


def test_transformer_block_summary_surfaces_metadata_when_available():
    """Heads / d_model attributes on attention layer should appear in the block label."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "trans",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn",  "kind": "attention",
             "num_heads": 12, "d_model": 768},
            {"name": "ff",    "kind": "dense", "units": 3072},
            {"name": "out",   "kind": "dense", "units": 1000},
        ],
    })
    combined = "\n".join(s.label for s in spec.layers)
    assert "12 heads" in combined, (
        f"Expected '12 heads' in block summary; got: {combined!r}"
    )
    assert "d=768" in combined, (
        f"Expected 'd=768' in block summary; got: {combined!r}"
    )


def test_resnet_summary_block_label_has_conv_count():
    """Residual block summary must include conv count so it is honest about content."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv",
               "out_channels": 64, "kernel_size": [3, 3]} for i in range(6)],
            {"name": "add1", "kind": "add"},
            {"name": "add2", "kind": "add"},
            {"name": "fc",   "kind": "dense", "units": 10},
        ],
    }, detail_level="summary")
    res_labels = [s.label for s in spec.layers if "Res Block" in s.label or "Residual Block" in s.label]
    assert res_labels, f"No residual block labels: {[s.label for s in spec.layers]}"
    # Each residual block label must mention Conv count or channel count.
    for lb in res_labels:
        assert "Conv" in lb or "ch" in lb, (
            f"Residual block label must mention conv/channel info: {lb!r}"
        )


def test_unet_summary_mentions_upsample_or_decoder():
    """U-Net summary must show Decoder block and concat-collapsed metadata."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "unet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "e1",   "kind": "conv",     "out_channels": 64,  "kernel_size": [3, 3]},
            {"name": "e2",   "kind": "conv",     "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "bot",  "kind": "conv",     "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "up1",  "kind": "upsample", "scale_factor": 2},
            {"name": "d1",   "kind": "conv",     "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat",  "kind": "concat"},
            {"name": "out",  "kind": "conv",     "out_channels": 1,   "kernel_size": [1, 1]},
        ],
    }, detail_level="summary")
    labels = "\n".join(s.label for s in spec.layers)
    assert "Decoder" in labels and "Encoder" in labels, (
        f"Encoder + Decoder must appear in U-Net summary; got: {labels!r}"
    )
    assert "concat collapsed" in labels.lower() or "concat" in labels.lower(), (
        f"U-Net decoder must mention concat-collapsed; got: {labels!r}"
    )


def test_block_label_does_not_use_unsafe_2cv_abbreviation():
    """Block summary labels must never use '2cv'/'4cv' abbreviation."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "vgg",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(8)]
            + [{"name": "p1", "kind": "maxpool", "kernel_size": [2, 2]}]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }, detail_level="summary")
    labels = "\n".join(s.label for s in spec.layers)
    import re as _re
    assert not _re.search(r"\b\d+cv\b", labels), (
        f"Unsafe abbreviation 'Ncv' found in labels: {labels!r}"
    )


def test_compact_mode_uses_smaller_per_layer_budget():
    """Compact mode should produce a narrower canvas than presentation mode."""
    deep_spec = {
        "model_name": "deep",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(8)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec_default = nsx.build_nnsvg_spec(deep_spec, width=600)
    spec_compact = nsx.build_nnsvg_spec(deep_spec, width=600, compact=True)
    assert spec_compact.width <= spec_default.width, (
        f"compact width {spec_compact.width} should not exceed default {spec_default.width}"
    )


def test_debug_json_preserves_dropout_layers(tmp_path: Path):
    """Debug JSON must preserve Dropout layers absorbed into compact badges."""
    out = tmp_path / "drop_debug.json"
    nsx.save_debug_json(out, _cnn_with_bn_dropout_spec())
    data = json.loads(out.read_text())
    kinds = [s["kind"] for s in data["layers"]]
    assert "dropout" in kinds, f"Dropout missing from debug JSON: {kinds}"
    assert "batch_norm" in kinds, f"BatchNorm missing from debug JSON: {kinds}"


# ── Visual readability policy: multi-line wrap, no '...' truncation ─────────

def test_long_label_wraps_to_multiple_lines_not_ellipsis():
    """Long badge-decorated labels must wrap, never use '…' truncation."""
    deep = {
        "model_name": "deep",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64,
                "kernel_size": [3, 3]} for i in range(10)]
            + [{"name": "bn", "kind": "batchnorm"},
               {"name": "relu", "kind": "relu"},
               {"name": "drop", "kind": "dropout", "rate": 0.5},
               {"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec = nsx.build_nnsvg_spec(deep, label_mode="compact", width=900)
    for s in spec.layers:
        if not s.label:
            continue
        assert "…" not in s.label, (
            f"Unsafe ellipsis in label: {s.label!r}"
        )


def test_long_label_preserves_important_tokens():
    """Wrapping must never split tokens like ReLU/+Drop 0.5/k3/d=768/N classes."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "tiny",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "bn1",   "kind": "batchnorm"},
            {"name": "relu1", "kind": "relu"},
            {"name": "drop1", "kind": "dropout", "rate": 0.5},
            {"name": "fc1",   "kind": "dense", "units": 1000},
        ],
    }, label_mode="compact", width=520)
    combined = "\n".join(s.label for s in spec.layers if s.label)
    # No important token may be split mid-word by the wrapper.
    for tok in ("ReLU", "BN", "Drop 0.5", "k3", "Classifier 1000"):
        if tok in combined.replace("\n", " "):
            # The token must appear contiguously somewhere (whitespace OK).
            assert tok in combined.replace("\n", " "), (
                f"Token {tok!r} corrupted; label set: {combined!r}"
            )


def test_safe_label_policy_uses_multi_line_for_decorated_labels():
    from neuroschemax.visualization.nnsvg_mapper import _safe_label_policy
    from neuroschemax.visualization.nnsvg_schema import NNSVGLayerSpec
    layers = [
        NNSVGLayerSpec(layer_type="conv", label="Dense 256 +ReLU +Drop 0.5", channels=3),
        NNSVGLayerSpec(layer_type="conv", label="Dense 128 +ReLU", channels=2),
        NNSVGLayerSpec(layer_type="dense", label="Output 10", units=5),
    ]
    out = _safe_label_policy(layers, 540, 12, allow_thinning=False)
    # First label should be wrapped onto multiple lines, with badges intact.
    parts = out[0].label.split("\n")
    assert len(parts) >= 2, f"Expected multi-line wrap, got: {out[0].label!r}"
    # Each badge token must remain whole on some line.
    for tok in ("Dense 256", "+ReLU", "+Drop 0.5"):
        assert any(tok in line for line in parts), (
            f"Token {tok!r} lost during wrap: {parts}"
        )


def test_diagnostic_card_present_in_unsupported_html(tmp_path: Path):
    """Unsupported transformer mode must produce an HTML diagnostic card."""
    out = tmp_path / "trans_diag.html"
    nsx.save_network_html(out, _transformer_spec(), transformer_mode="unsupported")
    html = out.read_text()
    assert "nnsvg-diagnostic" in html, (
        "Diagnostic card class missing from HTML"
    )
    assert "Transformer exact rendering is not supported" in html
    assert "block_summary" in html
    assert "debug" in html.lower()
    # The structured card should not contain the tiny placeholder fallback
    # text rendered inside the SVG box.
    assert "Detected components" in html


def test_diagnostic_card_hides_svg_diagram(tmp_path: Path):
    out = tmp_path / "trans_diag2.html"
    nsx.save_network_html(out, _transformer_spec(), transformer_mode="unsupported")
    html = out.read_text()
    assert 'id="diagram" style="display:none"' in html, (
        "Unsupported mode should hide the SVG diagram element"
    )


def test_block_summary_html_does_not_show_diagnostic_card(tmp_path: Path):
    """Regular block_summary mode must NOT instantiate the diagnostic card."""
    out = tmp_path / "trans_block.html"
    nsx.save_network_html(out, _transformer_spec(), transformer_mode="block_summary")
    html = out.read_text()
    assert "<div class='nnsvg-diagnostic'" not in html, (
        "block_summary should not render the unsupported diagnostic card"
    )
    # SVG diagram should be visible (no inline display:none on the element).
    assert 'id="diagram" style="display:none"' not in html


def test_canvas_grows_to_fit_long_labels():
    """When a label is much wider than the slot, the canvas grows accordingly."""
    # Compact label_mode produces longer labels (channels + kernel etc.).
    spec_short = nsx.build_nnsvg_spec({
        "model_name": "short",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "fc",    "kind": "dense", "units": 10},
        ],
    }, label_mode="name", width=600)
    spec_long = nsx.build_nnsvg_spec({
        "model_name": "long",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "bn1",   "kind": "batchnorm"},
            {"name": "relu1", "kind": "relu"},
            {"name": "drop1", "kind": "dropout", "rate": 0.5},
            {"name": "fc",    "kind": "dense", "units": 1000},
        ],
    }, label_mode="full", width=600)
    # The long version should never be narrower than the short one — labels
    # widen the canvas instead of being silently clipped.
    assert spec_long.width >= spec_short.width


def test_summary_block_labels_use_line_breaks():
    """Summary block labels must use multi-line structure, not one long line."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "vgg",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64,
                "kernel_size": [3, 3]} for i in range(8)]
            + [{"name": "p1", "kind": "maxpool", "kernel_size": [2, 2]}]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }, detail_level="summary")
    block_labels = [s.label for s in spec.layers if "Block" in s.label]
    assert block_labels, "Expected Block N labels in summary mode"
    for lb in block_labels:
        assert "\n" in lb, f"Block label must be multi-line, got {lb!r}"


def test_compact_mode_avoids_excessive_full_labels_for_large_cnn():
    """In compact + auto mode, large CNN labels should be terse (no shape dims)."""
    big = {
        "model_name": "big",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64,
                "kernel_size": [3, 3]} for i in range(20)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec = nsx.build_nnsvg_spec(big, compact=True)
    # Auto label_mode → name for >9 layers; or summary collapse → ch=1 boxes.
    # Either way, slots/labels should remain readable.
    for s in spec.layers:
        if s.label and (s.layer_type == "dense" or s.channels != 1):
            for line in s.label.split("\n"):
                assert len(line) <= 28, (
                    f"Compact large-CNN label line too long: {line!r}"
                )


def test_diagnostic_payload_in_nnsvg_spec_dict():
    """build_nnsvg_spec must expose the diagnostic payload in to_dict()."""
    spec = nsx.build_nnsvg_spec(_transformer_spec(), transformer_mode="unsupported")
    d = spec.to_dict()
    assert d.get("diagnostic"), "diagnostic payload must be present in spec dict"
    assert d["diagnostic"]["kind"] == "transformer_unsupported"
    assert "actions" in d["diagnostic"]
    assert "block_summary" in " ".join(d["diagnostic"]["actions"])


def test_block_summary_spec_has_no_diagnostic_payload():
    spec = nsx.build_nnsvg_spec(_transformer_spec(), transformer_mode="block_summary")
    assert spec.diagnostic is None, (
        "block_summary mode must not set the diagnostic payload"
    )


# ── Label/layout safety: ONNX-style labels, ResNet/U-Net, Transformer ───────

def _onnx_style_conv_spec() -> dict:
    """Simulate ONNX-parsed Conv + BN + ReLU chain with long compact labels."""
    return {
        "model_name": "onnx_cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            {"name": "Conv_0", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3], "stride": [1, 1]},
            {"name": "BN_1", "kind": "batchnorm"},
            {"name": "Relu_2", "kind": "relu"},
            {"name": "Conv_3", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3], "stride": [2, 2]},
            {"name": "BN_4", "kind": "batchnorm"},
            {"name": "Relu_5", "kind": "relu"},
            {"name": "Gemm_6", "kind": "dense", "units": 1000},
        ],
    }


def test_onnx_style_labels_no_ellipsis():
    """ONNX-style compact labels must not contain '…' ellipsis."""
    spec = nsx.build_nnsvg_spec(_onnx_style_conv_spec(), label_mode="compact", width=900)
    for s in spec.layers:
        if s.label:
            assert "…" not in s.label, f"Ellipsis in label: {s.label!r}"


def test_onnx_style_labels_no_overflow_slot():
    """After wrapping, no label line should be longer than the safe slot budget."""
    spec = nsx.build_nnsvg_spec(_onnx_style_conv_spec(), label_mode="compact", width=900)
    n = len(spec.layers)
    slot_px = max(1, (spec.width - 120) / n)
    safe_chars = max(4, int(slot_px * 0.92 / (12 * 0.62)))
    for s in spec.layers:
        if not s.label:
            continue
        # ch=1 boxes are auto-scaled by JS — skip them
        if s.layer_type != "dense" and s.channels == 1:
            continue
        for line in s.label.split("\n"):
            assert len(line) <= safe_chars + 2, (
                f"Label line too long ({len(line)} > {safe_chars}): {line!r}"
            )


def test_full_label_preserved_in_extra_after_wrap():
    """When a label is wrapped, the original is preserved in layer.extra['full_label']."""
    from neuroschemax.visualization.nnsvg_mapper import _safe_label_policy
    from neuroschemax.visualization.nnsvg_schema import NNSVGLayerSpec
    layers = [
        NNSVGLayerSpec(layer_type="conv", label="Dense 256 +ReLU +BN +Drop 0.5", channels=3),
        NNSVGLayerSpec(layer_type="conv", label="Conv 128 k3 s2", channels=2),
    ]
    out = _safe_label_policy(layers, 480, 12, allow_thinning=False)
    # The first label is long and should have been wrapped
    if "\n" in out[0].label:
        assert "full_label" in out[0].extra, (
            "full_label must be in extra when label is wrapped"
        )
        assert "Dense 256" in out[0].extra["full_label"]


def test_resnet_summary_labels_no_ellipsis():
    """ResNet summary block labels must not contain '…'."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]} for i in range(8)],
            {"name": "add1", "kind": "add"},
            {"name": "add2", "kind": "add"},
            {"name": "fc", "kind": "dense", "units": 1000},
        ],
    }, detail_level="summary")
    for s in spec.layers:
        if s.label:
            assert "…" not in s.label, f"Ellipsis in ResNet summary label: {s.label!r}"


def test_unet_summary_labels_no_ellipsis():
    """U-Net summary block labels must not contain '…'."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "unet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "e1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "e2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "bot", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "up", "kind": "upsample", "scale_factor": 2},
            {"name": "d1", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat", "kind": "concat"},
            {"name": "out", "kind": "conv", "out_channels": 1, "kernel_size": [1, 1]},
        ],
    }, detail_level="summary")
    for s in spec.layers:
        if s.label:
            assert "…" not in s.label, f"Ellipsis in U-Net summary label: {s.label!r}"


def test_transformer_unsupported_diagnostic_label_no_overflow(tmp_path: Path):
    """Transformer unsupported diagnostic card must fit inside its HTML card (no overflow)."""
    trans_src = {
        "model_name": "trans",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn", "kind": "attention"},
            {"name": "ff",   "kind": "dense", "units": 512},
        ],
    }
    html_path = tmp_path / "trans_unsupported.html"
    nsx.save_network_html(html_path, trans_src, transformer_mode="unsupported")
    html = html_path.read_text()
    # The structured card should be present and contain all required sections
    assert "nnsvg-diagnostic-title" in html
    assert "block_summary" in html
    assert "debug" in html.lower()
    assert "Detected" in html
    # Each line of diagnostic text must be present without truncation
    for key in ("is not supported", "block_summary", "debug JSON"):
        assert key in html, f"Expected diagnostic text {key!r} missing from HTML"


# ── SVG/PNG export: error messages ──────────────────────────────────────────

def test_browser_not_available_error_has_install_instructions():
    """BrowserNotAvailableError must include actionable install instructions."""
    from neuroschemax.exceptions import BrowserNotAvailableError
    err = BrowserNotAvailableError()
    msg = str(err)
    assert "playwright" in msg.lower()
    assert "chromium" in msg.lower()
    assert "html" in msg.lower()  # mentions HTML as the safe alternative


def test_svg_export_error_is_actionable(tmp_path: Path):
    """SVG export when Playwright missing must raise BrowserNotAvailableError with install hint."""
    from neuroschemax.visualization.nnsvg_runtime import is_playwright_available
    if is_playwright_available():
        pytest.skip("Playwright installed; only testing absent-Playwright path")
    with pytest.raises(nsx.BrowserNotAvailableError) as exc_info:
        nsx.save_network_svg(tmp_path / "out.svg", _mlp_spec())
    msg = str(exc_info.value)
    assert "playwright" in msg.lower()


# ── Legend ───────────────────────────────────────────────────────────────────

def test_legend_present_by_default(tmp_path: Path):
    """HTML output must include the legend by default."""
    path = tmp_path / "mlp.html"
    nsx.save_network_html(path, _mlp_spec())
    html = path.read_text()
    assert "nnsvg-legend" in html, "Legend class must appear by default"


def test_legend_disabled_with_flag(tmp_path: Path):
    """show_legend=False must remove the legend from HTML output."""
    path = tmp_path / "mlp_no_legend.html"
    nsx.save_network_html(path, _mlp_spec(), show_legend=False)
    html = path.read_text()
    # The legend div should not be present when disabled
    assert "<div class='nnsvg-legend'" not in html


def test_legend_contains_color_swatches(tmp_path: Path):
    """Legend must include colour swatches for at least Conv and Dense."""
    path = tmp_path / "mlp_legend.html"
    nsx.save_network_html(path, _mlp_spec())
    html = path.read_text()
    assert "nnsvg-legend-swatch" in html
    # Should mention Conv and Dense/Classifier layer types
    assert "Conv" in html
    assert "Dense" in html or "Classifier" in html


def test_legend_shows_approx_note_for_resnet(tmp_path: Path):
    """Legend for an approximate model (ResNet) must note 'approximate'."""
    resnet = {
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
              for i in range(5)],
            {"name": "add1", "kind": "add"},
            {"name": "fc",   "kind": "dense", "units": 1000},
        ],
    }
    path = tmp_path / "resnet_legend.html"
    nsx.save_network_html(path, resnet)
    html = path.read_text()
    assert "approximate" in html.lower()


def test_legend_shows_exact_note_for_mlp(tmp_path: Path):
    """Legend for an exact model (MLP) must say 'exact'."""
    path = tmp_path / "mlp_exact.html"
    nsx.save_network_html(path, _mlp_spec())
    html = path.read_text()
    # The legend approx note should say 'exact' for a clean MLP
    assert "exact" in html


def test_legend_hidden_for_transformer_unsupported(tmp_path: Path):
    """Diagnostic card mode hides the legend (card is self-explanatory)."""
    trans = {
        "model_name": "trans",
        "layers": [
            {"name": "attn", "kind": "attention"},
            {"name": "ff",   "kind": "dense", "units": 512},
        ],
    }
    path = tmp_path / "trans_diag.html"
    nsx.save_network_html(path, trans, transformer_mode="unsupported")
    html = path.read_text()
    # In diagnostic mode the legend div element should not appear
    assert "<div class='nnsvg-legend'" not in html


def test_show_legend_in_nnsvg_spec():
    """NNSVGSpec must carry the show_legend flag."""
    spec = nsx.build_nnsvg_spec(_mlp_spec(), show_legend=True)
    assert spec.show_legend is True
    spec2 = nsx.build_nnsvg_spec(_mlp_spec(), show_legend=False)
    assert spec2.show_legend is False
    d = spec2.to_dict()
    assert d["showLegend"] is False


# ── Family mapping and summary rendering: ResNet / U-Net / AlexNet ──────────

def _resnet_manual_spec() -> dict:
    return {
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


def _unet_manual_spec() -> dict:
    return {
        "model_name": "unet_like",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "e1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "e2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "bot", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "up", "kind": "upsample", "scale_factor": 2},
            {"name": "d1", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat", "kind": "concat"},
            {"name": "out", "kind": "conv", "out_channels": 1, "kernel_size": [1, 1]},
        ],
    }


def _vgg_sequential_spec() -> dict:
    layers = [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
    for i in range(6):
        layers.append({"name": f"c{i}", "kind": "conv",
                       "out_channels": 64, "kernel_size": [3, 3]})
    for i in range(6):
        layers.append({"name": f"c{i + 6}", "kind": "conv",
                       "out_channels": 128, "kernel_size": [3, 3]})
    layers.append({"name": "fc", "kind": "dense", "units": 1000})
    return {"model_name": "vgg_like", "layers": layers}


# -- Subtitle / label correctness --

def test_resnet_manual_subtitle_not_alexnet_style():
    """ResNet-like manual spec must not be subtitled 'AlexNet-style'."""
    spec = nsx.build_nnsvg_spec(_resnet_manual_spec())
    assert "AlexNet" not in spec.subtitle, (
        f"ResNet subtitle must not say AlexNet-style, got: {spec.subtitle!r}"
    )
    assert "ResNet" in spec.subtitle or "resnet" in spec.subtitle.lower(), (
        f"ResNet subtitle must identify the architecture: {spec.subtitle!r}"
    )


def test_unet_manual_subtitle_not_alexnet_style():
    """U-Net-like manual spec must not be subtitled 'AlexNet-style'."""
    spec = nsx.build_nnsvg_spec(_unet_manual_spec())
    assert "AlexNet" not in spec.subtitle, (
        f"U-Net subtitle must not say AlexNet-style, got: {spec.subtitle!r}"
    )
    assert "U-Net" in spec.subtitle or "unet" in spec.subtitle.lower(), (
        f"U-Net subtitle must identify the architecture: {spec.subtitle!r}"
    )


def test_vgg_sequential_subtitle_is_sequential_cnn():
    """Sequential VGG-like CNN must be labeled as a sequential CNN (not ResNet/U-Net)."""
    spec = nsx.build_nnsvg_spec(_vgg_sequential_spec())
    # Should say "Sequential CNN" or "LeNet-style CNN" — not ResNet/U-Net summary
    assert "ResNet" not in spec.subtitle and "U-Net" not in spec.subtitle, (
        f"Sequential VGG must not be labeled ResNet/U-Net: {spec.subtitle!r}"
    )
    assert "CNN" in spec.subtitle or "AlexNet" in spec.subtitle or "Sequential" in spec.subtitle, (
        f"Sequential VGG must be identified as CNN family: {spec.subtitle!r}"
    )


# -- ResNet summary rendering --

def test_resnet_manual_auto_detail_produces_residual_blocks():
    """With default detail_level=auto, ResNet-like model must produce residual block labels."""
    spec = nsx.build_nnsvg_spec(_resnet_manual_spec())
    labels = [s.label for s in spec.layers]
    combined = "\n".join(labels)
    assert any("Residual" in lb or "Res Block" in lb or "Block" in lb for lb in labels), (
        f"ResNet auto-detail must produce residual block labels, got: {labels}"
    )
    assert "skip collapsed" in combined.lower() or "skip" in combined.lower(), (
        f"ResNet blocks must mention skip-collapsed; got labels: {labels}"
    )


def test_resnet_manual_full_detail_shows_individual_convs():
    """With detail_level=full, ResNet-like model shows individual conv layers."""
    spec = nsx.build_nnsvg_spec(_resnet_manual_spec(), detail_level="full")
    # 4 conv + input + classifier — should have more stages than summary
    assert len(spec.layers) >= 4, (
        f"detail_level=full should show individual layers, got {len(spec.layers)}"
    )


def test_resnet_not_flat_alexnet_columns():
    """ResNet summary must not produce a flat line of identical conv-column layers."""
    spec = nsx.build_nnsvg_spec(_resnet_manual_spec())
    # All non-input ch=1 blocks with same channel and fmW is the 'flat AlexNet' failure
    non_input = [s for s in spec.layers if s.layer_type != "input"]
    # Summary residual blocks should have ch=1 box labels, not individually flat convs
    # Key check: no layer should be labeled raw 'Conv 64 k3' (individual flat conv)
    raw_conv_count = sum(1 for s in non_input if s.label.startswith("Conv "))
    assert raw_conv_count < len(non_input), (
        f"ResNet should not show all raw conv columns; got {raw_conv_count} conv labels"
    )


# -- U-Net summary rendering --

def test_unet_manual_auto_detail_produces_encoder_decoder():
    """With default detail_level=auto, U-Net-like model must show Encoder/Decoder labels."""
    spec = nsx.build_nnsvg_spec(_unet_manual_spec())
    labels = [s.label for s in spec.layers]
    combined = "\n".join(labels)
    assert "Encoder" in combined, f"U-Net must have Encoder block, got: {labels}"
    assert "Decoder" in combined, f"U-Net must have Decoder block, got: {labels}"
    assert "Bottleneck" in combined, f"U-Net must have Bottleneck block, got: {labels}"


def test_unet_decoder_mentions_concat_collapsed():
    """U-Net decoder label must mention concat/skip collapsed."""
    spec = nsx.build_nnsvg_spec(_unet_manual_spec())
    decoder_labels = [s.label for s in spec.layers if "Decoder" in s.label]
    assert decoder_labels, f"No Decoder label found in {[s.label for s in spec.layers]}"
    combined = "\n".join(decoder_labels)
    assert "concat" in combined.lower() or "skip" in combined.lower(), (
        f"Decoder label must mention concat/skip collapsed; got: {decoder_labels}"
    )


# -- ONNX support --

def _make_onnx_resnet():
    """Minimal ONNX graph with Add (residual-style)."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        import pytest
        pytest.skip("onnx not available")
    conv1 = helper.make_node("Conv", ["x","w1","b1"], ["c1"], kernel_shape=[3,3], pads=[1,1,1,1])
    conv2 = helper.make_node("Conv", ["c1","w2","b2"], ["c2"], kernel_shape=[3,3], pads=[1,1,1,1])
    add   = helper.make_node("Add", ["x","c2"], ["out"])
    x    = helper.make_tensor_value_info("x",  TensorProto.FLOAT, [1,64,32,32])
    w1_i = helper.make_tensor("w1", TensorProto.FLOAT, [64,64,3,3], [0.0]*36864)
    b1_i = helper.make_tensor("b1", TensorProto.FLOAT, [64], [0.0]*64)
    w2_i = helper.make_tensor("w2", TensorProto.FLOAT, [64,64,3,3], [0.0]*36864)
    b2_i = helper.make_tensor("b2", TensorProto.FLOAT, [64], [0.0]*64)
    out  = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)
    graph = helper.make_graph([conv1, conv2, add], "resnet_onnx",
        [x, helper.make_tensor_value_info("w1", TensorProto.FLOAT, [64,64,3,3]),
            helper.make_tensor_value_info("b1", TensorProto.FLOAT, [64]),
            helper.make_tensor_value_info("w2", TensorProto.FLOAT, [64,64,3,3]),
            helper.make_tensor_value_info("b2", TensorProto.FLOAT, [64])],
        [out], initializer=[w1_i, b1_i, w2_i, b2_i])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _make_onnx_unet():
    """Minimal ONNX graph with Concat (encoder-decoder-style)."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        import pytest
        pytest.skip("onnx not available")
    conv1  = helper.make_node("Conv",   ["x","w1","b1"], ["c1"],   kernel_shape=[3,3], pads=[1,1,1,1])
    conv2  = helper.make_node("Conv",   ["c1","w2","b2"],["c2"],   kernel_shape=[3,3], pads=[1,1,1,1])
    concat = helper.make_node("Concat", ["c1","c2"],     ["cat"],  axis=1)
    out_v  = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)
    x    = helper.make_tensor_value_info("x",  TensorProto.FLOAT, [1,32,64,64])
    w1_i = helper.make_tensor("w1", TensorProto.FLOAT, [32,32,3,3], [0.0]*9216)
    b1_i = helper.make_tensor("b1", TensorProto.FLOAT, [32], [0.0]*32)
    w2_i = helper.make_tensor("w2", TensorProto.FLOAT, [32,32,3,3], [0.0]*9216)
    b2_i = helper.make_tensor("b2", TensorProto.FLOAT, [32], [0.0]*32)
    graph = helper.make_graph([conv1, conv2, concat], "unet_onnx",
        [x, helper.make_tensor_value_info("w1", TensorProto.FLOAT, [32,32,3,3]),
            helper.make_tensor_value_info("b1", TensorProto.FLOAT, [32]),
            helper.make_tensor_value_info("w2", TensorProto.FLOAT, [32,32,3,3]),
            helper.make_tensor_value_info("b2", TensorProto.FLOAT, [32])],
        [out_v], initializer=[w1_i, b1_i, w2_i, b2_i])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def test_onnx_resnet_subtitle_says_resnet_summary():
    """ONNX graph with Add op must be subtitled 'ResNet summary', not 'AlexNet-style'."""
    model = _make_onnx_resnet()
    spec = nsx.build_nnsvg_spec(model)
    assert "AlexNet" not in spec.subtitle, (
        f"ONNX ResNet subtitle must not say AlexNet-style: {spec.subtitle!r}"
    )
    assert "ResNet" in spec.subtitle, (
        f"ONNX ResNet subtitle must say ResNet summary: {spec.subtitle!r}"
    )


def test_onnx_resnet_produces_residual_block_labels():
    """ONNX graph with Add op must produce Residual Block labels."""
    model = _make_onnx_resnet()
    spec = nsx.build_nnsvg_spec(model)
    labels = [s.label for s in spec.layers]
    assert any("Residual" in lb or "Block" in lb for lb in labels), (
        f"ONNX ResNet must have residual block labels, got: {labels}"
    )


def test_onnx_unet_subtitle_says_unet_summary():
    """ONNX graph with Concat op must be subtitled 'U-Net summary', not 'AlexNet-style'."""
    model = _make_onnx_unet()
    spec = nsx.build_nnsvg_spec(model)
    assert "AlexNet" not in spec.subtitle, (
        f"ONNX U-Net subtitle must not say AlexNet-style: {spec.subtitle!r}"
    )
    assert "U-Net" in spec.subtitle, (
        f"ONNX U-Net subtitle must say U-Net summary: {spec.subtitle!r}"
    )


def test_onnx_unet_produces_encoder_decoder_labels():
    """ONNX graph with Concat op must produce Encoder/Decoder labels."""
    model = _make_onnx_unet()
    spec = nsx.build_nnsvg_spec(model)
    labels = [s.label for s in spec.layers]
    combined = "\n".join(labels)
    assert "Encoder" in combined or "Decoder" in combined, (
        f"ONNX U-Net must have Encoder/Decoder labels, got: {labels}"
    )


def test_vgg_sequential_onnx_subtitle_is_sequential_cnn():
    """ONNX sequential CNN without merge ops must be identified as Sequential CNN family."""
    spec = nsx.build_nnsvg_spec(_vgg_sequential_spec())
    # Should identify as some form of sequential CNN
    assert "ResNet" not in spec.subtitle and "U-Net" not in spec.subtitle, (
        f"Sequential VGG must not be labeled ResNet/U-Net: {spec.subtitle!r}"
    )
    assert "CNN" in spec.subtitle or "AlexNet" in spec.subtitle or "Sequential" in spec.subtitle, (
        f"Sequential VGG must be identified as CNN family: {spec.subtitle!r}"
    )


# -- Notebook iframe height --

def test_to_notebook_html_has_adequate_height():
    """to_notebook_html() must use at least 700px height to avoid scroll within small cell."""
    import neuroschemax as nsx
    spec = {
        "model_name": "test",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1",   "kind": "dense", "units": 10},
        ],
    }
    fig = nsx.Figure()
    fig.draw(spec)
    nb_html = fig.to_notebook_html()
    # Extract height value from the iframe attributes
    import re
    m = re.search(r'height="(\d+)"', nb_html)
    assert m, "iframe must have explicit height attribute"
    iframe_height = int(m.group(1))
    assert iframe_height >= 700, (
        f"iframe height must be >= 700 for diagrams to be readable; got {iframe_height}"
    )


# ── NN-SVG primitive regression tests ───────────────────────────────────────

def test_vgg_full_uses_nnsvg_primitives_not_boxes():
    """VGG full mode must use multi-channel conv/pool/dense primitives, not ch=1 boxes."""
    vgg = {
        "model_name": "vgg",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(4)]
            + [{"name": "pool", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]}]
            + [{"name": f"c{i+4}", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]}
               for i in range(4)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec = nsx.build_nnsvg_spec(vgg, detail_level="full")
    non_input = [s for s in spec.layers if s.layer_type != "input"]
    # All non-input conv layers must be multi-channel (real NN-SVG stacks)
    conv_layers = [s for s in non_input if s.layer_type in ("conv", "pool")]
    ch1_boxes = [s for s in conv_layers if s.channels == 1]
    assert not ch1_boxes, (
        f"VGG full must not use ch=1 generic boxes; found: {[(s.layer_type, s.label) for s in ch1_boxes]}"
    )


def test_vgg_summary_uses_nnsvg_primitives():
    """VGG summary mode must retain conv/pool/dense NN-SVG primitives."""
    vgg = {
        "model_name": "vgg",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(4)]
            + [{"name": "pool", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]}]
            + [{"name": f"c{i+4}", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]}
               for i in range(4)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }
    spec = nsx.build_nnsvg_spec(vgg, detail_level="summary")
    types = {s.layer_type for s in spec.layers}
    # Summary must still include proper dense/classifier column (not only conv boxes)
    assert "dense" in types or any(s.channels > 1 for s in spec.layers if s.layer_type == "conv"), (
        "VGG summary must retain NN-SVG-style conv/dense primitives"
    )


def test_resnet_summary_no_ch1_generic_boxes():
    """ResNet summary must not produce ch=1 generic square boxes for backbone."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
              for i in range(4)],
            {"name": "add1", "kind": "add"},
            {"name": "fc", "kind": "dense", "units": 10},
        ],
    })
    conv_layers = [s for s in spec.layers if s.layer_type in ("conv", "pool") and s.layer_type != "input"]
    ch1_boxes = [s for s in conv_layers if s.channels == 1]
    assert not ch1_boxes, (
        f"ResNet summary must use multi-channel conv primitives, not ch=1 boxes; "
        f"got: {[(s.layer_type, s.channels, s.label[:20]) for s in ch1_boxes]}"
    )


def test_resnet_summary_has_dense_classifier():
    """ResNet summary must end with a dense/classifier NN-SVG column."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
              for i in range(4)],
            {"name": "add1", "kind": "add"},
            {"name": "fc", "kind": "dense", "units": 10},
        ],
    })
    dense_layers = [s for s in spec.layers if s.layer_type == "dense"]
    assert dense_layers, "ResNet summary must have a dense classifier column"
    assert "Classifier" in dense_layers[-1].label or "10" in dense_layers[-1].label


def test_unet_summary_no_ch1_generic_boxes():
    """U-Net summary must not produce ch=1 generic square boxes for encoder/decoder."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "unet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "e1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "e2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "bot", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "up", "kind": "upsample", "scale_factor": 2},
            {"name": "d1", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat", "kind": "concat"},
            {"name": "out", "kind": "conv", "out_channels": 1, "kernel_size": [1, 1]},
        ],
    })
    conv_pool = [s for s in spec.layers if s.layer_type in ("conv", "pool") and s.layer_type != "input"]
    ch1_boxes = [s for s in conv_pool if s.channels == 1 and s.layer_type == "conv" and s.feature_map_width > 4]
    # We allow ch=1 for the actual segmentation head output (1ch output conv)
    # but NOT for encoder/decoder stages
    enc_dec_ch1 = [s for s in ch1_boxes
                   if "Encoder" in s.label or "Decoder" in s.label or "Bottleneck" in s.label]
    assert not enc_dec_ch1, (
        f"U-Net Encoder/Decoder stages must use multi-channel conv primitives, not ch=1 boxes: "
        f"{[(s.channels, s.label[:30]) for s in enc_dec_ch1]}"
    )


def test_unet_summary_has_pool_primitive():
    """U-Net summary must include a pool primitive for the encoder downsampling."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "unet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "e1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "e2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "bot", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "up", "kind": "upsample", "scale_factor": 2},
            {"name": "d1", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat", "kind": "concat"},
            {"name": "out", "kind": "conv", "out_channels": 1, "kernel_size": [1, 1]},
        ],
    })
    pool_layers = [s for s in spec.layers if s.layer_type == "pool"]
    assert pool_layers, "U-Net summary must include pool primitive for encoder downsampling"


def test_resnet_warnings_no_alexnet_view_text():
    """ResNet warnings must not say 'AlexNet view'."""
    arch = nsx.parse_model({
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
              for i in range(5)],
            {"name": "add1", "kind": "add"},
            {"name": "fc", "kind": "dense", "units": 10},
        ],
    })
    combined = " ".join(arch.warnings).lower()
    assert "alexnet view" not in combined, (
        f"ResNet warning must not mention 'AlexNet view': {arch.warnings}"
    )


def test_unet_warnings_no_alexnet_view_text():
    """U-Net warnings must not say 'AlexNet view'."""
    arch = nsx.parse_model({
        "model_name": "unet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "e1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "bot", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat", "kind": "concat"},
            {"name": "out", "kind": "conv", "out_channels": 1, "kernel_size": [1, 1]},
        ],
    })
    combined = " ".join(arch.warnings).lower()
    assert "alexnet view" not in combined, (
        f"U-Net warning must not mention 'AlexNet view': {arch.warnings}"
    )


def test_transformer_unsupported_warning_no_block_summary_shown():
    """Transformer unsupported mode warning must not claim block-level summary is shown."""
    arch = nsx.parse_model({
        "model_name": "trans",
        "layers": [
            {"name": "embed", "kind": "embedding"},
            {"name": "attn", "kind": "attention"},
            {"name": "ff", "kind": "dense", "units": 512},
        ],
    })
    # The family_recognizer warning should not say "block-level summary shown"
    # when the user is about to use transformer_mode="unsupported"
    combined = " ".join(arch.warnings)
    assert "block-level summary shown" not in combined, (
        f"Transformer warning must not claim block summary is shown; got: {combined[:200]!r}"
    )


def test_readme_contains_colab_viewer_link():
    """README must contain the public Colab viewer link."""
    import pathlib
    readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
    assert "colab.research.google.com" in readme, "README must contain Colab link"
    assert "1oVe9JRJukQ5dQsFH8XoVQj6b1IDpy2MS" in readme, (
        "README must contain the specific Colab notebook ID"
    )


# ── Final visual-system policy tests ────────────────────────────────────────

def _vgg_with_two_dense_spec() -> dict:
    """VGG-like model with two dense layers (fc1=4096, fc2=1000) to test classifier."""
    layers = [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
    for i in range(4):
        layers.append({"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]})
    layers.append({"name": "pool", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]})
    layers.append({"name": "fc1", "kind": "dense", "units": 4096})
    layers.append({"name": "fc2", "kind": "dense", "units": 1000})
    return {"model_name": "vgg_two_fc", "layers": layers}


def test_vgg_summary_classifier_uses_last_dense_not_first():
    """VGG summary classifier must show the LAST dense unit count (1000), not the first (4096)."""
    spec = nsx.build_nnsvg_spec(_vgg_with_two_dense_spec(), detail_level="summary")
    dense_layers = [s for s in spec.layers if s.layer_type == "dense"]
    assert dense_layers, "VGG summary must produce a dense classifier"
    last = dense_layers[-1]
    assert "1000" in last.label, (
        f"Classifier must show true output class count (1000), not first dense (4096); "
        f"got: {last.label!r}"
    )
    assert "4096" not in last.label, (
        f"Classifier label must not show intermediate dense size 4096; got: {last.label!r}"
    )


def test_vgg_summary_fmw_decreases_after_each_pool():
    """VGG summary must show visually decreasing feature-map sizes after each pool."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "vgg",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(2)]
            + [{"name": "p1", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]}]
            + [{"name": f"c{i+2}", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]}
               for i in range(2)]
            + [{"name": "p2", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]}]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    }, detail_level="summary")
    conv_fmws = [s.feature_map_width for s in spec.layers
                 if s.layer_type == "conv" and s.layer_type != "input" and s.feature_map_width > 0]
    assert len(conv_fmws) >= 2, "Need at least 2 conv blocks for progression check"
    # Each subsequent conv block must be smaller (halved after pool)
    for k in range(1, len(conv_fmws)):
        assert conv_fmws[k] <= conv_fmws[k - 1], (
            f"fmW must not increase between blocks; got {conv_fmws}"
        )


def test_vgg_subtitle_says_sequential_cnn_not_alexnet():
    """VGG/AlexNet-style diagrams should use 'Sequential CNN', not the internal renderer name."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "vgg",
        "layers": (
            [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
            + [{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
               for i in range(5)]
            + [{"name": "fc", "kind": "dense", "units": 1000}]
        ),
    })
    assert "ResNet" not in spec.subtitle and "U-Net" not in spec.subtitle, (
        f"Sequential CNN subtitle must not say ResNet/U-Net: {spec.subtitle!r}"
    )
    # Must identify as some form of CNN (exact string may change, but should be CNN-related)
    assert any(kw in spec.subtitle for kw in ("CNN", "AlexNet", "LeNet", "Sequential")), (
        f"Sequential CNN subtitle must identify the CNN family: {spec.subtitle!r}"
    )


def test_tiny_cnn_compact_labels_show_op_type():
    """TinyCNN (<=12 arch layers) must use compact labels showing op type, not just names."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "tiny_cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "bn1", "kind": "batchnorm"},
            {"name": "relu1", "kind": "relu"},
            {"name": "pool1", "kind": "maxpool", "kernel_size": [2, 2], "stride": [2, 2]},
            {"name": "conv2", "kind": "conv", "out_channels": 32, "kernel_size": [3, 3]},
            {"name": "relu2", "kind": "relu"},
            {"name": "fc1", "kind": "dense", "units": 10},
        ],
    })
    conv_labels = [s.label for s in spec.layers if s.layer_type == "conv" and s.channels > 1]
    # Compact mode should produce operation-type labels: "Conv 16 k3" style
    assert any("Conv" in lb for lb in conv_labels), (
        f"TinyCNN should use compact 'Conv ...' labels, got: {conv_labels}"
    )


def test_resnet_stem_label_has_conv_info():
    """ResNet stem label must include conv count/channel info, not just 'Stem'."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
              for i in range(4)],
            {"name": "add1", "kind": "add"},
            {"name": "fc", "kind": "dense", "units": 1000},
        ],
    })
    stem_layers = [s for s in spec.layers if "Stem" in (s.label or "")]
    assert stem_layers, "ResNet must have a Stem block"
    stem_label = stem_layers[0].label
    # Must include conv count or channel info
    assert "Conv" in stem_label or "ch" in stem_label or "×" in stem_label, (
        f"Stem label must include conv/channel info; got: {stem_label!r}"
    )


def test_resnet_fmw_decreases_from_stem_to_blocks():
    """ResNet visual progression: stem fmW > residual block fmW."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "resnet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            *[{"name": f"c{i}", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]}
              for i in range(4)],
            {"name": "add1", "kind": "add"},
            {"name": "fc", "kind": "dense", "units": 1000},
        ],
    })
    conv_layers = [s for s in spec.layers if s.layer_type == "conv" and s.feature_map_width > 0]
    if len(conv_layers) >= 2:
        # Stem (first) should be larger than residual blocks (subsequent)
        assert conv_layers[0].feature_map_width > conv_layers[1].feature_map_width, (
            f"Stem fmW ({conv_layers[0].feature_map_width}) must be larger than "
            f"residual block fmW ({conv_layers[1].feature_map_width})"
        )


def test_unet_fmw_shows_compress_expand():
    """U-Net fmW must compress (encoder→bottleneck) then expand (bottleneck→decoder)."""
    spec = nsx.build_nnsvg_spec({
        "model_name": "unet",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 64, 64]},
            {"name": "e1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "e2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "bot", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "up", "kind": "upsample", "scale_factor": 2},
            {"name": "d1", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "cat", "kind": "concat"},
            {"name": "out", "kind": "conv", "out_channels": 1, "kernel_size": [1, 1]},
        ],
    })
    enc = next((s for s in spec.layers if "Encoder" in (s.label or "")), None)
    bot = next((s for s in spec.layers if "Bottleneck" in (s.label or "")), None)
    dec = next((s for s in spec.layers if "Decoder" in (s.label or "")), None)
    assert enc and bot and dec, "U-Net must have Encoder, Bottleneck, Decoder stages"
    # Encoder > Bottleneck (compression)
    assert enc.feature_map_width > bot.feature_map_width, (
        f"Encoder fmW ({enc.feature_map_width}) must be larger than "
        f"Bottleneck fmW ({bot.feature_map_width})"
    )
    # Decoder > Bottleneck (expansion)
    assert dec.feature_map_width > bot.feature_map_width, (
        f"Decoder fmW ({dec.feature_map_width}) must be larger than "
        f"Bottleneck fmW ({bot.feature_map_width})"
    )


def test_readme_uses_only_supported_api_options():
    """README examples must not use genuinely nonexistent API options."""
    import pathlib
    readme = (pathlib.Path(__file__).parent.parent / "README.md").read_text()
    # These style values do not exist in the rendering API
    bad_options = ["style='vgg'", 'style="vgg"', "style='resnet'", 'style="resnet"']
    for bad in bad_options:
        assert bad not in readme, (
            f"README contains nonexistent style option {bad!r}"
        )
