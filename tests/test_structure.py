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
    """Unit test for _safe_label_policy: long labels get truncated."""
    from neuroschemax.visualization.nnsvg_mapper import _safe_label_policy
    from neuroschemax.visualization.nnsvg_schema import NNSVGLayerSpec
    layers = [
        NNSVGLayerSpec(layer_type="conv", label="very_long_layer_name_here", channels=3),
        NNSVGLayerSpec(layer_type="conv", label="another_very_long_name", channels=2),
        NNSVGLayerSpec(layer_type="dense", label="classifier_head_output", units=5),
    ]
    # 480px canvas, 3 layers: slot=120px, safe_chars≈12
    result = _safe_label_policy(layers, 480, 12, allow_thinning=False)
    for s in result:
        if s.label:
            assert len(s.label) <= 15, f"Label not truncated: {s.label!r}"


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
    block_labels = [s.label for s in spec.layers if s.layer_type == "conv" and s.channels == 1]
    assert len(block_labels) > 0, "Summary should produce ch=1 block labels"
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
