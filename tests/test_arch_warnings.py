"""Tests for architecture detection, family recommendation, and warnings."""

from __future__ import annotations

import neuroschemax as nsx
from neuroschemax.core.enums import ConfidenceLevel, RenderFamily

# ---------------------------------------------------------------------------
# Helper specs
# ---------------------------------------------------------------------------

def _mlp_spec() -> dict:
    return {
        "model_name": "mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 256},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2", "kind": "dense", "units": 128},
            {"name": "relu2", "kind": "relu"},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    }


def _small_cnn_spec() -> dict:
    return {
        "model_name": "small_cnn",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 1, 28, 28]},
            {"name": "conv1", "kind": "conv", "out_channels": 16, "kernel_size": [3, 3]},
            {"name": "relu1", "kind": "relu"},
            {"name": "conv2", "kind": "conv", "out_channels": 32, "kernel_size": [3, 3]},
            {"name": "relu2", "kind": "relu"},
            {"name": "pool", "kind": "maxpool", "kernel_size": [2, 2]},
            {"name": "flatten", "kind": "flatten"},
            {"name": "fc1", "kind": "dense", "units": 10},
        ],
    }


def _deep_cnn_spec() -> dict:
    layers = [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
    for i in range(6):
        layers.append({
            "name": f"conv{i}", "kind": "conv",
            "out_channels": 64, "kernel_size": [3, 3],
        })
        layers.append({"name": f"relu{i}", "kind": "relu"})
    layers.append({"name": "pool", "kind": "globalaveragepool"})
    layers.append({"name": "fc", "kind": "dense", "units": 1000})
    return {"model_name": "deep_cnn", "layers": layers}


def _attention_spec() -> dict:
    """A spec that contains attention layers (Transformer-like)."""
    return {
        "model_name": "transformer_like",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 512]},
            {"name": "attn1", "kind": "attention"},
            {"name": "attn2", "kind": "attention"},
            {"name": "fc1", "kind": "dense", "units": 512},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    }


def _rnn_spec() -> dict:
    """A spec that contains recurrent layers."""
    return {
        "model_name": "rnn_model",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 100, 64]},
            {"name": "lstm1", "kind": "lstm"},
            {"name": "lstm2", "kind": "lstm"},
            {"name": "fc1", "kind": "dense", "units": 10},
        ],
    }


def _skip_conn_spec() -> dict:
    """A spec with Add nodes indicating skip connections (ResNet-like, 4+ convs)."""
    return {
        "model_name": "skip_net",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 224, 224]},
            {"name": "conv1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "conv2", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "add1", "kind": "add"},
            {"name": "conv3", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "conv4", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "add2", "kind": "add"},
            {"name": "fc1", "kind": "dense", "units": 10},
        ],
    }


def _unknown_op_spec() -> dict:
    """A spec with only unknown ops."""
    return {
        "model_name": "unknown_ops",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 64]},
            {"name": "custom1", "kind": "unknown"},
            {"name": "custom2", "kind": "unknown"},
        ],
    }


def _deep_skip_spec() -> dict:
    """Deep CNN (>3 convs) with skip connections (Add layer)."""
    layers = [{"name": "input", "kind": "input", "shape": [1, 3, 224, 224]}]
    for i in range(6):
        layers.append({
            "name": f"conv{i}", "kind": "conv",
            "out_channels": 64, "kernel_size": [3, 3],
        })
    layers.append({"name": "add1", "kind": "add"})
    layers.append({"name": "fc1", "kind": "dense", "units": 1000})
    return {"model_name": "deep_skip", "layers": layers}


def _unet_like_spec() -> dict:
    """Encoder-decoder with concat ops (U-Net style)."""
    return {
        "model_name": "unet_like",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 3, 64, 64]},
            {"name": "enc1", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "enc2", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "bottleneck", "kind": "conv", "out_channels": 256, "kernel_size": [3, 3]},
            {"name": "dec1", "kind": "conv", "out_channels": 128, "kernel_size": [3, 3]},
            {"name": "concat1", "kind": "concat"},
            {"name": "dec2", "kind": "conv", "out_channels": 64, "kernel_size": [3, 3]},
            {"name": "concat2", "kind": "concat"},
            {"name": "out", "kind": "conv", "out_channels": 1, "kernel_size": [1, 1]},
        ],
    }


# ---------------------------------------------------------------------------
# Tests — family detection
# ---------------------------------------------------------------------------

def test_mlp_family():
    arch = nsx.parse_model(_mlp_spec())
    assert arch.recommended_family == RenderFamily.FCNN
    assert arch.family_confidence == ConfidenceLevel.HIGH
    assert arch.warnings == []  # clean MLP has no warnings


def test_small_cnn_family():
    arch = nsx.parse_model(_small_cnn_spec())
    assert arch.recommended_family == RenderFamily.LENET
    assert arch.family_confidence == ConfidenceLevel.HIGH


def test_deep_cnn_family():
    arch = nsx.parse_model(_deep_cnn_spec())
    assert arch.recommended_family == RenderFamily.ALEXNET
    assert arch.family_confidence == ConfidenceLevel.HIGH


def test_transformer_like_family():
    """Attention layers: FCNN with LOW confidence and a warning."""
    arch = nsx.parse_model(_attention_spec())
    assert arch.recommended_family == RenderFamily.FCNN
    # Must be LOW confidence — attention is NOT representable in NN-SVG
    assert arch.family_confidence == ConfidenceLevel.LOW
    # Must have at least one warning about attention/approximation
    assert len(arch.warnings) > 0
    warn_text = " ".join(arch.warnings).lower()
    assert "attention" in warn_text or "transformer" in warn_text


def test_rnn_family():
    """LSTM layers: FCNN with LOW confidence and a warning."""
    arch = nsx.parse_model(_rnn_spec())
    assert arch.recommended_family == RenderFamily.FCNN
    assert arch.family_confidence == ConfidenceLevel.LOW
    assert len(arch.warnings) > 0
    warn_text = " ".join(arch.warnings).lower()
    assert "recurrent" in warn_text or "lstm" in warn_text


def test_skip_conn_warning():
    """A model with Add nodes (ResNet-like) gets MEDIUM confidence and a warning."""
    arch = nsx.parse_model(_skip_conn_spec())
    # 4 convs + 2 add ops → AlexNet MEDIUM with residual warning
    assert arch.recommended_family == RenderFamily.ALEXNET
    assert arch.family_confidence == ConfidenceLevel.MEDIUM
    assert len(arch.warnings) > 0
    warn_text = " ".join(arch.warnings).lower()
    assert "skip" in warn_text or "residual" in warn_text or "add" in warn_text


def test_unknown_op_family():
    """All unknown ops should fall back to FCNN with LOW/UNKNOWN confidence."""
    arch = nsx.parse_model(_unknown_op_spec())
    assert arch.recommended_family == RenderFamily.FCNN
    assert arch.family_confidence in (ConfidenceLevel.LOW, ConfidenceLevel.UNKNOWN)


def test_warn_on_skip():
    """Deep CNN with Add layer produces a residual/skip warning."""
    arch = nsx.parse_model(_deep_skip_spec())
    # 6 convs + 1 add → AlexNet MEDIUM with warning
    assert arch.recommended_family == RenderFamily.ALEXNET
    assert arch.family_confidence == ConfidenceLevel.MEDIUM
    assert any(
        "skip" in w.lower() or "residual" in w.lower() or "backbone" in w.lower()
        for w in arch.warnings
    ), f"Expected skip/residual warning, got: {arch.warnings}"


def test_unet_like():
    """Encoder-decoder with Concat layers: AlexNet MEDIUM with U-Net/Concat warning."""
    arch = nsx.parse_model(_unet_like_spec())
    assert arch.recommended_family == RenderFamily.ALEXNET
    assert arch.family_confidence == ConfidenceLevel.MEDIUM
    assert len(arch.warnings) > 0
    warn_text = " ".join(arch.warnings).lower()
    assert "concat" in warn_text or "encoder" in warn_text or "branch" in warn_text


# ---------------------------------------------------------------------------
# Tests — recommend_view output
# ---------------------------------------------------------------------------

def test_recommend_view_has_reason():
    info = nsx.recommend_view(_mlp_spec())
    assert "reason" in info
    assert isinstance(info["reason"], str)
    assert len(info["reason"]) > 0


def test_recommend_view_has_is_approximate():
    """recommend_view must include an is_approximate boolean."""
    info = nsx.recommend_view(_mlp_spec())
    assert "is_approximate" in info
    assert info["is_approximate"] is False  # clean MLP


def test_recommend_view_approximate_for_transformer():
    info = nsx.recommend_view(_attention_spec())
    assert info["is_approximate"] is True
    assert info["confidence"] == "low"


def test_recommend_view_approximate_for_resnet():
    info = nsx.recommend_view(_skip_conn_spec())
    assert info["is_approximate"] is True
    assert info["confidence"] == "medium"


def test_recommend_view_reason_mlp():
    info = nsx.recommend_view(_mlp_spec())
    assert "dense" in info["reason"].lower() or "mlp" in info["reason"].lower()


def test_recommend_view_reason_cnn():
    info = nsx.recommend_view(_small_cnn_spec())
    assert "conv" in info["reason"].lower()


# ---------------------------------------------------------------------------
# Tests — NN-SVG layer mapping
# ---------------------------------------------------------------------------

def test_merge_ops_not_in_nnsvg_layers():
    """Add/Concat/Multiply layers must not appear as diagram columns."""
    spec = nsx.build_nnsvg_spec(_skip_conn_spec())
    # No layer in the NN-SVG spec should represent a raw merge op
    # (they are invisible structural ops, not neuron columns)
    for lay in spec.layers:
        assert lay.label not in ("add1", "add2", "concat1", "concat2"), (
            f"Merge op found as diagram layer: {lay.label}"
        )


def test_add_layers_removed_from_fcnn():
    """An FCNN model with Add ops should not show Add as a neuron column."""
    spec_dict = {
        "model_name": "mlp_with_add",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "skip_add", "kind": "add"},
            {"name": "fc2", "kind": "dense", "units": 10},
        ],
    }
    spec = nsx.build_nnsvg_spec(spec_dict)
    labels = [l.label for l in spec.layers]
    assert "skip_add" not in labels, f"Add layer appeared as diagram column: {labels}"
