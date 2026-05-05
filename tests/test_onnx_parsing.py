"""Regression tests for ONNX adapter parsing robustness."""

from __future__ import annotations

import pytest

import neuroschemax as nsx
from neuroschemax.core.enums import LayerKind


def _make_minimal_onnx():
    """Build an in-memory ONNX model with Conv + Relu."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx not available")

    conv = helper.make_node("Conv", ["x", "w", "b"], ["conv_out"],
                            kernel_shape=[3, 3], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["conv_out"], ["out"])

    x   = helper.make_tensor_value_info("x",   TensorProto.FLOAT, [1, 1, 28, 28])
    w_i = helper.make_tensor("w", TensorProto.FLOAT, [16, 1, 3, 3], [0.0] * 576)
    b_i = helper.make_tensor("b", TensorProto.FLOAT, [16], [0.0] * 16)
    w_v = helper.make_tensor_value_info("w", TensorProto.FLOAT, [16, 1, 3, 3])
    b_v = helper.make_tensor_value_info("b", TensorProto.FLOAT, [16])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, None)

    graph = helper.make_graph(
        [conv, relu], "tiny",
        [x, w_v, b_v], [out],
        initializer=[w_i, b_i],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    model.ir_version = 8
    return model


def _make_gemm_onnx():
    """Build a Gemm (Linear) ONNX node with known input/output sizes."""
    try:
        from onnx import TensorProto, helper
    except ImportError:
        pytest.skip("onnx not available")

    gemm = helper.make_node("Gemm", ["x", "W", "B"], ["out"],
                            transB=1)
    x    = helper.make_tensor_value_info("x",   TensorProto.FLOAT, [1, 128])
    W_i  = helper.make_tensor("W", TensorProto.FLOAT, [10, 128], [0.0] * 1280)
    B_i  = helper.make_tensor("B", TensorProto.FLOAT, [10], [0.0] * 10)
    W_v  = helper.make_tensor_value_info("W",  TensorProto.FLOAT, [10, 128])
    B_v  = helper.make_tensor_value_info("B",  TensorProto.FLOAT, [10])
    out  = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 10])

    graph = helper.make_graph(
        [gemm], "gemm_model",
        [x, W_v, B_v], [out],
        initializer=[W_i, B_i],
    )
    try:
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 8
    except Exception:
        pytest.skip("could not build Gemm ONNX model")
    return model


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOnnxAdapterRobustness:
    def test_minimal_conv_relu_parses(self):
        """Minimal Conv+Relu ONNX model parses without error."""
        model = _make_minimal_onnx()
        arch = nsx.parse_model(model)
        assert arch is not None
        kinds = [lay.kind for lay in arch.layers]
        assert LayerKind.CONV in kinds
        assert LayerKind.RELU in kinds

    def test_conv_output_shape_inferred(self):
        """ONNX shape inference populates Conv output shape."""
        model = _make_minimal_onnx()
        arch = nsx.parse_model(model)
        conv = next(lay for lay in arch.layers if lay.kind == LayerKind.CONV)
        # Conv2d: 28x28 input, 3x3 kernel, no padding → 26x26 output, 16 channels
        assert conv.output_shape == [1, 16, 26, 26]

    def test_conv_channels_extracted(self):
        """Conv layer channels should be inferred from output shape."""
        model = _make_minimal_onnx()
        arch = nsx.parse_model(model)
        conv = next(lay for lay in arch.layers if lay.kind == LayerKind.CONV)
        assert conv.channels == 16

    def test_gemm_units_inferred(self):
        """Gemm (Linear) layer units inferred from ONNX shape inference."""
        model = _make_gemm_onnx()
        arch = nsx.parse_model(model)
        dense = next(lay for lay in arch.layers if lay.kind == LayerKind.DENSE)
        assert dense.units == 10

    def test_no_crash_on_dynamic_shape(self, tmp_path):
        """Dynamic/symbolic ONNX shapes (batch='N') do not crash parsing."""
        try:
            from onnx import TensorProto, helper
        except ImportError:
            pytest.skip("onnx not available")
        relu = helper.make_node("Relu", ["x"], ["out"])
        x   = helper.make_tensor_value_info("x",   TensorProto.FLOAT, ["N", 3, 224, 224])
        out = helper.make_tensor_value_info("out",  TensorProto.FLOAT, None)
        graph = helper.make_graph([relu], "dyn", [x], [out])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 8
        arch = nsx.parse_model(model)
        assert len(arch.layers) > 0
        relu_l = arch.layers[0]
        # Symbolic dim should survive as a string, not cause a crash
        shape = relu_l.input_shape or relu_l.output_shape
        assert any(isinstance(d, (int, str)) for d in shape)

    def test_onnx_file_parsing(self, tmp_path):
        """ONNX file on disk parses without errors."""
        try:
            import onnx
        except ImportError:
            pytest.skip("onnx not available")
        model = _make_minimal_onnx()
        path = tmp_path / "test.onnx"
        onnx.save(model, str(path))
        arch = nsx.parse_model(str(path))
        assert arch.model_name  # non-empty

    def test_onnx_debug_json_preserves_layers(self, tmp_path):
        """Debug JSON for ONNX model includes Conv and Relu layers."""
        import json
        model = _make_minimal_onnx()
        path = tmp_path / "debug.json"
        nsx.save_debug_json(path, model)
        data = json.loads(path.read_text())
        kinds = [entry["kind"] for entry in data["layers"]]
        assert "conv" in kinds or "Conv" in " ".join(kinds)


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("torch"),
    reason="torch not installed",
)
class TestPyTorchOnnxRoundtrip:
    """Tests that require torch — skipped cleanly when torch is absent."""

    def test_torch_cnn_via_onnx(self, tmp_path):
        """TinyCNN exported with torch.onnx.export parses through NeuroSchemaX.

        Note: torch >= 2.x requires ``onnxscript`` for the new exporter path.
        If it is not available this test is skipped gracefully.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")

        try:
            import onnxscript  # type: ignore[import]  # noqa: F401
        except ImportError:
            pytest.skip("onnxscript not installed; needed by torch.onnx.export in torch>=2.x")

        class TinyCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(1, 8, 3)
                self.fc   = nn.Linear(8 * 26 * 26, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.conv(x)
                return self.fc(x.view(x.size(0), -1))

        model = TinyCNN().eval()
        dummy = torch.zeros(1, 1, 28, 28)
        onnx_path = tmp_path / "tiny_cnn.onnx"
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"],
            opset_version=17,
        )
        arch = nsx.parse_model(str(onnx_path))
        kinds = [lay.kind for lay in arch.layers]
        assert LayerKind.CONV in kinds
        assert LayerKind.DENSE in kinds
