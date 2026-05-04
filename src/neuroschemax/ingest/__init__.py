"""Model ingestion adapters."""

from .base import AdapterRegistry, BaseAdapter, build_default_registry
from .manual_spec_adapter import ManualSpecAdapter
from .onnx_adapter import OnnxAdapter
from .pytorch_adapter import PyTorchAdapter
from .tensorflow_adapter import TensorFlowAdapter

__all__ = [
    "AdapterRegistry",
    "BaseAdapter",
    "ManualSpecAdapter",
    "OnnxAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "build_default_registry",
]
