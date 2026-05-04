"""Enumerations used throughout NeuroSchemaX."""

from __future__ import annotations

from enum import Enum, auto


class RenderFamily(Enum):
    """NN-SVG rendering family. Each corresponds to one NN-SVG diagram type."""

    FCNN = "fcnn"
    LENET = "lenet"
    ALEXNET = "alexnet"


class OutputFormat(Enum):
    """Supported output formats."""

    HTML = "html"
    SVG = "svg"
    PNG = "png"
    NNSVG_JSON = "nnsvg_json"
    PAPER_JSON = "paper_json"
    DEBUG_JSON = "debug_json"
    TEXT = "text"
    MARKDOWN = "markdown"


class InputFormat(Enum):
    """Supported model input formats."""

    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    MANUAL_SPEC = "manual_spec"


class Theme(Enum):
    """Visual theme presets."""

    PAPER = "paper"
    THESIS = "thesis"
    DEBUG = "debug"
    README = "readme"


class LineStyle(Enum):
    """Connection line styles."""

    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"


class Orientation(Enum):
    """Layout orientation."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class LayoutMode(Enum):
    """Layout density mode."""

    COMPACT = "compact"
    PRESENTATION = "presentation"


class LabelDensity(Enum):
    """How many labels to display on the diagram."""

    NONE = "none"
    MINIMAL = "minimal"
    NORMAL = "normal"
    VERBOSE = "verbose"


class ConfidenceLevel(Enum):
    """Confidence in semantic interpretation of a layer or block."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class LayerKind(Enum):
    """Canonical layer/operation kinds recognised during normalisation."""

    INPUT = auto()
    CONV = auto()
    DEPTHWISE_CONV = auto()
    TRANSPOSED_CONV = auto()
    POOL_MAX = auto()
    POOL_AVG = auto()
    POOL_GLOBAL = auto()
    DENSE = auto()
    BATCH_NORM = auto()
    LAYER_NORM = auto()
    GROUP_NORM = auto()
    INSTANCE_NORM = auto()
    DROPOUT = auto()
    ACTIVATION = auto()
    RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()
    GELU = auto()
    FLATTEN = auto()
    RESHAPE = auto()
    CONCAT = auto()
    ADD = auto()
    MULTIPLY = auto()
    ATTENTION = auto()
    EMBEDDING = auto()
    RECURRENT = auto()
    LSTM = auto()
    GRU = auto()
    UPSAMPLE = auto()
    PAD = auto()
    OUTPUT = auto()
    UNKNOWN = auto()
