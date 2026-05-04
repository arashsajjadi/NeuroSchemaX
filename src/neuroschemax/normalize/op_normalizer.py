"""Map raw op-type strings to canonical :class:`LayerKind` values."""

from __future__ import annotations

from ..core.enums import ConfidenceLevel, LayerKind

# Mapping of raw op-type strings → (LayerKind, ConfidenceLevel).
# Keys are lowercased for matching.
_OP_TABLE: dict[str, tuple[LayerKind, ConfidenceLevel]] = {
    # Convolution
    "conv": (LayerKind.CONV, ConfidenceLevel.HIGH),
    "conv1d": (LayerKind.CONV, ConfidenceLevel.HIGH),
    "conv2d": (LayerKind.CONV, ConfidenceLevel.HIGH),
    "conv3d": (LayerKind.CONV, ConfidenceLevel.HIGH),
    "convolution": (LayerKind.CONV, ConfidenceLevel.HIGH),
    "depthwiseconv2d": (LayerKind.DEPTHWISE_CONV, ConfidenceLevel.HIGH),
    "convtranspose": (LayerKind.TRANSPOSED_CONV, ConfidenceLevel.HIGH),
    "convtranspose2d": (LayerKind.TRANSPOSED_CONV, ConfidenceLevel.HIGH),
    "deconv": (LayerKind.TRANSPOSED_CONV, ConfidenceLevel.MEDIUM),
    # Pooling
    "maxpool": (LayerKind.POOL_MAX, ConfidenceLevel.HIGH),
    "maxpool2d": (LayerKind.POOL_MAX, ConfidenceLevel.HIGH),
    "maxpooling2d": (LayerKind.POOL_MAX, ConfidenceLevel.HIGH),
    "averagepool": (LayerKind.POOL_AVG, ConfidenceLevel.HIGH),
    "avgpool2d": (LayerKind.POOL_AVG, ConfidenceLevel.HIGH),
    "averagepooling2d": (LayerKind.POOL_AVG, ConfidenceLevel.HIGH),
    "globalaveragepool": (LayerKind.POOL_GLOBAL, ConfidenceLevel.HIGH),
    "globalaveragepooling2d": (LayerKind.POOL_GLOBAL, ConfidenceLevel.HIGH),
    "globalmaxpool": (LayerKind.POOL_GLOBAL, ConfidenceLevel.HIGH),
    "adaptiveavgpool2d": (LayerKind.POOL_GLOBAL, ConfidenceLevel.MEDIUM),
    # Dense / Linear
    "gemm": (LayerKind.DENSE, ConfidenceLevel.HIGH),
    "matmul": (LayerKind.DENSE, ConfidenceLevel.MEDIUM),
    "linear": (LayerKind.DENSE, ConfidenceLevel.HIGH),
    "dense": (LayerKind.DENSE, ConfidenceLevel.HIGH),
    "fc": (LayerKind.DENSE, ConfidenceLevel.HIGH),
    # Normalization
    "batchnormalization": (LayerKind.BATCH_NORM, ConfidenceLevel.HIGH),
    "batchnorm2d": (LayerKind.BATCH_NORM, ConfidenceLevel.HIGH),
    "batchnorm1d": (LayerKind.BATCH_NORM, ConfidenceLevel.HIGH),
    "layernormalization": (LayerKind.LAYER_NORM, ConfidenceLevel.HIGH),
    "layernorm": (LayerKind.LAYER_NORM, ConfidenceLevel.HIGH),
    "groupnorm": (LayerKind.GROUP_NORM, ConfidenceLevel.HIGH),
    "instancenormalization": (LayerKind.INSTANCE_NORM, ConfidenceLevel.HIGH),
    "instancenorm2d": (LayerKind.INSTANCE_NORM, ConfidenceLevel.HIGH),
    # Activation
    "relu": (LayerKind.RELU, ConfidenceLevel.HIGH),
    "sigmoid": (LayerKind.SIGMOID, ConfidenceLevel.HIGH),
    "tanh": (LayerKind.TANH, ConfidenceLevel.HIGH),
    "softmax": (LayerKind.SOFTMAX, ConfidenceLevel.HIGH),
    "gelu": (LayerKind.GELU, ConfidenceLevel.HIGH),
    "leakyrelu": (LayerKind.ACTIVATION, ConfidenceLevel.HIGH),
    "elu": (LayerKind.ACTIVATION, ConfidenceLevel.HIGH),
    "prelu": (LayerKind.ACTIVATION, ConfidenceLevel.HIGH),
    "selu": (LayerKind.ACTIVATION, ConfidenceLevel.HIGH),
    "silu": (LayerKind.ACTIVATION, ConfidenceLevel.HIGH),
    "swish": (LayerKind.ACTIVATION, ConfidenceLevel.HIGH),
    "mish": (LayerKind.ACTIVATION, ConfidenceLevel.HIGH),
    "activation": (LayerKind.ACTIVATION, ConfidenceLevel.MEDIUM),
    # Reshape / Flatten
    "flatten": (LayerKind.FLATTEN, ConfidenceLevel.HIGH),
    "reshape": (LayerKind.RESHAPE, ConfidenceLevel.HIGH),
    "squeeze": (LayerKind.RESHAPE, ConfidenceLevel.MEDIUM),
    "unsqueeze": (LayerKind.RESHAPE, ConfidenceLevel.MEDIUM),
    "transpose": (LayerKind.RESHAPE, ConfidenceLevel.MEDIUM),
    # Merge / Skip
    "add": (LayerKind.ADD, ConfidenceLevel.HIGH),
    "concat": (LayerKind.CONCAT, ConfidenceLevel.HIGH),
    "concatenate": (LayerKind.CONCAT, ConfidenceLevel.HIGH),
    "mul": (LayerKind.MULTIPLY, ConfidenceLevel.HIGH),
    "multiply": (LayerKind.MULTIPLY, ConfidenceLevel.HIGH),
    # Dropout
    "dropout": (LayerKind.DROPOUT, ConfidenceLevel.HIGH),
    # Recurrent
    "lstm": (LayerKind.LSTM, ConfidenceLevel.HIGH),
    "gru": (LayerKind.GRU, ConfidenceLevel.HIGH),
    "rnn": (LayerKind.RECURRENT, ConfidenceLevel.HIGH),
    # Attention
    "multiheadattention": (LayerKind.ATTENTION, ConfidenceLevel.HIGH),
    "attention": (LayerKind.ATTENTION, ConfidenceLevel.MEDIUM),
    # Other
    "embedding": (LayerKind.EMBEDDING, ConfidenceLevel.HIGH),
    "upsample": (LayerKind.UPSAMPLE, ConfidenceLevel.HIGH),
    "resize": (LayerKind.UPSAMPLE, ConfidenceLevel.MEDIUM),
    "pad": (LayerKind.PAD, ConfidenceLevel.HIGH),
    "input": (LayerKind.INPUT, ConfidenceLevel.HIGH),
    "inputlayer": (LayerKind.INPUT, ConfidenceLevel.HIGH),
    "output": (LayerKind.OUTPUT, ConfidenceLevel.HIGH),
}


def normalize_op(op_type: str) -> tuple[LayerKind, ConfidenceLevel]:
    """Resolve a raw op-type string to a canonical ``(LayerKind, confidence)``."""
    key = op_type.lower().replace("_", "").replace("-", "").strip()
    if key in _OP_TABLE:
        return _OP_TABLE[key]
    # Fuzzy fallbacks
    for pattern, result in [
        ("conv", (LayerKind.CONV, ConfidenceLevel.LOW)),
        ("pool", (LayerKind.POOL_MAX, ConfidenceLevel.LOW)),
        ("norm", (LayerKind.BATCH_NORM, ConfidenceLevel.LOW)),
        ("linear", (LayerKind.DENSE, ConfidenceLevel.LOW)),
        ("attention", (LayerKind.ATTENTION, ConfidenceLevel.LOW)),
    ]:
        if pattern in key:
            return result
    return (LayerKind.UNKNOWN, ConfidenceLevel.UNKNOWN)
