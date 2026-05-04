"""Extract semantic attributes (units, channels, kernel_size, etc.) from raw node attributes."""

from __future__ import annotations

from typing import Any

from ..core.enums import LayerKind


def _int_or_none(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _int_list(val: Any) -> list[int] | None:
    if val is None:
        return None
    if isinstance(val, int):
        return [val]
    if isinstance(val, (list, tuple)):
        try:
            return [int(v) for v in val]
        except (TypeError, ValueError):
            return None
    return None


def extract_units(attrs: dict[str, Any], kind: LayerKind) -> int | None:
    """Extract the number of output units/neurons for a Dense-like layer."""
    if kind == LayerKind.DENSE:
        for key in ("out_features", "units", "transB_out"):
            if key in attrs:
                return _int_or_none(attrs[key])
    return None


def extract_channels(attrs: dict[str, Any], kind: LayerKind) -> int | None:
    """Extract the number of output channels for a Conv-like layer."""
    if kind in (LayerKind.CONV, LayerKind.DEPTHWISE_CONV, LayerKind.TRANSPOSED_CONV):
        for key in ("out_channels", "filters", "channels"):
            if key in attrs:
                return _int_or_none(attrs[key])
    return None


def extract_kernel_size(attrs: dict[str, Any]) -> list[int] | None:
    for key in ("kernel_size", "kernel_shape"):
        if key in attrs:
            return _int_list(attrs[key])
    return None


def extract_stride(attrs: dict[str, Any]) -> list[int] | None:
    for key in ("stride", "strides"):
        if key in attrs:
            return _int_list(attrs[key])
    return None


def extract_padding(attrs: dict[str, Any]) -> list[int] | None:
    for key in ("padding", "pads"):
        val = attrs.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            return None  # e.g. "same", "valid"
        return _int_list(val)
    return None


def extract_activation(attrs: dict[str, Any]) -> str | None:
    if "activation" in attrs:
        act = attrs["activation"]
        if isinstance(act, str) and act.lower() not in ("none", "linear", ""):
            return act.lower()
    return None
