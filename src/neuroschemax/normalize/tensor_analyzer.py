"""Analyse tensor shapes to derive feature-map dimensions and channel counts."""

from __future__ import annotations


def classify_shape(shape: list[int | str]) -> str:
    """Classify a tensor shape into a category.

    Returns one of: ``"scalar"``, ``"vector"``, ``"matrix"``, ``"3d"``,
    ``"4d"``, ``"5d+"``, or ``"dynamic"`` (if symbolic dims are present).
    """
    if any(isinstance(d, str) for d in shape):
        return "dynamic"
    ndim = len(shape)
    if ndim == 0:
        return "scalar"
    if ndim == 1:
        return "vector"
    if ndim == 2:
        return "matrix"
    if ndim == 3:
        return "3d"
    if ndim == 4:
        return "4d"
    return "5d+"


def extract_feature_map_dims(shape: list[int | str]) -> dict[str, int | None]:
    """Attempt to extract spatial / channel info from a shape (assumes NCHW)."""
    result: dict[str, int | None] = {
        "batch": None, "channels": None, "height": None, "width": None,
    }
    ints = [d for d in shape if isinstance(d, int)]
    if len(ints) == 4:
        result["batch"] = ints[0]
        result["channels"] = ints[1]
        result["height"] = ints[2]
        result["width"] = ints[3]
    elif len(ints) == 3:
        result["channels"] = ints[0]
        result["height"] = ints[1]
        result["width"] = ints[2]
    elif len(ints) == 2:
        result["batch"] = ints[0]
        result["channels"] = ints[1]
    elif len(ints) == 1:
        result["channels"] = ints[0]
    return result
