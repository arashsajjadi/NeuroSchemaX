"""Compatibility / feature-detection helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"

REQUIRED_ASSETS = [
    "util.js",
    "FCNN.js",
    "LeNet.js",
    "AlexNet.js",
]


def check_assets() -> dict[str, bool]:
    """Return a map of ``asset_name -> exists?`` for each required asset."""
    return {name: (_ASSETS_DIR / name).exists() for name in REQUIRED_ASSETS}


def check_playwright() -> bool:
    """Return True if Playwright is importable."""
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


def check_chromium() -> bool:
    """Return True if Playwright is importable AND the Chromium browser is installed."""
    if not check_playwright():
        return False
    try:
        from playwright.sync_api import sync_playwright
        p = sync_playwright().__enter__()
        exe = p.chromium.executable_path
        p.__exit__(None, None, None)
        return Path(exe).exists()
    except Exception:  # noqa: BLE001
        return False


def check_onnx() -> bool:
    """Return True if onnx is importable."""
    try:
        import onnx  # noqa: F401
        return True
    except ImportError:
        return False


def check_torch() -> bool:
    """Return True if torch is importable."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_tensorflow() -> bool:
    """Return True if tensorflow is importable."""
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        return False


def check_yaml() -> bool:
    """Return True if PyYAML is importable."""
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def _get_package_version() -> str:
    """Return the installed NeuroSchemaX version string."""
    try:
        from ..version import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


def environment_summary() -> dict[str, object]:
    """Return a structured environment status summary.

    The result has the following shape::

        {
            "status": "ok" | "partial" | "error",
            "version": "0.1.0",
            "python": "3.12.0",
            "assets": {"util.js": True, ...},
            "dependencies": {
                "onnx": True,
                "playwright": True,
                "chromium": False,
                "torch": False,
                "tensorflow": False,
                "yaml": True,
            },
            "messages": [
                "Chromium not installed. Run: playwright install chromium",
                ...
            ],
        }

    ``status`` is:

    - ``"ok"``      all required assets present and onnx importable
    - ``"partial"`` optional deps (playwright/torch/tf) missing, but core works
    - ``"error"``   required assets missing or onnx not importable
    """
    assets = check_assets()
    playwright_ok = check_playwright()
    chromium_ok = check_chromium()
    onnx_ok = check_onnx()
    torch_ok = check_torch()
    tf_ok = check_tensorflow()
    yaml_ok = check_yaml()

    messages: list[str] = []

    # Asset checks (required)
    for name, present in assets.items():
        if not present:
            messages.append(
                f"Required asset '{name}' is missing. "
                "Reinstall the package: pip install --force-reinstall neuroschemax"
            )

    # onnx (required for ONNX model input)
    if not onnx_ok:
        messages.append(
            "onnx not installed. ONNX model support requires: pip install onnx"
        )

    # yaml (required for YAML spec input, but ships with PyYAML which is a dep)
    if not yaml_ok:
        messages.append(
            "PyYAML not installed. YAML spec support requires: pip install PyYAML"
        )

    # Playwright (optional, required only for SVG export)
    if not playwright_ok:
        messages.append(
            "playwright not installed. SVG export requires: pip install playwright"
        )
    elif not chromium_ok:
        messages.append(
            "Chromium not installed. Run: playwright install chromium"
        )

    # torch (optional)
    if not torch_ok:
        messages.append(
            "torch not installed. PyTorch model support requires: pip install torch"
        )

    # tensorflow (optional)
    if not tf_ok:
        messages.append(
            "tensorflow not installed. "
            "TensorFlow/Keras model support requires: pip install tensorflow"
        )

    # Determine top-level status
    all_assets_ok = all(assets.values())
    if not all_assets_ok or not onnx_ok:
        status = "error"
    elif not playwright_ok or not chromium_ok or not torch_ok or not tf_ok:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "version": _get_package_version(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "assets": assets,
        "dependencies": {
            "onnx": onnx_ok,
            "playwright": playwright_ok,
            "chromium": chromium_ok,
            "torch": torch_ok,
            "tensorflow": tf_ok,
            "yaml": yaml_ok,
        },
        "messages": messages,
    }
