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
            "version": "0.1.1",
            "python": "3.12.0",
            "assets": {"util.js": True, ...},
            "capabilities": {
                "html_export": True,     # always True when assets are present
                "svg_export": False,     # requires Playwright + Chromium
                "onnx_input": True,
                "torch_input": False,
                "tensorflow_input": False,
                "yaml_input": True,
                "notebook_display": False,
            },
            "dependencies": {
                "onnx": True,
                "playwright": True,
                "chromium": False,
                "torch": False,
                "tensorflow": False,
                "yaml": True,
                "ipython": False,
            },
            "messages": [
                "Chromium not installed. Run: playwright install chromium",
                ...
            ],
        }

    ``status`` is:

    - ``"ok"``      all required assets present and ONNX importable
    - ``"partial"`` optional deps (playwright/torch/tf/ipython) missing; core works
    - ``"error"``   required assets missing or ONNX not importable
    """
    assets = check_assets()
    playwright_ok = check_playwright()
    chromium_ok = check_chromium()
    onnx_ok = check_onnx()
    torch_ok = check_torch()
    tf_ok = check_tensorflow()
    yaml_ok = check_yaml()

    # Test the actual display API, not just the package name.
    # ``import ipython`` (lowercase) may fail even when the display API is
    # available (Colab installs IPython without registering the lowercase name).
    from .._display import _can_display_html
    ipython_ok = _can_display_html()

    messages: list[str] = []
    all_assets_ok = all(assets.values())

    # Asset checks (required for any rendering)
    for name, present in assets.items():
        if not present:
            messages.append(
                f"Required asset '{name}' is missing. "
                "Reinstall: pip install --force-reinstall neuroschemax"
            )

    # ONNX (required for .onnx file input)
    if not onnx_ok:
        messages.append(
            "onnx not installed. ONNX input requires: pip install onnx  "
            "(or: pip install neuroschemax[onnx])"
        )

    # YAML
    if not yaml_ok:
        messages.append(
            "PyYAML not installed. YAML spec input requires: pip install PyYAML"
        )

    # SVG export path
    if not playwright_ok:
        messages.append(
            "playwright not installed. SVG export requires: "
            "pip install playwright && playwright install chromium  "
            "(or: pip install neuroschemax[svg])"
        )
    elif not chromium_ok:
        messages.append(
            "Chromium not found. SVG export requires: playwright install chromium"
        )

    # PyTorch
    if not torch_ok:
        messages.append(
            "torch not installed. PyTorch model input requires: "
            "pip install torch  (or: pip install neuroschemax[torch])"
        )

    # TensorFlow
    if not tf_ok:
        messages.append(
            "tensorflow not installed. TF/Keras input requires: "
            "pip install tensorflow  (or: pip install neuroschemax[tf])"
        )

    # Notebook / Colab inline display
    if not ipython_ok:
        messages.append(
            "IPython not installed. Inline notebook/Colab display requires: "
            "pip install ipython  (or: pip install neuroschemax[colab])"
        )

    # Capabilities summary (easier to check than raw deps)
    html_ok = all_assets_ok
    svg_ok = playwright_ok and chromium_ok

    capabilities = {
        "html_export": html_ok,
        "svg_export": svg_ok,
        "onnx_input": onnx_ok,
        "torch_input": torch_ok,
        "tensorflow_input": tf_ok,
        "yaml_input": yaml_ok,
        "notebook_display": ipython_ok,
    }

    # Status
    if not all_assets_ok or not onnx_ok:
        status = "error"
    elif not svg_ok or not torch_ok or not tf_ok or not ipython_ok:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "version": _get_package_version(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "assets": assets,
        "capabilities": capabilities,
        "dependencies": {
            "onnx": onnx_ok,
            "playwright": playwright_ok,
            "chromium": chromium_ok,
            "torch": torch_ok,
            "tensorflow": tf_ok,
            "yaml": yaml_ok,
            "ipython": ipython_ok,
        },
        "messages": messages,
    }
