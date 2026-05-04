"""Tests that verify package data (JS assets and fonts) are present and accessible."""

from __future__ import annotations

from pathlib import Path

import pytest

# Path to the installed (or editable-installed) assets directory
_ASSETS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src" / "neuroschemax" / "visualization" / "assets"
)

REQUIRED_JS = ["util.js", "FCNN.js", "LeNet.js", "AlexNet.js"]
REQUIRED_FONTS = ["helvetiker_regular.typeface.json"]


# ---------------------------------------------------------------------------
# Asset existence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", REQUIRED_JS)
def test_assets_exist(filename: str):
    """Each required JS asset must exist on the filesystem."""
    asset_path = _ASSETS_DIR / filename
    assert asset_path.exists(), f"Missing required asset: {filename}"
    assert asset_path.stat().st_size > 0, f"Asset is empty: {filename}"


@pytest.mark.parametrize("filename", REQUIRED_FONTS)
def test_font_exists(filename: str):
    """Font JSON files must exist in the fonts sub-directory."""
    font_path = _ASSETS_DIR / "fonts" / filename
    assert font_path.exists(), f"Missing font file: {filename}"
    assert font_path.stat().st_size > 0, f"Font file is empty: {filename}"


# ---------------------------------------------------------------------------
# check_assets() helper
# ---------------------------------------------------------------------------

def test_compat_check_assets():
    """check_assets() must report all required assets as present."""
    from neuroschemax.visualization.compat import check_assets
    result = check_assets()
    assert isinstance(result, dict)
    for name, present in result.items():
        assert present, f"check_assets() reports missing: {name}"


# ---------------------------------------------------------------------------
# Generated HTML contains real JS content
# ---------------------------------------------------------------------------

def test_html_embeds_js():
    """Generated HTML must embed actual JS content, not an error placeholder."""
    import neuroschemax as nsx
    spec = {
        "model_name": "tiny_mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    }
    html = nsx.render_network_html(spec)
    # Must contain runtime hooks
    assert "__nnsvg_ready" in html
    assert "__nnsvg_export_svg" in html
    # Must NOT contain the error placeholder string
    assert "Asset not found" not in html
    # Should contain actual JS identifiers from FCNN.js / util.js
    assert "renderFCNN" in html or "renderNNSVG" in html or "FCNN" in html
