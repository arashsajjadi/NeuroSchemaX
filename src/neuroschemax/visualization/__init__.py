"""Visualisation layer: NN-SVG spec, HTML generation, and SVG extraction."""

from .compat import check_assets, environment_summary
from .nnsvg_html import generate_html
from .nnsvg_mapper import map_to_nnsvg
from .nnsvg_runtime import (
    extract_svg_from_html,
    is_playwright_available,
    save_svg_from_html,
)
from .nnsvg_schema import NNSVGLayerSpec, NNSVGSpec
from .text import markdown_summary, text_summary

__all__ = [
    "NNSVGLayerSpec",
    "NNSVGSpec",
    "check_assets",
    "environment_summary",
    "extract_svg_from_html",
    "generate_html",
    "is_playwright_available",
    "map_to_nnsvg",
    "markdown_summary",
    "save_svg_from_html",
    "text_summary",
]
