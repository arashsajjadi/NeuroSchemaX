"""Verify that all example scripts exist and are syntactically valid Python."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

EXPECTED_EXAMPLES = [
    "tiny_mlp_to_html.py",
    "tiny_cnn_to_svg.py",
    "manual_spec_to_svg.py",
    "export_nnsvg_json.py",
    "summarize_model.py",
    "recommend_view.py",
]


@pytest.mark.parametrize("name", EXPECTED_EXAMPLES)
def test_example_exists(name: str):
    path = EXAMPLES_DIR / name
    assert path.is_file(), f"Example script missing: {path}"


@pytest.mark.parametrize("name", EXPECTED_EXAMPLES)
def test_example_is_valid_python(name: str):
    path = EXAMPLES_DIR / name
    source = path.read_text()
    # Will raise SyntaxError if the file is not valid Python
    ast.parse(source)


def test_sample_configs_exist():
    cfg_dir = Path(__file__).resolve().parents[1] / "sample_configs"
    for name in ("paper.yaml", "thesis.yaml", "debug.yaml", "readme.yaml"):
        assert (cfg_dir / name).is_file(), f"Missing: {name}"
