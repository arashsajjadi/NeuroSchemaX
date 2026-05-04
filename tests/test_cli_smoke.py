"""Smoke tests for the CLI using a tiny manual-spec model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuroschemax.cli import main


@pytest.fixture
def tiny_spec(tmp_path: Path) -> Path:
    spec = {
        "model_name": "tiny_mlp",
        "layers": [
            {"name": "input", "kind": "input", "shape": [1, 784]},
            {"name": "fc1", "kind": "dense", "units": 128},
            {"name": "relu1", "kind": "relu"},
            {"name": "fc2", "kind": "dense", "units": 64},
            {"name": "relu2", "kind": "relu"},
            {"name": "out", "kind": "dense", "units": 10},
        ],
    }
    p = tmp_path / "tiny_mlp.json"
    p.write_text(json.dumps(spec))
    return p


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_cli_version():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_cli_inspect(tiny_spec: Path, capsys: pytest.CaptureFixture[str]):
    rc = main(["inspect", str(tiny_spec)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tiny_mlp" in out
    assert "Layer count" in out


def test_cli_summarize_text(tiny_spec: Path, capsys: pytest.CaptureFixture[str]):
    rc = main(["summarize", str(tiny_spec)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tiny_mlp" in out
    assert "fc1" in out


def test_cli_summarize_markdown(tiny_spec: Path, capsys: pytest.CaptureFixture[str]):
    rc = main(["summarize", str(tiny_spec), "--format", "markdown"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# tiny_mlp" in out
    assert "| # |" in out


def test_cli_recommend_view(tiny_spec: Path, capsys: pytest.CaptureFixture[str]):
    rc = main(["recommend-view", str(tiny_spec)])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["family"] == "fcnn"


def test_cli_render_html(tiny_spec: Path, tmp_path: Path):
    out = tmp_path / "diagram.html"
    rc = main(["render", str(tiny_spec), "-o", str(out)])
    assert rc == 0
    assert out.exists()
    html = out.read_text()
    assert "__nnsvg_ready" in html
    assert "__nnsvg_export_svg" in html


def test_cli_export_paper_json(tiny_spec: Path, tmp_path: Path):
    out = tmp_path / "paper.json"
    rc = main(["export-paper-json", str(tiny_spec), "-o", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["model_name"] == "tiny_mlp"
    assert data["recommended_family"] == "fcnn"


def test_cli_export_debug_json(tiny_spec: Path, tmp_path: Path):
    out = tmp_path / "debug.json"
    rc = main(["export-debug-json", str(tiny_spec), "-o", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["model_name"] == "tiny_mlp"
    assert "layers" in data


def test_cli_export_nnsvg(tiny_spec: Path, tmp_path: Path):
    out = tmp_path / "nnsvg.json"
    rc = main(["export-nnsvg", str(tiny_spec), "-o", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["family"] == "fcnn"
    assert len(data["layers"]) > 0


def test_cli_doctor(capsys: pytest.CaptureFixture[str]):
    rc = main(["doctor"])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert "assets" in info
