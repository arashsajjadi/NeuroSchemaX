"""Test that all documented public symbols import cleanly."""

from __future__ import annotations

import pytest


def test_import_package():
    import neuroschemax
    assert hasattr(neuroschemax, "__version__")


def test_import_public_api():
    from neuroschemax import (  # noqa: F401
        NNSVGSpec,
        RenderConfig,
        RenderFamily,
        SemanticArchitecture,
        Theme,
        build_nnsvg_spec,
        export_debug_json,
        export_nnsvg_spec,
        export_paper_json,
        parse_graph,
        parse_model,
        recommend_view,
        render_network_html,
        render_network_svg,
        save_debug_json,
        save_network_html,
        save_network_svg,
        save_nnsvg_spec,
        save_paper_json,
        summarize_model,
    )


def test_import_cli_main():
    from neuroschemax.cli import main  # noqa: F401
    assert callable(main)


def test_import_exceptions():
    from neuroschemax import (  # noqa: F401
        BrowserNotAvailableError,
        NeuroSchemaXError,
        ParseError,
        RenderError,
        UnsupportedFormatError,
    )


def test_import_submodules():
    import neuroschemax.api  # noqa: F401
    import neuroschemax.core  # noqa: F401
    import neuroschemax.exporters  # noqa: F401
    import neuroschemax.ingest  # noqa: F401
    import neuroschemax.ir  # noqa: F401
    import neuroschemax.normalize  # noqa: F401
    import neuroschemax.presets  # noqa: F401
    import neuroschemax.visualization  # noqa: F401


@pytest.mark.parametrize("symbol", [
    "parse_model", "summarize_model", "recommend_view",
    "build_nnsvg_spec", "render_network_html", "render_network_svg",
    "save_network_html", "save_network_svg",
    "export_paper_json", "export_debug_json", "export_nnsvg_spec",
    "save_paper_json", "save_debug_json", "save_nnsvg_spec",
    "RenderConfig", "RenderFamily", "Theme",
])
def test_public_symbol_exists(symbol: str):
    import neuroschemax
    assert hasattr(neuroschemax, symbol), f"{symbol} is missing"
    assert symbol in neuroschemax.__all__, f"{symbol} is not in __all__"
