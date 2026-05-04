"""Preset :class:`RenderConfig` builders."""

from ..core.config import RenderConfig
from ..core.enums import Theme
from .debug import debug_preset
from .paper import paper_preset
from .readme import readme_preset
from .thesis import thesis_preset

_PRESETS = {
    Theme.PAPER: paper_preset,
    Theme.THESIS: thesis_preset,
    Theme.DEBUG: debug_preset,
    Theme.README: readme_preset,
}


def get_preset(theme: Theme) -> RenderConfig:
    """Return the preset :class:`RenderConfig` for *theme*."""
    return _PRESETS[theme]()


__all__ = [
    "debug_preset",
    "get_preset",
    "paper_preset",
    "readme_preset",
    "thesis_preset",
]
