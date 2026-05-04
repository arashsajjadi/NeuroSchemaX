"""NeuroSchemaX exception hierarchy."""


class NeuroSchemaXError(Exception):
    """Base exception for all NeuroSchemaX errors."""


class ParseError(NeuroSchemaXError):
    """Raised when a model cannot be parsed."""


class UnsupportedFormatError(ParseError):
    """Raised when the input format is not supported."""


class AdapterImportError(NeuroSchemaXError):
    """Raised when an optional adapter dependency is missing."""

    def __init__(self, adapter_name: str, package: str) -> None:
        super().__init__(
            f"The '{adapter_name}' adapter requires the '{package}' package. "
            f"Install it with: pip install {package}"
        )
        self.adapter_name = adapter_name
        self.package = package


class ValidationError(NeuroSchemaXError):
    """Raised when configuration or input validation fails."""


class RenderError(NeuroSchemaXError):
    """Raised when visualization rendering fails."""


class BrowserNotAvailableError(RenderError):
    """Raised when headless browser is needed but not installed."""

    def __init__(self) -> None:
        super().__init__(
            "SVG export requires a headless browser. Install Playwright:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )


class ExportError(NeuroSchemaXError):
    """Raised when an export operation fails."""


class AssetError(NeuroSchemaXError):
    """Raised when required JS/font assets are missing or corrupt."""
