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
    """Raised when headless browser is needed but not installed.

    Carries a user-friendly message with exact install commands.
    Can be constructed with a custom message (e.g. when Playwright is
    installed but Chromium is missing).
    """

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "SVG export requires Playwright and Chromium.\n"
                "Install:  pip install playwright\n"
                "Then:     playwright install chromium\n"
                "Or:       pip install \"neuroschemax[svg]\"\n"
                "\nStandalone HTML export works without any browser."
            )
        super().__init__(message)


class ExportError(NeuroSchemaXError):
    """Raised when an export operation fails."""


class AssetError(NeuroSchemaXError):
    """Raised when required JS/font assets are missing or corrupt."""
