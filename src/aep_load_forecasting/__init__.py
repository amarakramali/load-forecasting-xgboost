"""AEP load-forecasting package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aep-load-forecasting")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
