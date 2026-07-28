"""Shared command-line interface helpers."""

from __future__ import annotations

import argparse

from aep_load_forecasting import __version__


def add_version_argument(parser: argparse.ArgumentParser) -> None:
    """Add a standard package-version option to an argument parser."""

    parser.add_argument(
        "--version",
        action="version",
        version=f"aep-load-forecasting {__version__}",
    )
