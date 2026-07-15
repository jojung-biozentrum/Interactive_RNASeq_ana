"""Analysis module protocol for pluggable GUI tabs."""

from __future__ import annotations

from typing import Protocol

from dash import Dash
from dash.development.base_component import Component


class AnalysisModule(Protocol):
    """Contract for analysis tabs. Implement and register in modules/__init__.py."""

    id: str
    label: str

    def layout(self) -> Component:
        """Return the tab body layout."""
        ...

    def register_callbacks(self, app: Dash) -> None:
        """Register Dash callbacks for this module."""
        ...
