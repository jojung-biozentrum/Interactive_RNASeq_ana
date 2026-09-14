"""Analysis module protocol for pluggable GUI tabs."""

from __future__ import annotations

from typing import Protocol

from dash import Dash
from dash.development.base_component import Component


class AnalysisModule(Protocol):
    """Contract for analysis tabs. Implement and register in modules/__init__.py."""

    id: str
    label: str

    def layout(self, *, readonly: bool = False) -> Component:
        """Return the tab body layout (``readonly`` hides disk-write UI)."""
        ...

    def register_callbacks(self, app: Dash, *, readonly: bool = False) -> None:
        """Register Dash callbacks (``readonly`` skips browse/save/export)."""
        ...
