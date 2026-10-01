"""Runtime helpers for the Virtual-server (always read-only) dashboard."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dash import Dash

_CONFIG_KEY = "INTERACTIVE_RNASEQ_MODE"

# Lab data folder for gunicorn / default CLI.
DATA_ROOT = "/home/lab/data/Johannes/biofilm-microenvironments2"
SERVER_DEFAULT_DATA_ROOT = DATA_ROOT


@dataclass(frozen=True)
class AppMode:
    """Fixed data root for this process (always read-only viewer)."""

    data_root: str

    @property
    def default_project(self) -> str:
        return self.data_root


def normalize_url_base_pathname(raw: str | None) -> str | None:
    """Return a Dash ``url_base_pathname`` (leading + trailing slash) or None."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text == "/":
        return None
    if not text.startswith("/"):
        text = "/" + text
    if not text.endswith("/"):
        text = text + "/"
    return text


def data_root_from_env(default_project: str | None = None) -> str:
    """Resolve data root: explicit arg, then env, then ``DATA_ROOT``."""
    return (
        (default_project or "").strip()
        or (os.environ.get("DASH_DEFAULT_PROJECT") or "").strip()
        or (os.environ.get("DASH_DATA_ROOT") or "").strip()
        or DATA_ROOT
    )


def attach_mode(app: Dash, mode: AppMode) -> None:
    app.server.config[_CONFIG_KEY] = mode


def get_mode(app: Dash | None = None) -> AppMode:
    if app is not None:
        mode = app.server.config.get(_CONFIG_KEY)
        if isinstance(mode, AppMode):
            return mode
    return AppMode(data_root=data_root_from_env())
