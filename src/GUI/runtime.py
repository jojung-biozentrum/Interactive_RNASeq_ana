"""Runtime mode shared by desktop and gunicorn/nginx deployments.

On the ``Virtual-server`` branch the **safe default is read-only**. Writable
desktop UI requires an explicit opt-in (``--writable`` / ``DASH_WRITABLE=1``).
That way a mistaken ``python -m src.GUI.app`` on the lab VM cannot expose
register / export / Filter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dash import Dash

_CONFIG_KEY = "INTERACTIVE_RNASEQ_READONLY"
_DEFAULT_DATA_ROOT = "/home/lab/data/Johannes/biofilm-microenvironments2"
# Public alias for app.py / deploy (readonly fallback).
SERVER_DEFAULT_DATA_ROOT = _DEFAULT_DATA_ROOT

# Virtual-server branch: default to readonly. On ``main`` this should be False.
DEPLOYMENT_READONLY_DEFAULT = True


@dataclass(frozen=True)
class AppMode:
    """How the Dash app is presented and which write paths are enabled."""

    readonly: bool = True
    data_root: str | None = None  # fixed folder when set (server / --project)

    @property
    def default_project(self) -> str | None:
        return self.data_root


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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


def mode_from_env(
    *,
    readonly: bool | None = None,
    default_project: str | None = None,
) -> AppMode:
    """Resolve mode from explicit kwargs, then environment.

    Precedence for readonly:
    1. Explicit ``readonly=`` kwarg (``True`` / ``False``)
    2. ``DASH_WRITABLE=1`` → writable (escape hatch)
    3. ``DASH_READONLY`` if set
    4. ``DEPLOYMENT_READONLY_DEFAULT`` (True on Virtual-server)
    """
    if readonly is not None:
        ro = bool(readonly)
    elif env_flag("DASH_WRITABLE", False):
        ro = False
    elif os.environ.get("DASH_READONLY") is not None and str(
        os.environ.get("DASH_READONLY")
    ).strip() != "":
        ro = env_flag("DASH_READONLY", True)
    else:
        ro = DEPLOYMENT_READONLY_DEFAULT

    root = default_project or os.environ.get("DASH_DEFAULT_PROJECT") or None
    if ro and not root:
        root = os.environ.get("DASH_DATA_ROOT") or _DEFAULT_DATA_ROOT
    return AppMode(readonly=ro, data_root=root)


def attach_mode(app: Dash, mode: AppMode) -> None:
    app.server.config[_CONFIG_KEY] = mode


def get_mode(app: Dash | None = None) -> AppMode:
    if app is not None:
        mode = app.server.config.get(_CONFIG_KEY)
        if isinstance(mode, AppMode):
            return mode
    return mode_from_env()


def is_readonly(app: Dash | None = None) -> bool:
    return get_mode(app).readonly
