"""Gunicorn WSGI entry for the virtual-server dashboard.

    gunicorn src.GUI.wsgi:server -b :8052

This module **forces** read-only mode (overrides a mistaken DASH_READONLY=0).
Writable UI is never available through this entry point.

Environment:

    DASH_URL_BASE_PATHNAME   nginx location, default ``/interactive/``
    DASH_DEFAULT_PROJECT     override the fixed data folder
    DASH_SECRET_CONFIG       TOML with [auth] user / pwd; required
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.GUI.app import create_app
from src.GUI.runtime import SERVER_DEFAULT_DATA_ROOT, normalize_url_base_pathname

# Force readonly — do not use setdefault (that would keep a bad DASH_READONLY=0).
os.environ["DASH_READONLY"] = "1"
os.environ.pop("DASH_WRITABLE", None)

app = create_app(
    default_project=(
        os.environ.get("DASH_DEFAULT_PROJECT")
        or os.environ.get("DASH_DATA_ROOT")
        or SERVER_DEFAULT_DATA_ROOT
    ),
    readonly=True,
    url_base_pathname=normalize_url_base_pathname(
        os.environ.get("DASH_URL_BASE_PATHNAME", "/interactive/")
    ),
    secret_config=os.environ.get("DASH_SECRET_CONFIG", ""),
)
server = app.server
