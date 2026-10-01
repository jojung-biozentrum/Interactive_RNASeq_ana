"""Gunicorn WSGI entry for the virtual-server dashboard.

    gunicorn src.GUI.wsgi:server -b 127.0.0.1:8052

Always read-only; data root defaults to the lab folder (override with env).

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
from src.GUI.runtime import DATA_ROOT, normalize_url_base_pathname

app = create_app(
    default_project=(
        os.environ.get("DASH_DEFAULT_PROJECT")
        or os.environ.get("DASH_DATA_ROOT")
        or DATA_ROOT
    ),
    url_base_pathname=normalize_url_base_pathname(
        os.environ.get("DASH_URL_BASE_PATHNAME", "/interactive/")
    ),
    secret_config=os.environ.get("DASH_SECRET_CONFIG", ""),
)
server = app.server
