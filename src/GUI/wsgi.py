"""Gunicorn WSGI entry for the virtual-server dashboard.

    gunicorn src.GUI.wsgi:server -b :8052

Environment:

    DASH_URL_BASE_PATHNAME   nginx location, default ``/interactive/``
    DASH_DEFAULT_PROJECT     override the fixed data folder (see app.DATA_ROOT)
    DASH_SECRET_CONFIG       TOML with [auth] user / pwd; omit to skip login
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.GUI.app import create_app, normalize_url_base_pathname

app = create_app(
    default_project=os.environ.get("DASH_DEFAULT_PROJECT") or None,
    url_base_pathname=normalize_url_base_pathname(
        os.environ.get("DASH_URL_BASE_PATHNAME", "/interactive/")
    ),
    secret_config=os.environ.get("DASH_SECRET_CONFIG") or None,
)
server = app.server
