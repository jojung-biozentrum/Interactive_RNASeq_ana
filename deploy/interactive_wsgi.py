"""Drop-in WSGI module for the lab dashboard (read-only).

Prefer pointing gunicorn at ``src.GUI.wsgi:server``. If an existing unit must
load a file under ``~/www/dashboard/``, copy or symlink this module and use::

    gunicorn deploy.interactive_wsgi:server -b :8052

or from that directory after putting this file there as ``interactive.py``::

    gunicorn interactive:server -b :8052

Any ``create_app(url_base_pathname="/interactive/", ...)`` wrapper is also
forced read-only by ``create_app`` itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Re-use the forced-readonly entry (sets DASH_READONLY, clears DASH_WRITABLE).
from src.GUI.wsgi import app, server  # noqa: E402

__all__ = ["app", "server"]
