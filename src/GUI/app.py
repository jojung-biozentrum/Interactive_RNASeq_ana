"""Dash entry point — Virtual-server read-only viewer.

    python -m src.GUI.app --secret-config path/to/secret.toml
    gunicorn src.GUI.wsgi:server -b 127.0.0.1:8052
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python src/GUI/app.py` as well as `python -m src.GUI.app`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from src.GUI.components.dataset_picker import dataset_picker_layout
from src.GUI.data_store import load_selected, session_to_store
from src.GUI.modules import MODULE_REGISTRY
from src.GUI.project import DatasetEntry, open_project
from src.GUI.runtime import (
    DATA_ROOT,
    AppMode,
    attach_mode,
    data_root_from_env,
    normalize_url_base_pathname,
)


def _ds_opts(datasets: list) -> list[dict]:
    names = []
    for d in datasets:
        names.append(d["name"] if isinstance(d, dict) else d.name)
    return [{"label": n, "value": n} for n in names]


def _first_ds(datasets: list) -> str | None:
    opts = _ds_opts(datasets)
    return opts[0]["value"] if opts else None


def _load_project_blob(root: str) -> tuple[dict | None, str, list, str | None]:
    try:
        project = open_project(root)
    except Exception as exc:  # noqa: BLE001
        return None, f"Cannot open {root}: {exc}", [], None
    blob = {
        "root": str(project.root),
        "datasets": [d.to_dict() for d in project.datasets],
        "settings": dict(project.settings or {}),
    }
    msg = f"Opened {project.root} ({len(project.datasets)} datasets)"
    return blob, msg, _ds_opts(project.datasets), _first_ds(project.datasets)


def create_app(
    default_project: str | None = None,
    *,
    url_base_pathname: str | None = None,
    secret_config: str | None = None,
) -> Dash:
    """Build the always read-only Dash viewer."""
    prefix = normalize_url_base_pathname(
        url_base_pathname
        if url_base_pathname is not None
        else os.environ.get("DASH_URL_BASE_PATHNAME")
    )
    root = data_root_from_env(default_project)
    mode = AppMode(data_root=root)

    dash_kwargs: dict = {
        "external_stylesheets": [dbc.themes.FLATLY],
        "suppress_callback_exceptions": True,
        "title": "Biofilm RNA-Seq Viewer",
    }
    if prefix:
        dash_kwargs["url_base_pathname"] = prefix

    app = Dash(__name__, **dash_kwargs)
    attach_mode(app, mode)

    secret = (secret_config or os.environ.get("DASH_SECRET_CONFIG") or "").strip()
    if not secret:
        raise ValueError(
            "Login required: set --secret-config or DASH_SECRET_CONFIG to a TOML "
            "with [auth] user / pwd."
        )
    from src.GUI.auth import enable_basic_auth, load_basic_auth_users, load_secret_key

    enable_basic_auth(app, load_basic_auth_users(secret), load_secret_key(secret))

    modules = list(MODULE_REGISTRY)
    module_tabs = [
        dbc.Tab(mod.layout(), label=mod.label, tab_id=mod.id) for mod in modules
    ]
    active_tab = modules[0].id if modules else "pca"

    blob, _status, opts, first = _load_project_blob(root)
    app.layout = dbc.Container(
        [
            dcc.Store(id="project-store", data=blob),
            dcc.Store(id="session-store"),
            html.H2("Biofilm microenvironments — interactive viewer", className="mt-3 mb-1"),
            html.Div(id="project-status", style={"display": "none"}),
            dcc.Input(id="project-path", type="hidden", value=root),
            dbc.Card(
                dbc.CardBody(dataset_picker_layout(options=opts, value=first)),
                className="mb-3",
            ),
            dbc.Tabs(module_tabs, id="analysis-tabs", active_tab=active_tab),
        ],
        fluid=True,
        className="pb-5",
    )

    _register_core_callbacks(app)
    for mod in modules:
        mod.register_callbacks(app)

    return app


def _register_core_callbacks(app: Dash) -> None:
    @app.callback(
        Output("session-store", "data"),
        Output("ds-status", "children"),
        Input("ds-active", "value"),
        Input("project-store", "data"),
    )
    def _load_session(active, blob):
        if not blob or not blob.get("root"):
            return (
                {"ready": False, "error": "No project", "meta_columns": []},
                "No readable data folder.",
            )
        name = active if isinstance(active, str) else (active[0] if active else None)
        if not name:
            return (
                {"ready": False, "error": "No dataset selected", "meta_columns": []},
                "Select an active dataset.",
            )
        from src.GUI.project import Project

        project = Project.load(blob["root"])
        project.datasets = [
            DatasetEntry.from_dict(d) for d in blob.get("datasets", project.datasets)
        ]
        session = load_selected(project, [name])
        store = session_to_store(session)
        if session.ready:
            msg = (
                f"Loaded {session.expression.shape[0]} samples × "
                f"{session.expression.shape[1]} features."
            )
        else:
            msg = session.error or "Failed to load."
        return store, msg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Biofilm RNA-Seq interactive viewer")
    parser.add_argument(
        "--project",
        default=None,
        help=f"Data folder (default: {DATA_ROOT})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--url-base-pathname",
        default=None,
        help="Dash URL prefix when reverse-proxied (e.g. /interactive/)",
    )
    parser.add_argument(
        "--secret-config",
        default=None,
        help="TOML with [auth] user / pwd (required)",
    )
    args = parser.parse_args(argv)

    secret = args.secret_config or os.environ.get("DASH_SECRET_CONFIG")
    app = create_app(
        default_project=args.project,
        url_base_pathname=args.url_base_pathname,
        secret_config=secret,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
