"""Dash entry point: fixed data folder, dataset picker, modular analysis tabs.

Launch from repo root:
    python -m src.GUI.app
    python -m src.GUI.app --project path/to/biofilm-microenvironments

Gunicorn (virtual server):
    gunicorn src.GUI.wsgi:server -b :8052
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/GUI/app.py` as well as `python -m src.GUI.app`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc

from src.GUI.auth import enable_basic_auth, load_basic_auth_users
from src.GUI.components.dataset_picker import dataset_picker_layout
from src.GUI.data_store import load_selected, session_to_store
from src.GUI.modules import MODULE_REGISTRY
from src.GUI.project import DatasetEntry, open_project

# The only folder this viewer reads. Datasets must be registered in its
# project.yaml; there is no way to point the app somewhere else from the UI.
DATA_ROOT = "/home/lab/data/Johannes/biofilm-microenvironments1"


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


def _ds_opts(datasets: list) -> list[dict]:
    names = []
    for d in datasets:
        names.append(d["name"] if isinstance(d, dict) else d.name)
    return [{"label": n, "value": n} for n in names]


def _first_ds(datasets: list) -> str | None:
    opts = _ds_opts(datasets)
    return opts[0]["value"] if opts else None


def _load_project(root: str) -> tuple[dict | None, str, list, str | None]:
    """Read the fixed folder's ``project.yaml``. Read-only; never writes."""
    try:
        project = open_project(root)
    except Exception as exc:  # noqa: BLE001
        return None, f"Cannot read {root}: {exc}", [], None
    blob = {
        "root": str(project.root),
        "datasets": [d.to_dict() for d in project.datasets],
        "settings": dict(project.settings or {}),
    }
    msg = f"{project.root} — {len(project.datasets)} dataset(s), read-only"
    return blob, msg, _ds_opts(project.datasets), _first_ds(project.datasets)


def create_app(
    default_project: str | None = None,
    url_base_pathname: str | None = None,
    secret_config: str | None = None,
) -> Dash:
    dash_kwargs: dict = {
        "external_stylesheets": [dbc.themes.FLATLY],
        "suppress_callback_exceptions": True,
        "title": "Biofilm RNA-Seq Viewer",
    }
    prefix = normalize_url_base_pathname(url_base_pathname)
    if prefix:
        dash_kwargs["url_base_pathname"] = prefix

    app = Dash(__name__, **dash_kwargs)

    secret = (secret_config or "").strip()
    if secret:
        enable_basic_auth(app, load_basic_auth_users(secret))

    module_tabs = [
        dbc.Tab(mod.layout(), label=mod.label, tab_id=mod.id) for mod in MODULE_REGISTRY
    ]

    root = str(default_project or DATA_ROOT)
    blob, status, opts, first = _load_project(root)

    app.layout = dbc.Container(
        [
            dcc.Store(id="project-store", data=blob),
            dcc.Store(id="session-store"),
            html.H2("Biofilm microenvironments — interactive viewer", className="mt-3 mb-1"),
            html.P(
                "Read-only viewer: datasets come from the folder below and nothing "
                "is written to disk.",
                className="text-muted",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6("Data folder", className="mb-1"),
                        html.Div(status, className="font-monospace small text-muted"),
                    ]
                ),
                className="mb-3",
            ),
            dbc.Card(
                dbc.CardBody(dataset_picker_layout(options=opts, value=first)),
                className="mb-3",
            ),
            dbc.Tabs(module_tabs, id="analysis-tabs", active_tab=MODULE_REGISTRY[0].id),
        ],
        fluid=True,
        className="pb-5",
    )

    _register_core_callbacks(app)
    for mod in MODULE_REGISTRY:
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
                "No readable data folder — check the server configuration.",
            )
        name = active if isinstance(active, str) else (active[0] if active else None)
        if not name:
            return (
                {"ready": False, "error": "No dataset selected", "meta_columns": []},
                "Select a dataset.",
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
        help=f"Override the data folder for local runs (default: {DATA_ROOT})",
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
        help="TOML with [auth] user / pwd (omit for no login, e.g. local use)",
    )
    args = parser.parse_args(argv)

    app = create_app(
        default_project=args.project,
        url_base_pathname=args.url_base_pathname,
        secret_config=args.secret_config,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
