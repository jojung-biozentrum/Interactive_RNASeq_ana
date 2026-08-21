"""Dash entry point: project bar, dataset picker, modular analysis tabs.

Launch from repo root:
    python -m src.GUI.app
    python -m src.GUI.app --project path/to/biofilm-microenvironments1

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

from dash import Dash, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc

from src.GUI.auth import enable_basic_auth, load_basic_auth_users
from src.GUI.components.dataset_picker import dataset_picker_layout
from src.GUI.data_store import load_selected, session_to_store
from src.GUI.modules import MODULE_REGISTRY
from src.GUI.project import DatasetEntry, open_project


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


def _project_blob(project) -> dict:
    return {
        "root": str(project.root),
        "datasets": [d.to_dict() for d in project.datasets],
        "settings": dict(project.settings or {}),
    }


def _try_open_project(path: str | None) -> tuple[dict | None, str, list, str | None]:
    """Load an existing ``project.yaml`` (read-only). Never writes."""
    if not path or not str(path).strip():
        return None, "", [], None
    path = str(path).strip()
    try:
        project = open_project(path)
        blob = _project_blob(project)
        opts = _ds_opts(project.datasets)
        msg = f"Opened {project.root} ({len(project.datasets)} datasets, read-only)"
        return blob, msg, opts, _first_ds(project.datasets)
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: {exc}", [], None


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

    blob, status, opts, first = _try_open_project(default_project)

    app.layout = dbc.Container(
        [
            dcc.Store(id="project-store", data=blob),
            dcc.Store(id="session-store"),
            html.H2("Biofilm microenvironments — interactive viewer", className="mt-3 mb-1"),
            html.P(
                "Open a working folder that already has project.yaml, then explore "
                "modular analyses. This build is read-only (no file exports).",
                className="text-muted",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Working folder"),
                        html.P(
                            "Must contain a project.yaml prepared ahead of time. "
                            "Nothing is written back to disk.",
                            className="text-muted small",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Input(
                                        id="project-path",
                                        type="text",
                                        value=default_project or "",
                                    ),
                                    md=8,
                                ),
                                dbc.Col(
                                    dbc.Button("Open", id="project-open", color="primary"),
                                    md=4,
                                ),
                            ],
                            className="g-2",
                        ),
                        html.Div(
                            id="project-status",
                            className="mt-2 text-muted small",
                            children=status,
                        ),
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
        Output("project-store", "data"),
        Output("project-status", "children"),
        Output("ds-active", "options"),
        Output("ds-active", "value"),
        Input("project-open", "n_clicks"),
        State("project-path", "value"),
        prevent_initial_call=True,
    )
    def _open_project(n_open, path):
        blob, msg, opts, first = _try_open_project(path)
        if blob is None and not (path and str(path).strip()):
            return None, "Enter a working folder path.", [], None
        return blob, msg, opts, first

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
                "Open a working folder first.",
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
        help="Working folder that already contains project.yaml",
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
