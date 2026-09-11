"""Dash entry point: project bar, dataset picker, modular analysis tabs.

Launch from repo root:
    python -m src.GUI.app
    python -m src.GUI.app --project path/to/other/folder
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/GUI/app.py` as well as `python -m src.GUI.app`
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.GUI.components.dataset_picker import dataset_picker_layout
from src.GUI.components.folder_browser import (
    path_relative_to,
    pick_file_dialog,
    pick_folder_dialog,
)
from src.GUI.data_store import load_selected, session_to_store
from src.GUI.modules import MODULE_REGISTRY
from src.GUI.project import DatasetEntry, open_project

# Pre-filled working folder in the UI and ``--project`` default.
DEFAULT_PROJECT = "/mnt/bronto/Johannes"


def _ds_opts(datasets: list) -> list[dict]:
    names = []
    for d in datasets:
        names.append(d["name"] if isinstance(d, dict) else d.name)
    return [{"label": n, "value": n} for n in names]


def _first_ds(datasets: list) -> str | None:
    opts = _ds_opts(datasets)
    return opts[0]["value"] if opts else None


def _working_folder(blob: dict | None, project_path: str | None = None) -> str | None:
    root = (blob or {}).get("root") or project_path
    return str(root).strip() if root and str(root).strip() else None


def create_app(default_project: str | None = None) -> Dash:
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
        title="Biofilm RNA-Seq Viewer",
    )

    module_tabs = [
        dbc.Tab(mod.layout(), label=mod.label, tab_id=mod.id) for mod in MODULE_REGISTRY
    ]

    app.layout = dbc.Container(
        [
            dcc.Store(id="project-store"),
            dcc.Store(id="session-store"),
            html.H2("Biofilm microenvironments — interactive viewer", className="mt-3 mb-1"),
            html.P(
                "Open a working folder, register datasets, then explore modular analyses.",
                className="text-muted",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Working folder"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Input(
                                        id="project-path",
                                        type="text",
                                        value=default_project if default_project is not None else DEFAULT_PROJECT,
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button("Browse…", id="project-browse", color="info", outline=True, className="me-2"),
                                        dbc.Button("Open", id="project-open", color="primary"),
                                    ],
                                    md=6,
                                ),
                            ],
                            className="g-2",
                        ),
                        html.Div(id="project-status", className="mt-2 text-muted small"),
                    ]
                ),
                className="mb-3",
            ),
            dbc.Card(dbc.CardBody(dataset_picker_layout()), className="mb-3"),
            dbc.Tabs(module_tabs, id="analysis-tabs", active_tab=MODULE_REGISTRY[0].id),
        ],
        fluid=True,
        className="pb-5",
    )

    _register_core_callbacks(app)
    _register_browse_callbacks(app)
    for mod in MODULE_REGISTRY:
        mod.register_callbacks(app)

    return app


def _register_browse_callbacks(app: Dash) -> None:
    @app.callback(
        Output("project-path", "value", allow_duplicate=True),
        Output("project-status", "children", allow_duplicate=True),
        Input("project-browse", "n_clicks"),
        State("project-path", "value"),
        prevent_initial_call=True,
    )
    def _browse_project(n_clicks, current):
        chosen = pick_folder_dialog(initial=current, title="Select working folder")
        if not chosen:
            return no_update, "Browse cancelled (or folder dialog unavailable)."
        return chosen, f"Selected folder: {chosen} — click Open to load the project."

    @app.callback(
        Output("ds-expr", "value"),
        Output("ds-name", "value"),
        Output("ds-status", "children", allow_duplicate=True),
        Input("ds-expr-browse", "n_clicks"),
        State("project-store", "data"),
        State("project-path", "value"),
        State("ds-name", "value"),
        prevent_initial_call=True,
    )
    def _browse_expr(n_clicks, blob, project_path, current_name):
        root = _working_folder(blob, project_path)
        chosen = pick_file_dialog(initial=root, title="Select expression / count matrix")
        if not chosen:
            return no_update, no_update, "Expression browse cancelled."
        rel = path_relative_to(root, chosen)
        name = current_name if current_name else Path(chosen).stem
        return rel, name, f"Expression: {rel}"

    @app.callback(
        Output("ds-meta", "value"),
        Output("ds-status", "children", allow_duplicate=True),
        Input("ds-meta-browse", "n_clicks"),
        State("project-store", "data"),
        State("project-path", "value"),
        State("ds-meta", "value"),
        prevent_initial_call=True,
    )
    def _browse_meta(n_clicks, blob, project_path, current_meta):
        root = _working_folder(blob, project_path)
        datasets = (blob or {}).get("datasets") or []
        initial = root
        # With registered datasets, start near an existing metadata path if present;
        # otherwise always the project folder.
        if datasets and current_meta and str(current_meta).strip() and root:
            cand = Path(str(current_meta).strip())
            if not cand.is_absolute():
                cand = Path(root) / cand
            if cand.exists():
                initial = str(cand.parent if cand.is_file() else cand)
        chosen = pick_file_dialog(initial=initial, title="Select metadata table")
        if not chosen:
            return no_update, "Metadata browse cancelled."
        rel = path_relative_to(root, chosen)
        return rel, f"Metadata: {rel}"

    @app.callback(
        Output("ds-locus", "value"),
        Output("ds-status", "children", allow_duplicate=True),
        Input("ds-locus-browse", "n_clicks"),
        State("project-store", "data"),
        State("project-path", "value"),
        State("ds-locus", "value"),
        prevent_initial_call=True,
    )
    def _browse_locus(n_clicks, blob, project_path, current_locus):
        root = _working_folder(blob, project_path)
        datasets = (blob or {}).get("datasets") or []
        initial = root
        if datasets and current_locus and str(current_locus).strip() and root:
            cand = Path(str(current_locus).strip())
            if not cand.is_absolute():
                cand = Path(root) / cand
            if cand.exists():
                initial = str(cand.parent if cand.is_file() else cand)
        chosen = pick_file_dialog(initial=initial, title="Select locus lookup CSV")
        if not chosen:
            return no_update, "Locus browse cancelled."
        rel = path_relative_to(root, chosen)
        return rel, f"Locus lookup: {rel}"


def _autofill_dataset_fields(datasets: list) -> tuple[str | None, str | None, str | None]:
    """Return metadata / locus / celov-id-col from the first registered dataset that has them."""
    meta = None
    locus = None
    celov_id = None
    for d in datasets:
        dd = d if isinstance(d, dict) else d.to_dict()
        if meta is None and dd.get("metadata"):
            meta = str(dd["metadata"])
        if locus is None and dd.get("locus_lookup"):
            locus = str(dd["locus_lookup"])
        if celov_id is None and dd.get("celov_id_col"):
            celov_id = str(dd["celov_id_col"])
        if meta is not None and locus is not None and celov_id is not None:
            break
    return meta, locus, celov_id


def _register_core_callbacks(app: Dash) -> None:
    @app.callback(
        Output("project-store", "data"),
        Output("project-status", "children"),
        Output("ds-active", "options"),
        Output("ds-active", "value"),
        Output("ds-unregister", "options"),
        Output("ds-unregister", "value"),
        Output("ds-meta", "value", allow_duplicate=True),
        Output("ds-locus", "value", allow_duplicate=True),
        Output("ds-celov-id-col", "value", allow_duplicate=True),
        Input("project-open", "n_clicks"),
        State("project-path", "value"),
        prevent_initial_call=True,
    )
    def _open_project(n_open, path):
        if not path or not str(path).strip():
            return None, "Enter a working folder path.", [], None, [], [], None, None, None
        path = str(path).strip()
        try:
            project = open_project(path)
            msg = f"Opened {project.root} ({len(project.datasets)} datasets)"
            blob = {
                "root": str(project.root),
                "datasets": [d.to_dict() for d in project.datasets],
                "settings": dict(project.settings or {}),
            }
            opts = _ds_opts(project.datasets)
            meta_f, locus_f, celov_f = _autofill_dataset_fields(project.datasets)
            return (
                blob,
                msg,
                opts,
                _first_ds(project.datasets),
                opts,
                [],
                meta_f,
                locus_f,
                celov_f,
            )
        except Exception as exc:  # noqa: BLE001
            return None, f"Error: {exc}", [], None, [], [], None, None, None

    @app.callback(
        Output("project-store", "data", allow_duplicate=True),
        Output("ds-active", "options", allow_duplicate=True),
        Output("ds-unregister", "options", allow_duplicate=True),
        Output("ds-status", "children"),
        Output("ds-meta", "value", allow_duplicate=True),
        Output("ds-locus", "value", allow_duplicate=True),
        Output("ds-celov-id-col", "value", allow_duplicate=True),
        Input("ds-register-btn", "n_clicks"),
        State("project-store", "data"),
        State("ds-name", "value"),
        State("ds-expr", "value"),
        State("ds-meta", "value"),
        State("ds-locus", "value"),
        State("ds-sample-col", "value"),
        State("ds-celov-id-col", "value"),
        prevent_initial_call=True,
    )
    def _register(n_clicks, blob, name, expr, meta, locus, sample_col, celov_id_col):
        if not blob or not blob.get("root"):
            return blob, [], [], "Open a working folder first.", no_update, no_update, no_update
        datasets = list(blob.get("datasets", []))
        if not name or not expr or not meta or not locus:
            opts = _ds_opts(datasets)
            return (
                blob,
                opts,
                opts,
                "Name, expression, metadata, and locus lookup paths are required.",
                no_update,
                no_update,
                no_update,
            )
        entry = DatasetEntry(
            name=name.strip(),
            expression=expr.strip(),
            metadata=meta.strip(),
            locus_lookup=locus.strip(),
            sample_id_col=(sample_col or "fileName").strip(),
            celov_id_col=(celov_id_col or "biocyc_id").strip() or "biocyc_id",
        ).to_dict()
        datasets = [d for d in datasets if d["name"] != entry["name"]] + [entry]
        blob = {**blob, "datasets": datasets}
        from src.GUI.project import Project

        project = Project.load(blob["root"])
        project.datasets = [DatasetEntry.from_dict(d) for d in datasets]
        project.settings = dict(blob.get("settings") or {})
        project.save()
        opts = _ds_opts(datasets)
        meta_f, locus_f, celov_f = _autofill_dataset_fields(datasets)
        return blob, opts, opts, f"Registered '{entry['name']}'.", meta_f, locus_f, celov_f

    @app.callback(
        Output("project-store", "data", allow_duplicate=True),
        Output("ds-active", "options", allow_duplicate=True),
        Output("ds-active", "value", allow_duplicate=True),
        Output("ds-unregister", "options", allow_duplicate=True),
        Output("ds-unregister", "value", allow_duplicate=True),
        Output("ds-status", "children", allow_duplicate=True),
        Output("ds-meta", "value", allow_duplicate=True),
        Output("ds-locus", "value", allow_duplicate=True),
        Output("ds-celov-id-col", "value", allow_duplicate=True),
        Input("ds-unregister-btn", "n_clicks"),
        State("project-store", "data"),
        State("ds-unregister", "value"),
        State("ds-active", "value"),
        prevent_initial_call=True,
    )
    def _unregister(n_clicks, blob, to_remove, active):
        if not blob or not blob.get("root"):
            return (
                blob,
                [],
                None,
                [],
                [],
                "Open a working folder first.",
                no_update,
                no_update,
                no_update,
            )
        names = [str(x) for x in (to_remove or [])]
        if not names:
            opts = _ds_opts(blob.get("datasets", []))
            return (
                blob,
                opts,
                active,
                opts,
                [],
                "Select one or more datasets to unregister.",
                no_update,
                no_update,
                no_update,
            )
        remove = set(names)
        datasets = [d for d in blob.get("datasets", []) if d["name"] not in remove]
        blob = {**blob, "datasets": datasets}
        from src.GUI.project import Project

        project = Project.load(blob["root"])
        project.datasets = [DatasetEntry.from_dict(d) for d in datasets]
        project.settings = dict(blob.get("settings") or {})
        project.save()
        opts = _ds_opts(datasets)
        next_active = active if active in {o["value"] for o in opts} else _first_ds(datasets)
        meta_f, locus_f, celov_f = _autofill_dataset_fields(datasets)
        return (
            blob,
            opts,
            next_active,
            opts,
            [],
            f"Unregistered: {', '.join(sorted(remove))}",
            meta_f,
            locus_f,
            celov_f,
        )

    @app.callback(
        Output("session-store", "data"),
        Output("ds-status", "children", allow_duplicate=True),
        Input("ds-active", "value"),
        State("project-store", "data"),
        prevent_initial_call=True,
    )
    def _load_session(active, blob):
        if not blob or not blob.get("root"):
            return {"ready": False, "error": "No project", "meta_columns": []}, "Open a working folder first."
        name = active if isinstance(active, str) else (active[0] if active else None)
        if not name:
            return {"ready": False, "error": "No dataset selected", "meta_columns": []}, "Select an active dataset."
        from src.GUI.project import Project

        project = Project.load(blob["root"])
        project.datasets = [DatasetEntry.from_dict(d) for d in blob.get("datasets", project.datasets)]
        session = load_selected(project, [name])
        store = session_to_store(session)
        if session.ready:
            msg = f"Loaded {session.expression.shape[0]} samples × {session.expression.shape[1]} features."
        else:
            msg = session.error or "Failed to load."
        return store, msg


def no_update_triple(msg: str):
    from dash import no_update

    return no_update, msg, no_update


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Biofilm RNA-Seq interactive viewer")
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"Working folder to pre-fill (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    app = create_app(default_project=args.project)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
