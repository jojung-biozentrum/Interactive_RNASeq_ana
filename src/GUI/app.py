"""Dash entry point: desktop (writable) and Virtual-server (readonly) share one tree.

On this ``Virtual-server`` branch the **default is read-only** (safe for the lab
VM). Accidental ``python -m src.GUI.app`` will not expose write UI.

Read-only (default here / gunicorn)::

    python -m src.GUI.app --secret-config path/to/secret.toml
    gunicorn src.GUI.wsgi:server -b :8052

Writable desktop (explicit opt-in only)::

    python -m src.GUI.app --writable
    DASH_WRITABLE=1 python -m src.GUI.app
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

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.GUI.components.dataset_picker import dataset_picker_layout
from src.GUI.data_store import load_selected, session_to_store
from src.GUI.modules import build_registry
from src.GUI.project import DatasetEntry, open_project
from src.GUI.runtime import (
    SERVER_DEFAULT_DATA_ROOT,
    attach_mode,
    mode_from_env,
    normalize_url_base_pathname,
)

# Prefill for --writable desktop runs. Readonly uses SERVER_DEFAULT_DATA_ROOT.
DEFAULT_PROJECT = SERVER_DEFAULT_DATA_ROOT

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


def _load_project_blob(root: str, *, readonly: bool) -> tuple[dict | None, str, list, str | None]:
    try:
        project = open_project(root, readonly=readonly)
    except Exception as exc:  # noqa: BLE001
        return None, f"Cannot open {root}: {exc}", [], None
    blob = {
        "root": str(project.root),
        "datasets": [d.to_dict() for d in project.datasets],
        "settings": dict(project.settings or {}),
    }
    suffix = ", read-only" if readonly else ""
    msg = f"Opened {project.root} ({len(project.datasets)} datasets{suffix})"
    return blob, msg, _ds_opts(project.datasets), _first_ds(project.datasets)


def create_app(
    default_project: str | None = None,
    *,
    readonly: bool | None = None,
    url_base_pathname: str | None = None,
    secret_config: str | None = None,
) -> Dash:
    """Build the Dash app.

    Safety rules (Virtual-server):
    - Any non-empty ``url_base_pathname`` (nginx proxy) **forces read-only**.
      Custom wrappers that only pass the URL prefix cannot accidentally open
      writable UI.
    - Writable requires explicit ``readonly=False`` **and** no URL prefix
      (plus ``--writable`` / ``DASH_WRITABLE=1`` at the CLI/env layer).
    """
    prefix = normalize_url_base_pathname(
        url_base_pathname
        if url_base_pathname is not None
        else os.environ.get("DASH_URL_BASE_PATHNAME")
    )
    # Proxied dashboard deploys are always read-only — no escape hatch.
    if prefix is not None:
        if readonly is False:
            raise ValueError(
                f"Writable mode is not allowed with url_base_pathname={prefix!r}. "
                "Omit the URL prefix for local --writable use."
            )
        readonly = True
        os.environ["DASH_READONLY"] = "1"
        os.environ.pop("DASH_WRITABLE", None)

    mode = mode_from_env(readonly=readonly, default_project=default_project)
    # Readonly always pins a data root (never an empty / wrong desktop path).
    if mode.readonly and not mode.data_root:
        mode = mode_from_env(
            readonly=True,
            default_project=default_project or SERVER_DEFAULT_DATA_ROOT,
        )

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
    if secret:
        from src.GUI.auth import enable_basic_auth, load_basic_auth_users, load_secret_key

        enable_basic_auth(app, load_basic_auth_users(secret), load_secret_key(secret))
    elif mode.readonly:
        raise ValueError(
            "Read-only / Virtual-server mode requires a login: set --secret-config "
            "or DASH_SECRET_CONFIG to a TOML with [auth] user / pwd."
        )

    modules = build_registry(readonly=mode.readonly)
    module_tabs = [
        dbc.Tab(mod.layout(readonly=mode.readonly), label=mod.label, tab_id=mod.id)
        for mod in modules
    ]
    active_tab = modules[0].id if modules else "pca"

    if mode.readonly:
        # Prefer mode.data_root (DASH_DEFAULT_PROJECT / DASH_DATA_ROOT /
        # biofilm-microenvironments2). Never fall back to a desktop WSL path.
        root = str(mode.data_root or SERVER_DEFAULT_DATA_ROOT)
        blob, _status, opts, first = _load_project_blob(root, readonly=True)
        # Hidden anchors so shared callbacks keep their component ids.
        project_block: list = [
            html.Div(id="project-status", style={"display": "none"}),
            dcc.Input(id="project-path", type="hidden", value=root),
        ]
        picker = dataset_picker_layout(readonly=True, options=opts, value=first)
        stores = [
            dcc.Store(id="project-store", data=blob),
            dcc.Store(id="session-store"),
        ]
    else:
        prefill = mode.data_root if mode.data_root is not None else DEFAULT_PROJECT
        project_block = [
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
                                        value=prefill,
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Browse…",
                                            id="project-browse",
                                            color="info",
                                            outline=True,
                                            className="me-2",
                                        ),
                                        dbc.Button(
                                            "Open", id="project-open", color="primary"
                                        ),
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
        ]
        picker = dataset_picker_layout(readonly=False)
        stores = [dcc.Store(id="project-store"), dcc.Store(id="session-store")]

    app.layout = dbc.Container(
        [
            *stores,
            html.H2("Biofilm microenvironments — interactive viewer", className="mt-3 mb-1"),
            *project_block,
            dbc.Card(dbc.CardBody(picker), className="mb-3"),
            dbc.Tabs(module_tabs, id="analysis-tabs", active_tab=active_tab),
        ],
        fluid=True,
        className="pb-5",
    )

    # Never register disk-write callbacks in readonly (defense in depth).
    _register_core_callbacks(app, readonly=mode.readonly)
    if not mode.readonly:
        _register_browse_callbacks(app)
        _register_write_callbacks(app)
    for mod in modules:
        mod.register_callbacks(app, readonly=mode.readonly)

    return app


def _register_browse_callbacks(app: Dash) -> None:
    from src.GUI.components.folder_browser import (
        path_relative_to,
        pick_file_dialog,
        pick_folder_dialog,
    )

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


def _register_core_callbacks(app: Dash, *, readonly: bool) -> None:
    # In writable mode other callbacks also write ds-status (allow_duplicate).
    # In readonly those writers are absent, so a normal Output + initial call is fine.
    ds_status_out = (
        Output("ds-status", "children")
        if readonly
        else Output("ds-status", "children", allow_duplicate=True)
    )

    @app.callback(
        Output("session-store", "data"),
        ds_status_out,
        Input("ds-active", "value"),
        Input("project-store", "data"),
        prevent_initial_call=not readonly,
    )
    def _load_session(active, blob):
        if not blob or not blob.get("root"):
            return (
                {"ready": False, "error": "No project", "meta_columns": []},
                "Open a working folder first." if not readonly else "No readable data folder.",
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

    @app.callback(
        Output("ds-gene-meta-cols", "options"),
        Output("ds-gene-meta-cols", "value"),
        Input("ds-active", "value"),
        Input("project-store", "data"),
        State("ds-gene-meta-cols", "value"),
    )
    def _sync_gene_meta_cols(active, blob, current):
        if not blob or not blob.get("root") or not active:
            return [], None
        name = active if isinstance(active, str) else (active[0] if active else None)
        if not name:
            return [], None
        entry = next(
            (d for d in blob.get("datasets", []) if d.get("name") == name),
            None,
        )
        locus = (entry or {}).get("locus_lookup") or ""
        if not locus:
            return [], None
        from src.biocyc.celov_multiomics_post import load_locus_lookup
        from src.GUI.project import Project

        try:
            project = Project.load(blob["root"])
            path = project.resolve(locus)
            if not Path(path).is_file():
                return [], None
            lookup = load_locus_lookup(path)
            cols = [str(c) for c in lookup.columns]
        except Exception:  # noqa: BLE001
            return [], None
        opts = [{"label": c, "value": c} for c in cols]
        cur = [c for c in (current or []) if c in cols]
        return opts, (cur or None)


def _register_write_callbacks(app: Dash) -> None:
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
            blob, msg, opts, first = _load_project_blob(path, readonly=False)
            meta_f, locus_f, celov_f = _autofill_dataset_fields(
                (blob or {}).get("datasets", [])
            )
            return blob, msg, opts, first, opts, [], meta_f, locus_f, celov_f
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


def no_update_triple(msg: str):
    return no_update, msg, no_update


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Biofilm RNA-Seq interactive viewer")
    parser.add_argument(
        "--project",
        default=None,
        help=(
            f"Data / working folder (readonly default: {SERVER_DEFAULT_DATA_ROOT}; "
            f"writable default: {DEFAULT_PROJECT})"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--readonly",
        action="store_true",
        help="Force read-only (default on Virtual-server; also DASH_READONLY=1)",
    )
    mode.add_argument(
        "--writable",
        action="store_true",
        help="Allow register / Filter / Celov writes (desktop only; also DASH_WRITABLE=1)",
    )
    parser.add_argument(
        "--url-base-pathname",
        default=None,
        help="Dash URL prefix when reverse-proxied (e.g. /interactive/)",
    )
    parser.add_argument(
        "--secret-config",
        default=None,
        help="TOML with [auth] user / pwd (required unless --writable)",
    )
    args = parser.parse_args(argv)

    if args.writable:
        readonly: bool | None = False
    elif args.readonly:
        readonly = True
    else:
        readonly = None  # → DEPLOYMENT_READONLY_DEFAULT (True on this branch)

    secret = args.secret_config or os.environ.get("DASH_SECRET_CONFIG")
    app = create_app(
        default_project=args.project,
        readonly=readonly,
        url_base_pathname=args.url_base_pathname,
        secret_config=secret,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
