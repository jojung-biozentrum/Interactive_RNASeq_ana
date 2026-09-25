"""Volcano by metadata condition — volcano_by_condition.ipynb."""

from __future__ import annotations

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from src.biocyc.celov_multiomics_post import load_locus_lookup
from src.GUI.project import resolve_celov_id_col

from ..components.controls import (
    EXPORT_H,
    EXPORT_W,
    fig_size_controls,
    plotly_title as _plotly_title,
    set_fig_size,
)
from ..components.folder_browser import pick_save_file_dialog
from ..components.gene_meta_mark import (
    entry_dropdown_options,
    genes_with_meta_entry,
    locus_mark_columns,
    mark_controls,
    mark_legend_banner,
)
from ..components.sample_detail import gene_detail_placeholder, gene_detail_table
from ..data_store import (
    active_dataset_entry as _active_dataset_entry,
    locus_path_from_session as _locus_path_from_session,
    session_from_store,
)
from .clustering import _volcano_weighed_for_celov, volcano_cluster_vs_cluster
from .condition_enrichment import _apply_op, classify_meta_columns
from .hc_plots import volcano_fig_from_results
from .pca_classifier import save_classifier_celov

_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "volcano_condition"},
    "displaylogo": False,
}
_VC_RUNTIME: dict = {}


def _entry_options(series: pd.Series) -> list[dict]:
    vals = series.fillna("NA").astype(str)
    entries = sorted(vals.unique(), key=lambda x: (x == "NA", x.lower()))
    return [{"label": e, "value": e} for e in entries]


def _vc_bins(
    meta: pd.DataFrame,
    col: str,
    spec: dict,
    a_vals,
    b_vals,
    mode,
    a_op,
    a_thr,
    b_op,
    b_thr,
) -> tuple[pd.Index, pd.Index, str]:
    col_s = meta[col].fillna("NA").astype(str)
    a_vals = [str(v) for v in (a_vals or [])]
    b_vals = [str(v) for v in (b_vals or [])]
    if a_vals:
        mask_a = col_s.isin(a_vals)
        a_lab = ",".join(a_vals)
    elif spec["kind"] == "num" and a_thr is not None and a_thr != "":
        mask_a = _apply_op(meta[col], a_op or ">", a_thr)
        a_lab = f"{a_op or '>'}{a_thr}"
    else:
        raise ValueError("Select Bin A entries, or a numeric operator and value.")

    if str(mode) in {"rest", "~", "~Bin A", "~bin a"}:
        mask_b = ~mask_a
        if spec["kind"] == "num":
            mask_b &= pd.to_numeric(meta[col], errors="coerce").notna()
        b_lab = "~Bin A"
    elif b_vals:
        mask_b = col_s.isin(b_vals)
        b_lab = ",".join(b_vals)
    elif spec["kind"] == "num" and b_thr is not None and b_thr != "":
        mask_b = _apply_op(meta[col], b_op or "<", b_thr)
        b_lab = f"{b_op or '<'}{b_thr}"
    else:
        raise ValueError("Select Bin B entries, a numeric operator, or use ~Bin A.")

    ids_a = meta.index[mask_a]
    ids_b = meta.index[mask_b]
    overlap = ids_a.intersection(ids_b)
    if len(overlap):
        ids_a = ids_a.difference(overlap)
        ids_b = ids_b.difference(overlap)
    return ids_a, ids_b, f"{col}:{{{a_lab}}} vs {{{b_lab}}}"


class VolcanoConditionModule:
    id = "volcano-condition"
    label = "Volcano by condition"

    def layout(self):
        return html.Div(
            [
                html.P(
                    "Pick a metadata column. Define Bin A by one or more entries "
                    "(also on numeric columns) or by a numeric operator. Bin B is "
                    "~Bin A, selected entries, or a numeric operator. Overlapping "
                    "samples are dropped.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Condition column"),
                                dcc.Dropdown(id="vc-col"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Bin A"),
                                dcc.Dropdown(id="vc-a", multi=True),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Bin B"),
                                dcc.RadioItems(
                                    id="vc-mode",
                                    options=[
                                        {"label": " ~Bin A", "value": "rest"},
                                        {"label": " select entries / operator", "value": "b"},
                                    ],
                                    value="rest",
                                    inline=True,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            html.Div(
                                id="vc-b-entry-wrap",
                                style={"display": "none"},
                                children=[
                                    html.Label("Bin B entries"),
                                    dcc.Dropdown(id="vc-b", multi=True),
                                ],
                            ),
                            md=3,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(
                    id="vc-num-a-wrap",
                    style={"display": "none"},
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Bin A operator"),
                                        dcc.Dropdown(
                                            id="vc-a-op",
                                            options=[
                                                {"label": ">", "value": ">"},
                                                {"label": "<", "value": "<"},
                                            ],
                                            value=">",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Bin A value"),
                                        dbc.Input(id="vc-a-thr", type="number"),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                    ],
                ),
                html.Div(
                    id="vc-num-b-wrap",
                    style={"display": "none"},
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Bin B operator"),
                                        dcc.Dropdown(
                                            id="vc-b-op",
                                            options=[
                                                {"label": ">", "value": ">"},
                                                {"label": "<", "value": "<"},
                                            ],
                                            value="<",
                                            clearable=False,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Bin B value"),
                                        dbc.Input(id="vc-b-thr", type="number"),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                    ],
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Difference"),
                                dcc.Dropdown(
                                    id="vc-center",
                                    options=[
                                        {"label": "means", "value": "mean"},
                                        {"label": "medians", "value": "median"},
                                    ],
                                    value="mean",
                                    clearable=False,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("−log10(padj) thr"),
                                dbc.Input(id="vc-padj", type="number", value=2.0, step=0.1),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("|ΔCLR| thr"),
                                dbc.Input(id="vc-fc", type="number", value=0.5, step=0.1),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run volcano", id="vc-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="vc-status", className="text-muted small mb-2"),
                mark_controls(col_id="vc-mark-col", entry_id="vc-mark-entry", wrap_id="vc-mark-wrap"),
                html.Div(id="vc-mark-legend", children=mark_legend_banner(0, None)),
                fig_size_controls("vc", default_width=EXPORT_W, default_height=EXPORT_H),
                dcc.Loading(
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id="vc-fig", figure={}, config=_GRAPH_CONFIG),
                                md=8,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6("Gene metadata", className="mb-2"),
                                        html.Div(id="vc-gene-detail", children=gene_detail_placeholder()),
                                    ],
                                    className="border rounded p-2 bg-light",
                                ),
                                md=4,
                            ),
                        ],
                        className="g-2",
                    ),
                    type="default",
                ),
                dcc.Store(id="vc-last-gene"),
                dcc.Store(id="vc-cache"),
                html.H6("Save Celov", className="mt-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="vc-celov-score",
                                options=[
                                    {"label": "expression difference", "value": "fold_change"},
                                    {"label": "−log10(padj)", "value": "neg_log10_padj"},
                                    {"label": "product of both", "value": "product"},
                                ],
                                value="fold_change",
                                clearable=False,
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            dcc.RadioItems(
                                id="vc-celov-mode",
                                options=[
                                    {"label": "up & down", "value": "up_and_down"},
                                    {"label": "up", "value": "up"},
                                    {"label": "down", "value": "down"},
                                    {"label": "all together", "value": "all"},
                                ],
                                value="up_and_down",
                                inline=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="vc-celov-out", type="text"),
                                    dbc.Button("Browse…", id="vc-celov-browse", color="info", outline=True),
                                ]
                            ),
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button("Save Celov", id="vc-celov-save", color="secondary", className="mb-2"),
                html.Div(id="vc-celov-status", className="text-muted small"),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output("vc-col", "options"),
            Output("vc-col", "value"),
            Input("session-store", "data"),
            State("vc-col", "value"),
        )
        def _fill_cols(session_blob, current):
            session = session_from_store(session_blob)
            if session.metadata is None:
                return [], None
            info = classify_meta_columns(session.metadata)
            _VC_RUNTIME["col_info"] = info
            opts = [{"label": f"{c} ({info[c]['kind']})", "value": c} for c in info]
            value = current if current in info else (next(iter(info), None))
            return opts, value

        @app.callback(
            Output("vc-a", "options"),
            Output("vc-b", "options"),
            Output("vc-a", "value"),
            Output("vc-b", "value"),
            Output("vc-num-a-wrap", "style"),
            Output("vc-a-thr", "value"),
            Output("vc-b-thr", "value"),
            Input("vc-col", "value"),
            State("session-store", "data"),
        )
        def _fill_entries(col, session_blob):
            hide = {"display": "none"}
            session = session_from_store(session_blob)
            if session.metadata is None or not col:
                return [], [], None, None, hide, None, None
            info = _VC_RUNTIME.get("col_info") or classify_meta_columns(session.metadata)
            spec = info.get(col)
            if not spec:
                return [], [], None, None, hide, None, None
            opts = _entry_options(session.metadata[col])
            if spec["kind"] == "num":
                med = spec.get("median")
                return opts, opts, None, None, {"display": "block"}, med, med
            return opts, opts, None, None, hide, None, None

        @app.callback(
            Output("vc-b-entry-wrap", "style"),
            Output("vc-num-b-wrap", "style"),
            Input("vc-mode", "value"),
            Input("vc-col", "value"),
            State("session-store", "data"),
        )
        def _toggle_bin_b(mode, col, session_blob):
            hide = {"display": "none"}
            show = {"display": "block"}
            select_b = str(mode) not in {"rest", "~"}
            if not select_b:
                return hide, hide
            info = _VC_RUNTIME.get("col_info")
            if info is None:
                session = session_from_store(session_blob)
                if session.metadata is not None:
                    info = classify_meta_columns(session.metadata)
            is_num = bool(info and col and info.get(col, {}).get("kind") == "num")
            return show, show if is_num else hide

        @app.callback(
            Output("vc-fig", "figure"),
            Output("vc-status", "children"),
            Output("vc-cache", "data"),
            Output("vc-mark-wrap", "style"),
            Output("vc-mark-col", "options"),
            Output("vc-mark-col", "value"),
            Output("vc-last-gene", "data", allow_duplicate=True),
            Input("vc-run", "n_clicks"),
            State("vc-fig-w", "value"),
            State("vc-fig-h", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            State("vc-col", "value"),
            State("vc-a", "value"),
            State("vc-b", "value"),
            State("vc-mode", "value"),
            State("vc-a-op", "value"),
            State("vc-a-thr", "value"),
            State("vc-b-op", "value"),
            State("vc-b-thr", "value"),
            State("vc-center", "value"),
            State("vc-padj", "value"),
            State("vc-fc", "value"),
            prevent_initial_call=True,
        )
        def _run(
            n_clicks,
            fig_w,
            fig_h,
            session_blob,
            project_blob,
            col,
            a_vals,
            b_vals,
            mode,
            a_op,
            a_thr,
            b_op,
            b_thr,
            center,
            padj_thr,
            fc_thr,
        ):
            empty = go.Figure()
            hide = {"display": "none"}
            session = session_from_store(session_blob)
            if not session.ready or session.metadata is None:
                return empty, session.error or "Load a dataset first.", None, hide, [], None, None
            if not col:
                return empty, "Choose a condition column.", None, hide, [], None, None
            info = _VC_RUNTIME.get("col_info") or classify_meta_columns(session.metadata)
            spec = info.get(col)
            if not spec:
                return empty, f"Column {col!r} was skipped as ID-like or constant.", None, hide, [], None, None
            try:
                ids_a, ids_b, title = _vc_bins(
                    session.metadata,
                    col,
                    spec,
                    a_vals,
                    b_vals,
                    mode or "rest",
                    a_op,
                    a_thr,
                    b_op,
                    b_thr,
                )
            except Exception as exc:  # noqa: BLE001
                return empty, f"Bin error: {exc}", None, hide, [], None, None
            if len(ids_a) < 2 or len(ids_b) < 2:
                return (
                    empty,
                    f"Need ≥2 samples per side (A={len(ids_a)}, B={len(ids_b)}).",
                    None,
                    hide,
                    [],
                    None,
                    None,
                )
            expr = session.expression.T
            expr.columns = expr.columns.astype(str)
            keep_a = [s for s in ids_a.astype(str) if s in expr.columns]
            keep_b = [s for s in ids_b.astype(str) if s in expr.columns]
            try:
                results, _fig = volcano_cluster_vs_cluster(
                    expr[keep_a],
                    expr[keep_b],
                    title=_plotly_title(title),
                    neg_log10_padj_threshold=float(padj_thr or 2.0),
                    fold_change_threshold=float(fc_thr or 0.5),
                    center=center or "mean",
                )
            except Exception as exc:  # noqa: BLE001
                return empty, f"Volcano error: {exc}", None, hide, [], None, None
            path = _locus_path_from_session(session_blob, project_blob)
            lookup = None
            if path:
                try:
                    lookup = load_locus_lookup(path)
                except Exception:  # noqa: BLE001
                    lookup = None
            _VC_RUNTIME["results"] = results
            _VC_RUNTIME["locus_lookup"] = lookup
            _VC_RUNTIME["plot_meta"] = {
                "padj": float(padj_thr or 2.0),
                "fc": float(fc_thr or 0.5),
                "title": title,
            }
            fig = volcano_fig_from_results(
                results,
                title=_plotly_title(title, f"Bin A n={len(keep_a)} vs Bin B n={len(keep_b)}"),
                neg_log10_padj_threshold=float(padj_thr or 2.0),
                fold_change_threshold=float(fc_thr or 0.5),
            )
            fig = set_fig_size(fig, fig_w, fig_h)
            mark_cols = locus_mark_columns(lookup)
            mark_opts = [{"label": c, "value": c} for c in mark_cols]
            wrap = {"display": "block"} if mark_cols else hide
            return (
                fig,
                f"Volcano: {title}. Bin A n={len(keep_a)}, Bin B n={len(keep_b)}.",
                {"n_a": len(keep_a), "n_b": len(keep_b), "title": title},
                wrap,
                mark_opts,
                None,
                None,
            )

        @app.callback(
            Output("vc-fig", "figure", allow_duplicate=True),
            Output("vc-mark-legend", "children"),
            Input("vc-mark-col", "value"),
            Input("vc-mark-entry", "value"),
            Input("vc-fig-w", "value"),
            Input("vc-fig-h", "value"),
            prevent_initial_call=True,
        )
        def _replot_marks(mark_col, mark_entry, fig_w, fig_h):
            results = _VC_RUNTIME.get("results")
            meta = _VC_RUNTIME.get("plot_meta")
            if results is None or not meta:
                return no_update, no_update
            mark_genes = genes_with_meta_entry(
                _VC_RUNTIME.get("locus_lookup"), mark_col, mark_entry
            )
            mark_label = f"{mark_col}={mark_entry}" if mark_col and mark_entry else None
            fig = volcano_fig_from_results(
                results,
                title=_plotly_title(meta.get("title", "Volcano")),
                neg_log10_padj_threshold=meta["padj"],
                fold_change_threshold=meta["fc"],
                mark_genes=mark_genes,
                mark_label=mark_label,
            )
            n_mark = len(mark_genes) if mark_genes else 0
            return set_fig_size(fig, fig_w, fig_h), mark_legend_banner(n_mark, mark_entry)

        @app.callback(
            Output("vc-mark-entry", "options"),
            Output("vc-mark-entry", "value"),
            Output("vc-mark-entry", "disabled"),
            Output("vc-mark-entry", "placeholder"),
            Input("vc-mark-col", "value"),
        )
        def _mark_entries(col):
            lookup = _VC_RUNTIME.get("locus_lookup")
            opts = entry_dropdown_options(lookup, col) if col else []
            return opts, None, not bool(opts), "entry" if opts else "choose a column"

        @app.callback(
            Output("vc-last-gene", "data"),
            Input("vc-fig", "clickData"),
        )
        def _click(click):
            if not click:
                return no_update
            raw = click["points"][0].get("customdata")
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            return str(raw) if raw is not None else no_update

        @app.callback(
            Output("vc-gene-detail", "children"),
            Input("vc-last-gene", "data"),
        )
        def _detail(gid):
            if not gid:
                return gene_detail_placeholder()
            return gene_detail_table(str(gid), _VC_RUNTIME.get("locus_lookup"))

        @app.callback(
            Output("vc-celov-out", "value"),
            Output("vc-celov-status", "children", allow_duplicate=True),
            Input("vc-celov-browse", "n_clicks"),
            State("vc-celov-out", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save condition volcano Celov (use {} for type)",
                defaultextension=".txt",
                initialfile="Condition_Volcano_{}.txt",
            )
            if not chosen:
                return no_update, "Celov path browse cancelled."
            return chosen, f"Celov output template: {chosen}"

        @app.callback(
            Output("vc-celov-status", "children"),
            Input("vc-celov-save", "n_clicks"),
            State("vc-celov-out", "value"),
            State("vc-celov-mode", "value"),
            State("vc-celov-score", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _save(n_clicks, out_path, mode, score, session_blob, project_blob):
            results = _VC_RUNTIME.get("results")
            if results is None:
                return "Run the volcano first."
            if not out_path:
                return "Choose an output path."
            try:
                weighed = _volcano_weighed_for_celov(results, score or "fold_change")
                entry = _active_dataset_entry(session_blob, project_blob)
                paths = save_classifier_celov(
                    weighed,
                    out_path,
                    mode=mode or "up_and_down",
                    locus_lookup=_VC_RUNTIME.get("locus_lookup"),
                    id_column=resolve_celov_id_col(entry),
                )
                return "Saved Celov: " + ", ".join(str(p) for p in paths)
            except Exception as exc:  # noqa: BLE001
                return f"Celov save error: {exc}"
