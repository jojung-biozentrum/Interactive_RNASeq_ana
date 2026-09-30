"""Shared level-3 condition enrichment + sample-ID export."""

from __future__ import annotations

from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from src.GUI.components.controls import EXPORT_W, fig_size_controls, plotly_title, set_fig_size
from src.GUI.data_store import live_session, meta_columns_from_store
from src.GUI.modules.condition_enrichment import (
    DEFAULT_CAT_COLS,
    DEFAULT_NUM_COLS,
    cat_end_mats,
    cat_entry_mats,
    corr_vs_axes,
    fisher_bin_enrichment,
    fisher_odds_fig,
    heatmap_matrix_fig,
    mwu_bin_enrichment,
    present_cols,
    table_html,
)

CLF_AXIS = "Linear classifier"
_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "condition_enrichment"},
    "displaylogo": False,
}
_AXIS_HELP = (
    "Pick PC axes, the Linear classifier axis (n PCs chosen above in the "
    "classifier) or gene expressions."
)
_GRAD_HELP = "Pick gradient axis or gene expressions."
_HC_HELP = (
    "Enrichment uses dendrogram Bin A vs Bin B. Categorical: Fisher odds ratio per "
    "entry (2×2, BH-adjusted). Numeric: Mann–Whitney U (BH-adjusted). Gene "
    "expressions: same end-split / Pearson–Spearman enrichment as PCA, using "
    "selected gene expression values as continuous scores."
)


def enrichment_layout() -> html.Div:
    return html.Div(
        [
            html.H6("Condition enrichment"),
            html.P(id="enr-help", children=_AXIS_HELP, className="text-muted small"),
            html.Div(
                id="enr-axis-controls",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(id="enr-axes-label", children="Sample axes"),
                                    html.Div(
                                        id="enr-axes-dd-wrap",
                                        children=[dcc.Dropdown(id="enr-axes", multi=True)],
                                    ),
                                    html.Div(
                                        id="enr-grad-axis-wrap",
                                        style={"display": "none"},
                                        children=[
                                            dcc.Checklist(
                                                id="enr-grad-axis",
                                                options=[
                                                    {
                                                        "label": " gradient rank",
                                                        "value": "gradient rank",
                                                    }
                                                ],
                                                value=[],
                                                inline=True,
                                            ),
                                        ],
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Gene expressions"),
                                    dcc.Dropdown(id="enr-genes", multi=True),
                                ],
                                md=6,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    html.Div(
                        id="enr-fig-size-wrap",
                        children=[
                            fig_size_controls("enr", default_width=EXPORT_W, default_height=720),
                        ],
                    ),
                    html.Hr(),
                    html.H6("Categorical"),
                    html.P(
                        "For each unique entry in a categorical column, the axis is split "
                        "one-vs-rest: if that entry’s median lies higher on the axis, the "
                        "cut is its minimum score (everything at or above = entry-end); "
                        "if lower, the cut is its maximum (at or below = entry-end).",
                        className="text-muted small mb-1",
                    ),
                    html.P(
                        "Measures: ARI, homogeneity, completeness of that split vs the "
                        "true entry/other labels. var_ratio = within-entry / total "
                        "variance (small = tight). distance = |entry mean − label "
                        "mean| / √within-var (large = separated).",
                        className="text-muted small",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Metadata columns"),
                                    dcc.Dropdown(id="enr-cat-cols", multi=True),
                                ],
                                md=5,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Measures"),
                                    dcc.Checklist(
                                        id="enr-cat-measures",
                                        options=[
                                            {"label": " ARI (end)", "value": "ARI"},
                                            {"label": " homogeneity", "value": "homogeneity"},
                                            {"label": " completeness", "value": "completeness"},
                                            {"label": " var_ratio", "value": "var_ratio"},
                                            {"label": " distance", "value": "distance"},
                                        ],
                                        value=["ARI", "homogeneity", "completeness", "var_ratio"],
                                        inline=True,
                                    ),
                                ],
                                md=7,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    dbc.Button("Run enrichment", id="enr-cat-run", color="primary", className="mb-2"),
                    dcc.Graph(id="enr-cat-fig", figure=go.Figure(), config=_GRAPH_CONFIG),
                    html.Div(id="enr-cat-status", className="text-muted small mb-3"),
                    html.Hr(),
                    html.H6("Rankable"),
                    html.P(
                        "Correlate ranked numeric metadata with the selected axes.",
                        className="text-muted small mb-1",
                    ),
                    html.P(
                        "Measures: Pearson and Spearman.",
                        className="text-muted small",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Metadata columns"),
                                    dcc.Dropdown(id="enr-rank-cols", multi=True),
                                ],
                                md=5,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Measures"),
                                    dcc.Checklist(
                                        id="enr-rank-measures",
                                        options=[
                                            {"label": " Pearson", "value": "Pearson"},
                                            {"label": " Spearman", "value": "Spearman"},
                                        ],
                                        value=["Pearson", "Spearman"],
                                        inline=True,
                                    ),
                                ],
                                md=7,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    dbc.Button("Run enrichment", id="enr-rank-run", color="primary", className="mb-2"),
                    dcc.Graph(id="enr-rank-fig", figure=go.Figure(), config=_GRAPH_CONFIG),
                    html.Div(id="enr-rank-status", className="text-muted small mb-3"),
                ],
            ),
            html.Div(
                id="enr-hc-controls",
                style={"display": "none"},
                children=[
                    html.Div(id="enr-hc-bins", className="text-muted small mb-2"),
                    html.Hr(),
                    html.H6("Categorical"),
                    html.P(
                        "Fisher exact (2×2 entry × Bin A vs Bin B), BH-adjusted. "
                        "Odds ratio > 1: entry more common in Bin A; < 1: more "
                        "common in Bin B. Color = −log10(padj); details on hover.",
                        className="text-muted small",
                    ),
                    dcc.Dropdown(id="enr-hc-cat-cols", multi=True, className="mb-2"),
                    dbc.Button("Run enrichment", id="enr-hc-cat-run", color="primary", className="mb-2"),
                    dcc.Graph(id="enr-hc-cat-fig", figure=go.Figure(), config=_GRAPH_CONFIG),
                    html.Div(id="enr-hc-cat-status", className="text-muted small mb-3"),
                    html.Hr(),
                    html.H6("Numeric"),
                    html.P(
                        "Mann–Whitney U of ranked / numeric metadata between Bin A and Bin B, "
                        "BH-adjusted.",
                        className="text-muted small",
                    ),
                    dcc.Dropdown(id="enr-hc-num-cols", multi=True, className="mb-2"),
                    dbc.Button("Run enrichment", id="enr-hc-num-run", color="primary", className="mb-2"),
                    html.Div(id="enr-hc-num-table"),
                    html.Div(id="enr-hc-num-status", className="text-muted small mb-3"),
                    html.Hr(),
                    html.H6("Gene expressions"),
                    html.P(
                        "As in PCA: use expression of selected genes as continuous scores "
                        "(pick genes from the HC Genes volcano / gene-PCA selection).",
                        className="text-muted small",
                    ),
                    dcc.Dropdown(
                        id="enr-hc-genes",
                        multi=True,
                        placeholder="Select genes…",
                        className="mb-2",
                    ),
                    html.H6("Categorical (on gene expression)", className="mt-2"),
                    html.P(
                        "For each unique entry, each gene-expression score is split one-vs-rest "
                        "(entry-end cut at min or max of that entry). Measures: ARI, "
                        "homogeneity, completeness. var_ratio = within/total variance "
                        "(small = tight). distance = |entry mean − label mean| / "
                        "√within-var (large = separated).",
                        className="text-muted small",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Dropdown(id="enr-hc-gene-cat-cols", multi=True), md=5),
                            dbc.Col(
                                dcc.Checklist(
                                    id="enr-hc-gene-cat-measures",
                                    options=[
                                        {"label": " ARI (end)", "value": "ARI"},
                                        {"label": " homogeneity", "value": "homogeneity"},
                                        {"label": " completeness", "value": "completeness"},
                                        {"label": " var_ratio", "value": "var_ratio"},
                                        {"label": " distance", "value": "distance"},
                                    ],
                                    value=["ARI", "homogeneity", "completeness", "var_ratio"],
                                    inline=True,
                                ),
                                md=7,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    dbc.Button(
                        "Run enrichment",
                        id="enr-hc-gene-cat-run",
                        color="primary",
                        className="mb-2",
                    ),
                    dcc.Graph(id="enr-hc-gene-cat-fig", figure=go.Figure(), config=_GRAPH_CONFIG),
                    html.Div(id="enr-hc-gene-cat-status", className="text-muted small mb-3"),
                    html.H6("Rankable (on gene expression)", className="mt-2"),
                    html.P(
                        "Pearson / Spearman of ranked metadata vs selected gene expressions.",
                        className="text-muted small",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(dcc.Dropdown(id="enr-hc-gene-rank-cols", multi=True), md=5),
                            dbc.Col(
                                dcc.Checklist(
                                    id="enr-hc-gene-rank-measures",
                                    options=[
                                        {"label": " Pearson", "value": "Pearson"},
                                        {"label": " Spearman", "value": "Spearman"},
                                    ],
                                    value=["Pearson", "Spearman"],
                                    inline=True,
                                ),
                                md=7,
                            ),
                        ],
                        className="g-2 mb-2",
                    ),
                    dbc.Button(
                        "Run enrichment",
                        id="enr-hc-gene-rank-run",
                        color="primary",
                        className="mb-2",
                    ),
                    dcc.Graph(id="enr-hc-gene-rank-fig", figure=go.Figure(), config=_GRAPH_CONFIG),
                    html.Div(id="enr-hc-gene-rank-status", className="text-muted small mb-3"),
                ],
            ),
            html.Div(id="enr-table"),
            html.Div(id="enr-status", className="text-muted small mb-3"),
            html.Hr(),
        ]
    )


def _pca_frame():
    from src.GUI.modules.pca import _PCA_RUNTIME

    df = _PCA_RUNTIME.get("score_df")
    return df if isinstance(df, pd.DataFrame) else None


def _hc_frame():
    from src.GUI.modules.clustering import _HC_RUNTIME, _cut_clusters

    rt = _HC_RUNTIME
    df = rt.get("score_df")
    if not isinstance(df, pd.DataFrame) or "Z_samples" not in rt:
        return None
    out = df.copy()
    t = int(rt.get("t_use") or 2)
    labels = _cut_clusters(rt["Z_samples"], t)
    ids = list(rt.get("sample_ids") or out.index)
    cluster = pd.Series(labels, index=pd.Index(ids))
    out["cluster"] = pd.to_numeric(cluster.reindex(out.index), errors="coerce")
    return out


def _par_frame():
    from src.GUI.modules.condition_enrichment import assign_region_labels, region_rank_series
    from src.GUI.modules.parallel_conditions import _PAR_RUNTIME

    df = _PAR_RUNTIME.get("score_df")
    if not isinstance(df, pd.DataFrame):
        return None
    out = df.copy()
    levels = list(_PAR_RUNTIME.get("levels") or [])
    closest = _PAR_RUNTIME.get("closest") or {}
    out["_closest"] = assign_region_labels(out.index, closest, levels)
    out["closest rank"] = region_rank_series(out["_closest"], levels)
    return out


def _grad_frame():
    from src.GUI.modules.gene_gradients import _GRAD_RUNTIME

    meta = _GRAD_RUNTIME.get("meta")
    if not isinstance(meta, pd.DataFrame):
        return None
    out = meta.copy()
    order_col = _GRAD_RUNTIME.get("order_col")
    levels = list(_GRAD_RUNTIME.get("order_levels") or [])
    if order_col and order_col in out.columns and levels:
        rank_map = {str(lv): float(i + 1) for i, lv in enumerate(levels)}
        out["gradient rank"] = out[order_col].astype(str).map(rank_map)
    return out


def method_sample_frame(method: str) -> pd.DataFrame | None:
    if method == "pca":
        return _pca_frame()
    if method == "hc":
        return _hc_frame()
    if method == "parallel-conditions":
        return _par_frame()
    if method == "gene-gradients":
        return _grad_frame()
    return None


def available_sample_axes(method: str, df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    if method == "pca":
        pcs = [c for c in df.columns if str(c).startswith("PC") and str(c)[2:].isdigit()]
        extra = [CLF_AXIS] if CLF_AXIS in df.columns else []
        return pcs + extra
    if method == "hc":
        return []
    if method == "parallel-conditions":
        pcs = [c for c in df.columns if str(c).startswith("PC") and str(c)[2:].isdigit()]
        extra = ["closest rank"] if "closest rank" in df.columns else []
        return extra + pcs[:4]
    if method == "gene-gradients":
        return ["gradient rank"] if "gradient rank" in df.columns else []
    return []


def add_gene_axes(df: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    session = live_session()
    if session is None or session.expression is None:
        return df, []
    expr = session.expression
    axes = []
    out = df.copy()
    for g in genes:
        gid = str(g)
        if gid not in expr.columns:
            continue
        name = gid if gid not in out.columns else f"{gid} expr"
        out[name] = pd.to_numeric(expr[gid].reindex(out.index), errors="coerce")
        axes.append(name)
    return out, axes


def _hc_bin_sample_ids(selected) -> tuple[list[str], list[str], str]:
    """Resolve dendrogram Bin A vs Bin B sample IDs at the current cut."""
    from src.GUI.modules.clustering import (
        _HC_RUNTIME,
        _brace_clusters,
        _cut_clusters,
        _normalize_bin_sel,
    )

    rt = _HC_RUNTIME
    if "Z_samples" not in rt:
        raise ValueError("Run hierarchical clustering first.")
    bins = _normalize_bin_sel(selected)
    if not bins["a"]:
        raise ValueError("Assign at least one dendrogram cluster to Bin A.")
    if not bins["b"]:
        raise ValueError("Assign at least one dendrogram cluster to Bin B.")
    n = int(rt["n_samples"])
    t = max(2, min(int(rt.get("t_use") or 2), n))
    labels = _cut_clusters(rt["Z_samples"], t)
    sample_ids = [str(s) for s in rt["sample_ids"]]
    set_a = set(int(c) for c in bins["a"])
    set_b = set(int(c) for c in bins["b"])
    ids_a = [sid for sid, lab in zip(sample_ids, labels) if int(lab) in set_a]
    ids_b = [sid for sid, lab in zip(sample_ids, labels) if int(lab) in set_b]
    if not ids_a or not ids_b:
        raise ValueError(f"Empty group after cut (Bin A n={len(ids_a)}, Bin B n={len(ids_b)}).")
    title = (
        f"Bin A {{{_brace_clusters(bins['a'])}}} vs "
        f"Bin B {{{_brace_clusters(bins['b'])}}} ({t} clusters)"
    )
    return ids_a, ids_b, title


def register_enrichment_callbacks(app: Dash) -> None:
    @app.callback(
        Output("analysis-selected-genes", "data", allow_duplicate=True),
        Input("par-grad-genes", "value"),
        Input("grad-selected-genes", "data"),
        prevent_initial_call=True,
    )
    def _sync_other_genes(par_genes, grad_genes):
        genes = []
        for src in (par_genes, grad_genes):
            for g in src or []:
                gid = str(g)
                if gid not in genes:
                    genes.append(gid)
        return genes

    @app.callback(
        Output("enr-help", "children"),
        Output("enr-axis-controls", "style"),
        Output("enr-hc-controls", "style"),
        Output("enr-axes-label", "children"),
        Output("enr-axes-dd-wrap", "style"),
        Output("enr-grad-axis-wrap", "style"),
        Input("analysis-method", "data"),
    )
    def _mode_ui(method):
        if method == "hc":
            return (
                _HC_HELP,
                {"display": "none"},
                {"display": "block"},
                "Sample axes",
                {},
                {"display": "none"},
            )
        if method == "gene-gradients":
            return (
                _GRAD_HELP,
                {},
                {"display": "none"},
                "Gradient axis",
                {"display": "none"},
                {},
            )
        return (
            _AXIS_HELP,
            {},
            {"display": "none"},
            "Sample axes",
            {},
            {"display": "none"},
        )

    @app.callback(
        Output("enr-hc-bins", "children"),
        Input("analysis-method", "data"),
        Input("hc-cluster-sel", "data"),
        Input("hc-maxclust-applied", "data"),
        Input("hc-cache", "data"),
    )
    def _hc_bin_status(method, selected, _t, _cache):
        if method != "hc":
            return ""
        try:
            ids_a, ids_b, title = _hc_bin_sample_ids(selected)
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return f"{title} — n_A={len(ids_a)}, n_B={len(ids_b)}."

    @app.callback(
        Output("enr-axes", "options"),
        Output("enr-axes", "value"),
        Output("enr-grad-axis", "options"),
        Output("enr-grad-axis", "value"),
        Input("analysis-method", "data"),
        Input("pca-cache", "data"),
        Input("pca-clf-cache", "data"),
        Input("hc-cache", "data"),
        Input("par-cache", "data"),
        Input("grad-cache", "data"),
        State("enr-axes", "value"),
        State("enr-grad-axis", "value"),
    )
    def _fill_axes(method, *rest):
        current_axes, current_grad = rest[-2], rest[-1]
        method = method or "pca"
        df = method_sample_frame(method)
        axes = available_sample_axes(method, df)
        opts = [{"label": a, "value": a} for a in axes]
        if method == "gene-gradients":
            grad_opts = (
                [{"label": " gradient rank", "value": "gradient rank"}]
                if "gradient rank" in axes
                else []
            )
            keep_g = [a for a in (current_grad or []) if a in axes]
            grad_val = keep_g if keep_g else (["gradient rank"] if grad_opts else [])
            return [], [], grad_opts, grad_val
        keep = [a for a in (current_axes or []) if a in axes]
        if keep:
            value = keep
        elif method == "pca":
            value = axes[:2]
        else:
            value = axes[:1]
        return opts, value, [], []

    @app.callback(
        Output("enr-genes", "options"),
        Output("enr-genes", "value"),
        Input("analysis-selected-genes", "data"),
        Input("pca-expr-genes", "value"),
        Input("par-grad-genes", "value"),
        Input("grad-selected-genes", "data"),
        State("enr-genes", "value"),
    )
    def _fill_genes(shared, pca_genes, par_genes, grad_genes, current):
        genes = []
        for src in (shared, pca_genes, par_genes, grad_genes):
            for g in src or []:
                gid = str(g)
                if gid not in genes:
                    genes.append(gid)
        opts = [{"label": g, "value": g} for g in genes]
        keep = [g for g in (current or []) if g in genes]
        return opts, keep

    @app.callback(
        Output("enr-cat-cols", "options"),
        Output("enr-cat-cols", "value"),
        Output("enr-rank-cols", "options"),
        Output("enr-rank-cols", "value"),
        Input("session-store", "data"),
        State("enr-cat-cols", "value"),
        State("enr-rank-cols", "value"),
    )
    def _fill_cols(session_blob, cat_current, rank_current):
        cols = meta_columns_from_store(session_blob)
        opts = [{"label": c, "value": c} for c in cols]
        cat_pref = present_cols(cols, DEFAULT_CAT_COLS) or cols
        rank_pref = present_cols(cols, DEFAULT_NUM_COLS) or cols
        cat_val = [c for c in (cat_current or cat_pref) if c in cols] or list(cat_pref)
        rank_val = [c for c in (rank_current or rank_pref) if c in cols] or list(rank_pref)
        return opts, cat_val, opts, rank_val

    @app.callback(
        Output("enr-hc-cat-cols", "options"),
        Output("enr-hc-cat-cols", "value"),
        Output("enr-hc-num-cols", "options"),
        Output("enr-hc-num-cols", "value"),
        Output("enr-hc-gene-cat-cols", "options"),
        Output("enr-hc-gene-cat-cols", "value"),
        Output("enr-hc-gene-rank-cols", "options"),
        Output("enr-hc-gene-rank-cols", "value"),
        Input("session-store", "data"),
        State("enr-hc-cat-cols", "value"),
        State("enr-hc-num-cols", "value"),
        State("enr-hc-gene-cat-cols", "value"),
        State("enr-hc-gene-rank-cols", "value"),
    )
    def _fill_hc_cols(session_blob, cat_cur, num_cur, gcat_cur, grank_cur):
        cols = meta_columns_from_store(session_blob)
        opts = [{"label": c, "value": c} for c in cols]
        cat_pref = present_cols(cols, DEFAULT_CAT_COLS) or cols
        num_pref = present_cols(cols, DEFAULT_NUM_COLS) or cols
        def _pick(cur, pref):
            return [c for c in (cur or pref) if c in cols] or list(pref)
        return (
            opts, _pick(cat_cur, cat_pref),
            opts, _pick(num_cur, num_pref),
            opts, _pick(gcat_cur, cat_pref),
            opts, _pick(grank_cur, num_pref),
        )

    @app.callback(
        Output("enr-hc-genes", "options"),
        Output("enr-hc-genes", "value"),
        Input("analysis-selected-genes", "data"),
        Input("hc-gene-pca-genes", "data"),
        State("enr-hc-genes", "value"),
    )
    def _fill_hc_genes(shared, pca_genes, current):
        genes = []
        for src in (shared, pca_genes):
            for g in src or []:
                gid = str(g)
                if gid not in genes:
                    genes.append(gid)
        opts = [{"label": g, "value": g} for g in genes]
        keep = [g for g in (current or []) if g in genes]
        if not keep:
            keep = genes[: min(8, len(genes))]
        return opts, keep

    @app.callback(
        Output("enr-hc-cat-fig", "figure"),
        Output("enr-hc-cat-status", "children"),
        Input("enr-hc-cat-run", "n_clicks"),
        State("hc-cluster-sel", "data"),
        State("enr-hc-cat-cols", "value"),
        prevent_initial_call=True,
    )
    def _run_hc_cat(_n, selected, cols):
        empty = go.Figure()
        try:
            ids_a, ids_b, title = _hc_bin_sample_ids(selected)
        except Exception as exc:  # noqa: BLE001
            return empty, str(exc)
        df = _hc_frame()
        if df is None:
            return empty, "Run hierarchical clustering first."
        use = [c for c in (cols or []) if c in df.columns]
        if not use:
            return empty, "Select categorical metadata columns."
        table = fisher_bin_enrichment(df, ids_a, ids_b, use)
        if table is None or table.empty:
            return empty, "No Fisher results for the selected columns."
        fig = fisher_odds_fig(
            table,
            title=plotly_title("Fisher odds ratio", title),
        )
        return fig, f"Fisher odds ratio: {title}. {len(table)} entr{'y' if len(table) == 1 else 'ies'}."

    @app.callback(
        Output("enr-hc-num-table", "children"),
        Output("enr-hc-num-status", "children"),
        Input("enr-hc-num-run", "n_clicks"),
        State("hc-cluster-sel", "data"),
        State("enr-hc-num-cols", "value"),
        prevent_initial_call=True,
    )
    def _run_hc_num(_n, selected, cols):
        try:
            ids_a, ids_b, title = _hc_bin_sample_ids(selected)
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)
        df = _hc_frame()
        if df is None:
            return "", "Run hierarchical clustering first."
        use = [c for c in (cols or []) if c in df.columns]
        if not use:
            return "", "Select numeric metadata columns."
        table = mwu_bin_enrichment(df, ids_a, ids_b, use)
        if table is None or table.empty:
            return "", "No Mann–Whitney results for the selected columns."
        return table_html(table, max_rows=80), f"Mann–Whitney U: {title}. {len(table)} row(s)."

    @app.callback(
        Output("enr-hc-gene-cat-fig", "figure"),
        Output("enr-hc-gene-cat-status", "children"),
        Input("enr-hc-gene-cat-run", "n_clicks"),
        State("enr-hc-genes", "value"),
        State("enr-hc-gene-cat-cols", "value"),
        State("enr-hc-gene-cat-measures", "value"),
        prevent_initial_call=True,
    )
    def _run_hc_gene_cat(_n, genes, cols, measures):
        empty = go.Figure()
        df = _hc_frame()
        if df is None:
            return empty, "Run hierarchical clustering first."
        df, axes = add_gene_axes(df, list(genes or []))
        if not axes:
            return empty, "Select one or more gene expressions."
        cols = [c for c in (cols or []) if c in df.columns]
        measures = list(measures or [])
        if not cols or not measures:
            return empty, "Select categorical columns and measures."
        mats, titles = [], []
        end_want = [m for m in ("ARI", "homogeneity", "completeness") if m in measures]
        if end_want:
            end_mats = cat_end_mats(df, cols, axes)
            for m in end_want:
                mats.append(end_mats[m])
                titles.append(m)
        if "var_ratio" in measures or "distance" in measures:
            var_mat, dist_mat = cat_entry_mats(df, cols, axes)
            if "var_ratio" in measures:
                mats.append(var_mat)
                titles.append("var_ratio (within / total)")
            if "distance" in measures:
                mats.append(dist_mat)
                titles.append("mean |distance| / √within-var")
        if not mats:
            return empty, "Select at least one categorical measure."
        fig = heatmap_matrix_fig(
            mats,
            titles,
            title=plotly_title("Categorical entries vs gene expression", ", ".join(axes[:6])),
            zmin=0 if any(t.startswith("var") or t.startswith("mean") for t in titles) else None,
        )
        return fig, f"Categorical: {len(axes)} gene(s) × {len(cols)} columns."

    @app.callback(
        Output("enr-hc-gene-rank-fig", "figure"),
        Output("enr-hc-gene-rank-status", "children"),
        Input("enr-hc-gene-rank-run", "n_clicks"),
        State("enr-hc-genes", "value"),
        State("enr-hc-gene-rank-cols", "value"),
        State("enr-hc-gene-rank-measures", "value"),
        prevent_initial_call=True,
    )
    def _run_hc_gene_rank(_n, genes, cols, measures):
        empty = go.Figure()
        df = _hc_frame()
        if df is None:
            return empty, "Run hierarchical clustering first."
        df, axes = add_gene_axes(df, list(genes or []))
        if not axes:
            return empty, "Select one or more gene expressions."
        cols = [c for c in (cols or []) if c in df.columns]
        if not cols:
            return empty, "Select rankable metadata columns."
        want = [m for m in ("Pearson", "Spearman") if m in (measures or [])] or [
            "Pearson",
            "Spearman",
        ]
        mats = corr_vs_axes(df, cols, axes)
        fig = heatmap_matrix_fig(
            [mats[m] for m in want],
            want,
            title=plotly_title("Rankable metadata vs gene expression", ", ".join(axes[:6])),
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
        )
        return fig, f"Rankable: {len(axes)} gene(s) × {len(cols)} columns."

    @app.callback(
        Output("enr-cat-fig", "figure"),
        Output("enr-rank-fig", "figure"),
        Output("enr-cat-status", "children"),
        Output("enr-rank-status", "children"),
        Output("enr-table", "children"),
        Output("enr-status", "children"),
        Input("enr-cat-run", "n_clicks"),
        Input("enr-rank-run", "n_clicks"),
        Input("enr-fig-w", "value"),
        Input("enr-fig-h", "value"),
        State("analysis-method", "data"),
        State("enr-axes", "value"),
        State("enr-grad-axis", "value"),
        State("enr-genes", "value"),
        State("enr-cat-cols", "value"),
        State("enr-cat-measures", "value"),
        State("enr-rank-cols", "value"),
        State("enr-rank-measures", "value"),
        prevent_initial_call=True,
    )
    def _run(
        n_cat,
        n_rank,
        fig_w,
        fig_h,
        method,
        axes,
        grad_axis,
        genes,
        cat_cols,
        cat_measures,
        rank_cols,
        rank_measures,
    ):
        empty = go.Figure()
        tid = callback_context.triggered_id
        if method == "hc":
            return no_update, no_update, no_update, no_update, no_update, no_update

        do_cat = tid in ("enr-cat-run", "enr-fig-w", "enr-fig-h")
        do_rank = tid in ("enr-rank-run", "enr-fig-w", "enr-fig-h")
        if not do_cat and not do_rank:
            return no_update, no_update, no_update, no_update, no_update, no_update

        df = method_sample_frame(method or "pca")
        if df is None:
            msg = "Run the sample-level analysis first."
            return (
                empty if do_cat else no_update,
                empty if do_rank else no_update,
                msg if do_cat else no_update,
                msg if do_rank else no_update,
                "",
                msg,
            )
        if method == "gene-gradients":
            axes = [a for a in (grad_axis or []) if a in df.columns]
        else:
            axes = [a for a in (axes or []) if a in df.columns]
        df, gene_axes = add_gene_axes(df, list(genes or []))
        axes = axes + gene_axes
        if not axes:
            msg = (
                "Select the gradient axis and/or gene expressions."
                if method == "gene-gradients"
                else "Select one or more axes or gene expressions."
            )
            return (
                empty if do_cat else no_update,
                empty if do_rank else no_update,
                msg if do_cat else no_update,
                msg if do_rank else no_update,
                "",
                msg,
            )

        cat_cols = [c for c in (cat_cols or []) if c in df.columns]
        rank_cols = [c for c in (rank_cols or []) if c in df.columns]
        cat_measures = list(cat_measures or [])
        rank_measures = list(rank_measures or [])
        cat_fig, cat_msg = no_update, no_update
        rank_fig, rank_msg = no_update, no_update

        try:
            if do_cat:
                cat_fig, cat_msg = empty, "Select categorical columns and measures."
                if cat_cols and cat_measures:
                    mats = []
                    titles = []
                    end_want = [
                        m for m in ("ARI", "homogeneity", "completeness") if m in cat_measures
                    ]
                    if end_want:
                        end_mats = cat_end_mats(df, cat_cols, axes)
                        for m in end_want:
                            mats.append(end_mats[m])
                            titles.append(m)
                    if "var_ratio" in cat_measures or "distance" in cat_measures:
                        var_mat, dist_mat = cat_entry_mats(df, cat_cols, axes)
                        if "var_ratio" in cat_measures:
                            mats.append(var_mat)
                            titles.append("var_ratio (within / total)")
                        if "distance" in cat_measures:
                            mats.append(dist_mat)
                            titles.append("mean |distance| / √within-var")
                    if mats:
                        cat_fig = heatmap_matrix_fig(
                            mats,
                            titles,
                            title=plotly_title(
                                "Categorical entries vs axes", ", ".join(axes[:6])
                            ),
                            zmin=0
                            if any(
                                t.startswith("var") or t.startswith("mean") for t in titles
                            )
                            else None,
                        )
                        needed = int(cat_fig.layout.height or 720)
                        cat_fig = set_fig_size(
                            cat_fig, fig_w, max(fig_h or 0, needed), default_height=needed
                        )
                        cat_msg = f"Categorical: {len(axes)} axis/axes × {len(cat_cols)} columns."
                    else:
                        cat_msg = "Select at least one categorical measure."
                elif cat_cols:
                    cat_msg = "Select at least one categorical measure."

            if do_rank:
                rank_fig, rank_msg = empty, "Select rankable columns and measures."
                if rank_cols:
                    want = [m for m in ("Pearson", "Spearman") if m in rank_measures] or [
                        "Pearson",
                        "Spearman",
                    ]
                    mats = corr_vs_axes(df, rank_cols, axes)
                    rank_fig = heatmap_matrix_fig(
                        [mats[m] for m in want],
                        want,
                        title=plotly_title(
                            "Rankable metadata vs axes", ", ".join(axes[:6])
                        ),
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu_r",
                    )
                    needed = int(rank_fig.layout.height or 720)
                    rank_fig = set_fig_size(
                        rank_fig, fig_w, max(fig_h or 0, needed), default_height=needed
                    )
                    rank_msg = f"Rankable: {len(axes)} axis/axes × {len(rank_cols)} columns."
        except Exception as exc:  # noqa: BLE001
            msg = f"Enrichment error: {exc}"
            return (
                empty if do_cat else no_update,
                empty if do_rank else no_update,
                msg if do_cat else no_update,
                msg if do_rank else no_update,
                "",
                msg,
            )

        return cat_fig, rank_fig, cat_msg, rank_msg, "", ""
