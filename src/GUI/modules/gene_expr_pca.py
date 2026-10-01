"""Gene → PCA: color PCA by expression of selected genes (Virtual-server)."""

from __future__ import annotations

import math
import re

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from src.biocyc.celov_multiomics_post import load_locus_lookup

from ..components.controls import (
    EXPORT_H,
    EXPORT_W,
    _marker_sizes,
    _point_symbols,
    aesthetic_options,
    apply_export_layout,
    fig_size_controls,
    parse_aes_choice,
    set_fig_size,
)
from ..data_store import session_from_store
from .clustering import _locus_path_from_session
from .pca import _PCA_RUNTIME, _run_pca

# Hardcoded locus / ID fields (no free column picker).
_NAME_FIELDS = (
    "geneID",
    "geneName",
    "geneNameA",
    "geneName A",
    "geneNameC",
    "gene Name C",
    "geneName C",
)

_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "gene_expr_pca"},
    "displaylogo": False,
}
_EXPR_COLS = 2


def _plotly_title(*lines: str) -> dict:
    text = "<br>".join(x for x in lines if x is not None and str(x).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


def _parse_paste(text: str | None) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[\s,;]+", str(text).strip())
    return [p for p in parts if p]


def _available_name_fields(lookup: pd.DataFrame | None) -> list[str]:
    out = ["geneID"]
    if lookup is None or lookup.empty:
        return out
    cols = {str(c) for c in lookup.columns}
    for name in _NAME_FIELDS:
        if name == "geneID":
            continue
        if name in cols:
            out.append(name)
    return out


def _resolve_gene_ids(
    tokens: list[str],
    field: str,
    expr_genes: list[str],
    lookup: pd.DataFrame | None,
) -> tuple[list[str], list[str]]:
    """Map paste/search tokens → expression geneIDs; return (ids, titles)."""
    have = set(map(str, expr_genes))
    ids: list[str] = []
    titles: list[str] = []
    if field == "geneID" or lookup is None or lookup.empty or field not in lookup.columns:
        for t in tokens:
            if t in have and t not in ids:
                ids.append(t)
                titles.append(t)
        return ids, titles
    # map field value → geneID via locusTag / geneID columns
    id_col = "geneID" if "geneID" in lookup.columns else (
        "locusTag" if "locusTag" in lookup.columns else None
    )
    if id_col is None:
        return [], []
    series = lookup[field].fillna("").astype(str)
    id_series = lookup[id_col].astype(str)
    for t in tokens:
        hits = id_series[series.str.lower() == t.lower()]
        for gid in hits.astype(str):
            if gid in have and gid not in ids:
                ids.append(gid)
                titles.append(f"{t} ({gid})" if t != gid else gid)
    return ids, titles


def _resolve_shape_size(shape, size, columns):
    shape_col, shape_const = parse_aes_choice(shape, columns)
    size_col, size_raw = parse_aes_choice(size, columns)
    size_const = None
    if size_raw is not None:
        try:
            size_const = float(size_raw)
        except (TypeError, ValueError):
            size_const = 10.0
    return shape_col, shape_const, size_col, size_const


def _expr_pca_figure(
    score_df,
    expr,
    genes,
    titles,
    x_col,
    y_col,
    var_ratio,
    shape=None,
    size=None,
) -> go.Figure:
    genes = [str(g) for g in genes if str(g) in expr.columns]
    empty = go.Figure()
    if not genes:
        empty.add_annotation(text="Select or paste genes present in the matrix.", showarrow=False)
        return empty
    n = len(genes)
    n_cols = min(_EXPR_COLS, n)
    n_rows = int(math.ceil(n / n_cols))
    titles_full = list(titles[:n]) + [""] * (n_rows * n_cols - n)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles_full,
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )
    x = pd.to_numeric(score_df[x_col], errors="coerce")
    y = pd.to_numeric(score_df[y_col], errors="coerce")
    hover = score_df.index.astype(str)
    shape_col, shape_const, size_col, size_const = _resolve_shape_size(
        shape, size, score_df.columns
    )
    if shape_col and shape_col in score_df.columns:
        symbols = _point_symbols(score_df[shape_col])
    else:
        symbols = shape_const or "circle"
    if size_col and size_col in score_df.columns:
        sizes = _marker_sizes(score_df[size_col])
    else:
        sizes = float(size_const) if size_const is not None else 8.0
    for i, gene in enumerate(genes):
        row, col = i // n_cols + 1, i % n_cols + 1
        title = titles[i] if i < len(titles) else gene
        vals = pd.to_numeric(expr[gene].reindex(score_df.index), errors="coerce")
        cmin = float(np.nanmin(vals.to_numpy(dtype=float))) if vals.notna().any() else 0.0
        cmax = float(np.nanmax(vals.to_numpy(dtype=float))) if vals.notna().any() else 1.0
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color=vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(thickness=12, outlinewidth=0),
                    cmin=cmin,
                    cmax=cmax,
                    symbol=symbols,
                    size=sizes,
                ),
                customdata=np.stack([hover, vals.to_numpy(dtype=float)], axis=1),
                hovertemplate=(
                    "%{customdata[0]}<br>" + title + ": %{customdata[1]:.4g}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=x_col if row == n_rows else "", row=row, col=col)
        fig.update_yaxes(title_text=y_col if col == 1 else "", row=row, col=col)
    var = list(map(float, var_ratio or []))
    sub = ""
    if len(var) >= 2:
        sub = f"PC1 {100 * var[0]:.1f}%, PC2 {100 * var[1]:.1f}%"
    fig.update_layout(title=_plotly_title("Gene expression on PCA", sub))
    apply_export_layout(fig, title_lines=2, legend=False, height=max(420, 320 * n_rows))
    return fig


class GeneExprPCAModule:
    id = "gene-expr-pca"
    label = "Gene → PCA"

    def layout(self, *, readonly: bool = False):
        return html.Div(
            [
                html.P(
                    "Paste gene IDs / names, or search by a hardcoded name field. "
                    "Color is always gene expression (viridis); shape / size use "
                    "metadata columns or fixed values (same as PCA).",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Name field"),
                                dcc.Dropdown(
                                    id="gxp-name-field",
                                    options=[{"label": f, "value": f} for f in ("geneID",)],
                                    value="geneID",
                                    clearable=False,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Search / pick"),
                                dcc.Dropdown(
                                    id="gxp-gene-pick",
                                    multi=True,
                                    searchable=True,
                                    placeholder="Type 2+ characters…",
                                ),
                            ],
                            md=5,
                        ),
                        dbc.Col(
                            [
                                html.Label("X"),
                                dcc.Dropdown(id="gxp-x", value="PC1", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Y"),
                                dcc.Dropdown(id="gxp-y", value="PC2", clearable=False),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Shape"),
                                dcc.Dropdown(
                                    id="gxp-shape",
                                    clearable=True,
                                    placeholder="metadata or fixed…",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Size"),
                                dcc.Dropdown(
                                    id="gxp-size",
                                    clearable=True,
                                    placeholder="metadata or fixed…",
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Label("Paste gene IDs or names (whitespace / comma / semicolon)"),
                dcc.Textarea(
                    id="gxp-paste",
                    placeholder="VC_00610 VC_01274 …",
                    style={"width": "100%", "height": "72px", "fontFamily": "monospace"},
                    className="mb-2",
                ),
                dbc.Button("Plot on PCA", id="gxp-run", color="primary", className="mb-2"),
                html.Div(id="gxp-status", className="text-muted small mb-2"),
                fig_size_controls("gxp", default_width=EXPORT_W, default_height=EXPORT_H),
                dcc.Loading(
                    dcc.Graph(id="gxp-fig", figure={}, config=_GRAPH_CONFIG),
                    type="default",
                ),
                dcc.Store(id="gxp-cache"),
            ]
        )

    def register_callbacks(self, app: Dash, *, readonly: bool = False) -> None:
        @app.callback(
            Output("gxp-name-field", "options"),
            Output("gxp-name-field", "value"),
            Input("session-store", "data"),
            Input("project-store", "data"),
            State("gxp-name-field", "value"),
        )
        def _fill_fields(session_blob, project_blob, current):
            lookup = None
            path = _locus_path_from_session(session_blob, project_blob)
            if path:
                try:
                    lookup = load_locus_lookup(path)
                except Exception:  # noqa: BLE001
                    lookup = None
            fields = _available_name_fields(lookup)
            opts = [{"label": f, "value": f} for f in fields]
            value = current if current in fields else fields[0]
            return opts, value

        @app.callback(
            Output("gxp-gene-pick", "options"),
            Input("gxp-name-field", "value"),
            Input("gxp-gene-pick", "search_value"),
            Input("gxp-gene-pick", "value"),
            Input("session-store", "data"),
            Input("project-store", "data"),
        )
        def _pick_options(field, search, selected, session_blob, project_blob):
            session = session_from_store(session_blob)
            if not session.ready or session.expression is None:
                return []
            field = field or "geneID"
            q = (search or "").strip().lower()
            chosen = [str(x) for x in (selected or [])]
            opts = [{"label": g, "value": g} for g in chosen]
            if len(q) < 2:
                return opts
            seen = set(chosen)
            if field == "geneID":
                for gid in session.expression.columns.astype(str):
                    if gid in seen:
                        continue
                    if q in gid.lower():
                        opts.append({"label": gid, "value": gid})
                        seen.add(gid)
                    if len(opts) >= 80:
                        break
                return opts
            path = _locus_path_from_session(session_blob, project_blob)
            if not path:
                return opts
            try:
                lookup = load_locus_lookup(path)
            except Exception:  # noqa: BLE001
                return opts
            if field not in lookup.columns:
                return opts
            id_col = "geneID" if "geneID" in lookup.columns else "locusTag"
            if id_col not in lookup.columns:
                return opts
            have = set(session.expression.columns.astype(str))
            for _, row in lookup.iterrows():
                label = str(row.get(field, "") or "")
                gid = str(row.get(id_col, "") or "")
                if not label or gid not in have or gid in seen:
                    continue
                if q in label.lower() or q in gid.lower():
                    opts.append({"label": f"{label} ({gid})", "value": gid})
                    seen.add(gid)
                if len(opts) >= 80:
                    break
            return opts

        @app.callback(
            Output("gxp-x", "options"),
            Output("gxp-y", "options"),
            Output("gxp-x", "value"),
            Output("gxp-y", "value"),
            Input("gxp-cache", "data"),
            Input("pca-cache", "data"),
        )
        def _axes(gxp_cache, pca_cache):
            n = 0
            if "score_df" in _PCA_RUNTIME:
                n = sum(
                    1
                    for c in _PCA_RUNTIME["score_df"].columns
                    if str(c).startswith("PC") and str(c)[2:].isdigit()
                )
            if not n and gxp_cache:
                n = int(gxp_cache.get("n_pcs") or 0)
            pcs = [f"PC{i}" for i in range(1, max(n, 2) + 1)]
            opts = [{"label": c, "value": c} for c in pcs]
            return opts, opts, "PC1", "PC2" if "PC2" in pcs else "PC1"

        @app.callback(
            Output("gxp-shape", "options"),
            Output("gxp-size", "options"),
            Input("session-store", "data"),
            Input("gxp-cache", "data"),
        )
        def _fill_aes(session_blob, _cache):
            cols = list((session_blob or {}).get("meta_columns", []))
            _color_opts, shape_opts, size_opts = aesthetic_options(cols)
            return shape_opts, size_opts

        @app.callback(
            Output("gxp-fig", "figure"),
            Output("gxp-status", "children"),
            Output("gxp-cache", "data"),
            Input("gxp-run", "n_clicks"),
            Input("gxp-fig-w", "value"),
            Input("gxp-fig-h", "value"),
            Input("gxp-shape", "value"),
            Input("gxp-size", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            State("gxp-name-field", "value"),
            State("gxp-gene-pick", "value"),
            State("gxp-paste", "value"),
            State("gxp-x", "value"),
            State("gxp-y", "value"),
            prevent_initial_call=True,
        )
        def _run(
            n_clicks,
            fig_w,
            fig_h,
            shape,
            size,
            session_blob,
            project_blob,
            field,
            picked,
            paste,
            x_col,
            y_col,
        ):
            empty = go.Figure()
            session = session_from_store(session_blob)
            if not session.ready or session.expression is None:
                return empty, session.error or "Load a dataset first.", None
            tokens = list(picked or []) + _parse_paste(paste)
            # de-dupe preserving order
            seen = set()
            uniq = []
            for t in tokens:
                if t not in seen:
                    seen.add(t)
                    uniq.append(str(t))
            if not uniq:
                return empty, "Paste or pick at least one gene.", None
            lookup = None
            path = _locus_path_from_session(session_blob, project_blob)
            if path:
                try:
                    lookup = load_locus_lookup(path)
                except Exception:  # noqa: BLE001
                    lookup = None
            genes, titles = _resolve_gene_ids(
                uniq,
                field or "geneID",
                list(session.expression.columns.astype(str)),
                lookup,
            )
            if not genes:
                return empty, "No pasted/picked names matched genes in the expression matrix.", None
            # Prefer existing PCA runtime; else compute
            if "score_df" in _PCA_RUNTIME and "var_ratio" in _PCA_RUNTIME:
                score_df = _PCA_RUNTIME["score_df"]
                var = _PCA_RUNTIME["var_ratio"]
            else:
                score_df, var, _explained, _loadings = _run_pca(session)
                _PCA_RUNTIME.update(
                    {
                        "score_df": score_df,
                        "var_ratio": list(map(float, var)),
                        "explained_variance": np.asarray(_explained, dtype=float),
                        "loadings": _loadings,
                    }
                )
            # Ensure metadata columns are present for shape/size
            if session.metadata is not None:
                missing = [c for c in session.metadata.columns if c not in score_df.columns]
                if missing:
                    score_df = score_df.join(session.metadata[missing])
            x_col = x_col if x_col in score_df.columns else "PC1"
            y_col = y_col if y_col in score_df.columns else (
                "PC2" if "PC2" in score_df.columns else "PC1"
            )
            fig = _expr_pca_figure(
                score_df,
                session.expression,
                genes,
                titles,
                x_col,
                y_col,
                var,
                shape=shape,
                size=size,
            )
            fig = set_fig_size(fig, fig_w, fig_h)
            n_pcs = sum(
                1
                for c in score_df.columns
                if str(c).startswith("PC") and str(c)[2:].isdigit()
            )
            msg = f"Showing {len(genes)} gene(s) on {x_col} vs {y_col}."
            return fig, msg, {"n_pcs": n_pcs, "genes": genes}
