"""Gene expression profiles: color PCA by expression of selected genes."""

from __future__ import annotations

import math
import re

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.biocyc.celov_multiomics_post import load_locus_lookup

from ..components.controls import (
    _marker_sizes,
    _point_symbols,
    aesthetic_options,
    apply_export_layout,
    graph_export_config,
    parse_aes_choice,
)
from ..components.gene_meta_mark import GENE_HOVER_COLUMNS
from ..components.sample_detail import (
    plot_with_sample_detail,
    register_sample_detail_callback,
)
from ..data_store import session_from_store
from .clustering import _locus_path_from_session
from .pca import _PCA_RUNTIME, _run_pca, _score_frame_for_plot

_GRAPH_CONFIG = graph_export_config("gene_expression_profiles")
_EXPR_COLS = 3
_PANEL = 380  # square panel cell (px)
_PANEL_MARGIN = dict(l=55, r=90, t=56, b=50)  # colorbar + titles
_SEARCH_LIMIT = 80
_GXP_RUNTIME: dict = {}


def _gxp_fig_size(n_rows: int, n_cols: int, *, use_3d: bool = False) -> tuple[int, int]:
    """Overall figure size so each subplot panel is roughly square."""
    cell = _PANEL + (40 if use_3d else 0)
    w = _PANEL_MARGIN["l"] + _PANEL_MARGIN["r"] + n_cols * cell
    h = _PANEL_MARGIN["t"] + _PANEL_MARGIN["b"] + n_rows * cell
    return int(w), int(h)


def _plotly_title(*lines: str) -> dict:
    text = "<br>".join(x for x in lines if x is not None and str(x).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


def _parse_gene_ids(text: str | None) -> list[str]:
    """Split pasted geneIDs (same space-joined form as PCA Top genes)."""
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[\s,;]+", str(text).strip())
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        gid = str(p).strip()
        if not gid or gid in seen:
            continue
        seen.add(gid)
        out.append(gid)
    return out


def _lookup_id_col(lookup: pd.DataFrame) -> str | None:
    if "geneID" in lookup.columns:
        return "geneID"
    if "locusTag" in lookup.columns:
        return "locusTag"
    return None


def _display_name_from_row(row: pd.Series) -> str | None:
    for col in GENE_HOVER_COLUMNS:
        if col not in row.index:
            continue
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        text = str(val).strip()
        if text:
            return text.split(";")[0].strip() or text
    return None


def _titles_for_genes(genes: list[str], lookup: pd.DataFrame | None) -> list[str]:
    """Panel titles from locus name columns when available."""
    if lookup is None or lookup.empty:
        return list(genes)
    id_col = _lookup_id_col(lookup)
    if id_col is None:
        return list(genes)
    by_id: dict[str, str] = {}
    for _, row in lookup.iterrows():
        gid = str(row.get(id_col, "") or "")
        if not gid or gid in by_id:
            continue
        label = _display_name_from_row(row)
        if label and label != gid:
            by_id[gid] = f"{label} ({gid})"
    return [by_id.get(g, g) for g in genes]


def _gene_pick_options(
    expression: pd.DataFrame,
    lookup: pd.DataFrame | None,
    search: str | None,
    selected,
    *,
    limit: int = _SEARCH_LIMIT,
) -> list[dict]:
    """Search geneID and all locus-lookup columns (2+ characters)."""
    have = set(expression.columns.astype(str))
    chosen = [str(x) for x in (selected or []) if str(x) in have]
    opts = [{"label": g, "value": g} for g in chosen]
    q = (search or "").strip().lower()
    if len(q) < 2:
        return opts
    seen = set(chosen)
    id_col = _lookup_id_col(lookup) if lookup is not None and not lookup.empty else None
    if id_col is not None and lookup is not None:
        for _, row in lookup.iterrows():
            gid = str(row.get(id_col, "") or "")
            if not gid or gid not in have or gid in seen:
                continue
            pieces = [gid]
            for c in lookup.columns:
                val = row.get(c)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                text = str(val).strip()
                if text:
                    pieces.append(text)
            if q not in " ".join(pieces).lower():
                continue
            display = _display_name_from_row(row)
            label = f"{display} ({gid})" if display and display != gid else gid
            opts.append({"label": label, "value": gid})
            seen.add(gid)
            if len(opts) >= limit:
                return opts
    for gid in expression.columns.astype(str):
        if gid in seen:
            continue
        if q in gid.lower():
            opts.append({"label": gid, "value": gid})
            seen.add(gid)
        if len(opts) >= limit:
            break
    return opts


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
    z_col=None,
) -> go.Figure:
    genes = [str(g) for g in genes if str(g) in expr.columns]
    empty = go.Figure()
    if not genes:
        empty.add_annotation(text="Select genes present in the matrix.", showarrow=False)
        return empty
    n = len(genes)
    n_cols = min(_EXPR_COLS, n)
    n_rows = int(math.ceil(n / n_cols))
    titles_full = list(titles[:n]) + [""] * (n_rows * n_cols - n)
    use_3d = bool(z_col) and z_col in score_df.columns
    specs = (
        [[{"type": "scatter3d"} for _ in range(n_cols)] for _ in range(n_rows)]
        if use_3d
        else None
    )
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles_full,
        specs=specs,
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )
    x = pd.to_numeric(score_df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(score_df[y_col], errors="coerce").to_numpy(dtype=float)
    z = (
        pd.to_numeric(score_df[z_col], errors="coerce").to_numpy(dtype=float)
        if use_3d
        else None
    )
    hover = score_df.index.astype(str).to_numpy()
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
    var = list(map(float, var_ratio or []))

    def _pc_axis_label(col: str) -> str:
        if not str(col).startswith("PC"):
            return str(col)
        try:
            i = int(str(col)[2:]) - 1
        except ValueError:
            return str(col)
        if 0 <= i < len(var):
            return f"{col} ({100 * var[i]:.1f}%)"
        return str(col)

    x_label = _pc_axis_label(x_col)
    y_label = _pc_axis_label(y_col)
    z_label = _pc_axis_label(z_col) if use_3d else ""
    for i, gene in enumerate(genes):
        row, col = i // n_cols + 1, i % n_cols + 1
        title = titles[i] if i < len(titles) else gene
        vals = pd.to_numeric(expr[gene].reindex(score_df.index), errors="coerce")
        color = vals.to_numpy(dtype=float)
        finite = color[np.isfinite(color)]
        cmin = float(np.nanmin(finite)) if finite.size else 0.0
        cmax = float(np.nanmax(finite)) if finite.size else 1.0
        if cmin == cmax:
            cmax = cmin + 1.0
        marker = dict(
            color=color,
            colorscale="Viridis",
            showscale=(i == 0),
            colorbar=dict(thickness=12, outlinewidth=0) if i == 0 else None,
            cmin=cmin,
            cmax=cmax,
            symbol=symbols,
            size=sizes,
        )
        if use_3d:
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=marker,
                    customdata=np.stack([hover, color], axis=1),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        + title
                        + ": %{customdata[1]:.4g}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.update_scenes(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title=z_label,
                row=row,
                col=col,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=marker,
                    customdata=np.stack([hover, color], axis=1),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        + title
                        + ": %{customdata[1]:.4g}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text=x_label if row == n_rows else "", row=row, col=col)
            fig.update_yaxes(title_text=y_label if col == 1 else "", row=row, col=col)
    if not use_3d:
        # Same square PC domain in every panel.
        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        half = 0.525 * max(xmax - xmin, ymax - ymin, 1e-9)
        for gi in range(n):
            row, col = gi // n_cols + 1, gi % n_cols + 1
            fig.update_xaxes(
                range=[cx - half, cx + half],
                scaleanchor="y",
                scaleratio=1,
                constrain="domain",
                row=row,
                col=col,
            )
            fig.update_yaxes(
                range=[cy - half, cy + half],
                constrain="domain",
                row=row,
                col=col,
            )
    fig_w, fig_h = _gxp_fig_size(n_rows, n_cols, use_3d=use_3d)
    fig.update_layout(title=_plotly_title("Gene expression on PCA"))
    apply_export_layout(
        fig,
        title_lines=1,
        legend=False,
        width=fig_w,
        height=fig_h,
        bottom=_PANEL_MARGIN["b"],
    )
    fig.update_layout(margin=dict(**_PANEL_MARGIN))
    return fig


class GeneExprPCAModule:
    id = "gene-expr-pca"
    label = "Gene expression profiles"

    def layout(self):
        return html.Div(
            [
                html.P(
                    "Paste geneIDs from PCA Top genes, and/or search any locus-lookup "
                    "column (geneID, names, …). Color is gene expression (viridis); "
                    "shape / size can use metadata columns or fixed values.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("geneIDs (paste from PCA)"),
                                dcc.Textarea(
                                    id="gxp-gene-ids",
                                    placeholder="e.g. PA0001 PA0002 …",
                                    style={
                                        "width": "100%",
                                        "height": "72px",
                                        "fontFamily": "monospace",
                                        "fontSize": "12px",
                                    },
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            [
                                html.Label("\u00a0"),
                                dbc.Button(
                                    "Add pasted IDs to selection",
                                    id="gxp-ids-add",
                                    color="secondary",
                                    outline=True,
                                    size="sm",
                                    className="w-100",
                                ),
                            ],
                            md=4,
                            className="d-flex flex-column justify-content-end",
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Search / pick (all locus columns)"),
                                dcc.Dropdown(
                                    id="gxp-gene-pick",
                                    multi=True,
                                    searchable=True,
                                    placeholder="Type 2+ characters…",
                                ),
                            ],
                            md=12,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
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
                        dbc.Col(
                            [
                                html.Label("Z (optional 3D)"),
                                dcc.Dropdown(id="gxp-z", placeholder="None", clearable=True),
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
                dbc.Button("Plot on PCA", id="gxp-run", color="primary", className="mb-2"),
                html.Div(id="gxp-status", className="text-muted small mb-2"),
                dcc.Loading(
                    plot_with_sample_detail(
                        "gxp-fig",
                        "gxp-sample-detail",
                        graph_config=_GRAPH_CONFIG,
                        graph_md=8,
                        detail_md=4,
                    ),
                    type="default",
                ),
                dcc.Store(id="gxp-cache"),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output("gxp-gene-pick", "options"),
            Input("gxp-gene-pick", "search_value"),
            Input("gxp-gene-pick", "value"),
            Input("session-store", "data"),
            Input("project-store", "data"),
        )
        def _pick_options(search, selected, session_blob, project_blob):
            session = session_from_store(session_blob)
            if not session.ready or session.expression is None:
                return []
            lookup = None
            path = _locus_path_from_session(session_blob, project_blob)
            if path:
                try:
                    lookup = load_locus_lookup(path)
                except Exception:  # noqa: BLE001
                    lookup = None
            return _gene_pick_options(session.expression, lookup, search, selected)

        @app.callback(
            Output("gxp-gene-pick", "value"),
            Output("gxp-status", "children", allow_duplicate=True),
            Input("gxp-ids-add", "n_clicks"),
            State("gxp-gene-ids", "value"),
            State("gxp-gene-pick", "value"),
            State("session-store", "data"),
            prevent_initial_call=True,
        )
        def _add_pasted_ids(n_clicks, paste_text, selected, session_blob):
            session = session_from_store(session_blob)
            if not session.ready or session.expression is None:
                return no_update, session.error or "Load a dataset first."
            have = set(session.expression.columns.astype(str))
            chosen = [str(x) for x in (selected or []) if str(x) in have]
            pasted = _parse_gene_ids(paste_text)
            if not pasted:
                return no_update, "Paste geneIDs first (from PCA Top genes)."
            added = 0
            missing = []
            for gid in pasted:
                if gid in have:
                    if gid not in chosen:
                        chosen.append(gid)
                        added += 1
                else:
                    missing.append(gid)
            msg = f"Added {added} geneID(s); selection has {len(chosen)}."
            if missing:
                msg += (
                    f" Not in matrix: {', '.join(missing[:5])}"
                    f"{'…' if len(missing) > 5 else ''}."
                )
            return chosen, msg

        @app.callback(
            Output("gxp-x", "options"),
            Output("gxp-y", "options"),
            Output("gxp-z", "options"),
            Output("gxp-x", "value"),
            Output("gxp-y", "value"),
            Output("gxp-z", "value"),
            Input("gxp-cache", "data"),
            Input("pca-cache", "data"),
            State("gxp-z", "value"),
        )
        def _axes(gxp_cache, pca_cache, z_cur):
            n = int((pca_cache or {}).get("n_pcs") or 0)
            if not n and "score_df" in _PCA_RUNTIME:
                n = sum(
                    1
                    for c in _PCA_RUNTIME["score_df"].columns
                    if str(c).startswith("PC") and str(c)[2:].isdigit()
                )
            if not n and gxp_cache:
                n = int(gxp_cache.get("n_pcs") or 0)
            pcs = [f"PC{i}" for i in range(1, max(n, 2) + 1)]
            opts = [{"label": c, "value": c} for c in pcs]
            z_val = z_cur if z_cur in pcs else None
            return opts, opts, opts, "PC1", "PC2" if "PC2" in pcs else "PC1", z_val

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
            Output("gxp-fig", "style"),
            Output("gxp-status", "children"),
            Output("gxp-cache", "data"),
            Input("gxp-run", "n_clicks"),
            Input("gxp-shape", "value"),
            Input("gxp-size", "value"),
            Input("gxp-x", "value"),
            Input("gxp-y", "value"),
            Input("gxp-z", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            State("pca-cache", "data"),
            State("gxp-gene-pick", "value"),
            State("gxp-gene-ids", "value"),
            State("gxp-cache", "data"),
            prevent_initial_call=True,
        )
        def _run(
            n_clicks,
            shape,
            size,
            x_col,
            y_col,
            z_col,
            session_blob,
            project_blob,
            pca_cache,
            picked,
            paste_text,
            gxp_cache,
        ):
            empty = go.Figure()
            style = {}
            session = session_from_store(session_blob)
            if not session.ready or session.expression is None:
                return empty, style, session.error or "Load a dataset first.", no_update
            have = set(session.expression.columns.astype(str))
            genes = []
            for gid in list(picked or []) + _parse_gene_ids(paste_text):
                gid = str(gid)
                if gid in have and gid not in genes:
                    genes.append(gid)
            if not genes and gxp_cache and gxp_cache.get("genes"):
                genes = [g for g in gxp_cache["genes"] if g in have]
            if not genes:
                missing = [g for g in _parse_gene_ids(paste_text) if g not in have]
                if missing:
                    return (
                        empty,
                        style,
                        f"No matching geneIDs in matrix "
                        f"(e.g. {', '.join(missing[:5])}{'…' if len(missing) > 5 else ''}).",
                        no_update,
                    )
                return empty, style, "Paste or pick at least one geneID.", no_update
            lookup = None
            path = _locus_path_from_session(session_blob, project_blob)
            if path:
                try:
                    lookup = load_locus_lookup(path)
                except Exception:  # noqa: BLE001
                    lookup = None
            titles = _titles_for_genes(genes, lookup)
            score_df = _score_frame_for_plot(pca_cache)
            var = list((pca_cache or {}).get("var_ratio") or [])
            if score_df is None or not any(
                str(c).startswith("PC") and str(c)[2:].isdigit() for c in score_df.columns
            ):
                if "score_df" in _PCA_RUNTIME and "var_ratio" in _PCA_RUNTIME:
                    score_df = _PCA_RUNTIME["score_df"]
                    var = list(_PCA_RUNTIME["var_ratio"])
                else:
                    try:
                        score_df, var, _explained, _loadings = _run_pca(session)
                        _PCA_RUNTIME.update(
                            {
                                "score_df": score_df,
                                "var_ratio": list(map(float, var)),
                                "explained_variance": np.asarray(_explained, dtype=float),
                                "loadings": _loadings,
                            }
                        )
                        var = list(map(float, var))
                    except Exception as exc:  # noqa: BLE001
                        return empty, style, f"PCA error: {exc}", no_update
            if session.metadata is not None:
                missing = [c for c in session.metadata.columns if c not in score_df.columns]
                if missing:
                    score_df = score_df.join(session.metadata[missing])
            x_col = x_col if x_col in score_df.columns else "PC1"
            y_col = y_col if y_col in score_df.columns else (
                "PC2" if "PC2" in score_df.columns else "PC1"
            )
            z_col = z_col if z_col and z_col in score_df.columns else None
            n_cols = min(_EXPR_COLS, len(genes))
            n_rows = int(math.ceil(len(genes) / n_cols))
            fig_w, fig_h = _gxp_fig_size(n_rows, n_cols, use_3d=bool(z_col))
            style = {"width": f"{fig_w}px", "height": f"{fig_h}px"}
            try:
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
                    z_col=z_col,
                )
            except Exception as exc:  # noqa: BLE001
                return empty, style, f"Plot error: {exc}", no_update
            _GXP_RUNTIME["score_df"] = score_df
            n_pcs = sum(
                1
                for c in score_df.columns
                if str(c).startswith("PC") and str(c)[2:].isdigit()
            )
            axes = f"{x_col} vs {y_col}" + (f" vs {z_col}" if z_col else "")
            msg = f"Showing {len(genes)} gene(s) on {axes}."
            return fig, style, msg, {"n_pcs": n_pcs, "genes": genes}

        register_sample_detail_callback(
            app,
            graph_id="gxp-fig",
            detail_id="gxp-sample-detail",
            cache_id="gxp-cache",
            get_score_df=lambda _c: _GXP_RUNTIME.get("score_df"),
        )
