"""PCA: interactive scatter + variance barplot, optional linear classifier, Celov export."""

from __future__ import annotations

from pathlib import Path

from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from src.GUI.components.gene_scores import weighed_genes_from_pc
from src.GUI.project import resolve_celov_id_col

from ..components.controls import (
    EXPORT_H,
    EXPORT_PCA_SCREE_H,
    EXPORT_W,
    aesthetic_options,
    aesthetic_panel,
    apply_export_layout,
    build_scatter,
    equal_xy_axes,
    fig_size_controls,
    plotly_title as _plotly_title,
    resolve_aes as _resolve_aes,
    set_fig_size,
)
from ..components.sample_detail import plot_with_sample_detail, register_sample_detail_callback
from ..components.folder_browser import pick_file_dialog, pick_save_file_dialog
from ..components.gene_meta_mark import filter_ids_by_search
from ..data_store import (
    SessionData,
    active_dataset_name as _active_dataset_name,
    live_session,
    meta_columns_from_store,
    meta_levels,
    session_from_store,
)
from .condition_enrichment import (
    DEFAULT_CAT_COLS,
    DEFAULT_NUM_COLS,
    ari_figure,
    biofilm_axis_scores,
    cat_entry_mats,
    corr_vs_axes,
    heatmap_matrix_fig,
    pc_separation_ari,
    present_cols,
)
from .pca_classifier import (
    add_decision_boundary,
    build_weighed_genes,
    classifier_performance,
    encode_binary_labels,
    fit_pc_classifier,
    performance_figure,
    save_classifier_celov,
)

_AES_ALL = "__all__"
_AES_PREFIX = "pca-aes"
_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "pca_plot"},
    "displaylogo": False,
}
_DEFAULT_AES = {
    "color": None,
    "shape": None,
    "size": None,
    "alpha": 0.85,
}

# Server-side PCA result (full scores / loadings) — local single-user app
_PCA_RUNTIME: dict = {}
_GENE_BY_ID = "__geneID__"


def _run_pca(session: SessionData):
    """Same as notebook: ``PCA().fit(normalizedCounts)`` then ``transform``."""
    X = session.numeric_matrix()
    pca = PCA()
    pca.fit(X)
    scores = pca.transform(X)
    cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    score_df = pd.DataFrame(scores, index=session.expression.index, columns=cols)
    score_df = score_df.join(session.metadata)
    # Correlation loadings: components.T * sqrt(explained_variance) (notebook pcaLoadings)
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=session.expression.columns.astype(str),
        columns=cols,
    )
    return score_df, pca.explained_variance_ratio_, pca.explained_variance_, loadings


def _scree_trace(var_ratio, top_n: int = 10) -> go.Bar:
    var_ratio = list(map(float, var_ratio))
    n_show = min(int(top_n), len(var_ratio))
    pcs = [f"PC{i + 1}" for i in range(n_show)]
    vals = [round(100 * var_ratio[i], 1) for i in range(n_show)]
    return go.Bar(
        x=pcs,
        y=vals,
        text=vals,
        textposition="outside",
        showlegend=False,
        name="Variance explained",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    )


def _combine_pca_and_variance(scatter: go.Figure, var_ratio, title: dict | str) -> go.Figure:
    """Attach a variance-explained bar panel under the 2D PCA scatter (same figure)."""
    title_dict = title if isinstance(title, dict) else _plotly_title(str(title))
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    is_3d = any(getattr(tr, "type", None) == "scatter3d" for tr in scatter.data)
    if is_3d:
        scatter.update_layout(title=title_dict)
        apply_export_layout(scatter, title_lines=title_lines, legend=True, uirevision="pca-scatter")
        return scatter

    # Compact PCA on top (inset-like), scree underneath — same outer width as other figs
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.14,
        subplot_titles=("", "Variance explained"),
    )
    for tr in scatter.data:
        fig.add_trace(tr, row=1, col=1)
    fig.add_trace(_scree_trace(var_ratio, top_n=10), row=2, col=1)
    vals = [round(100 * float(v), 1) for v in list(var_ratio)[:10]]
    ymax = max(vals) * 1.22 if vals else 1.0

    # Copy axis titles from scatter
    xlab = scatter.layout.xaxis.title.text if scatter.layout.xaxis.title else ""
    ylab = scatter.layout.yaxis.title.text if scatter.layout.yaxis.title else ""
    fig.update_xaxes(title_text=xlab or None, row=1, col=1)
    fig.update_yaxes(title_text=ylab or None, row=1, col=1)
    # Preserve equal aspect from the stand-alone scatter when available
    if scatter.layout.xaxis and scatter.layout.xaxis.range:
        fig.update_xaxes(range=list(scatter.layout.xaxis.range), row=1, col=1)
    if scatter.layout.yaxis and scatter.layout.yaxis.range:
        fig.update_yaxes(range=list(scatter.layout.yaxis.range), scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="PC", row=2, col=1)
    fig.update_yaxes(
        title_text="Variance explained [%]",
        range=[0, ymax],
        row=2,
        col=1,
    )
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))

    fig.update_layout(title=title_dict)
    leg_title = None
    if scatter.layout.legend and scatter.layout.legend.title:
        leg_title = scatter.layout.legend.title.text
    apply_export_layout(
        fig,
        title_lines=title_lines,
        width=EXPORT_W,
        height=EXPORT_PCA_SCREE_H,
        legend=True,
        legend_kwargs={"title_text": leg_title} if leg_title else None,
        uirevision="pca-scatter",
    )
    return fig


def _pc_options(n_pcs: int) -> list[dict]:
    n = max(0, int(n_pcs))
    return [{"label": f"PC{i}", "value": f"PC{i}"} for i in range(1, n + 1)]


def _pc_index(pc: str | None) -> int | None:
    if not pc or not str(pc).startswith("PC"):
        return None
    tail = str(pc)[2:]
    if not tail.isdigit():
        return None
    return int(tail) - 1


def _axis_label(pc: str | None, var) -> str:
    if not pc:
        return ""
    var = list(var) if var is not None else []
    idx = _pc_index(pc)
    if idx is not None and 0 <= idx < len(var):
        return f"{pc} ({100 * var[idx]:.1f}%)"
    return str(pc)


def _scatter_single(
    score_df,
    x_col,
    y_col,
    z_col,
    color,
    shape,
    size,
    alpha,
    var,
    title="PCA",
) -> go.Figure:
    score_df = score_df.copy()
    aes = _resolve_aes(color, shape, size, alpha, score_df.columns)
    xlab = _axis_label(x_col, var)
    ylab = _axis_label(y_col, var)
    zcol = z_col if z_col and z_col in score_df.columns else None
    fig = build_scatter(
        score_df,
        x=x_col,
        y=y_col,
        z=zcol,
        legend_name="samples",
        title=title,
        **aes,
    )
    layout_kw = dict(uirevision="pca-scatter", showlegend=True)
    if zcol:
        fig.update_layout(
            scene=dict(
                xaxis_title=xlab,
                yaxis_title=ylab,
                zaxis_title=_axis_label(zcol, var),
            ),
            **layout_kw,
        )
        apply_export_layout(fig, title_lines=1, legend=True, uirevision="pca-scatter")
    else:
        fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, **layout_kw)
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
        apply_export_layout(fig, title_lines=1, legend=True, uirevision="pca-scatter")
    return fig


def _ensure_pca_lookup(locus_path: str | None):
    """Load locus lookup once per path into ``_PCA_RUNTIME``."""
    path = (locus_path or "").strip()
    if _PCA_RUNTIME.get("locus_lookup") is not None and _PCA_RUNTIME.get("locus_path") == path:
        return _PCA_RUNTIME["locus_lookup"]
    if not path:
        _PCA_RUNTIME["locus_lookup"] = None
        _PCA_RUNTIME["locus_path"] = ""
        return None
    try:
        from src.biocyc.celov_multiomics_post import load_locus_lookup

        lookup = load_locus_lookup(path)
    except Exception:  # noqa: BLE001
        _PCA_RUNTIME["locus_lookup"] = None
        _PCA_RUNTIME["locus_path"] = path
        return None
    _PCA_RUNTIME["locus_lookup"] = lookup
    _PCA_RUNTIME["locus_path"] = path
    return lookup


def _gene_select_by_options(lookup) -> list[dict]:
    opts = [{"label": "gene ID", "value": _GENE_BY_ID}]
    if lookup is None or not isinstance(lookup, pd.DataFrame) or lookup.empty:
        return opts
    skip = {"locustag", "geneid"}
    for col in lookup.columns:
        key = str(col).lower().replace(" ", "")
        if key in skip:
            continue
        if "name" in key or "symbol" in key:
            opts.append({"label": str(col), "value": str(col)})
    return opts


_LOOKUP_ID_COLS = ("locusTag", "geneID", "biocyc_id", "old locusTag", "old_locusTag")


def _name_column(lookup, by_col: str | None) -> str | None:
    if lookup is None or not isinstance(lookup, pd.DataFrame) or lookup.empty:
        return None
    if by_col and by_col != _GENE_BY_ID and by_col in lookup.columns:
        return by_col
    for opt in _gene_select_by_options(lookup):
        if opt["value"] != _GENE_BY_ID and opt["value"] in lookup.columns:
            return opt["value"]
    return None


def _gene_name_map(lookup, by_col: str | None) -> dict[str, str]:
    name_col = _name_column(lookup, by_col)
    if lookup is None or not name_col:
        return {}
    id_cols = [c for c in _LOOKUP_ID_COLS if c in lookup.columns]
    if not id_cols:
        return {}
    out: dict[str, str] = {}
    for raw, *ids in zip(lookup[name_col], *(lookup[c] for c in id_cols)):
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        text = str(raw).strip()
        if not text or text.lower() == "nan":
            continue
        for gid in ids:
            if gid is None or (isinstance(gid, float) and pd.isna(gid)):
                continue
            key = str(gid).strip()
            if key and key.lower() != "nan" and key not in out:
                out[key] = text
    return out


def _gene_label(gid: str, names: dict[str, str], by_col: str | None, weight: float | None = None) -> str:
    name = names.get(gid, "")
    if by_col and by_col != _GENE_BY_ID and name:
        label = f"{name} ({gid})"
    else:
        label = gid
    if weight is not None and np.isfinite(weight):
        label = f"{label}  {weight:+.3g}"
    return label


def _gene_dropdown_options(
    gene_ids,
    lookup,
    by_col: str | None,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    names = _gene_name_map(lookup, by_col)
    opts = []
    for gid in gene_ids:
        gid = str(gid)
        w = None if weights is None else weights.get(gid)
        opts.append({"label": _gene_label(gid, names, by_col, w), "value": gid})
    return opts


def _top_weighed(weighed: pd.DataFrame, n, side: str | None) -> pd.DataFrame:
    w = weighed.copy()
    if side == "up":
        w = w[w["gene_weight"] > 0]
    elif side == "down":
        w = w[w["gene_weight"] < 0]
    try:
        n_keep = max(1, int(n))
    except (TypeError, ValueError):
        n_keep = 20
    return w.head(n_keep)


def _gene_expr_picker(prefix: str, *, from_weights: bool) -> html.Div:
    """Select-by + gene dropdown (+ top-N weights) and viridis PCA plot."""
    blurb = (
        "Pick genes from the top weights (gene ID or name). Shown: expression on "
        "the current PC axes, one panel per gene."
        if from_weights
        else "Pick genes by gene ID or name. Shown: expression on the current PC "
        "axes, one panel per gene."
    )
    row = [
        dbc.Col(
            [
                html.Label("Select by"),
                dcc.Dropdown(id=f"{prefix}-by", value=_GENE_BY_ID, clearable=False),
            ],
            md=3,
        ),
    ]
    if from_weights:
        row += [
            dbc.Col(
                [
                    html.Label("Top N"),
                    dbc.Input(id=f"{prefix}-topn", type="number", value=20, min=1, step=1),
                ],
                md=2,
            ),
            dbc.Col(
                [
                    html.Label("Weights"),
                    dcc.RadioItems(
                        id=f"{prefix}-side",
                        options=[
                            {"label": "|weight|", "value": "abs"},
                            {"label": "up", "value": "up"},
                            {"label": "down", "value": "down"},
                        ],
                        value="abs",
                        inline=True,
                    ),
                ],
                md=4,
            ),
        ]
    return html.Div(
        [
            html.H6("Gene expression", className="mt-3"),
            html.P(blurb, className="text-muted small"),
            dbc.Row(row, className="g-2 mb-2"),
            html.Label("Genes"),
            dcc.Dropdown(
                id=f"{prefix}-genes",
                multi=True,
                placeholder="Type 2+ characters to search genes…",
                className="mb-2",
            ),
            fig_size_controls(prefix, default_width=EXPORT_W, default_height=480),
            plot_with_sample_detail(
                f"{prefix}-fig",
                f"{prefix}-sample-detail",
                graph_config=_GRAPH_CONFIG,
            ),
            html.Div(id=f"{prefix}-status", className="text-muted small mb-2"),
            html.H6("Condition enrichment on gene profile", className="mt-3"),
            html.P(
                "Uses the selected genes' raw expression as the profile. "
                "Rankable: Pearson / Spearman vs that profile. Categorical: "
                "within-entry variance and distance on that profile.",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Columns"),
                            dcc.RadioItems(
                                id=f"{prefix}-enr-kind",
                                options=[
                                    {"label": " categorical", "value": "cat"},
                                    {"label": " rankable", "value": "rank"},
                                ],
                                value="cat",
                                inline=True,
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Metadata columns"),
                            dcc.Dropdown(id=f"{prefix}-enr-cols", multi=True),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Axis label"),
                            dcc.Dropdown(id=f"{prefix}-enr-label", clearable=True),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Run enrichment",
                            id=f"{prefix}-enr-run",
                            color="secondary",
                            className="mt-4",
                        ),
                        md=2,
                    ),
                ],
                className="g-2 mb-2",
            ),
            dcc.Graph(id=f"{prefix}-enr-fig", figure={}, config=_GRAPH_CONFIG),
            html.Div(id=f"{prefix}-enr-status", className="text-muted small mb-2"),
        ]
    )


def _live_expression() -> pd.DataFrame | None:
    session = live_session()
    if session is None or session.expression is None:
        return None
    expr = session.expression
    if any(not isinstance(c, str) for c in expr.columns):
        expr = expr.copy()
        expr.columns = expr.columns.astype(str)
    return expr


def _render_gene_expr_plot(genes, lookup, by_col, x_col, y_col, cache, fig_w, fig_h):
    empty = go.Figure()
    expr = _live_expression()
    if not cache or expr is None:
        return empty, "Run PCA first."
    score_df = _score_frame_for_plot(cache)
    if score_df is None:
        return empty, "Run PCA first."
    genes = [str(g) for g in (genes or [])]
    if not genes:
        return empty, "Select one or more genes."
    missing = [g for g in genes if g not in expr.columns]
    if missing:
        return empty, f"Unknown gene ID: {', '.join(missing)}"
    var = cache.get("var_ratio") or _PCA_RUNTIME.get("var_ratio", [])
    names = _gene_name_map(lookup, by_col)
    titles = [names.get(g) or g for g in genes]
    try:
        fig = _gene_expression_figure(
            score_df, expr, genes, x_col or "PC1", y_col or "PC2", var, titles=titles
        )
        n = max(1, len(genes))
        h = int(fig_h) if fig_h else 480
        if n > 1:
            h = max(h, 400 * n)
        fig = set_fig_size(fig, fig_w, h, default_height=max(480, 400 * n))
        return fig, f"{n} gene{'s' if n != 1 else ''} on {x_col or 'PC1'} vs {y_col or 'PC2'}."
    except Exception as exc:  # noqa: BLE001
        err = go.Figure()
        err.add_annotation(text=f"Plot error: {exc}", showarrow=False)
        return err, f"Plot error: {exc}"


def _gene_expression_figure(score_df, expr, genes, x_col, y_col, var, titles=None) -> go.Figure:
    """PCA scatter colored by each selected gene (viridis, own min–max)."""
    genes = [str(g) for g in (genes or []) if str(g) in expr.columns]
    empty = go.Figure()
    if not genes:
        empty.add_annotation(text="Select genes.", showarrow=False)
        return empty
    if x_col not in score_df.columns or y_col not in score_df.columns:
        empty.add_annotation(text="Run PCA and pick PC axes.", showarrow=False)
        return empty
    if titles is None:
        titles = genes
    else:
        titles = [str(t) for t in titles]

    n = len(genes)
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.12 if n > 1 else 0.08,
    )
    xlab = _axis_label(x_col, var)
    ylab = _axis_label(y_col, var)
    x = pd.to_numeric(score_df[x_col], errors="coerce")
    y = pd.to_numeric(score_df[y_col], errors="coerce")
    hover = score_df.index.astype(str)
    for i, gene in enumerate(genes, start=1):
        title = titles[i - 1] if i - 1 < len(titles) else gene
        vals = pd.to_numeric(expr[gene].reindex(score_df.index), errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    color=vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=title, side="right"),
                        len=max(0.2, 0.85 / n),
                        y=1 - (i - 0.5) / n,
                        yanchor="middle",
                        thickness=14,
                    ),
                    cmin=float(np.nanmin(vals.to_numpy(dtype=float))) if vals.notna().any() else 0,
                    cmax=float(np.nanmax(vals.to_numpy(dtype=float))) if vals.notna().any() else 1,
                ),
                customdata=np.stack([hover, vals.to_numpy(dtype=float)], axis=1),
                hovertemplate=(
                    "%{customdata[0]}<br>"
                    + title
                    + ": %{customdata[1]:.4g}<extra></extra>"
                ),
                name=title,
                showlegend=False,
            ),
            row=i,
            col=1,
        )
        fig.update_xaxes(title_text=xlab, row=i, col=1)
        fig.update_yaxes(title_text=ylab, row=i, col=1)
    if n == 1:
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
    else:
        xmin, xmax = float(x.min()), float(x.max())
        ymin, ymax = float(y.min()), float(y.max())
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * max(xmax - xmin, ymax - ymin, 1e-9)
        half += 0.05 * half
        fig.update_xaxes(range=[cx - half, cx + half], constrain="domain")
        fig.update_yaxes(range=[cy - half, cy + half], constrain="domain")
    apply_export_layout(fig, title_lines=1, legend=False, uirevision="pca-expr")
    return fig


def _gene_profile_enrich_fig(genes, kind, cols, label_col):
    empty = go.Figure()
    if "score_df" not in _PCA_RUNTIME:
        return empty, "Run PCA first."
    expr = _live_expression()
    if expr is None:
        return empty, "Load a dataset first."
    genes = [str(g) for g in (genes or []) if str(g) in expr.columns]
    if not genes:
        return empty, "Select one or more genes."
    cols = [c for c in (cols or []) if c]
    if not cols:
        return empty, "Select metadata columns."
    df = _PCA_RUNTIME["score_df"].copy()
    axes = []
    for g in genes:
        name = g if g not in df.columns else f"{g} expr"
        df[name] = pd.to_numeric(expr[g].reindex(df.index), errors="coerce")
        axes.append(name)
    use = [c for c in cols if c in df.columns]
    if not use:
        return empty, "Selected columns are not in metadata."
    label = label_col if label_col in df.columns else None
    names = ", ".join(axes)
    if kind == "rank":
        mats = corr_vs_axes(df, use, axes)
        fig = heatmap_matrix_fig(
            [mats["Pearson"], mats["Spearman"]],
            ["Pearson", "Spearman"],
            title=_plotly_title("Rankable metadata vs gene expression", names),
            zmin=-1,
            zmax=1,
            colorscale="RdBu_r",
        )
        return fig, f"{len(use)} rankable columns vs {len(axes)} gene profile(s)."
    var_mat, dist_mat = cat_entry_mats(df, use, axes, label_col=label)
    fig = heatmap_matrix_fig(
        [var_mat, dist_mat],
        [
            "var_ratio (within / total) — small = clusters",
            "mean |distance| to label / √within-var",
        ],
        title=_plotly_title("Categorical entries vs gene expression", names),
        zmin=0,
    )
    return fig, f"{len(use)} categorical columns vs {len(axes)} gene profile(s)."


def _scatter_split(score_df, x_col, y_col, z_col, split_col, group_aes: dict, var) -> go.Figure:
    xlab = _axis_label(x_col, var)
    ylab = _axis_label(y_col, var)
    zcol = z_col if z_col and z_col in score_df.columns else None
    fig = go.Figure()
    levels = [str(v) for v in score_df[split_col].astype(str).unique()]
    for level in sorted(levels):
        sub = score_df.loc[score_df[split_col].astype(str) == level].copy()
        if sub.empty:
            continue
        raw = {**_DEFAULT_AES, **(group_aes or {}).get(level, {})}
        aes = _resolve_aes(
            raw.get("color"),
            raw.get("shape"),
            raw.get("size"),
            raw.get("alpha"),
            sub.columns,
        )
        part = build_scatter(
            sub,
            x=x_col,
            y=y_col,
            z=zcol,
            legend_name=level,
            title="",
            **aes,
        )
        for tr in part.data:
            name = tr.name
            if name in (None, "", "trace 0", "samples"):
                tr.name = level
            elif not str(name).startswith(level):
                tr.name = f"{level} | {name}"
            tr.legendgroup = level
            tr.showlegend = True
            fig.add_trace(tr)

    title_lines = 1
    layout_kw = dict(
        title=f"PCA (split by {split_col})",
        uirevision="pca-scatter",
        legend_title_text=split_col,
        showlegend=True,
    )
    if zcol:
        fig.update_layout(
            scene=dict(
                xaxis_title=xlab,
                yaxis_title=ylab,
                zaxis_title=_axis_label(zcol, var),
            ),
            **layout_kw,
        )
        apply_export_layout(
            fig,
            title_lines=title_lines,
            legend=True,
            legend_kwargs={"title_text": split_col},
            uirevision="pca-scatter",
        )
    else:
        fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, **layout_kw)
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
        apply_export_layout(
            fig,
            title_lines=title_lines,
            legend=True,
            legend_kwargs={"title_text": split_col},
            uirevision="pca-scatter",
        )
    return fig


def _score_frame_for_plot(cache: dict) -> pd.DataFrame | None:
    if not cache:
        return None
    key = cache.get("key")
    if key and _PCA_RUNTIME.get("key") and _PCA_RUNTIME.get("key") != key:
        return None
    df = _PCA_RUNTIME.get("score_df")
    return df if isinstance(df, pd.DataFrame) else None


class PCAModule:
    id = "pca"
    label = "PCA"

    def layout(self):
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("X"),
                                dcc.Dropdown(id="pca-x", placeholder="PC1", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Y"),
                                dcc.Dropdown(id="pca-y", placeholder="PC2", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Z (optional 3D)"),
                                dcc.Dropdown(id="pca-z", placeholder="None", clearable=True),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run PCA", id="pca-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.H6("Split by group"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Split column"),
                                dcc.Dropdown(
                                    id="pca-split-col",
                                    placeholder="None (single aesthetics)",
                                    clearable=True,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.P(
                    "Set color, shape, and size from metadata or a fixed value. "
                    "A split column gives each value its own settings.",
                    className="text-muted small mb-2",
                ),
                html.Div(id="pca-aes-panels"),
                fig_size_controls(
                    "pca",
                    default_width=EXPORT_W,
                    default_height=EXPORT_PCA_SCREE_H,
                ),
                dcc.Loading(
                    plot_with_sample_detail(
                        "pca-scatter",
                        "pca-sample-detail",
                        graph_config=_GRAPH_CONFIG,
                    ),
                    type="default",
                ),
                html.Hr(),
                dcc.Tabs(
                    id="pca-analysis-tabs",
                    value="ari",
                    children=[
                        dcc.Tab(
                            label="ARI of separation",
                            value="ari",
                            children=[
                html.H6("ARI of separation", className="mt-2"),
                html.P(
                    "Pick a label column and the positive class. Shown: how well each "
                    "PC separates that class.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Label column"),
                                dcc.Dropdown(id="pca-ari-label"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Positive class"),
                                dcc.Dropdown(id="pca-ari-positive"),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("n PCs"),
                                dbc.Input(id="pca-ari-n", type="number", value=10, min=1, step=1),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run ARI", id="pca-ari-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                fig_size_controls("pca-ari", default_width=EXPORT_W, default_height=360),
                dcc.Graph(id="pca-ari", figure=go.Figure(), config=_GRAPH_CONFIG),
                html.Div(id="pca-ari-status", className="text-muted small mb-2"),
                            ],
                        ),
                        dcc.Tab(
                            label="Weighed genes (PC, Celov)",
                            value="pc-celov",
                            children=[
                html.H6("Save weighed genes (PC, Celov)", className="mt-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("PC"),
                                dcc.Dropdown(id="pca-weight-pc", placeholder="PC1", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Locus lookup CSV (optional)"),
                                dbc.InputGroup(
                                    [
                                        dbc.Input(id="pca-pc-locus", type="text"),
                                        dbc.Button(
                                            "Browse…",
                                            id="pca-pc-locus-browse",
                                            color="info",
                                            outline=True,
                                        ),
                                    ]
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Output path ({} = type)"),
                                dbc.InputGroup(
                                    [
                                        dbc.Input(id="pca-weight-out", type="text"),
                                        dbc.Button(
                                            "Browse…",
                                            id="pca-weight-browse",
                                            color="info",
                                            outline=True,
                                        ),
                                    ]
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Genes"),
                                dcc.RadioItems(
                                    id="pca-weight-celov-mode",
                                    options=[
                                        {"label": "up & down", "value": "up_and_down"},
                                        {"label": "up", "value": "up"},
                                        {"label": "down", "value": "down"},
                                        {"label": "all together", "value": "all"},
                                    ],
                                    value="up_and_down",
                                    inline=True,
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button("Save Celov", id="pca-weight-save", color="secondary", className="mb-2"),
                _gene_expr_picker("pca-pc-expr", from_weights=True),
                            ],
                        ),
                        dcc.Tab(
                            label="Linear classifier",
                            value="clf",
                            children=[
                html.H6("Linear classifier", className="mt-2"),
                html.P(
                    "Pick a label column and the positive class. Shown: accuracy vs "
                    "number of PCs. Optionally draw the 2-PC boundary on the PCA plot. "
                    "Export weighed genes as Celov.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Label column"),
                                dcc.Dropdown(id="pca-clf-label"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Positive class"),
                                dcc.Dropdown(id="pca-clf-positive"),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("PCs min"),
                                dbc.Input(id="pca-clf-pc-min", type="number", value=2, min=2, step=1),
                            ],
                            md=1,
                        ),
                        dbc.Col(
                            [
                                html.Label("PCs max"),
                                dbc.Input(id="pca-clf-pc-max", type="number", value=12, min=2, step=1),
                            ],
                            md=1,
                        ),
                        dbc.Col(
                            [
                                html.Label("Genes: n PCs"),
                                dbc.Input(id="pca-clf-gene-pcs", type="number", value=5, min=2, step=1),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Checklist(
                                    id="pca-clf-overlay",
                                    options=[{"label": "Plot 2-PC boundary on PCA", "value": "on"}],
                                    value=[],
                                    inline=True,
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button("Run linear classifier", id="pca-clf-run", color="primary", className="mb-2"),
                fig_size_controls(
                    "pca-clf",
                    default_width=EXPORT_W,
                    default_height=EXPORT_H,
                ),
                dcc.Graph(
                    id="pca-clf-perf",
                    figure=go.Figure(),
                    config={
                        **_GRAPH_CONFIG,
                        "toImageButtonOptions": {
                            **_GRAPH_CONFIG["toImageButtonOptions"],
                            "filename": "pca_classifier_performance",
                        },
                    },
                ),
                html.Div(
                    id="pca-clf-celov-section",
                    style={"display": "none"},
                    children=[
                        html.H6("Save weighed genes (Celov)", className="mt-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label(
                                            "Locus lookup CSV (optional; Celov ID column is set on the dataset)"
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(id="pca-clf-locus", type="text"),
                                                dbc.Button(
                                                    "Browse…",
                                                    id="pca-clf-locus-browse",
                                                    color="info",
                                                    outline=True,
                                                ),
                                            ]
                                        ),
                                    ],
                                    md=5,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Output path ({} = type)"),
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(id="pca-clf-celov-out", type="text"),
                                                dbc.Button(
                                                    "Browse…",
                                                    id="pca-clf-celov-browse",
                                                    color="info",
                                                    outline=True,
                                                ),
                                            ]
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Genes"),
                                        dcc.RadioItems(
                                            id="pca-clf-celov-mode",
                                            options=[
                                                {"label": "up & down", "value": "up_and_down"},
                                                {"label": "up", "value": "up"},
                                                {"label": "down", "value": "down"},
                                                {"label": "all together", "value": "all"},
                                            ],
                                            value="up_and_down",
                                            inline=True,
                                        ),
                                    ],
                                    md=3,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                        dbc.Button(
                            "Save Celov",
                            id="pca-clf-celov-save",
                            color="secondary",
                            className="mb-2",
                        ),
                    ],
                ),
                html.Div(id="pca-clf-status", className="text-muted small mb-2"),
                _gene_expr_picker("pca-clf-expr", from_weights=True),
                            ],
                        ),
                        dcc.Tab(
                            label="Condition enrichment",
                            value="enrich",
                            children=[
                html.H6("Condition enrichment", className="mt-2"),
                html.P(
                    "Pick numeric and categorical metadata columns, and an optional "
                    "axis label. Shown: how those columns relate to the top PCs and "
                    "the biofilm axis.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Numeric / rank columns"),
                                dcc.Dropdown(id="pca-enr-num", multi=True),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Categorical columns"),
                                dcc.Dropdown(id="pca-enr-cat", multi=True),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Axis label column"),
                                dcc.Dropdown(id="pca-enr-label"),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("PCs in heatmaps"),
                                dbc.Input(id="pca-enr-show", type="number", value=2, min=1, step=1),
                            ],
                            md=1,
                        ),
                        dbc.Col(
                            [
                                html.Label("PCs for axis"),
                                dbc.Input(id="pca-enr-axis", type="number", value=7, min=1, step=1),
                            ],
                            md=1,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button("Run condition enrichment", id="pca-enr-run", color="primary", className="mb-2"),
                fig_size_controls("pca-enr-num", default_width=EXPORT_W, default_height=720),
                dcc.Graph(id="pca-enr-num-fig", figure=go.Figure(), config=_GRAPH_CONFIG),
                fig_size_controls("pca-enr-cat", default_width=EXPORT_W, default_height=1100),
                dcc.Graph(id="pca-enr-cat-fig", figure=go.Figure(), config=_GRAPH_CONFIG),
                html.Div(id="pca-enr-status", className="text-muted small mb-2"),
                            ],
                        ),
                        dcc.Tab(
                            label="Gene expression",
                            value="gene-expr",
                            children=[
                _gene_expr_picker("pca-expr", from_weights=False),
                            ],
                        ),
                    ],
                ),
                html.Div(id="pca-status", className="text-muted small"),
                dcc.Store(id="pca-cache"),
                dcc.Store(id="pca-group-aes", data={}),
                dcc.Store(id="pca-split-options", data=[]),
                dcc.Store(id="pca-clf-cache", data=None),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output("pca-split-col", "options"),
            Output("pca-split-options", "data"),
            Input("session-store", "data"),
            Input("pca-cache", "data"),
        )
        def _fill_split(session_blob, cache):
            cols = meta_columns_from_store(session_blob)
            split_opts = [{"label": c, "value": c} for c in cols]
            return split_opts, cols

        @app.callback(
            Output("pca-aes-panels", "children"),
            Input("pca-split-col", "value"),
            Input("pca-cache", "data"),
            Input("pca-split-options", "data"),
            State("pca-group-aes", "data"),
        )
        def _build_aes_panels(split_col, cache, meta_cols, group_aes):
            meta_cols = list(meta_cols or [])
            color_opts, shape_opts, size_opts = aesthetic_options(meta_cols)
            group_aes = group_aes or {}
            panels: list = []

            if split_col and cache:
                levels = meta_levels(split_col)
                if levels:
                    for level in levels:
                        panels.append(
                            aesthetic_panel(
                                _AES_PREFIX,
                                level,
                                heading=level,
                                color_opts=color_opts,
                                shape_opts=shape_opts,
                                size_opts=size_opts,
                                values={**_DEFAULT_AES, **group_aes.get(level, {})},
                            )
                        )
                    return panels

            panels.append(
                aesthetic_panel(
                    _AES_PREFIX,
                    _AES_ALL,
                    heading="Aesthetics",
                    color_opts=color_opts,
                    shape_opts=shape_opts,
                    size_opts=size_opts,
                    values={**_DEFAULT_AES, **group_aes.get(_AES_ALL, {})},
                )
            )
            return panels

        @app.callback(
            Output("pca-group-aes", "data"),
            Input({"type": f"{_AES_PREFIX}-color", "group": ALL}, "value"),
            Input({"type": f"{_AES_PREFIX}-shape", "group": ALL}, "value"),
            Input({"type": f"{_AES_PREFIX}-size", "group": ALL}, "value"),
            Input({"type": f"{_AES_PREFIX}-alpha", "group": ALL}, "value"),
            State({"type": f"{_AES_PREFIX}-color", "group": ALL}, "id"),
            prevent_initial_call=True,
        )
        def _sync_group_aes(colors, shapes, sizes, alphas, ids):
            if not ids:
                return {}
            data = {}
            for i, id_dict in enumerate(ids):
                g = str(id_dict["group"])
                data[g] = {
                    "color": colors[i] if i < len(colors) else None,
                    "shape": shapes[i] if i < len(shapes) else None,
                    "size": sizes[i] if i < len(sizes) else None,
                    "alpha": float(alphas[i]) if i < len(alphas) and alphas[i] is not None else 0.85,
                }
            return data

        @app.callback(
            Output("pca-cache", "data"),
            Output("pca-status", "children"),
            Output("pca-clf-cache", "data", allow_duplicate=True),
            Output("pca-x", "options"),
            Output("pca-y", "options"),
            Output("pca-z", "options"),
            Output("pca-weight-pc", "options"),
            Output("pca-x", "value"),
            Output("pca-y", "value"),
            Output("pca-z", "value"),
            Output("pca-weight-pc", "value"),
            Input("pca-run", "n_clicks"),
            State("session-store", "data"),
            prevent_initial_call=True,
        )
        def _compute(n_clicks, session_blob):
            session = session_from_store(session_blob)
            if not session.ready:
                return (
                    no_update,
                    session.error or "Load datasets first.",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            try:
                score_df, var_ratio, explained_var, loadings = _run_pca(session)
                _PCA_RUNTIME.clear()
                _PCA_RUNTIME.update(
                    {
                        "score_df": score_df,
                        "loadings": loadings,
                        "explained_variance": np.asarray(explained_var, dtype=float),
                        "var_ratio": list(map(float, var_ratio)),
                        "key": "|".join(session.active_datasets),
                    }
                )
                n_pcs = int(len(var_ratio))
                opts = _pc_options(n_pcs)
                cache = {
                    "ready": True,
                    "n_pcs": n_pcs,
                    "key": "|".join(session.active_datasets),
                }
                return (
                    cache,
                    f"PCA done: {score_df.shape[0]} samples, {n_pcs} components.",
                    None,
                    opts,
                    opts,
                    opts,
                    opts,
                    "PC1" if n_pcs >= 1 else None,
                    "PC2" if n_pcs >= 2 else ("PC1" if n_pcs >= 1 else None),
                    None,
                    "PC1" if n_pcs >= 1 else None,
                )
            except Exception as exc:  # noqa: BLE001
                return (
                    no_update,
                    f"PCA error: {exc}",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )

        @app.callback(
            Output("pca-clf-label", "options"),
            Output("pca-ari-label", "options"),
            Output("pca-ari-label", "value"),
            Output("pca-enr-label", "options"),
            Output("pca-enr-label", "value"),
            Output("pca-enr-num", "options"),
            Output("pca-enr-num", "value"),
            Output("pca-enr-cat", "options"),
            Output("pca-enr-cat", "value"),
            Input("session-store", "data"),
            Input("pca-cache", "data"),
            State("pca-ari-label", "value"),
            State("pca-enr-label", "value"),
            State("pca-enr-num", "value"),
            State("pca-enr-cat", "value"),
        )
        def _fill_clf_label(session_blob, cache, ari_cur, enr_lab, num_cur, cat_cur):
            cols = meta_columns_from_store(session_blob)
            opts = [{"label": c, "value": c} for c in cols]
            ari_val = ari_cur if ari_cur in cols else ("Biofilm" if "Biofilm" in cols else (cols[0] if cols else None))
            enr_val = enr_lab if enr_lab in cols else ("Biofilm" if "Biofilm" in cols else (cols[0] if cols else None))
            num_pref = present_cols(cols, DEFAULT_NUM_COLS) or cols
            cat_pref = present_cols(cols, DEFAULT_CAT_COLS) or cols
            num_val = [c for c in (num_cur or num_pref) if c in cols]
            cat_val = [c for c in (cat_cur or cat_pref) if c in cols]
            return opts, opts, ari_val, opts, enr_val, opts, num_val, opts, cat_val

        @app.callback(
            Output("pca-clf-positive", "options"),
            Output("pca-clf-positive", "value"),
            Input("pca-clf-label", "value"),
            State("pca-cache", "data"),
            State("pca-clf-positive", "value"),
        )
        def _fill_positive(label_col, cache, current):
            if not label_col or not cache:
                return [], None
            levels = meta_levels(label_col)
            if not levels:
                return [], None
            opts = [{"label": v, "value": v} for v in levels]
            value = current if current in levels else (levels[-1] if levels else None)
            return opts, value

        @app.callback(
            Output("pca-ari-positive", "options"),
            Output("pca-ari-positive", "value"),
            Input("pca-ari-label", "value"),
            State("pca-cache", "data"),
            State("pca-ari-positive", "value"),
        )
        def _fill_ari_positive(label_col, cache, current):
            if not label_col or not cache:
                return [], None
            levels = meta_levels(label_col)
            if not levels:
                return [], None
            opts = [{"label": v, "value": v} for v in levels]
            prefer = next((v for v in ("True", "true", "1") if v in levels), None)
            value = current if current in levels else (prefer or (levels[-1] if levels else None))
            return opts, value

        @app.callback(
            Output("pca-clf-celov-section", "style"),
            Input("pca-clf-cache", "data"),
        )
        def _toggle_celov(clf_cache):
            if clf_cache:
                return {"display": "block"}
            return {"display": "none"}

        @app.callback(
            Output("pca-clf-cache", "data", allow_duplicate=True),
            Output("pca-clf-perf", "figure"),
            Output("pca-clf-status", "children"),
            Input("pca-clf-run", "n_clicks"),
            Input("pca-clf-fig-w", "value"),
            Input("pca-clf-fig-h", "value"),
            State("pca-cache", "data"),
            State("pca-clf-label", "value"),
            State("pca-clf-positive", "value"),
            State("pca-clf-pc-min", "value"),
            State("pca-clf-pc-max", "value"),
            State("pca-clf-gene-pcs", "value"),
            State("pca-clf-cache", "data"),
            State("session-store", "data"),
            State("ds-active", "value"),
            prevent_initial_call=True,
        )
        def _run_classifier(
            n_clicks,
            fig_w,
            fig_h,
            cache,
            label_col,
            positive,
            pc_min,
            pc_max,
            gene_pcs,
            clf_cache,
            session_blob,
            active,
        ):
            empty = go.Figure()
            triggered = callback_context.triggered_id
            dataset = _active_dataset_name(session_blob, active)

            if triggered in ("pca-clf-fig-w", "pca-clf-fig-h"):
                if not clf_cache or not clf_cache.get("perf"):
                    return no_update, no_update, no_update
                perf = pd.DataFrame(clf_cache["perf"])
                fig = performance_figure(
                    perf,
                    title=_plotly_title(
                        f"Linear Classifier on PCs for {dataset}",
                        f"split on {clf_cache.get('label_col')} "
                        f"(positive = {clf_cache.get('positive')})",
                    ),
                )
                return no_update, set_fig_size(fig, fig_w, fig_h), no_update

            if not cache or "score_df" not in _PCA_RUNTIME:
                return no_update, empty, "Run PCA first."
            if not label_col or positive is None or positive == "":
                return no_update, empty, "Choose label column and positive class."
            try:
                score_df = _PCA_RUNTIME["score_df"]
                y = encode_binary_labels(score_df[label_col], positive)
                perf = classifier_performance(score_df, y, int(pc_min or 2), int(pc_max or 12))
                fig = performance_figure(
                    perf,
                    title=_plotly_title(
                        f"Linear Classifier on PCs for {dataset}",
                        f"split on {label_col} (positive = {positive})",
                    ),
                )
                fig = set_fig_size(fig, fig_w, fig_h)
                res2 = fit_pc_classifier(score_df, y, 2)
                n_gene = int(gene_pcs or 5)
                res_g = fit_pc_classifier(score_df, y, n_gene)
                new_cache = {
                    "label_col": label_col,
                    "positive": str(positive),
                    "perf": perf.to_dict(orient="list") if perf is not None else {},
                    "w2": list(map(float, res2["w"])) if res2 else None,
                    "b2": float(res2["b"]) if res2 else None,
                    "cv2": float(res2["cv_acc"]) if res2 else None,
                    "gene_pcs": n_gene,
                    "w_gene": list(map(float, res_g["w"])) if res_g else None,
                    "train_gene": float(res_g["train_acc"]) if res_g else None,
                    "cv_gene": float(res_g["cv_acc"]) if res_g else None,
                }
                msg = (
                    f"Classifier done. 2-PC CV={new_cache['cv2']:.3f}; "
                    f"{n_gene}-PC train={new_cache['train_gene']:.3f}, CV={new_cache['cv_gene']:.3f}."
                    if res2 and res_g
                    else "Classifier finished with missing fits (check class sizes / n_pcs)."
                )
                return new_cache, fig, msg
            except Exception as exc:  # noqa: BLE001
                return no_update, empty, f"Classifier error: {exc}"

        @app.callback(
            Output("pca-ari", "figure"),
            Output("pca-ari-status", "children"),
            Input("pca-ari-run", "n_clicks"),
            Input("pca-ari-fig-w", "value"),
            Input("pca-ari-fig-h", "value"),
            State("pca-ari-label", "value"),
            State("pca-ari-positive", "value"),
            State("pca-ari-n", "value"),
            State("session-store", "data"),
            State("ds-active", "value"),
            prevent_initial_call=True,
        )
        def _run_ari(n_clicks, fig_w, fig_h, label_col, positive, n_pcs, session_blob, active):
            empty = go.Figure()
            if "score_df" not in _PCA_RUNTIME:
                return empty, "Run PCA first."
            if not label_col or positive is None or positive == "":
                return empty, "Choose label column and positive class."
            try:
                score_df = _PCA_RUNTIME["score_df"]
                n_use = max(1, int(n_pcs or 10))
                sep = pc_separation_ari(score_df, label_col, positive, n_use)
                dataset = _active_dataset_name(session_blob, active)
                fig = ari_figure(
                    sep,
                    title=_plotly_title(
                        f"ARI of {label_col}={positive} on each PC",
                        dataset,
                    ),
                )
                return set_fig_size(fig, fig_w, fig_h, default_height=360), (
                    f"ARI on first {len(sep)} PCs; max={sep['ARI'].max():.3f} "
                    f"at PC{int(sep.loc[sep['ARI'].idxmax(), 'PC'])}."
                )
            except Exception as exc:  # noqa: BLE001
                return empty, f"ARI error: {exc}"

        @app.callback(
            Output("pca-enr-num-fig", "figure"),
            Output("pca-enr-cat-fig", "figure"),
            Output("pca-enr-status", "children"),
            Input("pca-enr-run", "n_clicks"),
            Input("pca-enr-num-fig-w", "value"),
            Input("pca-enr-num-fig-h", "value"),
            Input("pca-enr-cat-fig-w", "value"),
            Input("pca-enr-cat-fig-h", "value"),
            State("pca-enr-num", "value"),
            State("pca-enr-cat", "value"),
            State("pca-enr-label", "value"),
            State("pca-enr-show", "value"),
            State("pca-enr-axis", "value"),
            State("session-store", "data"),
            State("ds-active", "value"),
            prevent_initial_call=True,
        )
        def _run_pca_enrich(
            n_clicks,
            num_w,
            num_h,
            cat_w,
            cat_h,
            num_cols,
            cat_cols,
            label_col,
            n_show,
            n_axis,
            session_blob,
            active,
        ):
            empty = go.Figure()
            if "score_df" not in _PCA_RUNTIME:
                return empty, empty, "Run PCA first."
            try:
                score_df = _PCA_RUNTIME["score_df"].copy()
                n_show = max(1, int(n_show or 2))
                n_axis = max(1, int(n_axis or 7))
                num_cols = list(num_cols or [])
                cat_cols = list(cat_cols or [])
                if not num_cols and not cat_cols:
                    return empty, empty, "Select numeric and/or categorical columns."
                axes = [f"PC{i}" for i in range(1, n_show + 1) if f"PC{i}" in score_df.columns]
                if label_col and label_col in score_df.columns:
                    axis, _w = biofilm_axis_scores(score_df, label_col, n_axis)
                    score_df["Biofilm axis"] = axis
                    axes = axes + ["Biofilm axis"]
                dataset = _active_dataset_name(session_blob, active)
                num_fig = empty
                if num_cols:
                    mats = corr_vs_axes(score_df, num_cols, axes)
                    num_fig = heatmap_matrix_fig(
                        [mats["Pearson"], mats["Spearman"]],
                        ["Pearson", "Spearman"],
                        title=_plotly_title(
                            "Correlation of rankable metadata with PCs / biofilm axis",
                            f"{dataset} (axis from {n_axis} PCs)",
                        ),
                        zmin=-1,
                        zmax=1,
                    )
                    needed = int(num_fig.layout.height or 720)
                    num_fig = set_fig_size(
                        num_fig,
                        num_w,
                        max(num_h or 0, needed),
                        default_height=needed,
                    )
                cat_fig = empty
                if cat_cols:
                    var_mat, dist_mat = cat_entry_mats(score_df, cat_cols, axes, label_col=label_col)
                    cat_fig = heatmap_matrix_fig(
                        [var_mat, dist_mat],
                        [
                            "var_ratio (within / total) — small = clusters",
                            "mean |distance| to label / √within-var",
                        ],
                        title=_plotly_title("Categorical entries vs PCs / biofilm axis", dataset),
                        zmin=0,
                    )
                    needed = int(cat_fig.layout.height or 1100)
                    cat_fig = set_fig_size(
                        cat_fig,
                        cat_w,
                        max(cat_h or 0, needed),
                        default_height=needed,
                    )
                return num_fig, cat_fig, f"Condition enrichment on axes {axes}."
            except Exception as exc:  # noqa: BLE001
                return empty, empty, f"Enrichment error: {exc}"

        @app.callback(
            Output("pca-scatter", "figure"),
            Input("pca-cache", "data"),
            Input("pca-x", "value"),
            Input("pca-y", "value"),
            Input("pca-z", "value"),
            Input("pca-split-col", "value"),
            Input("pca-group-aes", "data"),
            Input("pca-clf-cache", "data"),
            Input("pca-clf-overlay", "value"),
            Input("pca-fig-w", "value"),
            Input("pca-fig-h", "value"),
            Input("session-store", "data"),
            Input("ds-active", "value"),
        )
        def _replot(
            cache,
            x_col,
            y_col,
            z_col,
            split_col,
            group_aes,
            clf_cache,
            overlay,
            fig_w,
            fig_h,
            session_blob,
            active,
        ):
            empty = go.Figure()
            if not cache:
                return empty
            score_df = _score_frame_for_plot(cache)
            var = cache.get("var_ratio") or _PCA_RUNTIME.get("var_ratio", [])
            if score_df is None:
                return empty
            x_col = x_col or "PC1"
            y_col = y_col or "PC2"
            if x_col not in score_df.columns or y_col not in score_df.columns:
                return empty
            group_aes = group_aes or {}
            dataset = _active_dataset_name(session_blob, active)
            if split_col and split_col in score_df.columns:
                title = _plotly_title(f"PCA for {dataset}", f"split by {split_col}")
            else:
                title = _plotly_title(f"PCA for {dataset}")
            try:
                if split_col and split_col in score_df.columns:
                    levels = [str(v) for v in score_df[split_col].astype(str).unique()]
                    merged = {lv: {**_DEFAULT_AES, **group_aes.get(lv, {})} for lv in levels}
                    scatter = _scatter_split(score_df, x_col, y_col, z_col, split_col, merged, var)
                else:
                    raw = {**_DEFAULT_AES, **group_aes.get(_AES_ALL, {})}
                    scatter = _scatter_single(
                        score_df,
                        x_col,
                        y_col,
                        z_col,
                        raw.get("color"),
                        raw.get("shape"),
                        raw.get("size"),
                        raw.get("alpha"),
                        var,
                    )
                if (
                    overlay
                    and "on" in (overlay or [])
                    and not z_col
                    and x_col == "PC1"
                    and y_col == "PC2"
                    and clf_cache
                    and clf_cache.get("w2") is not None
                ):
                    name = f"2-PC classifier (CV {clf_cache.get('cv2', float('nan')):.2f})"
                    scatter = add_decision_boundary(
                        scatter, score_df, clf_cache["w2"], clf_cache["b2"], name=name
                    )
                fig = _combine_pca_and_variance(scatter, var, title)
                return set_fig_size(
                    fig, fig_w, fig_h, default_height=EXPORT_PCA_SCREE_H
                )
            except Exception as exc:  # noqa: BLE001
                err = go.Figure()
                err.add_annotation(text=f"Plot error: {exc}", showarrow=False)
                return err

        @app.callback(
            Output("pca-pc-locus", "value"),
            Output("pca-clf-locus", "value"),
            Input("ds-active", "value"),
            Input("project-store", "data"),
        )
        def _sync_locus_from_active(active, blob):
            if not blob or not active:
                return "", ""
            name = active if isinstance(active, str) else (active[0] if active else None)
            entry = next(
                (d for d in blob.get("datasets", []) if d.get("name") == name),
                None,
            )
            if not entry:
                return "", ""
            locus = entry.get("locus_lookup") or ""
            root = blob.get("root")
            if locus and root and not Path(locus).is_absolute():
                locus = str(Path(root) / locus)
            return locus, locus

        @app.callback(
            Output("pca-pc-locus", "value", allow_duplicate=True),
            Output("pca-status", "children", allow_duplicate=True),
            Input("pca-pc-locus-browse", "n_clicks"),
            State("pca-pc-locus", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_pc_locus(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_file_dialog(initial=initial, title="Select locus lookup CSV")
            if not chosen:
                return no_update, "Locus browse cancelled."
            return chosen, f"Locus lookup: {chosen}"

        @app.callback(
            Output("pca-weight-out", "value"),
            Output("pca-status", "children", allow_duplicate=True),
            Input("pca-weight-browse", "n_clicks"),
            State("pca-weight-out", "value"),
            State("pca-weight-pc", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_weight(n_clicks, current, pc, project_blob):
            initial = (project_blob or {}).get("root") or current
            pc_name = pc or "PC1"
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save Celov weighed genes (use {} for type)",
                defaultextension=".txt",
                initialfile=f"PCA_{pc_name}_{{}}.txt",
            )
            if not chosen:
                return no_update, "Celov path browse cancelled."
            p = Path(chosen)
            if "{}" not in p.name:
                chosen = str(p.with_name(f"{p.stem}_{{}}{p.suffix or '.txt'}"))
            return chosen, f"Celov output template: {chosen}"

        @app.callback(
            Output("pca-status", "children", allow_duplicate=True),
            Input("pca-weight-save", "n_clicks"),
            State("pca-weight-pc", "value"),
            State("pca-weight-out", "value"),
            State("pca-weight-celov-mode", "value"),
            State("pca-pc-locus", "value"),
            State("project-store", "data"),
            State("ds-active", "value"),
            prevent_initial_call=True,
        )
        def _save_pc_celov(n_clicks, pc, out_path, mode, locus_path, project_blob, active):
            if "loadings" not in _PCA_RUNTIME:
                return "Run PCA first."
            if not pc:
                return "Choose a PC for weighed genes."
            if not out_path or not str(out_path).strip():
                return "Choose an output .txt path (Browse)."
            try:
                from src.biocyc.celov_multiomics_post import load_locus_lookup

                weighed = weighed_genes_from_pc(_PCA_RUNTIME["loadings"], str(pc))
                lookup = None
                if locus_path and str(locus_path).strip():
                    lookup = load_locus_lookup(str(locus_path).strip())
                name = active if isinstance(active, str) else (active[0] if active else None)
                entry = next(
                    (
                        d
                        for d in (project_blob or {}).get("datasets", [])
                        if d.get("name") == name
                    ),
                    None,
                )
                paths = save_classifier_celov(
                    weighed,
                    str(out_path).strip(),
                    mode=mode or "up_and_down",
                    locus_lookup=lookup,
                    id_column=resolve_celov_id_col(entry),
                )
                return f"Saved Celov ({pc}): " + ", ".join(str(p) for p in paths)
            except Exception as exc:  # noqa: BLE001
                return f"Celov save error: {exc}"

        @app.callback(
            Output("pca-clf-locus", "value", allow_duplicate=True),
            Output("pca-clf-status", "children", allow_duplicate=True),
            Input("pca-clf-locus-browse", "n_clicks"),
            State("pca-clf-locus", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_locus(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_file_dialog(initial=initial, title="Select locus lookup CSV")
            if not chosen:
                return no_update, "Locus browse cancelled."
            return chosen, f"Locus lookup: {chosen}"

        @app.callback(
            Output("pca-clf-celov-out", "value"),
            Output("pca-clf-status", "children", allow_duplicate=True),
            Input("pca-clf-celov-browse", "n_clicks"),
            State("pca-clf-celov-out", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_celov(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save Celov weighed genes (use {} for type)",
                defaultextension=".txt",
                initialfile="PCA_LinearClass_{}.txt",
            )
            if not chosen:
                return no_update, "Celov path browse cancelled."
            p = Path(chosen)
            if "{}" not in p.name:
                chosen = str(p.with_name(f"{p.stem}_{{}}{p.suffix or '.txt'}"))
            return chosen, f"Celov output template: {chosen}"

        @app.callback(
            Output("pca-clf-status", "children", allow_duplicate=True),
            Input("pca-clf-celov-save", "n_clicks"),
            State("pca-clf-cache", "data"),
            State("pca-clf-celov-out", "value"),
            State("pca-clf-celov-mode", "value"),
            State("pca-clf-locus", "value"),
            State("pca-clf-gene-pcs", "value"),
            State("project-store", "data"),
            State("ds-active", "value"),
            prevent_initial_call=True,
        )
        def _save_celov(
            n_clicks, clf_cache, out_path, mode, locus_path, gene_pcs, project_blob, active
        ):
            if not clf_cache or not clf_cache.get("w_gene"):
                return "Run the linear classifier first."
            if not out_path or not str(out_path).strip():
                return "Choose an output .txt path (Browse)."
            if "loadings" not in _PCA_RUNTIME:
                return "Run PCA first."
            try:
                from src.biocyc.celov_multiomics_post import load_locus_lookup

                n_pcs = int(clf_cache.get("gene_pcs") or gene_pcs or 5)
                weighed = build_weighed_genes(
                    _PCA_RUNTIME["loadings"],
                    _PCA_RUNTIME["explained_variance"],
                    np.asarray(clf_cache["w_gene"], dtype=float),
                    n_pcs,
                )
                lookup = None
                if locus_path and str(locus_path).strip():
                    lookup = load_locus_lookup(str(locus_path).strip())
                name = active if isinstance(active, str) else (active[0] if active else None)
                entry = next(
                    (
                        d
                        for d in (project_blob or {}).get("datasets", [])
                        if d.get("name") == name
                    ),
                    None,
                )
                paths = save_classifier_celov(
                    weighed,
                    str(out_path).strip(),
                    mode=mode or "up_and_down",
                    locus_lookup=lookup,
                    id_column=resolve_celov_id_col(entry),
                )
                return "Saved Celov: " + ", ".join(str(p) for p in paths)
            except Exception as exc:  # noqa: BLE001
                return f"Celov save error: {exc}"

        @app.callback(
            Output("pca-expr-by", "options"),
            Output("pca-pc-expr-by", "options"),
            Output("pca-clf-expr-by", "options"),
            Input("pca-cache", "data"),
            Input("pca-pc-locus", "value"),
            Input("pca-clf-locus", "value"),
        )
        def _pca_expr_by_options(cache, pc_locus, clf_locus):
            lookup = _ensure_pca_lookup(pc_locus or clf_locus)
            opts = _gene_select_by_options(lookup)
            return opts, opts, opts

        @app.callback(
            Output("pca-expr-genes", "options"),
            Input("pca-cache", "data"),
            Input("pca-expr-by", "value"),
            Input("pca-pc-locus", "value"),
            Input("pca-expr-genes", "search_value"),
            Input("pca-expr-genes", "value"),
        )
        def _pca_expr_gene_options(cache, by_col, locus_path, search, selected):
            expr = _live_expression()
            if expr is None or not cache:
                return []
            lookup = _ensure_pca_lookup(locus_path)
            names = _gene_name_map(lookup, by_col)
            hits = filter_ids_by_search(
                expr.columns.astype(str), search, selected, labels=names
            )
            return _gene_dropdown_options(hits, lookup, by_col)

        @app.callback(
            Output("pca-pc-expr-genes", "options"),
            Input("pca-cache", "data"),
            Input("pca-weight-pc", "value"),
            Input("pca-pc-expr-by", "value"),
            Input("pca-pc-expr-topn", "value"),
            Input("pca-pc-expr-side", "value"),
            Input("pca-pc-locus", "value"),
        )
        def _pca_pc_expr_gene_options(cache, pc, by_col, topn, side, locus_path):
            if not cache or "loadings" not in _PCA_RUNTIME or not pc:
                return []
            try:
                weighed = weighed_genes_from_pc(_PCA_RUNTIME["loadings"], str(pc))
            except Exception:  # noqa: BLE001
                return []
            top = _top_weighed(weighed, topn, side)
            weights = dict(zip(top["geneID"].astype(str), top["gene_weight"].astype(float)))
            lookup = _ensure_pca_lookup(locus_path)
            return _gene_dropdown_options(list(top["geneID"].astype(str)), lookup, by_col, weights)

        @app.callback(
            Output("pca-clf-expr-genes", "options"),
            Input("pca-clf-cache", "data"),
            Input("pca-cache", "data"),
            Input("pca-clf-expr-by", "value"),
            Input("pca-clf-expr-topn", "value"),
            Input("pca-clf-expr-side", "value"),
            Input("pca-clf-locus", "value"),
            Input("pca-clf-gene-pcs", "value"),
        )
        def _pca_clf_expr_gene_options(
            clf_cache, cache, by_col, topn, side, locus_path, gene_pcs
        ):
            if not cache or not clf_cache or not clf_cache.get("w_gene"):
                return []
            if "loadings" not in _PCA_RUNTIME:
                return []
            try:
                n_pcs = int(clf_cache.get("gene_pcs") or gene_pcs or 5)
                weighed = build_weighed_genes(
                    _PCA_RUNTIME["loadings"],
                    _PCA_RUNTIME["explained_variance"],
                    np.asarray(clf_cache["w_gene"], dtype=float),
                    n_pcs,
                )
            except Exception:  # noqa: BLE001
                return []
            top = _top_weighed(weighed, topn, side)
            weights = dict(zip(top["geneID"].astype(str), top["gene_weight"].astype(float)))
            lookup = _ensure_pca_lookup(locus_path)
            return _gene_dropdown_options(list(top["geneID"].astype(str)), lookup, by_col, weights)

        @app.callback(
            Output("pca-expr-fig", "figure"),
            Output("pca-expr-status", "children"),
            Input("pca-expr-genes", "value"),
            Input("pca-expr-by", "value"),
            Input("pca-x", "value"),
            Input("pca-y", "value"),
            Input("pca-cache", "data"),
            Input("pca-expr-fig-w", "value"),
            Input("pca-expr-fig-h", "value"),
            Input("pca-pc-locus", "value"),
        )
        def _pca_expr_plot(genes, by_col, x_col, y_col, cache, fig_w, fig_h, locus_path):
            return _render_gene_expr_plot(
                genes, _ensure_pca_lookup(locus_path), by_col, x_col, y_col, cache, fig_w, fig_h
            )

        @app.callback(
            Output("pca-pc-expr-fig", "figure"),
            Output("pca-pc-expr-status", "children"),
            Input("pca-pc-expr-genes", "value"),
            Input("pca-pc-expr-by", "value"),
            Input("pca-x", "value"),
            Input("pca-y", "value"),
            Input("pca-cache", "data"),
            Input("pca-pc-expr-fig-w", "value"),
            Input("pca-pc-expr-fig-h", "value"),
            Input("pca-pc-locus", "value"),
        )
        def _pca_pc_expr_plot(genes, by_col, x_col, y_col, cache, fig_w, fig_h, locus_path):
            return _render_gene_expr_plot(
                genes, _ensure_pca_lookup(locus_path), by_col, x_col, y_col, cache, fig_w, fig_h
            )

        @app.callback(
            Output("pca-clf-expr-fig", "figure"),
            Output("pca-clf-expr-status", "children"),
            Input("pca-clf-expr-genes", "value"),
            Input("pca-clf-expr-by", "value"),
            Input("pca-x", "value"),
            Input("pca-y", "value"),
            Input("pca-cache", "data"),
            Input("pca-clf-expr-fig-w", "value"),
            Input("pca-clf-expr-fig-h", "value"),
            Input("pca-clf-locus", "value"),
            Input("pca-clf-cache", "data"),
        )
        def _pca_clf_expr_plot(
            genes, by_col, x_col, y_col, cache, fig_w, fig_h, locus_path, clf_cache
        ):
            if not clf_cache or not clf_cache.get("w_gene"):
                return go.Figure(), "Run the linear classifier first."
            return _render_gene_expr_plot(
                genes, _ensure_pca_lookup(locus_path), by_col, x_col, y_col, cache, fig_w, fig_h
            )

        register_sample_detail_callback(
            app,
            graph_id="pca-scatter",
            detail_id="pca-sample-detail",
            cache_id="pca-cache",
            get_score_df=_score_frame_for_plot,
        )

        def _wire_gene_expr_extras(prefix: str) -> None:
            register_sample_detail_callback(
                app,
                graph_id=f"{prefix}-fig",
                detail_id=f"{prefix}-sample-detail",
                cache_id="pca-cache",
                get_score_df=_score_frame_for_plot,
            )

            @app.callback(
                Output(f"{prefix}-enr-cols", "options"),
                Output(f"{prefix}-enr-cols", "value"),
                Output(f"{prefix}-enr-label", "options"),
                Output(f"{prefix}-enr-label", "value"),
                Input("session-store", "data"),
                Input(f"{prefix}-enr-kind", "value"),
                State(f"{prefix}-enr-cols", "value"),
                State(f"{prefix}-enr-label", "value"),
            )
            def _fill_gene_enr(session_blob, kind, current, label_cur, _prefix=prefix):
                cols = meta_columns_from_store(session_blob)
                opts = [{"label": c, "value": c} for c in cols]
                pref_src = DEFAULT_NUM_COLS if kind == "rank" else DEFAULT_CAT_COLS
                pref = present_cols(cols, pref_src) or cols
                triggered = callback_context.triggered_id
                if triggered == f"{_prefix}-enr-kind" or not current:
                    value = list(pref)
                else:
                    value = [c for c in current if c in cols]
                label = label_cur if label_cur in cols else (
                    "Biofilm" if "Biofilm" in cols else (cols[0] if cols else None)
                )
                return opts, value, opts, label

            @app.callback(
                Output(f"{prefix}-enr-fig", "figure"),
                Output(f"{prefix}-enr-status", "children"),
                Input(f"{prefix}-enr-run", "n_clicks"),
                State(f"{prefix}-genes", "value"),
                State(f"{prefix}-enr-kind", "value"),
                State(f"{prefix}-enr-cols", "value"),
                State(f"{prefix}-enr-label", "value"),
                prevent_initial_call=True,
            )
            def _run_gene_enr(n_clicks, genes, kind, cols, label_col):
                try:
                    return _gene_profile_enrich_fig(genes, kind, cols, label_col)
                except Exception as exc:  # noqa: BLE001
                    return go.Figure(), f"Enrichment error: {exc}"

        for _prefix in ("pca-expr", "pca-pc-expr", "pca-clf-expr"):
            _wire_gene_expr_extras(_prefix)
