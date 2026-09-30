"""PCA: interactive scatter + variance barplot, optional linear classifier, Celov export."""

from __future__ import annotations

from pathlib import Path
import math

from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from src.GUI.components.gene_scores import weighed_genes_from_classifier, weighed_genes_from_pc
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
    cat_entry_mats,
    corr_vs_axes,
    heatmap_matrix_fig,
    pc_separation_ari,
)
from .pca_classifier import (
    add_decision_boundary,
    classifier_axis_scores,
    classifier_performance,
    encode_binary_labels,
    fit_pc_classifier,
    performance_figure,
    save_classifier_celov,
)
from .volcano_condition import volcano_passing_gene_ids
from .workspace_enrichment import CLF_AXIS

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
_EXPR_COLS = 4


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


def _ranked_expr_gene_ids(source, pc, topn, side, clf_cache) -> tuple[list[str], dict[str, float]]:
    ranked: list[str] = []
    weights: dict[str, float] = {}
    if source == "clf":
        if clf_cache and clf_cache.get("w_gene") and "loadings" in _PCA_RUNTIME:
            try:
                n_pcs = int(clf_cache.get("gene_pcs") or 5)
                weighed = weighed_genes_from_classifier(
                    _PCA_RUNTIME["loadings"],
                    _PCA_RUNTIME["explained_variance"],
                    np.asarray(clf_cache["w_gene"], dtype=float),
                    n_pcs,
                )
                top = _top_weighed(weighed, topn, side)
                ranked = list(top["geneID"].astype(str))
                weights = dict(
                    zip(top["geneID"].astype(str), top["gene_weight"].astype(float))
                )
            except Exception:  # noqa: BLE001
                ranked = []
    elif pc and "loadings" in _PCA_RUNTIME:
        try:
            weighed = weighed_genes_from_pc(_PCA_RUNTIME["loadings"], str(pc))
            top = _top_weighed(weighed, topn, side)
            ranked = list(top["geneID"].astype(str))
            weights = dict(
                zip(top["geneID"].astype(str), top["gene_weight"].astype(float))
            )
        except Exception:  # noqa: BLE001
            ranked = []
    return ranked, weights


def _gene_expr_picker(prefix: str, *, from_weights: bool) -> html.Div:
    """Select-by + gene dropdown (+ top-N weights) and viridis PCA plot."""
    blurb = (
        "Pick genes from the top weights (gene ID or name), or take genes that pass "
        "the volcano fold-change / padj lines. Shown: expression on the current PC "
        "axes, one panel per gene."
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
        n_cols = min(_EXPR_COLS, n)
        n_rows = int(math.ceil(n / n_cols))
        w = int(fig_w) if fig_w else EXPORT_W
        h = int(fig_h) if fig_h else 480
        w = max(w, 260 * n_cols + 90)
        h = max(h, 280 * n_rows + 50)
        fig = set_fig_size(fig, w, h, default_width=w, default_height=h)
        _place_expr_colorbars(fig, n)
        return fig, f"{n} gene{'s' if n != 1 else ''} on {x_col or 'PC1'} vs {y_col or 'PC2'}."
    except Exception as exc:  # noqa: BLE001
        err = go.Figure()
        err.add_annotation(text=f"Plot error: {exc}", showarrow=False)
        return err, f"Plot error: {exc}"


def _gene_expression_figure(score_df, expr, genes, x_col, y_col, var, titles=None) -> go.Figure:
    """PCA scatter colored by each selected gene (viridis, own min–max).

    Up to four square panels per row; colorbar height matches each plot.
    """
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
    n_cols = min(_EXPR_COLS, n)
    n_rows = int(math.ceil(n / n_cols))
    titles_full = list(titles[:n]) + [""] * (n_rows * n_cols - n)
    v_space = 0.10 if n_rows <= 1 else min(0.16, 0.72 / (n_rows - 1))
    h_space = 0.08 if n_cols <= 1 else min(0.12, 0.36 / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles_full,
        horizontal_spacing=h_space,
        vertical_spacing=v_space,
    )
    xlab = _axis_label(x_col, var)
    ylab = _axis_label(y_col, var)
    x = pd.to_numeric(score_df[x_col], errors="coerce")
    y = pd.to_numeric(score_df[y_col], errors="coerce")
    hover = score_df.index.astype(str)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin, 1e-9)
    half += 0.05 * half
    xr = [cx - half, cx + half]
    yr = [cy - half, cy + half]

    for i, gene in enumerate(genes):
        row = i // n_cols + 1
        col = i % n_cols + 1
        title = titles[i] if i < len(titles) else gene
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
                    colorbar=dict(thickness=12, outlinewidth=0, xpad=4, ypad=0),
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
            row=row,
            col=col,
        )
        fig.update_xaxes(
            title_text=xlab if row == n_rows else "",
            range=xr,
            constrain="domain",
            matches=None,
            row=row,
            col=col,
        )
        xref = "x" if i == 0 else f"x{i + 1}"
        fig.update_yaxes(
            title_text=ylab if col == 1 else "",
            range=yr,
            scaleanchor=xref,
            scaleratio=1,
            constrain="domain",
            matches=None,
            row=row,
            col=col,
        )

    apply_export_layout(fig, title_lines=1, legend=False, uirevision="pca-expr")
    fig.update_layout(margin=dict(r=70, t=40))
    return fig


def _place_expr_colorbars(fig: go.Figure, n: int) -> None:
    """Match each colorbar height to the square PCA panel."""
    w = int(fig.layout.width or EXPORT_W)
    h = int(fig.layout.height or EXPORT_H)
    margin = fig.layout.margin
    inner_w = max(1.0, w - float(margin.l or 55) - float(margin.r or 70))
    inner_h = max(1.0, h - float(margin.t or 40) - float(margin.b or 50))
    for i in range(n):
        axis = "xaxis" if i == 0 else f"xaxis{i + 1}"
        yax = "yaxis" if i == 0 else f"yaxis{i + 1}"
        xd = fig.layout[axis].domain
        yd = fig.layout[yax].domain
        side = min(inner_w * (xd[1] - xd[0]), inner_h * (yd[1] - yd[0]))
        fig.data[i].marker.colorbar.update(
            dict(
                x=xd[1],
                y=(yd[0] + yd[1]) / 2,
                len=max(40.0, float(side)),
                lenmode="pixels",
                yanchor="middle",
                xanchor="left",
                thickness=12,
            )
        )


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

    def level1(self):
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
                html.H6("Linear classifier (optional)"),
                html.P(
                    "Pick a label and n PCs. That fit defines the extra sample axis "
                    "and the gene weights. The curve is only to choose n.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col([html.Label("Label column"), dcc.Dropdown(id="pca-clf-label")], md=3),
                        dbc.Col([html.Label("Positive class"), dcc.Dropdown(id="pca-clf-positive")], md=2),
                        dbc.Col(
                            [
                                html.Label("n PCs"),
                                dbc.Input(id="pca-clf-n", type="number", value=5, min=2, step=1),
                            ],
                            md=1,
                        ),
                        dbc.Col(
                            [
                                html.Label("min PCs"),
                                dbc.Input(id="pca-clf-pc-min", type="number", value=2, min=2, step=1),
                            ],
                            md=1,
                        ),
                        dbc.Col(
                            [
                                html.Label("max PCs"),
                                dbc.Input(id="pca-clf-pc-max", type="number", value=12, min=2, step=1),
                            ],
                            md=1,
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
                fig_size_controls("pca-clf", default_width=EXPORT_W, default_height=EXPORT_H),
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
                html.Div(id="pca-clf-status", className="text-muted small mb-2"),
                html.Div(id="pca-status", className="text-muted small"),
                dcc.Store(id="pca-cache"),
                dcc.Store(id="pca-group-aes", data={}),
                dcc.Store(id="pca-split-options", data=[]),
                dcc.Store(id="pca-clf-cache", data=None),
            ]
        )

    def level2(self):
        return html.Div(
            [
                html.H6("Weigh genes"),
                html.P(
                    "Weights from a PC or from the classifier axis (same n as step 1). "
                    "Select or deselect top genes, or search to add others. Celov uses "
                    "the raw weights (not scaled to max |w| = 1).",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Weight source"),
                                dcc.RadioItems(
                                    id="pca-weight-source",
                                    options=[
                                        {"label": " PC", "value": "pc"},
                                        {"label": " Linear classifier", "value": "clf"},
                                    ],
                                    value="pc",
                                    inline=True,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("PC"),
                                dcc.Dropdown(id="pca-weight-pc", placeholder="PC1", clearable=False),
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
                            md=5,
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
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Select top N", id="pca-expr-select-top", color="secondary", outline=True, size="sm"),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Select volcano genes",
                                id="pca-expr-select-volcano",
                                color="secondary",
                                outline=True,
                                size="sm",
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button("Clear genes", id="pca-expr-clear", color="secondary", outline=True, size="sm"),
                            width="auto",
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.P(
                    "Select volcano genes uses the last run under Volcano by condition "
                    "(hits past the fold-change / padj lines).",
                    className="text-muted small mb-2",
                ),
                _gene_expr_picker("pca-expr", from_weights=True),
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
            Output("pca-clf-label", "value"),
            Input("session-store", "data"),
            Input("pca-cache", "data"),
            State("pca-clf-label", "value"),
        )
        def _fill_clf_label(session_blob, cache, current):
            cols = meta_columns_from_store(session_blob)
            opts = [{"label": c, "value": c} for c in cols]
            value = current if current in cols else ("Biofilm" if "Biofilm" in cols else (cols[0] if cols else None))
            return opts, value

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
            State("pca-clf-n", "value"),
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
                ari = (
                    pd.DataFrame(clf_cache["ari"])
                    if clf_cache.get("ari")
                    else None
                )
                fig = performance_figure(
                    perf,
                    title=_plotly_title(
                        f"Linear Classifier on PCs for {dataset}",
                        f"split on {clf_cache.get('label_col')} "
                        f"(positive = {clf_cache.get('positive')})",
                    ),
                    ari_df=ari,
                )
                return no_update, set_fig_size(fig, fig_w, fig_h), no_update

            if not cache or "score_df" not in _PCA_RUNTIME:
                return no_update, empty, "Run PCA first."
            if not label_col or positive is None or positive == "":
                return no_update, empty, "Choose label column and positive class."
            try:
                score_df = _PCA_RUNTIME["score_df"]
                y = encode_binary_labels(score_df[label_col], positive)
                n_lo = int(pc_min or 2)
                n_hi = int(pc_max or 12)
                perf = classifier_performance(score_df, y, n_lo, n_hi)
                try:
                    ari = pc_separation_ari(score_df, label_col, positive, n_hi)
                except Exception:  # noqa: BLE001
                    ari = None
                fig = performance_figure(
                    perf,
                    title=_plotly_title(
                        f"Linear Classifier on PCs for {dataset}",
                        f"split on {label_col} (positive = {positive})",
                    ),
                    ari_df=ari,
                )
                fig = set_fig_size(fig, fig_w, fig_h)
                res2 = fit_pc_classifier(score_df, y, 2)
                n_gene = int(gene_pcs or 5)
                res_g = fit_pc_classifier(score_df, y, n_gene)
                if res_g:
                    axis = classifier_axis_scores(score_df, res_g["w"], n_gene, CLF_AXIS)
                    score_df[CLF_AXIS] = axis
                    _PCA_RUNTIME["score_df"] = score_df
                new_cache = {
                    "label_col": label_col,
                    "positive": str(positive),
                    "perf": perf.to_dict(orient="list") if perf is not None else {},
                    "ari": ari.to_dict(orient="list") if ari is not None else {},
                    "w2": list(map(float, res2["w"])) if res2 else None,
                    "b2": float(res2["b"]) if res2 else None,
                    "cv2": float(res2["cv_acc"]) if res2 else None,
                    "gene_pcs": n_gene,
                    "w_gene": list(map(float, res_g["w"])) if res_g else None,
                    "train_gene": float(res_g["train_acc"]) if res_g else None,
                    "cv_gene": float(res_g["cv_acc"]) if res_g else None,
                }
                msg = (
                    f"Classifier done. {n_gene}-PC axis stored. "
                    f"2-PC CV={new_cache['cv2']:.3f}; "
                    f"{n_gene}-PC train={new_cache['train_gene']:.3f}, CV={new_cache['cv_gene']:.3f}."
                    if res2 and res_g
                    else "Classifier finished with missing fits (check class sizes / n_pcs)."
                )
                return new_cache, fig, msg
            except Exception as exc:  # noqa: BLE001
                return no_update, empty, f"Classifier error: {exc}"

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
            Input("ds-active", "value"),
            Input("project-store", "data"),
        )
        def _sync_locus_from_active(active, blob):
            if not blob or not active:
                return ""
            name = active if isinstance(active, str) else (active[0] if active else None)
            entry = next(
                (d for d in blob.get("datasets", []) if d.get("name") == name),
                None,
            )
            if not entry:
                return ""
            locus = entry.get("locus_lookup") or ""
            root = blob.get("root")
            if locus and root and not Path(locus).is_absolute():
                locus = str(Path(root) / locus)
            return locus

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
            State("pca-weight-source", "value"),
            State("pca-weight-pc", "value"),
            State("pca-weight-out", "value"),
            State("pca-weight-celov-mode", "value"),
            State("pca-pc-locus", "value"),
            State("pca-clf-cache", "data"),
            State("project-store", "data"),
            State("ds-active", "value"),
            prevent_initial_call=True,
        )
        def _save_pc_celov(
            n_clicks, source, pc, out_path, mode, locus_path, clf_cache, project_blob, active
        ):
            if "loadings" not in _PCA_RUNTIME:
                return "Run PCA first."
            if not out_path or not str(out_path).strip():
                return "Choose an output .txt path (Browse)."
            try:
                from src.biocyc.celov_multiomics_post import load_locus_lookup

                if source == "clf":
                    if not clf_cache or not clf_cache.get("w_gene"):
                        return "Run the linear classifier first."
                    n_pcs = int(clf_cache.get("gene_pcs") or 5)
                    weighed = weighed_genes_from_classifier(
                        _PCA_RUNTIME["loadings"],
                        _PCA_RUNTIME["explained_variance"],
                        np.asarray(clf_cache["w_gene"], dtype=float),
                        n_pcs,
                    )
                    label = f"{CLF_AXIS} ({n_pcs} PCs)"
                else:
                    if not pc:
                        return "Choose a PC for weighed genes."
                    weighed = weighed_genes_from_pc(_PCA_RUNTIME["loadings"], str(pc))
                    label = str(pc)
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
                return f"Saved Celov ({label}): " + ", ".join(str(p) for p in paths)
            except Exception as exc:  # noqa: BLE001
                return f"Celov save error: {exc}"

        @app.callback(
            Output("pca-expr-by", "options"),
            Input("pca-cache", "data"),
            Input("pca-pc-locus", "value"),
        )
        def _pca_expr_by_options(cache, locus_path):
            return _gene_select_by_options(_ensure_pca_lookup(locus_path))

        @app.callback(
            Output("pca-expr-genes", "options"),
            Input("pca-cache", "data"),
            Input("pca-clf-cache", "data"),
            Input("pca-weight-source", "value"),
            Input("pca-weight-pc", "value"),
            Input("pca-expr-by", "value"),
            Input("pca-expr-topn", "value"),
            Input("pca-expr-side", "value"),
            Input("pca-pc-locus", "value"),
            Input("pca-expr-genes", "search_value"),
            Input("pca-expr-genes", "value"),
            Input("vc-cache", "data"),
        )
        def _pca_expr_gene_options(
            cache, clf_cache, source, pc, by_col, topn, side, locus_path, search, selected, _vc_cache
        ):
            expr = _live_expression()
            if expr is None or not cache:
                return []
            lookup = _ensure_pca_lookup(locus_path)
            names = _gene_name_map(lookup, by_col)
            ranked, weights = _ranked_expr_gene_ids(source, pc, topn, side, clf_cache)
            search_hits = filter_ids_by_search(
                expr.columns.astype(str), search, selected, labels=names
            )
            ordered = list(ranked)
            extra = list(selected or []) + list(search_hits) + volcano_passing_gene_ids()
            for gid in extra:
                if str(gid) not in ordered:
                    ordered.append(str(gid))
            return _gene_dropdown_options(ordered, lookup, by_col, weights)

        @app.callback(
            Output("pca-expr-genes", "value"),
            Output("analysis-selected-genes", "data", allow_duplicate=True),
            Input("pca-expr-select-top", "n_clicks"),
            Input("pca-expr-select-volcano", "n_clicks"),
            Input("pca-expr-clear", "n_clicks"),
            Input("pca-expr-genes", "value"),
            State("pca-weight-source", "value"),
            State("pca-weight-pc", "value"),
            State("pca-expr-topn", "value"),
            State("pca-expr-side", "value"),
            State("pca-clf-cache", "data"),
            prevent_initial_call=True,
        )
        def _pca_expr_select_or_clear(
            n_top, n_volcano, n_clear, current, source, pc, topn, side, clf_cache
        ):
            tid = callback_context.triggered_id
            if tid == "pca-expr-clear":
                return [], []
            if tid == "pca-expr-select-volcano":
                ids = volcano_passing_gene_ids()
                return ids, ids
            if tid == "pca-expr-select-top":
                ids, _ = _ranked_expr_gene_ids(source, pc, topn, side, clf_cache)
                return ids, ids
            genes = [str(g) for g in (current or [])]
            return no_update, genes

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

        register_sample_detail_callback(
            app,
            graph_id="pca-scatter",
            detail_id="pca-sample-detail",
            cache_id="pca-cache",
            get_score_df=_score_frame_for_plot,
        )
        register_sample_detail_callback(
            app,
            graph_id="pca-expr-fig",
            detail_id="pca-expr-sample-detail",
            cache_id="pca-cache",
            get_score_df=_score_frame_for_plot,
        )
