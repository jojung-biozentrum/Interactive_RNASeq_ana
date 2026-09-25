"""Parallel conditions — per-level Ward + closest LC (Parallel_conditions.ipynb)."""

from __future__ import annotations

import math
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

from src.biocyc.celov_multiomics_post import load_locus_lookup
from src.GUI.project import resolve_celov_id_col

from ..components.controls import (
    COLOR_CONSTANTS,
    EXPORT_H,
    EXPORT_W,
    SHAPE_CONSTANTS,
    SIZE_CONSTANTS,
    apply_export_layout,
    equal_xy_axes,
    fig_size_controls,
    plotly_title as _plotly_title,
    set_fig_size,
)
from ..components.folder_browser import pick_save_file_dialog
from ..components.sample_detail import (
    lookup_sample,
    sample_detail_placeholder,
    sample_detail_table,
)
from ..components.gene_meta_mark import filter_ids_by_search
from ..data_store import (
    active_dataset_entry as _active_dataset_entry,
    locus_path_from_session as _locus_path_from_session,
    session_from_store,
)
from .condition_enrichment import (
    DEFAULT_CAT_COLS,
    DEFAULT_NUM_COLS,
    assign_region_labels,
    corr_vs_region_rank,
    entry_fractions_fig,
    heatmap_matrix_fig,
    present_cols,
    table_html,
)
from .gene_gradients import (
    _MAX_PROFILE_GENES,
    _METHOD_COLS,
    _METHOD_LABELS,
    _PROFILE_BOTTOM,
    _PROFILE_CELL_H,
    _PROFILE_CELL_W,
    _PROFILE_COLS,
    _PROFILE_LEFT,
    _PROFILE_LEGEND_ROW_H,
    _PROFILE_RIGHT,
    _PROFILE_TITLE_H,
    _REP_COLORS,
    _click_gene_id,
    _display_name_for_gene,
    _level_tick_label,
    _natural_sorted,
    _subset_mask,
    gene_profile_grid_fig,
    gene_region_gradients,
    gradient_scatter_fig,
    save_gradient_celov,
)

_PAR_RUNTIME: dict = {}
_PAR_GRAD_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "par_gene_gradients"},
    "displaylogo": False,
}

_SYMBOL_3D = {
    "triangle-up": "diamond",
    "triangle-down": "diamond-open",
    "star": "cross",
    "hexagon": "square",
    "pentagon": "square-open",
}
_SYMBOL_3D_OK = {
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "circle-open",
    "square-open",
    "diamond-open",
}


def separation_k(Z: np.ndarray, is_bio: np.ndarray) -> tuple[int, np.ndarray]:
    """Smallest maxclust k where no cluster mixes biofilm and LC."""
    n = len(is_bio)
    for k in range(2, n + 1):
        labels = hierarchy.fcluster(Z, t=k, criterion="maxclust")
        mixed = False
        for c in np.unique(labels):
            m = labels == c
            if is_bio[m].any() and (~is_bio[m]).any():
                mixed = True
                break
        if not mixed:
            return k, labels
    labels = hierarchy.fcluster(Z, t=n, criterion="maxclust")
    return n, labels


def closest_non_bio_cluster(X: np.ndarray, labels: np.ndarray, is_bio: np.ndarray):
    """Non-biofilm cluster nearest the biofilm mean expression vector (centroid)."""
    bio_cent = X[is_bio].mean(axis=0)
    bio_clusters = set(labels[is_bio].tolist())
    best_c, best_d = None, np.inf
    for c in np.unique(labels):
        if c in bio_clusters:
            continue
        cent = X[labels == c].mean(axis=0)
        d = float(np.linalg.norm(cent - bio_cent))
        if d < best_d:
            best_c, best_d = int(c), d
    return best_c, best_d


def unique_sets(by_region: dict[str, set[str]], regions: list[str]) -> dict[str, set[str]]:
    unique = {}
    for r in regions:
        others = [by_region[o] for o in regions if o != r]
        u = by_region[r] - set.union(*others) if others else set(by_region[r])
        unique[r] = u if u else set(by_region[r])
    return unique


def unique_sets_strict(by_region: dict[str, set[str]], regions: list[str]) -> dict[str, set[str]]:
    unique = {}
    for r in regions:
        others = [by_region[o] for o in regions if o != r]
        unique[r] = by_region[r] - (set.union(*others) if others else set())
    return unique


def _symbol_for(symbol: str, is_3d: bool) -> str:
    symbol = str(symbol or "circle")
    if not is_3d:
        return symbol
    if symbol in _SYMBOL_3D_OK:
        return symbol
    return _SYMBOL_3D.get(symbol, "circle")


def _fit_par_pca(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    arr = np.asarray(X, dtype=float)
    n_pcs = max(2, min(50, int(arr.shape[0]) - 1, int(arr.shape[1])))
    scores = PCA(n_components=n_pcs).fit_transform(arr)
    cols = [f"PC{i + 1}" for i in range(n_pcs)]
    score_df = pd.DataFrame(scores, index=X.index.astype(str), columns=cols)
    if meta is not None:
        score_df = score_df.join(meta.reindex(score_df.index))
    return score_df


def _scatter_trace(
    score_df: pd.DataFrame,
    sids: list[str],
    *,
    x_col: str,
    y_col: str,
    z_col: str | None,
    color: str,
    symbol: str,
    size: float,
    opacity: float,
    name: str,
) -> go.Scatter | go.Scatter3d | None:
    idx = [s for s in sids if s in score_df.index]
    if not idx:
        return None
    sub = score_df.loc[idx]
    if "fileName" in sub.columns:
        text = sub["fileName"].astype(str)
    else:
        text = pd.Index(idx).astype(str)
    marker = dict(
        color=color,
        symbol=_symbol_for(symbol, bool(z_col)),
        size=float(size),
        opacity=float(opacity),
    )
    common = dict(
        x=sub[x_col],
        y=sub[y_col],
        mode="markers",
        name=name,
        text=text,
        customdata=idx,
        hovertemplate="%{text}<extra></extra>",
        marker=marker,
    )
    if z_col:
        return go.Scatter3d(z=sub[z_col], **common)
    return go.Scatter(**common)


def level_pca_fig(
    score_df: pd.DataFrame,
    *,
    level: str,
    bio_ids: set[str],
    labels: pd.Series,
    highlight_ids: set[str],
    highlight_name: str,
    x_col: str,
    y_col: str,
    z_col: str | None,
    hl_color: str,
    hl_shape: str,
    hl_size: float,
    title: str | dict,
) -> go.Figure:
    ids = [str(i) for i in score_df.index]
    bio_by_c: dict[int, list[str]] = {}
    highlight: list[str] = []
    rest: list[str] = []
    for sid in ids:
        if sid in bio_ids:
            if sid in labels.index:
                bio_by_c.setdefault(int(labels[sid]), []).append(sid)
            else:
                bio_by_c.setdefault(-1, []).append(sid)
        elif sid in highlight_ids:
            highlight.append(sid)
        else:
            rest.append(sid)
    fig = go.Figure()
    tr = _scatter_trace(
        score_df,
        rest,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        color="#888888",
        symbol="circle",
        size=7,
        opacity=0.1,
        name="rest",
    )
    if tr is not None:
        fig.add_trace(tr)
    for i, cid in enumerate(sorted(c for c in bio_by_c if c >= 0)):
        shape = SHAPE_CONSTANTS[i % len(SHAPE_CONSTANTS)]
        tr = _scatter_trace(
            score_df,
            bio_by_c[cid],
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            color="black",
            symbol=shape,
            size=10,
            opacity=0.7,
            name=f"{level} cluster {cid}",
        )
        if tr is not None:
            fig.add_trace(tr)
    if -1 in bio_by_c:
        tr = _scatter_trace(
            score_df,
            bio_by_c[-1],
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            color="black",
            symbol="circle",
            size=10,
            opacity=0.7,
            name=f"{level}",
        )
        if tr is not None:
            fig.add_trace(tr)
    tr = _scatter_trace(
        score_df,
        highlight,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        color=hl_color,
        symbol=hl_shape,
        size=hl_size,
        opacity=0.7,
        name=highlight_name,
    )
    if tr is not None:
        fig.add_trace(tr)
    if isinstance(title, dict):
        fig.update_layout(title=title)
        title_lines = str(title.get("text", "")).count("<br>") + 1
    else:
        fig.update_layout(title=_plotly_title(str(title)))
        title_lines = 1
    if not z_col:
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
    apply_export_layout(
        fig,
        title_lines=title_lines,
        legend=True,
        legend_kwargs={"title_text": level},
        uirevision=f"par-pca-{level}",
    )
    return fig


def biofilm_vs_close_fig(
    results: pd.DataFrame,
    method: str,
    *,
    mark_genes: set[str] | None = None,
    mark_label: str | None = None,
) -> go.Figure:
    y_col = _METHOD_COLS[method]
    x_col = f"{y_col}_biofilm"
    fig = go.Figure()
    if x_col not in results.columns or y_col not in results.columns:
        fig.add_annotation(text="Biofilm vs close not available", showarrow=False)
        return fig
    mark = {str(g) for g in (mark_genes or [])}
    marked = results["geneID"].astype(str).isin(mark)
    base = results.loc[~marked] if marked.any() else results
    if len(base):
        fig.add_trace(
            go.Scatter(
                x=base[x_col],
                y=base[y_col],
                mode="markers",
                marker=dict(size=6, color="#888888", opacity=0.45),
                customdata=base["geneID"].astype(str),
                text=base["geneID"].astype(str),
                hovertemplate=(
                    "%{text}<br>levels=%{x:.3g}<br>close=%{y:.3g}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    if marked.any():
        sub = results.loc[marked]
        fig.add_trace(
            go.Scatter(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                marker=dict(size=9, color="#d62728"),
                customdata=sub["geneID"].astype(str),
                text=sub["geneID"].astype(str),
                name=mark_label or "selected",
                hovertemplate=(
                    "%{text}<br>levels=%{x:.3g}<br>close=%{y:.3g}<extra></extra>"
                ),
            )
        )
    xs = pd.concat([results[x_col], results[y_col]], ignore_index=True)
    lim = float(np.nanmax(np.abs(xs.to_numpy(dtype=float)))) if len(xs) else 0.05
    lim = max(lim, 0.05)
    fig.add_trace(
        go.Scatter(
            x=[-lim, lim],
            y=[-lim, lim],
            mode="lines",
            line=dict(color="#808080", width=0.8, dash="dash"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    label = _METHOD_LABELS[method]
    fig.update_layout(
        title=_plotly_title(f"{label}: levels vs closest"),
        xaxis_title=f"{label} (levels)",
        yaxis_title=f"{label} (closest conditions)",
        height=420,
        width=420,
        uirevision=f"par-cmp-{method}",
        showlegend=bool(marked.any()),
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="closest",
        clickmode="event+select",
    )
    fig.update_xaxes(range=[-lim * 1.05, lim * 1.05], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-lim * 1.05, lim * 1.05])
    return fig


def gene_profile_biofilm_and_close_fig(
    genes: list[str],
    *,
    expr_bio: pd.DataFrame,
    meta_bio: pd.DataFrame,
    expr_close: pd.DataFrame,
    meta_close: pd.DataFrame,
    order_col: str,
    order_levels: list[str],
    rep_col: str | None,
    results: pd.DataFrame | None = None,
) -> go.Figure:
    """Biofilm replicate lines + closest mean ± std and grey samples, one panel per gene."""
    gene_ids = set(expr_bio.index.astype(str)) | set(expr_close.index.astype(str))
    genes = [str(g) for g in genes if str(g) in gene_ids][:_MAX_PROFILE_GENES]
    if not genes:
        return gene_profile_grid_fig(
            [],
            expr=pd.DataFrame(),
            meta=pd.DataFrame(),
            order_col=order_col,
            order_levels=order_levels,
            rep_col=None,
        )
    n = len(genes)
    n_cols = _PROFILE_COLS
    n_rows = int(math.ceil(n / n_cols))
    titles = [_display_name_for_gene(results, g) for g in genes]
    titles_full = titles + [""] * (n_rows * n_cols - n)
    v_space = 0.12 if n_rows <= 1 else min(0.22, 0.72 / (n_rows - 1))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles_full,
        horizontal_spacing=0.05,
        vertical_spacing=v_space,
    )
    meta_bio = meta_bio.copy()
    meta_bio.index = meta_bio.index.astype(str)
    meta_bio[order_col] = meta_bio[order_col].astype(str)
    meta_close = meta_close.copy()
    meta_close.index = meta_close.index.astype(str)
    meta_close[order_col] = meta_close[order_col].astype(str)
    x_pos = list(range(len(order_levels)))
    x_labels = [_level_tick_label(lv) for lv in order_levels]
    if rep_col and rep_col in meta_bio.columns:
        replicates = _natural_sorted(meta_bio[rep_col].dropna().unique())
    else:
        replicates = ["all"]
    legend_done: set[str] = set()
    for gi, gene in enumerate(genes):
        row = gi // n_cols + 1
        col = gi % n_cols + 1
        for ri, replicate in enumerate(replicates):
            if rep_col and rep_col in meta_bio.columns and replicate != "all":
                rep_meta = meta_bio.loc[meta_bio[rep_col].astype(str) == str(replicate)]
            else:
                rep_meta = meta_bio
            ys: list[float | None] = []
            for level in order_levels:
                samples = [s for s in rep_meta.index[rep_meta[order_col] == level] if s in expr_bio.columns]
                if not samples or gene not in expr_bio.index:
                    ys.append(None)
                    continue
                ys.append(float(np.nanmean(expr_bio.loc[gene, samples].to_numpy(dtype=float))))
            if all(v is None for v in ys):
                continue
            color = _REP_COLORS[ri % len(_REP_COLORS)]
            show_leg = str(replicate) not in legend_done
            if show_leg:
                legend_done.add(str(replicate))
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=ys,
                    mode="lines+markers",
                    name=str(replicate),
                    line=dict(color=color, width=1.5),
                    marker=dict(size=6, color=color),
                    legendgroup=str(replicate),
                    showlegend=show_leg,
                    hovertemplate=(
                        f"{titles[gi]}<br>levels {replicate}<br>"
                        "level=%{x}<br>expr=%{y:.3g}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        xs_pts: list[float] = []
        ys_pts: list[float] = []
        ys_m: list[float | None] = []
        yerr: list[float] = []
        for i, level in enumerate(order_levels):
            samples = [s for s in meta_close.index[meta_close[order_col] == level] if s in expr_close.columns]
            if not samples or gene not in expr_close.index:
                ys_m.append(None)
                yerr.append(0.0)
                continue
            vals = pd.to_numeric(expr_close.loc[gene, samples], errors="coerce").to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            for v in finite:
                xs_pts.append(float(i))
                ys_pts.append(float(v))
            if len(finite) == 0:
                ys_m.append(None)
                yerr.append(0.0)
                continue
            ys_m.append(float(np.mean(finite)))
            yerr.append(float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0)
        if xs_pts:
            show_pts = "closest samples" not in legend_done
            if show_pts:
                legend_done.add("closest samples")
            fig.add_trace(
                go.Scatter(
                    x=xs_pts,
                    y=ys_pts,
                    mode="markers",
                    name="closest samples",
                    marker=dict(size=6, color="#888888", opacity=0.55),
                    legendgroup="closest samples",
                    showlegend=show_pts,
                    hovertemplate=f"{titles[gi]}<br>closest<br>expr=%{{y:.3g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        if not all(v is None for v in ys_m):
            show_mean = "closest mean" not in legend_done
            if show_mean:
                legend_done.add("closest mean")
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=ys_m,
                    mode="lines+markers",
                    name="closest mean",
                    line=dict(color="black", width=2),
                    marker=dict(size=8, color="black"),
                    error_y=dict(type="data", array=yerr, color="black", thickness=1.2, width=4),
                    legendgroup="closest mean",
                    showlegend=show_mean,
                    hovertemplate=f"{titles[gi]}<br>closest mean=%{{y:.3g}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.update_xaxes(
            tickmode="array",
            tickvals=x_pos,
            ticktext=x_labels,
            title_text=order_col if row == n_rows else "",
            row=row,
            col=col,
        )
        fig.update_yaxes(title_text="expression" if col == 1 else "", row=row, col=col)
    for gi in range(n, n_rows * n_cols):
        row = gi // n_cols + 1
        col = gi % n_cols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)
    n_leg = max(1, len(legend_done))
    legend_w = 110
    top_margin = max(_PROFILE_TITLE_H, n_leg * _PROFILE_LEGEND_ROW_H) + 12
    left_margin = _PROFILE_LEFT + legend_w
    fig_h = top_margin + n_rows * _PROFILE_CELL_H + _PROFILE_BOTTOM
    fig_w = left_margin + n_cols * _PROFILE_CELL_W + _PROFILE_RIGHT
    title = _plotly_title("Levels (replicates) and closest mean ± std")
    title.update(x=(legend_w + 12) / max(fig_w, 1), xref="container", xanchor="left", y=0.995, yref="container", yanchor="top")
    fig.update_layout(
        title=title,
        height=fig_h,
        width=fig_w,
        margin=dict(l=left_margin, r=_PROFILE_RIGHT, t=top_margin, b=_PROFILE_BOTTOM),
        legend=dict(
            orientation="v",
            font=dict(size=10),
            xref="container",
            yref="container",
            x=0.004,
            y=0.995,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
        ),
        uirevision="par-profiles",
        hovermode="closest",
    )
    fig.update_annotations(font_size=11)
    return fig


def run_per_level_clusters(
    X: pd.DataFrame,
    meta: pd.DataFrame,
    is_bio: pd.Series,
    order_col: str,
    regions: list[str],
    linkage_method: str = "ward",
):
    """One Ward tree per region ∪ all LC; first unmixed cut; closest LC centroid."""
    is_lc = ~is_bio
    rows = []
    closest = {}
    level_detail = {}
    for region in regions:
        in_region = is_bio & meta[order_col].astype(str).eq(region)
        keep = is_lc | in_region
        sub = X.loc[keep]
        if sub.shape[0] < 3 or int(in_region.sum()) < 1 or int(is_lc.sum()) < 1:
            closest[region] = set()
            level_detail[region] = {
                "labels": pd.Series(dtype=int),
                "closest_c": None,
                "bio_ids": set(meta.index.astype(str)[in_region.fillna(False)]),
            }
            rows.append(
                {
                    "region": region,
                    "k_sep": np.nan,
                    "n_closest": 0,
                    "dist": np.nan,
                }
            )
            continue
        bio_m = in_region.loc[sub.index].to_numpy()
        Z = hierarchy.linkage(sub.to_numpy(dtype=float), method=linkage_method)
        k_sep, labels = separation_k(Z, bio_m)
        closest_c, dist = closest_non_bio_cluster(sub.to_numpy(dtype=float), labels, bio_m)
        mask = (labels == closest_c) & (~bio_m) if closest_c is not None else np.zeros(len(sub), dtype=bool)
        conds = set(sub.index.astype(str)[mask])
        closest[region] = conds
        level_detail[region] = {
            "labels": pd.Series(labels, index=sub.index.astype(str)),
            "closest_c": closest_c,
            "bio_ids": set(sub.index.astype(str)[bio_m]),
        }
        rows.append(
            {
                "region": region,
                "k_sep": int(k_sep),
                "n_closest": len(conds),
                "dist": dist,
            }
        )
    summary = pd.DataFrame(rows)
    unique = unique_sets(closest, regions)
    unique_strict = unique_sets_strict(closest, regions)
    summary["n_unique"] = summary["region"].map(lambda r: len(unique_strict[r]))
    return summary, closest, unique, unique_strict, level_detail


class ParallelConditionsModule:
    id = "parallel-conditions"
    label = "Parallel conditions & genes"

    def layout(self):
        return html.Div(
            [
                html.P(
                    "Pick the biofilm subset, the level column, and the level order. "
                    "Each level is clustered with all non-subset samples. Shown: a PCA "
                    "per level, then enrichment or gene gradients on closest or "
                    "uniquely-close samples.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Subset column (biofilm)"),
                                dcc.Dropdown(id="par-subset-col"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Subset value"),
                                dcc.Dropdown(id="par-subset-val"),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Level column"),
                                dcc.Dropdown(id="par-order-col"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Levels (order)"),
                                dcc.Dropdown(id="par-levels", multi=True),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button("Run per-level clustering", id="par-run", color="primary", className="mb-2"),
                html.Div(id="par-status", className="text-muted small mb-2"),
                html.Div(id="par-summary"),
                html.H6("PCA per level", className="mt-3"),
                html.P(
                    "Subset-value samples of the current level are black (alpha 0.7); "
                    "each of their clusters is a different shape. Closest or unique "
                    "closest (set difference; empty unique uses all closest) use the "
                    "chosen color, shape, and size (alpha 0.7). Other samples in that "
                    "level's tree are grey (alpha 0.1). Four panels per row.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("X"),
                                dcc.Dropdown(id="par-pca-x", value="PC1", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Y"),
                                dcc.Dropdown(id="par-pca-y", value="PC2", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Z (optional 3D)"),
                                dcc.Dropdown(id="par-pca-z", clearable=True),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Highlight"),
                                dcc.RadioItems(
                                    id="par-pca-hl",
                                    options=[
                                        {"label": " closest", "value": "closest"},
                                        {"label": " unique closest", "value": "unique"},
                                    ],
                                    value="closest",
                                    inline=True,
                                ),
                            ],
                            md=3,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Closest color"),
                                dcc.Dropdown(
                                    id="par-pca-color",
                                    options=[{"label": c, "value": c} for c in COLOR_CONSTANTS],
                                    value="crimson",
                                    clearable=False,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Closest shape"),
                                dcc.Dropdown(
                                    id="par-pca-shape",
                                    options=[{"label": s, "value": s} for s in SHAPE_CONSTANTS],
                                    value="circle",
                                    clearable=False,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Closest size"),
                                dcc.Dropdown(
                                    id="par-pca-size",
                                    options=[{"label": s, "value": int(s)} for s in SIZE_CONSTANTS],
                                    value=12,
                                    clearable=False,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            fig_size_controls(
                                "par-pca",
                                default_width=EXPORT_W,
                                default_height=EXPORT_H,
                                heading="PCA size (px)",
                            ),
                            md=6,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="par-pca-panels"),
                html.Div(
                    [
                        html.H6("Sample metadata", className="mb-2"),
                        html.Div(
                            id="par-pca-sample-detail",
                            children=sample_detail_placeholder(),
                        ),
                    ],
                    className="border rounded p-2 bg-light mb-2",
                ),
                html.Hr(),
                dcc.Tabs(
                    id="par-analysis-tabs",
                    value="enrich",
                    children=[
                        dcc.Tab(
                            label="Condition enrichment",
                            value="enrich",
                            children=[
                html.H6("Condition enrichment", className="mt-2"),
                html.P(
                    "Categorical: stacked entry fractions per closest or uniquely-close "
                    "level. Rankable: Pearson and Spearman of ranked metadata vs "
                    "level rank.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Columns"),
                                dcc.RadioItems(
                                    id="par-enr-kind",
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
                                html.Label("Grouping"),
                                dcc.RadioItems(
                                    id="par-enr-mode",
                                    options=[
                                        {"label": " closest", "value": "closest"},
                                        {"label": " uniquely-close", "value": "unique"},
                                    ],
                                    value="closest",
                                    inline=True,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Metadata columns"),
                                dcc.Dropdown(id="par-enr-cols", multi=True),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Run enrichment",
                                id="par-enr-run",
                                color="secondary",
                                className="mt-4",
                            ),
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="par-enr-status", className="text-muted small mb-2"),
                dcc.Graph(id="par-enr-fig", figure={}, config=_PAR_GRAD_CONFIG),
                            ],
                        ),
                        dcc.Tab(
                            label="Pooled gene gradients",
                            value="gradients",
                            children=[
                html.H6("Pooled gene gradients", className="mt-2"),
                html.P(
                    "Run on closest or uniquely-close samples. Shown: Pearson and "
                    "Spearman vs dynamic range, and each measure as levels (x) vs "
                    "closest (y). Select a gene to overlay biofilm replicate lines "
                    "with closest mean ± std (grey points).",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.RadioItems(
                                id="par-grad-mode",
                                options=[
                                    {"label": " closest", "value": "closest"},
                                    {"label": " uniquely-close", "value": "unique"},
                                ],
                                value="closest",
                                inline=True,
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Run pooled gradients",
                                id="par-grad-run",
                                color="secondary",
                            ),
                            md=3,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="par-grad-status", className="text-muted small mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="par-grad-pearson",
                                    figure={},
                                    config=_PAR_GRAD_CONFIG,
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("|ρ|", className="small mb-0"),
                                                dbc.Input(
                                                    id="par-grad-rho-p",
                                                    type="number",
                                                    value=0.7,
                                                    step=0.05,
                                                    min=0,
                                                    max=1,
                                                    size="sm",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Dyn. range", className="small mb-0"),
                                                dbc.Input(
                                                    id="par-grad-dr-p",
                                                    type="number",
                                                    value=1.0,
                                                    step=0.1,
                                                    size="sm",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ],
                                    className="g-1",
                                ),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                dcc.Graph(
                                    id="par-grad-spearman",
                                    figure={},
                                    config=_PAR_GRAD_CONFIG,
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("|ρ|", className="small mb-0"),
                                                dbc.Input(
                                                    id="par-grad-rho-s",
                                                    type="number",
                                                    value=0.7,
                                                    step=0.05,
                                                    min=0,
                                                    max=1,
                                                    size="sm",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("Dyn. range", className="small mb-0"),
                                                dbc.Input(
                                                    id="par-grad-dr-s",
                                                    type="number",
                                                    value=1.0,
                                                    step=0.1,
                                                    size="sm",
                                                ),
                                            ],
                                            md=6,
                                        ),
                                    ],
                                    className="g-1",
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                id="par-grad-cmp-pearson",
                                figure={},
                                config=_PAR_GRAD_CONFIG,
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="par-grad-cmp-spearman",
                                figure={},
                                config=_PAR_GRAD_CONFIG,
                            ),
                            md=6,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.H6("Selected gene profiles", className="mt-3"),
                html.P(
                    "Click a gene or pick IDs. Biofilm lines follow the replicate "
                    "column (same as gene gradients). Closest samples are grey; "
                    "closest mean is black with std bars.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Replicate column (levels)"),
                                dcc.Dropdown(id="par-grad-rep-col", clearable=True),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="par-grad-genes",
                                multi=True,
                                placeholder="Type 2+ characters to search genes…",
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Clear selected genes",
                                id="par-grad-clear",
                                color="secondary",
                                outline=True,
                                size="sm",
                            ),
                            md=3,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="par-grad-profiles-status", className="text-muted small mb-1"),
                dcc.Graph(
                    id="par-grad-profiles",
                    figure={},
                    config=_PAR_GRAD_CONFIG,
                ),
                html.Hr(),
                html.H6("Save Celov"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Score"),
                                dcc.Dropdown(
                                    id="par-grad-celov-score",
                                    options=[
                                        {"label": "Pearson ρ (closest)", "value": "pearson_rho"},
                                        {"label": "Spearman ρ (closest)", "value": "spearman_rho"},
                                        {
                                            "label": "Pearson ρ (levels)",
                                            "value": "pearson_rho_biofilm",
                                        },
                                        {
                                            "label": "Spearman ρ (levels)",
                                            "value": "spearman_rho_biofilm",
                                        },
                                    ],
                                    value="pearson_rho",
                                    clearable=False,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Genes"),
                                dcc.RadioItems(
                                    id="par-grad-celov-mode",
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
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Output path ({} = type)"),
                                dbc.InputGroup(
                                    [
                                        dbc.Input(id="par-grad-celov-out", type="text"),
                                        dbc.Button(
                                            "Browse…",
                                            id="par-grad-celov-browse",
                                            color="info",
                                            outline=True,
                                        ),
                                    ]
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button(
                    "Save Celov",
                    id="par-grad-celov-save",
                    color="secondary",
                    className="mb-2",
                ),
                html.Div(id="par-grad-celov-status", className="text-muted small mb-2"),
                dcc.Store(id="par-grad-cache"),
                dcc.Store(id="par-grad-selected", data=[]),
                            ],
                        ),
                    ],
                ),
                dcc.Store(id="par-cache"),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output("par-subset-col", "options"),
            Output("par-order-col", "options"),
            Output("par-subset-col", "value"),
            Output("par-order-col", "value"),
            Input("session-store", "data"),
            State("par-subset-col", "value"),
            State("par-order-col", "value"),
        )
        def _fill_cols(session_blob, sub_cur, ord_cur):
            cols = list((session_blob or {}).get("meta_columns", []))
            opts = [{"label": c, "value": c} for c in cols]
            sub = sub_cur if sub_cur in cols else ("Biofilm" if "Biofilm" in cols else (cols[0] if cols else None))
            order = ord_cur if ord_cur in cols else ("GrowthPhase" if "GrowthPhase" in cols else (cols[1] if len(cols) > 1 else sub))
            return opts, opts, sub, order

        @app.callback(
            Output("par-grad-rep-col", "options"),
            Output("par-grad-rep-col", "value"),
            Input("session-store", "data"),
            State("par-grad-rep-col", "value"),
        )
        def _fill_par_rep(session_blob, current):
            cols = list((session_blob or {}).get("meta_columns", []))
            opts = [{"label": c, "value": c} for c in cols]
            prefer = current if current in cols else (
                "Replicate" if "Replicate" in cols else (
                    "replicate" if "replicate" in cols else None
                )
            )
            return opts, prefer

        @app.callback(
            Output("par-subset-val", "options"),
            Output("par-subset-val", "value"),
            Input("par-subset-col", "value"),
            State("session-store", "data"),
            State("par-subset-val", "value"),
        )
        def _fill_sub_val(col, session_blob, current):
            session = session_from_store(session_blob)
            if session.metadata is None or not col or col not in session.metadata.columns:
                return [], None
            levels = _natural_sorted(session.metadata[col].dropna().astype(str).unique())
            opts = [{"label": v, "value": v} for v in levels]
            prefer = next((v for v in levels if v.lower() in {"true", "1", "yes"}), None)
            value = current if current in levels else (prefer or (levels[0] if levels else None))
            return opts, value

        @app.callback(
            Output("par-levels", "options"),
            Output("par-levels", "value"),
            Input("par-order-col", "value"),
            Input("par-subset-col", "value"),
            Input("par-subset-val", "value"),
            State("session-store", "data"),
            State("par-levels", "value"),
        )
        def _fill_levels(order_col, sub_col, sub_val, session_blob, current):
            session = session_from_store(session_blob)
            if session.metadata is None or not order_col or order_col not in session.metadata.columns:
                return [], []
            meta = session.metadata
            if sub_col and sub_val is not None and sub_col in meta.columns:
                meta = meta.loc[_subset_mask(meta, sub_col, str(sub_val))]
            levels = _natural_sorted(meta[order_col].dropna().astype(str).unique())
            opts = [{"label": v, "value": v} for v in levels]
            default = [v for v in ("region1", "region2", "region3", "region4") if v in levels] or levels
            value = [v for v in (current or default) if v in levels] or default
            return opts, value

        @app.callback(
            Output("par-cache", "data"),
            Output("par-status", "children"),
            Output("par-summary", "children"),
            Input("par-run", "n_clicks"),
            State("session-store", "data"),
            State("par-subset-col", "value"),
            State("par-subset-val", "value"),
            State("par-order-col", "value"),
            State("par-levels", "value"),
            prevent_initial_call=True,
        )
        def _run(n_clicks, session_blob, sub_col, sub_val, order_col, levels):
            session = session_from_store(session_blob)
            if not session.ready:
                return no_update, session.error or "Load a dataset first.", no_update
            if not sub_col or sub_val is None or not order_col or not levels:
                return no_update, "Choose subset, level column, and at least two levels.", no_update
            levels = [str(x) for x in levels]
            if len(levels) < 2:
                return no_update, "Need at least two levels.", no_update
            meta = session.metadata.copy()
            is_bio = _subset_mask(meta, sub_col, str(sub_val))
            X = session.expression.copy()
            X.index = X.index.astype(str)
            meta = meta.reindex(X.index)
            is_bio = is_bio.reindex(X.index).fillna(False)
            try:
                summary, closest, unique, unique_strict, level_detail = run_per_level_clusters(
                    X, meta, is_bio, order_col, levels
                )
                score_df = _fit_par_pca(X, meta)
            except Exception as exc:  # noqa: BLE001
                return no_update, f"Clustering error: {exc}", no_update
            _PAR_RUNTIME.clear()
            _PAR_RUNTIME.update(
                {
                    "summary": summary,
                    "closest": closest,
                    "unique": unique,
                    "unique_strict": unique_strict,
                    "level_detail": level_detail,
                    "score_df": score_df,
                    "meta": meta,
                    "X": X,
                    "is_bio": is_bio,
                    "levels": levels,
                    "order_col": order_col,
                }
            )
            return (
                {"n_regions": len(levels)},
                f"Per-level Ward done for {len(levels)} levels.",
                table_html(summary),
            )

        @app.callback(
            Output("par-pca-x", "options"),
            Output("par-pca-y", "options"),
            Output("par-pca-z", "options"),
            Output("par-pca-x", "value"),
            Output("par-pca-y", "value"),
            Output("par-pca-z", "value"),
            Input("par-cache", "data"),
        )
        def _fill_par_pca_axes(cache):
            score_df = _PAR_RUNTIME.get("score_df")
            if score_df is None or not cache:
                return [], [], [], None, None, None
            pcs = [
                c
                for c in score_df.columns
                if str(c).startswith("PC") and str(c)[2:].isdigit()
            ]
            opts = [{"label": c, "value": c} for c in pcs]
            return (
                opts,
                opts,
                opts,
                "PC1" if "PC1" in pcs else (pcs[0] if pcs else None),
                "PC2" if "PC2" in pcs else (pcs[1] if len(pcs) > 1 else None),
                None,
            )

        @app.callback(
            Output("par-pca-panels", "children"),
            Input("par-cache", "data"),
            Input("par-pca-x", "value"),
            Input("par-pca-y", "value"),
            Input("par-pca-z", "value"),
            Input("par-pca-hl", "value"),
            Input("par-pca-color", "value"),
            Input("par-pca-shape", "value"),
            Input("par-pca-size", "value"),
            Input("par-pca-fig-w", "value"),
            Input("par-pca-fig-h", "value"),
        )
        def _plot_par_pca(
            cache,
            x_col,
            y_col,
            z_col,
            hl_mode,
            hl_color,
            hl_shape,
            hl_size,
            fig_w,
            fig_h,
        ):
            if not cache or "score_df" not in _PAR_RUNTIME:
                return html.P(
                    "Run per-level clustering to show PCA.",
                    className="text-muted small mb-0",
                )
            score_df = _PAR_RUNTIME["score_df"]
            pcs = [
                c
                for c in score_df.columns
                if str(c).startswith("PC") and str(c)[2:].isdigit()
            ]
            x_col = x_col if x_col in score_df.columns else (pcs[0] if pcs else None)
            y_col = y_col if y_col in score_df.columns else (pcs[1] if len(pcs) > 1 else x_col)
            z_col = z_col if z_col in score_df.columns else None
            if not x_col or not y_col:
                return html.P("PCA has no components to plot.", className="text-muted small mb-0")
            hl_mode = "unique" if hl_mode == "unique" else "closest"
            hl_name = "unique closest" if hl_mode == "unique" else "closest"
            hl_map = (
                _PAR_RUNTIME.get("unique")
                if hl_mode == "unique"
                else _PAR_RUNTIME.get("closest")
            ) or {}
            detail = _PAR_RUNTIME.get("level_detail") or {}
            levels = list(_PAR_RUNTIME.get("levels") or [])
            panels = []
            for level in levels:
                info = detail.get(level) or {}
                labels = info.get("labels")
                if labels is None:
                    labels = pd.Series(dtype=int)
                tree_ids = set(labels.index.astype(str)) if len(labels) else set()
                keep = tree_ids | set(info.get("bio_ids") or []) | set(hl_map.get(level) or [])
                score_use = (
                    score_df.loc[score_df.index.astype(str).isin(keep)]
                    if keep
                    else score_df
                )
                fig = level_pca_fig(
                    score_use,
                    level=level,
                    bio_ids=set(info.get("bio_ids") or []),
                    labels=labels,
                    highlight_ids=set(hl_map.get(level) or []),
                    highlight_name=hl_name,
                    x_col=x_col,
                    y_col=y_col,
                    z_col=z_col,
                    hl_color=hl_color or "crimson",
                    hl_shape=hl_shape or "circle",
                    hl_size=float(hl_size if hl_size is not None else 12),
                    title=_plotly_title(
                        f"PCA · {level}",
                        f"{hl_name} n={len(hl_map.get(level) or [])}",
                    ),
                )
                fig = set_fig_size(fig, fig_w, fig_h)
                panels.append(
                    dcc.Graph(
                        id={"type": "par-pca-fig", "level": level},
                        figure=fig,
                        config={
                            "toImageButtonOptions": {
                                "format": "svg",
                                "filename": f"par_pca_{level}",
                            },
                            "displaylogo": False,
                        },
                    )
                )
            if not panels:
                return html.P("No levels to plot.", className="text-muted small mb-0")
            rows = []
            for i in range(0, len(panels), 4):
                chunk = [dbc.Col(p, md=3) for p in panels[i : i + 4]]
                rows.append(dbc.Row(chunk, className="g-2 mb-2"))
            return rows

        @app.callback(
            Output("par-pca-sample-detail", "children"),
            Input({"type": "par-pca-fig", "level": ALL}, "clickData"),
            Input("par-cache", "data"),
        )
        def _par_pca_sample(clicks, cache):
            if not cache or "score_df" not in _PAR_RUNTIME:
                return sample_detail_placeholder()
            triggered = callback_context.triggered_id
            if triggered == "par-cache" or not any(clicks or []):
                return sample_detail_placeholder()
            trig = callback_context.triggered
            if not trig:
                return sample_detail_placeholder()
            click = trig[0].get("value")
            if not click:
                return sample_detail_placeholder()
            point = click["points"][0]
            click_id = point.get("customdata")
            if isinstance(click_id, (list, tuple)):
                click_id = click_id[0] if click_id else None
            elif click_id is not None:
                click_id = str(click_id)
            row = lookup_sample(_PAR_RUNTIME.get("score_df"), click_id)
            return sample_detail_table(row)

        @app.callback(
            Output("par-enr-cols", "options"),
            Output("par-enr-cols", "value"),
            Input("session-store", "data"),
            Input("par-cache", "data"),
            Input("par-enr-kind", "value"),
            State("par-enr-cols", "value"),
        )
        def _enr_cols(session_blob, cache, kind, current):
            cols = list((session_blob or {}).get("meta_columns", []))
            opts = [{"label": c, "value": c} for c in cols]
            pref_src = DEFAULT_NUM_COLS if kind == "rank" else DEFAULT_CAT_COLS
            pref = present_cols(cols, pref_src) or cols
            triggered = callback_context.triggered_id
            if triggered == "par-enr-kind" or not current:
                value = list(pref)
            else:
                value = [c for c in current if c in cols]
            return opts, value

        @app.callback(
            Output("par-enr-fig", "figure"),
            Output("par-enr-status", "children"),
            Input("par-enr-run", "n_clicks"),
            State("par-enr-cols", "value"),
            State("par-enr-mode", "value"),
            State("par-enr-kind", "value"),
            prevent_initial_call=True,
        )
        def _enr(n_clicks, cols, mode, kind):
            empty = go.Figure()
            if "closest" not in _PAR_RUNTIME:
                return empty, "Run per-level clustering first."
            cols = [c for c in (cols or []) if c]
            if not cols:
                return empty, "Select metadata columns."
            meta = _PAR_RUNTIME["meta"]
            levels = list(_PAR_RUNTIME.get("levels") or [])
            if not levels:
                return empty, "No levels in the last clustering run."
            use = [c for c in cols if c in meta.columns]
            if not use:
                return empty, "Selected columns are not in metadata."
            df = meta.copy()
            df["_closest"] = assign_region_labels(df.index, _PAR_RUNTIME.get("closest") or {}, levels)
            df["_unique"] = assign_region_labels(df.index, _PAR_RUNTIME.get("unique") or {}, levels)
            label = "uniquely-close" if mode == "unique" else "closest"
            try:
                if kind == "rank":
                    target = "_unique" if mode == "unique" else "_closest"
                    mats = corr_vs_region_rank(df, use, [target], levels)
                    for mat in mats.values():
                        mat.columns = [label]
                    fig = heatmap_matrix_fig(
                        [mats["Pearson"], mats["Spearman"]],
                        ["Pearson", "Spearman"],
                        title=_plotly_title(
                            "Rankable metadata vs level rank",
                            f"{label} samples",
                        ),
                        zmin=-1,
                        zmax=1,
                        colorscale="RdBu_r",
                    )
                    n = int(df[target].notna().sum())
                    return fig, f"{len(use)} rankable columns vs {label} level rank ({n} labeled samples)."
                region_col = "_unique" if mode == "unique" else "_closest"
                fig = entry_fractions_fig(
                    df,
                    region_col,
                    use,
                    title=_plotly_title(
                        "Categorical entry fractions",
                        f"within {label} samples",
                    ),
                    levels=levels,
                )
                n = int(df[region_col].notna().sum())
                return fig, f"{n} samples labeled as {label}."
            except Exception as exc:  # noqa: BLE001
                return empty, f"Enrichment error: {exc}"

        @app.callback(
            Output("par-grad-cache", "data"),
            Output("par-grad-status", "children"),
            Output("par-grad-genes", "value", allow_duplicate=True),
            Output("par-grad-selected", "data", allow_duplicate=True),
            Input("par-grad-run", "n_clicks"),
            State("par-grad-mode", "value"),
            prevent_initial_call=True,
        )
        def _grads(n_clicks, mode):
            if "closest" not in _PAR_RUNTIME:
                return no_update, "Run per-level clustering first.", no_update, no_update
            X = _PAR_RUNTIME["X"]
            levels = _PAR_RUNTIME["levels"]
            if mode == "unique":
                by_level = _PAR_RUNTIME.get("unique") or {}
                label = "uniquely-close"
            else:
                by_level = _PAR_RUNTIME.get("closest") or {}
                label = "closest"
            order_col = _PAR_RUNTIME.get("order_col") or "region"
            samples = []
            labels = []
            for region in levels:
                for sid in by_level.get(region, ()):
                    if sid in X.index:
                        samples.append(sid)
                        labels.append(region)
            if len(samples) < 3:
                return no_update, f"Fewer than 3 {label} samples.", no_update, no_update
            expr = X.loc[samples].T
            meta_close = pd.DataFrame({order_col: labels}, index=samples)
            is_bio = _PAR_RUNTIME["is_bio"].reindex(X.index).fillna(False)
            meta_all = _PAR_RUNTIME["meta"]
            bio_keep = is_bio & meta_all[order_col].astype(str).isin(levels)
            bio_ids = [str(s) for s in meta_all.index[bio_keep] if str(s) in X.index]
            if len(bio_ids) < 3:
                return no_update, "Fewer than 3 level (biofilm) samples.", no_update, no_update
            expr_bio = X.loc[bio_ids].T
            meta_bio = meta_all.loc[bio_ids].copy()
            meta_bio[order_col] = meta_bio[order_col].astype(str)
            try:
                results = gene_region_gradients(
                    expr,
                    meta_close,
                    order_col,
                    levels,
                    ["pearson", "spearman"],
                    pool=True,
                )
                results_bio = gene_region_gradients(
                    expr_bio,
                    meta_bio,
                    order_col,
                    levels,
                    ["pearson", "spearman"],
                    pool=True,
                )
            except Exception as exc:  # noqa: BLE001
                return no_update, f"Gradient error: {exc}", no_update, no_update
            bio = results_bio.set_index("geneID")
            results["pearson_rho_biofilm"] = results["geneID"].map(bio["pearson_rho"])
            results["spearman_rho_biofilm"] = results["geneID"].map(bio["spearman_rho"])
            _PAR_RUNTIME["grad_results"] = results
            _PAR_RUNTIME["grad_expr"] = expr
            _PAR_RUNTIME["grad_meta"] = meta_close
            _PAR_RUNTIME["grad_expr_bio"] = expr_bio
            _PAR_RUNTIME["grad_meta_bio"] = meta_bio
            _PAR_RUNTIME["grad_mode"] = label
            return (
                {"n": len(samples), "mode": label, "n_genes": int(len(results))},
                f"Pooled gradients on {len(samples)} {label} samples.",
                [],
                [],
            )

        @app.callback(
            Output("par-grad-genes", "options"),
            Input("par-grad-cache", "data"),
            Input("par-grad-genes", "search_value"),
            Input("par-grad-genes", "value"),
            Input("par-grad-selected", "data"),
        )
        def _par_grad_gene_options(cache, search, selected, clicked):
            results = _PAR_RUNTIME.get("grad_results")
            if not cache or results is None:
                return []
            chosen = list(selected or []) + list(clicked or [])
            hits = filter_ids_by_search(results["geneID"].astype(str), search, chosen)
            return [{"label": g, "value": g} for g in hits]

        @app.callback(
            Output("par-grad-pearson", "figure"),
            Output("par-grad-spearman", "figure"),
            Output("par-grad-cmp-pearson", "figure"),
            Output("par-grad-cmp-spearman", "figure"),
            Input("par-grad-cache", "data"),
            Input("par-grad-rho-p", "value"),
            Input("par-grad-dr-p", "value"),
            Input("par-grad-rho-s", "value"),
            Input("par-grad-dr-s", "value"),
            Input("par-grad-selected", "data"),
        )
        def _replot_par_grads(cache, rho_p, dr_p, rho_s, dr_s, selected):
            empty = go.Figure()
            results = _PAR_RUNTIME.get("grad_results")
            if not cache or results is None:
                return empty, empty, empty, empty
            mark = {str(g) for g in (selected or []) if g}
            mode = cache.get("mode") or "closest"
            pearson = gradient_scatter_fig(
                results,
                "pearson",
                rho_thr=float(rho_p if rho_p is not None else 0.7),
                dr_thr=float(dr_p if dr_p is not None else 1.0),
                title=_plotly_title("Pearson gene gradients", f"{mode} samples"),
                mark_genes=mark,
                mark_label="selected" if mark else None,
            )
            spearman = gradient_scatter_fig(
                results,
                "spearman",
                rho_thr=float(rho_s if rho_s is not None else 0.7),
                dr_thr=float(dr_s if dr_s is not None else 1.0),
                title=_plotly_title("Spearman gene gradients", f"{mode} samples"),
                mark_genes=mark,
                mark_label="selected" if mark else None,
            )
            cmp_p = biofilm_vs_close_fig(
                results, "pearson", mark_genes=mark, mark_label="selected" if mark else None
            )
            cmp_s = biofilm_vs_close_fig(
                results, "spearman", mark_genes=mark, mark_label="selected" if mark else None
            )
            return pearson, spearman, cmp_p, cmp_s

        @app.callback(
            Output("par-grad-selected", "data"),
            Output("par-grad-genes", "value"),
            Output("par-grad-profiles-status", "children"),
            Input("par-grad-pearson", "clickData"),
            Input("par-grad-spearman", "clickData"),
            Input("par-grad-cmp-pearson", "clickData"),
            Input("par-grad-cmp-spearman", "clickData"),
            Input("par-grad-clear", "n_clicks"),
            Input("par-grad-genes", "value"),
            State("par-grad-selected", "data"),
            prevent_initial_call=True,
        )
        def _select_par_genes(
            click_p, click_s, click_cp, click_cs, n_clear, dropdown, selected
        ):
            triggered = callback_context.triggered_id
            selected = list(selected or [])
            if triggered == "par-grad-clear":
                return [], [], "Cleared selected genes."
            if triggered == "par-grad-genes":
                genes = [str(g) for g in (dropdown or [])][:_MAX_PROFILE_GENES]
                return genes, no_update, f"Profiles {len(genes)}/{_MAX_PROFILE_GENES}."
            click_map = {
                "par-grad-pearson": click_p,
                "par-grad-spearman": click_s,
                "par-grad-cmp-pearson": click_cp,
                "par-grad-cmp-spearman": click_cs,
            }
            gene = _click_gene_id(click_map.get(triggered)) if isinstance(triggered, str) else None
            if not gene:
                return no_update, no_update, no_update
            if gene in selected:
                selected = [g for g in selected if g != gene]
                msg = f"Removed {gene} ({len(selected)}/{_MAX_PROFILE_GENES})."
            elif len(selected) >= _MAX_PROFILE_GENES:
                return (
                    no_update,
                    no_update,
                    f"Already at max {_MAX_PROFILE_GENES} profiles — remove one or Clear.",
                )
            else:
                selected = selected + [gene]
                msg = f"Profiles {len(selected)}/{_MAX_PROFILE_GENES}."
            return selected, selected, msg

        @app.callback(
            Output("par-grad-profiles", "figure"),
            Input("par-grad-selected", "data"),
            Input("par-grad-cache", "data"),
            Input("par-grad-rep-col", "value"),
        )
        def _par_profiles(selected, cache, rep_col):
            expr = _PAR_RUNTIME.get("grad_expr")
            meta = _PAR_RUNTIME.get("grad_meta")
            expr_bio = _PAR_RUNTIME.get("grad_expr_bio")
            meta_bio = _PAR_RUNTIME.get("grad_meta_bio")
            levels = _PAR_RUNTIME.get("levels") or []
            order_col = _PAR_RUNTIME.get("order_col") or "region"
            results = _PAR_RUNTIME.get("grad_results")
            if not cache or expr is None or meta is None or expr_bio is None or meta_bio is None:
                return gene_profile_grid_fig(
                    [],
                    expr=pd.DataFrame(),
                    meta=pd.DataFrame(),
                    order_col=order_col,
                    order_levels=[],
                    rep_col=None,
                )
            return gene_profile_biofilm_and_close_fig(
                list(selected or []),
                expr_bio=expr_bio,
                meta_bio=meta_bio,
                expr_close=expr,
                meta_close=meta,
                order_col=order_col,
                order_levels=list(levels),
                rep_col=rep_col if rep_col in meta_bio.columns else None,
                results=results,
            )

        @app.callback(
            Output("par-grad-celov-out", "value"),
            Output("par-grad-celov-status", "children", allow_duplicate=True),
            Input("par-grad-celov-browse", "n_clicks"),
            State("par-grad-celov-out", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_par_grad_celov(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save pooled gradient Celov (use {} for type)",
                defaultextension=".txt",
                initialfile="ParGeneGradient_{}.txt",
            )
            if not chosen:
                return no_update, "Celov path browse cancelled."
            p = Path(chosen)
            if "{}" not in p.name:
                chosen = str(p.with_name(f"{p.stem}_{{}}{p.suffix or '.txt'}"))
            return chosen, f"Celov output template: {chosen}"

        @app.callback(
            Output("par-grad-celov-status", "children"),
            Input("par-grad-celov-save", "n_clicks"),
            State("par-grad-celov-out", "value"),
            State("par-grad-celov-mode", "value"),
            State("par-grad-celov-score", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _save_par_grad_celov(n_clicks, out_path, mode, score_col, session_blob, project_blob):
            results = _PAR_RUNTIME.get("grad_results")
            if results is None or not isinstance(results, pd.DataFrame) or results.empty:
                return "Run pooled gradients first."
            if not out_path or not str(out_path).strip():
                return "Choose an output .txt path (Browse)."
            score_col = score_col or "pearson_rho"
            if score_col not in results.columns:
                return f"Score column {score_col!r} was not computed — re-run gradients."
            try:
                entry = _active_dataset_entry(session_blob, project_blob)
                lookup = None
                locus = _locus_path_from_session(session_blob, project_blob)
                if locus:
                    lookup = load_locus_lookup(locus)
                paths = save_gradient_celov(
                    results,
                    score_col,
                    str(out_path).strip(),
                    mode=mode or "up_and_down",
                    locus_lookup=lookup,
                    id_column=resolve_celov_id_col(entry),
                )
                return "Saved Celov: " + ", ".join(str(p) for p in paths)
            except Exception as exc:  # noqa: BLE001
                return f"Celov save error: {exc}"
