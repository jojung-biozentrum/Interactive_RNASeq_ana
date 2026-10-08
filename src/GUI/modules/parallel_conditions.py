"""Parallel conditions — per-level Ward + closest LC (Parallel_conditions.ipynb)."""

from __future__ import annotations

import math
from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

from src.biocyc.celov_multiomics_post import load_locus_lookup
from ..components.controls import (
    SHAPE_CONSTANTS,
    apply_export_layout,
    equal_xy_axes,
    graph_export_config,
    _LEGEND_GREY,
    _LEGEND_STD_SIZE,
    _MISSING_COLOR,
    _MISSING_LABEL,
    _MISSING_SYMBOL,
    _color_map,
    _has_missing,
    _is_missing,
    _legend_dummy,
    _shape_map,
    _size_map,
)
from ..components.sample_detail import (
    lookup_sample,
    sample_detail_placeholder,
    sample_detail_table,
)
from ..components.gene_meta_mark import (
    gene_hover_text,
    gene_meta_hover_map,
    gene_pick_controls,
    gene_search_options,
    merge_gene_selection,
    parse_gene_ids,
)
from ..data_store import session_from_store
from .clustering import (
    _locus_path_from_session,
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
)

_PAR_RUNTIME: dict = {}
_PAR_CACHE_N_PCS = 20


def _plotly_title(*lines: str) -> dict:
    text = "<br>".join(x for x in lines if x is not None and str(x).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


def table_html(df: pd.DataFrame, *, max_rows: int = 40):
    if df is None or df.empty:
        return html.P("No rows.", className="text-muted small mb-0")
    show = df.head(max_rows)
    header = [html.Th(str(c), className="text-nowrap") for c in show.columns]
    body = []
    for _, row in show.iterrows():
        cells = []
        for c in show.columns:
            val = row[c]
            if isinstance(val, float) and np.isfinite(val):
                text = f"{val:.4g}"
            else:
                text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
            cells.append(html.Td(text, className="text-nowrap"))
        body.append(html.Tr(cells))
    return html.Div(
        dbc.Table([html.Thead(html.Tr(header)), html.Tbody(body)], bordered=True, striped=True, size="sm", className="mb-0"),
        className="overflow-auto",
        style={"maxHeight": "420px"},
    )

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


def _symbol_for(symbol: str, is_3d: bool) -> str:
    symbol = str(symbol or "circle")
    if not is_3d:
        return symbol
    if symbol in _SYMBOL_3D_OK:
        return symbol
    return _SYMBOL_3D.get(symbol, "circle")


def _fit_par_pca(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    arr = np.asarray(X, dtype=float)
    n_pcs = max(2, min(_PAR_CACHE_N_PCS, int(arr.shape[0]) - 1, int(arr.shape[1])))
    scores = PCA(n_components=n_pcs).fit_transform(arr)
    cols = [f"PC{i + 1}" for i in range(n_pcs)]
    score_df = pd.DataFrame(scores, index=X.index.astype(str), columns=cols)
    if meta is not None:
        score_df = score_df.join(meta.reindex(score_df.index))
    return score_df


def _pack_score_df(score_df: pd.DataFrame) -> dict:
    return score_df.reset_index(names="_sample_id").to_dict(orient="list")


def _unpack_score_df(cache: dict | None) -> pd.DataFrame | None:
    if not cache or "scores" not in cache:
        return None
    df = pd.DataFrame(cache["scores"])
    if "_sample_id" in df.columns:
        df = df.set_index("_sample_id")
    df.index = df.index.astype(str)
    return df


def _pack_level_detail(level_detail: dict) -> dict:
    out = {}
    for level, info in (level_detail or {}).items():
        labels = info.get("labels")
        if isinstance(labels, pd.Series) and len(labels):
            lab_ids = list(labels.index.astype(str))
            lab_vals = [int(v) for v in labels.to_numpy()]
        else:
            lab_ids, lab_vals = [], []
        out[str(level)] = {
            "bio_ids": list(map(str, info.get("bio_ids") or [])),
            "label_ids": lab_ids,
            "label_vals": lab_vals,
            "closest_c": info.get("closest_c"),
        }
    return out


def _unpack_level_detail(blob: dict | None) -> dict:
    out = {}
    for level, info in (blob or {}).items():
        ids = list(info.get("label_ids") or [])
        vals = list(info.get("label_vals") or [])
        labels = (
            pd.Series(vals, index=pd.Index(ids, dtype=object), dtype=int)
            if ids
            else pd.Series(dtype=int)
        )
        out[str(level)] = {
            "bio_ids": set(map(str, info.get("bio_ids") or [])),
            "labels": labels,
            "closest_c": info.get("closest_c"),
        }
    return out


def _closest_from_cache(cache: dict | None) -> dict[str, set[str]]:
    return {str(k): set(map(str, v or [])) for k, v in ((cache or {}).get("closest") or {}).items()}


def _hydrate_par_from_cache(cache: dict | None, session_blob=None) -> bool:
    """Rebuild ``_PAR_RUNTIME`` from ``par-cache`` (+ session) for multi-worker use."""
    if not cache:
        return False
    if isinstance(_PAR_RUNTIME.get("score_df"), pd.DataFrame) and _PAR_RUNTIME.get("closest"):
        return True
    score_df = _unpack_score_df(cache)
    if score_df is None:
        return False
    levels = list(cache.get("levels") or [])
    order_col = cache.get("order_col")
    closest = _closest_from_cache(cache)
    level_detail = _unpack_level_detail(cache.get("level_detail"))
    X = meta = is_bio = None
    if session_blob:
        session = session_from_store(session_blob)
        if session.ready and session.expression is not None:
            X = session.expression.copy()
            X.index = X.index.astype(str)
            meta = session.metadata.reindex(X.index) if session.metadata is not None else None
            sub_col = cache.get("subset_col")
            sub_val = cache.get("subset_val")
            if meta is not None and sub_col and sub_val is not None and sub_col in meta.columns:
                is_bio = _subset_mask(meta, sub_col, str(sub_val)).reindex(X.index).fillna(False)
    _PAR_RUNTIME.update(
        {
            "closest": closest,
            "level_detail": level_detail,
            "score_df": score_df,
            "levels": levels,
            "order_col": order_col,
            "X": X if X is not None else _PAR_RUNTIME.get("X"),
            "meta": meta if meta is not None else _PAR_RUNTIME.get("meta"),
            "is_bio": is_bio if is_bio is not None else _PAR_RUNTIME.get("is_bio"),
        }
    )
    return True


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


def _closest_aes_maps(
    score_df: pd.DataFrame,
    closest_ids: set[str],
    color_col: str | None,
    shape_col: str | None,
    size_col: str | None,
) -> dict:
    """Color/shape/size maps from all closest samples (shared across level panels)."""
    keep = [i for i in score_df.index if str(i) in closest_ids]
    sub = score_df.loc[keep] if keep else score_df.iloc[0:0]
    out: dict = {
        "color_col": None,
        "shape_col": None,
        "size_col": None,
        "color_map": {},
        "shape_map": {},
        "size_map": {},
        "color_missing": False,
        "shape_missing": False,
        "size_missing": False,
    }
    if color_col and color_col in sub.columns and len(sub):
        out["color_col"] = color_col
        out["color_map"] = _color_map(sub[color_col])
        out["color_missing"] = _has_missing(sub[color_col])
    if shape_col and shape_col in sub.columns and len(sub):
        out["shape_col"] = shape_col
        out["shape_map"] = _shape_map(sub[shape_col])
        out["shape_missing"] = _has_missing(sub[shape_col])
    if size_col and size_col in sub.columns and len(sub):
        out["size_col"] = size_col
        out["size_map"] = _size_map(sub[size_col], default=12.0)
        out["size_missing"] = _has_missing(sub[size_col])
    return out


def _aes_point_style(
    sub: pd.DataFrame,
    aes: dict,
    *,
    is_3d: bool,
) -> tuple[list, list, list]:
    n = len(sub)
    color_col = aes.get("color_col")
    shape_col = aes.get("shape_col")
    size_col = aes.get("size_col")
    cmap = aes.get("color_map") or {}
    smap = aes.get("shape_map") or {}
    zmap = aes.get("size_map") or {}
    if color_col and color_col in sub.columns:
        colors = [
            _MISSING_COLOR if _is_missing(v) else cmap.get(str(v), "#d62728")
            for v in sub[color_col]
        ]
    else:
        colors = ["crimson"] * n
    if shape_col and shape_col in sub.columns:
        symbols = [
            _symbol_for(
                _MISSING_SYMBOL if _is_missing(v) else smap.get(str(v), "circle"),
                is_3d,
            )
            for v in sub[shape_col]
        ]
    else:
        symbols = [_symbol_for("circle", is_3d)] * n
    if size_col and size_col in sub.columns:
        sizes = [
            12.0 if _is_missing(v) else float(zmap.get(str(v), 12.0))
            for v in sub[size_col]
        ]
    else:
        sizes = [12.0] * n
    return colors, symbols, sizes


def _add_closest_aes_legend(fig: go.Figure, aes: dict, *, is_3d: bool) -> None:
    """Seaborn-style color / shape / size legend groups (same maps on every panel)."""
    color_col = aes.get("color_col")
    shape_col = aes.get("shape_col")
    size_col = aes.get("size_col")
    if not color_col and not shape_col and not size_col:
        _legend_dummy(
            fig,
            name="closest",
            color="crimson",
            symbol="circle",
            size=_LEGEND_STD_SIZE,
            group="legend-closest",
            group_title="Closest",
            first_in_group=True,
        )
        return
    if color_col:
        cmap = aes.get("color_map") or {}
        first = True
        if aes.get("color_missing"):
            _legend_dummy(
                fig,
                name=_MISSING_LABEL,
                color=_MISSING_COLOR,
                symbol="circle",
                size=_LEGEND_STD_SIZE,
                group="legend-color",
                group_title=color_col,
                first_in_group=True,
            )
            first = False
        for i, cat in enumerate(cmap):
            _legend_dummy(
                fig,
                name=cat,
                color=cmap[cat],
                symbol="circle",
                size=_LEGEND_STD_SIZE,
                group="legend-color",
                group_title=color_col,
                first_in_group=first and i == 0,
            )
    if shape_col:
        smap = aes.get("shape_map") or {}
        first = True
        if aes.get("shape_missing"):
            _legend_dummy(
                fig,
                name=_MISSING_LABEL,
                color=_LEGEND_GREY,
                symbol=_MISSING_SYMBOL,
                size=_LEGEND_STD_SIZE,
                group="legend-shape",
                group_title=shape_col,
                first_in_group=True,
            )
            first = False
        for i, cat in enumerate(smap):
            _legend_dummy(
                fig,
                name=cat,
                color=_LEGEND_GREY,
                symbol=_symbol_for(smap[cat], is_3d),
                size=_LEGEND_STD_SIZE,
                group="legend-shape",
                group_title=shape_col,
                first_in_group=first and i == 0,
            )
    if size_col:
        zmap = aes.get("size_map") or {}
        first = True
        if aes.get("size_missing"):
            _legend_dummy(
                fig,
                name=_MISSING_LABEL,
                color=_LEGEND_GREY,
                symbol="circle",
                size=_LEGEND_STD_SIZE,
                group="legend-size",
                group_title=size_col,
                first_in_group=True,
            )
            first = False
        cats = sorted(zmap.keys(), key=lambda c: (zmap[c], str(c)))
        for i, cat in enumerate(cats):
            _legend_dummy(
                fig,
                name=cat,
                color=_LEGEND_GREY,
                symbol="circle",
                size=float(zmap[cat]),
                group="legend-size",
                group_title=size_col,
                first_in_group=first and i == 0,
            )


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
    color_col: str | None = None,
    shape_col: str | None = None,
    size_col: str | None = None,
    aes_maps: dict | None = None,
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
    aes = aes_maps or _closest_aes_maps(
        score_df, set(highlight), color_col, shape_col, size_col
    )
    has_aes = bool(aes.get("color_col") or aes.get("shape_col") or aes.get("size_col"))
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
        opacity=0.2,
        name="rest",
    )
    if tr is not None:
        tr.showlegend = False
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
            tr.showlegend = not has_aes
            tr.legendgroup = "bio"
            if i == 0 and not has_aes:
                tr.legendgrouptitle = {"text": level}
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
            tr.showlegend = not has_aes
            fig.add_trace(tr)
    if highlight:
        sub = score_df.loc[[i for i in score_df.index.astype(str) if i in set(highlight)]]
        if sub.empty:
            keep = [i for i in score_df.index if str(i) in set(highlight)]
            sub = score_df.loc[keep]
        colors, symbols, sizes = _aes_point_style(sub, aes, is_3d=bool(z_col))
        idx = sub.index.astype(str)
        text = idx
        if "fileName" in sub.columns:
            text = sub["fileName"].astype(str)
        common = dict(
            x=sub[x_col],
            y=sub[y_col],
            mode="markers",
            name=highlight_name,
            text=text,
            customdata=idx,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            marker=dict(color=colors, symbol=symbols, size=sizes, opacity=1.0),
        )
        if z_col and z_col in sub.columns:
            fig.add_trace(go.Scatter3d(z=sub[z_col], **common))
        else:
            fig.add_trace(go.Scatter(**common))
    if not z_col:
        _add_closest_aes_legend(fig, aes, is_3d=False)
    if isinstance(title, dict):
        fig.update_layout(title=title)
        title_lines = str(title.get("text", "")).count("<br>") + 1
    else:
        fig.update_layout(title=_plotly_title(str(title)))
        title_lines = 1
    if not z_col:
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
    leg_kw = None if has_aes else {"title_text": level}
    apply_export_layout(
        fig,
        title_lines=title_lines,
        legend=True,
        legend_kwargs=leg_kw,
        uirevision=f"par-pca-{level}",
    )
    return fig


def biofilm_vs_close_fig(
    results: pd.DataFrame,
    method: str,
    *,
    mark_genes: set[str] | None = None,
    mark_label: str | None = None,
    hover_map: dict[str, str] | None = None,
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
                text=gene_hover_text(base["geneID"], hover_map),
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
                text=gene_hover_text(sub["geneID"], hover_map),
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
    lookup: pd.DataFrame | None = None,
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
    if lookup is None:
        lookup = _PAR_RUNTIME.get("locus_lookup")
    titles = [_display_name_for_gene(results, g, lookup) for g in genes]
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
    return summary, closest, level_detail


class ParallelConditionsModule:
    id = "parallel-conditions"
    label = "Parallel conditions"

    def layout(self):
        return html.Div([self._samples(), self._genes()])

    def _samples(self):
        return html.Div(
            [
                html.P(
                    "Pick the biofilm subset, the level column, and the level order. "
                    "Each level is clustered with all non-subset samples.",
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
                    "Closest LC cluster samples α=1; other points α=0.2. Two panels per row.",
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
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Color"),
                                dcc.Dropdown(id="par-pca-color-col", clearable=True),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Shape"),
                                dcc.Dropdown(id="par-pca-shape-col", clearable=True),
                            ],
                            md=4,
                        ),
                        dbc.Col(
                            [
                                html.Label("Size"),
                                dcc.Dropdown(id="par-pca-size-col", clearable=True),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Div(id="par-pca-panels"), md=8),
                        dbc.Col(
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
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dcc.Store(id="par-cache"),
            ]
        )

    def _genes(self):
        return html.Div(
            [
                html.H6("Pooled gene gradients"),
                html.P(
                    "Run on closest samples, then search / paste / click genes for "
                    "marking + profiles. Pearson / Spearman vs dynamic range, and "
                    "levels (x) vs closest (y).",
                    className="text-muted small",
                ),
                dbc.Button(
                    "Run pooled gradients",
                    id="par-grad-run",
                    color="primary",
                    className="mb-2",
                ),
                html.Div(id="par-grad-status", className="text-muted small mb-2"),
                gene_pick_controls(
                    search_id="par-grad-genes",
                    clear_id="par-grad-clear",
                    paste_id="par-grad-gene-paste",
                    paste_btn_id="par-grad-gene-paste-add",
                    label="Search / pick genes",
                    help_text=(
                        "Search, paste PCA geneIDs, or click plots "
                        f"(max {_MAX_PROFILE_GENES} for profiles)."
                    ),
                ),
                html.Div(id="par-grad-profiles-status", className="text-muted small mb-1"),
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
                dcc.Store(id="par-grad-cache"),
                dcc.Store(id="par-grad-selected", data=[]),
                html.Hr(),
                html.H6("Selected gene profiles"),
                html.P(
                    "Closest samples are grey; closest mean is black with std bars. "
                    "Replicate column is optional — if set, biofilm lines are split by "
                    "that column.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label(
                                    "Replicate column for subset of interest (optional)"
                                ),
                                dcc.Dropdown(id="par-grad-rep-col", clearable=True),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dcc.Graph(
                    id="par-grad-profiles",
                    figure={},
                    config=_PAR_GRAD_CONFIG,
                ),
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
                "TechnicalReplicate"
                if "TechnicalReplicate" in cols
                else (
                    "Replicate"
                    if "Replicate" in cols
                    else ("replicate" if "replicate" in cols else None)
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
            Output("par-cache", "data", allow_duplicate=True),
            Output("par-status", "children"),
            Output("par-summary", "children"),
            Input("par-run", "n_clicks"),
            State("session-store", "data"),
            State("project-store", "data"),
            State("par-subset-col", "value"),
            State("par-subset-val", "value"),
            State("par-order-col", "value"),
            State("par-levels", "value"),
            prevent_initial_call=True,
        )
        def _run(n_clicks, session_blob, project_blob, sub_col, sub_val, order_col, levels):
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
                summary, closest, level_detail = run_per_level_clusters(
                    X, meta, is_bio, order_col, levels
                )
                score_df = _fit_par_pca(X, meta)
            except Exception as exc:  # noqa: BLE001
                return no_update, f"Clustering error: {exc}", no_update
            _PAR_RUNTIME.clear()
            lookup = None
            locus = _locus_path_from_session(session_blob, project_blob)
            if locus:
                try:
                    lookup = load_locus_lookup(locus)
                except Exception:  # noqa: BLE001
                    lookup = None
            _PAR_RUNTIME.update(
                {
                    "summary": summary,
                    "closest": closest,
                    "level_detail": level_detail,
                    "score_df": score_df,
                    "meta": meta,
                    "X": X,
                    "is_bio": is_bio,
                    "levels": levels,
                    "order_col": order_col,
                    "locus_lookup": lookup,
                    "subset_col": sub_col,
                    "subset_val": str(sub_val),
                }
            )
            n_pcs = sum(
                1
                for c in score_df.columns
                if str(c).startswith("PC") and str(c)[2:].isdigit()
            )
            cache = {
                "n_regions": len(levels),
                "n_pcs": n_pcs,
                "levels": levels,
                "order_col": order_col,
                "subset_col": sub_col,
                "subset_val": str(sub_val),
                "closest": {k: list(v) for k, v in closest.items()},
                "level_detail": _pack_level_detail(level_detail),
                "scores": _pack_score_df(score_df),
            }
            return (
                cache,
                f"Per-level Ward done for {len(levels)} levels.",
                table_html(summary),
            )

        @app.callback(
            Output("par-pca-color-col", "options"),
            Output("par-pca-shape-col", "options"),
            Output("par-pca-size-col", "options"),
            Input("par-cache", "data"),
            Input("session-store", "data"),
        )
        def _fill_par_aes_cols(cache, session_blob):
            cols = list((session_blob or {}).get("meta_columns", []))
            opts = [{"label": c, "value": c} for c in cols]
            return opts, opts, opts

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
            if not cache:
                return [], [], [], None, None, None
            score_df = _PAR_RUNTIME.get("score_df")
            if not isinstance(score_df, pd.DataFrame):
                score_df = _unpack_score_df(cache)
            if score_df is None:
                n = int(cache.get("n_pcs") or 0)
                pcs = [f"PC{i}" for i in range(1, max(n, 0) + 1)]
            else:
                pcs = [
                    c
                    for c in score_df.columns
                    if str(c).startswith("PC") and str(c)[2:].isdigit()
                ]
            if not pcs:
                return [], [], [], None, None, None
            opts = [{"label": c, "value": c} for c in pcs]
            return (
                opts,
                opts,
                opts,
                "PC1" if "PC1" in pcs else pcs[0],
                "PC2" if "PC2" in pcs else (pcs[1] if len(pcs) > 1 else pcs[0]),
                None,
            )

        @app.callback(
            Output("par-pca-panels", "children"),
            Input("par-cache", "data"),
            Input("par-pca-color-col", "value"),
            Input("par-pca-shape-col", "value"),
            Input("par-pca-size-col", "value"),
            Input("par-pca-x", "value"),
            Input("par-pca-y", "value"),
            Input("par-pca-z", "value"),
            State("session-store", "data"),
        )
        def _plot_par_pca(
            cache,
            color_col,
            shape_col,
            size_col,
            x_col,
            y_col,
            z_col,
            session_blob,
        ):
            if not cache or not _hydrate_par_from_cache(cache, session_blob):
                return html.P(
                    "Run per-level clustering to plot PCA.",
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
            hl_map = _PAR_RUNTIME.get("closest") or {}
            detail = _PAR_RUNTIME.get("level_detail") or {}
            levels = list(_PAR_RUNTIME.get("levels") or [])
            all_closest: set[str] = set()
            for level in levels:
                all_closest |= set(hl_map.get(level) or [])
            aes_maps = _closest_aes_maps(
                score_df, all_closest, color_col, shape_col, size_col
            )
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
                    highlight_name="closest",
                    x_col=x_col,
                    y_col=y_col,
                    z_col=z_col,
                    color_col=color_col,
                    shape_col=shape_col,
                    size_col=size_col,
                    aes_maps=aes_maps,
                    title=_plotly_title(
                        f"PCA · {level}",
                        f"closest n={len(hl_map.get(level) or [])}",
                    ),
                )
                panels.append(
                    dcc.Graph(
                        id={"type": "par-pca-fig", "level": level},
                        figure=fig,
                        config=graph_export_config(f"par_pca_{level}"),
                    )
                )
            if not panels:
                return html.P("No levels to plot.", className="text-muted small mb-0")
            rows = []
            for i in range(0, len(panels), 2):
                chunk = [dbc.Col(p, md=6) for p in panels[i : i + 2]]
                rows.append(dbc.Row(chunk, className="g-2 mb-2"))
            return rows

        @app.callback(
            Output("par-pca-sample-detail", "children"),
            Input({"type": "par-pca-fig", "level": ALL}, "clickData"),
            Input("par-cache", "data"),
            State("session-store", "data"),
        )
        def _par_pca_sample(clicks, cache, session_blob):
            if not cache or not _hydrate_par_from_cache(cache, session_blob):
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
            Output("par-grad-cache", "data", allow_duplicate=True),
            Output("par-grad-status", "children"),
            Output("par-grad-genes", "value", allow_duplicate=True),
            Output("par-grad-selected", "data", allow_duplicate=True),
            Input("par-grad-run", "n_clicks"),
            State("par-cache", "data"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _grads(n_clicks, par_cache, session_blob, project_blob):
            if not par_cache or not _hydrate_par_from_cache(par_cache, session_blob):
                return no_update, "Run per-level clustering first.", no_update, no_update
            X = _PAR_RUNTIME.get("X")
            meta_all = _PAR_RUNTIME.get("meta")
            is_bio = _PAR_RUNTIME.get("is_bio")
            if X is None or meta_all is None or is_bio is None:
                return no_update, "Run per-level clustering first.", no_update, no_update
            levels = list(_PAR_RUNTIME.get("levels") or [])
            by_level = _PAR_RUNTIME.get("closest") or {}
            order_col = _PAR_RUNTIME.get("order_col") or "region"
            samples = []
            labels = []
            for region in levels:
                for sid in by_level.get(region, ()):
                    if sid in X.index:
                        samples.append(sid)
                        labels.append(region)
            if len(samples) < 3:
                return no_update, "Fewer than 3 closest samples.", no_update, no_update
            expr = X.loc[samples].T
            meta_close = pd.DataFrame({order_col: labels}, index=samples)
            is_bio = is_bio.reindex(X.index).fillna(False)
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
                )
                results_bio = gene_region_gradients(
                    expr_bio,
                    meta_bio,
                    order_col,
                    levels,
                    ["pearson", "spearman"],
                )
            except Exception as exc:  # noqa: BLE001
                return no_update, f"Gradient error: {exc}", no_update, no_update
            bio = results_bio.set_index("geneID")
            results["pearson_rho_biofilm"] = results["geneID"].map(bio["pearson_rho"])
            results["spearman_rho_biofilm"] = results["geneID"].map(bio["spearman_rho"])
            lookup = _PAR_RUNTIME.get("locus_lookup")
            if lookup is None:
                locus = _locus_path_from_session(session_blob, project_blob)
                if locus:
                    try:
                        lookup = load_locus_lookup(locus)
                    except Exception:  # noqa: BLE001
                        lookup = None
                _PAR_RUNTIME["locus_lookup"] = lookup
            _PAR_RUNTIME["grad_results"] = results
            _PAR_RUNTIME["grad_expr"] = expr
            _PAR_RUNTIME["grad_meta"] = meta_close
            _PAR_RUNTIME["grad_expr_bio"] = expr_bio
            _PAR_RUNTIME["grad_meta_bio"] = meta_bio
            _PAR_RUNTIME["grad_mode"] = "closest"
            return (
                {
                    "n": len(samples),
                    "mode": "closest",
                    "n_genes": int(len(results)),
                    "results": results.to_dict(orient="list"),
                    "levels": levels,
                    "order_col": order_col,
                    "close_sample_ids": samples,
                    "close_labels": labels,
                    "bio_ids": bio_ids,
                },
                f"Pooled gradients on {len(samples)} closest samples.",
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
            if results is None and cache and cache.get("results"):
                results = pd.DataFrame(cache["results"])
            if not cache or results is None or results.empty:
                return []
            chosen = list(selected or []) + list(clicked or [])
            return gene_search_options(
                results["geneID"].astype(str),
                search,
                chosen,
                lookup=_PAR_RUNTIME.get("locus_lookup"),
            )

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
            State("session-store", "data"),
            State("project-store", "data"),
        )
        def _replot_par_grads(
            cache, rho_p, dr_p, rho_s, dr_s, selected, session_blob, project_blob
        ):
            empty = go.Figure()
            results = _PAR_RUNTIME.get("grad_results")
            if results is None and cache and cache.get("results"):
                results = pd.DataFrame(cache["results"])
                _PAR_RUNTIME["grad_results"] = results
            if not cache or results is None or results.empty:
                return empty, empty, empty, empty
            mark = {str(g) for g in (selected or []) if g}
            mode = cache.get("mode") or "closest"
            lookup = _PAR_RUNTIME.get("locus_lookup")
            if lookup is None:
                locus = _locus_path_from_session(session_blob, project_blob)
                if locus:
                    try:
                        lookup = load_locus_lookup(locus)
                    except Exception:  # noqa: BLE001
                        lookup = None
            hover_map = gene_meta_hover_map(lookup)
            pearson = gradient_scatter_fig(
                results,
                "pearson",
                rho_thr=float(rho_p if rho_p is not None else 0.7),
                dr_thr=float(dr_p if dr_p is not None else 1.0),
                title=_plotly_title("Pearson gene gradients", f"{mode} samples"),
                mark_genes=mark,
                mark_label="selected" if mark else None,
                hover_map=hover_map,
            )
            spearman = gradient_scatter_fig(
                results,
                "spearman",
                rho_thr=float(rho_s if rho_s is not None else 0.7),
                dr_thr=float(dr_s if dr_s is not None else 1.0),
                title=_plotly_title("Spearman gene gradients", f"{mode} samples"),
                mark_genes=mark,
                mark_label="selected" if mark else None,
                hover_map=hover_map,
            )
            cmp_p = biofilm_vs_close_fig(
                results,
                "pearson",
                mark_genes=mark,
                mark_label="selected" if mark else None,
                hover_map=hover_map,
            )
            cmp_s = biofilm_vs_close_fig(
                results,
                "spearman",
                mark_genes=mark,
                mark_label="selected" if mark else None,
                hover_map=hover_map,
            )
            return pearson, spearman, cmp_p, cmp_s

        @app.callback(
            Output("par-grad-selected", "data"),
            Output("par-grad-genes", "value"),
            Output("par-grad-gene-paste", "value"),
            Output("par-grad-profiles-status", "children"),
            Input("par-grad-pearson", "clickData"),
            Input("par-grad-spearman", "clickData"),
            Input("par-grad-cmp-pearson", "clickData"),
            Input("par-grad-cmp-spearman", "clickData"),
            Input("par-grad-clear", "n_clicks"),
            Input("par-grad-gene-paste-add", "n_clicks"),
            Input("par-grad-genes", "value"),
            State("par-grad-selected", "data"),
            State("par-grad-gene-paste", "value"),
            prevent_initial_call=True,
        )
        def _select_par_genes(
            click_p,
            click_s,
            click_cp,
            click_cs,
            n_clear,
            n_paste,
            dropdown,
            selected,
            paste_text,
        ):
            triggered = callback_context.triggered_id
            selected = list(selected or [])
            if triggered == "par-grad-clear":
                return [], [], "", "Cleared selected genes."
            if triggered == "par-grad-gene-paste-add":
                results = _PAR_RUNTIME.get("grad_results")
                have = (
                    set(results["geneID"].astype(str))
                    if isinstance(results, pd.DataFrame) and not results.empty
                    else set()
                )
                add = [g for g in parse_gene_ids(paste_text) if not have or g in have]
                genes = merge_gene_selection(
                    selected, add=add, max_n=_MAX_PROFILE_GENES
                )
                return genes, genes, "", f"Profiles {len(genes)}/{_MAX_PROFILE_GENES}."
            if triggered == "par-grad-genes":
                genes = [str(g) for g in (dropdown or [])][:_MAX_PROFILE_GENES]
                return genes, genes, no_update, f"Profiles {len(genes)}/{_MAX_PROFILE_GENES}."
            click_map = {
                "par-grad-pearson": click_p,
                "par-grad-spearman": click_s,
                "par-grad-cmp-pearson": click_cp,
                "par-grad-cmp-spearman": click_cs,
            }
            gene = _click_gene_id(click_map.get(triggered)) if isinstance(triggered, str) else None
            if not gene:
                return no_update, no_update, no_update, no_update
            if gene in selected:
                selected = [g for g in selected if g != gene]
                msg = f"Removed {gene} ({len(selected)}/{_MAX_PROFILE_GENES})."
            elif len(selected) >= _MAX_PROFILE_GENES:
                return (
                    no_update,
                    no_update,
                    no_update,
                    f"Already at max {_MAX_PROFILE_GENES} profiles — remove one or Clear.",
                )
            else:
                selected = selected + [gene]
                msg = f"Profiles {len(selected)}/{_MAX_PROFILE_GENES}."
            return selected, selected, no_update, msg

        @app.callback(
            Output("par-grad-profiles", "figure"),
            Input("par-grad-selected", "data"),
            Input("par-grad-cache", "data"),
            Input("par-grad-rep-col", "value"),
            State("session-store", "data"),
        )
        def _par_profiles(selected, cache, rep_col, session_blob):
            order_col = (cache or {}).get("order_col") or _PAR_RUNTIME.get("order_col") or "region"
            levels = list((cache or {}).get("levels") or _PAR_RUNTIME.get("levels") or [])
            results = _PAR_RUNTIME.get("grad_results")
            if results is None and cache and cache.get("results"):
                results = pd.DataFrame(cache["results"])
            expr = _PAR_RUNTIME.get("grad_expr")
            meta = _PAR_RUNTIME.get("grad_meta")
            expr_bio = _PAR_RUNTIME.get("grad_expr_bio")
            meta_bio = _PAR_RUNTIME.get("grad_meta_bio")
            if (
                (expr is None or meta is None or expr_bio is None or meta_bio is None)
                and cache
                and session_blob
            ):
                session = session_from_store(session_blob)
                if session.ready and session.expression is not None:
                    X = session.expression.copy()
                    X.index = X.index.astype(str)
                    meta_all = session.metadata.reindex(X.index) if session.metadata is not None else None
                    close_ids = list(cache.get("close_sample_ids") or [])
                    close_labs = list(cache.get("close_labels") or [])
                    bio_ids = [s for s in (cache.get("bio_ids") or []) if s in X.index]
                    close_ids = [s for s in close_ids if s in X.index]
                    if close_ids and bio_ids and meta_all is not None:
                        expr = X.loc[close_ids].T
                        meta = pd.DataFrame({order_col: close_labs[: len(close_ids)]}, index=close_ids)
                        expr_bio = X.loc[bio_ids].T
                        meta_bio = meta_all.loc[bio_ids].copy()
                        meta_bio[order_col] = meta_bio[order_col].astype(str)
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
                lookup=_PAR_RUNTIME.get("locus_lookup"),
            )

