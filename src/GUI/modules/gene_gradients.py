"""Gene gradients along ordered metadata levels (distances.ipynb Pearson / Spearman + Kendall)."""

from __future__ import annotations

import math
import re
from pathlib import Path

from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.biocyc.celov_multiomics_post import (
    annotate_gene_table,
    celov_multiomics_file_generation,
    load_locus_lookup,
)
from src.GUI.project import resolve_celov_id_col

from ..components.folder_browser import pick_save_file_dialog
from ..components.gene_meta_mark import (
    add_marked_gene_trace,
    entry_dropdown_options,
    gene_mark_mask,
    genes_with_meta_entry,
    locus_mark_columns,
    mark_controls,
)
from ..components.sample_detail import (
    gene_detail_placeholder,
    gene_detail_table,
)
from ..data_store import session_from_store

_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "gene_gradients"},
    "displaylogo": False,
}

_METHODS = ("pearson", "spearman", "kendall")
_METHOD_LABELS = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "kendall": "Kendall τ",
}
_METHOD_COLS = {
    "pearson": "pearson_rho",
    "spearman": "spearman_rho",
    "kendall": "kendall_tau",
}
_ABS_COLS = {
    "pearson": "abs_pearson_rho",
    "spearman": "abs_spearman_rho",
    "kendall": "abs_kendall_tau",
}

_COLOR_PASS_ONLY = "#111111"
_COLOR_FAIL = "#9e9e9e"
_COLOR_MARK = "#e41a1c"
_REP_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
)

_MAX_PROFILE_GENES = 36  # 6 columns × 6 rows
_PROFILE_COLS = 6
# Fixed panel size (px): first gene = ax[0,0] of a 6-col grid; does not grow with fewer genes
_PROFILE_CELL_H = 200
_PROFILE_CELL_W = 180
_PROFILE_TITLE_H = 36
_PROFILE_LEGEND_ROW_H = 16
_PROFILE_BOTTOM = 50
_PROFILE_LEFT = 55
_PROFILE_RIGHT = 20

_GRAD_RUNTIME: dict = {}


def _plotly_title(*lines: str) -> dict:
    text = "<br>".join(line for line in lines if line is not None and str(line).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


def _active_dataset_name(session_blob) -> str:
    names = (session_blob or {}).get("active_datasets") or []
    return str(names[0]) if names else "dataset"


def _natural_key(value) -> list:
    parts = re.split(r"(\d+)", str(value))
    key: list = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        elif p:
            key.append(p.lower())
    return key


def _natural_sorted(values) -> list[str]:
    return sorted((str(v) for v in values), key=_natural_key)


def _subset_mask(meta: pd.DataFrame, column: str, value: str) -> pd.Series:
    col = meta[column]
    want = str(value).strip()
    if want.lower() in {"true", "false"}:
        want_bool = want.lower() == "true"
        if col.dtype == bool or set(map(bool, col.dropna().unique())).issubset({True, False}):
            try:
                return col.astype(bool) == want_bool
            except Exception:  # noqa: BLE001
                pass
    return col.astype(str) == want


# Correlation measures:
def _correlate(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        rho, _ = stats.pearsonr(x, y)
    elif method == "spearman":
        rho, _ = stats.spearmanr(x, y)
    elif method == "kendall":
        rho, _ = stats.kendalltau(x, y)
    else:
        raise ValueError(method)
    return float(rho) if np.isfinite(rho) else float("nan")


def gene_region_gradients(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    order_col: str,
    order_levels: list[str],
    methods: list[str],
) -> pd.DataFrame:
    """Port of ``gene_region_gradient_{pearson,spearman}`` (genes × samples).

    Correlation of each gene's sample values vs integer ranks of ordered levels.
    Dynamic range = max(region means) − min(region means) over ``order_levels``.
    """
    methods = [m for m in methods if m in _METHODS]
    if not methods:
        raise ValueError("Select at least one correlation measure.")
    if len(order_levels) < 2:
        raise ValueError("Need at least two ordered levels.")

    sample_ids = [s for s in expr.columns.astype(str) if s in meta.index]
    if len(sample_ids) < 3:
        raise ValueError("Fewer than 3 samples after subset / order filter.")
    expr = expr[sample_ids]
    meta = meta.loc[sample_ids].copy()
    meta[order_col] = meta[order_col].astype(str)

    keep = meta[order_col].isin(order_levels)
    expr = expr.loc[:, keep]
    meta = meta.loc[keep]
    if expr.shape[1] < 3:
        raise ValueError("Fewer than 3 samples in ordered levels.")

    rank_map = {level: rank for rank, level in enumerate(order_levels, start=1)} # rank the levels numerically following the custom set order of the levels
    region_rank = meta[order_col].map(rank_map)

    region_means = pd.DataFrame(
        {
            level: expr.loc[:, meta.index[meta[order_col] == level]].mean(axis=1)
            for level in order_levels
            if (meta[order_col] == level).any()
        }
    )
    dynamic_range = region_means.max(axis=1) - region_means.min(axis=1)

    out = pd.DataFrame(
        {
            "geneID": expr.index.astype(str),
            "dynamic_range": dynamic_range.reindex(expr.index).to_numpy(dtype=float),
        }
    )

    for method in methods:
        rhos: list[float] = []
        for gene in expr.index:
            values = expr.loc[gene]
            mask = values.notna() & region_rank.notna()
            if int(mask.sum()) < 3:
                rhos.append(float("nan"))
                continue
            rhos.append(
                _correlate(
                    values[mask].to_numpy(dtype=float),
                    region_rank[mask].to_numpy(dtype=float),
                    method,
                )
            )
        col = _METHOD_COLS[method]
        abs_col = _ABS_COLS[method]
        out[col] = rhos
        out[abs_col] = np.abs(out[col])

    return out


def _sig_mask(results: pd.DataFrame, method: str, rho_thr: float, dr_thr: float) -> pd.Series:
    abs_col = _ABS_COLS[method]
    if abs_col not in results.columns:
        return pd.Series(False, index=results.index)
    return (results[abs_col] > rho_thr) & (results["dynamic_range"] > dr_thr)


def _click_gene_id(click_data) -> str | None:
    if not click_data or not click_data.get("points"):
        return None
    pt = click_data["points"][0]
    raw = pt.get("customdata")
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is not None:
        return str(raw)
    text = pt.get("text")
    return str(text) if text is not None else None


def _display_name_for_gene(results: pd.DataFrame | None, gene: str) -> str:
    if results is None or "geneID" not in results.columns:
        return gene
    hit = results.loc[results["geneID"].astype(str) == gene]
    if hit.empty:
        return gene
    row = hit.iloc[0]
    for col in ("geneName", "gene_name", "old locusTag", "locusTag"):
        if col in hit.columns and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return gene


def gradient_scatter_fig(
    results: pd.DataFrame,
    focus: str,
    *,
    rho_thr: float,
    dr_thr: float,
    title: dict | str,
    mark_genes: set[str] | None = None,
    mark_label: str | None = None,
) -> go.Figure:
    """Notebook-style scatter: grey below threshold, black above (this measure only)."""
    y_col = _METHOD_COLS[focus]
    if y_col not in results.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"{focus} not computed", showarrow=False)
        return fig

    selected = _sig_mask(results, focus, rho_thr, dr_thr)
    marked = gene_mark_mask(results, mark_genes)
    fig = go.Figure()
    for mask, color, alpha, size in (
        (~selected & ~marked, _COLOR_FAIL, 0.45, 6),
        (selected & ~marked, _COLOR_PASS_ONLY, 1.0, 8),
    ):
        if not mask.any():
            continue
        sub = results.loc[mask]
        fig.add_trace(
            go.Scatter(
                x=sub["dynamic_range"],
                y=sub[y_col],
                mode="markers",
                marker=dict(size=size, color=color, opacity=alpha),
                customdata=sub["geneID"].astype(str),
                text=sub["geneID"].astype(str),
                hovertemplate=(
                    "%{text}<br>dynamic range=%{x:.3g}<br>"
                    f"{_METHOD_LABELS[focus]}=%{{y:.3g}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    n_mark = add_marked_gene_trace(
        fig,
        results,
        "dynamic_range",
        y_col,
        mark_genes,
        legend_name=mark_label or "marked",
        size=8,
        opacity=1.0,
    )

    fig.add_hline(y=rho_thr, line=dict(dash="dash", color="#808080", width=0.8))
    fig.add_hline(y=-rho_thr, line=dict(dash="dash", color="#808080", width=0.8))
    fig.add_vline(x=dr_thr, line=dict(dash="dash", color="#808080", width=0.8))

    title_dict = title if isinstance(title, dict) else _plotly_title(str(title))
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    fig.update_layout(
        title=title_dict,
        xaxis_title="Dynamic range of gene expression",
        yaxis_title=f"{_METHOD_LABELS[focus]} (expression vs ordered levels)",
        height=480,
        width=520,
        uirevision=f"grad-{focus}",
        showlegend=bool(n_mark),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=20, t=40 + 22 * title_lines, b=50),
        hovermode="closest",
        clickmode="event+select",
    )
    return fig


def pairwise_coeff_fig(
    results: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    mark_genes: set[str] | None = None,
    mark_label: str | None = None,
) -> go.Figure:
    """Scatter of two correlation coefficients against each other."""
    xa, ya = _METHOD_COLS[method_a], _METHOD_COLS[method_b]
    if xa not in results.columns or ya not in results.columns:
        fig = go.Figure()
        fig.add_annotation(text="Pair not available", showarrow=False)
        return fig
    fig = go.Figure()
    marked = gene_mark_mask(results, mark_genes)
    base = results.loc[~marked] if marked.any() else results
    if len(base):
        fig.add_trace(
            go.Scatter(
                x=base[xa],
                y=base[ya],
                mode="markers",
                marker=dict(size=6, color=_COLOR_FAIL, opacity=0.45),
                customdata=base["geneID"].astype(str),
                text=base["geneID"].astype(str),
                hovertemplate=(
                    "%{text}<br>"
                    f"{_METHOD_LABELS[method_a]}=%{{x:.3g}}<br>"
                    f"{_METHOD_LABELS[method_b]}=%{{y:.3g}}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    n_mark = add_marked_gene_trace(
        fig,
        results,
        xa,
        ya,
        mark_genes,
        legend_name=mark_label or "marked",
        size=8,
        opacity=1.0,
    )
    lim = float(
        np.nanmax(np.abs(np.concatenate([results[xa].to_numpy(), results[ya].to_numpy()])))
    )
    lim = max(lim, 0.05)
    title = _plotly_title(
        f"{_METHOD_LABELS[method_a]} vs {_METHOD_LABELS[method_b]}",
    )
    fig.update_layout(
        title=title,
        xaxis_title=_METHOD_LABELS[method_a],
        yaxis_title=_METHOD_LABELS[method_b],
        height=420,
        width=420,
        uirevision=f"grad-pair-{method_a}-{method_b}",
        showlegend=bool(n_mark),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=20, t=50, b=50),
        hovermode="closest",
        clickmode="event+select",
    )
    fig.update_xaxes(range=[-lim * 1.05, lim * 1.05], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-lim * 1.05, lim * 1.05])
    return fig


def _level_tick_label(level: str) -> str:
    s = str(level)
    return s.replace("region", "") if s.lower().startswith("region") else s


def gene_profile_grid_fig(
    genes: list[str],
    *,
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    order_col: str,
    order_levels: list[str],
    rep_col: str | None,
    results: pd.DataFrame | None = None,
) -> go.Figure:
    """Notebook-style region profiles (line + markers per replicate), ≤6×6 grid.

    Always lays out on a fixed ``cols=_PROFILE_COLS`` grid so gene 0 is ax[0,0]
    with the same panel size whether 1 or 36 genes are selected.
    """
    genes = [str(g) for g in genes if str(g) in set(expr.index.astype(str))][:_MAX_PROFILE_GENES]
    if not genes:
        fig = go.Figure()
        fig.add_annotation(
            text="Click genes on a gradient / pairwise plot to show expression profiles.",
            showarrow=False,
        )
        fig.update_layout(
            height=_PROFILE_TITLE_H + _PROFILE_CELL_H + _PROFILE_BOTTOM,
            width=_PROFILE_LEFT + _PROFILE_COLS * _PROFILE_CELL_W + _PROFILE_RIGHT,
            margin=dict(
                l=_PROFILE_LEFT,
                r=_PROFILE_RIGHT,
                t=_PROFILE_TITLE_H,
                b=_PROFILE_BOTTOM,
            ),
        )
        return fig

    n = len(genes)
    n_cols = _PROFILE_COLS
    n_rows = int(math.ceil(n / n_cols))
    titles = [_display_name_for_gene(results, g) for g in genes]
    # Pad so make_subplots keeps a full 6-col row (empty cells stay blank)
    titles_full = titles + [""] * (n_rows * n_cols - n)
    # Extra vertical gap so row i+1 subplot titles clear row i x-tick labels.
    # Constraint: (n_rows - 1) * vertical_spacing < 1.
    if n_rows <= 1:
        v_space = 0.12
    else:
        v_space = min(0.22, 0.72 / (n_rows - 1))
    h_space = 0.05
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles_full,
        horizontal_spacing=h_space,
        vertical_spacing=v_space,
    )

    meta = meta.copy()
    meta.index = meta.index.astype(str)
    meta[order_col] = meta[order_col].astype(str)
    x_pos = list(range(len(order_levels)))
    x_labels = [_level_tick_label(lv) for lv in order_levels]

    if rep_col and rep_col in meta.columns:
        replicates = _natural_sorted(meta[rep_col].dropna().unique())
    else:
        replicates = ["all"]

    legend_done: set[str] = set()
    for gi, gene in enumerate(genes):
        row = gi // n_cols + 1
        col = gi % n_cols + 1
        for ri, replicate in enumerate(replicates):
            if rep_col and rep_col in meta.columns and replicate != "all":
                rep_meta = meta.loc[meta[rep_col].astype(str) == str(replicate)]
            else:
                rep_meta = meta
            ys: list[float | None] = []
            for level in order_levels:
                samples = rep_meta.index[rep_meta[order_col] == level]
                samples = [s for s in samples if s in expr.columns]
                if not samples or gene not in expr.index:
                    ys.append(None)
                    continue
                ys.append(float(np.nanmean(expr.loc[gene, samples].to_numpy(dtype=float))))
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
                    customdata=[gene] * len(x_pos),
                    hovertemplate=(
                        f"{titles[gi]}<br>rep={replicate}<br>"
                        "level=%{x}<br>expr=%{y:.3g}<extra></extra>"
                    ),
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

    # Hide unused cells (keep layout/size stable)
    for gi in range(n, n_rows * n_cols):
        row = gi // n_cols + 1
        col = gi % n_cols + 1
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    n_leg = max(1, len(legend_done))
    # Legend (vertical) left of title in the figure header — clear of subplot titles
    legend_w = 88
    top_margin = max(_PROFILE_TITLE_H, n_leg * _PROFILE_LEGEND_ROW_H) + 12
    left_margin = _PROFILE_LEFT + legend_w
    fig_h = top_margin + n_rows * _PROFILE_CELL_H + _PROFILE_BOTTOM
    fig_w = left_margin + n_cols * _PROFILE_CELL_W + _PROFILE_RIGHT
    title = _plotly_title("Gene expression gradients along ordered levels")
    # Title immediately right of the stacked legend (both in the top banner)
    title_x = (legend_w + 12) / max(fig_w, 1)
    title.update(
        x=title_x,
        xref="container",
        xanchor="left",
        y=0.995,
        yref="container",
        yanchor="top",
    )
    fig.update_layout(
        title=title,
        height=fig_h,
        width=fig_w,
        margin=dict(
            l=left_margin,
            r=_PROFILE_RIGHT,
            t=top_margin,
            b=_PROFILE_BOTTOM,
        ),
        legend=dict(
            orientation="v",
            traceorder="normal",
            itemsizing="constant",
            font=dict(size=10),
            xref="container",
            yref="container",
            x=0.004,
            y=0.995,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            tracegroupgap=2,
            itemwidth=30,
        ),
        uirevision="grad-profiles",
        clickmode="event+select",
        hovermode="closest",
    )
    fig.update_annotations(font_size=11)
    return fig


def save_gradient_celov(
    results: pd.DataFrame,
    score_col: str,
    out_path: str | Path,
    mode: str,
    locus_lookup: pd.DataFrame | None = None,
    id_column: str = "biocyc_id",
) -> list[Path]:
    """Celov export like distances.ipynb: score + dynamic_range (not inverted)."""
    df = results.copy()
    id_column = (id_column or "biocyc_id").strip() or "biocyc_id"
    if locus_lookup is not None:
        if id_column not in df.columns:
            df = annotate_gene_table(df, locus_lookup)
        if id_column not in df.columns:
            raise ValueError(
                f"Celov ID column {id_column!r} not in locus lookup "
                f"(columns: {list(locus_lookup.columns)})."
            )
    elif id_column not in df.columns:
        id_column = "geneID"

    subsets = {
        "up_and_down": df,
        "up": df[df[score_col] > 0],
        "down": df[df[score_col] < 0],
    }
    mode = (mode or "up_and_down").lower().replace(" ", "_")
    if mode in ("combined", "both"):
        mode = "up_and_down"
    if mode == "all":
        types = ["up_and_down", "up", "down"]
    elif mode in subsets:
        types = [mode]
    else:
        raise ValueError(f"Unknown Celov mode {mode!r}")

    template = str(out_path)
    if "{}" not in template:
        p = Path(template)
        template = str(p.with_name(f"{p.stem}_{{}}{p.suffix or '.txt'}"))

    written: list[Path] = []
    for kind in types:
        subset = subsets[kind]
        path = Path(template.format(kind))
        path.parent.mkdir(parents=True, exist_ok=True)
        celov_multiomics_file_generation(
            subset,
            path,
            score_col,
            column2="dynamic_range" if "dynamic_range" in subset.columns else None,
            column2_invert=False,
            id_column=id_column,
            dataset_label=f"gene_gradient_{score_col}_{kind}",
        )
        written.append(path)
    return written


def _thr_block(method: str) -> html.Div:
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("|ρ| / |τ|", className="small mb-0"),
                            dbc.Input(
                                id=f"grad-rho-thr-{method}",
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
                                id=f"grad-dr-thr-{method}",
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
        id=f"grad-thr-wrap-{method}",
        style={"display": "none"},
    )


def _measure_col(method: str) -> dbc.Col:
    return dbc.Col(
        [
            dcc.Graph(id=f"grad-plot-{method}", figure={}, config=_GRAPH_CONFIG),
            _thr_block(method),
        ],
        md=4,
    )


class GeneGradientsModule:
    id = "gene-gradients"
    label = "Gene gradients"

    def layout(self):
        return html.Div(
            [
                html.P(
                    "Correlate gene expression with ordered metadata levels "
                    "(distances.ipynb gradients; Pearson / Spearman + Kendall τ). "
                    "Subset samples, drag level order, then set per-plot thresholds after Run. "
                    "Click genes on correlation / pairwise plots for profiles (below Celov) "
                    "and locus-lookup metadata (beside pairwise plots).",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Subset column"),
                                dcc.Dropdown(id="grad-subset-col", clearable=True),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Subset value"),
                                dcc.Dropdown(id="grad-subset-val", clearable=True),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Order column"),
                                dcc.Dropdown(id="grad-order-col", clearable=True),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Replicate column"),
                                dcc.Dropdown(id="grad-rep-col", clearable=True),
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
                                html.Label("Level order (drag to reorder; ranks 1, 2, …)"),
                                html.Div(
                                    id="grad-order-list",
                                    className="border rounded p-2 bg-light",
                                ),
                                dcc.Store(id="grad-order-store", data=[]),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Run gradients",
                                id="grad-run",
                                color="primary",
                                className="mt-4",
                            ),
                            md=4,
                            className="d-flex align-items-start",
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="grad-status", className="text-muted small mb-2"),
                mark_controls(
                    col_id="grad-mark-col",
                    entry_id="grad-mark-entry",
                    wrap_id="grad-mark-wrap",
                ),
                dcc.Loading(
                    html.Div(
                        [
                            dbc.Row(
                                [_measure_col(m) for m in _METHODS],
                                className="g-2 mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H6("Pairwise coefficient comparison"),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="grad-pair-ps",
                                                            figure={},
                                                            config=_GRAPH_CONFIG,
                                                        ),
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="grad-pair-pk",
                                                            figure={},
                                                            config=_GRAPH_CONFIG,
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        dcc.Graph(
                                                            id="grad-pair-sk",
                                                            figure={},
                                                            config=_GRAPH_CONFIG,
                                                        ),
                                                        md=6,
                                                    ),
                                                ],
                                                className="g-2",
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.H6("Gene metadata", className="mb-2"),
                                                html.P(
                                                    "Click a gene on a correlation or pairwise plot.",
                                                    className="text-muted small mb-2",
                                                ),
                                                html.Div(
                                                    id="grad-gene-detail",
                                                    children=gene_detail_placeholder(),
                                                ),
                                            ],
                                            className="border rounded p-2 bg-light",
                                        ),
                                        md=4,
                                    ),
                                ],
                                className="g-2 mb-2 align-items-start",
                            ),
                        ]
                    ),
                    type="default",
                ),
                html.Hr(),
                html.H6("Save Celov"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Score"),
                                dcc.Dropdown(
                                    id="grad-celov-score",
                                    options=[
                                        {"label": "Pearson ρ", "value": "pearson_rho"},
                                        {"label": "Spearman ρ", "value": "spearman_rho"},
                                        {"label": "Kendall τ", "value": "kendall_tau"},
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
                                    id="grad-celov-mode",
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
                                        dbc.Input(id="grad-celov-out", type="text"),
                                        dbc.Button(
                                            "Browse…",
                                            id="grad-celov-browse",
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
                dbc.Button("Save Celov", id="grad-celov-save", color="secondary", className="mb-2"),
                html.Div(id="grad-celov-status", className="text-muted small mb-2"),
                html.Hr(),
                html.H6("Clicked gene profiles"),
                html.P(
                    "Click a gene on any gradient or pairwise plot (click again to remove). "
                    f"Up to {_PROFILE_COLS} per row, {_MAX_PROFILE_GENES // _PROFILE_COLS} rows "
                    f"(max {_MAX_PROFILE_GENES}).",
                    className="text-muted small",
                ),
                dbc.Button(
                    "Clear selected genes",
                    id="grad-profiles-clear",
                    color="secondary",
                    outline=True,
                    size="sm",
                    className="mb-2",
                ),
                html.Div(id="grad-profiles-status", className="text-muted small mb-1"),
                dcc.Graph(id="grad-profiles-grid", figure={}, config=_GRAPH_CONFIG),
                dcc.Store(id="grad-cache"),
                dcc.Store(id="grad-selected-genes", data=[]),
                dcc.Store(id="grad-last-gene", data=None),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output("grad-subset-col", "options"),
            Output("grad-order-col", "options"),
            Output("grad-rep-col", "options"),
            Output("grad-subset-col", "value"),
            Output("grad-order-col", "value"),
            Output("grad-rep-col", "value"),
            Input("session-store", "data"),
        )
        def _fill_meta_cols(session_blob):
            session = session_from_store(session_blob)
            if not session.ready or session.metadata is None:
                return [], [], [], None, None, None
            cols = [
                c
                for c in session.meta_columns()
                if not str(c).startswith("_")
                and not (str(c).startswith("PC") and str(c)[2:].isdigit())
            ]
            opts = [{"label": c, "value": c} for c in cols]
            subset = "Biofilm" if "Biofilm" in cols else (cols[0] if cols else None)
            order = (
                "GrowthPhase"
                if "GrowthPhase" in cols
                else ("region" if "region" in cols else None)
            )
            if order is None and cols:
                order = cols[min(1, len(cols) - 1)]
            rep = "BiologicalReplicate" if "BiologicalReplicate" in cols else None
            return opts, opts, opts, subset, order, rep

        @app.callback(
            Output("grad-subset-val", "options"),
            Output("grad-subset-val", "value"),
            Input("grad-subset-col", "value"),
            State("session-store", "data"),
        )
        def _fill_subset_vals(col, session_blob):
            session = session_from_store(session_blob)
            if (
                not col
                or not session.ready
                or session.metadata is None
                or col not in session.metadata.columns
            ):
                return [], None
            vals = _natural_sorted(session.metadata[col].dropna().unique())
            opts = [{"label": v, "value": v} for v in vals]
            default = "True" if "True" in vals else (vals[0] if vals else None)
            return opts, default

        @app.callback(
            Output("grad-order-store", "data"),
            Input("grad-order-col", "value"),
            Input("grad-subset-col", "value"),
            Input("grad-subset-val", "value"),
            State("session-store", "data"),
            prevent_initial_call=False,
        )
        def _update_order_store(order_col, subset_col, subset_val, session_blob):
            session = session_from_store(session_blob)
            if (
                not order_col
                or not session.ready
                or session.metadata is None
                or order_col not in session.metadata.columns
            ):
                return []
            meta = session.metadata
            if subset_col and subset_val is not None and subset_col in meta.columns:
                meta = meta.loc[_subset_mask(meta, subset_col, str(subset_val))]
            return _natural_sorted(meta[order_col].dropna().unique())

        @app.callback(
            Output("grad-order-list", "children"),
            Input("grad-order-store", "data"),
        )
        def _render_order_list(levels):
            levels = list(levels or [])
            if not levels:
                return html.P(
                    "No levels — choose order column / subset.",
                    className="text-muted small mb-0",
                )
            return [
                html.Div(
                    f"{i + 1}. {level}",
                    className="grad-ord-item",
                    draggable="true",
                    **{"data-level": level},
                )
                for i, level in enumerate(levels)
            ]

        @app.callback(
            Output("grad-cache", "data"),
            Output("grad-status", "children"),
            Output("grad-selected-genes", "data", allow_duplicate=True),
            Output("grad-last-gene", "data", allow_duplicate=True),
            Output("grad-thr-wrap-pearson", "style"),
            Output("grad-thr-wrap-spearman", "style"),
            Output("grad-thr-wrap-kendall", "style"),
            Output("grad-mark-wrap", "style"),
            Output("grad-mark-col", "options"),
            Output("grad-mark-col", "value"),
            Input("grad-run", "n_clicks"),
            State("session-store", "data"),
            State("project-store", "data"),
            State("grad-subset-col", "value"),
            State("grad-subset-val", "value"),
            State("grad-order-col", "value"),
            State("grad-order-store", "data"),
            State("grad-rep-col", "value"),
            prevent_initial_call=True,
        )
        def _run(
            n_clicks,
            session_blob,
            project_blob,
            subset_col,
            subset_val,
            order_col,
            order_levels,
            rep_col,
        ):
            hide = {"display": "none"}
            show = {"display": "block"}
            empty_mark = (hide, [], None)
            session = session_from_store(session_blob)
            if not session.ready:
                return (
                    no_update,
                    session.error or "Load a dataset first.",
                    no_update,
                    no_update,
                    hide,
                    hide,
                    hide,
                    *empty_mark,
                )
            if not subset_col or subset_val is None or not order_col:
                return (
                    no_update,
                    "Choose subset column/value and order column.",
                    no_update,
                    no_update,
                    hide,
                    hide,
                    hide,
                    *empty_mark,
                )
            methods = list(_METHODS)
            try:
                meta = session.metadata.copy()
                meta.index = meta.index.astype(str)
                mask = _subset_mask(meta, subset_col, str(subset_val))
                meta_sub = meta.loc[mask]
                if meta_sub.empty:
                    return (
                        no_update,
                        f"No samples with {subset_col}={subset_val!r}.",
                        no_update,
                        no_update,
                        hide,
                        hide,
                        hide,
                        *empty_mark,
                    )

                available = set(meta_sub[order_col].dropna().astype(str).unique().tolist())
                order_levels = [str(x) for x in (order_levels or []) if str(x) in available]
                if len(order_levels) < 2:
                    return (
                        no_update,
                        "Need at least two ordered levels (drag to reorder).",
                        no_update,
                        no_update,
                        hide,
                        hide,
                        hide,
                        *empty_mark,
                    )

                expr = session.expression.T
                expr.columns = expr.columns.astype(str)
                samples = [s for s in meta_sub.index.astype(str) if s in expr.columns]
                if len(samples) < 3:
                    return (
                        no_update,
                        "Fewer than 3 matching samples in the expression matrix.",
                        no_update,
                        no_update,
                        hide,
                        hide,
                        hide,
                        *empty_mark,
                    )
                expr_sub = expr[samples]
                meta_use = meta_sub.loc[samples]

                results = gene_region_gradients(
                    expr_sub, meta_use, order_col, order_levels, methods
                )
                lookup = None
                active = (session_blob or {}).get("active_datasets") or []
                name = active[0] if active else None
                entry = next(
                    (
                        d
                        for d in (project_blob or {}).get("datasets", [])
                        if d.get("name") == name
                    ),
                    None,
                )
                locus = (entry or {}).get("locus_lookup") or ""
                root = (project_blob or {}).get("root")
                if locus:
                    path = Path(locus)
                    if root and not path.is_absolute():
                        path = Path(root) / path
                    try:
                        lookup = load_locus_lookup(path)
                    except Exception:  # noqa: BLE001
                        lookup = None

                _GRAD_RUNTIME.clear()
                _GRAD_RUNTIME["results"] = results
                _GRAD_RUNTIME["methods"] = methods
                _GRAD_RUNTIME["order_levels"] = order_levels
                _GRAD_RUNTIME["order_col"] = order_col
                _GRAD_RUNTIME["rep_col"] = rep_col
                _GRAD_RUNTIME["meta"] = meta_use
                _GRAD_RUNTIME["expr"] = expr_sub
                _GRAD_RUNTIME["locus_lookup"] = lookup
                cache = {
                    "n_genes": int(len(results)),
                    "n_samples": int(expr_sub.shape[1]),
                    "methods": methods,
                    "order_levels": order_levels,
                    "dataset": _active_dataset_name(session_blob),
                    "subset": f"{subset_col}={subset_val}",
                    "order_col": order_col,
                }
                thr_styles = (show, show, show)
                mark_cols = locus_mark_columns(lookup)
                mark_opts = [{"label": c, "value": c} for c in mark_cols]
                mark_wrap = show if mark_cols else hide
                return (
                    cache,
                    html.Span(
                        [
                            f"Gradients done: {len(results)} genes × {expr_sub.shape[1]} samples; "
                            f"levels {order_levels}; Pearson / Spearman / Kendall. "
                            "Set thresholds under each plot. ",
                            html.Strong(
                                "Select genes for profiles and metadata information."
                            ),
                        ]
                    ),
                    [],
                    None,
                    thr_styles[0],
                    thr_styles[1],
                    thr_styles[2],
                    mark_wrap,
                    mark_opts,
                    None,
                )
            except Exception as exc:  # noqa: BLE001
                _GRAD_RUNTIME.clear()
                return (
                    no_update,
                    f"Gradient error: {exc}",
                    [],
                    None,
                    hide,
                    hide,
                    hide,
                    hide,
                    [],
                    None,
                )

        @app.callback(
            Output("grad-plot-pearson", "figure"),
            Output("grad-plot-spearman", "figure"),
            Output("grad-plot-kendall", "figure"),
            Output("grad-pair-ps", "figure"),
            Output("grad-pair-pk", "figure"),
            Output("grad-pair-sk", "figure"),
            Input("grad-cache", "data"),
            Input("grad-rho-thr-pearson", "value"),
            Input("grad-dr-thr-pearson", "value"),
            Input("grad-rho-thr-spearman", "value"),
            Input("grad-dr-thr-spearman", "value"),
            Input("grad-rho-thr-kendall", "value"),
            Input("grad-dr-thr-kendall", "value"),
            Input("grad-mark-col", "value"),
            Input("grad-mark-entry", "value"),
        )
        def _replot(cache, rho_p, dr_p, rho_s, dr_s, rho_k, dr_k, mark_col, mark_entry):
            empty = go.Figure()
            results = _GRAD_RUNTIME.get("results")
            if not cache or results is None or not isinstance(results, pd.DataFrame):
                return empty, empty, empty, empty, empty, empty
            methods = list(_METHODS)
            thr = {
                "pearson": (
                    float(rho_p if rho_p is not None else 0.7),
                    float(dr_p if dr_p is not None else 1.0),
                ),
                "spearman": (
                    float(rho_s if rho_s is not None else 0.7),
                    float(dr_s if dr_s is not None else 1.0),
                ),
                "kendall": (
                    float(rho_k if rho_k is not None else 0.7),
                    float(dr_k if dr_k is not None else 1.0),
                ),
            }
            dataset = cache.get("dataset") or "dataset"
            subset = cache.get("subset") or ""
            order_col = cache.get("order_col") or "order"
            mark_genes = genes_with_meta_entry(
                _GRAD_RUNTIME.get("locus_lookup"),
                mark_col,
                mark_entry,
            )
            mark_label = f"{mark_col}={mark_entry}" if mark_col and mark_entry else None

            measure_figs = []
            for method in methods:
                rho_thr, dr_thr = thr[method]
                title = _plotly_title(
                    f"{_METHOD_LABELS[method]} gene gradients",
                    f"in {dataset} ({subset}; ordered by {order_col})",
                )
                measure_figs.append(
                    gradient_scatter_fig(
                        results,
                        method,
                        rho_thr=rho_thr,
                        dr_thr=dr_thr,
                        title=title,
                        mark_genes=mark_genes,
                        mark_label=mark_label,
                    )
                )

            pairs = [
                ("pearson", "spearman"),
                ("pearson", "kendall"),
                ("spearman", "kendall"),
            ]
            pair_figs = [
                pairwise_coeff_fig(
                    results,
                    a,
                    b,
                    mark_genes=mark_genes,
                    mark_label=mark_label,
                )
                for a, b in pairs
            ]

            return (
                measure_figs[0],
                measure_figs[1],
                measure_figs[2],
                pair_figs[0],
                pair_figs[1],
                pair_figs[2],
            )

        @app.callback(
            Output("grad-mark-entry", "options"),
            Output("grad-mark-entry", "value"),
            Output("grad-mark-entry", "disabled"),
            Output("grad-mark-entry", "placeholder"),
            Input("grad-mark-col", "value"),
            State("grad-cache", "data"),
        )
        def _mark_entries(col, cache):
            if not cache or not col:
                return [], None, True, "Select a column first…"
            opts = entry_dropdown_options(_GRAD_RUNTIME.get("locus_lookup"), col)
            return opts, None, False, "Entry…"

        @app.callback(
            Output("grad-selected-genes", "data"),
            Output("grad-last-gene", "data"),
            Output("grad-profiles-status", "children"),
            Input("grad-plot-pearson", "clickData"),
            Input("grad-plot-spearman", "clickData"),
            Input("grad-plot-kendall", "clickData"),
            Input("grad-pair-ps", "clickData"),
            Input("grad-pair-pk", "clickData"),
            Input("grad-pair-sk", "clickData"),
            Input("grad-profiles-clear", "n_clicks"),
            State("grad-selected-genes", "data"),
            prevent_initial_call=True,
        )
        def _select_genes(
            click_p,
            click_s,
            click_k,
            click_ps,
            click_pk,
            click_sk,
            n_clear,
            selected,
        ):
            triggered = callback_context.triggered_id
            selected = list(selected or [])
            if triggered == "grad-profiles-clear":
                return [], None, "Cleared selected genes."
            click_map = {
                "grad-plot-pearson": click_p,
                "grad-plot-spearman": click_s,
                "grad-plot-kendall": click_k,
                "grad-pair-ps": click_ps,
                "grad-pair-pk": click_pk,
                "grad-pair-sk": click_sk,
            }
            gene = (
                _click_gene_id(click_map.get(triggered))
                if isinstance(triggered, str)
                else None
            )
            if not gene:
                return no_update, no_update, no_update
            # Metadata table always shows the gene just clicked (correlation / pairwise)
            if gene in selected:
                selected = [g for g in selected if g != gene]
                return (
                    selected,
                    gene,
                    f"Removed {gene} from profiles ({len(selected)}/{_MAX_PROFILE_GENES}).",
                )
            if len(selected) >= _MAX_PROFILE_GENES:
                return (
                    selected,
                    gene,
                    f"Already at max {_MAX_PROFILE_GENES} profiles — remove one or Clear.",
                )
            selected = selected + [gene]
            return (
                selected,
                gene,
                f"Profiles {len(selected)}/{_MAX_PROFILE_GENES}; metadata = {gene}.",
            )

        @app.callback(
            Output("grad-gene-detail", "children"),
            Input("grad-last-gene", "data"),
            Input("grad-cache", "data"),
        )
        def _gene_meta_table(last_gene, cache):
            if not last_gene:
                return gene_detail_placeholder()
            return gene_detail_table(str(last_gene), _GRAD_RUNTIME.get("locus_lookup"))

        @app.callback(
            Output("grad-profiles-grid", "figure"),
            Input("grad-selected-genes", "data"),
            Input("grad-cache", "data"),
        )
        def _profiles(selected, cache):
            results = _GRAD_RUNTIME.get("results")
            expr = _GRAD_RUNTIME.get("expr")
            meta = _GRAD_RUNTIME.get("meta")
            order_levels = list(_GRAD_RUNTIME.get("order_levels") or [])
            order_col = _GRAD_RUNTIME.get("order_col")
            rep_col = _GRAD_RUNTIME.get("rep_col")
            if (
                expr is None
                or meta is None
                or not order_levels
                or not order_col
            ):
                return gene_profile_grid_fig(
                    [],
                    expr=pd.DataFrame(),
                    meta=pd.DataFrame(),
                    order_col="level",
                    order_levels=[],
                    rep_col=None,
                    results=None,
                )
            return gene_profile_grid_fig(
                list(selected or []),
                expr=expr,
                meta=meta,
                order_col=str(order_col),
                order_levels=order_levels,
                rep_col=rep_col if isinstance(rep_col, str) else None,
                results=results if isinstance(results, pd.DataFrame) else None,
            )

        @app.callback(
            Output("grad-celov-out", "value"),
            Output("grad-celov-status", "children", allow_duplicate=True),
            Input("grad-celov-browse", "n_clicks"),
            State("grad-celov-out", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_celov(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save gradient Celov (use {} for type)",
                defaultextension=".txt",
                initialfile="GeneGradient_{}.txt",
            )
            if not chosen:
                return no_update, "Celov path browse cancelled."
            p = Path(chosen)
            if "{}" not in p.name:
                chosen = str(p.with_name(f"{p.stem}_{{}}{p.suffix or '.txt'}"))
            return chosen, f"Celov output template: {chosen}"

        @app.callback(
            Output("grad-celov-status", "children"),
            Input("grad-celov-save", "n_clicks"),
            State("grad-celov-out", "value"),
            State("grad-celov-mode", "value"),
            State("grad-celov-score", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _save_celov(n_clicks, out_path, mode, score_col, session_blob, project_blob):
            results = _GRAD_RUNTIME.get("results")
            if results is None or not isinstance(results, pd.DataFrame) or results.empty:
                return "Run gradients first."
            if not out_path or not str(out_path).strip():
                return "Choose an output .txt path (Browse)."
            score_col = score_col or "pearson_rho"
            if score_col not in results.columns:
                return f"Score column {score_col!r} was not computed — re-run with that measure."
            try:
                active = (session_blob or {}).get("active_datasets") or []
                name = active[0] if active else None
                entry = next(
                    (
                        d
                        for d in (project_blob or {}).get("datasets", [])
                        if d.get("name") == name
                    ),
                    None,
                )
                lookup = None
                locus = (entry or {}).get("locus_lookup") or ""
                root = (project_blob or {}).get("root")
                if locus:
                    path = Path(locus)
                    if root and not path.is_absolute():
                        path = Path(root) / path
                    lookup = load_locus_lookup(path)
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
