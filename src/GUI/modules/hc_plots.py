"""Helpers for hierarchical clustering figures (notebook Hierarchical_clustering.ipynb)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy

from ..components.controls import apply_export_layout, equal_xy_axes
from ..components.gene_meta_mark import add_marked_gene_trace, gene_mark_mask
from ..components.sample_detail import (
    gene_detail_table,
    sample_detail_placeholder,
    sample_detail_table,
)
from dash import html
import dash_bootstrap_components as dbc

_QUAL = px.colors.qualitative.Plotly
_MIXED = "#888888"


def cluster_label(c: int | np.integer) -> str:
    """Numeric cluster id as string (no 'C' prefix — keeps 9 before 10 in legends)."""
    return str(int(c))


def _cluster_color(c: int) -> str:
    """Map fcluster id (1..t) to a stable color; shared by dendrogram, PCA, legends."""
    if c < 0:
        return _MIXED
    # 1-based cluster ids → sequence index 0,1,… so PCA and dendro match
    return _QUAL[int(c - 1) % len(_QUAL)]


def cluster_color_map(cluster_ids) -> dict[str, str]:
    return {cluster_label(c): _cluster_color(int(c)) for c in sorted({int(x) for x in cluster_ids})}


def _scale_icoord(icoord: np.ndarray) -> np.ndarray:
    """Map scipy leaf centers (5, 15, …) → heatmap indices (0, 1, …)."""
    return (np.asarray(icoord, dtype=float) - 5.0) / 10.0


def _leaf_clusters_in_dendro_order(Z: np.ndarray, labels: np.ndarray) -> tuple[list[int], list[int]]:
    dendro = hierarchy.dendrogram(Z, no_plot=True)
    leaves = list(dendro["leaves"])
    leaf_clusters = [int(labels[i]) for i in leaves]
    return leaves, leaf_clusters


def _segment_color(xs_scaled: list[float], leaf_clusters: list[int]) -> str:
    lo = int(round(min(xs_scaled)))
    hi = int(round(max(xs_scaled)))
    lo = max(0, lo)
    hi = min(len(leaf_clusters) - 1, hi)
    subset = leaf_clusters[lo : hi + 1]
    if subset and len(set(subset)) == 1:
        return _cluster_color(subset[0])
    return _MIXED


def dendro_polylines(
    Z: np.ndarray,
    *,
    orientation: str,
    leaf_clusters: list[int] | None = None,
) -> list[tuple[list[float], list[float], str]]:
    """Scaled dendrogram segments; optional leaf_clusters colors pure subtrees.

    ``left`` orientation reflects distance across the vertical axis so leaves sit
    on the right (toward the heatmap) and the tree grows leftward.
    """
    dendro = hierarchy.dendrogram(Z, no_plot=True)
    icoord = np.array(dendro["icoord"], dtype=float)
    dcoord = np.array(dendro["dcoord"], dtype=float)
    out = []
    for xs_raw, ys_raw in zip(icoord, dcoord):
        xs = _scale_icoord(xs_raw).tolist()
        ys = ys_raw.tolist()
        color = _segment_color(xs, leaf_clusters) if leaf_clusters is not None else "#444444"
        if orientation == "top":
            out.append((xs + [None], ys + [None], color))
        elif orientation == "left":
            # x = −distance (flip wrt vertical axis); y = leaf position
            out.append(([-y for y in ys] + [None], xs + [None], color))
        else:
            raise ValueError(orientation)
    return out


def _add_cluster_legend(fig: go.Figure, leaf_clusters: list[int] | None, *, title: str = "cluster") -> None:
    if not leaf_clusters:
        return
    first = True
    for cid in sorted(set(leaf_clusters)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=_cluster_color(cid)),
                name=cluster_label(cid),
                legendgroup=title,
                legendgrouptitle_text=title if first else None,
                showlegend=True,
                hoverinfo="skip",
            )
        )
        first = False


def _heatmap_size(n_row: int, n_col: int) -> tuple[int, int]:
    """Fixed figure size (not scaled by number of rows/columns)."""
    del n_row, n_col  # size is independent of matrix shape
    return 720, 640


def heatmap_with_dendro(
    z: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    Z_row: np.ndarray | None,
    Z_col: np.ndarray | None,
    *,
    title: str | dict,
    colorscale: str = "Viridis",
    colorbar_title: str = "Pairwise distances",
    row_leaf_clusters: list[int] | None = None,
    col_leaf_clusters: list[int] | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Viridis heatmap; axis dendrograms aligned to cells (scipy coords scaled)."""
    n_row, n_col = z.shape
    if Z_row is not None:
        row_order = list(hierarchy.dendrogram(Z_row, no_plot=True)["leaves"])
        z = z[row_order, :]
        row_labels = [row_labels[i] for i in row_order]
    if Z_col is not None:
        col_order = list(hierarchy.dendrogram(Z_col, no_plot=True)["leaves"])
        z = z[:, col_order]
        col_labels = [col_labels[i] for i in col_order]

    custom = np.empty(z.shape, dtype=object)
    text_hover = np.empty(z.shape, dtype=object)
    for i, r in enumerate(row_labels):
        for j, c in enumerate(col_labels):
            custom[i, j] = f"{r}||{c}"
            text_hover[i, j] = f"row: {r}<br>col: {c}"

    has_row = Z_row is not None
    has_col = Z_col is not None
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.15, 0.85] if has_row else [0.001, 0.999],
        row_heights=[0.15, 0.85] if has_col else [0.001, 0.999],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
    )

    if has_col:
        for xs, ys, color in dendro_polylines(
            Z_col,
            orientation="top",
            leaf_clusters=col_leaf_clusters,
        ):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        # Axis label on top of the column dendrogram (e.g. "Genes")
        fig.update_xaxes(
            range=[-0.5, n_col - 0.5],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title_text=xaxis_title or "",
            title_standoff=4,
            side="top",
            row=1,
            col=2,
        )
        fig.update_yaxes(visible=False, row=1, col=2)

    if has_row:
        for xs, ys, color in dendro_polylines(
            Z_row,
            orientation="left",
            leaf_clusters=row_leaf_clusters,
        ):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig.update_xaxes(visible=False, row=2, col=1)
        # Leaf 0 at top — same as heatmap; label left of row dendrogram (e.g. "Samples")
        fig.update_yaxes(
            range=[n_row - 0.5, -0.5],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title_text=yaxis_title or "",
            title_standoff=4,
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=list(range(n_col)),
            y=list(range(n_row)),
            colorscale=colorscale,
            customdata=custom,
            text=text_hover,
            hovertemplate="%{text}<br>value=%{z:.4g}<extra></extra>",
            colorbar=dict(
                title=dict(text=colorbar_title, side="right"),
                len=0.55,
                y=0.38,
                yanchor="middle",
                # Sit to the right of the cluster legend (legend uses x≈1.02).
                x=1.16,
                xpad=4,
                thickness=14,
            ),
            showscale=True,
        ),
        row=2,
        col=2,
    )
    # Heatmap axes: no "Samples"/"Genes" here — those sit on the dendrograms
    fig.update_xaxes(
        range=[-0.5, n_col - 0.5],
        showticklabels=False,
        title_text="",
        row=2,
        col=2,
    )
    fig.update_yaxes(
        range=[n_row - 0.5, -0.5],
        showticklabels=False,
        title_text="",
        row=2,
        col=2,
    )
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Keep dendrogram leaf axes in lockstep with heatmap zoom/pan (Plotly subplot
    # ids: col dendro=x2/y2, row dendro=x3/y3, heatmap=x4/y4). Distance axes stay
    # independent. Remove this block to restore independent zoom.
    if has_col:
        fig.update_xaxes(matches="x4", row=1, col=2)
        fig.update_yaxes(fixedrange=True, row=1, col=2)
    if has_row:
        fig.update_yaxes(matches="y4", row=2, col=1)
        fig.update_xaxes(fixedrange=True, row=2, col=1)

    legend_clusters = row_leaf_clusters or col_leaf_clusters
    if legend_clusters:
        for cid in sorted(set(legend_clusters)):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=_cluster_color(cid)),
                    name=cluster_label(cid),
                    showlegend=True,
                    hoverinfo="skip",
                ),
                row=2,
                col=2,
            )

    w, h = _heatmap_size(n_row, n_col)
    title_lines = 1
    if isinstance(title, dict):
        title_lines = str(title.get("text", "")).count("<br>") + 1
    elif isinstance(title, str):
        title_lines = title.count("<br>") + 1
    # Cluster legend along the right edge of the heatmap; colorbar further right
    fig.update_layout(
        title=title if isinstance(title, dict) else {"text": title, "x": 0.5, "xanchor": "center"},
        height=h,
        width=w + 40,
        margin=dict(
            l=70 if (has_row and yaxis_title) else 40,
            r=200,
            t=55 + 22 * title_lines if (has_col and xaxis_title) else 40 + 22 * title_lines,
            b=40,
        ),
        showlegend=bool(legend_clusters),
        legend=dict(
            title_text="Cluster",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            tracegroupgap=2,
            itemsizing="constant",
            font=dict(size=11),
        ),
        uirevision="hc-heatmap",
    )
    return fig


def pca_cluster_fig(
    score_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    x_col: str,
    y_col: str,
    z_col: str | None = None,
    *,
    selected: list[int] | None = None,
    bin_a: list[int] | None = None,
    bin_b: list[int] | None = None,
    title: str | dict | None = None,
) -> go.Figure:
    df = score_df.copy()
    df["HC_cluster"] = [cluster_label(c) for c in cluster_labels]
    cats = sorted({int(c) for c in cluster_labels})
    cat_str = [str(c) for c in cats]
    color_map = cluster_color_map(cats)
    df["HC_cluster"] = pd.Categorical(df["HC_cluster"], categories=cat_str, ordered=True)
    df["_click_id"] = df.index.astype(str)
    zcol = z_col if z_col and z_col in df.columns else None
    if title is None:
        title = f"PCA colored by HC cluster (t={len(cats)})"
    common = dict(
        color="HC_cluster",
        hover_name="fileName" if "fileName" in df.columns else "_click_id",
        custom_data=["_click_id"],
        category_orders={"HC_cluster": cat_str},
        color_discrete_map=color_map,
        title=title if isinstance(title, str) else None,
    )
    plot_df = df.reset_index(drop=True)
    if zcol:
        fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=zcol, **common)
        fig.update_traces(hovertemplate="%{hovertext}<extra></extra>")
    else:
        fig = px.scatter(plot_df, x=x_col, y=y_col, **common)
        fig.update_traces(hovertemplate="%{hovertext}<extra></extra>")
        fig = equal_xy_axes(fig, df, x_col, y_col)
    title_lines = 1
    if isinstance(title, dict):
        title_lines = str(title.get("text", "")).count("<br>") + 1
        fig.update_layout(title=title)
    apply_export_layout(
        fig,
        title_lines=title_lines,
        legend=True,
        legend_kwargs={"title_text": "Cluster"},
        uirevision="hc-pca",
    )
    set_a = {str(int(c)) for c in (bin_a or [])}
    set_b = {str(int(c)) for c in (bin_b or [])}
    sel = {str(int(c)) for c in (selected or [])} or (set_a | set_b)
    for tr in fig.data:
        name = str(tr.name) if tr.name is not None else ""
        if name in set_a:
            # Bin A → crosses
            tr.marker.symbol = "cross"
            tr.marker.size = 12
            if hasattr(tr.marker, "line"):
                tr.marker.line = dict(width=1.5, color="#111111")
            else:
                tr.update(marker_line=dict(width=1.5, color="#111111"))
        elif name in set_b:
            # Bin B → filled circles
            tr.marker.symbol = "circle"
            tr.marker.size = 12
            if hasattr(tr.marker, "line"):
                tr.marker.line = dict(width=1.5, color="#111111")
            else:
                tr.update(marker_line=dict(width=1.5, color="#111111"))
        elif sel and name not in sel:
            tr.marker.opacity = 0.35
            tr.marker.size = 7
            tr.marker.symbol = "circle"
    return fig


def dendrogram_colored_fig(
    Z: np.ndarray,
    labels: np.ndarray,
    sample_ids: list[str],
    *,
    selected: list[int] | None = None,
    bin_a: list[int] | None = None,
    bin_b: list[int] | None = None,
) -> go.Figure:
    """Standalone sample dendrogram; click any leaf of a cluster to select that cluster."""
    leaves, leaf_clusters = _leaf_clusters_in_dendro_order(Z, labels)
    a = [int(c) for c in (bin_a or [])]
    b = [int(c) for c in (bin_b or [])]
    set_a = set(a)
    set_b = set(b)
    selected_set = {int(c) for c in (selected if selected is not None else (a + b))}
    fig = go.Figure()
    for xs, ys, color in dendro_polylines(
        Z, orientation="top", leaf_clusters=leaf_clusters
    ):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=color, width=1.5),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    leaf_x = list(range(len(leaves)))
    leaf_y = [0.0] * len(leaves)
    leaf_names = [sample_ids[i] for i in leaves]
    df_leaf = pd.DataFrame(
        {
            "x": leaf_x,
            "y": leaf_y,
            "cluster": [cluster_label(c) for c in leaf_clusters],
            "name": leaf_names,
            "cid": leaf_clusters,
        }
    )
    for cid in sorted(set(leaf_clusters)):
        sub = df_leaf.loc[df_leaf["cid"] == cid]
        in_a = int(cid) in set_a
        in_b = int(cid) in set_b
        is_sel = int(cid) in selected_set
        if in_a:
            symbol = "cross"
            suffix = " (A)"
        elif in_b:
            symbol = "circle"
            suffix = " (B)"
        else:
            symbol = "circle"
            suffix = ""
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=cluster_label(cid) + suffix,
                text=sub["name"],
                customdata=[[int(cid)]] * len(sub),
                hovertemplate="cluster %{customdata[0]}<br>%{text}<extra></extra>",
                marker=dict(
                    symbol=symbol,
                    size=14 if is_sel else 9,
                    color=_cluster_color(cid),
                    line=dict(width=2 if is_sel else 0, color="#111111"),
                ),
            )
        )

    def _brace(cids: list[int]) -> str:
        return "{" + ", ".join(cluster_label(c) for c in cids) + "}"

    sel_txt = f"Selected: {_brace(a)} vs {_brace(b)}"
    fig.update_layout(
        title={
            "text": f"Sample dendrogram (t={len(set(labels))})<br>"
            f"<span style='font-size:12px'>{sel_txt}</span>",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis=dict(visible=False, range=[-0.5, len(leaves) - 0.5]),
        yaxis_title="distance",
        clickmode="event+select",
    )
    apply_export_layout(
        fig,
        title_lines=2,
        height=360,
        legend=True,
        legend_kwargs={"title_text": "Cluster"},
        uirevision=f"hc-dendro-v2-{_brace(a)}-{_brace(b)}-{len(set(labels))}",
    )
    return fig


def volcano_fig_from_results(
    results: pd.DataFrame,
    *,
    title: str | dict,
    neg_log10_padj_threshold: float,
    fold_change_threshold: float,
    xaxis_title: str = "Mean expression difference between clusters",
    mark_genes: set[str] | None = None,
    mark_label: str | None = None,
) -> go.Figure:
    """Plotly volcano matching distances.ipynb scatter + threshold lines."""
    padj_thr = float(neg_log10_padj_threshold)
    fc_thr = float(fold_change_threshold)
    selected = (results["abs_fold_change"] > fc_thr) & (
        results["neg_log10_padj"] > padj_thr
    )
    marked = gene_mark_mask(results, mark_genes)
    fig = go.Figure()
    for mask, color, alpha, name in (
        (~selected & ~marked, "grey", 0.45, "other"),
        (selected & ~marked, "black", 1.0, "selected"),
    ):
        if not mask.any():
            continue
        sub = results.loc[mask]
        fig.add_trace(
            go.Scatter(
                x=sub["fold_change"],
                y=sub["neg_log10_padj"],
                mode="markers",
                marker=dict(size=6, color=color, opacity=alpha),
                name=name,
                customdata=sub["geneID"].astype(str),
                text=sub["geneID"].astype(str),
                hovertemplate="%{text}<br>fc=%{x:.3g}<br>-log10(padj)=%{y:.3g}<extra></extra>",
                showlegend=False,
            )
        )
    n_mark = add_marked_gene_trace(
        fig,
        results,
        "fold_change",
        "neg_log10_padj",
        mark_genes,
        legend_name=mark_label or "marked",
        size=8,
        opacity=1.0,
    )
    fig.add_hline(y=padj_thr, line=dict(dash="dash", color="#808080", width=0.8))
    fig.add_vline(x=fc_thr, line=dict(dash="dash", color="#808080", width=0.8))
    fig.add_vline(x=-fc_thr, line=dict(dash="dash", color="#808080", width=0.8))
    title_dict = (
        title
        if isinstance(title, dict)
        else {"text": title, "x": 0.5, "xanchor": "center"}
    )
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    fig.update_layout(
        title=title_dict,
        xaxis_title=xaxis_title,
        yaxis_title="-log10(padj)",
        # Include thresholds so dashed lines refresh (uirevision freezes layout otherwise)
        clickmode="event+select",
        hovermode="closest",
    )
    apply_export_layout(
        fig,
        title_lines=title_lines,
        width=640,
        height=520,
        legend=bool(n_mark),
        uirevision=f"hc-volcano-{padj_thr:g}-{fc_thr:g}",
    )
    return fig


def _scroll_x(children: list) -> html.Div:
    """Horizontal scroll only (vertical scrolling is on the shared parent)."""
    return html.Div(children, style={"overflowX": "auto", "overflowY": "hidden"})


def detail_from_heatmap_click(
    click_data,
    rt: dict,
    which: str,
    *,
    labels: np.ndarray | None = None,
    t: int | None = None,
) -> html.Div:
    if not click_data or not rt:
        return sample_detail_placeholder()
    point = click_data["points"][0]
    custom = point.get("customdata")
    if not custom or not isinstance(custom, str) or "||" not in custom:
        return html.P("Click a heatmap cell.", className="text-muted small mb-0")
    row_id, col_id = custom.split("||", 1)
    score_df = rt.get("score_df")
    sample_ids = list(rt.get("sample_ids") or [])
    locus_lookup = rt.get("locus_lookup")
    cluster_col = f"maxclust :{t}" if t is not None else "HC_cluster"

    def _cluster_extra(sid: str) -> dict | None:
        if labels is None or sid not in sample_ids:
            return None
        return {cluster_col: int(labels[sample_ids.index(sid)])}

    def _sample_table(sid: str) -> html.Div:
        if score_df is None or sid not in score_df.index:
            return html.Div([html.P(sid, className="small mb-0")])
        return sample_detail_table(
            score_df.loc[sid], scroll=False, extra=_cluster_extra(sid)
        )

    shared_vscroll = {
        "maxHeight": "560px",
        "overflowY": "auto",
        "overflowX": "hidden",
    }

    if which == "ss":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            _scroll_x(
                                [
                                    html.H6("Row sample", className="mb-2"),
                                    _sample_table(row_id),
                                ]
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            _scroll_x(
                                [
                                    html.H6("Column sample", className="mb-2"),
                                    _sample_table(col_id),
                                ]
                            ),
                            md=6,
                        ),
                    ],
                    className="g-2",
                )
            ],
            style=shared_vscroll,
        )
    if which == "gg":
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            _scroll_x(
                                [
                                    html.H6("Row gene", className="mb-2"),
                                    gene_detail_table(row_id, locus_lookup),
                                ]
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            _scroll_x(
                                [
                                    html.H6("Column gene", className="mb-2"),
                                    gene_detail_table(col_id, locus_lookup),
                                ]
                            ),
                            md=6,
                        ),
                    ],
                    className="g-2",
                )
            ],
            style=shared_vscroll,
        )
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        _scroll_x(
                            [
                                html.H6("Sample", className="mb-2"),
                                _sample_table(row_id),
                            ]
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        _scroll_x(
                            [
                                html.H6("Gene", className="mb-2"),
                                gene_detail_table(col_id, locus_lookup),
                            ]
                        ),
                        md=6,
                    ),
                ],
                className="g-2",
            )
        ],
        style=shared_vscroll,
    )
