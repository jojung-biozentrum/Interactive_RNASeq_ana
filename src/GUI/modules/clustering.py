"""Hierarchical clustering: heatmaps + PCA overlay (notebook Hierarchical_clustering.ipynb)."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html, no_update, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from scipy.stats import false_discovery_control
from sklearn.decomposition import PCA

from ..components.folder_browser import pick_save_file_dialog
from ..components.controls import (
    EXPORT_H,
    EXPORT_W,
    fig_size_controls,
    set_fig_size,
)
from ..components.gene_meta_mark import (
    entry_dropdown_options,
    genes_with_meta_entry,
    locus_mark_columns,
    mark_controls,
)
from ..components.sample_detail import (
    gene_detail_placeholder,
    gene_detail_table,
    sample_detail_placeholder,
    sample_detail_table,
)
from ..data_store import SessionData, session_from_store
from .hc_plots import (
    detail_from_heatmap_click,
    dendrogram_colored_fig,
    heatmap_with_dendro,
    pca_cluster_fig,
    volcano_fig_from_results,
    _cluster_color,
    _leaf_clusters_in_dendro_order,
    cluster_label,
)
from .pca_classifier import save_classifier_celov
from src.biocyc.celov_multiomics_post import load_locus_lookup
from src.GUI.project import resolve_celov_id_col


_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "hc_plot"},
    "displaylogo": False,
}

_LINKAGE_METHODS = [
    {"label": "ward", "value": "ward"},
    {"label": "average", "value": "average"},
    {"label": "complete", "value": "complete"},
    {"label": "single", "value": "single"},
]

_HC_RUNTIME: dict = {}


def _plotly_title(*lines: str) -> dict:
    """Centered multi-line Plotly title."""
    text = "<br>".join(line for line in lines if line is not None and str(line).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


def _active_dataset_name(session_blob) -> str:
    names = (session_blob or {}).get("active_datasets") or []
    return str(names[0]) if names else "dataset"


def _brace_clusters(cids: list[int]) -> str:
    return "{" + ", ".join(cluster_label(c) for c in cids) + "}"


def _heatmap_title(kind: str, dataset: str, t: int) -> dict:
    if kind == "ss":
        line1 = "Pairwise expression differences within samples"
    elif kind == "gg":
        line1 = "Pairwise expression differences within genes"
    else:
        line1 = "Pairwise expression differences between samples and genes"
    return _plotly_title(
        line1,
        f"in {dataset} normalization ({t} clusters)",
    )


def _linkage(X: np.ndarray, method: str = "ward") -> np.ndarray:
    """Notebook: ``hierarchy.linkage(normalizedCounts.values, method='ward')``."""
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 rows for hierarchical clustering.")
    return hierarchy.linkage(X, method=method)


def _cut_clusters(Z: np.ndarray, t: int) -> np.ndarray:
    t = max(2, min(int(t), Z.shape[0] + 1))
    return hierarchy.fcluster(Z, t=t, criterion="maxclust")


def volcano_cluster_vs_cluster(
    expr_a: pd.DataFrame,
    expr_b: pd.DataFrame,
    title: str | dict,
    neg_log10_padj_threshold: float,
    fold_change_threshold: float,
    *,
    center: str = "mean",
) -> tuple[pd.DataFrame, go.Figure]:
    """Port of ``volcano_biofilm_vs_lc`` from distances.ipynb (genes × samples).

    ``center='mean'`` is the notebook default (option 1); ``center='median'`` uses
    per-gene medians for the difference instead.
    """
    if center == "median":
        fold_change = expr_a.median(axis=1) - expr_b.median(axis=1)
        x_label = "Median expression difference between clusters"
    else:
        fold_change = expr_a.mean(axis=1) - expr_b.mean(axis=1)
        x_label = "Mean expression difference between clusters"
    p_values = pd.Series(
        {
            gene: stats.ttest_ind(
                expr_a.loc[gene],
                expr_b.loc[gene],
                equal_var=False,
                nan_policy="omit",
            ).pvalue
            for gene in expr_a.index
        }
    ) 
    # Welch's t-test for each gene: we don't have a lot of replicates, 
    # so we use the Welch's t-test instead of DESeq2 or other statistically more elaborate models
    # The basic assumption is that a gene is Gaussian distributed within each group (different variances)
    # NOTE: maybe, later on, we can do some more sophisticated model for biofilms
    padj = pd.Series(
        false_discovery_control(p_values.fillna(1).values),
        index=p_values.index,
    ) # multiple testing correction (FDR) using the Benjamini-Hochberg procedure

    pi_values = fold_change * padj

    results = (
        pd.DataFrame(
            {
                "geneID": expr_a.index.astype(str),
                "fold_change": fold_change.values,
                "abs_fold_change": np.abs(fold_change.values),
                "padj": padj.values,
                "pi_value": pi_values.values,
            }
        )
        .sort_values("abs_fold_change", ascending=False)
    )

    neg_log10_padj = -np.log10(results["padj"].clip(lower=1e-300))
    results["neg_log10_padj"] = neg_log10_padj

    fig = volcano_fig_from_results(
        results,
        title=title,
        neg_log10_padj_threshold=neg_log10_padj_threshold,
        fold_change_threshold=fold_change_threshold,
        xaxis_title=x_label,
    )
    return results, fig


def _expr_genes_x_samples(rt: dict) -> pd.DataFrame:
    """Notebook orientation: genes × samples from cached sample×gene matrix."""
    return pd.DataFrame(
        rt["X"].T,
        index=pd.Index(rt["gene_ids"], name="geneID"),
        columns=rt["sample_ids"],
    )


def _volcano_weighed_for_celov(results: pd.DataFrame, score: str) -> pd.DataFrame:
    """Build gene_weight table for Celov; direction from expression difference sign."""
    fold = results["fold_change"].astype(float)
    neg = results["neg_log10_padj"].astype(float)
    if score == "neg_log10_padj":
        weight = np.sign(fold.replace(0, np.nan)).fillna(0.0) * neg
    elif score == "product":
        weight = fold * neg
    else:
        weight = fold
    return pd.DataFrame(
        {"geneID": results["geneID"].astype(str).values, "gene_weight": weight.values}
    )


def _active_dataset_entry(session_blob, project_blob) -> dict | None:
    active = (session_blob or {}).get("active_datasets") or []
    name = active[0] if active else None
    if not name:
        return None
    return next(
        (d for d in (project_blob or {}).get("datasets", []) if d.get("name") == name),
        None,
    )


def _locus_path_from_session(session_blob, project_blob) -> str | None:
    """Resolve locus lookup path from the first active dataset (same as PCA)."""
    entry = _active_dataset_entry(session_blob, project_blob)
    if not entry:
        return None
    locus = entry.get("locus_lookup") or ""
    if not locus:
        return None
    root = (project_blob or {}).get("root")
    if root and not Path(locus).is_absolute():
        return str(Path(root) / locus)
    return str(locus)


def _empty_bin_sel() -> dict:
    return {"a": [], "b": [], "target": "a"}


def _normalize_bin_sel(data) -> dict:
    if not isinstance(data, dict):
        return _empty_bin_sel()
    out = _empty_bin_sel()
    out["a"] = [int(x) for x in (data.get("a") or [])]
    out["b"] = [int(x) for x in (data.get("b") or [])]
    out["target"] = "b" if data.get("target") == "b" else "a"
    return out


def _flat_selected(data) -> list[int]:
    d = _normalize_bin_sel(data)
    return sorted(set(d["a"]) | set(d["b"]))


def _selected_clusters_banner(selected) -> html.Div:
    d = _normalize_bin_sel(selected)

    def _bin_badges(cids: list[int], name: str) -> list:
        parts = [html.Strong(f"{name}: ", className="me-1")]
        if not cids:
            parts.append(html.Span("(empty)", className="text-muted"))
            return parts
        for i, cid in enumerate(cids):
            if i:
                parts.append(html.Span(" ", className="me-1"))
            parts.append(
                html.Span(
                    f"{cluster_label(cid)}",
                    className="badge me-1",
                    style={
                        "backgroundColor": _cluster_color(cid),
                        "color": "#fff",
                        "fontSize": "0.95rem",
                        "padding": "0.35em 0.65em",
                    },
                )
            )
        return parts

    return html.Div(
        [
            html.Div(_bin_badges(d["a"], "Bin A"), className="mb-1"),
            html.Div(_bin_badges(d["b"], "Bin B"), className="mb-0"),
        ]
    )


def first_homogeneous_maxclust(
    Z: np.ndarray,
    meta_values: pd.Series,
    target: str,
) -> tuple[int | None, int | None]:
    """First ``maxclust`` t with a cluster whose members all equal ``target``."""
    n = len(meta_values)
    target_s = str(target)
    vals = meta_values.astype(str)
    for t in range(2, n + 1):
        labels = _cut_clusters(Z, t)
        for cid in range(1, t + 1):
            mask = labels == cid
            if not mask.any():
                continue
            if (vals.iloc[np.where(mask)[0]] == target_s).all():
                return t, int(cid)
    return None, None


def _run_hc(session: SessionData, *, method: str = "ward") -> dict:
    """Cluster samples and genes on the unscaled count matrix (notebook).

    Linkage + PCA + distance matrices are computed once and stored; tab / cut
    changes only rebuild Plotly figures from this cache.
    """
    if session.expression is None or session.metadata is None:
        raise RuntimeError("No session data")
    expr = session.expression
    X = session.numeric_matrix()
    sample_ids = list(expr.index.astype(str))
    gene_ids = list(expr.columns.astype(str))
    meta = session.metadata

    Z_samples = _linkage(X, method=method)
    Z_genes = _linkage(X.T, method=method)
    dist_samples = squareform(pdist(X, metric="euclidean"))
    dist_genes = squareform(pdist(X.T, metric="euclidean"))

    pca = PCA()
    pca.fit(X)
    scores = pca.transform(X)
    n_pcs = min(scores.shape[1], 50)
    pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]
    score_df = pd.DataFrame(scores[:, :n_pcs], index=sample_ids, columns=pc_cols)
    score_df = score_df.join(meta)

    return {
        "Z_samples": Z_samples,
        "Z_genes": Z_genes,
        "X": X,
        "dist_samples": dist_samples,
        "dist_genes": dist_genes,
        "sample_ids": sample_ids,
        "gene_ids": gene_ids,
        "score_df": score_df,
        "method": method,
        "n_samples": len(sample_ids),
        "n_genes": len(gene_ids),
    }


def _sample_distance_fig(rt: dict, sample_labels: np.ndarray, *, dataset: str, t: int) -> go.Figure:
    ids = rt["sample_ids"]
    Z = rt["Z_samples"]
    dist = rt["dist_samples"]
    _, leaf_c = _leaf_clusters_in_dendro_order(Z, sample_labels)
    return heatmap_with_dendro(
        dist,
        ids,
        ids,
        Z,
        Z,
        title=_heatmap_title("ss", dataset, t),
        row_leaf_clusters=leaf_c,
        col_leaf_clusters=leaf_c,
        xaxis_title="Samples",
        yaxis_title="Samples",
    )


def _gene_distance_fig(rt: dict, *, dataset: str, t: int) -> go.Figure:
    ids = rt["gene_ids"]
    Z = rt["Z_genes"]
    return heatmap_with_dendro(
        rt["dist_genes"],
        ids,
        ids,
        Z,
        Z,
        title=_heatmap_title("gg", dataset, t),
        xaxis_title="Genes",
        yaxis_title="Genes",
    )


def _sample_gene_fig(rt: dict, sample_labels: np.ndarray, *, dataset: str, t: int) -> go.Figure:
    Z = rt["Z_samples"]
    _, leaf_c = _leaf_clusters_in_dendro_order(Z, sample_labels)
    return heatmap_with_dendro(
        rt["X"],
        rt["sample_ids"],
        rt["gene_ids"],
        Z,
        rt["Z_genes"],
        title=_heatmap_title("sg", dataset, t),
        colorbar_title="Expression",
        row_leaf_clusters=leaf_c,
        xaxis_title="Genes",
        yaxis_title="Samples",
    )


class ClusteringModule:
    id = "hc"
    label = "Hierarchical clustering"

    def layout(self):
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Linkage method"),
                                dcc.Dropdown(
                                    id="hc-method",
                                    options=_LINKAGE_METHODS,
                                    value="ward",
                                    clearable=False,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run clustering", id="hc-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(
                    id="hc-cut-section",
                    children=[
                        html.Hr(),
                        html.H6("Cluster cut (samples)"),
                        html.P(
                            "Set the total number of clusters, or find the first pure cluster "
                            "for a metadata column/value and color the clusters at that "
                            "clustering step.",
                            className="text-muted small mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Total number of clusters"),
                                        dbc.Input(
                                            id="hc-maxclust",
                                            type="number",
                                            value=2,
                                            min=2,
                                            step=1,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Br(),
                                        dbc.Button(
                                            "Apply",
                                            id="hc-maxclust-apply",
                                            color="primary",
                                            outline=True,
                                        ),
                                    ],
                                    md=1,
                                ),
                                dbc.Col(html.Div("— or —", className="text-muted mt-4"), md=1),
                                dbc.Col(
                                    [
                                        html.Label("Metadata column"),
                                        dcc.Dropdown(id="hc-homo-col", clearable=True),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Value"),
                                        dcc.Dropdown(id="hc-homo-val", clearable=True),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Br(),
                                        dbc.Button(
                                            "Find pure cluster step",
                                            id="hc-homo-find",
                                            color="secondary",
                                            outline=True,
                                        ),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                        html.Div(id="hc-homo-status", className="text-muted small mb-2"),
                        dcc.Store(id="hc-maxclust-applied", data=2),
                    ],
                ),
                html.Hr(),
                html.H6("Heatmap"),
                dcc.Tabs(
                    id="hc-heatmap-tabs",
                    value="ss",
                    children=[
                        dcc.Tab(label="Samples × samples", value="ss"),
                        dcc.Tab(label="Genes × genes", value="gg"),
                        dcc.Tab(label="Samples × genes", value="sg"),
                    ],
                ),
                fig_size_controls(
                    "hc-heat",
                    default_width=760,
                    default_height=640,
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                dcc.Graph(id="hc-heatmap", figure={}, config=_GRAPH_CONFIG),
                                type="default",
                            ),
                            md=8,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H6("Click detail", className="mb-2"),
                                    html.P(
                                        "Samples: sample metadata. Genes: locus-lookup columns.",
                                        className="text-muted small mb-2",
                                    ),
                                    html.Div(
                                        id="hc-heat-detail",
                                        children=sample_detail_placeholder(),
                                    ),
                                ],
                                className="border rounded p-2 bg-light",
                            ),
                            md=4,
                        ),
                    ],
                    className="g-2 mb-3",
                ),
                html.H6("Export cluster assignments"),
                html.P(
                    "Exports sample metadata with an added column named "
                    "maxclust :{t} (current cut).",
                    className="text-muted small mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="hc-export-path", type="text"),
                                    dbc.Button(
                                        "Browse…",
                                        id="hc-export-browse",
                                        color="info",
                                        outline=True,
                                    ),
                                ]
                            ),
                            md=8,
                        ),
                        dbc.Col(
                            dbc.Button(
                                "Export CSV", id="hc-export", color="secondary"
                            ),
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="hc-status", className="text-muted small mb-2"),
                html.Div(
                    id="hc-pca-section",
                    children=[
                        html.Hr(),
                        html.H6("PCA + dendrogram (samples × samples)"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("X"),
                                        dcc.Dropdown(id="hc-pca-x", value="PC1", clearable=False),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Y"),
                                        dcc.Dropdown(id="hc-pca-y", value="PC2", clearable=False),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Z (optional 3D)"),
                                        dcc.Dropdown(id="hc-pca-z", clearable=True),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    fig_size_controls(
                                        "hc-pca",
                                        default_width=EXPORT_W,
                                        default_height=EXPORT_H,
                                        heading="PCA size (px)",
                                    ),
                                    md=5,
                                ),
                                dbc.Col(
                                    fig_size_controls(
                                        "hc-dendro",
                                        default_width=EXPORT_W,
                                        default_height=360,
                                        heading="Dendrogram size (px)",
                                    ),
                                    md=4,
                                ),
                            ],
                            className="g-2 mb-1",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id="hc-pca", figure={}, config=_GRAPH_CONFIG),
                                    md=5,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="hc-dendro", figure={}, config=_GRAPH_CONFIG
                                        ),
                                        html.P(
                                            "Select dendrogram clusters into Bin A or Bin B "
                                            "for volcano contrast.",
                                            className="text-muted small mb-1 mt-2",
                                        ),
                                        html.Label("Assign dendrogram clicks to"),
                                        dcc.RadioItems(
                                            id="hc-volcano-bin-target",
                                            options=[
                                                {"label": " Bin A", "value": "a"},
                                                {"label": " Bin B", "value": "b"},
                                            ],
                                            value="a",
                                            inline=True,
                                            className="mb-2",
                                        ),
                                        html.Div(
                                            id="hc-volcano-selected",
                                            children=_selected_clusters_banner(
                                                _empty_bin_sel()
                                            ),
                                            className="mb-1",
                                        ),
                                        dbc.Button(
                                            "Clear selection",
                                            id="hc-volcano-clear",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.H6("Sample metadata", className="mb-2"),
                                            html.Div(
                                                id="hc-pca-detail",
                                                children=sample_detail_placeholder(),
                                            ),
                                        ],
                                        className="border rounded p-2 bg-light",
                                    ),
                                    md=3,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                        html.Hr(),
                        html.H6("Cluster contrast volcano"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Difference"),
                                        dcc.Dropdown(
                                            id="hc-volcano-center",
                                            options=[
                                                {
                                                    "label": "means",
                                                    "value": "mean",
                                                },
                                                {
                                                    "label": "medians",
                                                    "value": "median",
                                                },
                                            ],
                                            value="mean",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("−log10(padj) threshold"),
                                        dbc.Input(
                                            id="hc-volcano-padj",
                                            type="number",
                                            value=2,
                                            step=0.1,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("|fold change| threshold"),
                                        dbc.Input(
                                            id="hc-volcano-fc",
                                            type="number",
                                            value=1,
                                            step=0.1,
                                        ),
                                    ],
                                    md=2,
                                ),
                                dbc.Col(
                                    [
                                        html.Br(),
                                        dbc.Button(
                                            "Run volcano",
                                            id="hc-volcano-run",
                                            color="primary",
                                        ),
                                    ],
                                    md=2,
                                ),
                            ],
                            className="g-2 mb-2",
                        ),
                        html.Div(id="hc-volcano-status", className="text-muted small mb-2"),
                        mark_controls(
                            col_id="hc-volcano-mark-col",
                            entry_id="hc-volcano-mark-entry",
                            wrap_id="hc-volcano-mark-wrap",
                        ),
                        fig_size_controls(
                            "hc-volcano",
                            default_width=640,
                            default_height=520,
                        ),
                        dcc.Loading(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="hc-volcano",
                                            figure={},
                                            config=_GRAPH_CONFIG,
                                        ),
                                        md=8,
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.H6("Gene metadata", className="mb-2"),
                                                html.P(
                                                    "Click a gene on the volcano plot.",
                                                    className="text-muted small mb-2",
                                                ),
                                                html.Div(
                                                    id="hc-volcano-gene-detail",
                                                    children=gene_detail_placeholder(),
                                                ),
                                            ],
                                            className="border rounded p-2 bg-light",
                                        ),
                                        md=4,
                                    ),
                                ],
                                className="g-2 align-items-start",
                            ),
                            type="default",
                        ),
                        dcc.Store(id="hc-volcano-last-gene", data=None),
                        html.H6("Save volcano genes (Celov)", className="mt-3"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Score"),
                                        dcc.Dropdown(
                                            id="hc-volcano-celov-score",
                                            options=[
                                                {
                                                    "label": "expression difference",
                                                    "value": "fold_change",
                                                },
                                                {
                                                    "label": "−log10(padj)",
                                                    "value": "neg_log10_padj",
                                                },
                                                {
                                                    "label": "product of both",
                                                    "value": "product",
                                                },
                                            ],
                                            value="fold_change",
                                            clearable=False,
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Genes"),
                                        dcc.RadioItems(
                                            id="hc-volcano-celov-mode",
                                            options=[
                                                {
                                                    "label": "up & down",
                                                    "value": "up_and_down",
                                                },
                                                {"label": "up", "value": "up"},
                                                {"label": "down", "value": "down"},
                                                {
                                                    "label": "all together",
                                                    "value": "all",
                                                },
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
                                                dbc.Input(
                                                    id="hc-volcano-celov-out", type="text"
                                                ),
                                                dbc.Button(
                                                    "Browse…",
                                                    id="hc-volcano-celov-browse",
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
                            id="hc-volcano-celov-save",
                            color="secondary",
                            className="mb-2",
                        ),
                        html.Div(
                            id="hc-volcano-celov-status",
                            className="text-muted small mb-2",
                        ),
                        dcc.Store(id="hc-cluster-sel", data=_empty_bin_sel()),
                    ],
                ),
                dcc.Store(id="hc-cache"),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output("hc-homo-col", "options"),
            Input("session-store", "data"),
            Input("hc-cache", "data"),
        )
        def _fill_meta_cols(session_blob, cache):
            cols = list((session_blob or {}).get("meta_columns", []))
            if _HC_RUNTIME.get("score_df") is not None:
                df = _HC_RUNTIME["score_df"]
                pc = {c for c in df.columns if c.startswith("PC") and c[2:].isdigit()}
                cols = [c for c in df.columns if c not in pc]
            return [{"label": c, "value": c} for c in cols]

        @app.callback(
            Output("hc-homo-val", "options"),
            Input("hc-homo-col", "value"),
            State("session-store", "data"),
        )
        def _fill_vals(col, session_blob):
            session = session_from_store(session_blob)
            if not col or session.metadata is None or col not in session.metadata.columns:
                return []
            vals = sorted(session.metadata[col].dropna().astype(str).unique())
            return [{"label": v, "value": v} for v in vals]

        @app.callback(
            Output("hc-cache", "data"),
            Output("hc-status", "children"),
            Output("hc-pca-x", "options"),
            Output("hc-pca-y", "options"),
            Output("hc-pca-z", "options"),
            Output("hc-pca-x", "value"),
            Output("hc-pca-y", "value"),
            Output("hc-pca-z", "value"),
            Output("hc-maxclust", "max"),
            Output("hc-maxclust", "value", allow_duplicate=True),
            Output("hc-maxclust-applied", "data", allow_duplicate=True),
            Input("hc-run", "n_clicks"),
            State("session-store", "data"),
            State("project-store", "data"),
            State("hc-method", "value"),
            prevent_initial_call=True,
        )
        def _compute(n_clicks, session_blob, project_blob, method):
            session = session_from_store(session_blob)
            if not session.ready:
                return (
                    no_update,
                    session.error or "Load a dataset first.",
                    [],
                    [],
                    [],
                    None,
                    None,
                    None,
                    no_update,
                    no_update,
                    no_update,
                )
            try:
                rt = _run_hc(session, method=method or "ward")
                locus_path = _locus_path_from_session(session_blob, project_blob)
                rt["locus_lookup"] = None
                if locus_path:
                    try:
                        rt["locus_lookup"] = load_locus_lookup(locus_path)
                    except Exception as locus_exc:  # noqa: BLE001
                        rt["locus_lookup"] = None
                        locus_note = f" (locus lookup not loaded: {locus_exc})"
                    else:
                        locus_note = f" (locus lookup: {len(rt['locus_lookup'])} genes)"
                else:
                    locus_note = " (no locus lookup on dataset)"
                _HC_RUNTIME.clear()
                _HC_RUNTIME.update(rt)
                pcs = [c for c in rt["score_df"].columns if c.startswith("PC") and c[2:].isdigit()]
                opts = [{"label": c, "value": c} for c in pcs]
                cache = {
                    "ready": True,
                    "n_samples": rt["n_samples"],
                    "n_genes": rt["n_genes"],
                    "method": rt["method"],
                }
                return (
                    cache,
                    f"Clustering done ({rt['method']}): {rt['n_samples']} samples × "
                    f"{rt['n_genes']} genes.{locus_note}",
                    opts,
                    opts,
                    opts,
                    "PC1" if "PC1" in pcs else (pcs[0] if pcs else None),
                    "PC2" if "PC2" in pcs else (pcs[1] if len(pcs) > 1 else None),
                    None,
                    max(2, rt["n_samples"]),
                    2,
                    2,
                )
            except Exception as exc:  # noqa: BLE001
                return (
                    no_update,
                    f"HC error: {exc}",
                    [],
                    [],
                    [],
                    None,
                    None,
                    None,
                    no_update,
                    no_update,
                    no_update,
                )

        @app.callback(
            Output("hc-maxclust-applied", "data"),
            Output("hc-homo-status", "children", allow_duplicate=True),
            Input("hc-maxclust-apply", "n_clicks"),
            State("hc-maxclust", "value"),
            prevent_initial_call=True,
        )
        def _apply_maxclust(n_clicks, t):
            if "Z_samples" not in _HC_RUNTIME:
                return no_update, "Run clustering first."
            n = _HC_RUNTIME["n_samples"]
            t_use = max(2, min(int(t or 2), n))
            return t_use, f"Colored clusters at total number of clusters = {t_use}."

        @app.callback(
            Output("hc-maxclust", "value"),
            Output("hc-maxclust-applied", "data", allow_duplicate=True),
            Output("hc-homo-status", "children"),
            Input("hc-homo-find", "n_clicks"),
            State("hc-homo-col", "value"),
            State("hc-homo-val", "value"),
            prevent_initial_call=True,
        )
        def _find_homo(n_clicks, col, val):
            if "Z_samples" not in _HC_RUNTIME:
                return no_update, no_update, "Run clustering first."
            if not col or val is None or val == "":
                return no_update, no_update, "Choose metadata column and value."
            score_df = _HC_RUNTIME["score_df"]
            if col not in score_df.columns:
                return no_update, no_update, f"Column {col!r} not in metadata."
            t, cid = first_homogeneous_maxclust(
                _HC_RUNTIME["Z_samples"], score_df[col], str(val)
            )
            if t is None:
                return no_update, no_update, f"No pure cluster step found for {col}={val!r}."
            return (
                t,
                t,
                f"Step t = {t} (first pure cluster {cid}) for {col}={val!r} — clusters colored.",
            )

        @app.callback(
            Output("hc-pca-section", "style"),
            Input("hc-heatmap-tabs", "value"),
        )
        def _toggle_pca(tab):
            if tab == "ss":
                return {"display": "block"}
            return {"display": "none"}

        @app.callback(
            Output("hc-heatmap", "figure"),
            Input("hc-cache", "data"),
            Input("hc-heatmap-tabs", "value"),
            Input("hc-maxclust-applied", "data"),
            Input("hc-heat-fig-w", "value"),
            Input("hc-heat-fig-h", "value"),
            Input("session-store", "data"),
        )
        def _plot_heatmap(cache, tab, t, fig_w, fig_h, session_blob):
            empty = go.Figure()
            if not cache or "Z_samples" not in _HC_RUNTIME:
                return empty
            rt = _HC_RUNTIME
            dataset = _active_dataset_name(session_blob)
            try:
                n = rt["n_samples"]
                t = max(2, min(int(t or 2), n))
                labels = _cut_clusters(rt["Z_samples"], t)
                if tab == "gg":
                    fig = _gene_distance_fig(rt, dataset=dataset, t=t)
                elif tab == "sg":
                    fig = _sample_gene_fig(rt, labels, dataset=dataset, t=t)
                else:
                    fig = _sample_distance_fig(rt, labels, dataset=dataset, t=t)
                return set_fig_size(
                    fig, fig_w, fig_h, default_width=760, default_height=640
                )
            except Exception as exc:  # noqa: BLE001
                err = go.Figure()
                err.add_annotation(text=f"Heatmap error: {exc}", showarrow=False)
                return err

        @app.callback(
            Output("hc-heat-detail", "children"),
            Input("hc-heatmap", "clickData"),
            Input("hc-heatmap-tabs", "value"),
            Input("hc-cache", "data"),
            Input("hc-maxclust-applied", "data"),
        )
        def _heat_detail(click, tab, cache, t):
            triggered = callback_context.triggered_id
            if triggered in ("hc-cache", "hc-heatmap-tabs") or not cache:
                return sample_detail_placeholder()
            labels = None
            t_use = None
            if "Z_samples" in _HC_RUNTIME:
                n = _HC_RUNTIME["n_samples"]
                t_use = max(2, min(int(t or 2), n))
                labels = _cut_clusters(_HC_RUNTIME["Z_samples"], t_use)
            return detail_from_heatmap_click(
                click, _HC_RUNTIME, tab or "ss", labels=labels, t=t_use
            )

        @app.callback(
            Output("hc-cluster-sel", "data"),
            Output("hc-volcano-status", "children", allow_duplicate=True),
            Output("hc-volcano-selected", "children"),
            Input("hc-dendro", "clickData"),
            Input("hc-volcano-clear", "n_clicks"),
            Input("hc-maxclust-applied", "data"),
            Input("hc-cache", "data"),
            Input("hc-volcano-bin-target", "value"),
            State("hc-cluster-sel", "data"),
            prevent_initial_call=True,
        )
        def _select_clusters(click, n_clear, t, cache, bin_target, current):
            triggered = callback_context.triggered_id
            if triggered in ("hc-volcano-clear", "hc-maxclust-applied", "hc-cache"):
                empty = _empty_bin_sel()
                empty["target"] = "b" if bin_target == "b" else "a"
                return (
                    empty,
                    "Assign clusters to Bin A and Bin B via the dendrogram.",
                    _selected_clusters_banner(empty),
                )
            d = _normalize_bin_sel(current)
            if triggered == "hc-volcano-bin-target":
                d["target"] = "b" if bin_target == "b" else "a"
                return d, no_update, _selected_clusters_banner(d)
            if not click:
                return no_update, no_update, no_update
            point = click["points"][0]
            raw = point.get("customdata")
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            if raw is None:
                return (
                    no_update,
                    "Click a colored cluster leaf (not a branch).",
                    no_update,
                )
            cid = int(raw)
            target = "b" if (bin_target or d.get("target")) == "b" else "a"
            other = "b" if target == "a" else "a"
            d["target"] = target
            # Toggle within target bin; remove from other bin if present
            if cid in d[other]:
                d[other] = [x for x in d[other] if x != cid]
            if cid in d[target]:
                d[target] = [x for x in d[target] if x != cid]
            else:
                d[target] = sorted(set(d[target]) | {cid})
            if not d["a"] and not d["b"]:
                msg = "Assign clusters to Bin A and Bin B via the dendrogram."
            elif not d["a"] or not d["b"]:
                msg = "Fill both bins (at least one cluster each), then run volcano."
            else:
                msg = (
                    f"Bin A ({len(d['a'])} cluster(s)) vs Bin B ({len(d['b'])} cluster(s)) "
                    "— run volcano."
                )
            return d, msg, _selected_clusters_banner(d)

        @app.callback(
            Output("hc-pca", "figure"),
            Output("hc-dendro", "figure"),
            Input("hc-cache", "data"),
            Input("hc-maxclust-applied", "data"),
            Input("hc-pca-x", "value"),
            Input("hc-pca-y", "value"),
            Input("hc-pca-z", "value"),
            Input("hc-heatmap-tabs", "value"),
            Input("hc-cluster-sel", "data"),
            Input("hc-pca-fig-w", "value"),
            Input("hc-pca-fig-h", "value"),
            Input("hc-dendro-fig-w", "value"),
            Input("hc-dendro-fig-h", "value"),
            Input("session-store", "data"),
        )
        def _plot_pca(
            cache,
            t,
            x_col,
            y_col,
            z_col,
            tab,
            selected,
            pca_w,
            pca_h,
            dendro_w,
            dendro_h,
            session_blob,
        ):
            empty = go.Figure()
            if tab != "ss" or not cache or "Z_samples" not in _HC_RUNTIME:
                return empty, empty
            rt = _HC_RUNTIME
            n = rt["n_samples"]
            t = max(2, min(int(t or 2), n))
            labels = _cut_clusters(rt["Z_samples"], t)
            score_df = rt["score_df"]
            pcs = [c for c in score_df.columns if c.startswith("PC") and c[2:].isdigit()]
            x_col = x_col if x_col in score_df.columns else (pcs[0] if pcs else None)
            y_col = y_col if y_col in score_df.columns else (pcs[1] if len(pcs) > 1 else x_col)
            if not x_col or not y_col:
                return empty, empty
            sel = _flat_selected(selected)
            bins = _normalize_bin_sel(selected)
            dataset = _active_dataset_name(session_blob)
            pca_title = _plotly_title(
                "PCA colored by hierarchical clusters",
                f"in {dataset} ({t} clusters)",
            )
            try:
                pca_fig = pca_cluster_fig(
                    score_df,
                    labels,
                    x_col,
                    y_col,
                    z_col,
                    selected=sel,
                    bin_a=bins["a"],
                    bin_b=bins["b"],
                    title=pca_title,
                )
                dendro_fig = dendrogram_colored_fig(
                    rt["Z_samples"],
                    labels,
                    rt["sample_ids"],
                    selected=sel,
                    bin_a=bins["a"],
                    bin_b=bins["b"],
                )
                return (
                    set_fig_size(pca_fig, pca_w, pca_h),
                    set_fig_size(
                        dendro_fig, dendro_w, dendro_h, default_height=360
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                err = go.Figure()
                err.add_annotation(text=f"PCA error: {exc}", showarrow=False)
                return err, err

        @app.callback(
            Output("hc-pca-detail", "children"),
            Input("hc-pca", "clickData"),
            Input("hc-cache", "data"),
            Input("hc-maxclust-applied", "data"),
        )
        def _pca_detail(click, cache, t):
            triggered = callback_context.triggered_id
            if triggered == "hc-cache" or not cache or "score_df" not in _HC_RUNTIME:
                return sample_detail_placeholder()
            if not click:
                return sample_detail_placeholder()
            point = click["points"][0]
            cid = point.get("customdata")
            if isinstance(cid, (list, tuple)):
                cid = cid[0] if cid else None
            df = _HC_RUNTIME["score_df"]
            if cid is None or str(cid) not in df.index:
                return sample_detail_placeholder()
            sid = str(cid)
            extra = None
            n = _HC_RUNTIME["n_samples"]
            t_use = max(2, min(int(t or 2), n))
            labels = _cut_clusters(_HC_RUNTIME["Z_samples"], t_use)
            sample_ids = _HC_RUNTIME["sample_ids"]
            if sid in sample_ids:
                extra = {f"maxclust :{t_use}": int(labels[sample_ids.index(sid)])}
            return sample_detail_table(df.loc[sid], extra=extra)

        @app.callback(
            Output("hc-volcano", "figure"),
            Output("hc-volcano-status", "children"),
            Output("hc-volcano-last-gene", "data", allow_duplicate=True),
            Output("hc-volcano-mark-wrap", "style"),
            Output("hc-volcano-mark-col", "options"),
            Output("hc-volcano-mark-col", "value"),
            Input("hc-volcano-run", "n_clicks"),
            State("hc-cluster-sel", "data"),
            State("hc-maxclust-applied", "data"),
            State("hc-volcano-padj", "value"),
            State("hc-volcano-fc", "value"),
            State("hc-volcano-center", "value"),
            State("hc-volcano-fig-w", "value"),
            State("hc-volcano-fig-h", "value"),
            State("session-store", "data"),
            prevent_initial_call=True,
        )
        def _run_volcano(
            n_clicks, selected, t, padj_thr, fc_thr, center, fig_w, fig_h, session_blob
        ):
            empty = go.Figure()
            hide = {"display": "none"}
            show = {"display": "block"}
            no_mark = (hide, [], None)
            if "Z_samples" not in _HC_RUNTIME:
                return empty, "Run clustering first.", None, *no_mark
            bins = _normalize_bin_sel(selected)
            if not bins["a"] or not bins["b"]:
                return (
                    empty,
                    "Put at least one cluster in Bin A and one in Bin B.",
                    None,
                    *no_mark,
                )
            rt = _HC_RUNTIME
            n = rt["n_samples"]
            t = max(2, min(int(t or 2), n))
            labels = _cut_clusters(rt["Z_samples"], t)
            set_a = set(bins["a"])
            set_b = set(bins["b"])
            ids_a = [
                sid for sid, lab in zip(rt["sample_ids"], labels) if int(lab) in set_a
            ]
            ids_b = [
                sid for sid, lab in zip(rt["sample_ids"], labels) if int(lab) in set_b
            ]
            if len(ids_a) < 2 or len(ids_b) < 2:
                return (
                    empty,
                    f"Need ≥2 samples per bin (got {len(ids_a)} vs {len(ids_b)}).",
                    None,
                    *no_mark,
                )
            expr = _expr_genes_x_samples(rt)
            expr_a = expr[ids_a]
            expr_b = expr[ids_b]
            padj_thr = float(padj_thr if padj_thr is not None else 2)
            fc_thr = float(fc_thr if fc_thr is not None else 1)
            center = "median" if center == "median" else "mean"
            dataset = _active_dataset_name(session_blob)
            title = _plotly_title(
                f"Volcano plot for cluster {_brace_clusters(bins['a'])} vs cluster "
                f"{_brace_clusters(bins['b'])}",
                f"in {dataset} ({t} clusters)",
            )
            a_lab = ",".join(str(c) for c in bins["a"])
            b_lab = ",".join(str(c) for c in bins["b"])
            try:
                results, _fig = volcano_cluster_vs_cluster(
                    expr_a,
                    expr_b,
                    title,
                    padj_thr,
                    fc_thr,
                    center=center,
                )
            except Exception as exc:  # noqa: BLE001
                _HC_RUNTIME.pop("volcano_results", None)
                _HC_RUNTIME.pop("volcano_plot_meta", None)
                return empty, f"Volcano error: {exc}", None, *no_mark

            x_label = (
                "Median expression difference between clusters"
                if center == "median"
                else "Mean expression difference between clusters"
            )
            _HC_RUNTIME["volcano_results"] = results
            _HC_RUNTIME["volcano_plot_meta"] = {
                "title": title,
                "padj_thr": padj_thr,
                "fc_thr": fc_thr,
                "xaxis_title": x_label,
            }
            fig = volcano_fig_from_results(
                results,
                title=title,
                neg_log10_padj_threshold=padj_thr,
                fold_change_threshold=fc_thr,
                xaxis_title=x_label,
            )
            fig = set_fig_size(
                fig, fig_w, fig_h, default_width=640, default_height=520
            )
            mark_cols = locus_mark_columns(rt.get("locus_lookup"))
            mark_opts = [{"label": c, "value": c} for c in mark_cols]
            mark_wrap = show if mark_cols else hide
            return (
                fig,
                f"Volcano ({center}): Bin A n={len(ids_a)} "
                f"(clusters {a_lab}) − Bin B n={len(ids_b)} (clusters {b_lab}). "
                "Click a gene for locus-lookup metadata.",
                None,
                mark_wrap,
                mark_opts,
                None,
            )

        @app.callback(
            Output("hc-volcano", "figure", allow_duplicate=True),
            Input("hc-volcano-mark-col", "value"),
            Input("hc-volcano-mark-entry", "value"),
            Input("hc-volcano-fig-w", "value"),
            Input("hc-volcano-fig-h", "value"),
            prevent_initial_call=True,
        )
        def _replot_volcano_marks(mark_col, mark_entry, fig_w, fig_h):
            results = _HC_RUNTIME.get("volcano_results")
            meta = _HC_RUNTIME.get("volcano_plot_meta")
            if results is None or not meta:
                return no_update
            mark_genes = genes_with_meta_entry(
                _HC_RUNTIME.get("locus_lookup"), mark_col, mark_entry
            )
            mark_label = f"{mark_col}={mark_entry}" if mark_col and mark_entry else None
            fig = volcano_fig_from_results(
                results,
                title=meta["title"],
                neg_log10_padj_threshold=meta["padj_thr"],
                fold_change_threshold=meta["fc_thr"],
                xaxis_title=meta["xaxis_title"],
                mark_genes=mark_genes,
                mark_label=mark_label,
            )
            return set_fig_size(
                fig, fig_w, fig_h, default_width=640, default_height=520
            )

        @app.callback(
            Output("hc-volcano-mark-entry", "options"),
            Output("hc-volcano-mark-entry", "value"),
            Output("hc-volcano-mark-entry", "disabled"),
            Output("hc-volcano-mark-entry", "placeholder"),
            Input("hc-volcano-mark-col", "value"),
        )
        def _volcano_mark_entries(col):
            if not col:
                return [], None, True, "Select a column first…"
            opts = entry_dropdown_options(_HC_RUNTIME.get("locus_lookup"), col)
            return opts, None, False, "Entry…"

        @app.callback(
            Output("hc-volcano-last-gene", "data"),
            Input("hc-volcano", "clickData"),
            prevent_initial_call=True,
        )
        def _volcano_gene_click(click):
            if not click or not click.get("points"):
                return no_update
            pt = click["points"][0]
            raw = pt.get("customdata")
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            if raw is None:
                raw = pt.get("text")
            return str(raw) if raw is not None else no_update

        @app.callback(
            Output("hc-volcano-gene-detail", "children"),
            Input("hc-volcano-last-gene", "data"),
            Input("hc-cache", "data"),
            Input("hc-volcano", "figure"),
        )
        def _volcano_gene_detail(last_gene, cache, fig):
            if not last_gene:
                return gene_detail_placeholder()
            return gene_detail_table(str(last_gene), _HC_RUNTIME.get("locus_lookup"))

        @app.callback(
            Output("hc-volcano-celov-out", "value"),
            Output("hc-volcano-celov-status", "children", allow_duplicate=True),
            Input("hc-volcano-celov-browse", "n_clicks"),
            State("hc-volcano-celov-out", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_volcano_celov(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save volcano Celov (use {} for type)",
                defaultextension=".txt",
                initialfile="HC_Volcano_{}.txt",
            )
            if not chosen:
                return no_update, "Celov path browse cancelled."
            p = Path(chosen)
            if "{}" not in p.name:
                chosen = str(p.with_name(f"{p.stem}_{{}}{p.suffix or '.txt'}"))
            return chosen, f"Celov output template: {chosen}"

        @app.callback(
            Output("hc-volcano-celov-status", "children"),
            Input("hc-volcano-celov-save", "n_clicks"),
            State("hc-volcano-celov-out", "value"),
            State("hc-volcano-celov-mode", "value"),
            State("hc-volcano-celov-score", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _save_volcano_celov(
            n_clicks, out_path, mode, score, session_blob, project_blob
        ):
            results = _HC_RUNTIME.get("volcano_results")
            if results is None or not isinstance(results, pd.DataFrame) or results.empty:
                return "Run volcano first."
            if not out_path or not str(out_path).strip():
                return "Choose an output .txt path (Browse)."
            try:
                weighed = _volcano_weighed_for_celov(results, score or "fold_change")
                lookup = None
                locus = _locus_path_from_session(session_blob, project_blob)
                if locus:
                    lookup = load_locus_lookup(locus)
                paths = save_classifier_celov(
                    weighed,
                    str(out_path).strip(),
                    mode=mode or "up_and_down",
                    locus_lookup=lookup,
                    id_column=resolve_celov_id_col(
                        _active_dataset_entry(session_blob, project_blob)
                    ),
                )
                return "Saved Celov: " + ", ".join(str(p) for p in paths)
            except Exception as exc:  # noqa: BLE001
                return f"Celov save error: {exc}"

        @app.callback(
            Output("hc-export-path", "value"),
            Output("hc-status", "children", allow_duplicate=True),
            Input("hc-export-browse", "n_clicks"),
            State("hc-export-path", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_export(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Export HC cluster assignments",
                defaultextension=".csv",
                initialfile="hc_clusters.csv",
            )
            if not chosen:
                return no_update, "Export browse cancelled."
            return chosen, f"Export path: {chosen}"

        @app.callback(
            Output("hc-status", "children", allow_duplicate=True),
            Input("hc-export", "n_clicks"),
            State("hc-export-path", "value"),
            State("hc-maxclust-applied", "data"),
            prevent_initial_call=True,
        )
        def _export(n_clicks, path, t):
            if "Z_samples" not in _HC_RUNTIME:
                return "Run clustering first."
            if not path or not str(path).strip():
                return "Choose an export path (Browse)."
            rt = _HC_RUNTIME
            t = max(2, min(int(t or 2), rt["n_samples"]))
            labels = _cut_clusters(rt["Z_samples"], t)
            out = rt["score_df"].copy()
            cluster_col = f"maxclust :{t}"
            out.insert(0, cluster_col, labels)
            cols = [c for c in ["fileName", cluster_col] if c in out.columns]
            extra = [
                c
                for c in out.columns
                if c not in cols and not (c.startswith("PC") and c[2:].isdigit())
            ]
            out = out[cols + extra]
            p = Path(str(path).strip())
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(p, index=True, index_label="sample_id")
            return f"Exported {len(out)} samples with column {cluster_col!r} → {p}"
