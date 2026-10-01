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

from ..components.controls import (
    graph_export_config,
)
from ..components.gene_meta_mark import (
    entry_dropdown_options,
    gene_meta_hover_map,
    genes_with_meta_entry,
    locus_mark_columns,
    mark_controls,
    mark_legend_banner,
)
from ..components.sample_detail import (
    gene_detail_placeholder,
    gene_detail_table,
    sample_detail_placeholder,
    sample_detail_table,
    samples_detail_table,
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
from src.biocyc.celov_multiomics_post import load_locus_lookup


_GRAPH_CONFIG = graph_export_config("hc_plot")

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


def _heatmap_title(dataset: str, t: int) -> dict:
    return _plotly_title(
        "Pairwise expression differences within samples",
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

    fig, _n_mark = volcano_fig_from_results(
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


_HC_CACHE_N_PCS = 20


def _run_hc(session: SessionData, *, method: str = "ward") -> dict:
    """Cluster samples on the unscaled count matrix (notebook samples × samples)."""
    if session.expression is None or session.metadata is None:
        raise RuntimeError("No session data")
    expr = session.expression
    X = session.numeric_matrix()
    sample_ids = list(expr.index.astype(str))
    gene_ids = list(expr.columns.astype(str))
    meta = session.metadata

    Z_samples = _linkage(X, method=method)
    dist_samples = squareform(pdist(X, metric="euclidean"))

    pca = PCA()
    pca.fit(X)
    scores = pca.transform(X)
    n_pcs = min(scores.shape[1], _HC_CACHE_N_PCS)
    pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]
    score_df = pd.DataFrame(scores[:, :n_pcs], index=sample_ids, columns=pc_cols)
    score_df = score_df.join(meta)

    return {
        "Z_samples": Z_samples,
        "X": X,
        "dist_samples": dist_samples,
        "sample_ids": sample_ids,
        "gene_ids": gene_ids,
        "score_df": score_df,
        "method": method,
        "n_samples": len(sample_ids),
        "n_genes": len(gene_ids),
    }


def _pack_hc_cache(rt: dict) -> dict:
    """Persist linkage + PCA scores only (recompute distances from session on hydrate)."""
    score_df = rt["score_df"]
    n_pcs = sum(
        1 for c in score_df.columns if str(c).startswith("PC") and str(c)[2:].isdigit()
    )
    return {
        "ready": True,
        "n_samples": rt["n_samples"],
        "n_genes": rt["n_genes"],
        "n_pcs": n_pcs,
        "method": rt["method"],
        "sample_ids": list(rt["sample_ids"]),
        "Z_samples": np.asarray(rt["Z_samples"], dtype=float).tolist(),
        "scores": score_df.reset_index(names="_sample_id").to_dict(orient="list"),
    }


def _ensure_hc_dist(rt: dict, session_blob=None) -> bool:
    """Ensure ``dist_samples`` exists (from runtime or session expression)."""
    dist = rt.get("dist_samples")
    if dist is not None:
        arr = np.asarray(dist)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] == int(rt.get("n_samples") or 0):
            return True
    X = rt.get("X")
    if X is None and session_blob:
        session = session_from_store(session_blob)
        if session.ready and session.expression is not None:
            X = session.numeric_matrix()
            rt["X"] = X
            rt["gene_ids"] = list(session.expression.columns.astype(str))
            rt["n_genes"] = len(rt["gene_ids"])
    if X is None:
        return False
    rt["dist_samples"] = squareform(pdist(X, metric="euclidean"))
    return True


def _hc_cache_sig(cache: dict | None) -> tuple | None:
    if not cache or not cache.get("ready") or "Z_samples" not in cache:
        return None
    ids = cache.get("sample_ids") or []
    return (
        cache.get("method"),
        cache.get("n_samples"),
        cache.get("n_genes"),
        cache.get("n_pcs"),
        ids[0] if ids else None,
        ids[-1] if ids else None,
        len(ids),
    )


def _hydrate_hc(cache: dict | None, session_blob=None, project_blob=None) -> bool:
    """Fill ``_HC_RUNTIME`` from ``hc-cache`` (+ session) for multi-worker use."""
    sig = _hc_cache_sig(cache)
    if (
        sig is not None
        and _HC_RUNTIME.get("_cache_sig") == sig
        and isinstance(_HC_RUNTIME.get("score_df"), pd.DataFrame)
        and "Z_samples" in _HC_RUNTIME
    ):
        if _HC_RUNTIME.get("locus_lookup") is None and session_blob is not None:
            path = _locus_path_from_session(session_blob, project_blob)
            if path:
                try:
                    _HC_RUNTIME["locus_lookup"] = load_locus_lookup(path)
                except Exception:  # noqa: BLE001
                    pass
        _ensure_hc_dist(_HC_RUNTIME, session_blob)
        return True
    if sig is None:
        # Same-worker fallback when Store did not round-trip the cache payload.
        if (
            isinstance(_HC_RUNTIME.get("score_df"), pd.DataFrame)
            and "Z_samples" in _HC_RUNTIME
        ):
            _ensure_hc_dist(_HC_RUNTIME, session_blob)
            return True
        return False
    score_df = pd.DataFrame(cache["scores"])
    if "_sample_id" in score_df.columns:
        score_df = score_df.set_index("_sample_id")
    score_df.index = score_df.index.astype(str)
    sample_ids = list(cache.get("sample_ids") or score_df.index.astype(str))
    X = None
    gene_ids: list[str] = []
    if session_blob:
        session = session_from_store(session_blob)
        if session.ready and session.expression is not None:
            X = session.numeric_matrix()
            gene_ids = list(session.expression.columns.astype(str))
    lookup = None
    if session_blob is not None:
        path = _locus_path_from_session(session_blob, project_blob)
        if path:
            try:
                lookup = load_locus_lookup(path)
            except Exception:  # noqa: BLE001
                lookup = None
    keep_volcano = {
        k: _HC_RUNTIME[k]
        for k in ("volcano_results", "volcano_plot_meta")
        if k in _HC_RUNTIME and _HC_RUNTIME.get("_cache_sig") == sig
    }
    dist = None
    if cache.get("dist_samples") is not None:
        dist = np.asarray(cache["dist_samples"], dtype=float)
    _HC_RUNTIME.clear()
    _HC_RUNTIME.update(
        {
            "_cache_sig": sig,
            "Z_samples": np.asarray(cache["Z_samples"], dtype=float),
            "dist_samples": dist,
            "sample_ids": sample_ids,
            "gene_ids": gene_ids,
            "score_df": score_df,
            "X": X,
            "method": cache.get("method") or "ward",
            "n_samples": int(cache.get("n_samples") or len(sample_ids)),
            "n_genes": int(cache.get("n_genes") or len(gene_ids)),
            "locus_lookup": lookup,
            **keep_volcano,
        }
    )
    _ensure_hc_dist(_HC_RUNTIME, session_blob)
    return True


def _sample_distance_fig(rt: dict, sample_labels: np.ndarray, *, dataset: str, t: int) -> go.Figure:
    ids = rt["sample_ids"]
    Z = rt["Z_samples"]
    dist = rt["dist_samples"]
    leaves, leaf_c = _leaf_clusters_in_dendro_order(Z, sample_labels)
    ordered = [ids[i] for i in leaves]
    # Click detail resolves cell (x,y) via these leaf-ordered ids (no n×n customdata).
    rt["heat_row_ids"] = ordered
    rt["heat_col_ids"] = ordered
    return heatmap_with_dendro(
        dist,
        ids,
        ids,
        Z,
        Z,
        title=_heatmap_title(dataset, t),
        row_leaf_clusters=leaf_c,
        col_leaf_clusters=leaf_c,
        xaxis_title="Samples",
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
                        html.H6("Cluster cut"),
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
                html.H6("Heatmap (samples × samples)"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                dcc.Graph(
                                    id="hc-heatmap",
                                    figure={},
                                    config=graph_export_config(
                                        "hc_heatmap", width=760, height=640
                                    ),
                                ),
                                type="default",
                            ),
                            md=8,
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H6("Click detail", className="mb-2"),
                                    html.P(
                                        "Click a cell for the two samples’ metadata.",
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
                html.Div(id="hc-status", className="text-muted small mb-2"),
                html.Div(
                    [
                        html.Hr(),
                        html.H6("PCA + dendrogram"),
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
                                dbc.Col(
                                    [
                                        html.Label("Bin A alpha"),
                                        dcc.Slider(
                                            id="hc-pca-alpha-a",
                                            min=0.05,
                                            max=1.0,
                                            step=0.05,
                                            value=1.0,
                                            marks={0.05: "0.05", 0.5: "0.5", 1.0: "1"},
                                        ),
                                    ],
                                    md=3,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Bin B alpha"),
                                        dcc.Slider(
                                            id="hc-pca-alpha-b",
                                            min=0.05,
                                            max=1.0,
                                            step=0.05,
                                            value=1.0,
                                            marks={0.05: "0.05", 0.5: "0.5", 1.0: "1"},
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
                                    dcc.Graph(
                                        id="hc-pca",
                                        figure={},
                                        config=_GRAPH_CONFIG,
                                    ),
                                    md=5,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="hc-dendro",
                                            figure={},
                                            config=graph_export_config(
                                                "hc_dendrogram", height=360
                                            ),
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
                                            html.Label(
                                                "Show metadata of",
                                                className="small mb-0",
                                            ),
                                            dcc.Dropdown(
                                                id="hc-pca-meta-src",
                                                options=[
                                                    {
                                                        "label": "Off (click a point)",
                                                        "value": "off",
                                                    },
                                                    {"label": "Bin A", "value": "a"},
                                                    {"label": "Bin B", "value": "b"},
                                                ],
                                                value="off",
                                                clearable=False,
                                                className="mb-2",
                                            ),
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
                        html.Div(
                            id="hc-volcano-mark-legend",
                            children=mark_legend_banner(0, None),
                        ),
                        dcc.Loading(
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Graph(
                                            id="hc-volcano",
                                            figure={},
                                            config=graph_export_config(
                                                "hc_volcano", width=640, height=520
                                            ),
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
            if _hydrate_hc(cache, session_blob) and isinstance(
                _HC_RUNTIME.get("score_df"), pd.DataFrame
            ):
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
                cache = _pack_hc_cache(rt)
                _HC_RUNTIME.clear()
                _HC_RUNTIME.update(rt)
                _HC_RUNTIME["_cache_sig"] = _hc_cache_sig(cache)
                pcs = [c for c in rt["score_df"].columns if c.startswith("PC") and c[2:].isdigit()]
                opts = [{"label": c, "value": c} for c in pcs]
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
            State("hc-cache", "data"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _apply_maxclust(n_clicks, t, cache, session_blob, project_blob):
            if not _hydrate_hc(cache, session_blob, project_blob):
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
            State("hc-cache", "data"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _find_homo(n_clicks, col, val, cache, session_blob, project_blob):
            if not _hydrate_hc(cache, session_blob, project_blob):
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
            Output("hc-heatmap", "figure"),
            Input("hc-cache", "data"),
            Input("hc-maxclust-applied", "data"),
            Input("session-store", "data"),
            State("project-store", "data"),
        )
        def _plot_heatmap(cache, t, session_blob, project_blob):
            empty = go.Figure()
            if not _hydrate_hc(cache, session_blob, project_blob):
                empty.add_annotation(
                    text="Run clustering to show the heatmap.",
                    showarrow=False,
                )
                return empty
            rt = _HC_RUNTIME
            if not _ensure_hc_dist(rt, session_blob):
                empty.add_annotation(
                    text="Heatmap needs session expression to rebuild distances.",
                    showarrow=False,
                )
                return empty
            dataset = _active_dataset_name(session_blob)
            try:
                n = rt["n_samples"]
                t = max(2, min(int(t or 2), n))
                labels = _cut_clusters(rt["Z_samples"], t)
                return _sample_distance_fig(rt, labels, dataset=dataset, t=t)
            except Exception as exc:  # noqa: BLE001
                err = go.Figure()
                err.add_annotation(text=f"Heatmap error: {exc}", showarrow=False)
                return err

        @app.callback(
            Output("hc-heat-detail", "children"),
            Input("hc-heatmap", "clickData"),
            Input("hc-cache", "data"),
            Input("hc-maxclust-applied", "data"),
            State("session-store", "data"),
            State("project-store", "data"),
        )
        def _heat_detail(click, cache, t, session_blob, project_blob):
            triggered = callback_context.triggered_id
            if triggered == "hc-cache" or not _hydrate_hc(cache, session_blob, project_blob):
                return sample_detail_placeholder()
            n = _HC_RUNTIME["n_samples"]
            t_use = max(2, min(int(t or 2), n))
            labels = _cut_clusters(_HC_RUNTIME["Z_samples"], t_use)
            return detail_from_heatmap_click(
                click, _HC_RUNTIME, "ss", labels=labels, t=t_use
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
            Input("hc-pca-alpha-a", "value"),
            Input("hc-pca-alpha-b", "value"),
            Input("hc-cluster-sel", "data"),
            Input("session-store", "data"),
            State("project-store", "data"),
        )
        def _plot_pca(
            cache,
            t,
            x_col,
            y_col,
            z_col,
            alpha_a,
            alpha_b,
            selected,
            session_blob,
            project_blob,
        ):
            empty = go.Figure()
            if not _hydrate_hc(cache, session_blob, project_blob):
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
                    alpha_a=float(alpha_a if alpha_a is not None else 1.0),
                    alpha_b=float(alpha_b if alpha_b is not None else 1.0),
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
                return pca_fig, dendro_fig
            except Exception as exc:  # noqa: BLE001
                err = go.Figure()
                err.add_annotation(text=f"PCA error: {exc}", showarrow=False)
                return err, err

        @app.callback(
            Output("hc-pca-detail", "children"),
            Input("hc-pca", "clickData"),
            Input("hc-pca-meta-src", "value"),
            Input("hc-cluster-sel", "data"),
            Input("hc-cache", "data"),
            Input("hc-maxclust-applied", "data"),
            State("session-store", "data"),
            State("project-store", "data"),
        )
        def _pca_detail(click, meta_src, selected, cache, t, session_blob, project_blob):
            triggered = callback_context.triggered_id
            if triggered == "hc-cache" or not _hydrate_hc(
                cache, session_blob, project_blob
            ):
                return sample_detail_placeholder()
            df = _HC_RUNTIME["score_df"]
            n = _HC_RUNTIME["n_samples"]
            t_use = max(2, min(int(t or 2), n))
            labels = _cut_clusters(_HC_RUNTIME["Z_samples"], t_use)
            sample_ids = _HC_RUNTIME["sample_ids"]
            cluster_col = f"maxclust :{t_use}"

            src = meta_src if meta_src in ("a", "b") else "off"
            if src in ("a", "b"):
                bins = _normalize_bin_sel(selected)
                cids = set(bins[src])
                if not cids:
                    return html.P(
                        f"Bin {src.upper()} is empty — assign clusters on the dendrogram.",
                        className="text-muted small mb-0",
                    )
                keep = [
                    sid
                    for sid, lab in zip(sample_ids, labels)
                    if int(lab) in cids
                ]
                if not keep:
                    return html.P(
                        f"No samples in Bin {src.upper()}.",
                        className="text-muted small mb-0",
                    )
                sub = df.loc[[s for s in keep if s in df.index]]
                extras = [
                    int(labels[sample_ids.index(sid)]) if sid in sample_ids else None
                    for sid in sub.index.astype(str)
                ]
                return samples_detail_table(
                    sub,
                    title=f"Bin {src.upper()}",
                    subtitle=f"Length: {len(sub)} samples",
                    extra_col=cluster_col,
                    extra_values=extras,
                )

            if not click:
                return sample_detail_placeholder()
            point = click["points"][0]
            cid = point.get("customdata")
            if isinstance(cid, (list, tuple)):
                cid = cid[0] if cid else None
            if cid is None or str(cid) not in df.index:
                return sample_detail_placeholder()
            sid = str(cid)
            extra = None
            if sid in sample_ids:
                extra = {cluster_col: int(labels[sample_ids.index(sid)])}
            return sample_detail_table(df.loc[sid], extra=extra)

        @app.callback(
            Output("hc-volcano", "figure"),
            Output("hc-volcano-status", "children"),
            Output("hc-volcano-last-gene", "data", allow_duplicate=True),
            Output("hc-volcano-mark-wrap", "style"),
            Output("hc-volcano-mark-col", "options"),
            Output("hc-volcano-mark-col", "value"),
            Output("hc-volcano-mark-legend", "children"),
            Input("hc-volcano-run", "n_clicks"),
            State("hc-cluster-sel", "data"),
            State("hc-maxclust-applied", "data"),
            State("hc-volcano-padj", "value"),
            State("hc-volcano-fc", "value"),
            State("hc-volcano-center", "value"),
            State("hc-cache", "data"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _run_volcano(
            n_clicks,
            selected,
            t,
            padj_thr,
            fc_thr,
            center,
            cache,
            session_blob,
            project_blob,
        ):
            empty = go.Figure()
            hide = {"display": "none"}
            show = {"display": "block"}
            clear_legend = mark_legend_banner(0, None)
            no_mark = (hide, [], None, clear_legend)
            if not _hydrate_hc(cache, session_blob, project_blob):
                return empty, "Run clustering first.", None, *no_mark
            if _HC_RUNTIME.get("X") is None:
                return empty, "Session expression missing for volcano.", None, *no_mark
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
            fig, _n_mark = volcano_fig_from_results(
                results,
                title=title,
                neg_log10_padj_threshold=padj_thr,
                fold_change_threshold=fc_thr,
                xaxis_title=x_label,
                hover_map=gene_meta_hover_map(rt.get("locus_lookup")),
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
                clear_legend,
            )

        @app.callback(
            Output("hc-volcano", "figure", allow_duplicate=True),
            Output("hc-volcano-mark-legend", "children", allow_duplicate=True),
            Input("hc-volcano-mark-col", "value"),
            Input("hc-volcano-mark-entry", "value"),
            prevent_initial_call=True,
        )
        def _replot_volcano_marks(mark_col, mark_entry):
            results = _HC_RUNTIME.get("volcano_results")
            meta = _HC_RUNTIME.get("volcano_plot_meta")
            if results is None or not meta:
                return no_update, no_update
            mark_genes = genes_with_meta_entry(
                _HC_RUNTIME.get("locus_lookup"), mark_col, mark_entry
            )
            mark_label = f"{mark_col}={mark_entry}" if mark_col and mark_entry else None
            fig, n_mark = volcano_fig_from_results(
                results,
                title=meta["title"],
                neg_log10_padj_threshold=meta["padj_thr"],
                fold_change_threshold=meta["fc_thr"],
                xaxis_title=meta["xaxis_title"],
                mark_genes=mark_genes,
                mark_label=mark_label,
                hover_map=gene_meta_hover_map(_HC_RUNTIME.get("locus_lookup")),
            )
            return fig, mark_legend_banner(n_mark, mark_label)

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

