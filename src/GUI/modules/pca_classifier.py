"""Linear classifier on PCs — same logic as PCA.ipynb ``fit_pc_classifier`` / weighed genes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression # expectation: roughly linearly seperable and no extreme outliers
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.biocyc.celov_multiomics_post import (
    annotate_gene_table,
    celov_multiomics_file_generation,
)
from src.GUI.components.controls import apply_export_layout
from src.GUI.components.gene_scores import weighed_genes_from_classifier


def fit_pc_classifier(df: pd.DataFrame, y: np.ndarray, n_pcs: int) -> dict | None:
    """Logistic regression on first n_pcs. Returns None if too few samples."""
    if len(df) <= n_pcs or len(np.unique(y)) < 2:
        return None
    pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]
    missing = [c for c in pc_cols if c not in df.columns]
    if missing:
        return None
    X = df[pc_cols].to_numpy(dtype=float)
    clf = LogisticRegression(max_iter=10000, random_state=0).fit(X, y)
    n_splits = min(5, int(np.bincount(y.astype(int)).min())) # number of splits for cross-validation, minimum of 5 or the minimum number of samples in either class
    if n_splits < 2:
        cv_acc = float("nan")
    else:
        cv_acc = float(
            cross_val_score(
                clf,
                X,
                y,
                cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0),
                scoring="balanced_accuracy",
            ).mean()
        )
    return {
        "clf": clf,
        "w": clf.coef_[0],
        "b": float(clf.intercept_[0]),
        "cv_acc": cv_acc,
        "train_acc": float(clf.score(X, y)),
    }


def encode_binary_labels(series: pd.Series, positive: str) -> np.ndarray:
    """Map label column to 0/1 with ``positive`` as class 1 (notebook Biofilm.astype(int))."""
    s = series.astype(str)
    pos = str(positive)
    if not (s == pos).any():
        raise ValueError(f"Positive class {pos!r} not found in labels.")
    if s.nunique(dropna=True) < 2:
        raise ValueError("Need at least two classes for the linear classifier.")
    return (s == pos).astype(int).to_numpy()


def classifier_performance(
    score_df: pd.DataFrame,
    y: np.ndarray,
    n_pc_min: int,
    n_pc_max: int,
) -> pd.DataFrame:
    """Train/CV accuracy vs number of PCs (notebook performance loop)."""
    max_available = sum(1 for c in score_df.columns if c.startswith("PC") and c[2:].isdigit())
    hi = min(int(n_pc_max), max_available)
    lo = max(2, int(n_pc_min))
    rows = []
    for n in range(lo, hi + 1):
        res = fit_pc_classifier(score_df, y, n)
        if res:
            rows.append({"n_pcs": n, "cv_acc": res["cv_acc"], "train_acc": res["train_acc"]})
    return pd.DataFrame(rows)


def performance_figure(perf_df: pd.DataFrame, title: str | dict) -> go.Figure:
    fig = go.Figure()
    if perf_df is None or perf_df.empty:
        fig.add_annotation(text="No classifier results", showarrow=False)
        return fig
    fig.add_trace(
        go.Scatter(
            x=perf_df["n_pcs"],
            y=perf_df["cv_acc"],
            mode="lines+markers",
            name="test (CV balanced accuracy)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=perf_df["n_pcs"],
            y=perf_df["train_acc"],
            mode="lines+markers",
            name="train accuracy",
            line=dict(dash="dash"),
            opacity=0.7,
        )
    )
    title_dict = (
        title
        if isinstance(title, dict)
        else {"text": title, "x": 0.5, "xanchor": "center"}
    )
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    fig.update_layout(
        title=title_dict,
        xaxis_title="Number of PCs",
        yaxis_title="Accuracy",
        xaxis=dict(tickmode="linear", dtick=1),
    )
    # Same outer size as PCA (not full-page wide); legend outside to the right
    apply_export_layout(
        fig,
        title_lines=title_lines,
        legend=True,
        uirevision="pca-clf-perf",
    )
    return fig


def decision_boundary_points(df: pd.DataFrame, w, b) -> tuple[list[float], list[float]] | None:
    """PC1/PC2 separation line clipped to data range (notebook ``plot_pc_separation``)."""
    w0, w1, b = float(w[0]), float(w[1]), float(b)
    x_min, x_max = float(df["PC1"].min()), float(df["PC1"].max())
    y_min, y_max = float(df["PC2"].min()), float(df["PC2"].max())
    pts: list[tuple[float, float]] = []
    if abs(w1) > 1e-12:
        for x in (x_min, x_max):
            y = -(w0 * x + b) / w1
            if y_min <= y <= y_max:
                pts.append((x, y))
    if abs(w0) > 1e-12:
        for y in (y_min, y_max):
            x = -(w1 * y + b) / w0
            if x_min <= x <= x_max:
                pts.append((x, y))
    # unique, keep endpoints for a line
    uniq = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-9 and abs(p[1] - q[1]) < 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    uniq = sorted(uniq)
    return [p[0] for p in uniq[:2]], [p[1] for p in uniq[:2]]


def add_decision_boundary(fig: go.Figure, df: pd.DataFrame, w, b, name: str = "2-PC classifier") -> go.Figure:
    pts = decision_boundary_points(df, w, b)
    if pts is None:
        return fig
    xs, ys = pts
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line=dict(color="crimson", width=2, dash="dash"),
            hoverinfo="name",
        )
    )
    return fig


def save_classifier_celov(
    weighed: pd.DataFrame,
    out_path: str | Path,
    mode: str,
    locus_lookup: pd.DataFrame | None = None,
    id_column: str = "biocyc_id",
) -> list[Path]:
    """Write Celov ``.txt`` only (no CSV).

    ``out_path`` should contain ``{}`` for the type token, e.g.
    ``PCA_LinearClass_{}.txt`` → ``…_up.txt``, ``…_down.txt``, ``…_up_and_down.txt``.

    ``id_column`` is the locus-lookup column used as Celov gene IDs (dataset
    ``celov_id_col``; e.g. biocyc_id, old locus tags, RefSeq IDs, gene names).

    ``mode``: ``up_and_down`` (both signs in one file), ``up``, ``down``, or ``all``
    (write the three type files).
    """
    df = weighed.copy()
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
        if "geneID" not in df.columns:
            raise ValueError("weighed genes table missing geneID")

    subsets = {
        "up_and_down": df,
        "up": df[df["gene_weight"] > 0],
        "down": df[df["gene_weight"] < 0],
    }
    mode = (mode or "up_and_down").lower().replace(" ", "_")
    if mode in ("combined", "both"):
        mode = "up_and_down"
    if mode == "all":
        types = ["up_and_down", "up", "down"]
    elif mode in subsets:
        types = [mode]
    else:
        raise ValueError(f"Unknown Celov mode {mode!r}; use up_and_down, up, down, or all.")

    template = str(out_path)
    if "{}" not in template:
        # insert before extension
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
            "gene_weight",
            id_column=id_column,
            dataset_label=f"linear_classifier_{kind}",
        )
        written.append(path)
    return written


def build_weighed_genes(
    loadings: pd.DataFrame,
    explained_variance: np.ndarray,
    coef: np.ndarray,
    n_pcs: int,
) -> pd.DataFrame:
    return weighed_genes_from_classifier(loadings, explained_variance, coef, n_pcs)
