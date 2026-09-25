"""Gene score tables for analysis export (PCA loadings, classifier, contrasts)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def weighed_genes_from_pc(loadings: pd.DataFrame, pc: str, flip_sign: bool = False) -> pd.DataFrame:
    """Gene weights = loadings on one PC (notebook ``weighed_genes``)."""
    if pc not in loadings.columns:
        raise KeyError(f"PC {pc!r} not in loadings columns {list(loadings.columns)}")
    weights = loadings[pc].astype(float).to_numpy()
    if flip_sign:
        weights = -weights
    out = (
        pd.DataFrame({"gene_weight": weights}, index=loadings.index.astype(str))
        .assign(abs_gene_weight=lambda d: d["gene_weight"].abs())
        .sort_values("abs_gene_weight", ascending=False)
        .reset_index(names="geneID")
    )
    return out


def weighed_genes_from_classifier(
    loadings: pd.DataFrame,
    explained_variance: np.ndarray | list[float],
    coef: np.ndarray | list[float],
    n_pcs: int,
) -> pd.DataFrame:
    """Combine PC loadings with linear classifier coefficients (notebook formula).

    Expects loadings as ``components.T * sqrt(explained_variance)`` (correlation loadings).
    """
    pc_cols = [f"PC{i + 1}" for i in range(n_pcs)]
    missing = [c for c in pc_cols if c not in loadings.columns]
    if missing:
        raise KeyError(f"Missing PC columns in loadings: {missing}")
    var = np.asarray(explained_variance, dtype=float).ravel()
    w = np.asarray(coef, dtype=float).ravel()[:n_pcs]
    if len(var) < n_pcs:
        raise ValueError(f"explained_variance length {len(var)} < n_pcs={n_pcs}")
    scale = np.sqrt(np.maximum(var[:n_pcs], 1e-12))
    weights = loadings[pc_cols].to_numpy(dtype=float) @ (w / scale)
    out = (
        pd.DataFrame({"gene_weight": weights}, index=loadings.index.astype(str))
        .assign(abs_gene_weight=lambda d: d["gene_weight"].abs())
        .sort_values("abs_gene_weight", ascending=False)
        .reset_index(names="geneID")
    )
    return out
