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


def group_contrast_scores(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    log1p: bool = True,
) -> pd.DataFrame:
    """Per-gene mean difference / fold-change style scores between two metadata groups.

    ``fold_change`` is mean(A) - mean(B) on log1p (or raw) expression.
    ``neg_log10_padj`` is a simple Welch t-test derived −log10(p) (BH-adjusted when possible).
    """
    from scipy import stats

    if group_col not in metadata.columns:
        raise KeyError(f"Metadata missing column {group_col!r}")
    meta = metadata.copy()
    meta[group_col] = meta[group_col].astype(str)
    idx_a = meta.index[meta[group_col] == str(group_a)]
    idx_b = meta.index[meta[group_col] == str(group_b)]
    if len(idx_a) == 0 or len(idx_b) == 0:
        raise ValueError(
            f"Empty group(s): A={group_a!r} (n={len(idx_a)}), B={group_b!r} (n={len(idx_b)})"
        )

    X = expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if log1p:
        X = np.log1p(X)
    xa = X.loc[idx_a]
    xb = X.loc[idx_b]
    mean_a = xa.mean(axis=0)
    mean_b = xb.mean(axis=0)
    fold = mean_a - mean_b

    pvals = []
    for gene in X.columns:
        a = xa[gene].to_numpy(dtype=float)
        b = xb[gene].to_numpy(dtype=float)
        if len(a) < 2 or len(b) < 2 or (np.nanstd(a) == 0 and np.nanstd(b) == 0):
            pvals.append(1.0)
            continue
        try:
            _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            pvals.append(float(p) if np.isfinite(p) else 1.0)
        except Exception:  # noqa: BLE001
            pvals.append(1.0)

    pvals_arr = np.asarray(pvals, dtype=float)
    try:
        padj = stats.false_discovery_control(pvals_arr, method="bh")
    except Exception:  # noqa: BLE001
        padj = pvals_arr

    neg_log = -np.log10(np.clip(padj, 1e-300, 1.0))
    out = pd.DataFrame(
        {
            "geneID": X.columns.astype(str),
            "fold_change": fold.to_numpy(dtype=float),
            "abs_fold_change": np.abs(fold.to_numpy(dtype=float)),
            "neg_log10_padj": neg_log,
            "pi_value": fold.to_numpy(dtype=float) * neg_log,
            "group_a": group_a,
            "group_b": group_b,
        }
    )
    return out.sort_values("abs_fold_change", ascending=False).reset_index(drop=True)
