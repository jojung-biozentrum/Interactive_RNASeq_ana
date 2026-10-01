"""Per-PC one-sided end-split ARI (notebook / main condition_enrichment)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def binary_mask(series: pd.Series, positive) -> np.ndarray:
    return (series.astype(str) == str(positive)).to_numpy()


def end_cluster_labels(x: np.ndarray, is_pos: np.ndarray) -> np.ndarray:
    """One-sided threshold covering all positives."""
    x = np.asarray(x, dtype=float)
    is_pos = np.asarray(is_pos, dtype=bool)
    if np.median(x[is_pos]) >= np.median(x[~is_pos]):
        return np.where(x >= x[is_pos].min(), "pos-end", "other")
    return np.where(x <= x[is_pos].max(), "pos-end", "other")


def pc_separation_ari(
    score_df: pd.DataFrame,
    label_col: str,
    positive,
    n_pcs: int,
) -> pd.DataFrame:
    """Per-PC one-sided threshold ARI."""
    is_pos = binary_mask(score_df[label_col], positive)
    if is_pos.sum() == 0 or (~is_pos).sum() == 0:
        raise ValueError(f"Need both {positive!r} and other samples in {label_col}.")
    rows = []
    for i in range(1, int(n_pcs) + 1):
        col = f"PC{i}"
        if col not in score_df.columns:
            break
        pred = end_cluster_labels(score_df[col].to_numpy(dtype=float), is_pos)
        true = np.where(is_pos, "pos", "other")
        rows.append({"PC": i, "ARI": float(adjusted_rand_score(true, pred))})
    return pd.DataFrame(rows)
