"""In-memory session: load/merge selected datasets into expression + metadata.

Expression matrices stay on the server (process cache + reload-from-disk).
``dcc.Store`` only carries metadata / gene IDs so large datasets (e.g.
NEB_KazukiRef ~115 MB as JSON) are not pushed through the browser/nginx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .project import DatasetEntry, Project

_GENE_LENGTH_MARKERS = ("geneLength", "gene_length", "GeneLength")

# Process-local cache: expression is too large for dcc.Store round-trips.
# Key = (project_root, active dataset names). Gunicorn multi-worker: each
# worker may miss and reload from disk via project_root in the store blob.
_SESSION_CACHE: dict[tuple[str, tuple[str, ...]], "SessionData"] = {}


def _csv_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _read_table(path: Path, index_col=0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=_csv_sep(path), index_col=index_col)


def _load_normalized_counts(path: Path) -> pd.DataFrame:
    """Prepare expression exactly like the notebook PCA block.

    ::

        df = pd.read_csv(...)
        df = df.set_index("geneID")
        sampleCol = df.columns.get_loc("geneLength") + 1
        df = df.iloc[:, sampleCol:]
        df = df.dropna()
        df = df.T   # samples × genes
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep=_csv_sep(path))
    if "geneID" in df.columns:
        df = df.set_index("geneID")
    else:
        df = df.set_index(df.columns[0])

    length_col = next((c for c in _GENE_LENGTH_MARKERS if c in df.columns), None)
    if length_col is None:
        raise ValueError(
            f"{path.name}: expected a geneLength column before sample columns "
            f"(got columns e.g. {list(df.columns[:8])})."
        )
    sample_col = df.columns.get_loc(length_col) + 1
    counts = df.iloc[:, sample_col:]
    counts = counts.dropna()
    return counts.T


def _ensure_sample_id_column(meta: pd.DataFrame, sample_id_col: str) -> pd.DataFrame:
    """Ensure metadata has the join key; build ``fileName`` from kdlibNr + barcode if needed."""
    meta = meta.copy()
    if sample_id_col in meta.columns:
        return meta

    # Common biofilm-microenvironments convention
    if sample_id_col == "fileName" or "fileName" not in meta.columns:
        barcode_cols = [c for c in meta.columns if "barcode" in str(c).lower()]
        if "kdlibNr" in meta.columns and barcode_cols:
            meta["fileName"] = (
                meta["kdlibNr"].astype(str).str.strip()
                + "_"
                + meta[barcode_cols[0]].astype(str).str.strip()
            )
            if sample_id_col != "fileName" and sample_id_col not in meta.columns:
                # requested column missing but fileName was built — use fileName
                return meta
            return meta

    if sample_id_col not in meta.columns and "fileName" in meta.columns:
        return meta

    raise ValueError(
        f"Metadata missing sample ID column {sample_id_col!r}. "
        "For biofilm-microenvironments use sample_id_col=fileName "
        "(built from kdlibNr + barcode when absent)."
    )


def _resolve_join_col(meta: pd.DataFrame, sample_id_col: str) -> str:
    if sample_id_col in meta.columns:
        return sample_id_col
    if "fileName" in meta.columns:
        return "fileName"
    raise ValueError(f"Cannot resolve sample ID column {sample_id_col!r}")


def _load_one(project: Project, entry: DatasetEntry) -> tuple[pd.DataFrame, pd.DataFrame]:
    expr_path = project.resolve(entry.expression)
    meta_path = project.resolve(entry.metadata)
    # Notebook prep: geneID index → after geneLength → dropna → T
    X = _load_normalized_counts(expr_path)
    X.index = X.index.astype(str)

    meta = _read_table(meta_path, index_col=None)
    meta = _ensure_sample_id_column(meta, entry.sample_id_col)
    join_col = _resolve_join_col(meta, entry.sample_id_col)

    meta = meta.set_index(join_col)
    meta.index = meta.index.astype(str)
    if "fileName" not in meta.columns:
        meta = meta.copy()
        meta.insert(0, "fileName", meta.index.astype(str))

    n_matched = int(X.index.isin(meta.index).sum())
    if n_matched == 0:
        raise ValueError(
            f"Dataset '{entry.name}': no overlapping sample IDs between expression and metadata "
            f"(join column={join_col!r}). "
            f"Expression has e.g. {list(X.index[:3])}; metadata has e.g. {list(meta.index[:3])}."
        )
    # Keep all count-matrix samples for PCA (notebook); attach metadata where present
    meta = meta.reindex(X.index).copy()
    meta["dataset"] = entry.name
    new_index = [f"{entry.name}::{sid}" for sid in X.index]
    X = X.copy()
    X.index = new_index
    meta.index = new_index
    return X, meta


@dataclass
class SessionData:
    expression: pd.DataFrame | None = None  # samples x genes
    metadata: pd.DataFrame | None = None
    active_datasets: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.expression is not None
            and self.metadata is not None
            and len(self.expression) > 0
            and self.error is None
        )

    def numeric_matrix(self) -> np.ndarray:
        """Samples × genes as float — no scaling, no log1p (notebook PCA input)."""
        if self.expression is None:
            raise RuntimeError("No expression loaded")
        return np.asarray(self.expression, dtype=float)

    def meta_columns(self) -> list[str]:
        if self.metadata is None:
            return []
        return list(self.metadata.columns)


def _cache_key(project_root: str, names: list[str]) -> tuple[str, tuple[str, ...]]:
    return (str(project_root), tuple(names))


def cache_session(project_root: str, session: SessionData) -> None:
    if project_root and session.ready:
        _SESSION_CACHE[_cache_key(project_root, session.active_datasets)] = session


def _meta_for_store(meta: pd.DataFrame) -> dict:
    """Serialize metadata with NaN/Inf → None (JSON-safe)."""
    out = meta.copy()
    out.insert(0, "_sample_id", out.index.astype(str))
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict(orient="list")


def session_to_store(session: SessionData, *, project_root: str | None = None) -> dict:
    """Serialize a light session for Dash ``dcc.Store`` (no expression matrix)."""
    if not session.ready:
        return {
            "ready": False,
            "error": session.error,
            "active_datasets": session.active_datasets,
            "meta_columns": [],
            "project_root": project_root,
        }
    assert session.expression is not None and session.metadata is not None
    if project_root:
        cache_session(project_root, session)
    return {
        "ready": True,
        "error": None,
        "project_root": project_root,
        "active_datasets": session.active_datasets,
        "meta_columns": session.meta_columns(),
        "gene_ids": list(session.expression.columns.astype(str)),
        "n_samples": int(session.expression.shape[0]),
        "metadata": _meta_for_store(session.metadata),
    }


def _expression_from_legacy_blob(blob: dict) -> pd.DataFrame | None:
    expr_blob = blob.get("expression")
    if not expr_blob or "data" not in expr_blob:
        return None
    return pd.DataFrame(
        expr_blob["data"],
        index=expr_blob["index"],
        columns=expr_blob["columns"],
    )


def _hydrate_expression(blob: dict, names: list[str]) -> pd.DataFrame | None:
    """Prefer process cache; else reload from disk; else legacy embedded matrix."""
    root = blob.get("project_root")
    if root and names:
        cached = _SESSION_CACHE.get(_cache_key(str(root), names))
        if cached is not None and cached.expression is not None:
            return cached.expression
        try:
            project = Project.load(str(root))
            sess = load_selected(project, names)
        except Exception:  # noqa: BLE001
            sess = None
        if sess is not None and sess.ready and sess.expression is not None:
            cache_session(str(root), sess)
            return sess.expression
    return _expression_from_legacy_blob(blob)


def session_from_store(blob: dict | None) -> SessionData:
    if not blob or not blob.get("ready"):
        return SessionData(error=(blob or {}).get("error") or "No session data.")

    names = list(blob.get("active_datasets", []))
    meta = pd.DataFrame(blob["metadata"])
    if "_sample_id" in meta.columns:
        meta = meta.set_index("_sample_id")
    meta.index = meta.index.astype(str)

    expression = _hydrate_expression(blob, names)
    if expression is None:
        return SessionData(
            error="Expression matrix unavailable (failed to reload from data folder).",
            active_datasets=names,
        )
    return SessionData(
        expression=expression,
        metadata=meta,
        active_datasets=names,
    )


def load_selected(project: Project, names: list[str]) -> SessionData:
    if not names:
        return SessionData(error="Select at least one dataset.")
    by_name = {d.name: d for d in project.datasets}
    missing = [n for n in names if n not in by_name]
    if missing:
        return SessionData(error=f"Unknown datasets: {', '.join(missing)}")

    frames_x: list[pd.DataFrame] = []
    frames_m: list[pd.DataFrame] = []
    try:
        for name in names:
            X, meta = _load_one(project, by_name[name])
            frames_x.append(X)
            frames_m.append(meta)
        all_genes = sorted(set().union(*(set(f.columns) for f in frames_x)))
        aligned = [f.reindex(columns=all_genes, fill_value=0) for f in frames_x]
        expression = pd.concat(aligned, axis=0)
        metadata = pd.concat(frames_m, axis=0).reindex(expression.index)
        return SessionData(
            expression=expression,
            metadata=metadata,
            active_datasets=list(names),
        )
    except Exception as exc:  # noqa: BLE001 — surface to UI
        return SessionData(error=str(exc), active_datasets=list(names))
