"""In-memory session: load/merge selected datasets into expression + metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .project import DatasetEntry, Project

_GENE_LENGTH_MARKERS = ("geneLength", "gene_length", "GeneLength")


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


_LIVE_SESSION: SessionData | None = None


def active_dataset_name(session_blob=None, active=None) -> str:
    if active:
        return active if isinstance(active, str) else (active[0] if active else "dataset")
    names = (session_blob or {}).get("active_datasets") or []
    return str(names[0]) if names else "dataset"


def session_key(session: SessionData | None) -> str | None:
    if session is None or not session.active_datasets:
        return None
    return "|".join(session.active_datasets)


def set_live_session(session: SessionData) -> None:
    """Keep the count matrix on the server (not in the browser store)."""
    global _LIVE_SESSION
    _LIVE_SESSION = session


def live_session() -> SessionData | None:
    return _LIVE_SESSION


def session_to_store(session: SessionData) -> dict:
    """Slim blob for Dash: columns and key only. Expression stays in ``_LIVE_SESSION``."""
    set_live_session(session)
    if not session.ready:
        return {
            "ready": False,
            "error": session.error,
            "active_datasets": session.active_datasets,
            "meta_columns": [],
            "key": None,
        }
    assert session.expression is not None
    return {
        "ready": True,
        "error": None,
        "active_datasets": session.active_datasets,
        "meta_columns": session.meta_columns(),
        "n_samples": int(session.expression.shape[0]),
        "n_genes": int(session.expression.shape[1]),
        "key": session_key(session),
    }


def session_from_store(blob: dict | None) -> SessionData:
    """Return the server-side session when the store key matches."""
    live = _LIVE_SESSION
    want = (blob or {}).get("key")
    if live is not None and live.ready:
        have = session_key(live)
        if want is None or want == have:
            return live
        return SessionData(
            error="Dataset changed — reload.",
            active_datasets=list((blob or {}).get("active_datasets") or []),
        )
    return SessionData(error=(blob or {}).get("error") or "No session data.")


def meta_columns_from_store(blob: dict | None) -> list[str]:
    return [str(c) for c in (blob or {}).get("meta_columns") or [] if c]


def meta_levels(col: str | None) -> list[str]:
    session = live_session()
    if (
        session is None
        or session.metadata is None
        or not col
        or col not in session.metadata.columns
    ):
        return []
    return sorted(session.metadata[col].dropna().astype(str).unique())


def active_dataset_entry(session_blob, project_blob) -> dict | None:
    active = (session_blob or {}).get("active_datasets") or []
    name = active[0] if active else None
    if not name:
        return None
    return next(
        (d for d in (project_blob or {}).get("datasets", []) if d.get("name") == name),
        None,
    )


def locus_path_from_session(session_blob, project_blob) -> str | None:
    entry = active_dataset_entry(session_blob, project_blob)
    if not entry:
        return None
    locus = entry.get("locus_lookup") or ""
    if not locus:
        return None
    root = (project_blob or {}).get("root")
    if root and not Path(locus).is_absolute():
        return str(Path(root) / locus)
    return str(locus)


def clear_analysis_runtimes() -> None:
    """Drop PCA / HC / UMAP / gradient / volcano results from a previous dataset."""
    from src.GUI.modules.clustering import _HC_RUNTIME
    from src.GUI.modules.gene_gradients import _GRAD_RUNTIME
    from src.GUI.modules.parallel_conditions import _PAR_RUNTIME
    from src.GUI.modules.pca import _PCA_RUNTIME
    from src.GUI.modules.umap_mod import _UMAP_RUNTIME
    from src.GUI.modules.volcano_condition import _VC_RUNTIME

    _PCA_RUNTIME.clear()
    _HC_RUNTIME.clear()
    _UMAP_RUNTIME.clear()
    _GRAD_RUNTIME.clear()
    _PAR_RUNTIME.clear()
    _VC_RUNTIME.clear()


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
