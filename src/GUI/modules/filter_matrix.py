"""Filter count-matrix samples (metadata) or genes (locus lookup)."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd

from src.biocyc.celov_multiomics_post import load_locus_lookup

from ..components.folder_browser import pick_save_file_dialog
from ..data_store import (
    _ensure_sample_id_column,
    _read_table,
    _resolve_join_col,
)
from ..project import DatasetEntry, Project

_GENE_LENGTH_MARKERS = ("geneLength", "gene_length", "GeneLength")


def _csv_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _load_count_matrix(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=_csv_sep(path))


def _sample_columns(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns.astype(str))
    length_col = next((c for c in _GENE_LENGTH_MARKERS if c in cols), None)
    if length_col is None:
        raise ValueError(
            f"Expected geneLength column before sample columns (got e.g. {cols[:8]})."
        )
    i = cols.index(length_col) + 1
    return cols[i:]


def _annotation_columns(df: pd.DataFrame) -> list[str]:
    sample_cols = set(_sample_columns(df))
    return [c for c in df.columns.astype(str) if c not in sample_cols]


def _gene_id_column(matrix: pd.DataFrame) -> str:
    if "geneID" in matrix.columns:
        return "geneID"
    return str(matrix.columns[0])


def _resolve_active_dataset(blob: dict | None, active) -> tuple[DatasetEntry | None, str | None]:
    if not blob or not blob.get("root"):
        return None, "Open a working folder first."
    name = active if isinstance(active, str) else (active[0] if active else None)
    if not name:
        return None, "Select an active dataset first."
    project = Project.load(blob["root"])
    project.datasets = [DatasetEntry.from_dict(d) for d in blob.get("datasets", [])]
    entry = next((d for d in project.datasets if d.name == name), None)
    if entry is None:
        return None, f"Dataset {name!r} not found."
    return entry, None


def _load_metadata(entry: DatasetEntry, project: Project) -> tuple[pd.DataFrame, str]:
    meta = _read_table(project.resolve(entry.metadata), index_col=None)
    meta = _ensure_sample_id_column(meta, entry.sample_id_col)
    join_col = _resolve_join_col(meta, entry.sample_id_col)
    return meta, join_col


def _resolve_locus_path(project: Project, locus_path: str | None) -> Path:
    raw = (locus_path or "").strip()
    if not raw:
        raise ValueError("Locus lookup path is required for gene-axis filtering.")
    p = Path(raw)
    if not p.is_absolute():
        p = project.resolve(raw)
    if not p.exists():
        raise FileNotFoundError(f"Locus lookup not found: {p}")
    return p


def _metadata_match_mask(meta: pd.DataFrame, column: str, mode: str, value: str) -> pd.Series:
    if column not in meta.columns:
        raise KeyError(f"Metadata missing column {column!r}")
    series = meta[column].astype(str)
    needle = "" if value is None else str(value)
    mode = (mode or "equals").lower()
    if mode == "contains":
        return series.str.contains(needle, case=False, na=False, regex=False)
    return series == needle


def filter_matrix_by_sample_metadata(
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    join_col: str,
    column: str,
    mode: str,
    value: str,
    action: str,
) -> tuple[pd.DataFrame, int, int]:
    """Keep or drop sample columns based on sample metadata."""
    sample_cols = _sample_columns(matrix)
    meta = meta.copy()
    meta[join_col] = meta[join_col].astype(str)
    mask = _metadata_match_mask(meta, column, mode, value)
    matched_ids = set(meta.loc[mask, join_col].astype(str))

    action = (action or "exclude").lower()
    if action == "keep":
        keep = [c for c in sample_cols if str(c) in matched_ids]
    else:
        keep = [c for c in sample_cols if str(c) not in matched_ids]

    if not keep:
        raise ValueError("Filter would remove all sample columns.")

    annot = _annotation_columns(matrix)
    out = matrix.loc[:, annot + keep].copy()
    return out, len(sample_cols), len(keep)


def filter_matrix_by_locus_lookup(
    matrix: pd.DataFrame,
    lookup: pd.DataFrame,
    column: str,
    mode: str,
    value: str,
    action: str,
) -> tuple[pd.DataFrame, int, int]:
    """Keep or drop gene rows using locus-lookup metadata (join geneID ↔ locusTag)."""
    if "locusTag" not in lookup.columns:
        raise ValueError("Locus lookup must contain a 'locusTag' column")
    gene_col = _gene_id_column(matrix)
    lookup = lookup.copy()
    lookup["locusTag"] = lookup["locusTag"].astype(str)
    mask = _metadata_match_mask(lookup, column, mode, value)
    matched_ids = set(lookup.loc[mask, "locusTag"].astype(str))

    gene_ids = matrix[gene_col].astype(str)
    action = (action or "exclude").lower()
    if action == "keep":
        row_mask = gene_ids.isin(matched_ids)
    else:
        row_mask = ~gene_ids.isin(matched_ids)

    n0 = int(len(matrix))
    out = matrix.loc[row_mask].copy()
    n1 = int(len(out))
    if n1 == 0:
        raise ValueError("Filter would remove all gene rows.")
    return out, n0, n1


class FilterModule:
    id = "filter"
    label = "Filter count matrix"

    def layout(self):
        return html.Div(
            [
                html.H5("Filter count matrix"),
                dbc.Alert(
                    "CLR trafo is normalized on the geometric mean of all conditions and all genes. "
                    "The filtered data is still normalized using the whole dataset as the reference.",
                    color="warning",
                    className="py-2 small",
                ),
                html.P(
                    "Filter sample columns via sample metadata, or gene rows via locus-lookup "
                    "metadata (geneID ↔ locusTag). Locus lookup is required for gene-axis filters.",
                    className="text-muted small",
                ),
                html.Div(id="flt-active-label", className="mb-2 fw-semibold"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Axis"),
                                dcc.RadioItems(
                                    id="flt-axis",
                                    options=[
                                        {"label": "samples", "value": "samples"},
                                        {"label": "genes", "value": "genes"},
                                    ],
                                    value="samples",
                                    inline=True,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            html.P(
                                "Gene-axis filters use the active dataset’s locus lookup path.",
                                className="text-muted small mt-4 mb-0",
                            ),
                            md=9,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Filter column"),
                                dcc.Dropdown(id="flt-column"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Match"),
                                dcc.RadioItems(
                                    id="flt-mode",
                                    options=[
                                        {"label": "equals", "value": "equals"},
                                        {"label": "contains", "value": "contains"},
                                    ],
                                    value="equals",
                                    inline=True,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Value"),
                                dbc.Input(id="flt-value", type="text"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Action"),
                                dcc.RadioItems(
                                    id="flt-action",
                                    options=[
                                        {"label": "exclude matches", "value": "exclude"},
                                        {"label": "keep matches only", "value": "keep"},
                                    ],
                                    value="exclude",
                                    inline=True,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Output CSV"),
                                dbc.InputGroup(
                                    [
                                        dbc.Input(id="flt-output", type="text"),
                                        dbc.Button(
                                            "Browse…",
                                            id="flt-output-browse",
                                            color="info",
                                            outline=True,
                                        ),
                                    ]
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run filter", id="flt-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="flt-status", className="text-muted small"),
                dcc.Store(id="flt-dataset-key"),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        @app.callback(
            Output("flt-dataset-key", "data"),
            Output("flt-active-label", "children"),
            Input("ds-active", "value"),
            Input("project-store", "data"),
        )
        def _sync_active(active, blob):
            entry, err = _resolve_active_dataset(blob, active)
            if err or entry is None:
                return None, err or "No active dataset."
            name = active if isinstance(active, str) else (active[0] if active else "")
            project = Project.load(blob["root"])
            project.datasets = [DatasetEntry.from_dict(d) for d in blob.get("datasets", [])]
            expr_path = str(project.resolve(entry.expression))
            key = {
                "dataset": name,
                "expression": entry.expression,
                "metadata": entry.metadata,
                "locus_lookup": entry.locus_lookup,
                "sample_id_col": entry.sample_id_col,
                "celov_id_col": entry.celov_id_col,
            }
            return key, f"Active dataset: {name} — {expr_path}"

        @app.callback(
            Output("flt-column", "options"),
            Output("flt-column", "value"),
            Input("flt-axis", "value"),
            Input("flt-dataset-key", "data"),
            State("project-store", "data"),
            State("flt-column", "value"),
        )
        def _fill_columns(axis, key, blob, current_col):
            if not key or not blob or not blob.get("root"):
                return [], None
            try:
                project = Project.load(blob["root"])
                project.datasets = [DatasetEntry.from_dict(d) for d in blob.get("datasets", [])]
                entry = DatasetEntry(
                    name=key["dataset"],
                    expression=key["expression"],
                    metadata=key["metadata"],
                    locus_lookup=key.get("locus_lookup") or "",
                    sample_id_col=key.get("sample_id_col", "fileName"),
                    celov_id_col=key.get("celov_id_col", "biocyc_id"),
                )
                if (axis or "samples") == "genes":
                    if not entry.locus_lookup:
                        return [], None
                    lookup = load_locus_lookup(_resolve_locus_path(project, entry.locus_lookup))
                    cols = [c for c in lookup.columns.astype(str)]
                else:
                    meta, _ = _load_metadata(entry, project)
                    cols = [c for c in meta.columns.astype(str) if c != "dataset"]
            except Exception:  # noqa: BLE001
                return [], None
            opts = [{"label": c, "value": c} for c in cols]
            value = current_col if current_col in cols else (opts[0]["value"] if opts else None)
            return opts, value

        @app.callback(
            Output("flt-output", "value"),
            Output("flt-status", "children", allow_duplicate=True),
            Input("flt-output-browse", "n_clicks"),
            State("flt-output", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse_out(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save filtered matrix CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
                initialfile="filtered_matrix.csv",
            )
            if not chosen:
                return no_update, "Output browse cancelled."
            return chosen, f"Output: {chosen}"

        @app.callback(
            Output("flt-status", "children"),
            Input("flt-run", "n_clicks"),
            State("flt-dataset-key", "data"),
            State("project-store", "data"),
            State("flt-axis", "value"),
            State("flt-column", "value"),
            State("flt-mode", "value"),
            State("flt-value", "value"),
            State("flt-action", "value"),
            State("flt-output", "value"),
            prevent_initial_call=True,
        )
        def _run(n_clicks, key, blob, axis, column, mode, value, action, out_path):
            if not key or not blob or not blob.get("root"):
                return "Select an active dataset first."
            if not column:
                return "Choose a filter column."
            if not out_path or not str(out_path).strip():
                return "Choose an output CSV path (Browse)."
            axis = axis or "samples"
            try:
                project = Project.load(blob["root"])
                entry = DatasetEntry(
                    name=key["dataset"],
                    expression=key["expression"],
                    metadata=key["metadata"],
                    locus_lookup=key.get("locus_lookup") or "",
                    sample_id_col=key.get("sample_id_col", "fileName"),
                    celov_id_col=key.get("celov_id_col", "biocyc_id"),
                )
                matrix = _load_count_matrix(project.resolve(entry.expression))
                if axis == "genes":
                    if not entry.locus_lookup:
                        return "Active dataset has no locus lookup path — re-register with one."
                    lookup = load_locus_lookup(_resolve_locus_path(project, entry.locus_lookup))
                    filtered, n0, n1 = filter_matrix_by_locus_lookup(
                        matrix, lookup, column, mode, value, action
                    )
                    unit = "gene rows"
                else:
                    meta, join_col = _load_metadata(entry, project)
                    filtered, n0, n1 = filter_matrix_by_sample_metadata(
                        matrix, meta, join_col, column, mode, value, action
                    )
                    unit = "sample columns"
                out = Path(str(out_path).strip())
                out.parent.mkdir(parents=True, exist_ok=True)
                filtered.to_csv(out, index=False)
                verb = "kept matches" if (action or "exclude") == "keep" else "excluded matches"
                return (
                    f"Wrote {out} — {unit} {n0} → {n1} "
                    f"({verb}; {axis} {column} {mode} {value!r})."
                )
            except Exception as exc:  # noqa: BLE001
                return f"Filter error: {exc}"
