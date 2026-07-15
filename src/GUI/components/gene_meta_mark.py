"""Mark genes on scatters by locus-lookup column entry (';'-'separated tokens)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_COLOR_MARK = "#e41a1c"


def split_meta_tokens(value) -> list[str]:
    """Split a cell on ``;`` and strip empties."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [t.strip() for t in str(value).split(";") if t.strip()]


def _natural_key(value) -> list:
    """Sort key that stays comparable when some tokens start with digits.

    Pathways include IDs like ``1CMET2-PWY|...``; a bare ``int`` then ``str``
    pair raises TypeError under ``sorted`` in Python 3.
    """
    parts = re.split(r"(\d+)", str(value))
    key: list = []
    for p in parts:
        if not p:
            continue
        if p.isdigit():
            key.append((0, int(p)))
        else:
            key.append((1, p.lower()))
    return key


def locus_mark_columns(lookup: pd.DataFrame | None) -> list[str]:
    if lookup is None or not isinstance(lookup, pd.DataFrame) or lookup.empty:
        return []
    return [str(c) for c in lookup.columns]


def locus_entry_values(lookup: pd.DataFrame | None, column: str | None) -> list[str]:
    if (
        lookup is None
        or not column
        or column not in lookup.columns
    ):
        return []
    entries: set[str] = set()
    for v in lookup[column]:
        entries.update(split_meta_tokens(v))
    return sorted(entries, key=_natural_key)


def entry_dropdown_options(lookup: pd.DataFrame | None, column: str | None) -> list[dict]:
    """Dash dropdown options for ``;``-tokens in ``column`` (empty until a column is set)."""
    if not column:
        return []
    return [{"label": v, "value": v} for v in locus_entry_values(lookup, column)]


def genes_with_meta_entry(
    lookup: pd.DataFrame | None,
    column: str | None,
    entry: str | None,
    *,
    id_column: str = "locusTag",
) -> set[str]:
    """Gene IDs (locusTag / geneID) whose ``column`` cell contains ``entry`` as a ``;`` token."""
    if (
        lookup is None
        or not column
        or entry is None
        or str(entry).strip() == ""
        or column not in lookup.columns
    ):
        return set()
    want = str(entry).strip()
    id_col = id_column if id_column in lookup.columns else (
        "geneID" if "geneID" in lookup.columns else None
    )
    if id_col is None and "locusTag" in lookup.columns:
        id_col = "locusTag"
    if id_col is None:
        return set()
    out: set[str] = set()
    for _, row in lookup.iterrows():
        if want in split_meta_tokens(row.get(column)):
            gid = row.get(id_col)
            if gid is not None and not (isinstance(gid, float) and pd.isna(gid)):
                out.add(str(gid).strip())
    return out


def gene_mark_mask(results: pd.DataFrame, gene_ids: set[str] | None) -> pd.Series:
    """Boolean Series aligned to ``results`` for genes in ``gene_ids``."""
    if not gene_ids or "geneID" not in results.columns:
        return pd.Series(False, index=results.index)
    return results["geneID"].astype(str).isin(gene_ids)


def add_marked_gene_trace(
    fig: go.Figure,
    results: pd.DataFrame,
    x_col: str,
    y_col: str,
    gene_ids: set[str] | None,
    *,
    legend_name: str = "marked",
    size: int = 8,
    opacity: float = 1.0,
) -> int:
    """Draw matching genes as solid red points (same style as threshold-pass black)."""
    if not gene_ids or x_col not in results.columns or y_col not in results.columns:
        return 0
    mask = gene_mark_mask(results, gene_ids)
    if not mask.any():
        return 0
    sub = results.loc[mask]
    fig.add_trace(
        go.Scatter(
            x=sub[x_col],
            y=sub[y_col],
            mode="markers",
            marker=dict(size=size, color=_COLOR_MARK, opacity=opacity),
            name=legend_name,
            customdata=sub["geneID"].astype(str),
            text=sub["geneID"].astype(str),
            hovertemplate="%{text} (marked)<extra></extra>",
            showlegend=True,
        )
    )
    return int(mask.sum())


def mark_controls(
    *,
    col_id: str,
    entry_id: str,
    wrap_id: str | None = None,
) -> "html.Div":
    """Dropdowns for locus column + entry (visible after analysis run)."""
    from dash import dcc, html
    import dash_bootstrap_components as dbc

    body = dbc.Row(
        [
            dbc.Col(
                [
                    html.Label("Mark genes by locus column", className="small mb-0"),
                    dcc.Dropdown(
                        id=col_id,
                        clearable=True,
                        placeholder="Column…",
                        searchable=True,
                    ),
                ],
                md=4,
            ),
            dbc.Col(
                [
                    html.Label("Entry (; -separated tokens)", className="small mb-0"),
                    dcc.Dropdown(
                        id=entry_id,
                        clearable=True,
                        placeholder="Select a column first…",
                        searchable=True,
                        disabled=True,
                        options=[],
                    ),
                ],
                md=4,
            ),
            dbc.Col(
                html.P(
                    "Highlights all genes that list this entry in the chosen column.",
                    className="text-muted small mb-0 mt-4",
                ),
                md=4,
            ),
        ],
        className="g-2 mb-2",
    )
    if wrap_id:
        return html.Div(body, id=wrap_id, style={"display": "none"})
    return html.Div(body)
