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


def filter_ids_by_search(
    ids,
    search,
    selected,
    *,
    labels: dict[str, str] | None = None,
    limit: int = 80,
) -> list[str]:
    """Keep current selections plus up to ``limit`` search hits (2+ characters)."""
    id_list = [str(x) for x in ids]
    have = set(id_list)
    keep = [str(x) for x in (selected or []) if x and str(x) in have]
    q = (search or "").strip().lower()
    if len(q) < 2:
        return keep
    extra: list[str] = []
    labels = labels or {}
    kept = set(keep)
    for gid in id_list:
        if gid in kept:
            continue
        if q in gid.lower() or q in str(labels.get(gid, "")).lower():
            extra.append(gid)
        if len(extra) >= limit:
            break
    return keep + extra


def locus_mark_columns(lookup: pd.DataFrame | None) -> list[str]:
    if lookup is None or not isinstance(lookup, pd.DataFrame) or lookup.empty:
        return []
    return [str(c) for c in lookup.columns]


def gene_meta_hover_map(
    lookup: pd.DataFrame | None,
    columns: list[str] | None,
) -> dict[str, str]:
    """Map locusTag / geneID → ``<br>col: value`` lines for Plotly hover text."""
    cols = [str(c) for c in (columns or []) if c]
    if (
        lookup is None
        or not isinstance(lookup, pd.DataFrame)
        or lookup.empty
        or not cols
    ):
        return {}
    use = [c for c in cols if c in lookup.columns]
    if not use:
        return {}
    id_cols = [c for c in ("locusTag", "geneID") if c in lookup.columns]
    if not id_cols:
        return {}
    out: dict[str, str] = {}
    for _, row in lookup.iterrows():
        lines: list[str] = []
        for c in use:
            val = row.get(c)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                text = ""
            else:
                text = str(val)
            lines.append(f"{c}: {text}")
        block = "<br>" + "<br>".join(lines)
        for id_col in id_cols:
            gid = row.get(id_col)
            if gid is None or (isinstance(gid, float) and pd.isna(gid)):
                continue
            key = str(gid).strip()
            if key and key not in out:
                out[key] = block
    return out


def gene_hover_text(
    gene_ids: pd.Series,
    hover_map: dict[str, str] | None,
) -> list[str]:
    """Gene ID plus optional locus-metadata lines for each point."""
    hmap = hover_map or {}
    out: list[str] = []
    for gid in gene_ids.astype(str):
        out.append(gid + hmap.get(gid, ""))
    return out


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
    hover_map: dict[str, str] | None = None,
    showlegend: bool = True,
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
            text=gene_hover_text(sub["geneID"], hover_map),
            hovertemplate="%{text} (marked)<extra></extra>",
            showlegend=showlegend,
        )
    )
    return int(mask.sum())


def mark_legend_banner(n_mark: int, mark_label: str | None) -> "html.Div | html.P":
    """HTML stand-in for the Plotly mark legend (keeps figure size stable)."""
    from dash import html

    if not n_mark or not mark_label:
        return html.P("", className="small mb-0", style={"minHeight": "1.25rem"})
    return html.Div(
        [
            html.Span(
                style={
                    "display": "inline-block",
                    "width": "10px",
                    "height": "10px",
                    "borderRadius": "50%",
                    "backgroundColor": _COLOR_MARK,
                    "marginRight": "8px",
                    "verticalAlign": "middle",
                }
            ),
            html.Span(
                f"{mark_label}  ({n_mark} gene{'s' if n_mark != 1 else ''})",
                className="small",
                style={"verticalAlign": "middle"},
            ),
        ],
        className="mb-1",
        style={"minHeight": "1.25rem"},
    )


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
                    html.Label(
                        "Gene column (Name, biological process, …)",
                        className="small mb-0",
                    ),
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
                    html.Label("Entry", className="small mb-0"),
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
