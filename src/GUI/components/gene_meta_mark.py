"""Mark genes on scatters by locus-lookup column entry (';'-'separated tokens)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_COLOR_MARK = "#e41a1c"

# Locus-lookup columns used for gene hover (and Gene expression profiles search).
# From locus_lookup_biocyc_ids_handcurated.csv.
GENE_HOVER_COLUMNS = (
    "geneName",
    "Gene name A1552",
    "Gene name C6706",
)


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


def gene_search_options(
    gene_ids,
    search,
    selected,
    *,
    lookup: pd.DataFrame | None = None,
    limit: int = 80,
) -> list[dict]:
    """Dropdown options: keep selection, add hits on geneID / all locus columns."""
    id_list = [str(x) for x in gene_ids]
    have = set(id_list)
    chosen = [str(x) for x in (selected or []) if x and str(x) in have]
    opts = [{"label": g, "value": g} for g in chosen]
    q = (search or "").strip().lower()
    if len(q) < 2:
        return opts
    seen = set(chosen)
    id_col = None
    if lookup is not None and not lookup.empty:
        if "geneID" in lookup.columns:
            id_col = "geneID"
        elif "locusTag" in lookup.columns:
            id_col = "locusTag"
    if id_col is not None and lookup is not None:
        for _, row in lookup.iterrows():
            gid = str(row.get(id_col, "") or "")
            if not gid or gid not in have or gid in seen:
                continue
            pieces = [gid]
            for c in lookup.columns:
                val = row.get(c)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                text = str(val).strip()
                if text:
                    pieces.append(text)
            if q not in " ".join(pieces).lower():
                continue
            # Prefer a short gene name in the label when available.
            display = None
            for c in GENE_HOVER_COLUMNS:
                if c in lookup.columns:
                    raw = row.get(c)
                    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                        continue
                    tok = split_meta_tokens(raw)
                    if tok:
                        display = tok[0]
                        break
            label = f"{display} ({gid})" if display and display != gid else gid
            opts.append({"label": label, "value": gid})
            seen.add(gid)
            if len(opts) >= limit:
                return opts
    for gid in id_list:
        if gid in seen:
            continue
        if q in gid.lower():
            opts.append({"label": gid, "value": gid})
            seen.add(gid)
        if len(opts) >= limit:
            break
    return opts


def gene_meta_hover_map(
    lookup: pd.DataFrame | None,
    columns: list[str] | None = None,
) -> dict[str, str]:
    """Map locusTag / geneID → ``<br>col: value`` lines for Plotly hover text.

    ``columns`` defaults to ``GENE_HOVER_COLUMNS`` present in the lookup.
    """
    if columns is None:
        have = set(map(str, lookup.columns)) if lookup is not None and not lookup.empty else set()
        cols = [c for c in GENE_HOVER_COLUMNS if c in have]
    else:
        cols = [str(c) for c in columns if c]
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


def parse_gene_ids(text: str | None) -> list[str]:
    """Split pasted geneIDs (space / comma / semicolon / newline — PCA Top genes form)."""
    if not text or not str(text).strip():
        return []
    parts = re.split(r"[\s,;]+", str(text).strip())
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        gid = str(p).strip()
        if not gid or gid in seen:
            continue
        seen.add(gid)
        out.append(gid)
    return out


def merge_gene_selection(
    current,
    *,
    add: list[str] | None = None,
    toggle: str | None = None,
    clear: bool = False,
    max_n: int | None = None,
) -> list[str]:
    """Update a multi-select gene list (add / toggle / clear), optionally capped."""
    if clear:
        return []
    selected = [str(g) for g in (current or []) if g]
    if toggle:
        gene = str(toggle)
        if gene in selected:
            selected = [g for g in selected if g != gene]
        else:
            selected = selected + [gene]
    if add:
        have = set(selected)
        for gid in add:
            g = str(gid)
            if not g or g in have:
                continue
            selected.append(g)
            have.add(g)
    if max_n is not None:
        selected = selected[: int(max_n)]
    return selected


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


def gene_pick_controls(
    *,
    search_id: str,
    clear_id: str,
    paste_id: str,
    paste_btn_id: str,
    label: str = "Search / pick genes",
    help_text: str | None = None,
) -> "html.Div":
    """Compact search + paste + clear for marking / profiling specific genes."""
    from dash import dcc, html
    import dash_bootstrap_components as dbc

    help_line = help_text or (
        "Type 2+ characters (geneID / locus name), paste IDs from PCA Top genes, "
        "or click points on the plot."
    )
    return html.Div(
        [
            html.P(help_line, className="text-muted small mb-1"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label(label, className="small mb-0"),
                            dcc.Dropdown(
                                id=search_id,
                                multi=True,
                                searchable=True,
                                placeholder="Type 2+ characters…",
                            ),
                        ],
                        md=7,
                    ),
                    dbc.Col(
                        [
                            html.Label("\u00a0", className="small mb-0"),
                            dbc.Button(
                                "Clear",
                                id=clear_id,
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="w-100",
                            ),
                        ],
                        md=2,
                        className="d-flex flex-column justify-content-end",
                    ),
                ],
                className="g-2 mb-1",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Textarea(
                            id=paste_id,
                            placeholder="Paste geneIDs (space / comma / newline)…",
                            style={
                                "width": "100%",
                                "height": "52px",
                                "fontFamily": "monospace",
                                "fontSize": "12px",
                            },
                        ),
                        md=7,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Add pasted IDs",
                            id=paste_btn_id,
                            color="secondary",
                            outline=True,
                            size="sm",
                            className="w-100",
                        ),
                        md=2,
                        className="d-flex align-items-center",
                    ),
                ],
                className="g-2 mb-2",
            ),
        ]
    )
