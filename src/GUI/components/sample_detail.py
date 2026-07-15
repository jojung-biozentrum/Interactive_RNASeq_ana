"""Hover (fileName only) and click → metadata table beside scatter plots."""

from __future__ import annotations

from dash import Dash, Input, Output, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

_COORD_PREFIXES = ("PC", "UMAP")


def _is_coord_col(name: str) -> bool:
    return any(name.startswith(p) and name[len(p) :].isdigit() for p in _COORD_PREFIXES)


def _internal_cols() -> set[str]:
    return {"_click_id", "_row_id", "_sample_id", "index", "level_0"}


def sample_detail_placeholder() -> html.P:
    return html.P("Click a point to see sample metadata.", className="text-muted small mb-0")


def gene_detail_placeholder() -> html.P:
    return html.P("Click a gene to see locus-lookup metadata.", className="text-muted small mb-0")


def gene_detail_table(gene_id: str, locus_lookup: pd.DataFrame | None) -> html.Div:
    """Locus-lookup metadata for a gene (same table pattern as sample metadata)."""
    gid = str(gene_id)
    if locus_lookup is None or not isinstance(locus_lookup, pd.DataFrame) or locus_lookup.empty:
        return html.Div(
            [
                html.H6(gid, className="mb-2"),
                html.P(
                    "No locus lookup loaded for this dataset.",
                    className="text-muted small mb-0",
                ),
            ]
        )
    lookup = locus_lookup.copy()
    hit = pd.DataFrame()
    if "locusTag" in lookup.columns:
        hit = lookup.loc[lookup["locusTag"].astype(str) == gid]
    if hit.empty and "geneID" in lookup.columns:
        hit = lookup.loc[lookup["geneID"].astype(str) == gid]
    if hit.empty:
        return html.Div(
            [
                html.H6(gid, className="mb-2"),
                html.P(
                    "No locus-lookup row for this gene.",
                    className="text-muted small mb-0",
                ),
            ]
        )
    row = hit.iloc[0]
    title = gid
    for col in ("geneName", "gene_name", "old locusTag", "biocyc_id"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            title = f"{row[col]} ({gid})"
            break
    return sample_detail_table(row, scroll=False, title=title, extra={"geneID": gid})


def sample_detail_table(
    row: pd.Series | None,
    *,
    scroll: bool = True,
    extra: dict | None = None,
    title: str | None = None,
) -> html.Div:
    if row is None:
        return sample_detail_placeholder()
    rows = []
    if extra:
        for col, val in extra.items():
            text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
            rows.append(
                html.Tr(
                    [
                        html.Th(str(col), className="text-nowrap fw-bold"),
                        html.Td(text),
                    ]
                )
            )
    for col, val in row.items():
        if col in _internal_cols() or _is_coord_col(str(col)):
            continue
        if extra and col in extra:
            continue
        if pd.isna(val):
            text = ""
        else:
            text = str(val)
        rows.append(html.Tr([html.Th(str(col), className="text-nowrap"), html.Td(text)]))
    if not rows:
        return html.P("No metadata columns for this sample.", className="text-muted small mb-0")
    if title is None:
        title = row.get("fileName")
        if title is None or (isinstance(title, float) and pd.isna(title)):
            title = row.name
    style = {"overflowX": "auto"}
    if scroll:
        style.update({"maxHeight": "560px", "overflowY": "auto"})
    return html.Div(
        [
            html.H6(str(title), className="mb-2"),
            dbc.Table(
                [html.Tbody(rows)],
                bordered=True,
                striped=True,
                size="sm",
                className="mb-0",
                style={"whiteSpace": "nowrap"},
            ),
        ],
        className="overflow-auto" if scroll else None,
        style=style,
    )


def plot_with_sample_detail(
    graph_id: str,
    detail_id: str,
    *,
    graph_config: dict | None = None,
    graph_md: int = 8,
    detail_md: int = 4,
) -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(
                dcc.Graph(id=graph_id, figure={}, config=graph_config or {}),
                md=graph_md,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H6("Sample metadata", className="mb-2"),
                        html.Div(id=detail_id, children=sample_detail_placeholder()),
                    ],
                    className="border rounded p-2 bg-light",
                ),
                md=detail_md,
            ),
        ],
        className="g-2 mb-2 align-items-start",
    )


def lookup_sample(score_df: pd.DataFrame | None, click_id: str | None) -> pd.Series | None:
    if score_df is None or not click_id:
        return None
    key = str(click_id)
    if key in score_df.index:
        return score_df.loc[key]
    if "_sample_id" in score_df.columns:
        hit = score_df.loc[score_df["_sample_id"].astype(str) == key]
        if len(hit) == 1:
            return hit.iloc[0]
    return None


def register_sample_detail_callback(
    app: Dash,
    *,
    graph_id: str,
    detail_id: str,
    cache_id: str,
    get_score_df,
) -> None:
    """Register click → metadata table; clear when analysis cache is refreshed."""

    @app.callback(
        Output(detail_id, "children"),
        Input(graph_id, "clickData"),
        Input(cache_id, "data"),
    )
    def _show_sample_detail(click_data, cache):
        from dash import callback_context

        if not cache:
            return sample_detail_placeholder()
        triggered = callback_context.triggered_id
        if triggered == cache_id or not click_data:
            return sample_detail_placeholder()
        point = click_data["points"][0]
        click_id = point.get("customdata")
        if isinstance(click_id, (list, tuple)):
            click_id = click_id[0] if click_id else None
        elif click_id is not None:
            click_id = str(click_id)
        row = lookup_sample(get_score_df(cache), click_id)
        return sample_detail_table(row)
