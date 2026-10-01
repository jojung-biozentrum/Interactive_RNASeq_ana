"""Active-dataset picker (read-only viewer)."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc


def dataset_picker_layout(
    *,
    options: list[dict] | None = None,
    value: str | None = None,
) -> html.Div:
    return html.Div(
        [
            html.H5("Dataset"),
            html.P(
                "Datasets are the ones registered in the data folder's project.yaml.",
                className="text-muted small",
            ),
            html.Label("Active dataset"),
            dcc.Dropdown(
                id="ds-active",
                multi=False,
                options=options or [],
                value=value,
                placeholder="Select a dataset to load",
                clearable=True,
            ),
            html.Div(id="ds-status", className="mt-2 text-muted small"),
        ],
        className="mb-3",
    )
