"""Dataset picker (read-only: datasets come from the data folder's project.yaml)."""

from __future__ import annotations

from dash import dcc, html


def dataset_picker_layout(
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
