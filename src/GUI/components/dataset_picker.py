"""Dataset picker (read-only: datasets come from project.yaml)."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc


def dataset_picker_layout(
    options: list[dict] | None = None,
    value: str | None = None,
) -> html.Div:
    return html.Div(
        [
            html.H5("Datasets"),
            html.P(
                "Datasets are listed from project.yaml. This server build cannot "
                "register, unregister, or write filtered / Celov files.",
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
