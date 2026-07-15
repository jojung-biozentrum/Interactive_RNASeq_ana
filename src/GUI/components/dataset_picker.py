"""Dataset registration and single-dataset picker."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc


def dataset_picker_layout() -> html.Div:
    return html.Div(
        [
            html.H5("Datasets"),
            html.P(
                "Browse for expression, sample metadata, and locus lookup "
                "(genes × samples matrices). Paths are stored relative to the "
                "working folder when possible. Only one dataset is active at a time.",
                className="text-muted small",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Name"),
                            dbc.Input(id="ds-name", type="text"),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Expression path"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="ds-expr", type="text"),
                                    dbc.Button(
                                        "Browse…",
                                        id="ds-expr-browse",
                                        color="info",
                                        outline=True,
                                    ),
                                ]
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.Label("Metadata path"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="ds-meta", type="text"),
                                    dbc.Button(
                                        "Browse…",
                                        id="ds-meta-browse",
                                        color="info",
                                        outline=True,
                                    ),
                                ]
                            ),
                        ],
                        md=5,
                    ),
                ],
                className="g-2 mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Locus lookup path"),
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="ds-locus", type="text"),
                                    dbc.Button(
                                        "Browse…",
                                        id="ds-locus-browse",
                                        color="info",
                                        outline=True,
                                    ),
                                ]
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(
                        [
                            html.Label("Sample ID column"),
                            dbc.Input(
                                id="ds-sample-col",
                                type="text",
                                value="fileName",
                            ),
                        ],
                        md=2,
                    ),
                    dbc.Col(
                        [
                            html.Label("Celov ID column"),
                            dbc.Input(
                                id="ds-celov-id-col",
                                type="text",
                                value="biocyc_id",
                                placeholder="biocyc_id",
                            ),
                            html.P(
                                "Locus-lookup column used as gene IDs in Celov exports "
                                "(e.g. Biocyc IDs, old locus tags, RefSeq IDs, gene names).",
                                className="text-muted small mb-0",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Br(),
                            dbc.Button(
                                "Register dataset",
                                id="ds-register-btn",
                                color="secondary",
                                size="sm",
                                className="me-2",
                            ),
                        ],
                        md=2,
                    ),
                ],
                className="g-2 mb-2",
            ),
            html.Hr(),
            html.Label("Active dataset"),
            dcc.Dropdown(
                id="ds-active",
                multi=False,
                placeholder="Select a dataset to load",
                clearable=True,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Unregister datasets", className="mt-2"),
                            dcc.Dropdown(
                                id="ds-unregister",
                                multi=True,
                                placeholder="Select registered datasets to remove",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            html.Br(),
                            dbc.Button(
                                "Unregister",
                                id="ds-unregister-btn",
                                color="warning",
                                outline=True,
                                size="sm",
                                className="mt-2",
                            ),
                        ],
                        md=3,
                    ),
                ],
                className="g-2 mb-2",
            ),
            html.Div(id="ds-status", className="mt-2 text-muted small"),
        ],
        className="mb-3",
    )
