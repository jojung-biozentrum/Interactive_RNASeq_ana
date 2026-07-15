"""Parallel conditions — placeholder (HC along a gradient; related conditions)."""

from __future__ import annotations

from dash import Dash, html
import dash_bootstrap_components as dbc


class ParallelConditionsModule:
    id = "parallel-conditions"
    label = "Parallel conditions"

    def layout(self):
        return html.Div(
            [
                dbc.Alert(
                    [
                        html.Strong("Placeholder — not implemented yet."),
                        html.Br(),
                        "This tab is reserved for upcoming analysis; controls and figures "
                        "will appear here when the method is ported.",
                    ],
                    color="warning",
                    className="mb-3",
                ),
                html.H5("Basic idea", className="mb-2"),
                html.P(
                    "Along an ordered “gradient” (e.g. biofilm regions / GrowthPhase), "
                    "compare transcriptome clusters step by step and find which other "
                    "metadata conditions best explain differences between neighbouring "
                    "(or matched) clusters — with significance assessed against the "
                    "rest of the dataset.",
                    className="mb-2",
                ),
                html.Ul(
                    [
                        html.Li(
                            "Subset samples (e.g. Biofilm = True) and pick a gradient "
                            "column (e.g. GrowthPhase) with a custom level order."
                        ),
                        html.Li(
                            "For each gradient level, run hierarchical clustering on "
                            "the subset and identify the closest / matching cluster "
                            "across adjacent levels (or vs a reference)."
                        ),
                        html.Li(
                            "Search condition columns for the one that best correlates "
                            "with differences between those paired clusters."
                        ),
                        html.Li(
                            "Assess significance using the remainder of the dataset "
                            "(held-out / background samples) as a null comparison."
                        ),
                    ],
                    className="mb-3",
                ),
                html.P(
                    "Nothing to run on this tab yet.",
                    className="text-muted small mb-0",
                ),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        return
