"""Condition prediction — placeholder (ML / decision tree from transcriptome)."""

from __future__ import annotations

from dash import Dash, html
import dash_bootstrap_components as dbc


class ConditionPredictionModule:
    id = "condition-prediction"
    label = "Condition prediction"

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
                    "Predict sample conditions (metadata labels) from the transcriptome "
                    "using supervised ML — e.g. decision trees / random forests, and "
                    "optionally other classifiers — trained on expression features.",
                    className="mb-2",
                ),
                html.Ul(
                    [
                        html.Li(
                            "Choose target condition column(s) from sample metadata "
                            "(e.g. Biofilm, GrowthPhase, medium)."
                        ),
                        html.Li(
                            "Train/test split (and/or cross-validation); report accuracy, "
                            "confusion matrix, and feature (gene) importances."
                        ),
                        html.Li(
                            "Inspect which genes drive the decision tree / model "
                            "and optionally export ranked gene scores for Celov."
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
