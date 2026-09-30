"""Shared analysis chrome: method row, then Samples / Genes / Enrichment stacked."""

from __future__ import annotations

from dash import Dash, Input, Output, State, callback_context, dcc, html
import dash_bootstrap_components as dbc

LEVEL_METHODS = [
    {"id": "pca", "label": "PCA"},
    {"id": "hc", "label": "Hierarchical clustering"},
    {"id": "volcano-condition", "label": "Volcano by condition"},
    {"id": "parallel-conditions", "label": "Parallel conditions"},
    {"id": "gene-gradients", "label": "Spatial gradient"},
]

# Methods where the shared level-3 enrichment panel is not applicable.
_NO_ENRICHMENT = frozenset({"volcano-condition"})


def workspace_layout(method_modules: list, level3) -> html.Div:
    by_id = {m.id: m for m in method_modules}
    l1_children = []
    l2_children = []
    l3_children = []
    for spec in LEVEL_METHODS:
        mod = by_id[spec["id"]]
        l1 = mod.level1()
        l2 = mod.level2() if hasattr(mod, "level2") else html.Div()
        l3 = mod.level3() if hasattr(mod, "level3") else html.Div()
        visible = {} if spec["id"] == "pca" else {"display": "none"}
        l1_children.append(html.Div(l1, id=f"ws-l1-{spec['id']}", style=visible))
        l2_children.append(html.Div(l2, id=f"ws-l2-{spec['id']}", style=visible))
        l3_children.append(html.Div(l3, id=f"ws-l3-{spec['id']}", style=visible))

    method_btns = [
        dbc.Button(
            spec["label"],
            id={"type": "ws-method", "index": spec["id"]},
            color="primary" if spec["id"] == "pca" else "secondary",
            outline=spec["id"] != "pca",
            className="ws-method-btn",
        )
        for spec in LEVEL_METHODS
    ]

    return html.Div(
        [
            dcc.Store(id="analysis-method", data="pca"),
            dcc.Store(id="analysis-selected-genes", data=[]),
            html.Div(
                id="ws-method-wrap",
                className="ws-method-bar",
                children=method_btns,
            ),
            html.Div(
                className="ws-section",
                children=[
                    html.H5("1. Samples", className="ws-section-title"),
                    html.Div(id="ws-level-1", children=l1_children),
                ],
            ),
            html.Div(
                className="ws-section",
                children=[
                    html.H5("2. Genes", className="ws-section-title"),
                    html.P("(optional)", className="text-muted small mb-2"),
                    html.Div(id="ws-level-2", children=l2_children),
                ],
            ),
            html.Div(
                id="ws-section-3",
                className="ws-section",
                children=[
                    html.H5("3. Condition enrichment", className="ws-section-title"),
                    html.P("(optional)", className="text-muted small mb-2"),
                    html.Div(
                        id="ws-level-3",
                        children=[html.Div(id="ws-level-3-shared", children=[level3])]
                        + l3_children,
                    ),
                ],
            ),
        ]
    )


def register_workspace_callbacks(app: Dash) -> None:
    method_ids = [s["id"] for s in LEVEL_METHODS]

    @app.callback(
        Output("analysis-method", "data"),
        *[Input({"type": "ws-method", "index": mid}, "n_clicks") for mid in method_ids],
        State("analysis-method", "data"),
        prevent_initial_call=True,
    )
    def _set_method(*args):
        current = args[-1]
        tid = callback_context.triggered_id
        if isinstance(tid, dict) and tid.get("type") == "ws-method":
            return tid["index"]
        return current or "pca"

    @app.callback(
        *[Output(f"ws-l1-{mid}", "style") for mid in method_ids],
        *[Output(f"ws-l2-{mid}", "style") for mid in method_ids],
        *[Output(f"ws-l3-{mid}", "style") for mid in method_ids],
        *[Output({"type": "ws-method", "index": mid}, "color") for mid in method_ids],
        *[Output({"type": "ws-method", "index": mid}, "outline") for mid in method_ids],
        Output("ws-section-3", "style"),
        Output("ws-level-3-shared", "style"),
        Input("analysis-method", "data"),
    )
    def _apply_visibility(method):
        method = method or "pca"
        l1_styles = [{} if mid == method else {"display": "none"} for mid in method_ids]
        l2_styles = [{} if mid == method else {"display": "none"} for mid in method_ids]
        l3_styles = [{} if mid == method else {"display": "none"} for mid in method_ids]
        m_color = ["primary" if mid == method else "secondary" for mid in method_ids]
        m_outline = [mid != method for mid in method_ids]
        # Hide whole section 3 for methods with no enrichment; otherwise show shared
        # panel except when the method supplies its own enrichment UI instead.
        if method in _NO_ENRICHMENT:
            section3 = {"display": "none"}
            shared = {"display": "none"}
        else:
            section3 = {}
            # Parallel conditions has its own enrichment UI in level3().
            shared = {"display": "none"} if method == "parallel-conditions" else {}
        return (*l1_styles, *l2_styles, *l3_styles, *m_color, *m_outline, section3, shared)
