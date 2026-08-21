"""PCA: interactive scatter + variance barplot, optional linear classifier."""

from __future__ import annotations

from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from ..components.controls import (
    EXPORT_H,
    EXPORT_PCA_SCREE_H,
    EXPORT_W,
    aesthetic_options,
    aesthetic_panel,
    apply_export_layout,
    build_scatter,
    equal_xy_axes,
    fig_size_controls,
    parse_aes_choice,
    set_fig_size,
)
from ..components.sample_detail import plot_with_sample_detail, register_sample_detail_callback
from ..data_store import SessionData, session_from_store
from .pca_classifier import (
    add_decision_boundary,
    classifier_performance,
    encode_binary_labels,
    fit_pc_classifier,
    performance_figure,
)

_AES_ALL = "__all__"
_AES_PREFIX = "pca-aes"
_CACHE_N_PCS = 50
_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "pca_plot"},
    "displaylogo": False,
}
_DEFAULT_AES = {
    "color": None,
    "shape": None,
    "size": None,
    "alpha": 0.85,
}

# Server-side PCA result (full scores / loadings) — local single-user app
_PCA_RUNTIME: dict = {}


def _run_pca(session: SessionData):
    """Same as notebook: ``PCA().fit(normalizedCounts)`` then ``transform``."""
    X = session.numeric_matrix()
    pca = PCA()
    pca.fit(X)
    scores = pca.transform(X)
    cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    score_df = pd.DataFrame(scores, index=session.expression.index, columns=cols)
    score_df = score_df.join(session.metadata)
    # Correlation loadings: components.T * sqrt(explained_variance) (notebook pcaLoadings)
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=session.expression.columns.astype(str),
        columns=cols,
    )
    return score_df, pca.explained_variance_ratio_, pca.explained_variance_, loadings


def _plotly_title(*lines: str) -> dict:
    text = "<br>".join(line for line in lines if line is not None and str(line).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


def _active_dataset_name(session_blob=None, active=None) -> str:
    if active:
        return active if isinstance(active, str) else (active[0] if active else "dataset")
    names = (session_blob or {}).get("active_datasets") or []
    return str(names[0]) if names else "dataset"


def _scree_trace(var_ratio, top_n: int = 10) -> go.Bar:
    var_ratio = list(map(float, var_ratio))
    n_show = min(int(top_n), len(var_ratio))
    pcs = [f"PC{i + 1}" for i in range(n_show)]
    vals = [round(100 * var_ratio[i], 1) for i in range(n_show)]
    return go.Bar(
        x=pcs,
        y=vals,
        text=vals,
        textposition="outside",
        showlegend=False,
        name="Variance explained",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    )


def _combine_pca_and_variance(scatter: go.Figure, var_ratio, title: dict | str) -> go.Figure:
    """Attach a variance-explained bar panel under the 2D PCA scatter (same figure)."""
    title_dict = title if isinstance(title, dict) else _plotly_title(str(title))
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    is_3d = any(getattr(tr, "type", None) == "scatter3d" for tr in scatter.data)
    if is_3d:
        scatter.update_layout(title=title_dict)
        apply_export_layout(scatter, title_lines=title_lines, legend=True, uirevision="pca-scatter")
        return scatter

    # Compact PCA on top (inset-like), scree underneath — same outer width as other figs
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.14,
        subplot_titles=("", "Variance explained"),
    )
    for tr in scatter.data:
        fig.add_trace(tr, row=1, col=1)
    fig.add_trace(_scree_trace(var_ratio, top_n=10), row=2, col=1)
    vals = [round(100 * float(v), 1) for v in list(var_ratio)[:10]]
    ymax = max(vals) * 1.22 if vals else 1.0

    # Copy axis titles from scatter
    xlab = scatter.layout.xaxis.title.text if scatter.layout.xaxis.title else ""
    ylab = scatter.layout.yaxis.title.text if scatter.layout.yaxis.title else ""
    fig.update_xaxes(title_text=xlab or None, row=1, col=1)
    fig.update_yaxes(title_text=ylab or None, row=1, col=1)
    # Preserve equal aspect from the stand-alone scatter when available
    if scatter.layout.xaxis and scatter.layout.xaxis.range:
        fig.update_xaxes(range=list(scatter.layout.xaxis.range), row=1, col=1)
    if scatter.layout.yaxis and scatter.layout.yaxis.range:
        fig.update_yaxes(range=list(scatter.layout.yaxis.range), scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="PC", row=2, col=1)
    fig.update_yaxes(
        title_text="Variance explained [%]",
        range=[0, ymax],
        row=2,
        col=1,
    )
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))

    fig.update_layout(title=title_dict)
    leg_title = None
    if scatter.layout.legend and scatter.layout.legend.title:
        leg_title = scatter.layout.legend.title.text
    apply_export_layout(
        fig,
        title_lines=title_lines,
        width=EXPORT_W,
        height=EXPORT_PCA_SCREE_H,
        legend=True,
        legend_kwargs={"title_text": leg_title} if leg_title else None,
        uirevision="pca-scatter",
    )
    return fig


def _pc_options(n_pcs: int) -> list[dict]:
    n = max(0, int(n_pcs))
    return [{"label": f"PC{i}", "value": f"PC{i}"} for i in range(1, n + 1)]


def _pc_index(pc: str | None) -> int | None:
    if not pc or not str(pc).startswith("PC"):
        return None
    tail = str(pc)[2:]
    if not tail.isdigit():
        return None
    return int(tail) - 1


def _axis_label(pc: str | None, var) -> str:
    if not pc:
        return ""
    var = list(var) if var is not None else []
    idx = _pc_index(pc)
    if idx is not None and 0 <= idx < len(var):
        return f"{pc} ({100 * var[idx]:.1f}%)"
    return str(pc)


def _resolve_aes(color, shape, size, alpha, columns):
    """Map UI choices → column/constant kwargs for build_scatter."""
    columns = list(columns)
    color_col, color_const = parse_aes_choice(color, columns)
    shape_col, shape_const = parse_aes_choice(shape, columns)
    size_col, size_raw = parse_aes_choice(size, columns)
    size_const = None
    if size_raw is not None:
        try:
            size_const = float(size_raw)
        except ValueError:
            size_const = 10.0
    return {
        "color_col": color_col,
        "color_const": color_const,
        "shape_col": shape_col,
        "shape_const": shape_const,
        "size_col": size_col,
        "size_const": size_const,
        "alpha": float(alpha) if alpha is not None else 0.85,
    }


def _scatter_single(
    score_df,
    x_col,
    y_col,
    z_col,
    color,
    shape,
    size,
    alpha,
    var,
    title="PCA",
) -> go.Figure:
    score_df = score_df.copy()
    aes = _resolve_aes(color, shape, size, alpha, score_df.columns)
    xlab = _axis_label(x_col, var)
    ylab = _axis_label(y_col, var)
    zcol = z_col if z_col and z_col in score_df.columns else None
    fig = build_scatter(
        score_df,
        x=x_col,
        y=y_col,
        z=zcol,
        legend_name="samples",
        title=title,
        **aes,
    )
    layout_kw = dict(uirevision="pca-scatter", showlegend=True)
    if zcol:
        fig.update_layout(
            scene=dict(
                xaxis_title=xlab,
                yaxis_title=ylab,
                zaxis_title=_axis_label(zcol, var),
            ),
            **layout_kw,
        )
        apply_export_layout(fig, title_lines=1, legend=True, uirevision="pca-scatter")
    else:
        fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, **layout_kw)
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
        apply_export_layout(fig, title_lines=1, legend=True, uirevision="pca-scatter")
    return fig


def _scatter_split(score_df, x_col, y_col, z_col, split_col, group_aes: dict, var) -> go.Figure:
    xlab = _axis_label(x_col, var)
    ylab = _axis_label(y_col, var)
    zcol = z_col if z_col and z_col in score_df.columns else None
    fig = go.Figure()
    levels = [str(v) for v in score_df[split_col].astype(str).unique()]
    for level in sorted(levels):
        sub = score_df.loc[score_df[split_col].astype(str) == level].copy()
        if sub.empty:
            continue
        raw = {**_DEFAULT_AES, **(group_aes or {}).get(level, {})}
        aes = _resolve_aes(
            raw.get("color"),
            raw.get("shape"),
            raw.get("size"),
            raw.get("alpha"),
            sub.columns,
        )
        part = build_scatter(
            sub,
            x=x_col,
            y=y_col,
            z=zcol,
            legend_name=level,
            title="",
            **aes,
        )
        for tr in part.data:
            name = tr.name
            if name in (None, "", "trace 0", "samples"):
                tr.name = level
            elif not str(name).startswith(level):
                tr.name = f"{level} | {name}"
            tr.legendgroup = level
            tr.showlegend = True
            fig.add_trace(tr)

    title_lines = 1
    layout_kw = dict(
        title=f"PCA (split by {split_col})",
        uirevision="pca-scatter",
        legend_title_text=split_col,
        showlegend=True,
    )
    if zcol:
        fig.update_layout(
            scene=dict(
                xaxis_title=xlab,
                yaxis_title=ylab,
                zaxis_title=_axis_label(zcol, var),
            ),
            **layout_kw,
        )
        apply_export_layout(
            fig,
            title_lines=title_lines,
            legend=True,
            legend_kwargs={"title_text": split_col},
            uirevision="pca-scatter",
        )
    else:
        fig.update_layout(xaxis_title=xlab, yaxis_title=ylab, **layout_kw)
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
        apply_export_layout(
            fig,
            title_lines=title_lines,
            legend=True,
            legend_kwargs={"title_text": split_col},
            uirevision="pca-scatter",
        )
    return fig


def _cache_plot_frame(score_df: pd.DataFrame) -> dict:
    pc_like = sorted(
        (c for c in score_df.columns if c.startswith("PC") and c[2:].isdigit()),
        key=lambda c: int(c[2:]),
    )
    keep = pc_like[:_CACHE_N_PCS]
    meta_cols = [c for c in score_df.columns if c not in set(pc_like)]
    slim = score_df.loc[:, keep + meta_cols].copy().reset_index(names="_sample_id")
    return slim.to_dict(orient="list")


def _frame_from_cache(cache: dict) -> pd.DataFrame | None:
    if not cache or "scores" not in cache:
        return None
    df = pd.DataFrame(cache["scores"])
    if "_sample_id" in df.columns:
        df = df.set_index("_sample_id")
    return df


def _score_frame_for_plot(cache: dict) -> pd.DataFrame | None:
    if "score_df" in _PCA_RUNTIME and isinstance(_PCA_RUNTIME["score_df"], pd.DataFrame):
        return _PCA_RUNTIME["score_df"]
    return _frame_from_cache(cache)


class PCAModule:
    id = "pca"
    label = "PCA"

    def layout(self):
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("X"),
                                dcc.Dropdown(id="pca-x", placeholder="PC1", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Y"),
                                dcc.Dropdown(id="pca-y", placeholder="PC2", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Z (optional 3D)"),
                                dcc.Dropdown(id="pca-z", placeholder="None", clearable=True),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run PCA", id="pca-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.H6("Split by group"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Split column"),
                                dcc.Dropdown(
                                    id="pca-split-col",
                                    placeholder="None (single aesthetics)",
                                    clearable=True,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.P(
                    "Dropdowns list metadata columns first, then fixed colors/shapes/sizes. "
                    "With a split column, each value gets its own aesthetics block below.",
                    className="text-muted small mb-2",
                ),
                html.Div(id="pca-aes-panels"),
                fig_size_controls(
                    "pca",
                    default_width=EXPORT_W,
                    default_height=EXPORT_PCA_SCREE_H,
                ),
                dcc.Loading(
                    plot_with_sample_detail(
                        "pca-scatter",
                        "pca-sample-detail",
                        graph_config=_GRAPH_CONFIG,
                    ),
                    type="default",
                ),
                html.Hr(),
                html.H6("Linear classifier"),
                html.P(
                    "Same as the notebook: sklearn LogisticRegression on PC scores → "
                    "train accuracy and CV balanced accuracy vs number of PCs; optional "
                    "2-PC decision boundary.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Label column"),
                                dcc.Dropdown(id="pca-clf-label"),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Positive class"),
                                dcc.Dropdown(id="pca-clf-positive"),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("PCs min"),
                                dbc.Input(id="pca-clf-pc-min", type="number", value=2, min=2, step=1),
                            ],
                            md=1,
                        ),
                        dbc.Col(
                            [
                                html.Label("PCs max"),
                                dbc.Input(id="pca-clf-pc-max", type="number", value=12, min=2, step=1),
                            ],
                            md=1,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Checklist(
                                    id="pca-clf-overlay",
                                    options=[{"label": "Plot 2-PC boundary on PCA", "value": "on"}],
                                    value=[],
                                    inline=True,
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button("Run linear classifier", id="pca-clf-run", color="primary", className="mb-2"),
                fig_size_controls(
                    "pca-clf",
                    default_width=EXPORT_W,
                    default_height=EXPORT_H,
                ),
                dcc.Graph(
                    id="pca-clf-perf",
                    figure=go.Figure(),
                    config={
                        **_GRAPH_CONFIG,
                        "toImageButtonOptions": {
                            **_GRAPH_CONFIG["toImageButtonOptions"],
                            "filename": "pca_classifier_performance",
                        },
                    },
                ),
                html.Div(id="pca-clf-status", className="text-muted small mb-2"),
                html.Div(id="pca-status", className="text-muted small"),
                dcc.Store(id="pca-cache"),
                dcc.Store(id="pca-group-aes", data={}),
                dcc.Store(id="pca-split-options", data=[]),
                dcc.Store(id="pca-clf-cache", data=None),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        def _meta_columns(session_blob, cache) -> list[str]:
            cols = list((session_blob or {}).get("meta_columns", []))
            if cache and "scores" in cache:
                sample = pd.DataFrame(cache["scores"])
                pc_like = {c for c in sample.columns if c.startswith("PC") and c[2:].isdigit()}
                cols = [c for c in sample.columns if c not in pc_like and c != "_sample_id"]
            return cols

        @app.callback(
            Output("pca-split-col", "options"),
            Output("pca-split-options", "data"),
            Input("session-store", "data"),
            Input("pca-cache", "data"),
        )
        def _fill_split(session_blob, cache):
            cols = _meta_columns(session_blob, cache)
            split_opts = [{"label": c, "value": c} for c in cols]
            return split_opts, cols

        @app.callback(
            Output("pca-aes-panels", "children"),
            Input("pca-split-col", "value"),
            Input("pca-cache", "data"),
            Input("pca-split-options", "data"),
            State("pca-group-aes", "data"),
        )
        def _build_aes_panels(split_col, cache, meta_cols, group_aes):
            meta_cols = list(meta_cols or [])
            color_opts, shape_opts, size_opts = aesthetic_options(meta_cols)
            group_aes = group_aes or {}
            panels: list = []

            if split_col and cache:
                df = _frame_from_cache(cache)
                if df is not None and split_col in df.columns:
                    levels = sorted(str(v) for v in df[split_col].astype(str).unique())
                    for level in levels:
                        panels.append(
                            aesthetic_panel(
                                _AES_PREFIX,
                                level,
                                heading=level,
                                color_opts=color_opts,
                                shape_opts=shape_opts,
                                size_opts=size_opts,
                                values={**_DEFAULT_AES, **group_aes.get(level, {})},
                            )
                        )
                    return panels

            panels.append(
                aesthetic_panel(
                    _AES_PREFIX,
                    _AES_ALL,
                    heading="Aesthetics",
                    color_opts=color_opts,
                    shape_opts=shape_opts,
                    size_opts=size_opts,
                    values={**_DEFAULT_AES, **group_aes.get(_AES_ALL, {})},
                )
            )
            return panels

        @app.callback(
            Output("pca-group-aes", "data"),
            Input({"type": f"{_AES_PREFIX}-color", "group": ALL}, "value"),
            Input({"type": f"{_AES_PREFIX}-shape", "group": ALL}, "value"),
            Input({"type": f"{_AES_PREFIX}-size", "group": ALL}, "value"),
            Input({"type": f"{_AES_PREFIX}-alpha", "group": ALL}, "value"),
            State({"type": f"{_AES_PREFIX}-color", "group": ALL}, "id"),
            prevent_initial_call=True,
        )
        def _sync_group_aes(colors, shapes, sizes, alphas, ids):
            if not ids:
                return {}
            data = {}
            for i, id_dict in enumerate(ids):
                g = str(id_dict["group"])
                data[g] = {
                    "color": colors[i] if i < len(colors) else None,
                    "shape": shapes[i] if i < len(shapes) else None,
                    "size": sizes[i] if i < len(sizes) else None,
                    "alpha": float(alphas[i]) if i < len(alphas) and alphas[i] is not None else 0.85,
                }
            return data

        @app.callback(
            Output("pca-cache", "data"),
            Output("pca-status", "children"),
            Output("pca-clf-cache", "data", allow_duplicate=True),
            Output("pca-x", "options"),
            Output("pca-y", "options"),
            Output("pca-z", "options"),
            Output("pca-x", "value"),
            Output("pca-y", "value"),
            Output("pca-z", "value"),
            Input("pca-run", "n_clicks"),
            State("session-store", "data"),
            prevent_initial_call=True,
        )
        def _compute(n_clicks, session_blob):
            session = session_from_store(session_blob)
            if not session.ready:
                return (
                    no_update,
                    session.error or "Load datasets first.",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )
            try:
                score_df, var_ratio, explained_var, loadings = _run_pca(session)
                _PCA_RUNTIME.clear()
                _PCA_RUNTIME.update(
                    {
                        "score_df": score_df,
                        "loadings": loadings,
                        "explained_variance": np.asarray(explained_var, dtype=float),
                        "var_ratio": list(map(float, var_ratio)),
                    }
                )
                n_pcs = int(len(var_ratio))
                opts = _pc_options(n_pcs)
                cache = {
                    "scores": _cache_plot_frame(score_df),
                    "var_ratio": list(map(float, var_ratio)),
                    "n_pcs": n_pcs,
                }
                return (
                    cache,
                    f"PCA done: {score_df.shape[0]} samples, {n_pcs} components.",
                    None,
                    opts,
                    opts,
                    opts,
                    "PC1" if n_pcs >= 1 else None,
                    "PC2" if n_pcs >= 2 else ("PC1" if n_pcs >= 1 else None),
                    None,
                )
            except Exception as exc:  # noqa: BLE001
                return (
                    no_update,
                    f"PCA error: {exc}",
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                    no_update,
                )

        @app.callback(
            Output("pca-clf-label", "options"),
            Input("session-store", "data"),
            Input("pca-cache", "data"),
        )
        def _fill_clf_label(session_blob, cache):
            cols = _meta_columns(session_blob, cache)
            return [{"label": c, "value": c} for c in cols]

        @app.callback(
            Output("pca-clf-positive", "options"),
            Output("pca-clf-positive", "value"),
            Input("pca-clf-label", "value"),
            State("pca-cache", "data"),
            State("pca-clf-positive", "value"),
        )
        def _fill_positive(label_col, cache, current):
            if not label_col or not cache:
                return [], None
            df = _frame_from_cache(cache)
            if df is None or label_col not in df.columns:
                return [], None
            levels = sorted(str(v) for v in df[label_col].dropna().astype(str).unique())
            opts = [{"label": v, "value": v} for v in levels]
            value = current if current in levels else (levels[-1] if levels else None)
            return opts, value

        @app.callback(
            Output("pca-clf-cache", "data", allow_duplicate=True),
            Output("pca-clf-perf", "figure"),
            Output("pca-clf-status", "children"),
            Input("pca-clf-run", "n_clicks"),
            Input("pca-clf-fig-w", "value"),
            Input("pca-clf-fig-h", "value"),
            State("pca-cache", "data"),
            State("pca-clf-label", "value"),
            State("pca-clf-positive", "value"),
            State("pca-clf-pc-min", "value"),
            State("pca-clf-pc-max", "value"),
            State("pca-clf-cache", "data"),
            State("session-store", "data"),
            State("ds-active", "value"),
            prevent_initial_call=True,
        )
        def _run_classifier(
            n_clicks,
            fig_w,
            fig_h,
            cache,
            label_col,
            positive,
            pc_min,
            pc_max,
            clf_cache,
            session_blob,
            active,
        ):
            empty = go.Figure()
            triggered = callback_context.triggered_id
            dataset = _active_dataset_name(session_blob, active)

            if triggered in ("pca-clf-fig-w", "pca-clf-fig-h"):
                if not clf_cache or not clf_cache.get("perf"):
                    return no_update, no_update, no_update
                perf = pd.DataFrame(clf_cache["perf"])
                fig = performance_figure(
                    perf,
                    title=_plotly_title(
                        f"Linear Classifier on PCs for {dataset}",
                        f"split on {clf_cache.get('label_col')} "
                        f"(positive = {clf_cache.get('positive')})",
                    ),
                )
                return no_update, set_fig_size(fig, fig_w, fig_h), no_update

            if not cache or "score_df" not in _PCA_RUNTIME:
                return no_update, empty, "Run PCA first."
            if not label_col or positive is None or positive == "":
                return no_update, empty, "Choose label column and positive class."
            try:
                score_df = _PCA_RUNTIME["score_df"]
                y = encode_binary_labels(score_df[label_col], positive)
                perf = classifier_performance(score_df, y, int(pc_min or 2), int(pc_max or 12))
                fig = performance_figure(
                    perf,
                    title=_plotly_title(
                        f"Linear Classifier on PCs for {dataset}",
                        f"split on {label_col} (positive = {positive})",
                    ),
                )
                fig = set_fig_size(fig, fig_w, fig_h)
                res2 = fit_pc_classifier(score_df, y, 2)
                new_cache = {
                    "label_col": label_col,
                    "positive": str(positive),
                    "perf": perf.to_dict(orient="list") if perf is not None else {},
                    "w2": list(map(float, res2["w"])) if res2 else None,
                    "b2": float(res2["b"]) if res2 else None,
                    "cv2": float(res2["cv_acc"]) if res2 else None,
                }
                msg = (
                    f"Classifier done. 2-PC CV={new_cache['cv2']:.3f}."
                    if res2
                    else "Classifier finished with missing fits (check class sizes / n_pcs)."
                )
                return new_cache, fig, msg
            except Exception as exc:  # noqa: BLE001
                return no_update, empty, f"Classifier error: {exc}"

        @app.callback(
            Output("pca-scatter", "figure"),
            Input("pca-cache", "data"),
            Input("pca-x", "value"),
            Input("pca-y", "value"),
            Input("pca-z", "value"),
            Input("pca-split-col", "value"),
            Input("pca-group-aes", "data"),
            Input("pca-clf-cache", "data"),
            Input("pca-clf-overlay", "value"),
            Input("pca-fig-w", "value"),
            Input("pca-fig-h", "value"),
            Input("session-store", "data"),
            Input("ds-active", "value"),
        )
        def _replot(
            cache,
            x_col,
            y_col,
            z_col,
            split_col,
            group_aes,
            clf_cache,
            overlay,
            fig_w,
            fig_h,
            session_blob,
            active,
        ):
            empty = go.Figure()
            if not cache:
                return empty
            score_df = _score_frame_for_plot(cache)
            var = cache.get("var_ratio") or _PCA_RUNTIME.get("var_ratio", [])
            if score_df is None:
                return empty
            x_col = x_col or "PC1"
            y_col = y_col or "PC2"
            if x_col not in score_df.columns or y_col not in score_df.columns:
                return empty
            group_aes = group_aes or {}
            dataset = _active_dataset_name(session_blob, active)
            if split_col and split_col in score_df.columns:
                title = _plotly_title(f"PCA for {dataset}", f"split by {split_col}")
            else:
                title = _plotly_title(f"PCA for {dataset}")
            try:
                if split_col and split_col in score_df.columns:
                    levels = [str(v) for v in score_df[split_col].astype(str).unique()]
                    merged = {lv: {**_DEFAULT_AES, **group_aes.get(lv, {})} for lv in levels}
                    scatter = _scatter_split(score_df, x_col, y_col, z_col, split_col, merged, var)
                else:
                    raw = {**_DEFAULT_AES, **group_aes.get(_AES_ALL, {})}
                    scatter = _scatter_single(
                        score_df,
                        x_col,
                        y_col,
                        z_col,
                        raw.get("color"),
                        raw.get("shape"),
                        raw.get("size"),
                        raw.get("alpha"),
                        var,
                    )
                if (
                    overlay
                    and "on" in (overlay or [])
                    and not z_col
                    and x_col == "PC1"
                    and y_col == "PC2"
                    and clf_cache
                    and clf_cache.get("w2") is not None
                ):
                    name = f"2-PC classifier (CV {clf_cache.get('cv2', float('nan')):.2f})"
                    scatter = add_decision_boundary(
                        scatter, score_df, clf_cache["w2"], clf_cache["b2"], name=name
                    )
                fig = _combine_pca_and_variance(scatter, var, title)
                return set_fig_size(
                    fig, fig_w, fig_h, default_height=EXPORT_PCA_SCREE_H
                )
            except Exception as exc:  # noqa: BLE001
                err = go.Figure()
                err.add_annotation(text=f"Plot error: {exc}", showarrow=False)
                return err

        register_sample_detail_callback(
            app,
            graph_id="pca-scatter",
            detail_id="pca-sample-detail",
            cache_id="pca-cache",
            get_score_df=_score_frame_for_plot,
        )
