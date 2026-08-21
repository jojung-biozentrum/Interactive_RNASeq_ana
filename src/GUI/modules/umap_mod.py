"""UMAP: interactive scatter with PCA-like aesthetics (notebook UMAP.ipynb)."""

from __future__ import annotations

from dash import ALL, Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

from ..components.controls import (
    EXPORT_H,
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

_AES_ALL = "__all__"
_AES_PREFIX = "umap-aes"
_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "umap_plot"},
    "displaylogo": False,
}
_DEFAULT_AES = {
    "color": None,
    "shape": None,
    "size": None,
    "alpha": 0.85,
}

# Notebook defaults: umap.UMAP(n_components=2, random_state=100, min_dist=0.5, n_neighbors=20)
_NOTEBOOK_N_NEIGHBORS = 20
_NOTEBOOK_MIN_DIST = 0.5
_NOTEBOOK_RANDOM_STATE = 100
_NOTEBOOK_N_COMPONENTS = 2

_UMAP_RUNTIME: dict = {}


def _run_umap(
    session: SessionData,
    *,
    n_neighbors: int = _NOTEBOOK_N_NEIGHBORS,
    min_dist: float = _NOTEBOOK_MIN_DIST,
    n_components: int = _NOTEBOOK_N_COMPONENTS,
    random_state: int = _NOTEBOOK_RANDOM_STATE,
) -> pd.DataFrame:
    """Same as UMAP.ipynb: unscaled samples × genes matrix, then fit_transform."""
    import umap

    if session.expression is None or session.metadata is None:
        raise RuntimeError("No session data")

    X = session.numeric_matrix()
    n_components = max(2, min(int(n_components), X.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        min_dist=min_dist,
        n_neighbors=n_neighbors,
    )
    embedding = reducer.fit_transform(X)
    cols = [f"UMAP{i + 1}" for i in range(embedding.shape[1])]
    score_df = pd.DataFrame(embedding, index=session.expression.index, columns=cols)
    score_df = score_df.join(session.metadata)
    return score_df


def _resolve_aes(color, shape, size, alpha, columns):
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


def _plotly_title(*lines: str) -> dict:
    text = "<br>".join(line for line in lines if line is not None and str(line).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


def _active_dataset_name(session_blob=None, active=None) -> str:
    if active:
        return active if isinstance(active, str) else (active[0] if active else "dataset")
    names = (session_blob or {}).get("active_datasets") or []
    return str(names[0]) if names else "dataset"


def _scatter_single(
    score_df,
    x_col,
    y_col,
    z_col=None,
    color=None,
    shape=None,
    size=None,
    alpha=0.85,
    title="UMAP",
) -> go.Figure:
    aes = _resolve_aes(color, shape, size, alpha, score_df.columns)
    zcol = z_col if z_col and z_col in score_df.columns else None
    title_dict = title if isinstance(title, dict) else _plotly_title(str(title))
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    fig = build_scatter(
        score_df,
        x=x_col,
        y=y_col,
        z=zcol,
        legend_name="samples",
        title="",
        **aes,
    )
    layout_kw = dict(
        title=title_dict,
        uirevision="umap-scatter",
        showlegend=True,
    )
    if zcol:
        fig.update_layout(
            scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=zcol),
            **layout_kw,
        )
        apply_export_layout(fig, title_lines=title_lines, legend=True, uirevision="umap-scatter")
    else:
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, **layout_kw)
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
        apply_export_layout(fig, title_lines=title_lines, legend=True, uirevision="umap-scatter")
    return fig


def _scatter_split(
    score_df, x_col, y_col, z_col, split_col, group_aes: dict, title: str | dict = "UMAP"
) -> go.Figure:
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

    title_dict = title if isinstance(title, dict) else _plotly_title(str(title))
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    layout_kw = dict(
        title=title_dict,
        uirevision="umap-scatter",
        legend_title_text=split_col,
        showlegend=True,
    )
    if zcol:
        fig.update_layout(
            scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=zcol),
            **layout_kw,
        )
        apply_export_layout(
            fig,
            title_lines=title_lines,
            legend=True,
            legend_kwargs={"title_text": split_col},
            uirevision="umap-scatter",
        )
    else:
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, **layout_kw)
        fig = equal_xy_axes(fig, score_df, x_col, y_col)
        apply_export_layout(
            fig,
            title_lines=title_lines,
            legend=True,
            legend_kwargs={"title_text": split_col},
            uirevision="umap-scatter",
        )
    return fig


def _umap_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        (c for c in df.columns if c.startswith("UMAP") and c[4:].isdigit()),
        key=lambda c: int(c[4:]),
    )


def _cache_plot_frame(score_df: pd.DataFrame) -> dict:
    keep = _umap_cols(score_df)
    meta_cols = [c for c in score_df.columns if c not in set(keep)]
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
    if "score_df" in _UMAP_RUNTIME and isinstance(_UMAP_RUNTIME["score_df"], pd.DataFrame):
        return _UMAP_RUNTIME["score_df"]
    return _frame_from_cache(cache)


class UMAPModule:
    id = "umap"
    label = "UMAP"

    def layout(self):
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("n_neighbors"),
                                dbc.Input(
                                    id="umap-n-neighbors",
                                    type="number",
                                    value=_NOTEBOOK_N_NEIGHBORS,
                                    min=2,
                                    step=1,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("min_dist"),
                                dbc.Input(
                                    id="umap-min-dist",
                                    type="number",
                                    value=_NOTEBOOK_MIN_DIST,
                                    min=0.0,
                                    max=1.0,
                                    step=0.05,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("n_components"),
                                dbc.Input(
                                    id="umap-n-components",
                                    type="number",
                                    value=_NOTEBOOK_N_COMPONENTS,
                                    min=2,
                                    max=10,
                                    step=1,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("random_state"),
                                dbc.Input(
                                    id="umap-random-state",
                                    type="number",
                                    value=_NOTEBOOK_RANDOM_STATE,
                                    step=1,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run UMAP", id="umap-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("X"),
                                dcc.Dropdown(id="umap-x", placeholder="UMAP1", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Y"),
                                dcc.Dropdown(id="umap-y", placeholder="UMAP2", clearable=False),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("Z (optional 3D)"),
                                dcc.Dropdown(id="umap-z", placeholder="None", clearable=True),
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
                                    id="umap-split-col",
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
                    "Legend lists color, shape, and size separately (like seaborn relplot). "
                    "With a split column, each value gets its own aesthetics block below.",
                    className="text-muted small mb-2",
                ),
                html.Div(id="umap-aes-panels"),
                fig_size_controls("umap", default_width=EXPORT_W, default_height=EXPORT_H),
                dcc.Loading(
                    plot_with_sample_detail(
                        "umap-scatter",
                        "umap-sample-detail",
                        graph_config=_GRAPH_CONFIG,
                    ),
                    type="default",
                ),
                html.Div(id="umap-status", className="text-muted small mt-2"),
                dcc.Store(id="umap-cache"),
                dcc.Store(id="umap-group-aes", data={}),
                dcc.Store(id="umap-split-options", data=[]),
            ]
        )

    def register_callbacks(self, app: Dash) -> None:
        def _meta_columns(session_blob, cache) -> list[str]:
            cols = list((session_blob or {}).get("meta_columns", []))
            if cache and "scores" in cache:
                sample = pd.DataFrame(cache["scores"])
                umap_like = {c for c in sample.columns if c.startswith("UMAP") and c[4:].isdigit()}
                cols = [c for c in sample.columns if c not in umap_like and c != "_sample_id"]
            return cols

        @app.callback(
            Output("umap-split-col", "options"),
            Output("umap-split-options", "data"),
            Input("session-store", "data"),
            Input("umap-cache", "data"),
        )
        def _fill_split(session_blob, cache):
            cols = _meta_columns(session_blob, cache)
            return [{"label": c, "value": c} for c in cols], cols

        @app.callback(
            Output("umap-aes-panels", "children"),
            Input("umap-split-col", "value"),
            Input("umap-cache", "data"),
            Input("umap-split-options", "data"),
            State("umap-group-aes", "data"),
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
            Output("umap-group-aes", "data"),
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
            Output("umap-cache", "data"),
            Output("umap-status", "children"),
            Output("umap-x", "options"),
            Output("umap-y", "options"),
            Output("umap-z", "options"),
            Output("umap-x", "value"),
            Output("umap-y", "value"),
            Output("umap-z", "value"),
            Input("umap-run", "n_clicks"),
            State("session-store", "data"),
            State("umap-n-neighbors", "value"),
            State("umap-min-dist", "value"),
            State("umap-n-components", "value"),
            State("umap-random-state", "value"),
            prevent_initial_call=True,
        )
        def _compute(n_clicks, session_blob, n_neighbors, min_dist, n_components, random_state):
            session = session_from_store(session_blob)
            if not session.ready:
                return no_update, session.error or "Load datasets first.", [], [], [], None, None, None
            try:
                score_df = _run_umap(
                    session,
                    n_neighbors=int(n_neighbors or _NOTEBOOK_N_NEIGHBORS),
                    min_dist=float(min_dist if min_dist is not None else _NOTEBOOK_MIN_DIST),
                    n_components=int(n_components or _NOTEBOOK_N_COMPONENTS),
                    random_state=int(random_state if random_state is not None else _NOTEBOOK_RANDOM_STATE),
                )
                _UMAP_RUNTIME.clear()
                _UMAP_RUNTIME["score_df"] = score_df
                cache = {"scores": _cache_plot_frame(score_df)}
                coords = _umap_cols(score_df)
                opts = [{"label": c, "value": c} for c in coords]
                x_val = coords[0] if coords else None
                y_val = coords[1] if len(coords) > 1 else x_val
                return (
                    cache,
                    f"UMAP done: {score_df.shape[0]} samples, {len(coords)} components.",
                    opts,
                    opts,
                    opts,
                    x_val,
                    y_val,
                    None,
                )
            except ImportError:
                return (
                    no_update,
                    "UMAP requires umap-learn (`pip install umap-learn`).",
                    [],
                    [],
                    [],
                    None,
                    None,
                    None,
                )
            except Exception as exc:  # noqa: BLE001
                return no_update, f"UMAP error: {exc}", [], [], [], None, None, None

        @app.callback(
            Output("umap-scatter", "figure"),
            Input("umap-cache", "data"),
            Input("umap-x", "value"),
            Input("umap-y", "value"),
            Input("umap-z", "value"),
            Input("umap-split-col", "value"),
            Input("umap-group-aes", "data"),
            Input("umap-fig-w", "value"),
            Input("umap-fig-h", "value"),
            Input("session-store", "data"),
            Input("ds-active", "value"),
        )
        def _replot(
            cache, x_col, y_col, z_col, split_col, group_aes, fig_w, fig_h, session_blob, active
        ):
            empty = go.Figure()
            if not cache:
                return empty
            score_df = _score_frame_for_plot(cache)
            if score_df is None:
                return empty
            coords = _umap_cols(score_df)
            x_col = x_col if x_col in score_df.columns else (coords[0] if coords else None)
            y_col = y_col if y_col in score_df.columns else (coords[1] if len(coords) > 1 else x_col)
            if not x_col or not y_col:
                return empty
            group_aes = group_aes or {}
            dataset = _active_dataset_name(session_blob, active)
            if split_col and split_col in score_df.columns:
                title = _plotly_title(f"UMAP for {dataset}", f"split by {split_col}")
            else:
                title = _plotly_title(f"UMAP for {dataset}")
            try:
                if split_col and split_col in score_df.columns:
                    levels = [str(v) for v in score_df[split_col].astype(str).unique()]
                    merged = {lv: {**_DEFAULT_AES, **group_aes.get(lv, {})} for lv in levels}
                    fig = _scatter_split(
                        score_df, x_col, y_col, z_col, split_col, merged, title=title
                    )
                else:
                    raw = {**_DEFAULT_AES, **group_aes.get(_AES_ALL, {})}
                    fig = _scatter_single(
                        score_df,
                        x_col,
                        y_col,
                        z_col,
                        raw.get("color"),
                        raw.get("shape"),
                        raw.get("size"),
                        raw.get("alpha"),
                        title=title,
                    )
                return set_fig_size(fig, fig_w, fig_h)
            except Exception as exc:  # noqa: BLE001
                err = go.Figure()
                err.add_annotation(text=f"Plot error: {exc}", showarrow=False)
                return err

        register_sample_detail_callback(
            app,
            graph_id="umap-scatter",
            detail_id="umap-sample-detail",
            cache_id="umap-cache",
            get_score_df=_score_frame_for_plot,
        )
