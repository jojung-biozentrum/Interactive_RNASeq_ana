"""Shared aesthetic controls (column mapping or constant values)."""

from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_ID_PRIORITY = (
    "fileName",
    "FileName",
    "filename",
    "geneName",
    "Gene name",
    "Gene Name",
    "sample_id",
    "sampleID",
    "geneID",
    "gene_id",
    "locusTag",
    "biocyc_id",
)

_COORD_PREFIXES = ("PC", "UMAP")
_COORD_EXACT = {"x", "y", "z", "hover_name", "abs_gene_weight"}

# Dropdown value prefixes
CONST = "const:"
COL = "col:"

COLOR_CONSTANTS = [
    "black",
    "steelblue",
    "crimson",
    "seagreen",
    "darkorange",
    "purple",
    "gray",
    "gold",
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
]

SHAPE_CONSTANTS = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-up",
    "triangle-down",
    "star",
    "hexagon",
]

SIZE_CONSTANTS = ["6", "8", "10", "12", "16", "20", "28"]

_LEGEND_GREY = "#808080"
_LEGEND_STD_SIZE = 10.0
_QUALITATIVE = px.colors.qualitative.Plotly
# Index 0 (circle) is reserved for missing/None and the unset default; column
# categories map from index 1 onward so None ≠ first real entry.
_SYMBOL_CYCLE = [
    "circle",
    "triangle-up",
    "square",
    "diamond",
    "cross",
    "x",
    "triangle-down",
    "star",
    "hexagon",
    "pentagon",
]
# scatter3d.marker.symbol only accepts this short list
_SYMBOL_CYCLE_3D = [
    "circle",
    "square",
    "diamond",
    "cross",
    "x",
    "circle-open",
    "square-open",
    "diamond-open",
]
_SYMBOL_3D_ALLOWED = set(_SYMBOL_CYCLE_3D)
_SYMBOL_TO_3D = {
    "triangle-up": "diamond",
    "triangle-down": "diamond-open",
    "star": "cross",
    "hexagon": "square",
    "pentagon": "square-open",
}

_MISSING_LABEL = "None"
_MISSING_COLOR = "#b0b0b0"
_MISSING_SYMBOL = "circle"
_DEFAULT_COLOR = _QUALITATIVE[0]
_DEFAULT_SYMBOL = "circle"
_DEFAULT_SIZE = 8.0

# Inkscape-ready export canvas: compact square plot + legend strip outside
EXPORT_PLOT = 420
EXPORT_LEGEND_W = 150
EXPORT_W = EXPORT_PLOT + EXPORT_LEGEND_W
EXPORT_H = EXPORT_PLOT + 60
# PCA + scree: smaller PCA panel, same outer width as other export figs
EXPORT_PCA_SCREE_H = 560


def outside_legend(**extra) -> dict:
    """Legend parked to the right of the plot domain (not over data)."""
    base = dict(
        x=1.02,
        y=1.0,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        borderwidth=0,
        itemsizing="constant",
        tracegroupgap=2,
        font=dict(size=11),
    )
    base.update(extra)
    return base


def export_margins(title_lines: int = 1, *, legend: bool = True, bottom: int = 50) -> dict:
    """Margins that keep axis labels inside and the legend outside on the right."""
    return dict(
        l=55,
        r=EXPORT_LEGEND_W if legend else 40,
        t=36 + 22 * max(1, int(title_lines)),
        b=bottom,
    )


def apply_export_layout(
    fig: go.Figure,
    *,
    title_lines: int = 1,
    width: int | None = EXPORT_W,
    height: int | None = EXPORT_H,
    legend: bool = True,
    legend_kwargs: dict | None = None,
    uirevision: str | None = None,
    bottom: int = 50,
) -> go.Figure:
    """Fixed pixel size + outside legend for drop-in SVG/Inkscape use."""
    layout: dict = {
        "margin": export_margins(title_lines, legend=legend, bottom=bottom),
        "showlegend": legend,
    }
    if width is not None:
        layout["width"] = int(width)
    if height is not None:
        layout["height"] = int(height)
    if legend:
        layout["legend"] = outside_legend(**(legend_kwargs or {}))
    if uirevision is not None:
        layout["uirevision"] = uirevision
    fig.update_layout(**layout)
    return fig


def coerce_fig_size(
    width,
    height,
    *,
    default_width: int = EXPORT_W,
    default_height: int = EXPORT_H,
    min_size: int = 200,
) -> tuple[int, int]:
    """Clamp interactive width/height inputs to usable pixel sizes."""
    try:
        w = int(width) if width is not None and str(width).strip() != "" else int(default_width)
    except (TypeError, ValueError):
        w = int(default_width)
    try:
        h = int(height) if height is not None and str(height).strip() != "" else int(default_height)
    except (TypeError, ValueError):
        h = int(default_height)
    return max(min_size, w), max(min_size, h)


def set_fig_size(
    fig: go.Figure,
    width=None,
    height=None,
    *,
    default_width: int = EXPORT_W,
    default_height: int = EXPORT_H,
) -> go.Figure:
    """Override figure pixel size (keeps margins/legend from ``apply_export_layout``)."""
    w, h = coerce_fig_size(
        width, height, default_width=default_width, default_height=default_height
    )
    fig.update_layout(width=w, height=h)
    return fig


def fig_size_controls(
    prefix: str,
    *,
    default_width: int = EXPORT_W,
    default_height: int = EXPORT_H,
    heading: str | None = "Figure size (px)",
) -> html.Div:
    """Width/height number inputs; ids are ``{prefix}-fig-w`` / ``{prefix}-fig-h``."""
    kids: list = []
    if heading:
        kids.append(html.Span(heading, className="small text-muted me-2"))
    kids.append(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Width", className="small mb-0"),
                        dbc.Input(
                            id=f"{prefix}-fig-w",
                            type="number",
                            value=int(default_width),
                            min=200,
                            step=10,
                            className="form-control-sm",
                        ),
                    ],
                    xs=6,
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Height", className="small mb-0"),
                        dbc.Input(
                            id=f"{prefix}-fig-h",
                            type="number",
                            value=int(default_height),
                            min=200,
                            step=10,
                            className="form-control-sm",
                        ),
                    ],
                    xs=6,
                    md=2,
                ),
            ],
            className="g-2 mb-2 align-items-end",
        )
    )
    return html.Div(kids, className="mb-1")


def const_value(v: str) -> str:
    return f"{CONST}{v}"


def col_value(v: str) -> str:
    return f"{COL}{v}"


def aesthetic_options(columns: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Build dropdown options: metadata columns first, then fixed constants."""
    color_opts = [{"label": f"column: {c}", "value": col_value(c)} for c in columns]
    color_opts += [{"label": f"color: {c}", "value": const_value(c)} for c in COLOR_CONSTANTS]

    shape_opts = [{"label": f"column: {c}", "value": col_value(c)} for c in columns]
    shape_opts += [{"label": f"shape: {s}", "value": const_value(s)} for s in SHAPE_CONSTANTS]

    size_opts = [{"label": f"column: {c}", "value": col_value(c)} for c in columns]
    size_opts += [{"label": f"size: {s}", "value": const_value(s)} for s in SIZE_CONSTANTS]
    return color_opts, shape_opts, size_opts


def parse_aes_choice(value: str | None, columns: list[str] | None = None) -> tuple[str | None, str | None]:
    """Return ``(column_name, constant_value)`` — at most one is set."""
    if value is None or value == "":
        return None, None
    value = str(value)
    cols = set(columns or [])

    if value.startswith(COL):
        return value[len(COL) :], None
    if value.startswith(CONST):
        return None, value[len(CONST) :]

    if value in cols:
        return value, None
    return None, value


_ALL_GROUP = "__all__"


def aesthetic_panel(
    prefix: str,
    group: str,
    *,
    heading: str | None = "Aesthetics",
    color_opts: list[dict] | None = None,
    shape_opts: list[dict] | None = None,
    size_opts: list[dict] | None = None,
    values: dict | None = None,
) -> html.Div:
    """One aesthetics block (pattern-matching ids). ``group`` is ``__all__`` or a split level."""
    values = values or {}
    color_opts = color_opts or []
    shape_opts = shape_opts or []
    size_opts = size_opts or []

    def pid(suffix: str) -> dict:
        return {"type": f"{prefix}-{suffix}", "group": group}

    children: list = []
    if heading:
        children.append(html.H6(heading, className="mt-2 mb-2"))
    children.append(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Color"),
                        dcc.Dropdown(
                            id=pid("color"),
                            options=color_opts,
                            value=values.get("color"),
                            placeholder="column or constant",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Shape"),
                        dcc.Dropdown(
                            id=pid("shape"),
                            options=shape_opts,
                            value=values.get("shape"),
                            placeholder="column or constant",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Size"),
                        dcc.Dropdown(
                            id=pid("size"),
                            options=size_opts,
                            value=values.get("size"),
                            placeholder="column or constant",
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Alpha"),
                        dcc.Slider(
                            id=pid("alpha"),
                            min=0.1,
                            max=1.0,
                            step=0.05,
                            value=float(values.get("alpha", 0.85)),
                            marks={0.1: "0.1", 0.5: "0.5", 1.0: "1"},
                        ),
                    ],
                    md=3,
                ),
            ],
            className="g-2",
        )
    )
    return html.Div(children, className="mb-3 border-bottom pb-2")


def aesthetic_controls(prefix: str = "aes") -> html.Div:
    """Legacy single-block aesthetics (fixed string ids). Prefer ``aesthetic_panel``."""
    return html.Div(
        [
            html.H6("Aesthetics", className="mt-2 mb-2"),
            html.P(
                "Choose a metadata column or a fixed value. "
                "Legend lists color, shape, and size mappings separately (like seaborn relplot).",
                className="text-muted small mb-2",
            ),
            aesthetic_panel(prefix, _ALL_GROUP, heading=None),
        ],
        className="mb-3",
    )


def _is_coord_col(name: str) -> bool:
    if name in _COORD_EXACT:
        return True
    return any(name.startswith(p) and name[len(p) :].isdigit() for p in _COORD_PREFIXES)


def prepare_hover_frame(
    df: pd.DataFrame,
    id_col: str | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Prepare plot frame with ``_click_id`` (sample index) for click lookup."""
    plot_df = df.copy()
    plot_df["_click_id"] = plot_df.index.astype(str)
    plot_df = plot_df.reset_index(drop=True)

    hover_name = id_col
    if hover_name is None or hover_name not in plot_df.columns:
        hover_name = next((c for c in _ID_PRIORITY if c in plot_df.columns), None)
    if hover_name is None:
        hover_name = "_click_id"

    return plot_df, hover_name, []


def equal_xy_axes(fig: go.Figure, df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
    """Square plot area: equal x/y axis ranges and 1:1 scaling."""
    if x_col not in df.columns or y_col not in df.columns:
        return fig
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin, 1e-9)
    half += 0.05 * half
    fig.update_xaxes(range=[cx - half, cx + half], scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(range=[cy - half, cy + half], constrain="domain")
    return fig


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _as_float(v) -> float | None:
    """Parse ints/floats and scientific strings (``1e-3``, ``2.5E+4``); else None."""
    if _is_missing(v) or isinstance(v, bool):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if np.isnan(f) else f
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _series_all_numeric(series: pd.Series) -> bool:
    """True when every non-missing value parses as a number (incl. sci notation)."""
    if pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return True
    present = [v for v in series if not _is_missing(v)]
    return bool(present) and all(_as_float(v) is not None for v in present)


def _sorted_categories(series: pd.Series) -> list[str]:
    """Unique labels; numeric / sci-notation values sort by magnitude, else as strings."""
    items: list[tuple[str, float | None]] = []
    seen: set[str] = set()
    for v in series:
        if _is_missing(v):
            continue
        label = str(v)
        if label in seen:
            continue
        seen.add(label)
        items.append((label, _as_float(v)))
    if not items:
        return []
    if all(num is not None for _, num in items):
        items.sort(key=lambda t: t[1])  # type: ignore[arg-type, return-value]
        return [lab for lab, _ in items]
    return sorted(lab for lab, _ in items)


def _has_missing(series: pd.Series) -> bool:
    return bool(series.isna().any())


def _color_map(series: pd.Series) -> dict[str, str]:
    cats = _sorted_categories(series)
    return {c: _QUALITATIVE[i % len(_QUALITATIVE)] for i, c in enumerate(cats)}


def _shape_map(series: pd.Series) -> dict[str, str]:
    """Map non-null categories to filled symbols; circle is reserved for None."""
    cats = _sorted_categories(series)
    cycle = _SYMBOL_CYCLE[1:]  # skip circle
    return {c: cycle[i % len(cycle)] for i, c in enumerate(cats)}


def _shape_map_3d(series: pd.Series) -> dict[str, str]:
    """Like ``_shape_map`` but only symbols Scatter3d accepts."""
    cats = _sorted_categories(series)
    cycle = _SYMBOL_CYCLE_3D[1:]  # skip circle
    return {c: cycle[i % len(cycle)] for i, c in enumerate(cats)}


def _coerce_symbol_3d(symbol: str | None) -> str:
    if not symbol:
        return _DEFAULT_SYMBOL
    if symbol in _SYMBOL_3D_ALLOWED:
        return symbol
    return _SYMBOL_TO_3D.get(symbol, _DEFAULT_SYMBOL)


def _size_map(series: pd.Series, default: float = 8.0) -> dict[str, float]:
    cats = _sorted_categories(series)
    if not cats:
        return {}
    if _series_all_numeric(series):
        nums = [_as_float(c) for c in cats]
        assert all(n is not None for n in nums)
        vmin = min(nums)  # type: ignore[type-var]
        vmax = max(nums)  # type: ignore[type-var]
        if vmax > vmin:
            return {
                c: 4.0 + 16.0 * (float(n) - float(vmin)) / (float(vmax) - float(vmin))
                for c, n in zip(cats, nums)
            }
        return {c: float(default) for c in cats}
    steps = [float(s) for s in SIZE_CONSTANTS]
    if len(cats) == 1:
        return {cats[0]: steps[len(steps) // 2]}
    return {
        c: steps[min(int(round(i * (len(steps) - 1) / (len(cats) - 1))), len(steps) - 1)]
        for i, c in enumerate(cats)
    }


def _point_colors(series: pd.Series) -> list[str]:
    cmap = _color_map(series)
    return [
        _MISSING_COLOR if _is_missing(v) else cmap.get(str(v), _QUALITATIVE[0])
        for v in series
    ]


def _point_symbols(series: pd.Series) -> list[str]:
    smap = _shape_map(series)
    fallback = _SYMBOL_CYCLE[1]
    return [
        _MISSING_SYMBOL if _is_missing(v) else smap.get(str(v), fallback)
        for v in series
    ]


def _marker_sizes(series: pd.Series, default: float = 8.0) -> list[float]:
    fill = float(default)
    if _series_all_numeric(series):
        vals = pd.Series(
            [_as_float(v) if not _is_missing(v) else float("nan") for v in series],
            dtype=float,
        )
        vmin = float(vals.min()) if vals.notna().any() else fill
        vmax = float(vals.max()) if vals.notna().any() else fill
        if vmax > vmin:
            scaled = 4.0 + 16.0 * (vals - vmin) / (vmax - vmin)
            return [fill if _is_missing(v) else float(s) for v, s in zip(series, scaled)]
        return [fill] * len(series)
    mapping = _size_map(series, default=default)
    return [
        float(default) if _is_missing(v) else mapping.get(str(v), default)
        for v in series
    ]


def _legend_dummy(
    fig: go.Figure,
    *,
    name: str,
    color: str,
    symbol: str,
    size: float,
    group: str,
    group_title: str | None,
    first_in_group: bool,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=name,
            legendgroup=group,
            legendgrouptitle_text=group_title if first_in_group else None,
            marker=dict(color=color, symbol=symbol, size=size),
            showlegend=True,
        )
    )


def build_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str | None = None,
    color_col: str | None = None,
    shape_col: str | None = None,
    size_col: str | None = None,
    color_const: str | None = None,
    shape_const: str | None = None,
    size_const: float | None = None,
    alpha: float = 0.85,
    hover_name: str | None = None,
    title: str = "",
    legend_name: str | None = None,
) -> go.Figure:
    """Build 2D/3D scatter; 2D legends list color, shape, and size separately."""
    plot_df, hover_name, _hover_cols = prepare_hover_frame(df, id_col=hover_name)
    columns = list(plot_df.columns)

    use_color_col = color_col if color_col in columns else None
    use_shape_col = shape_col if shape_col in columns else None
    use_size_col = size_col if size_col in columns else None

    if z and z in plot_df.columns:
        plot3 = plot_df.copy()
        color_discrete_map = None
        symbol_map = None
        category_orders: dict = {}

        if use_color_col:
            raw = plot_df[use_color_col]
            color_discrete_map = {
                _MISSING_LABEL: _MISSING_COLOR,
                **_color_map(raw),
            }
            cats = [_MISSING_LABEL] if _has_missing(raw) else []
            cats += _sorted_categories(raw)
            plot3[use_color_col] = [
                _MISSING_LABEL if _is_missing(v) else str(v) for v in raw
            ]
            category_orders[use_color_col] = cats

        if use_shape_col:
            raw = plot_df[use_shape_col]
            symbol_map = {
                _MISSING_LABEL: _MISSING_SYMBOL,
                **_shape_map_3d(raw),
            }
            cats = [_MISSING_LABEL] if _has_missing(raw) else []
            cats += _sorted_categories(raw)
            plot3[use_shape_col] = [
                _MISSING_LABEL if _is_missing(v) else str(v) for v in raw
            ]
            category_orders[use_shape_col] = cats

        if use_size_col and not pd.api.types.is_numeric_dtype(plot_df[use_size_col]):
            raw = plot_df[use_size_col]
            plot3[use_size_col] = [
                _MISSING_LABEL if _is_missing(v) else str(v) for v in raw
            ]

        common = dict(
            data_frame=plot3,
            x=x,
            y=y,
            z=z,
            color=use_color_col,
            symbol=use_shape_col,
            size=use_size_col,
            opacity=alpha,
            hover_name=hover_name,
            hover_data=[],
            custom_data=["_click_id"],
            title=title,
        )
        if color_discrete_map is not None:
            common["color_discrete_map"] = color_discrete_map
        if symbol_map is not None:
            common["symbol_map"] = symbol_map
        if category_orders:
            common["category_orders"] = category_orders
        fig = px.scatter_3d(**common)
        fig.update_traces(hovertemplate="%{hovertext}<extra></extra>")
        marker_updates: dict = {}
        if use_color_col is None and color_const:
            marker_updates["color"] = color_const
        elif use_color_col is None and not color_const:
            marker_updates["color"] = _DEFAULT_COLOR
        if use_shape_col is None and shape_const:
            marker_updates["symbol"] = _coerce_symbol_3d(shape_const)
        elif use_shape_col is None and not shape_const:
            marker_updates["symbol"] = _DEFAULT_SYMBOL
        if use_size_col is None and size_const is not None:
            marker_updates["size"] = float(size_const)
        elif use_size_col is None and size_const is None:
            marker_updates["size"] = _DEFAULT_SIZE
        if marker_updates:
            fig.update_traces(marker=marker_updates)
        apply_export_layout(fig, title_lines=1, legend=True)
        return fig

    # --- 2D: one data trace + separate legend entries per aesthetic (seaborn-style) ---
    if use_color_col:
        point_colors = _point_colors(plot_df[use_color_col])
    elif color_const:
        point_colors = color_const
    else:
        point_colors = _DEFAULT_COLOR

    if use_shape_col:
        point_symbols = _point_symbols(plot_df[use_shape_col])
    elif shape_const:
        point_symbols = shape_const
    else:
        point_symbols = _DEFAULT_SYMBOL

    if use_size_col:
        point_sizes = _marker_sizes(plot_df[use_size_col], default=_DEFAULT_SIZE)
    elif size_const is not None:
        point_sizes = float(size_const)
    else:
        point_sizes = _DEFAULT_SIZE

    hover_text = plot_df[hover_name].astype(str) if hover_name in plot_df.columns else plot_df["_click_id"].astype(str)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df[x],
            y=plot_df[y],
            mode="markers",
            name=legend_name or "samples",
            showlegend=False,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=plot_df["_click_id"],
            marker=dict(
                color=point_colors,
                symbol=point_symbols,
                size=point_sizes,
                opacity=alpha,
                line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
            ),
        )
    )

    if use_color_col:
        cmap = _color_map(plot_df[use_color_col])
        first = True
        if _has_missing(plot_df[use_color_col]):
            _legend_dummy(
                fig,
                name=_MISSING_LABEL,
                color=_MISSING_COLOR,
                symbol="circle",
                size=_LEGEND_STD_SIZE,
                group="legend-color",
                group_title=use_color_col,
                first_in_group=True,
            )
            first = False
        for i, cat in enumerate(_sorted_categories(plot_df[use_color_col])):
            _legend_dummy(
                fig,
                name=cat,
                color=cmap[cat],
                symbol="circle",
                size=_LEGEND_STD_SIZE,
                group="legend-color",
                group_title=use_color_col,
                first_in_group=first and i == 0,
            )
    elif color_const:
        _legend_dummy(
            fig,
            name=color_const,
            color=color_const,
            symbol="circle",
            size=_LEGEND_STD_SIZE,
            group="legend-color",
            group_title="Color",
            first_in_group=True,
        )

    if use_shape_col:
        smap = _shape_map(plot_df[use_shape_col])
        first = True
        if _has_missing(plot_df[use_shape_col]):
            _legend_dummy(
                fig,
                name=_MISSING_LABEL,
                color=_LEGEND_GREY,
                symbol=_MISSING_SYMBOL,
                size=_LEGEND_STD_SIZE,
                group="legend-shape",
                group_title=use_shape_col,
                first_in_group=True,
            )
            first = False
        for i, cat in enumerate(_sorted_categories(plot_df[use_shape_col])):
            _legend_dummy(
                fig,
                name=cat,
                color=_LEGEND_GREY,
                symbol=smap[cat],
                size=_LEGEND_STD_SIZE,
                group="legend-shape",
                group_title=use_shape_col,
                first_in_group=first and i == 0,
            )
    elif shape_const:
        _legend_dummy(
            fig,
            name=shape_const,
            color=_LEGEND_GREY,
            symbol=shape_const,
            size=_LEGEND_STD_SIZE,
            group="legend-shape",
            group_title="Shape",
            first_in_group=True,
        )

    if use_size_col:
        szmap = _size_map(plot_df[use_size_col])
        first = True
        if _has_missing(plot_df[use_size_col]) and not pd.api.types.is_numeric_dtype(
            plot_df[use_size_col]
        ):
            _legend_dummy(
                fig,
                name=_MISSING_LABEL,
                color=_LEGEND_GREY,
                symbol="circle",
                size=_DEFAULT_SIZE,
                group="legend-size",
                group_title=use_size_col,
                first_in_group=True,
            )
            first = False
        for i, cat in enumerate(_sorted_categories(plot_df[use_size_col])):
            _legend_dummy(
                fig,
                name=cat,
                color=_LEGEND_GREY,
                symbol="circle",
                size=szmap[cat],
                group="legend-size",
                group_title=use_size_col,
                first_in_group=first and i == 0,
            )
    elif size_const is not None:
        _legend_dummy(
            fig,
            name=str(size_const),
            color=_LEGEND_GREY,
            symbol="circle",
            size=float(size_const),
            group="legend-size",
            group_title="Size",
            first_in_group=True,
        )

    if not (use_color_col or color_const or use_shape_col or shape_const or use_size_col or size_const is not None):
        _legend_dummy(
            fig,
            name=legend_name or "samples",
            color=_DEFAULT_COLOR,
            symbol=_DEFAULT_SYMBOL,
            size=_LEGEND_STD_SIZE,
            group="legend-samples",
            group_title=None,
            first_in_group=True,
        )

    fig.update_layout(title=title)
    apply_export_layout(fig, title_lines=1, legend=True)
    return fig


def size_array(meta: pd.DataFrame, size_col: str | None, default: float = 8.0) -> np.ndarray | float:
    if size_col and size_col in meta.columns:
        return np.asarray(_marker_sizes(meta[size_col], default=default), dtype=float)
    return default
