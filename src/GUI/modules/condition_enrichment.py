"""Metadata ranks, PCA-axis enrichment, Fisher / MWU, condition bins."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score

from src.GUI.components.controls import apply_export_layout

GROWTHPHASE_RANK = {
    "No growth": 0,
    "Early Exponential": 1,
    "Mid Exponential": 2,
    "Late Exponential": 3,
    "Early Stationary": 4,
    "Mid Stationary": 5,
    "Late Stationary": 6,
}

DEFAULT_NUM_COLS = [
    "GrowthPhase",
    "Concentration",
    "Temperature",
    "OD_init",
    "OD_final",
    "Carbon_per_OD_init",
    "Carbon_left",
    "byproduct",
]
DEFAULT_CAT_COLS = [
    "Condition",
    "Carbon",
    "Treatment",
    "GrowthPhase",
    "Concentration",
    "Temperature",
    "Experiment",
]
EXCLUDE_ENTRIES = {"Biofilm", "region1", "region2", "region3", "region4"}

SKIP_COLS = {
    "fileName",
    "FileName",
    "sampleNr",
    "indexWell",
    "idx1",
    "idx2",
    "Well",
    "barcodeName",
    "barcode read in fastq (5′->3′)",
    "ConditionNr",
    "kdlibNr",
    "BiologicalConditionNr",
    "sample_id",
    "sampleID",
}
NUM_FRAC_THR = 0.9


def present_cols(available: Iterable[str], preferred: list[str]) -> list[str]:
    have = set(available)
    return [c for c in preferred if c in have]


def to_numeric_rank(series: pd.Series, col: str) -> pd.Series:
    if col == "GrowthPhase":
        return series.astype(str).map(GROWTHPHASE_RANK)
    if col == "Concentration":
        return pd.to_numeric(series.astype(str).str.extract(r"([0-9.]+)")[0], errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def binary_mask(series: pd.Series, positive) -> np.ndarray:
    return (series.astype(str) == str(positive)).to_numpy()


def pc_separation_ari(
    score_df: pd.DataFrame,
    label_col: str,
    positive,
    n_pcs: int,
) -> pd.DataFrame:
    """Per-PC one-sided threshold ARI."""
    is_pos = binary_mask(score_df[label_col], positive)
    if is_pos.sum() == 0 or (~is_pos).sum() == 0:
        raise ValueError(f"Need both {positive!r} and other samples in {label_col}.")
    rows = []
    for i in range(1, int(n_pcs) + 1):
        col = f"PC{i}"
        if col not in score_df.columns:
            break
        x = score_df[col].to_numpy(dtype=float)
        if np.median(x[is_pos]) >= np.median(x[~is_pos]):
            pred = np.where(x >= x[is_pos].min(), "pos-end", "other")
        else:
            pred = np.where(x <= x[is_pos].max(), "pos-end", "other")
        rows.append({"PC": i, "ARI": float(adjusted_rand_score(is_pos, pred))})
    return pd.DataFrame(rows)


def ari_figure(sep_df: pd.DataFrame, title: str | dict) -> go.Figure:
    fig = go.Figure()
    if sep_df is None or sep_df.empty:
        fig.add_annotation(text="No ARI results", showarrow=False)
        return fig
    fig.add_trace(
        go.Scatter(
            x=sep_df["PC"],
            y=sep_df["ARI"],
            mode="lines+markers",
            name="ARI",
        )
    )
    ymin = float(sep_df["ARI"].min())
    fig.update_layout(
        title=title,
        xaxis_title="PC",
        yaxis_title="ARI",
        yaxis=dict(range=[ymin, 1.0] if np.isfinite(ymin) else None),
        xaxis=dict(
            tickmode="array",
            tickvals=sep_df["PC"].tolist(),
            ticktext=[f"PC{i}" for i in sep_df["PC"]],
        ),
        showlegend=False,
    )
    apply_export_layout(fig, title_lines=2, legend=False, uirevision="pca-ari")
    return fig


def biofilm_axis_scores(
    score_df: pd.DataFrame,
    label_col: str,
    n_pcs_axis: int,
    axis_name: str = "Biofilm axis",
) -> tuple[pd.Series, np.ndarray]:
    """OLS of label (0/1) on the first n PCs; L2-normalized coefficients; project scores."""
    pc_cols = [f"PC{i}" for i in range(1, int(n_pcs_axis) + 1)]
    missing = [c for c in pc_cols if c not in score_df.columns]
    if missing:
        raise ValueError(f"Missing PCs for axis: {missing}")
    X = score_df[pc_cols].to_numpy(dtype=float)
    y = pd.to_numeric(score_df[label_col], errors="coerce").to_numpy()
    if np.isnan(y).all():
        y = (score_df[label_col].astype(str).str.lower().isin(["true", "1", "yes"])).astype(float).to_numpy()
    w = LinearRegression().fit(X, y).coef_
    nrm = float(np.linalg.norm(w))
    if nrm <= 0:
        raise ValueError("Axis coefficients have zero norm.")
    w = w / nrm
    return pd.Series(X @ w, index=score_df.index, name=axis_name), w


def corr_vs_axes(df: pd.DataFrame, meta_cols: list[str], axes: list[str]) -> dict[str, pd.DataFrame]:
    out = {
        "Pearson": pd.DataFrame(index=meta_cols, columns=axes, dtype=float),
        "Spearman": pd.DataFrame(index=meta_cols, columns=axes, dtype=float),
    }
    for m in meta_cols:
        if m not in df.columns:
            continue
        x = to_numeric_rank(df[m], m)
        for ax_name in axes:
            y = df[ax_name]
            mask = x.notna() & y.notna()
            if mask.sum() < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                out["Pearson"].loc[m, ax_name] = np.nan
                out["Spearman"].loc[m, ax_name] = np.nan
                continue
            xa, ya = x[mask].to_numpy(float), y[mask].to_numpy(float)
            out["Pearson"].loc[m, ax_name] = float(pearsonr(xa, ya)[0])
            out["Spearman"].loc[m, ax_name] = float(spearmanr(xa, ya)[0])
    return out


def cat_entry_mats(
    df: pd.DataFrame,
    cat_cols: list[str],
    axes: list[str],
    label_col: str | None = "Biofilm",
    exclude: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exclude = EXCLUDE_ENTRIES if exclude is None else set(exclude)
    row_labels: list[tuple[str, str]] = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        s = df[col].fillna("none")
        for entry in sorted(s.unique(), key=str):
            if str(entry) in exclude:
                continue
            row_labels.append((col, str(entry)))
    labels = [f"{c} | {e}" for c, e in row_labels]
    var_mat = pd.DataFrame(index=labels, columns=axes, dtype=float)
    dist_mat = pd.DataFrame(index=labels, columns=axes, dtype=float)
    is_pos = None
    if label_col and label_col in df.columns:
        is_pos = df[label_col].astype(str).str.lower().isin(["true", "1", "yes"])
    for ax_name in axes:
        y = df[ax_name]
        total_var = float(y.var())
        bio_mean = float(y[is_pos].mean()) if is_pos is not None and is_pos.any() else float(y.mean())
        for col, entry in row_labels:
            s = df[col].fillna("none")
            yi = y[s.astype(str).eq(entry)].dropna()
            lab = f"{col} | {entry}"
            if len(yi) == 0:
                continue
            v = float(yi.var()) if len(yi) > 1 else 0.0
            var_mat.loc[lab, ax_name] = (v / total_var) if total_var > 0 else np.nan
            d = abs(float(yi.mean()) - bio_mean)
            dist_mat.loc[lab, ax_name] = (d / np.sqrt(v)) if v > 0 else np.nan
    return var_mat, dist_mat


def heatmap_matrix_fig(
    mats: list[pd.DataFrame],
    titles: list[str],
    *,
    title: str | dict,
    zmin=None,
    zmax=None,
    text_fmt: str = ".2f",
    colorscale: str = "Viridis",
) -> go.Figure:
    mats = [m for m in mats if m is not None and not m.empty]
    if not mats:
        fig = go.Figure()
        fig.add_annotation(text="No enrichment matrix", showarrow=False)
        return fig
    n = len(mats)
    n_rows = int(mats[0].shape[0])
    max_lab = max((len(str(r)) for r in mats[0].index), default=8)
    # Stack panels so each heatmap keeps its own y-labels (side-by-side
    # put the second plot's ticks on top of the first heatmap).
    vspace = 0.10 if n == 1 else min(0.14, 0.28 / n)
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=titles[:n],
        vertical_spacing=vspace,
    )
    text_size = 9 if n_rows > 14 else 11
    for i, mat in enumerate(mats, start=1):
        z = mat.to_numpy(dtype=float)
        text = [
            ["" if not np.isfinite(v) else format(float(v), text_fmt) for v in row]
            for row in z
        ]
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=[str(c) for c in mat.columns],
                y=[str(r) for r in mat.index],
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=text_size, color="white"),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    len=max(0.22, 0.72 / n),
                    y=1 - (i - 0.5) / n,
                    yanchor="middle",
                    thickness=14,
                    x=1.02,
                    xanchor="left",
                    outlinewidth=0,
                ),
                hovertemplate="%{y}<br>%{x}: %{z:.3g}<extra></extra>",
            ),
            row=i,
            col=1,
        )
    fig.update_yaxes(
        autorange="reversed",
        automargin=True,
        tickfont=dict(size=10),
        ticks="outside",
        ticklen=3,
    )
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10), automargin=True)
    row_h = 24 if n_rows <= 12 else 20
    height = 90 + n * (max(140, row_h * n_rows + 70) + 24)
    left = min(340, max(80, 7 * max_lab + 20))
    apply_export_layout(fig, title_lines=2, legend=False, height=height, bottom=50)
    fig.update_layout(
        title=title,
        margin=dict(l=left, r=90, t=70, b=50),
        height=int(height),
    )
    return fig


def assign_region_labels(
    index: pd.Index,
    by_level: dict[str, set],
    levels: list[str],
) -> pd.Series:
    """Label samples by closest/unique level; overlaps become ``a ∩ b``."""
    ser = pd.Series(pd.NA, index=index, dtype=object)
    lookup = {str(i): i for i in index}
    for r in levels:
        for sid in by_level.get(r) or ():
            key = lookup.get(str(sid))
            if key is None:
                continue
            cur = ser.loc[key]
            ser.loc[key] = r if pd.isna(cur) else f"{cur} ∩ {r}"
    return ser


def region_rank_series(series: pd.Series, levels: list[str]) -> pd.Series:
    rank_map = {str(lv): float(i + 1) for i, lv in enumerate(levels)}

    def one(x):
        if pd.isna(x):
            return np.nan
        parts = [p.strip() for p in str(x).split("∩")]
        vals = [rank_map[p] for p in parts if p in rank_map]
        return float(np.mean(vals)) if vals else np.nan

    return series.map(one)


def corr_vs_region_rank(
    df: pd.DataFrame,
    rank_cols: list[str],
    target_cols: list[str],
    levels: list[str],
) -> dict[str, pd.DataFrame]:
    """Pearson / Spearman of ranked metadata vs closest-region rank (notebook §8)."""
    out = {
        "Pearson": pd.DataFrame(index=rank_cols, columns=target_cols, dtype=float),
        "Spearman": pd.DataFrame(index=rank_cols, columns=target_cols, dtype=float),
    }
    y_by_t = {
        t: region_rank_series(df[t], levels) for t in target_cols if t in df.columns
    }
    for m in rank_cols:
        if m not in df.columns:
            continue
        x = to_numeric_rank(df[m], m)
        for t, y in y_by_t.items():
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
                continue
            xa = x[mask].to_numpy(dtype=float)
            ya = y[mask].to_numpy(dtype=float)
            out["Pearson"].loc[m, t] = float(pearsonr(xa, ya)[0])
            out["Spearman"].loc[m, t] = float(spearmanr(xa, ya)[0])
    return out


def entry_fractions_fig(
    df: pd.DataFrame,
    region_col: str,
    columns: list[str],
    *,
    title: str | dict,
    levels: list[str],
) -> go.Figure:
    """Stacked entry fractions within each closest-region group (notebook §7b)."""
    fig = go.Figure()
    if df is None or df.empty or region_col not in df.columns or not columns:
        fig.add_annotation(text="No categorical fractions to plot.", showarrow=False)
        return fig
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    color_maps = {}
    for col in columns:
        if col not in df.columns:
            continue
        entries = sorted(
            e for e in df[col].fillna("none").astype(str).unique() if e != "none"
        )
        color_maps[col] = {
            "none": "#9e9e9e",
            **{e: palette[i % len(palette)] for i, e in enumerate(entries)},
        }
    reg_levels = [
        r
        for r in levels
        if df[region_col].fillna("").astype(str).str.contains(str(r), regex=False).any()
    ]
    if not reg_levels:
        fig.add_annotation(text="No samples labeled by level.", showarrow=False)
        return fig
    col_step, region_gap, bar_width = 1.0, 2.2, 0.92
    bar_specs, region_spans = [], []
    x = 0.0
    use_cols = [c for c in columns if c in df.columns]
    for ri, reg in enumerate(reg_levels):
        if ri:
            x += region_gap
        x0 = x
        sub = df[df[region_col].fillna("").astype(str).str.contains(str(reg), regex=False)]
        for col in use_cols:
            s = sub[col].fillna("none").astype(str)
            counts = s.value_counts()
            fr = s.value_counts(normalize=True)
            order = (["none"] if "none" in fr.index else []) + sorted(
                i for i in fr.index if i != "none"
            )
            fr, counts = fr.reindex(order), counts.reindex(order)
            bar_specs.append(
                {"x": x, "col": col, "region": reg, "fr": fr, "counts": counts, "n": len(s)}
            )
            x += col_step
        region_spans.append((reg, x0, x - col_step))
    for spec in bar_specs:
        cmap = color_maps.get(spec["col"], {})
        bottom = 0.0
        for entry, frac in spec["fr"].items():
            frac = float(frac)
            n_hit = int(spec["counts"].get(entry, 0))
            fig.add_trace(
                go.Bar(
                    x=[spec["x"]],
                    y=[frac],
                    base=[bottom],
                    width=bar_width,
                    marker_color=cmap.get(entry, "#333333"),
                    marker_line=dict(width=0.4, color="white"),
                    name=str(entry),
                    legendgroup=str(entry),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{entry}</b><br>level: {spec['region']}<br>"
                        f"column: {spec['col']}<br>"
                        f"n: {n_hit} / {spec['n']}<br>fraction: {frac:.3f}<extra></extra>"
                    ),
                )
            )
            bottom += frac
    fig.update_layout(
        title=title if isinstance(title, dict) else {"text": title, "x": 0.5, "xanchor": "center"},
        barmode="stack",
        xaxis=dict(
            tickmode="array",
            tickvals=[s["x"] for s in bar_specs],
            ticktext=[s["col"] for s in bar_specs],
            tickangle=-55,
            side="top",
            showgrid=False,
            zeroline=False,
            range=[-0.6, region_spans[-1][2] + 0.6] if region_spans else None,
        ),
        yaxis=dict(title="fraction within level", range=[0, 1.05]),
        height=520,
        margin=dict(b=80, t=100, r=20, l=50),
        showlegend=False,
        annotations=[
            dict(
                x=0.5 * (a + b),
                y=-0.16,
                xref="x",
                yref="paper",
                text=f"<b>{reg}</b>",
                showarrow=False,
                yanchor="top",
            )
            for reg, a, b in region_spans
        ],
        uirevision="par-enr-frac",
        hovermode="closest",
    )
    return fig


def classify_meta_columns(meta: pd.DataFrame) -> dict:
    """≥90% numeric → threshold filters; otherwise tickable categorical entries."""
    info = {}
    n = len(meta)
    for c in meta.columns:
        if c in SKIP_COLS:
            continue
        s = meta[c]
        nuniq = int(s.nunique(dropna=True))
        if nuniq <= 1:
            continue
        num = pd.to_numeric(s, errors="coerce")
        n_nonnull = int(s.notna().sum())
        frac_num = float(num.notna().sum()) / max(n_nonnull, 1)
        if frac_num >= NUM_FRAC_THR:
            info[c] = {
                "kind": "num",
                "vmin": float(num.min()),
                "vmax": float(num.max()),
                "median": float(num.median()),
            }
            continue
        if nuniq > min(200, max(50, int(0.4 * n))):
            continue
        vals = sorted(
            s.fillna("NA").astype(str).unique(),
            key=lambda x: (x == "NA", x.lower()),
        )
        info[c] = {"kind": "cat", "entries": vals}
    return info


def _apply_op(series: pd.Series, op: str, thr) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    t = float(thr)
    if op == "<":
        return (num < t) & num.notna()
    if op == ">":
        return (num > t) & num.notna()
    raise ValueError(f"Unknown operator {op!r}")


def resolve_condition_bins(
    meta: pd.DataFrame,
    col_info: dict,
    filters_a: dict,
    mode_b: str | dict,
    filters_b: dict | None = None,
) -> tuple[pd.Index, pd.Index, str]:
    """AND active column filters. ``mode_b`` is ``b`` / ``subset`` or ``rest`` / ``~treatment``."""
    filters_b = filters_b or {}

    def _mode(col: str) -> str:
        if isinstance(mode_b, dict):
            raw = mode_b.get(col, "rest")
        else:
            raw = mode_b
        return "rest" if str(raw) in {"rest", "~treatment", "not_treatment", "~"} else "subset"

    active = []
    for c, fa in (filters_a or {}).items():
        if c not in col_info:
            continue
        if col_info[c]["kind"] == "cat":
            if fa:
                active.append(c)
        elif fa and fa.get("enabled"):
            active.append(c)
    if not active:
        raise ValueError("Activate at least one column in group A / treatment.")

    mask_a = pd.Series(True, index=meta.index)
    mask_b = pd.Series(True, index=meta.index)
    title_parts = []
    for c in active:
        kind = col_info[c]["kind"]
        mb = _mode(c)
        if kind == "cat":
            a_vals = set(filters_a[c])
            all_vals = set(col_info[c]["entries"])
            remaining = sorted(all_vals - a_vals, key=lambda x: (x == "NA", x.lower()))
            if mb == "rest":
                b_vals = set(remaining)
                b_label = "~treatment"
            else:
                b_vals = set(filters_b.get(c) or [])
                if not b_vals:
                    raise ValueError(f"{c}: mode B needs selected entries.")
                if b_vals - set(remaining):
                    raise ValueError(f"{c}: B may only use entries not in A.")
                b_label = ",".join(sorted(b_vals))
            col_s = meta[c].fillna("NA").astype(str)
            mask_a &= col_s.isin(a_vals)
            mask_b &= col_s.isin(b_vals)
            title_parts.append(f"{c}:{{{','.join(sorted(a_vals))}}} vs {{{b_label}}}")
        else:
            fa = filters_a[c]
            a_op, a_thr = fa["op"], fa["value"]
            mask_a &= _apply_op(meta[c], a_op, a_thr)
            if mb == "rest":
                num = pd.to_numeric(meta[c], errors="coerce")
                mask_b &= num.notna() & ~_apply_op(meta[c], a_op, a_thr)
                b_label = "~treatment"
            else:
                fb = filters_b.get(c) or {}
                if not fb.get("enabled"):
                    raise ValueError(f"{c}: B needs an operator and value (or use ~treatment).")
                mask_b &= _apply_op(meta[c], fb["op"], fb["value"])
                b_label = f"{fb['op']}{fb['value']}"
            title_parts.append(f"{c}:{a_op}{a_thr} vs {{{b_label}}}")

    ids_a = meta.index[mask_a]
    ids_b = meta.index[mask_b]
    overlap = ids_a.intersection(ids_b)
    if len(overlap):
        ids_a = ids_a.difference(overlap)
        ids_b = ids_b.difference(overlap)
    return ids_a, ids_b, " | ".join(title_parts)


def table_html(df: pd.DataFrame, *, max_rows: int = 40):
    from dash import html
    import dash_bootstrap_components as dbc

    if df is None or df.empty:
        return html.P("No rows.", className="text-muted small mb-0")
    show = df.head(max_rows)
    header = [html.Th(str(c), className="text-nowrap") for c in show.columns]
    body = []
    for _, row in show.iterrows():
        cells = []
        for c in show.columns:
            val = row[c]
            if isinstance(val, float) and np.isfinite(val):
                text = f"{val:.4g}"
            else:
                text = "" if val is None or (isinstance(val, float) and pd.isna(val)) else str(val)
            cells.append(html.Td(text, className="text-nowrap"))
        body.append(html.Tr(cells))
    return html.Div(
        dbc.Table(
            [html.Thead(html.Tr(header)), html.Tbody(body)],
            bordered=True,
            striped=True,
            size="sm",
            className="mb-0",
        ),
        className="overflow-auto",
        style={"maxHeight": "420px", "overflowY": "auto", "overflowX": "auto"},
    )
