"""Volcano by metadata condition — volcano_by_condition.ipynb."""

from __future__ import annotations

from pathlib import Path

import ast
import re

from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from src.biocyc.celov_multiomics_post import load_locus_lookup
from src.GUI.project import resolve_celov_id_col

from ..components.controls import (
    EXPORT_H,
    EXPORT_W,
    apply_export_layout,
    equal_xy_axes,
    fig_size_controls,
    set_fig_size,
)
from ..components.folder_browser import pick_save_file_dialog
from ..components.gene_meta_mark import (
    entry_dropdown_options,
    gene_meta_hover_map,
    genes_with_meta_entry,
    locus_mark_columns,
    mark_controls,
    mark_legend_banner,
)
from ..components.sample_detail import (
    gene_detail_placeholder,
    gene_detail_table,
    plot_with_sample_detail,
    register_sample_detail_callback,
)
from ..data_store import session_from_store
from .clustering import (
    _active_dataset_entry,
    _locus_path_from_session,
    _volcano_weighed_for_celov,
    volcano_cluster_vs_cluster,
)
from .hc_plots import volcano_fig_from_results
from .pca_classifier import save_classifier_celov

_GRAPH_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "volcano_condition"},
    "displaylogo": False,
}
_BINS_PCA_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "volcano_condition_bins_pca"},
    "displaylogo": False,
}
_GENE_PCA_CONFIG = {
    "toImageButtonOptions": {"format": "svg", "filename": "volcano_condition_gene_pca"},
    "displaylogo": False,
}
_VC_RUNTIME: dict = {}


def _plotly_title(*lines: str) -> dict:
    text = "<br>".join(line for line in lines if line is not None and str(line).strip() != "")
    return {"text": text, "x": 0.5, "xanchor": "center"}


_BIN_ORDER = ["Bin A", "Bin B", "overlap", "rest"]
_BIN_COLORS = {
    "Bin A": "#1f77b4",
    "Bin B": "#d62728",
    "overlap": "#9467bd",
    "rest": "#bdbdbd",
}


def volcano_passing_gene_ids() -> list[str]:
    """Gene IDs that pass the last volcano fold-change / padj thresholds."""
    results = _VC_RUNTIME.get("results")
    meta = _VC_RUNTIME.get("plot_meta") or {}
    if results is None or "geneID" not in results.columns:
        return []
    padj = float(meta.get("padj") or 2.0)
    fc = float(meta.get("fc") or 0.5)
    hit = (results["abs_fold_change"] > fc) & (results["neg_log10_padj"] > padj)
    return list(results.loc[hit, "geneID"].astype(str))


_COMPLEMENT = {"~", "~a", "~A", "~bin a", "~Bin A", "~binA", "not A", "not a"}
_SAFE_TYPES = {"float": float, "int": int, "str": str, "bool": bool}
_RESERVED = set(_SAFE_TYPES) | {
    "True",
    "False",
    "None",
    "in",
    "not",
    "and",
    "or",
    "isin",
    "astype",
    "pd_to_numeric",
}


def _col_alias(name: str) -> str:
    alias = re.sub(r"[^0-9A-Za-z_]", "_", str(name)).strip("_")
    if alias and alias[0].isdigit():
        alias = f"c_{alias}"
    return alias


_MAX_ENTRY_OPTS = 400


def _expr_col_name(col: str) -> str:
    """Quoted column name for filter expressions (``'column'``)."""
    return repr(str(col))


def _entry_clause(col: str, values: list[str]) -> str:
    name = _expr_col_name(col)
    lits = [repr(str(v)) for v in values]
    if len(lits) == 1:
        return f"{name} == {lits[0]}"
    return f"{name} in [{', '.join(lits)}]"


def _append_clause(current: str | None, clause: str) -> str:
    cur = (current or "").strip()
    if not cur or _is_complement(cur):
        return clause
    return f"({cur}) | ({clause})"


def _unique_entry_options(series: pd.Series, col: str) -> tuple[list[dict], str]:
    vals = series.fillna("NA").astype(str)
    uniq = sorted(vals.unique(), key=lambda x: (x == "NA", x.lower()))
    n = len(uniq)
    num = pd.to_numeric(series, errors="coerce")
    n_ok = int(series.notna().sum())
    frac_num = float(num.notna().sum()) / max(n_ok, 1)
    hint_bits = [f"{n} unique value{'s' if n != 1 else ''}"]
    if frac_num >= 0.9 and n_ok:
        hint_bits.append(
            f"numeric range {float(num.min()):g} … {float(num.max()):g}; "
            f"use {_expr_col_name(col)} <= …"
        )
    if n > _MAX_ENTRY_OPTS:
        hint_bits.append(f"listing first {_MAX_ENTRY_OPTS}")
        uniq = uniq[:_MAX_ENTRY_OPTS]
    opts = [{"label": v, "value": v} for v in uniq]
    return opts, "; ".join(hint_bits)


def _meta_locals(meta: pd.DataFrame) -> dict:
    local: dict = dict(_SAFE_TYPES)
    local["pd_to_numeric"] = lambda s: pd.to_numeric(s, errors="coerce")
    for col in meta.columns:
        series = meta[col]
        key = str(col)
        if key.isidentifier() and key not in local:
            local[key] = series
        alias = _col_alias(key)
        if alias and alias not in local:
            local[alias] = series
    return local


def _rewrite_quoted_columns(expr: str, meta: pd.DataFrame) -> str:
    """Map ``'col'`` / ``\"col\"`` column refs to safe Series aliases."""
    name_to_alias = {str(c): _col_alias(str(c)) or str(c) for c in meta.columns}
    # Prefer real identifiers when possible
    for col in meta.columns:
        key = str(col)
        if key.isidentifier():
            name_to_alias[key] = key

    def repl(m: re.Match) -> str:
        name = m.group(2)
        if name in name_to_alias:
            return name_to_alias[name]
        return m.group(0)

    return re.sub(
        r"(['\"])([^'\"]+)\1(?=\s*(?:not\s+in\b|in\b|==|!=|<=|>=|<|>|\.))",
        repl,
        expr,
    )


def _rewrite_list_membership(expr: str) -> str:
    """``col in [...]`` / ``col not in [...]`` → ``col.isin([...])``."""
    expr = re.sub(
        r"\b([A-Za-z_][\w]*)\s+not\s+in\s+(\[[^\[\]]*\])",
        r"~\1.isin(\2)",
        expr,
    )
    expr = re.sub(
        r"\b([A-Za-z_][\w]*)\s+in\s+(\[[^\[\]]*\])",
        r"\1.isin(\2)",
        expr,
    )
    return expr


def _rewrite_numeric_cmp(expr: str) -> str:
    """``col <= 8`` → ``pd_to_numeric(col) <= 8`` for numeric literals."""
    return re.sub(
        r"\b([A-Za-z_][\w]*)\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b",
        r"pd_to_numeric(\1) \2 \3",
        expr,
    )


def _replace_logical_words(expr: str) -> str:
    """Turn ``and`` / ``or`` / ``not`` into ``&`` / ``|`` / ``~`` outside quotes."""
    out: list[str] = []
    i = 0
    quote = None
    while i < len(expr):
        ch = expr[i]
        if quote:
            out.append(ch)
            if ch == quote and (i == 0 or expr[i - 1] != "\\"):
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            out.append(ch)
            i += 1
            continue
        rest = expr[i:]
        start = i == 0 or not (expr[i - 1].isalnum() or expr[i - 1] == "_")
        if start:
            m = re.match(r"and\b", rest)
            if m:
                out.append("&")
                i += m.end()
                continue
            m = re.match(r"or\b", rest)
            if m:
                out.append("|")
                i += m.end()
                continue
            m = re.match(r"not\b", rest)
            if m:
                out.append("~")
                i += m.end()
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _inject_bare_strings(expr: str, local: dict) -> dict:
    extra = dict(local)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return extra
    known = set(extra) | _RESERVED
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in known:
            extra[node.id] = node.id
    return extra


def _split_top(expr: str, seps: list[str]) -> list[str]:
    """Split ``expr`` on ``seps`` outside quotes, parentheses, and brackets."""
    seps = sorted(seps, key=len, reverse=True)
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    depth_paren = 0
    depth_brack = 0
    quote = None
    while i < len(expr):
        ch = expr[i]
        if quote:
            buf.append(ch)
            if ch == quote and expr[i - 1] != "\\":
                quote = None
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack -= 1
        if depth_paren == 0 and depth_brack == 0:
            matched = None
            for sep in seps:
                if not expr.startswith(sep, i):
                    continue
                if sep.isalpha():
                    left_ok = i == 0 or not (expr[i - 1].isalnum() or expr[i - 1] == "_")
                    right = i + len(sep)
                    right_ok = right >= len(expr) or not (
                        expr[right].isalnum() or expr[right] == "_"
                    )
                    if not (left_ok and right_ok):
                        continue
                matched = sep
                break
            if matched:
                parts.append("".join(buf).strip())
                buf = []
                i += len(matched)
                continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf).strip())
    return [part for part in parts if part]


def _unwrap_parens(expr: str) -> str:
    text = expr.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        quote = None
        wraps = True
        for i, ch in enumerate(text):
            if quote:
                if ch == quote and (i == 0 or text[i - 1] != "\\"):
                    quote = None
                continue
            if ch in {"'", '"'}:
                quote = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(text) - 1:
                    wraps = False
                    break
        if wraps and depth == 0:
            text = text[1:-1].strip()
        else:
            break
    return text


def _eval_bool_expr(expr: str, meta: pd.DataFrame, local: dict) -> pd.Series:
    text = _unwrap_parens(expr)
    if not text:
        raise ValueError("Empty filter expression.")
    parts = _split_top(text, ["|", "or"])
    if len(parts) > 1:
        mask = _eval_bool_expr(parts[0], meta, local)
        for part in parts[1:]:
            mask = mask | _eval_bool_expr(part, meta, local)
        return mask
    parts = _split_top(text, ["&", "and"])
    if len(parts) > 1:
        mask = _eval_bool_expr(parts[0], meta, local)
        for part in parts[1:]:
            mask = mask & _eval_bool_expr(part, meta, local)
        return mask
    if text.startswith("~"):
        return ~_eval_bool_expr(text[1:], meta, local)
    if re.match(r"not\b", text):
        return ~_eval_bool_expr(text[3:], meta, local)
    atomic = _rewrite_list_membership(text)
    atomic = _rewrite_numeric_cmp(atomic)
    _require_columns(atomic, local)
    ns = _inject_bare_strings(atomic, local)
    try:
        mask = eval(atomic, {"__builtins__": {}}, ns)
    except Exception as err:
        raise ValueError(f"Could not evaluate {expr.strip()!r}: {err}") from err
    if isinstance(mask, pd.Series):
        out = mask.reindex(meta.index)
    else:
        out = pd.Series(mask, index=meta.index)
    return out.fillna(False).astype(bool)


def _require_columns(expr: str, local: dict) -> None:
    """Error if a comparison / method is applied to something that is not a column."""
    cols = {k for k, v in local.items() if isinstance(v, pd.Series)}
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as err:
        raise ValueError(f"Syntax error: {err.msg or err}") from err
    bad: list[str] = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = node.value.id
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Name):
            name = node.left.id
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "pd_to_numeric"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            name = node.args[0].id
        if name and name not in cols and name not in _RESERVED:
            bad.append(name)
    if bad:
        shown = ", ".join(repr(x) for x in dict.fromkeys(bad))
        raise ValueError(
            f"Unknown column{'s' if len(dict.fromkeys(bad)) > 1 else ''} {shown}. "
            "Use a metadata column listed above."
        )


def eval_meta_mask(meta: pd.DataFrame, expr: str) -> pd.Series:
    """Boolean Series over ``meta.index`` from a Python-like filter."""
    raw = (expr or "").strip()
    if not raw:
        raise ValueError("Empty filter expression.")
    if "__" in raw:
        raise ValueError("Invalid filter expression.")
    rewritten = _rewrite_quoted_columns(raw, meta)
    return _eval_bool_expr(rewritten, meta, _meta_locals(meta))


def _is_complement(expr: str | None) -> bool:
    s = (expr or "").strip()
    return (not s) or s in _COMPLEMENT or s.replace(" ", "") in {"~A", "~BinA"}


def vc_sample_groups(
    meta: pd.DataFrame,
    expr_a: str,
    expr_b: str | None,
) -> tuple[pd.Series, str]:
    """Label every sample as Bin A / Bin B / overlap / rest (overlap kept)."""
    mask_a = eval_meta_mask(meta, expr_a)
    a_lab = expr_a.strip()
    n_a = int(mask_a.sum())
    if n_a == 0:
        raise ValueError(
            "Bin A matched 0 samples — this filter does not match any metadata rows."
        )
    if _is_complement(expr_b):
        mask_b = ~mask_a
        b_lab = "~Bin A"
    else:
        mask_b = eval_meta_mask(meta, expr_b)
        b_lab = expr_b.strip()
    n_b = int(mask_b.sum())
    if n_b == 0:
        raise ValueError(
            "Bin B matched 0 samples — this filter does not match any metadata rows."
        )
    both = mask_a & mask_b
    if mask_a.equals(mask_b):
        raise ValueError("Bin A and Bin B select the same samples.")
    groups = pd.Series("rest", index=meta.index, dtype=object)
    groups.loc[mask_a & ~mask_b] = "Bin A"
    groups.loc[mask_b & ~mask_a] = "Bin B"
    groups.loc[both] = "overlap"
    return groups, f"{{{a_lab}}} vs {{{b_lab}}}"


def vc_bins_from_expr(
    meta: pd.DataFrame,
    expr_a: str,
    expr_b: str | None,
) -> tuple[pd.Index, pd.Index, str]:
    groups, title = vc_sample_groups(meta, expr_a, expr_b)
    ids_a = groups.index[groups == "Bin A"]
    ids_b = groups.index[groups == "Bin B"]
    n_ov = int((groups == "overlap").sum())
    if len(ids_a) < 2 or len(ids_b) < 2:
        extra = f" Dropped {n_ov} overlapping sample(s)." if n_ov else ""
        raise ValueError(
            f"Selection does not make sense: Bin A n={len(ids_a)}, Bin B n={len(ids_b)} "
            f"(need ≥2 samples in each bin).{extra}"
        )
    return ids_a, ids_b, title


def _err_status(msg: str):
    return dbc.Alert(str(msg), color="danger", className="py-2 px-3 mb-0")


def _ok_status(msg: str):
    return html.Div(msg, className="text-muted small mb-0")


def _error_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=str(msg),
        showarrow=False,
        font=dict(color="#842029", size=13),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _group_pca_fig(
    score_df: pd.DataFrame,
    groups: pd.Series,
    *,
    title,
    uirevision: str,
) -> go.Figure:
    df = score_df[["PC1", "PC2"]].copy()
    g = groups.copy()
    g.index = g.index.astype(str)
    df.index = df.index.astype(str)
    df["group"] = g.reindex(df.index).fillna("rest").astype(str)
    df["_id"] = df.index
    present = [c for c in _BIN_ORDER if c in set(df["group"])]
    fig = px.scatter(
        df.reset_index(drop=True),
        x="PC1",
        y="PC2",
        color="group",
        category_orders={"group": present},
        color_discrete_map=_BIN_COLORS,
        custom_data=["_id"],
    )
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")
    fig = equal_xy_axes(fig, df, "PC1", "PC2")
    title_dict = title if isinstance(title, dict) else _plotly_title(str(title))
    title_lines = str(title_dict.get("text", "")).count("<br>") + 1
    fig.update_layout(title=title_dict, clickmode="event+select")
    apply_export_layout(
        fig,
        title_lines=title_lines,
        legend=True,
        legend_kwargs={"title_text": "Group"},
        uirevision=uirevision,
    )
    return fig


def _sample_bins_pca(
    session, groups: pd.Series, title: str
) -> tuple[go.Figure, str, pd.DataFrame]:
    X = session.numeric_matrix()
    pca = PCA()
    scores = pca.fit_transform(X)
    score_df = pd.DataFrame(
        scores[:, :2],
        index=session.expression.index.astype(str),
        columns=["PC1", "PC2"],
    )
    if session.metadata is not None:
        score_df = score_df.join(session.metadata)
    fig = _group_pca_fig(
        score_df,
        groups.reindex(session.expression.index).fillna("rest"),
        title=_plotly_title("Sample PCA by filter bins", title),
        uirevision="vc-bins-pca",
    )
    counts = groups.value_counts()
    bits = [f"{k} n={int(counts.get(k, 0))}" for k in _BIN_ORDER if k in counts.index]
    msg = (
        f"PCA on {len(score_df)} samples "
        f"(PC1 {100 * float(pca.explained_variance_ratio_[0]):.1f}%, "
        f"PC2 {100 * float(pca.explained_variance_ratio_[1]):.1f}%): "
        + ", ".join(bits)
    )
    return fig, msg, score_df


def _bins_frame_from_cache(cache: dict | None) -> pd.DataFrame | None:
    if not cache or "scores" not in cache:
        return None
    df = pd.DataFrame(cache["scores"])
    if "_sample_id" in df.columns:
        df = df.set_index("_sample_id")
    df.index = df.index.astype(str)
    return df


def _bins_groups_from_cache(cache: dict | None) -> pd.Series | None:
    if not cache or "groups" not in cache or "sample_ids" not in cache:
        return None
    return pd.Series(cache["groups"], index=pd.Index(cache["sample_ids"], dtype=object))


class VolcanoConditionModule:
    id = "volcano-condition"
    label = "Volcano by condition"

    def layout(self, *, readonly: bool = False):
        write_style = {"display": "none"} if readonly else None
        body = html.Div([self._samples(), self._genes(write_style)])
        return body

    def _samples(self):
        return html.Div(
            [
                html.P(
                    "Filter samples based on metadata columns",
                    className="text-muted small mb-0",
                ),
                html.P(
                    "Operators: == != < <= > >=, & | ~",
                    className="text-muted small mb-0",
                ),
                html.P(
                    "Several entries: 'column' in ['entry1', 'entry2'] or "
                    "'column' <= value for numerical data",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Browse column", className="small mb-0"),
                                dcc.Dropdown(
                                    id="vc-help-col",
                                    placeholder="Metadata column…",
                                    searchable=True,
                                    clearable=True,
                                ),
                            ],
                            md=3,
                        ),
                        dbc.Col(
                            [
                                html.Label("Entries", className="small mb-0"),
                                dcc.Dropdown(
                                    id="vc-help-entries",
                                    multi=True,
                                    searchable=True,
                                    placeholder="Select a column to list values…",
                                ),
                            ],
                            md=5,
                        ),
                        dbc.Col(
                            [
                                html.Label("Insert into filter", className="small mb-0"),
                                html.Div(
                                    [
                                        dbc.Button(
                                            "Bin A",
                                            id="vc-help-ins-a",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                            className="me-1",
                                        ),
                                        dbc.Button(
                                            "Bin B",
                                            id="vc-help-ins-b",
                                            color="secondary",
                                            outline=True,
                                            size="sm",
                                        ),
                                    ],
                                    className="mt-1",
                                ),
                            ],
                            md=4,
                        ),
                    ],
                    className="g-2 mb-1",
                ),
                html.Div(id="vc-help-hint", className="text-muted small mb-2"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Bin A"),
                                dcc.Textarea(
                                    id="vc-a-expr",
                                    placeholder="~('columnA' == entry1) | ('columnB' <= 8)",
                                    style={
                                        "width": "100%",
                                        "height": "72px",
                                        "fontFamily": "monospace",
                                    },
                                ),
                            ],
                            md=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Bin B"),
                                dcc.Textarea(
                                    id="vc-b-expr",
                                    placeholder="leave empty = complement of Bin A, or same syntax as Bin A",
                                    style={
                                        "width": "100%",
                                        "height": "72px",
                                        "fontFamily": "monospace",
                                    },
                                ),
                            ],
                            md=6,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button(
                    "Show bins on PCA",
                    id="vc-bins-pca-run",
                    color="primary",
                    className="mb-2",
                ),
                html.Div(id="vc-bins-pca-status", className="mb-2"),
                fig_size_controls(
                    "vc-bins-pca",
                    default_width=EXPORT_W,
                    default_height=EXPORT_H,
                ),
                dcc.Loading(
                    plot_with_sample_detail(
                        "vc-bins-pca-fig",
                        "vc-bins-sample-detail",
                        graph_config=_BINS_PCA_CONFIG,
                    ),
                    type="default",
                ),
                dcc.Store(id="vc-bins-cache"),
            ]
        )

    def _genes(self, write_style):
        return html.Div(
            [
                html.P(
                    "Volcano contrast of Bin A vs Bin B (overlap dropped). Click a gene for locus-lookup metadata.",
                    className="text-muted small",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Difference"),
                                dcc.Dropdown(
                                    id="vc-center",
                                    options=[
                                        {"label": "means", "value": "mean"},
                                        {"label": "medians", "value": "median"},
                                    ],
                                    value="mean",
                                    clearable=False,
                                ),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("−log10(padj) thr"),
                                dbc.Input(id="vc-padj", type="number", value=2.0, step=0.1),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Label("|ΔCLR| thr"),
                                dbc.Input(id="vc-fc", type="number", value=0.5, step=0.1),
                            ],
                            md=2,
                        ),
                        dbc.Col(
                            [
                                html.Br(),
                                dbc.Button("Run volcano", id="vc-run", color="primary"),
                            ],
                            md=2,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                html.Div(id="vc-status", className="mb-2"),
                mark_controls(col_id="vc-mark-col", entry_id="vc-mark-entry", wrap_id="vc-mark-wrap"),
                html.Div(id="vc-mark-legend", children=mark_legend_banner(0, None)),
                fig_size_controls("vc", default_width=EXPORT_W, default_height=EXPORT_H),
                dcc.Loading(
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id="vc-fig", figure={}, config=_GRAPH_CONFIG),
                                md=8,
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.H6("Gene metadata", className="mb-2"),
                                        html.Div(
                                            id="vc-gene-detail",
                                            children=gene_detail_placeholder(),
                                        ),
                                    ],
                                    className="border rounded p-2 bg-light",
                                ),
                                md=4,
                            ),
                        ],
                        className="g-2",
                    ),
                    type="default",
                ),
                dcc.Store(id="vc-last-gene"),
                dcc.Store(id="vc-cache"),
                html.Div(
                    [
                html.H6("Save Celov", className="mt-3"),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Dropdown(
                                id="vc-celov-score",
                                options=[
                                    {"label": "expression difference", "value": "fold_change"},
                                    {"label": "−log10(padj)", "value": "neg_log10_padj"},
                                    {"label": "product of both", "value": "product"},
                                ],
                                value="fold_change",
                                clearable=False,
                            ),
                            md=3,
                        ),
                        dbc.Col(
                            dcc.RadioItems(
                                id="vc-celov-mode",
                                options=[
                                    {"label": "up & down", "value": "up_and_down"},
                                    {"label": "up", "value": "up"},
                                    {"label": "down", "value": "down"},
                                    {"label": "all together", "value": "all"},
                                ],
                                value="up_and_down",
                                inline=True,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dbc.InputGroup(
                                [
                                    dbc.Input(id="vc-celov-out", type="text"),
                                    dbc.Button(
                                        "Browse…",
                                        id="vc-celov-browse",
                                        color="info",
                                        outline=True,
                                    ),
                                ]
                            ),
                            md=4,
                        ),
                    ],
                    className="g-2 mb-2",
                ),
                dbc.Button("Save Celov", id="vc-celov-save", color="secondary", className="mb-2"),
                html.Div(id="vc-celov-status", className="text-muted small"),
                    ],
                    style=write_style,
                ),
            ]
        )

    def register_callbacks(self, app: Dash, *, readonly: bool = False) -> None:
        # Celov callbacks registered only when writable
        @app.callback(
            Output("vc-help-col", "options"),
            Output("vc-help-col", "value"),
            Input("session-store", "data"),
            State("vc-help-col", "value"),
        )
        def _fill_cols(session_blob, current_col):
            session = session_from_store(session_blob)
            if session.metadata is None or session.metadata.empty:
                return [], None
            cols = [str(c) for c in session.metadata.columns]
            opts = [{"label": c, "value": c} for c in cols]
            value = current_col if current_col in cols else None
            return opts, value

        @app.callback(
            Output("vc-help-entries", "options"),
            Output("vc-help-entries", "value"),
            Output("vc-help-hint", "children"),
            Input("vc-help-col", "value"),
            Input("session-store", "data"),
        )
        def _fill_entries(col, session_blob):
            session = session_from_store(session_blob)
            if session.metadata is None or not col or col not in session.metadata.columns:
                return (
                    [],
                    None,
                    "Pick a column to list its entries. Search the dropdown, then insert into Bin A or B.",
                )
            opts, hint = _unique_entry_options(session.metadata[col], str(col))
            return opts, None, hint

        @app.callback(
            Output("vc-a-expr", "value"),
            Output("vc-b-expr", "value"),
            Input("vc-help-ins-a", "n_clicks"),
            Input("vc-help-ins-b", "n_clicks"),
            State("vc-help-col", "value"),
            State("vc-help-entries", "value"),
            State("vc-a-expr", "value"),
            State("vc-b-expr", "value"),
            prevent_initial_call=True,
        )
        def _insert_entries(n_a, n_b, col, entries, expr_a, expr_b):
            triggered = callback_context.triggered_id
            if not col or triggered not in {"vc-help-ins-a", "vc-help-ins-b"}:
                return no_update, no_update
            values = [str(v) for v in (entries or []) if v is not None and str(v) != ""]
            if not values:
                return no_update, no_update
            clause = _entry_clause(str(col), values)
            if triggered == "vc-help-ins-a":
                return _append_clause(expr_a, clause), no_update
            return no_update, _append_clause(expr_b, clause)

        @app.callback(
            Output("vc-bins-pca-fig", "figure"),
            Output("vc-bins-pca-status", "children"),
            Output("vc-bins-cache", "data"),
            Input("vc-bins-pca-run", "n_clicks"),
            Input("vc-bins-pca-fig-w", "value"),
            Input("vc-bins-pca-fig-h", "value"),
            State("session-store", "data"),
            State("vc-a-expr", "value"),
            State("vc-b-expr", "value"),
            State("vc-bins-cache", "data"),
            prevent_initial_call=True,
        )
        def _bins_pca(n_clicks, fig_w, fig_h, session_blob, expr_a, expr_b, bins_cache):
            def _fail(msg: str):
                return _error_fig(msg), _err_status(msg), no_update

            triggered = callback_context.triggered_id
            if triggered in ("vc-bins-pca-fig-w", "vc-bins-pca-fig-h"):
                score_df = _VC_RUNTIME.get("bins_score_df")
                if not isinstance(score_df, pd.DataFrame):
                    score_df = _bins_frame_from_cache(bins_cache)
                groups = _VC_RUNTIME.get("groups")
                if not isinstance(groups, pd.Series):
                    groups = _bins_groups_from_cache(bins_cache)
                title = (
                    _VC_RUNTIME.get("groups_title")
                    or (bins_cache or {}).get("title")
                    or ""
                )
                if score_df is None or groups is None:
                    return no_update, no_update, no_update
                fig = _group_pca_fig(
                    score_df,
                    groups.reindex(score_df.index).fillna("rest"),
                    title=_plotly_title("Sample PCA by filter bins", title),
                    uirevision="vc-bins-pca",
                )
                return set_fig_size(fig, fig_w, fig_h), no_update, no_update

            session = session_from_store(session_blob)
            if not session.ready or session.metadata is None:
                return _fail(session.error or "Load a dataset first.")
            if not (expr_a or "").strip():
                return _fail("Enter a Bin A filter expression.")
            try:
                groups, title = vc_sample_groups(session.metadata, expr_a, expr_b)
            except Exception as exc:  # noqa: BLE001
                return _fail(str(exc))
            _VC_RUNTIME["groups"] = groups
            _VC_RUNTIME["groups_title"] = title
            try:
                fig, msg, score_df = _sample_bins_pca(session, groups, title)
            except Exception as exc:  # noqa: BLE001
                return _fail(f"PCA error: {exc}")
            _VC_RUNTIME["bins_score_df"] = score_df
            cache = {
                "scores": score_df.reset_index(names="_sample_id").to_dict(orient="list"),
                "sample_ids": list(groups.index.astype(str)),
                "groups": list(groups.astype(str)),
                "title": title,
            }
            return set_fig_size(fig, fig_w, fig_h), _ok_status(msg), cache

        @app.callback(
            Output("vc-fig", "figure"),
            Output("vc-status", "children"),
            Output("vc-cache", "data"),
            Output("vc-mark-wrap", "style"),
            Output("vc-mark-col", "options"),
            Output("vc-mark-col", "value"),
            Output("vc-last-gene", "data", allow_duplicate=True),
            Input("vc-run", "n_clicks"),
            State("vc-fig-w", "value"),
            State("vc-fig-h", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            State("vc-a-expr", "value"),
            State("vc-b-expr", "value"),
            State("vc-center", "value"),
            State("vc-padj", "value"),
            State("vc-fc", "value"),
            prevent_initial_call=True,
        )
        def _run(
            n_clicks,
            fig_w,
            fig_h,
            session_blob,
            project_blob,
            expr_a,
            expr_b,
            center,
            padj_thr,
            fc_thr,
        ):
            hide = {"display": "none"}

            def _fail(msg: str):
                return _error_fig(msg), _err_status(msg), None, hide, [], None, None

            session = session_from_store(session_blob)
            if not session.ready or session.metadata is None:
                return _fail(session.error or "Load a dataset first.")
            if not (expr_a or "").strip():
                return _fail("Enter a Bin A filter expression.")
            try:
                groups, _ = vc_sample_groups(session.metadata, expr_a, expr_b)
                ids_a, ids_b, title = vc_bins_from_expr(
                    session.metadata, expr_a, expr_b
                )
            except Exception as exc:  # noqa: BLE001
                return _fail(str(exc))
            _VC_RUNTIME["groups"] = groups
            _VC_RUNTIME["groups_title"] = title
            expr = session.expression.T
            expr.columns = expr.columns.astype(str)
            keep_a = [s for s in ids_a.astype(str) if s in expr.columns]
            keep_b = [s for s in ids_b.astype(str) if s in expr.columns]
            if len(keep_a) < 2 or len(keep_b) < 2:
                return _fail(
                    "Selection does not make sense: Bin A n="
                    + str(len(keep_a))
                    + ", Bin B n="
                    + str(len(keep_b))
                    + " after matching the count matrix (need ≥2 samples in each bin)."
                )
            try:
                results, _fig = volcano_cluster_vs_cluster(
                    expr[keep_a],
                    expr[keep_b],
                    title=_plotly_title(title),
                    neg_log10_padj_threshold=float(padj_thr or 2.0),
                    fold_change_threshold=float(fc_thr or 0.5),
                    center=center or "mean",
                )
            except Exception as exc:  # noqa: BLE001
                return _fail("Volcano error: " + str(exc))
            path = _locus_path_from_session(session_blob, project_blob)
            lookup = None
            if path:
                try:
                    lookup = load_locus_lookup(path)
                except Exception:  # noqa: BLE001
                    lookup = None
            _VC_RUNTIME["results"] = results
            _VC_RUNTIME["locus_lookup"] = lookup
            _VC_RUNTIME["plot_meta"] = {
                "padj": float(padj_thr or 2.0),
                "fc": float(fc_thr or 0.5),
                "title": title,
            }
            fig, _n_mark = volcano_fig_from_results(
                results,
                title=_plotly_title(title, f"Bin A n={len(keep_a)} vs Bin B n={len(keep_b)}"),
                neg_log10_padj_threshold=float(padj_thr or 2.0),
                fold_change_threshold=float(fc_thr or 0.5),
                hover_map=gene_meta_hover_map(lookup),
            )
            fig = set_fig_size(fig, fig_w, fig_h)
            mark_cols = locus_mark_columns(lookup)
            mark_opts = [{"label": c, "value": c} for c in mark_cols]
            wrap = {"display": "block"} if mark_cols else hide
            return (
                fig,
                _ok_status(f"Volcano: {title}. Bin A n={len(keep_a)}, Bin B n={len(keep_b)}."),
                {"n_a": len(keep_a), "n_b": len(keep_b), "title": title},
                wrap,
                mark_opts,
                None,
                None,
            )

        @app.callback(
            Output("vc-fig", "figure", allow_duplicate=True),
            Output("vc-mark-legend", "children"),
            Input("vc-mark-col", "value"),
            Input("vc-mark-entry", "value"),
            Input("vc-fig-w", "value"),
            Input("vc-fig-h", "value"),
            prevent_initial_call=True,
        )
        def _replot_marks(mark_col, mark_entry, fig_w, fig_h):
            results = _VC_RUNTIME.get("results")
            meta = _VC_RUNTIME.get("plot_meta")
            if results is None or not meta:
                return no_update, no_update
            lookup = _VC_RUNTIME.get("locus_lookup")
            mark_genes = genes_with_meta_entry(lookup, mark_col, mark_entry)
            mark_label = f"{mark_col}={mark_entry}" if mark_col and mark_entry else None
            fig, n_mark = volcano_fig_from_results(
                results,
                title=_plotly_title(meta.get("title", "Volcano")),
                neg_log10_padj_threshold=meta["padj"],
                fold_change_threshold=meta["fc"],
                mark_genes=mark_genes,
                mark_label=mark_label,
                hover_map=gene_meta_hover_map(lookup),
            )
            return set_fig_size(fig, fig_w, fig_h), mark_legend_banner(n_mark, mark_entry)

        @app.callback(
            Output("vc-mark-entry", "options"),
            Output("vc-mark-entry", "value"),
            Output("vc-mark-entry", "disabled"),
            Output("vc-mark-entry", "placeholder"),
            Input("vc-mark-col", "value"),
        )
        def _mark_entries(col):
            lookup = _VC_RUNTIME.get("locus_lookup")
            opts = entry_dropdown_options(lookup, col) if col else []
            return opts, None, not bool(opts), "entry" if opts else "choose a column"

        @app.callback(
            Output("vc-last-gene", "data"),
            Input("vc-fig", "clickData"),
        )
        def _click(click):
            if not click:
                return no_update
            raw = click["points"][0].get("customdata")
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            return str(raw) if raw is not None else no_update

        @app.callback(
            Output("vc-gene-detail", "children"),
            Input("vc-last-gene", "data"),
            State("session-store", "data"),
            State("project-store", "data"),
        )
        def _detail(gid, session_blob, project_blob):
            if not gid:
                return gene_detail_placeholder()
            lookup = _VC_RUNTIME.get("locus_lookup")
            if lookup is None:
                path = _locus_path_from_session(session_blob, project_blob)
                if path:
                    try:
                        lookup = load_locus_lookup(path)
                    except Exception:  # noqa: BLE001
                        lookup = None
            return gene_detail_table(str(gid), lookup)

        register_sample_detail_callback(
            app,
            graph_id="vc-bins-pca-fig",
            detail_id="vc-bins-sample-detail",
            cache_id="vc-bins-cache",
            get_score_df=lambda cache: (
                _VC_RUNTIME.get("bins_score_df")
                if isinstance(_VC_RUNTIME.get("bins_score_df"), pd.DataFrame)
                else _bins_frame_from_cache(cache)
            ),
        )

        if readonly:
            return

        @app.callback(
            Output("vc-celov-out", "value"),
            Output("vc-celov-status", "children", allow_duplicate=True),
            Input("vc-celov-browse", "n_clicks"),
            State("vc-celov-out", "value"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _browse(n_clicks, current, project_blob):
            initial = (project_blob or {}).get("root") or current
            chosen = pick_save_file_dialog(
                initial=initial,
                title="Save condition volcano Celov (use {} for type)",
                defaultextension=".txt",
                initialfile="Condition_Volcano_{}.txt",
            )
            if not chosen:
                return no_update, "Celov path browse cancelled."
            return chosen, f"Celov output template: {chosen}"

        @app.callback(
            Output("vc-celov-status", "children"),
            Input("vc-celov-save", "n_clicks"),
            State("vc-celov-out", "value"),
            State("vc-celov-mode", "value"),
            State("vc-celov-score", "value"),
            State("session-store", "data"),
            State("project-store", "data"),
            prevent_initial_call=True,
        )
        def _save(n_clicks, out_path, mode, score, session_blob, project_blob):
            results = _VC_RUNTIME.get("results")
            if results is None:
                return "Run the volcano first."
            if not out_path:
                return "Choose an output path."
            try:
                weighed = _volcano_weighed_for_celov(results, score or "fold_change")
                entry = _active_dataset_entry(session_blob, project_blob)
                paths = save_classifier_celov(
                    weighed,
                    out_path,
                    mode=mode or "up_and_down",
                    locus_lookup=_VC_RUNTIME.get("locus_lookup"),
                    id_column=resolve_celov_id_col(entry),
                )
                return "Saved Celov: " + ", ".join(str(p) for p in paths)
            except Exception as exc:  # noqa: BLE001
                return f"Celov save error: {exc}"
