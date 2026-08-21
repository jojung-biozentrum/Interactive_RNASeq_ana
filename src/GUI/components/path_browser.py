"""Folder picker that renders in the user's browser.

A native (tkinter) dialog opens a window on the machine running Dash, which is
the server — useless for a remote user and it crashes on a headless VM. This
modal lists the server filesystem as HTML instead.

Listing is restricted to ``roots``: nothing outside them is shown or selectable.
"""

from __future__ import annotations

import os
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, callback_context, dcc, html, no_update
import dash_bootstrap_components as dbc


def parse_roots(raw: str | list[str] | None) -> list[Path]:
    """Browsable roots from a list or an ``os.pathsep``-separated string."""
    if not raw:
        return []
    items = raw.split(os.pathsep) if isinstance(raw, str) else list(raw)
    roots: list[Path] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        try:
            path = path.resolve()
        except OSError:
            continue
        if path.is_dir() and path not in roots:
            roots.append(path)
    return roots


def is_within(path: str | Path, roots: list[Path]) -> bool:
    """True if ``path`` is one of ``roots`` or inside one of them."""
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        return False
    for root in roots:
        if resolved == root or root in resolved.parents:
            return True
    return False


def _start_dir(current: str | None, roots: list[Path]) -> Path:
    if current and str(current).strip():
        try:
            path = Path(str(current).strip()).expanduser().resolve()
        except OSError:
            path = None
        if path is not None:
            if not path.is_dir():
                path = path.parent
            if path.is_dir() and is_within(path, roots):
                return path
    return roots[0]


def folder_browser_modal(prefix: str, title: str = "Select working folder") -> html.Div:
    """Modal body for the picker. Pair with a button whose id is ``{prefix}-open``."""
    return html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(title)),
                    dbc.ModalBody(
                        [
                            html.Div(
                                id=f"{prefix}-current",
                                className="font-monospace small text-muted mb-2",
                            ),
                            dbc.Button(
                                "Up one level",
                                id=f"{prefix}-up",
                                size="sm",
                                color="secondary",
                                outline=True,
                                className="mb-2",
                            ),
                            html.Div(id=f"{prefix}-listing"),
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "Cancel",
                                id=f"{prefix}-cancel",
                                color="secondary",
                                outline=True,
                                className="me-2",
                            ),
                            dbc.Button(
                                "Use this folder",
                                id=f"{prefix}-select",
                                color="primary",
                            ),
                        ]
                    ),
                ],
                id=f"{prefix}-modal",
                is_open=False,
                size="lg",
                scrollable=True,
            ),
            dcc.Store(id=f"{prefix}-dir"),
        ]
    )


def register_folder_browser(
    app: Dash,
    prefix: str,
    *,
    roots: list[Path],
    target_id: str,
) -> None:
    """Wire the picker; the chosen folder is written into ``target_id``'s value."""
    entry_type = f"{prefix}-entry"

    @app.callback(
        Output(f"{prefix}-modal", "is_open"),
        Output(f"{prefix}-dir", "data"),
        Input(f"{prefix}-open", "n_clicks"),
        Input(f"{prefix}-cancel", "n_clicks"),
        Input(f"{prefix}-select", "n_clicks"),
        Input(f"{prefix}-up", "n_clicks"),
        Input({"type": entry_type, "index": ALL}, "n_clicks"),
        State(target_id, "value"),
        State(f"{prefix}-dir", "data"),
        prevent_initial_call=True,
    )
    def _navigate(open_click, cancel, select, up, entry_clicks, current_value, current_dir):
        if not roots:
            return False, None
        trigger = callback_context.triggered_id
        if trigger == f"{prefix}-open":
            return True, str(_start_dir(current_value, roots))
        if trigger in (f"{prefix}-cancel", f"{prefix}-select"):
            return False, no_update

        here = _start_dir(current_dir, roots)
        if trigger == f"{prefix}-up":
            parent = here.parent
            if parent != here and is_within(parent, roots):
                return True, str(parent)
            return True, no_update

        if isinstance(trigger, dict) and trigger.get("type") == entry_type:
            # Freshly rendered list items fire with n_clicks None; ignore those.
            if not any(entry_clicks or []):
                return no_update, no_update
            child = (here / str(trigger.get("index", ""))).resolve()
            if child.is_dir() and is_within(child, roots):
                return True, str(child)
        return no_update, no_update

    @app.callback(
        Output(f"{prefix}-listing", "children"),
        Output(f"{prefix}-current", "children"),
        Input(f"{prefix}-dir", "data"),
    )
    def _listing(current_dir):
        if not roots:
            return (
                dbc.Alert(
                    "No browsable folders configured (set DASH_BROWSE_ROOTS).",
                    color="warning",
                    className="mb-0",
                ),
                "",
            )
        # The store round-trips through the browser, so re-check the bounds here.
        here = _start_dir(current_dir, roots)
        try:
            children = sorted(here.iterdir(), key=lambda p: p.name.lower())
        except OSError as exc:
            return (
                dbc.Alert(f"Cannot list this folder: {exc}", color="danger", className="mb-0"),
                str(here),
            )

        items = []
        for child in children:
            if child.name.startswith("."):
                continue
            if child.is_dir():
                items.append(
                    dbc.ListGroupItem(
                        f"{child.name}/",
                        id={"type": entry_type, "index": child.name},
                        action=True,
                        n_clicks=0,
                    )
                )
            else:
                items.append(
                    dbc.ListGroupItem(child.name, disabled=True, className="text-muted")
                )
        if not items:
            items = [dbc.ListGroupItem("(empty)", disabled=True, className="text-muted")]

        has_yaml = (here / "project.yaml").is_file()
        label = html.Span(
            [
                html.Span(str(here)),
                dbc.Badge(
                    "project.yaml" if has_yaml else "no project.yaml",
                    color="success" if has_yaml else "secondary",
                    className="ms-2",
                ),
            ]
        )
        return dbc.ListGroup(items, flush=True), label

    @app.callback(
        Output(target_id, "value"),
        Input(f"{prefix}-select", "n_clicks"),
        State(f"{prefix}-dir", "data"),
        prevent_initial_call=True,
    )
    def _select(n_clicks, current_dir):
        if not current_dir or not is_within(current_dir, roots):
            return no_update
        return current_dir
