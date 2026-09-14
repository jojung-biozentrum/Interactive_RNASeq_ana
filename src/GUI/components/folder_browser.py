"""Local folder/file picker helpers for the Dash app."""

from __future__ import annotations

from pathlib import Path


def _initial_dir(initial: str | None) -> str:
    if initial and str(initial).strip():
        p = Path(str(initial).strip()).expanduser()
        try:
            if p.exists():
                return str(p if p.is_dir() else p.parent)
            if p.parent.exists():
                return str(p.parent)
        except OSError:
            pass
    return str(Path.home())


def _tk_root():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        root.wm_attributes("-topmost", True)
    except Exception:  # noqa: BLE001
        pass
    return root


def pick_folder_dialog(initial: str | None = None, title: str = "Select folder") -> str | None:
    """Open a native OS folder dialog. Returns absolute path or None."""
    try:
        from tkinter import filedialog
    except Exception:  # noqa: BLE001
        return None

    root = _tk_root()
    try:
        chosen = filedialog.askdirectory(
            parent=root,
            initialdir=_initial_dir(initial),
            title=title,
            mustexist=True,
        )
    finally:
        root.destroy()

    if not chosen:
        return None
    return str(Path(chosen).resolve())


def pick_file_dialog(
    initial: str | None = None,
    title: str = "Select file",
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Open a native OS file dialog. Returns absolute path or None."""
    try:
        from tkinter import filedialog
    except Exception:  # noqa: BLE001
        return None

    if filetypes is None:
        filetypes = [
            ("Tables", "*.csv *.tsv *.txt"),
            ("CSV", "*.csv"),
            ("TSV / TXT", "*.tsv *.txt"),
            ("All files", "*.*"),
        ]

    root = _tk_root()
    try:
        chosen = filedialog.askopenfilename(
            parent=root,
            initialdir=_initial_dir(initial),
            title=title,
            filetypes=filetypes,
        )
    finally:
        root.destroy()

    if not chosen:
        return None
    return str(Path(chosen).resolve())


def pick_save_file_dialog(
    initial: str | None = None,
    title: str = "Save file",
    defaultextension: str = ".txt",
    filetypes: list[tuple[str, str]] | None = None,
    initialfile: str | None = None,
) -> str | None:
    """Open a native OS save-file dialog. Returns absolute path or None."""
    try:
        from tkinter import filedialog
    except Exception:  # noqa: BLE001
        return None

    if filetypes is None:
        filetypes = [
            ("Text / Celov", "*.txt"),
            ("CSV", "*.csv"),
            ("All files", "*.*"),
        ]

    root = _tk_root()
    try:
        chosen = filedialog.asksaveasfilename(
            parent=root,
            initialdir=_initial_dir(initial),
            title=title,
            defaultextension=defaultextension,
            filetypes=filetypes,
            initialfile=initialfile or "",
        )
    finally:
        root.destroy()

    if not chosen:
        return None
    return str(Path(chosen).resolve())


def path_relative_to(project_root: str | Path | None, absolute: str | Path) -> str:
    """Prefer a path relative to the working folder; fall back to absolute."""
    abs_path = Path(absolute).resolve()
    if not project_root:
        return str(abs_path)
    root = Path(project_root).resolve()
    try:
        return str(abs_path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(abs_path)
