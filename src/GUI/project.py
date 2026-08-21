"""Working-folder project helpers: open/create, project.yaml, figure/export paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetEntry:
    name: str
    expression: str
    metadata: str
    locus_lookup: str = ""
    sample_id_col: str = "fileName"
    celov_id_col: str = "biocyc_id"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "metadata": self.metadata,
            "locus_lookup": self.locus_lookup,
            "sample_id_col": self.sample_id_col,
            "celov_id_col": self.celov_id_col,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetEntry:
        return cls(
            name=d["name"],
            expression=d["expression"],
            metadata=d["metadata"],
            locus_lookup=d.get("locus_lookup") or "",
            sample_id_col=d.get("sample_id_col", "fileName"),
            celov_id_col=(d.get("celov_id_col") or "biocyc_id").strip() or "biocyc_id",
        )


def resolve_celov_id_col(entry: DatasetEntry | dict[str, Any] | None) -> str:
    """Celov gene-ID column from a dataset entry (default ``biocyc_id``)."""
    if entry is None:
        return "biocyc_id"
    if isinstance(entry, DatasetEntry):
        col = entry.celov_id_col
    else:
        col = entry.get("celov_id_col")
    return (str(col).strip() if col else "") or "biocyc_id"


@dataclass
class Project:
    root: Path
    datasets: list[DatasetEntry] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def yaml_path(self) -> Path:
        return self.root / "project.yaml"

    def figures_dir(self) -> Path:
        """Recommended figures folder (``figs/``); not created until a figure is saved."""
        return self.root / "figs"

    def exports_dir(self) -> Path:
        """Recommended Celov / export folder; not created until a file is saved."""
        return self.root / "celov_output"

    def resolve(self, relative: str) -> Path:
        p = Path(relative)
        if p.is_absolute():
            return p
        return (self.root / p).resolve()

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasets": [d.to_dict() for d in self.datasets],
            "settings": self.settings,
        }

    def save(self) -> None:
        # Drop legacy project-wide keys (locus was moved per-dataset; exclude UI removed)
        drop = {"locus_lookup", "excluded_datasets"}
        if any(k in self.settings for k in drop):
            self.settings = {k: v for k, v in self.settings.items() if k not in drop}
        self.yaml_path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, root: str | Path) -> Project:
        root = Path(root).resolve()
        yaml_path = root / "project.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"No project.yaml in {root}")
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        datasets = [DatasetEntry.from_dict(d) for d in raw.get("datasets", [])]
        settings = raw.get("settings", {}) or {}
        # Migrate legacy settings.locus_lookup onto datasets that lack a path
        legacy = (settings.get("locus_lookup") or "").strip()
        if legacy:
            datasets = [
                DatasetEntry(
                    name=d.name,
                    expression=d.expression,
                    metadata=d.metadata,
                    locus_lookup=d.locus_lookup or legacy,
                    sample_id_col=d.sample_id_col,
                    celov_id_col=d.celov_id_col,
                )
                for d in datasets
            ]
        # Drop legacy exclude list (replaced by unregister)
        if "excluded_datasets" in settings:
            settings = {k: v for k, v in settings.items() if k != "excluded_datasets"}
        if "locus_lookup" in settings:
            settings = {k: v for k, v in settings.items() if k != "locus_lookup"}
        return cls(root=root, datasets=datasets, settings=settings)


def create_project(root: str | Path, name: str | None = None) -> Project:
    """Create working folder + ``project.yaml`` only (no data/fig subfolders)."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    project = Project(root=root, settings={"name": name or root.name})
    if not project.yaml_path.exists():
        project.save()
    else:
        project = Project.load(root)
    return project


def open_project(root: str | Path) -> Project:
    """Load an existing project.yaml. Does not create or write files."""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Working folder not found: {root}")
    yaml_path = root / "project.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"No project.yaml in {root}. Prepare the dataset registry ahead of "
            "time; this server build does not write project.yaml."
        )
    return Project.load(root)


def save_figure(
    fig,
    stem: str,
    out_dir: str | Path | None = None,
    formats: tuple[str, ...] = ("png",),
    project: Project | None = None,
) -> list[Path]:
    """Save a Plotly figure as png/svg/html.

    ``out_dir`` defaults to ``project.figures_dir()`` (``figs/``) when a project is given.
    Creates ``out_dir`` only when saving.
    """
    if out_dir is None:
        if project is None:
            raise ValueError("Provide out_dir or project")
        out_dir = project.figures_dir()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = out_dir / f"{stem}.{fmt}"
        if fmt == "html":
            fig.write_html(str(path))
        elif fmt in {"png", "svg", "pdf", "jpeg", "jpg", "webp"}:
            fig.write_image(str(path), format=fmt if fmt != "jpg" else "jpeg")
        else:
            raise ValueError(f"Unsupported figure format: {fmt}")
        saved.append(path)
    return saved


def save_export(project: Project, df, filename: str) -> Path:
    """Save a DataFrame CSV into ``celov_output/`` (created only when saving)."""
    out_dir = project.exports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=True)
    return path
