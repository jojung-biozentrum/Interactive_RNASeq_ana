"""Project helpers: load project.yaml and resolve dataset paths under the root."""

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


@dataclass
class Project:
    root: Path
    datasets: list[DatasetEntry] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def yaml_path(self) -> Path:
        return self.root / "project.yaml"

    def resolve(self, relative: str) -> Path:
        """Resolve a dataset path; must stay under ``self.root`` (no absolute escapes)."""
        raw = str(relative or "").strip()
        if not raw:
            raise ValueError("Empty path")
        p = Path(raw)
        if p.is_absolute():
            raise ValueError(f"Absolute dataset paths are not allowed: {raw}")
        root = self.root.resolve()
        resolved = (root / p).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Path escapes project root ({root}): {raw}"
            ) from exc
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasets": [d.to_dict() for d in self.datasets],
            "settings": self.settings,
        }

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
        if "excluded_datasets" in settings:
            settings = {k: v for k, v in settings.items() if k != "excluded_datasets"}
        if "locus_lookup" in settings:
            settings = {k: v for k, v in settings.items() if k != "locus_lookup"}
        return cls(root=root, datasets=datasets, settings=settings)


def open_project(root: str | Path) -> Project:
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data folder not found: {root}")
    yaml_path = root / "project.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No project.yaml in {root}")
    return Project.load(root)
