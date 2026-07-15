"""Prepare BioCyc Celov / multi-omics overview input files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_locus_lookup(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Locus lookup not found: {path}")
    lookup = pd.read_csv(path)
    if "locusTag" not in lookup.columns:
        raise ValueError("Locus lookup must contain a 'locusTag' column")
    return lookup


def annotate_gene_table(gene_df: pd.DataFrame, locus_lookup: pd.DataFrame) -> pd.DataFrame:
    """Merge gene scores (index or geneID column) with locus-lookup annotation columns."""
    annotated = gene_df.copy()
    if "geneID" not in annotated.columns:
        annotated = annotated.reset_index(names="geneID")
    annotated["geneID"] = annotated["geneID"].astype(str)
    merge_cols = [c for c in locus_lookup.columns if c != "locusTag"]
    annotated = annotated.merge(
        locus_lookup.rename(columns={"locusTag": "geneID"})[["geneID", *merge_cols]],
        on="geneID",
        how="left",
    )
    return annotated


def celov_multiomics_file_generation(
    data: pd.DataFrame,
    filepath: str | Path,
    column1: str,
    column2: str | None = None,
    column2_invert: bool = True,
    description: str | None = None,
    id_column: str = "biocyc_id",
    dataset_label: str | None = None,
    confidence_label: str | None = None,
) -> pd.DataFrame:
    """Write a BioCyc multi-omics input table from a scored gene DataFrame."""
    data = data.dropna(subset=[column1, id_column]).copy()
    data = data[
        data[id_column].astype(str).str.strip().ne("")
        & data[id_column].astype(str).str.lower().ne("nan")
    ]

    if dataset_label is None:
        dataset_label = column1
    if confidence_label is None and column2 is not None:
        confidence_label = column2

    lines: list[str] = []

    if description:
        for line in str(description).splitlines():
            lines.append(f"# {line}" if not line.startswith("#") else line)
        lines.append("")

    lines.extend(
        [
            "$Table=Table1",
            "$Column=1",
            "$Type=Gene",
            "$Target=Edge-Color",
            "$Counts=Relative",
            "$DataValueUse=1",
            "$NumColumns=1",
            f"$DatasetLabel={dataset_label}",
            "",
        ]
    )

    if column2 is not None:
        lines.extend(
            [
                "$Table=Table2",
                "$Column=1",
                "$Type=Gene",
                "$Target=Edge-Thickness",
                "$Counts=Absolute",
                "$NumColumns=1",
                f"$DatasetLabel={confidence_label}",
                "",
            ]
        )

    lines.append("$Table=Table1")
    lines.append(f"$ID\t{column1}")
    for _, row in data.iterrows():
        lines.append(f"{row[id_column]}\t{row[column1]}")

    if column2 is not None:
        confidence_data = data.dropna(subset=[column2]).copy()
        if column2_invert:
            confidence_data[column2] = 1 / (confidence_data[column2] + 1e-6)
        lines.append("")
        lines.append("$Table=Table2")
        lines.append(f"$ID\t{column2}")
        for _, row in confidence_data.iterrows():
            lines.append(f"{row[id_column]}\t{row[column2]}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved multi-omics input to %s (%s genes)", filepath, len(data))
    return data


def save_celov_up_down_combined(
    scored: pd.DataFrame,
    out_dir: str | Path,
    stem: str,
    score_column: str,
    locus_lookup: pd.DataFrame | None = None,
    confidence_column: str | None = None,
    column2_invert: bool = False,
    score_threshold: float = 0.0,
    id_column: str = "biocyc_id",
) -> dict[str, Path]:
    """Always write Celov files for upregulated, downregulated, and combined scores.

    Returns mapping ``{\"combined\": Path, \"upregulated\": Path, \"downregulated\": Path}``.
    Also writes matching CSV tables next to the ``.txt`` files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scored.copy()
    if locus_lookup is not None and id_column not in df.columns:
        df = annotate_gene_table(df, locus_lookup)

    if id_column not in df.columns:
        raise ValueError(
            f"Scored table needs '{id_column}' (provide locus_lookup or pre-annotate)."
        )
    if score_column not in df.columns:
        raise ValueError(f"Missing score column {score_column!r}")

    thr = float(score_threshold)
    combined = df.dropna(subset=[score_column, id_column])
    up = combined[combined[score_column] > thr]
    down = combined[combined[score_column] < -thr]

    paths: dict[str, Path] = {}
    subsets = {
        "combined": combined,
        "upregulated": up,
        "downregulated": down,
    }
    for label, subset in subsets.items():
        txt = out_dir / f"{stem}_{label}.txt"
        csv = out_dir / f"{stem}_{label}.csv"
        subset.to_csv(csv, index=False)
        celov_multiomics_file_generation(
            subset,
            txt,
            score_column,
            column2=confidence_column if confidence_column in subset.columns else None,
            column2_invert=column2_invert,
            id_column=id_column,
            dataset_label=f"{stem}_{label}",
        )
        paths[label] = txt
    return paths
