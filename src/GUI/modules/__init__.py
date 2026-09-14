"""MODULE_REGISTRY — add a new analysis tab by importing and appending here.

Filter (writes matrices) is omitted when ``readonly=True`` so the Virtual-server
build shares this file with desktop ``main``.
"""

from __future__ import annotations

from .base import AnalysisModule
from .clustering import ClusteringModule
from .condition_prediction import ConditionPredictionModule
from .filter_matrix import FilterModule
from .gene_gradients import GeneGradientsModule
from .parallel_conditions import ParallelConditionsModule
from .pca import PCAModule
from .umap_mod import UMAPModule

# Default full registry (desktop). Prefer ``build_registry`` from ``create_app``.
MODULE_REGISTRY: list[AnalysisModule] = [
    PCAModule(),
    UMAPModule(),
    ClusteringModule(),
    GeneGradientsModule(),
    FilterModule(),
    ConditionPredictionModule(),
    ParallelConditionsModule(),
]


def build_registry(*, readonly: bool = False) -> list[AnalysisModule]:
    """Analysis tabs for this process. Write-only tabs drop out in readonly mode."""
    mods: list[AnalysisModule] = [
        PCAModule(),
        UMAPModule(),
        ClusteringModule(),
        GeneGradientsModule(),
    ]
    if not readonly:
        mods.append(FilterModule())
    mods.extend(
        [
            ConditionPredictionModule(),
            ParallelConditionsModule(),
        ]
    )
    return mods
