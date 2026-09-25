"""MODULE_REGISTRY — add a new analysis tab by importing and appending here."""

from __future__ import annotations

from .clustering import ClusteringModule
from .filter_matrix import FilterModule
from .gene_gradients import GeneGradientsModule
from .parallel_conditions import ParallelConditionsModule
from .pca import PCAModule
from .umap_mod import UMAPModule
from .volcano_condition import VolcanoConditionModule

MODULE_REGISTRY = [
    PCAModule(),
    UMAPModule(),
    ClusteringModule(),
    VolcanoConditionModule(),
    GeneGradientsModule(),
    ParallelConditionsModule(),
    FilterModule(),
]
