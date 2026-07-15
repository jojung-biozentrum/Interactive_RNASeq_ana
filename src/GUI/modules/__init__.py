"""MODULE_REGISTRY — add a new analysis tab by importing and appending here."""

from __future__ import annotations

from .clustering import ClusteringModule
from .condition_prediction import ConditionPredictionModule
from .filter_matrix import FilterModule
from .gene_gradients import GeneGradientsModule
from .parallel_conditions import ParallelConditionsModule
from .pca import PCAModule
from .umap_mod import UMAPModule

MODULE_REGISTRY = [
    PCAModule(),
    UMAPModule(),
    ClusteringModule(),
    GeneGradientsModule(),
    FilterModule(),
    ConditionPredictionModule(),
    ParallelConditionsModule(),
]
