"""MODULE_REGISTRY — add a new analysis tab by importing and appending here."""

from __future__ import annotations

from .base import AnalysisModule
from .clustering import ClusteringModule
from .gene_expr_pca import GeneExprPCAModule
from .gene_gradients import GeneGradientsModule
from .parallel_conditions import ParallelConditionsModule
from .pca import PCAModule
from .umap_mod import UMAPModule
from .volcano_condition import VolcanoConditionModule

MODULE_REGISTRY: list[AnalysisModule] = [
    PCAModule(),
    UMAPModule(),
    ClusteringModule(),
    VolcanoConditionModule(),
    GeneGradientsModule(),
    ParallelConditionsModule(),
    GeneExprPCAModule(),
]
