"""MODULE_REGISTRY — add a new analysis tab by importing and appending here."""

from __future__ import annotations

from .clustering import ClusteringModule
from .gene_gradients import GeneGradientsModule
from .pca import PCAModule
from .umap_mod import UMAPModule

MODULE_REGISTRY = [
    PCAModule(),
    UMAPModule(),
    ClusteringModule(),
    GeneGradientsModule(),
]
