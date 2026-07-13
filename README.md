# Biofilm microenvironments - Analysis pipeline
Project to
  1. Analyze the GITCME protocoll
  2. Analyze the statistics of the data
  3. Use PCA, UMAP, hierarchical clustering and pairwise distances to identify differences/similarities between biofilm and LC conditions

# Project organization
```
.
├── bin                <- Compiled and external code, ignored by git
├── data               <- project data, ignored by git
│   ├── celov          <- Input files for pathway enrichment
│   ├── countMatrix    <- The final, canonical data sets for modeling
│   ├── referenceGenome
│   ├── sampleMetadata
│   └── temp           <- Intermediate data that has been transformed during the GITCME analysis
├── results
│   ├── figures        <- Figures
│   └── output         <- Other output
├── src                <- Source code for this project
├── .gitignore
├── README.md
└── Snakefile
```
