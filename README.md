# Biofilm microenvironments - Analysis pipeline

Interactive PCA viewer for biofilm / LC RNA-seq datasets.

This interactive interface was created using the cursor agent. The single analysis modules were first created as ipynb's to confirm that the analysis logic in the vibe coded visualization tool is correct by comparing expected outputs. For example, a first implementation of the PCA analysis resulted in a log2 transformation on the already CLR transformed dataset prior to the PCA.

## Project organization

```
.
├── src
│   ├── biocyc
│   ├── GUI              <- Dash + Plotly PCA viewer
│   ├── kazukis_rnaseq_pipeline
│   └── metaProcessing
├── requirements.txt
├── .gitignore
└── README.md
```

## Interactive viewer

**This `Virtual-server` branch is the read-only gunicorn/nginx build.**
It does not write `project.yaml`, register datasets, filter matrices, or
export Celov / CSV files (Bronto is mounted read-only on the lab VM).
See [deploy/README.md](deploy/README.md) for systemd + nginx.

Open a **working project folder** that already contains `project.yaml`.
The app never writes that file.

### Recommended project folder layout

```
<project>/
  project.yaml                 # dataset registry (expression, metadata, locus paths)
  data/
    countMatrix/               # expression / count matrices
    sampleMetadata/            # sample metadata tables
    locusMapping/              # gene metadata (locusTag / BioCyc / Pathways)
  figs/                        # optional local notes; this branch does not save figures to disk
```

Register paths relative to `<project>/` when possible (e.g.
`data/countMatrix/kdv1679_new_CLR.csv`).

### Expected table formats

**Expression / count matrix** — CSV, genes × samples. Must include a `geneID`
column (or use the first column as gene IDs) and a `geneLength` column; sample
columns start immediately after the `geneLength` column.

**Sample metadata** (`sampleMetadata/`) — CSV with one row per sample.
Must join to expression sample IDs via the configured sample-ID column
(default `fileName`). Remaining columns are used as metadata information for the analysis.

**Locus lookup** (`locusMapping/`) — CSV with one row per gene.
Additional columns might include information on the (GO) pathways these genes are included.


### PCA

Full PCA on the unscaled count/normalized matrix.
Choose which PCs to plot (X/Y, optional Z), scree plot (top 10 PCs), optional split aesthetics on a certain column and change color, shape and size based on different columns.

Additional option to run linear classification to separate LC from biofilm conditions. If the two can be separated for relatively low dimensions this means that the variance across LC conditions covers biological variance different from that of biofilms.

### UMAP

UMAP on the unscaled count matrix (same params as ``UMAP.ipynb``).
Same aesthetic / split-by controls as PCA.

### Hierarchical clustering

**Run**
- Linkage: ward, average, complete, or single on the unscaled samples×genes matrix (and genes×samples for gene clustering).

**Cluster cut (samples)**
- **Total number of clusters** (default 2) with **Apply** to color heatmaps / PCA / dendrogram at that cut.
- Or find the first pure cluster for a metadata column/value and color clusters at that step.

**Heatmap**
- Tabs: samples × samples (euclidean distances), genes × genes, samples × genes (expression).
- Cluster-colored dendrograms on axes; colorbar (“Pairwise distances” / “Expression”) + cluster legend.
- Click cells for sample metadata and/or locus-lookup gene annotation (when registered on the dataset).

**PCA + dendrogram** (samples × samples tab)
- PC X/Y/(optional Z) scatter colored by cluster; dendrogram with cluster leaf selection.
- Assign clicked clusters to **Bin A** / **Bin B** (any number of clusters per bin).

**Cluster contrast volcano**
- Welch t-test + FDR; difference via **means** or **medians** between bins (data is assumed to be already transformed into fold changes).
- Thresholds for −log10(padj) and |fold change| highlight points and place dashed lines.
- After Run: mark genes by locus-lookup **column** + **entry** (cells may list several tokens separated by `;`).
- Click a gene on the volcano for locus-lookup metadata (table on the right).

### Gene gradients

Per-gene correlation of expression vs ordered metadata levels
(Pearson / Spearman; plus Kendall τ).

- Subset samples (e.g. `Biofilm` = `True`), choose an order column (e.g. `GrowthPhase`),
  and **drag** levels to set order (natural sort as default: `region1`, `region2`, …).
- Optional replicate column (used for profile lines after clicking genes).
- Run always computes Pearson, Spearman, and Kendall τ (all three plots + pairwise).
- Scatter: correlation (y) vs dynamic range of region means (x); **per-plot** |ρ|/|τ|
  and dynamic-range thresholds (grey below / black above).
- After Run: mark genes by locus-lookup column + entry (`;`-separated tokens) on all correlation and pairwise plots.
- Pairwise coefficient plots: two side-by-side, optional third below; locus-lookup **gene metadata**
  table to the right (updated when you click a gene on a correlation or pairwise plot).
- Click genes on gradient/pairwise plots for expression-vs-level profiles
  (up to 6 per row, 6 rows). Click again to remove; Clear selected genes to reset.

### Filter count matrix

Not included on this branch (would write a filtered matrix to disk).

### Condition prediction *(placeholder)*

Planned: supervised ML (decision tree / related methods) to predict metadata
conditions from the transcriptome, with train/test metrics and gene importances.
Tab is a stub for now — see the in-app description.

### Parallel conditions *(placeholder)*

Planned: subset + ordered gradient column → hierarchical clustering per gradient
level → match closest clusters → find condition columns that best correlate with
cluster differences, with significance vs the rest of the dataset.
Tab is a stub for now — see the in-app description.

### Gene enrichment *(placeholder)*
Planned: pathway enrichment (this branch has no Celov file export).

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python -m src.GUI.app
```
or
```bash
python -m src.GUI.app --project path/to/your/project
```

Omit `--secret-config` for no login. On the VM, gunicorn sets
`DASH_SECRET_CONFIG` (see [deploy/README.md](deploy/README.md)).
