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

Open a **working project folder** (Browse → **Open**). The app writes
`project.yaml` under that folder when you open or register datasets; it does
**not** scaffold data or figure subfolders.

### Recommended project folder layout

```
<project>/
  project.yaml                 # dataset registry (expression, metadata, locus paths)
  data/
    countMatrix/               # expression / count matrices
    sampleMetadata/            # sample metadata tables
    locusMapping/              # gene metadata (locusTag / BioCyc / Pathways)
  figs/                        # saved figures (optional)
  celov_output/                # Celov / scored gene exports (optional)
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

**Locus lookup** (`locusMapping/`) — CSV with one row per gene
Biocyc IDs can be used for Celov
export (resulting txt files can be imported in the biocyc metabolic map as single omics file, this also works for old locus tags/gene names, but the mapping might be less complete). Additional columns might include information on the (GO) pathways these genes are included.


### PCA

Full PCA on the unscaled count/normalized matrix (`sklearn.decomposition.PCA`, no scaler).
Choose which PCs to plot (X/Y, optional Z), scree plot (top 10 PCs), optional split aesthetics on a certain column and change olor, shape and size based on different columns.
Celov export for PC weighed genes to find up-/downregulated pathways for this PC in the biocyc viewer.

**ARI of separation** — for each of the first *n* PCs (default 10), a one-sided threshold covers all positive-class samples (side from class medians). Score = `sklearn.metrics.adjusted_rand_score`.

Additional option to run linear classification to separate LC from biofilm conditions. If the two can be separated for relatively low dimensions this means that the variance across LC conditions covers biological variance different from that of biofilms. Celov export for genes weighed according to the linear classifier shows up-/downregulated pathways forbiofilm vs LC.

**Condition enrichment** — Pearson / Spearman of rankable metadata vs top PCs plus an optional biofilm axis (`sklearn.linear_model.LinearRegression` of the label on *k* PCs; samples are projected onto the normalized coefficients). Categorical: `var_ratio` = within-entry variance / total axis variance.

**Gene expression** — map selected genes onto the current PC axes with a viridis scale (own min–max per gene). Select by gene ID or a locus-lookup name column (`geneName`, …). The same plot is available from **Weighed genes** and **Linear classifier**, choosing from the top PC / classifier weights (top *N*, `|weight|` / up / down).

### UMAP

UMAP on the unscaled count matrix.
Same aesthetic / split-by controls as PCA.

### Hierarchical clustering

**Run**
- Linkage: ward, average, complete, or single on the unscaled samples×genes matrix (and genes×samples for gene clustering).

**Cluster cut (samples)**
- **Total number of clusters** (default 2) with **Apply** to color heatmaps / PCA / dendrogram at that cut.
- Or find the first pure cluster for a metadata column/value and color clusters at that step.

**Cluster table**
- Sample dataframe with a `cluster` column at the current cut; filter by cluster or Bin A / Bin B.

**Cluster cut (genes)**
- Separate `maxclust` on the gene linkage; genes × genes heatmap is colored by gene cluster; gene table + locus lookup, filterable by cluster.

**Heatmap**
- Tabs: samples × samples (euclidean distances), genes × genes, samples × genes (expression).
- Cluster-colored dendrograms on axes; colorbar (“Pairwise distances” / “Expression”) + cluster legend.
- Click cells for sample metadata and/or locus-lookup gene annotation (when registered on the dataset).

**PCA + dendrogram** (samples × samples tab)
- PC X/Y/(optional Z) scatter colored by cluster; dendrogram with cluster leaf selection.
- Assign clicked clusters to **Bin A** / **Bin B** (any number of clusters per bin).
- Export sample metadata with column `maxclust :{t}` (below the heatmap).

**Cluster contrast volcano**
- Welch t-test + FDR; difference via **means** or **medians** between bins (data is assumed to be already transformed into fold changes).
- Thresholds for −log10(padj) and |fold change| highlight points and place dashed lines.
- After Run: mark genes by locus-lookup **column** + **entry** (cells may list several tokens separated by `;`).
- Click a gene on the volcano for locus-lookup metadata (table on the right).
- Celov export (score: expression difference, −log10(padj), or product; genes: up / down / up&down / all).

**Condition enrichment**
- Fisher exact (`scipy.stats.fisher_exact` + BH `false_discovery_control`) and Mann–Whitney U on selected metadata columns.
- Background: **Bin B** or **~treatment** (complement of Bin A / rest).

### Gene gradients

Per-gene correlation of expression vs ordered metadata levels
(Pearson / Spearman; plus Kendall τ).

- Subset samples (e.g. `Biofilm` = `True`), choose an order column (e.g. `GrowthPhase`),
  and **drag** levels to set order (natural sort as default: `region1`, `region2`, …).
- Optional replicate column (used for profile lines after clicking genes).
- Run always computes Pearson, Spearman, and Kendall τ (all three plots + pairwise).
- **Pooled**: all samples vs integer ranks.
- **Within-replicate then mean**: ρ per replicate (≥3 points), then mean across replicates. Dynamic range stays region means.
- Scatter: correlation (y) vs dynamic range of region means (x); **per-plot** |ρ|/|τ|
  and dynamic-range thresholds (grey below / black above).
- After Run: mark genes by locus-lookup column + entry (`;`-separated tokens) on all correlation and pairwise plots.
- Pairwise coefficient plots: two side-by-side, optional third below; locus-lookup **gene metadata**
  table to the right (updated when you click a gene on a correlation or pairwise plot).
- Click genes on gradient/pairwise plots for expression-vs-level profiles (below Celov;
  up to 6 per row, 6 rows). Click again to remove; Clear selected genes to reset.
- Celov export of a chosen score (+ dynamic range), up / down / up&down / all.

### Filter count matrix

Filters the active dataset on **samples** (sample metadata) or **genes** (dataset
locus lookup: geneID ↔ locusTag). Register expression, metadata, and locus lookup
per dataset; metadata/locus fields autofill from existing registrations. ATTENTION: CLR transformation depends on the geometric mean of the WHOLE dataset. The absolute values of a filtered CLR transformed dataset should always be read wrt original full data, relative distances are unaffected since the CLR transformation is an isometry for $S^D\rightarrow \mathrm{R}^D$.

### Condition prediction *(placeholder)*

Planned: supervised ML (decision tree / related methods) to predict metadata
conditions from the transcriptome, with train/test metrics and gene importances.
Tab is a stub for now — see the in-app description.

### Volcano by condition

Same Welch + BH volcano as the cluster contrast, but groups come from a metadata
column. A = selected entries (or numeric `<`/`>`);
background is **B** (other selected entries) or **~treatment** (complement). Active
filters drop overlapping samples; need ≥2 samples per side.

### Parallel conditions

Per ordered level (e.g. `GrowthPhase` region1–4): Ward on `{level samples} ∪ {all LC}`;
`separation_k` = smallest `maxclust` with no mixed biofilm/LC cluster; closest LC
cluster = Euclidean distance of mean CLR vectors. Unique = set difference (empty
unique → all closest). Not replicate-wise.

Then optional: Fisher / MWU enrichment (closest vs rest) or **pooled** gene
gradients on uniquely-close LC vs region rank.

### Gene enrichment *(placeholder)*
Planned: whereever there is the celov option, write own pathway enrichment code/use other enrichment tools

## Methods (functions and computational tools)

Matrix recipe (every analysis): genes × samples after `geneLength` → `dropna` → transpose.
No `log1p`, no `StandardScaler`. Values are used as loaded (typically CLR).

| Analysis | Functions / libraries | What is computed |
|---|---|---|
| PCA | `sklearn.decomposition.PCA` | Unscaled SVD; scores = `fit_transform`; correlation loadings = `V √λ` |
| PCA ARI | `pc_separation_ari`, `sklearn.metrics.adjusted_rand_score` | One-sided PC threshold covering all positives; ARI vs label |
| Linear classifier | `LogisticRegression`, `StratifiedKFold`, balanced accuracy | First *n* PCs; gene weights = loadings @ (`w / √λ`) |
| PCA condition enrichment | `LinearRegression`, `pearsonr` / `spearmanr`, `var_ratio` | Biofilm axis = `X (β / ‖β‖)`; numeric ρ; categorical within/total var |
| UMAP | `umap.UMAP` | Unchanged tab (`n_neighbors=20`, `min_dist=0.5`) |
| Hierarchical clustering | `scipy.cluster.hierarchy.linkage` / `fcluster` | Ward (or average/complete/single) Euclidean; `maxclust` cut |
| Cluster volcano | `scipy.stats.ttest_ind` (Welch), `false_discovery_control` (BH) | ΔCLR (mean or median); `−log10(padj)` |
| HC condition enrichment | `fisher_exact`, `mannwhitneyu` | 2×2 entry×(A vs B or ~treatment); numeric ranks via `to_numeric_rank` |
| Volcano by condition | same Welch/BH + `resolve_condition_bins` | Metadata A vs B / ~treatment |
| Gene gradients | `pearsonr`, `spearmanr`, `kendalltau` | Pooled samples vs rank 1…k, or mean of per-replicate ρ |
| Parallel conditions | `separation_k`, centroid L2, Fisher/MWU, pooled gradients | Per-level Ward + unique closest LC |

Shared helpers live in `src/GUI/modules/condition_enrichment.py`.

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python -m src.GUI.app
```

The working-folder field pre-fills with `/mnt/bronto/Johannes`. Override with:

```bash
python -m src.GUI.app --project path/to/your/project
```
