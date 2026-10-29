# tbx5

the Tbx5 crispant data, from the cellranger output to the per embryo nn distortion tables `analysis/figures/tbx5` is drawn off. two stages: `demultiplexing/` calls which of the 8 pooled embryos each cell came from, `support_comparison/` fits the crispants and the E9.5 reference into one latent and compares their support against the model's prediction.

`common.py` in each directory holds the paths, cutoffs and loaders the rest import.

## demultiplexing/

- `preprocess_counts.py` loads both Tbx5 lanes, keeps cells at >= 200 genes and < 15% mito on the full transcriptome, subsets to the 3956 gene reference panel, and writes `X` as raw counts with 50 pcs of the log-normalised panel in `X_pca`. `MIN_UMI` / `MIN_GENES` / `MAX_PCT_MT` are a second, tighter qc flag applied at fit time rather than at load, so changing them does not force a rebuild.
- `model.md` is the generative model the calling fits: a doublet mixture, no ambient.
- `aggregate.py` cuts `N_BINS` equal read mass bins per chromosome out of a `FINE`-resolution read histogram taken over both Tbx5 lanes, then joins each Tbx5 lane's `allSNP` against its `read.BC` and aggregates one count per read at the read's first SNP. autosomes only, so nothing splits on sex. `counts.npz` and `histogram_f1000.csv` are prebuilt; `counts()` asserts the cache covers every cell in the h5ad and reindexes it into obs order, and builds on demand if the npz is missing.
- `snp_demux.py` runs the three passes of `model.md` and embeds the same features with umap. `components.csv` gives each embryo component a `geno_radius`, its genotype's distance from the pi-weighted pool mean genotype over the furthest component's, and the same statement on the umap layout as `umap_radius`. a component fit to the pool mean rather than to an embryo sits near 0 on both, and anything under `CENTRAL` is printed by name.
- `transcriptional_doublet_detection.py` builds one `K_NN` graph over the cells with a SNP profile in `X_pca` and writes both obs columns, so it is the last step. a cell that fails qc or has no SNP profile is `throwaway` too.

## support_comparison/

- `fit_scvi.py`: the integrated fit the distances are taken in: the demultiplexed singlets against `N_REF` E9.5 devmap cells, batch key `crispant` against `devmap` so the embryos are not corrected apart from each other, the reference thinned quantile to quantile down to the crispant depth first.
- `labels.py`: the devmap subtype transferred onto the crispant cells by a `K_LABEL` neighbour vote, which is what gives them a lineage.
- `predicted.py`: the screen's Tbx5 leaves resolved to E9.5 devmap cells. a target-time leaf that stayed on the manifold contributes the cell it snapped to under either convention; `MODE` says what happens to one that went off it. `prune` drops it, which is the screen's own convention; `freeze` contributes the snapped cell of its last on-manifold ancestor, and since a frozen state sits at E8.0-E9.0 it is snapped forward to its nearest E9.5 cell in the devmap latent. `OFF_RATIO` is the cutoff a node counts as off-manifold at, swept post-hoc off the per-node numbers the screen recorded.
- `integrated_umap.py`: `umap_integrated.parquet`, a umap of the integrated latent over `N_UMAP` cells per dataset. cached; delete the parquet to rebuild.
- `nn_distortion.py`: the distortion statistic itself. per reference cell, the ratio of its nearest neighbour distance into a perturbed draw over its distance into a draw from another reference embryo, the two draws subsampled to the same size. a library file, no outputs of its own.
- `per_embryo.py`: the version that scores each crispant embryo on its own. four differences from `nn_distortion.py`, in order of importance. every draw is taken inside the scope, so per embryo composition cancels: with a dataset-wide draw the crispant embryos' heart numbers track how many cells each embryo gave (rho +0.86) and what fraction of them were heart (rho -0.86), which is dissection and not phenotype, and scoping in drops the between-embryo sd from 0.110 to 0.065. the crispant embryo is the unit rather than the cell, and between-embryo sd runs about 4x the draw-to-draw sd, so a seed error bar understates the spread. the denominator embryo is a fixed cycle over the sorted names rather than a random pick. and the statistic is the log ratio centred on its median over the scope; the measured column carries a technical offset the predicted one does not, so only the centred columns compare, and the level inside a scope is not identified without a second crispant target through the same pipeline.
- `per_embryo_nn_distortion.py`: the caller, one scope at a time: every lineage, then `HEART_TYPES`. `DRAWS` numerator draws are averaged per unit before the subtype means, so draw noise stays out of the between-embryo spread. the subtype table carries `measured` with its sem, min, max, how many of the eight embryos are positive, and a one-sample t over them. the predicted side gets a leave-one-tree-out jackknife where the trees support it; in the heart they do not, because the screen grows its trees per reference replicate, so ~80% of a tree's leaves land on its home replicate's cells and after the third-embryo rule the largest `n` one tree covers for all three query embryos is 11. the centring is per draw and per scope, so the median taken off each column is written to `per_embryo_<scope>_offset.csv` and adding it back recovers the uncentred log ratio. `n_jack` in the support table is the `n` every jackknife replicate would run at, 86 for the heart against an auto `n` of 180, so `--n 86` buys the interval at the cost of a noisier point. `--scope` runs one on its own and does not rewrite the support table.

every reference cell is a query cell. each reference embryo takes its turn as the query, paired with one other embryo for the denominator: one choice per embryo, not per cell. there are three reference embryos, E9.5-R1/R2/R3. the predicted perturbed set is itself made of reference cells, so `distortion()` takes `p_embryo` and draws the numerator from a third embryo, neither the query cell's own nor the denominator's. without it the two draws share cells, the same cell wins both searches and the ratio is pinned at exactly 1.

## outputs

under `analysis/results/tbx5/`, one directory per code directory.

### demultiplexing/

- `counts.npz`, `histogram_f1000.csv`: the SNP bin caches. delete to rebuild.
- `demux.parquet`: per cell with a SNP profile: `snp_umi`, `embryo`, `partner`, `singlet_post`, `score`, `EM_embryo_call`.
- `cull.csv`: cells left after each step and at each threshold in `POSTERIORS`.
- `fit.csv`: one row per pass: loglik, BIC, iterations, bins, the fitted dispersion and concentration, the non-singlet share, and how many embryo components hold over 1% of the weight. the logliks drop `log C(n, a)`, so the `coarse` row has fewer features and is not comparable to the other two.
- `components.csv`: one row per embryo component, sorted by `geno_radius`.
- `snp_umap.parquet`, `snp_umap.png`: the layout, and the umap by called embryo with the thrown-away cells in grey beside the score distribution. a few thousand cells embed far out, so the panel is windowed to the `PCT` to `100 - PCT` percentile range and prints how many that leaves off.
- `nn_doublets.parquet`: per cell: `snp_singlet`, `dbl_frac`, `thresh`, `nn_doublet`, `keep`, `EM_NN_embryo_call`. `nn_doublets.csv` is the pool summary.

`../data/2608_crispants/tbx5.h5ad` gains `EM_embryo_call` and `EM_NN_embryo_call`.

### support_comparison/

- `latent_{crispant,devmap}.npy`, `obs_*.parquet`, `elbo.csv`, `model/`, `train_log/`: the fit.
- `subtype_crispant.parquet`: the transferred subtypes.
- `predicted_leaves.parquet`: per predicted leaf: `tree`, `node`, `cell` (an E9.5 devmap cell), `frozen`.
- `umap_integrated.parquet`: the integrated layout.
- `per_embryo_<scope>.csv`: one row per reference cell: `embryo`, `subtype`, `cell_type`, `predicted`, `measured` (the mean over the crispant embryos), `spread` (their sd), and one `measured_<embryo>` column each. all of them centred log ratios, so 0 is the scope's own median and not a 1:1 distance ratio.
- `per_embryo_<scope>_subtype.csv`: per subtype: `measured` with `measured_sem`/`_min`/`_max`/`_pos`/`_p` over the embryos, `predicted`, and `predicted_jk_se` where more than one jackknife tree ran.
- `per_embryo_<scope>_offset.csv`: per column of the per cell table, the median that was subtracted: `predicted`, and one `measured_<embryo>` row each. adding it back gives the uncentred log ratio, where 0 is a 1:1 distance ratio.
- `per_embryo_support.csv`: per scope the cells on each side, the units, the trees, `n`, `n_auto`, `n_jack`, and how many jackknife trees ran. a scope under `MIN_DRAW`, or one where some reference embryo holds no predicted cell, is listed with `n` 0 and writes no csv.

## running it

one gpu and the `rapids_singlecell` env, from the repo root:

```sh
python -m analysis.pipelines.tbx5.demultiplexing.preprocess_counts
python -m analysis.pipelines.tbx5.demultiplexing.snp_demux
python -m analysis.pipelines.tbx5.demultiplexing.transcriptional_doublet_detection

python -m analysis.pipelines.tbx5.support_comparison.fit_scvi
python -m analysis.pipelines.tbx5.support_comparison.labels
python -m analysis.pipelines.tbx5.support_comparison.predicted
python -m analysis.pipelines.tbx5.support_comparison.integrated_umap
python -m analysis.pipelines.tbx5.support_comparison.per_embryo_nn_distortion
```

`aggregate.py` is not in the list: its two caches are prebuilt, and `counts()` builds on demand if `counts.npz` is missing. `--scope heart --n 86` reruns the heart alone at the draw size its jackknife needs.

## knobs

demultiplexing: `N_BINS` and `FINE` (the SNP feature resolution, baked into `counts.npz`), the load and fit qc cutoffs, `MIN_SNP_UMI`, `N_PCS`, `DOUBLET_SCORE`, `M`, `K_NN`, `K_SNP`, `CENTRAL`.

support comparison: `N_REF`, `DOWNSAMPLE_REF`, the scvi hyperparameters, `K_LABEL`, `N_MAX`, `MIN_DRAW`, `SKIP_LINEAGES`, `HEART_TYPES`, `CALL`, which is the demultiplexing column the singlets are taken from, `SCREEN`, which tree set is read, `OFF_RATIO` / `MODE`, the off-manifold cutoff and convention, `N_UMAP` / `K_UMAP` (baked into the cached layout), and `DRAWS` / `MIN_DRAW` / `--n` on the per embryo pair.
