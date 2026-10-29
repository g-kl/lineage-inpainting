# tbx5 supplement

supporting panels for the tbx5 figure (`analysis/figures/tbx5/`).

- **qc of the crispant dataset**
- **expression alone vs. the model** as predictors of the measured nn distortion. this tests whether the model predicts it better than Tbx5 expression.
- **lineage age**: whether a crispant lineage looks younger than its E9.5 reference, and the neural ectoderm, where the differentiated neurons are depleted.

row 1: qc.
- (a) the SNP demultiplexing umap, kept cells by embryo, discarded cells grey, small SNP UMAP1 / 2 bars in the corner.
- (b) per embryo: cells recovered, UMIs and genes per recovered cell.

row 2: expression vs. the model, and age.
- (c) raw Tbx5 in each heart tube reference cell against its measured log nn distortion. the tbx5 figure's panel l with the model swapped for expression.
- (d) r with measured for Tbx5 expression and the model, paired bars per crispant embryo and for the mean over all of them.
- (e) one point per lineage: the energy distance from its crispant cells to E9.0 devmap cells of that lineage against the one to E9.5, log axes, the diagonal where the two are equal. above the diagonal a lineage sits nearer E9.0.

row 3: the neural ectoderm.
- (f) NE-only umap by cell type, the progenitors as one grey (other NE), the neuron types a hue each.
- (g) the same map by dataset.
- (h) neurons as a share of NE per embryo, reference against crispant.

# implementation

compute writes small caches into `numerical_results/`, the plot scripts and `assemble.py` read only those and write into `imgs/` (both gitignored). the scripts are run with `-m` from the repo root. `embed_stages.py` and `compute_umap_ne.py` need the gpu and run under `rapids_singlecell`; everything else runs under `lt` (`compute_age.py` puts the energy distance on the gpu if there is one).

`style.py`: paths, the panel constants, the palette, 10 / 9 pt type with e's lineage names at 8 pt, `figure`, `save_svg`. the tbx5 figure's `config.py` comes along by package path for its names (`TARGET`, `PRED`, `MEAS`, the heart scope), its subtype colours (c matches the main panel l) and its `key` and `window` helpers. e places its lineage names by hand (`plot_age.NAME_AT`), with a hairline for any set off far from its point.
- a and b: one hue per crispant embryo (`EMBRYO`).
- d: the model is the shared `P.MODEL` teal and expression the MGI supplement's ochre.
- f: `NE_OTHER` + `NE_NEURON`, chosen to stay distinguishable (including under colour-vision deficiency).
- g and h: `DATASET`, one hue per side.

## compute -> numerical_results/

everything but the demultiplexing is read through `analysis/pipelines/tbx5/support_comparison/common.py`: the demultiplexed crispants, the integrated latent and cells, the per-embryo nn distortion tables, the devmap and its annotation, and the saved scvi model.

- `compute_qc.py` -> `snp_umap.parquet`, `qc_cells.parquet` (a, b). the kept cells are the support comparison's crispants. depth is over the whole transcriptome, from the `tbx5.h5ad` obs. the SNP umap is `analysis/results/tbx5/demultiplexing/snp_umap.parquet`.
- `compute_expression.py` -> `expression_cells.csv`, `expression_r.csv` (c, d). the heart tube cells of the pipeline's heart scope, raw devmap Tbx5 (not smoothed, since the model is not smoothed either), correlated with the mean measured distortion and with each crispant embryo's own column.
- `embed_stages.py` -> `stage_latent_{stage}.npy`, `stage_obs_{stage}.parquet` (e). `N_STAGE` devmap cells per stage in `AGE_STAGES`, thinned to crispant depth and put through the support comparison's scvi encoder. the E9.5 cells are ones the fit did not draw, so every index cell is new to the encoder.
- `compute_age.py` -> `age_energy.csv` (e). a lineage is aged if it has `MIN_INDEX` cells at both stages and in the crispant, blood and the extraembryonic lineages left out. per lineage, a stage-balanced index of up to `N_INDEX` cells per stage, and the crispant cells pooled over the embryos (at most `N_QUERY`) get the U-statistic energy distance (`analysis/lib/metrics/energy_distance.py`) to each stage.
- `compute_umap_ne.py` -> `umap_ne.parquet`, `ne_counts.csv` (f, g, h). a cuml umap of the integrated latent over NE cells, the same number from each dataset. neurons are the annotation's `neuron` cluster.

## plot + assemble -> imgs/

`plot_qc.py` -> a, b, `plot_expression.py` -> c, d, `plot_age.py` -> e, `plot_ne.py` -> f, g, h, each into `imgs/panels/`. `assemble.py` renders them all and nests them into `imgs/supp_tbx5.{svg,png,pdf}` through `utils.svg_assembly`: three rows, each filling the page width.

## run order

    PY=python
    GPU=python  # an env with rapids-singlecell (see README)
    $PY -m analysis.figures.supp_tbx5.compute_qc
    $PY -m analysis.figures.supp_tbx5.compute_expression
    $GPU -m analysis.figures.supp_tbx5.embed_stages
    $PY -m analysis.figures.supp_tbx5.compute_age
    $GPU -m analysis.figures.supp_tbx5.compute_umap_ne
    $PY -m analysis.figures.supp_tbx5.assemble

the support comparison pipeline has to have run first (`analysis/pipelines/tbx5/support_comparison/`). only assemble needs rerunning for cosmetics.
