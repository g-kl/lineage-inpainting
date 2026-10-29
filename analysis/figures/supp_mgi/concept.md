# perturb supplement

supporting analysis for figure 5.

- (a) model depletion correlates with, but is not determined by, where a gene is expressed. every MGI pair, the gene's E9.0 mean expression in the cell type (log1p CP10K) against the model's depletion of it, the axis cut at [-1, 1].
- (b) examples of genes with low expression but a high perturbation effect. six MGI pairs from the knockout's lowest expression third, each at its rank among the knockout's cell types by E9.0 expression and by the model. the rule picking them is in `style.py`: TFs only, no wipe-out knockouts, the biggest rank gain first.
- (c) control subtrees showing where each example's knockout reaches the cell type's trajectories. one control subtree for each of three of b's examples, time along x, each edge hued by the gene's expression in the node's snapped cell, on its own scale per tree, each leaf by its cell type.

### readout protocol

a and b read depletion under the Tbx5 figure's protocol, freeze at off-manifold ratio 1.1.

# implementation

compute writes small caches into `numerical_results/`, the plot scripts and `assemble.py` read only those and write into `imgs/` (both gitignored). the scripts are run with `-m` from the repo root. everything runs under `lt` except `compute_pairs.py`, which needs cuml's knn for the freeze snap and runs under `rapids_singlecell`, like the Tbx5 pipeline's `predicted.py`.

`style.py`: paths, the protocol, the panel b and c constants, the palette, 10 pt type with b's second x label line at 9 pt, `figure`, `save_svg`. the screen paths and the TF list come from the perturb figure's `style.py`, so both read the same screen. the protocol, `MODE, OFF_RATIO = "freeze", 1.1`, is the Tbx5 figure's, not the perturb figure's prune. fonts are the shared ones from `utils.figure_labels`.
- a is ink.
- b's model is the shared `P.MODEL` teal and expression an ochre.
- c's edges are the shared expression colormap (`P.EXPR_CMAP`, viridis), one colour bar per tree. its leaves are one hue per example cell type and grey for any other, three hues off viridis chosen to stay distinguishable from each other and the grey (including under colour-vision deficiency).

## compute -> numerical_results/

- `readout.py`, used by `compute_pairs.py`: the Tbx5 support comparison's `predicted.readout` + `to_e95`, read at E9.0.
  - a readout leaf is a meta row within `RING_TOL` of E9.0. it is off the manifold if `perturb_result.off_manifold_descendants` at the ratio has it.
  - under freeze an off leaf takes `predicted.last_on_manifold`'s snapped cell, and a cell not named E9.0 is carried to its nearest E9.0 devmap cell (`common.knn`, brute euclidean on the gpu) in the devmap fit's own `X_scvi` (`common.devmap_latent`).
- `compute_expression.py` -> `expression.parquet` (cell type x gene), `expression_n.csv`.
  - mean log1p CP10K (the h5ad's X) over the E9.0 reference cells, typed with the same cell -> type map the screen readout uses.
  - X is streamed in row chunks through h5py.
- `compute_pairs.py` -> `pairs.csv`, `pairs.md`. the association set and `MIN_CTRL` come from the perturb figure's `compute_mgi_box.py` (its functions, not its outputs). the control and per-gene counts are `readout.py`'s under `style.MODE` at `style.OFF_RATIO`. depletion is 1 - perturbed / control.
- `compute_examples.py` -> `examples.csv` (b), `examples.md`. reads `pairs.csv`.
- `compute_subtrees.py` -> `subtrees.csv` (c), `subtrees.md`. reads the control trees and the h5ad's X row by row for the snapped cells it needs, on the cpu.
  - the examples are `style.SUBTREE_EXAMPLES`, three of b's six.
  - a knockout changes a node when its snapped cell expresses the gene, and up to its first change a knockout trajectory is the control one. so the control trees alone show where a knockout reaches a trajectory.
  - a control tree is a forest of rooted subtrees, one per start cell. each is scored by its readout leaves of the example's cell type with the gene on at a node above them, among those with at most `MAX_LEAVES` readout leaves. every other leaf is pruned away, with the internal nodes left holding none.

## plot + assemble -> imgs/

`plot_scatter.py` -> a, `plot_examples.py` -> b, `plot_subtrees.py` -> c, each into `imgs/panels/`. `assemble.py` renders all three and nests them into `imgs/supp_mgi.{svg,pdf,png}` through `utils.svg_assembly`: a | b, then c, 6.5 in wide.

## run order

    PY=python
    GPU=python  # an env with rapids-singlecell (see README)
    $PY -m analysis.figures.supp_mgi.compute_expression
    $GPU -m analysis.figures.supp_mgi.compute_pairs
    $PY -m analysis.figures.supp_mgi.compute_examples
    $PY -m analysis.figures.supp_mgi.compute_subtrees
    $PY -m analysis.figures.supp_mgi.assemble

only assemble needs rerunning for cosmetics.
