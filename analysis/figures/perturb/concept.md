# concept

the perturbation-map figure. one paper-ready panel (imgs/figure.svg / .png), 6.5 x 6.5in, fits an a4 page. two rows: the ring UMAP + its key on top (a), the two validation boxes below (b, c), sized small and centred as a pair.

- panel a -- the ring: one point per knockout in the genome-wide E9.0 HVG screen, placed by its perturbation delta and hued by the lineage it depletes most. representative TF knockouts ring the map, each a small embryo UMAP hued by nn distortion: how much farther a reference cell sits from that knockout's leaves than from a plain reference draw, i.e. where the knockout thinned the manifold. the key column on the right holds the reference UMAP, the most-perturbed-coarse-type legend, and the distortion colorbar.
- panel b -- MGI box: does a knockout perturb the cell type its curated MGI phenotype points at? perturbed fraction (1 - the fraction of the cell type's control leaves the knockout left unperturbed, i.e. on the manifold and at the same cell) over annotated gene->cell-type pairs, against a random-pair null, then broken out by coarse anatomical context.
- panel c -- gnomad box: perturbation effect size for the most vs least LoF-constrained screen genes (gnomad LOEUF). constrained genes are expected to have larger effects.

all three panels read the same screen and share its control set. a leaf that leaves the manifold is dropped (`prune`), which is the screen's own convention.

# implementation

compute is split from plotting. compute_*.py write small caches into numerical_results/; plot_*.py + assemble.py read only those caches and write to imgs/. cosmetic tweaks never touch the heavy data.

style.py holds everything shared: screen + dataset paths, the ring's compute knobs, the palettes and font, and the assembly geometry (panel widths, gaps). apply_rc() saves text as vector paths, so the svg needs no fonts installed.

## compute -> numerical_results/

`readout.py` (shared, no output) -- the tree walk, off-manifold pruning and cell-type map that panels b and c are both scored with.

`compute_ring.py` -> perturb_umap.csv/.npz + perturb_umap_panels.pkl (panel a). per knockout: mean scVI embedding of its E9.0 leaves minus control = a 50-d delta; UMAP the deltas, tag each gene with the lineage it depletes most. then picks the ring's representatives (the most lineage-specific TFs per lineage in `RING_SHOW_LINEAGES`) and scores each one's nn distortion over the reference cells. the cache stores raw per-cell values, so a palette or colour-limit change needs no recompute; changing which lineages are shown does. reads obs + X_scvi out of the h5ad through h5py, never the counts. the leaf/lineage helpers come from the perturb pipeline and the distortion helpers from the tbx5 support comparison, so neither can drift.

`compute_mgi_box.py` -> mgi_box.csv/.npz (+ mgi_box_medians.md) (panel b). loads the MGI associations (MP edges + phenotype map) restricted to screened genes, scores each by that KO's perturbed fraction of that cell type, plus a matched random-gene null.

`compute_gnomad_box.py` -> gnomad_box.csv (+ gnomad_box_medians.md) (panel c). bins the screen genes by LOEUF and scores each by effect_norm. the mann-whitney p goes to the md file, not onto the panel.

b and c are dataset-free and take seconds. only compute_ring is heavy (gene loop + kNN, wants a GPU).

## plot + assemble -> imgs/

`plot_ring.py` -> ring.svg + refkey.svg. central map, representative markers with subway connectors, ring insets rasterised from the cached per-cell values. reps depleting the same cell type sit in adjacent slots, and connectors are untangled by a 2-opt pass. one symmetric colour limit shared by every inset, so a hue means the same thing everywhere. refkey.svg is the key column.

`plot_mgi_box.py` -> mgi_box.svg, `plot_gnomad_box.py` -> gnomad_box.svg. box + jittered points.

`assemble.py` -> figure.svg (+ .png). nests the four pieces into one parent svg at the sizes and gaps in style.py, strips the ring's white background so its bottom whitespace can slide over the box row, and drops the a / b / c letters on last.

## data sources

- the screen (all panels): outputs/<experiment>/hvg_projection_screen_v2_dose0_E9.0_R1
- reference cells (panel a): qiu24_integrated_hvgs_and_scvi.h5ad obs + X_scvi, UMAP coords from final_umap.csv
- leaf/lineage + off-manifold helpers: analysis/pipelines/perturb/
- nn distortion: analysis/pipelines/tbx5/support_comparison/ -- the same statistic the tbx5 figure uses, read here off the screen rather than off crispants
- TF list, gnomad LOEUF: literature/
- MGI: mouse_phenotypes_gene_attribute_edges.txt + literature/phenotype_to_celltype_mapping.csv

## run order (lt conda env, except compute_ring)

    python -m analysis.figures.perturb.compute_mgi_box
    python -m analysis.figures.perturb.compute_gnomad_box
    python -m analysis.figures.perturb.compute_ring        # rapids_singlecell env, ~30min
    python -m analysis.figures.perturb.plot_ring
    python -m analysis.figures.perturb.plot_mgi_box
    python -m analysis.figures.perturb.plot_gnomad_box
    python -m analysis.figures.perturb.assemble

re-run only the plot_* + assemble steps for cosmetics.
