# concept

the perturb and tbx5 figures on one sheet, 6.5 in wide. the ring on top, then three rows of four small panels.

- row 0, the perturbation map: (a) the ring UMAP and its key (perturb a), drawn smaller: smaller insets on a tighter ring around a smaller central map, and a smaller reference UMAP in the key.
- row 1, the screen checks: (b) the MGI box and (c) the gnomad box (perturb b, c), then (d) heart vs total shift and (e) shift specificity vs shift time over the CHD panel (tbx5 a, b). tbx5's haploinsufficiency and dose panels (its c, d) are left out.
- row 2, the tbx5 knockout inside the model (tbx5 e-h): (f) the shift curves, (g) the Tbx5 subtree, (h) fate split vs Tbx5 onset, (i) the predicted distortion along the A-P program.
- row 3, the crispants against the prediction (tbx5 i-l): (j) the integrated map, (k) the heart tube map, (l) predicted and measured distortion, (m) predicted vs measured.

see `../perturb/concept.md` and `../tbx5/concept.md` for what each panel computes.

# implementation

this folder has no compute. it reads the other two folders' outputs and only re-plots what changes on this sheet.

- `style.py`: this sheet's geometry. it re-exports the tbx5 config's type, ink, page pad, letter box and inch-axes `figure`, and the perturb style's box palettes and its `numerical_results/` path.
- `plot_ring.py` -> `imgs/panels/{ring,refkey}.svg` (a). perturb's ring, re-drawn from `perturb/numerical_results/{perturb_umap.csv,perturb_umap_panels.pkl}` with `perturb/plot_ring.py`'s helpers and pixel scale, so strokes and text keep their size.
  - the slots sit at 15 + 30k degrees, so neighbouring insets are never on a diagonal of each other, and the radius `R` is the smallest that keeps them `GAP` apart. the central map side `C` is set so the map clears the insets at 45 degrees.
  - connectors are perturb's: from the inset edge towards the point, a straight leg, then a 45 degree run.
  - slots are assigned with perturb's rule: reps that deplete the same coarse type are adjacent. then the group order, direction and ring offset are searched for the fewest connector crossings plus connectors through another inset (the insets are transparent), then for the shortest connectors.
  - the key is perturb's column at a smaller reference UMAP, as wide as its widest line of text.
- `plot_boxes.py` -> `imgs/panels/{mgi_box,gnomad_box}.svg` (b, c). perturb's two boxes, re-drawn from `perturb/numerical_results/{mgi_box.csv,mgi_box.npz,gnomad_box.csv}` at the width row 1 leaves them, in the tbx5 type so they match d and e. the groups and labels are imported from `perturb/plot_mgi_box.py` and `perturb/plot_gnomad_box.py`. the n line drops its parentheses to fit.
- `assemble.py` -> `imgs/figure.{svg,png,pdf}`.
  - renders a, b and c, then nests the tbx5 panels as already rendered (`tbx5/imgs/panels/r{1,2,3}_*.svg`).
  - each panel goes in at the size its svg declares, and the whole sheet is scaled into tbx5's page pad. the ring and its key are centred as a pair, with letter a in the ring's empty top left corner.
  - a missing tbx5 svg fails with the name of the script that writes it.

so every tbx5 panel only changes when the tbx5 folder re-renders it.

## run order

    PY=python
    GPU=python  # an env with rapids-singlecell (see README)

    # perturb: the ring and box caches
    $PY  -m analysis.figures.perturb.compute_mgi_box
    $PY  -m analysis.figures.perturb.compute_gnomad_box
    $GPU -m analysis.figures.perturb.compute_ring

    # tbx5: its assemble renders every panel into tbx5/imgs/panels/
    cd analysis/figures/tbx5
    $GPU compute_tbx5.py
    $GPU compute_umap.py
    $PY  compute_metrics.py
    $PY  compute_dose.py         # not on this sheet, but tbx5's assemble renders it
    $PY  compute_curves.py       # after compute_tbx5
    $PY  compute_subtree.py      # after compute_tbx5
    $PY  compute_distortion.py
    $PY  compute_program.py      # after compute_distortion
    $PY  assemble.py
    cd -

    # this sheet
    $PY  -m analysis.figures.perturb_tbx5_combined.assemble

for cosmetics, rerun only the last line, after `tbx5/assemble.py` if those panels changed. `imgs/` is regenerable and not committed.
