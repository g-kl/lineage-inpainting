# Core figures

This directory contains the manuscript figures:

1. `06_methodology/` is the methodology-first schematic: destructive lineage
   records, the absent full-tree target, distributional constraints, tree-aware
   evidence routing, joint completion and conditional-sampling training, and
   conditional sampling from context masks.
2. `05_lineage_completion/` is the overview: observed destructive snapshots,
   the unobservable filled tree, learned conditional lineage inpainting, and the
   downstream analyses it enables.
3. `04_problem_setup/` sets up the completion-and-sampling framework: observed
   lineage records, completed-record draws, and generation-masked sampler
   training. `problem_setup_approach2` lays the problem out in four rows: (a)
   the single-snapshot observed data; (b) the modeling objective, a generative
   model that samples complete trees from an observed snapshot; (c) model
   fitting: impute, then mask everything but one snapshot and resample; and (d)
   the model implementation, the two pieces the model is built from, each named
   on the panel-letter line: stochastic interpolants over complete trees (the
   stacked-plane bridge of `03_inpainting`) on the left and the biased attention
   layer encoding lineage structure (`02_attention_module`) on the right. The two
   names set the column widths there -- at the 10 pt floor they nearly fill the
   row -- so each drawing is sized to sit inside its own column. Both are redrawn
   here rather than inlined: same composition, and 03's projection maths is
   imported so the planes match, but in this figure's palette, node vocabulary
   and type (CM serif), with the same titled group box the computed figures use
   (`utils.figure_labels`).
4. `01_slice_aligner_loss/` shows an observed tree generation and the aligner-loss construction.
5. `02_attention_module/` shows the tree-aware attention module.
6. `03_inpainting/` shows tree completion by inpainting.

Each figure directory contains the preview PNG, editable SVG, and source/generation files for that figure.

Regenerate them from the repository root with:

```sh
python3 cartoons/06_methodology/make_methodology_figure.py
python3 cartoons/04_problem_setup/export_problem_setup_svg.py
python3 cartoons/04_problem_setup/make_problem_setup_approach2_figure.py
python3 cartoons/05_lineage_completion/make_lineage_completion_figure.py
python3 cartoons/01_slice_aligner_loss/export_slice_feature_svg.py
python3 cartoons/02_attention_module/export_tree_attention_box_svg.py
python3 cartoons/03_inpainting/make_stacked_bridge_inpainting_figure.py
```

The problem-setup, slice-feature, and attention figures use their HTML files as the source
artwork and the export scripts to write SVG/PNG outputs. The inpainting,
lineage-completion, and methodology figures are generated directly from their
Python scripts, which write the SVG, PNG, and HTML preview.
