# concept (synthetic figure)

synthetic benchmarks split into two categories: non-least-action, and context-dependence. each has a synthetic and a semi-synthetic

- row a: coherent dynamics. true and reconstructed by different methods. "fate accuracy" refers to 1 - TV between the predicted and true sibling fate histograms, where fates called by nearest attractor on the latent axis. scored only on the 7 -> 8 transition in the main text, the rest in supp.
- row b: the same for the crossing variant.
- row c: the real-cell crossing landscape. on top, the cartoons of what cell type motifs are and how we classify them. clustermap shows fate accuracy computed the same way as in panels a/b (1 - TV after assigning nearest fate). label as OT-like if the OT coupling is correct, Fels-like if the felsenstein coupling is correct.
- row d: context (fate-bias) reconstruction. true dynamics vs. our model predictions at context size 1 and context size 8.
- row e: context experiment metrics: fate accuracy (fraction of predicted leaves nearer the true attractor than the other) vs context size n, one panel per sigma.
- row f: context semisynthetic: (left) a cartoon illustrating the two motif construction (mirroring style of panel c left, and set at the same height as the heatmaps beside it), followed by three tight heatmaps of motif vs. method: one for average ct accuracy, one for "within type transition" ct accuracy (non-ambiguous) and one for "between type transition" ct accuracy. the metric is progenitor-type accuracy on a 0-1 scale (did the imputed ancestor land nearest the *correct* progenitor-type centroid; 0.5 = chance) over 10 random real cell-type draws, one column per draw.

# implementation

compute is split from plotting. compute_*.py distill the upstream heavy eval outputs into small caches in numerical_results/; plot_*.py + assemble.py read only those caches and write into imgs/ (gitignored). max width 6.5 in, Latin Modern Sans (shared utils/fonts/, mathtext cm), nothing below 6 pt. shared paths / palette / geometry / rc live in style.py.

## reused upstream code (imported, not copied)
- `synthetic_evals.recon_figure` -- `_plot_dynamics`, `_recon_panel`, `_branch_rows` (crossing recon drawers).
- `synthetic_evals.context.plot` -- `_layout`, `_truth_panel`, `_pred_panel`, `_glyph_handles`, `_model_scaling`, `_baseline_level` (context recon drawers + metric aggregation).
- `synthetic_evals.viz` -- the method palette (`MODEL`/`FM`/`GS`/`GF`), `TRUTH`, `fate_color`, event colours, re-exported through style.py so every panel agrees (same method == same colour).
- `utils.figure_labels` -- `register_fonts`/`FONT_FAMILY`, `panel_letter_svg` (panel letters), and `box_title_svg` + `box_border_d` (the titled rounded group boxes, shared with `analysis.figures.summary_qc.assemble`).
- `utils.svg_assembly` -- `prefix_ids` / `nest_panel` / `write_master`: the mechanics of nesting panel svgs into one master and emitting svg+pdf+png. shared with the supplement figures.
- `glyphs.py` (local) -- the canvas helpers (`rect`, `assert_within_page`, unit `U`/`W`) and the motif-cartoon primitives (`curve`, `brace_h`, `parents`, `leaf_xy`, `edges`, `node_text`, `tint`) that rows c and f both draw with.

## compute scripts -> caches
`compute_crossing.py` -> `crossing_recon.pkl`, `crossing_fate.csv`. reads high_dim_crossing/results_<exp>_d*.pkl. writes the 1-d-projected forward recon data for rows a/b (RECON_DIM=1) and fate accuracy per variant x method x dim (1/4/16/64) x transition (mean over parents).

`compute_context.py` -> `context_recon.pkl`, `context_scaling.csv`. reads context/results_<exp>.pkl. re-pickles the case + cartoon (+ times) needed for the recon row d, and writes the pool-split fate-accuracy scaling curve per sigma (mean/sem vs n) plus the n-independent baseline levels for the metric strip e.

`compute_landscape.py` -> `landscape_heat.csv`, `landscape_bars.csv`, `landscape_examples.csv`, `landscape_meta.csv`. reads crossing_landscape/{candidates.csv,results.pkl}. thresholds both coherence axes at top-fractions of the full candidate landscape (style.OT_TOP_FRAC=1/6, FELS_TOP_FRAC=1/2; both thresholds land at coherence ~0, i.e. "that coupling is correct"), flags every evaluated motif OT-like / Felsenstein-like (OT-like implies Felsenstein-like, so with "neither" they partition the motifs into style.CATEGORIES), scores each method's stored forward samples by child-type accuracy (`_type_acc`; the true child type of every eval row is known from the dataset's row order, mA anchor / mA inner / mB anchor / mB inner, which it asserts), and writes: the per-motif accuracies clipped below at style.ACC_VMIN in ward/optimal-leaf-order (heat), the per-category mean/sem/n of the *unclipped* accuracies for the caption text (bars, not plotted), one representative motif per category, and the motif/candidate counts. the representative motifs come from one six-type set that candidates.csv rewires into all three categories (picked so the weakest category sits furthest from its threshold).

`compute_context_realcells.py` -> `context_realcells.csv`. reads context_landscape/{landscape,samples}.csv. keeps the canonical method order plus the `oracle` / `marg` reference rows, and joins the motif's drawn cell types back on. one row per motif x method.

## data sources
- rows a/b + crossing metric strip: `synthetic_evals/high_dim_crossing/results_unshared_vpsde_idnn_leaf_d{1,4,16,64}.pkl` (from `synthetic_evals.eval`).
- rows d/e (context): `synthetic_evals/context/results_synbench_fb_unshared_vpsde_idnn_leaf.pkl` (from the context benchmark).
- row c (landscape): `synthetic_evals/crossing_landscape/{candidates.csv,selected.csv,results.pkl}` (the pkl keeps every method's forward samples; from `synthetic_evals.crossing_landscape_eval`, which enumerates all 56,700 candidate motifs, draws `--n-select` of them uniformly at `--select-seed`, records the draw in selected.csv, then trains + evaluates each).
- row f (real-cell context): `synthetic_evals/context_landscape/{landscape,samples}.csv` (from `synthetic_evals.context_landscape_eval`, which trains one pool/split + the baselines per sampled motif over `datasets.cell_type_fate.CellTypeFateDataset`; the per-motif dataset configs it generates live in `config/dataset/context_landscape/`).

## plotting + assembly (read only numerical_results/)
`plot_crossing.py` -> recon rows a/b, each paired with its variant's fate-vs-d line plot (panel svgs in imgs/). recon rows reuse the recon_figure drawers; the line plot reads the style.CROSS_SPLIT_PAIR rows of crossing_fate.csv.
`plot_context.py` -> the glyph key, the context recon row d, the context metric strip e. the recon panel reuses context.plot's drawers; the metric drawer reads context_scaling.csv.
`plot_landscape.py` -> the whole row c: the construction cartoon and the three example motifs on top, then the two annotation rows and the 4 x 32 clustermap, with the colorbar and yes/no key in the right gutter. everything is positioned on one equal-aspect y-down canvas whose unit is 0.01 in (so the problem-setup `curve()` path math ports over verbatim and the image axes can be placed in the same coordinates); the per-cell-type palette (neural teal / cranial-mesenchyme purple) and the name abbreviations live here. the (OT likeness, Felsenstein likeness) plane version of this experiment is drawn standalone by `synthetic_evals/crossing_landscape_fancy_plot.py`.
`plot_context_realcells.py` -> the whole row f: the two-motif cartoon on the left, then the three accuracy heatmaps (motif x method) with a shared colorbar in the right gutter. same one-canvas trick as row c (unit 0.01 in, equal aspect, y-down) and it imports the same path primitives as row c from `glyphs.py` (`curve`, `node_text`, `parents`, `leaf_xy`, `edges`) so both constructions are drawn in one visual language; the letter palette (`glyphs.LETTER_COLORS`) is shared with row c's construction cartoon, so the same letter is the same colour in both rows. the accuracy colormap is `style.ACC_CMAP` (shared with row c) = the gist_earth ramp, flipped so tan = good / dark = bad, and cut short of its white end (`utils.palettes.truncate`) since accuracy hits 1 exactly. it spans chance to perfect (`style.ACC_VMIN/ACC_VMAX` = 0.5 to 1) and clips below at chance, with a bottom triangle reading <=0.5 -- below 0.5 a method is just uninformative, so the scale does not resolve it.
`assemble.py` -> `imgs/synthetic_figure.{svg,pdf,png}`. re-renders every panel, then nests them as positioned <svg> viewports in one master svg laid out in points (per-panel id prefixing so glyph/clip defs can't collide). page 6.5 in wide (fits A4 portrait). the rows are grouped into the two titled boxes (`style.BOX_TITLES`, geometry in `style.BOX_*`): each row is rendered at full page width and scaled down slightly so it clears the border, and each box title is a tight-cropped svg centred on the break in its top edge. `style.PAGE_PAD_IN` keeps the box strokes off the page edges. bold lowercase panel letters a-f (one per row, at each row's top-left; `utils.figure_labels.panel_letter_svg`) are placed last -- the category row's and the real-cell context row's letters are lifted into the gap above them since those rows are left-flush.

## run order
    python -m synthetic_evals.figure.compute_crossing
    python -m synthetic_evals.figure.compute_context
    python -m synthetic_evals.figure.compute_landscape
    python -m synthetic_evals.figure.compute_context_realcells
    python -m synthetic_evals.figure.assemble
