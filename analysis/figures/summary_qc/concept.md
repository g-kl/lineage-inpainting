# concept (fig 3)

- row 1: cell type transition map
- row 2: metric lollipop charts comparing lineage inpainting to GENOT-Sinkhorn and GENOT-cbFGW
    - clonal distortion (cycle consistency)
    - marginal distortion (energy distance between marginals)
    - lineage oscillation (computed from assignments)
    - linkage distortion (computed from assignments)
    - germ layer crossing (computed from assignments)
    - Hox-code distortion (ancestor/descendant Hox correlation from assignments)
- row 3: context scaling (same metric, energy distance vs. revealed context, both directions of inference). two boxes: left = ancestral (panels c, d), right = descendant (panels e, f); each box pairs an average panel with a per-type panel.
    - panel c (ancestral, average): cycle-consistency improvement with larger leaf context (leaf -> root -> leaf), against context size = leaf_time - t_MRCA of the revealed clade
    - panel e (descendant, average): same as c, but forward -- predict a focal node v's descendants by splitting DOWN from a complete cross-section layer above it. the layer crosses v's lineage at one node f_v and the rest are v's same-level cousins; the x-axis is how many cousins the layer reveals. energy distance falls as more cousin context is added, and the layer's position relative to v is ~flat across that range, so the gain is the added context, not proximity.
    - panels d, f: the same as c, e resolved BY CELL TYPE -- every terminal fate as a grey wispy per-type trend, with the two fates benefiting most/least from context highlighted. rare fates are excluded (`CONTEXT_MIN_LEAVES`): both samplers draw a fixed ~100 focals per fate, so row count is flat by construction and a fate with 371 leaves in the tree set would otherwise be drawn as confidently as one with 112,036. blood and extraembryonic fates (allantois included) are left out too (`CONTEXT_EXCLUDE_TYPES`). a single horizontal method legend runs below all four columns (the average panels c and e share the method trio).

# implementation

## compute scripts -> caches

`compute_transition_map.py` -> `numerical_results/transition_map.pkl` (row 1). Wraps `analysis.lib.celltype_map.eval.build_cell_type_map` on the model coupling over `N_TREES` trees: takes every transition pair within the engine's nn snap cutoff (q0.8), with no pmi cut (`PMI_QUANTILE` = 0) and no subsample, so the map is deterministic. It keeps a (cell_type, time) edge when it carries >= 5 cells and its share of the target's incoming cells plus its share of the source's outgoing cells is >= `MIN_COMPOSITION` = 0.2 (`COMPOSITION` = "sum"). Nodes are labelled at `RESOLUTION_BY_TIME`: germ layer at E7.5, lineage at E8.0, cell type after. E7.5 internal nodes snap to E8.0 cells, so at lineage resolution most E7.5 nodes have no E7.5 cell behind them.

`compute_assignment_metrics.py` -> `numerical_results/assignment_metrics.csv` (row 2: oscillation, linkage, germ). Recomputes the full-coupling `analysis/lib/coupling/eval` metrics for all three methods from the saved couplings: germ-layer crossing, lineage oscillation fraction, and linkage distortion (= |1 - normalised linkage correlation|, plus the raw normalised value). Columns: method, metric, tag, value.

`compute_distortion.py` -> `numerical_results/{clonal,marginal}_distortion.csv` (row 2: clonal, marginal). Scores clonal + marginal distortion on a random subset of subtrees, identically for all three methods.
- clonal distortion: sample `N_SUBTREE_LEAVES` (leaf, ancestor at leaf_time - `CLONAL_OFFSET`, descendant-set) tuples (`CLONAL_TIME_TOL` time match); round-trip leaf set -> ancestor -> leaf set (`SPLIT_STEPS` rollout) and take the energy distance to the true descendant cloud. Model uses the splitter for both halves (anchor leaves -> predict root = pool; anchor the predicted root, mask leaves, regenerate = split); GENOT round-trips the focal leaf `N_GENOT_SAMPLES` times via `analysis.lib.metrics.clonal_preservation.clonal_preservation_genot`.
- marginal distortion: per adjacent timepoint pair (`MARGINAL_TIME_TOL`), push `N_MARGINAL_CELLS` t_late cells back and take the energy distance to the true t_early marginal. Model = splitter pool-back (anchor the cell's subtree, read the root state at t_early); GENOT = flow transport.
- both distortions score each cloud pair with the unbiased U-statistic energy distance (`analysis/lib/metrics/clonal_preservation._energy_distance`)
Model rows are written before GENOT. JAX is pinned to a small GPU fraction (env at top of file).

`compute_hox.py` -> `numerical_results/hox_consistency.csv` (row 2: Hox). Scores the (cell_type, Hox gene) shortlist from `literature.hox_consistency.static_hox_pairs`. Walk each E9.5 (`LEAF_TIME`) leaf of a shortlisted type up to its ~E9.0 (`ANCESTOR_TIME`, `ANCESTOR_TOL`) ancestor; keep only ancestors assigned by all three methods (common support). Per (method, cell_type, gene) with n >= 3 and nonzero variance on both sides, report the ancestor/descendant Pearson r and distortion = 1 - r. Columns: method, cell_type, gene, n, r, distortion.

`compute_context_scaling.py` -> `numerical_results/context_scaling.csv` (row 3, panels c/d). Runs `analysis/pipelines/context/context_scaling_cycle_consistency.py` directly into numerical_results. Focal leaves sampled stratified by terminal cell type (>= ~100/type where available) via leaf -> root -> leaf rollouts. Columns: time_to_mrca (context size), energy_distance, cell_type, leaf_cell_type. Computes only if the CSV is missing.

`compute_context_scaling_forward.py` -> `numerical_results/context_scaling_forward.csv` (row 3, panels e/f). Runs `analysis/pipelines/context/context_scaling_forward_layer.py` directly into numerical_results. Per subtree: pool once (anchor all leaves -> every internal node's state), enumerate the distinct `cross_section` antichain layers (one node per lineage, no holes), and split down from each (anchor the layer's pooled states, mask the rest, regenerate leaves). For each focal internal node v (stratified by dominant descendant fate) the layer crosses v's lineage at one ancestor f_v; score v's descendant leaves against truth. Each split-down is reused across all focals the layer crosses. Columns: n_cousins (= n_frontier - 1), energy_distance, time_to_mrca (= v_time - f_time), desc_cell_type. Computes only if the CSV is missing.

`compute_celltype_abundance.py` -> `numerical_results/celltype_abundance.csv` (row 3, panels d/f). Leaves per terminal fate over the whole tree set, from topology + the subtype table. CPU-only, ~1 min, no model. The per-type panels need it because neither context sampler's row count reflects abundance. Computes only if the CSV is missing.

`compute_context_scaling_genot.py` -> `numerical_results/context_scaling_genot.csv` (row 3, c+e reference lines). Context-blind GENOT baselines (Sinkhorn / cbFGW) on the same stratified focal population, subsampled to `N_REF` focals per panel. Ancestral: focal leaf -> subtree-root time -> leaf time. Forward: anchor descendant leaf of v -> v's time -> leaf time. One energy distance per focal (vs the same canonical target). Columns: panel, method, energy_distance. JAX is pinned to a small GPU fraction.

## data sources
- row 1: model coupling pkl + scvi embedding / cell-type annotations from `MouseEmbryoDataset` (`analysis/lib/celltype_map`).
- row 2: the three couplings (assignment metrics), the model splitter + the two GENOT flow models (clonal/marginal distortion), and the three assignment pkls + `MouseEmbryoDataset` expression (Hox), all over `MouseEmbryoDataset` trees / leaf states.
- row 3: computed on demand into the three `context_scaling*.csv` caches; canonical scripts `analysis/pipelines/context/context_scaling_cycle_consistency.py` and its forward dual `context_scaling_forward_layer.py`.

## plotting + assembly (read only `numerical_results/`)

`style.py`: shared config -- paths (model assignments / couplings / GENOT pkls), the method trio (`METHOD_COLORS`: teal / purple / green), the germ-layer alluvial palette, the row-3 `CONTEXT_DIR_COLORS` (ancestral = teal, forward = blue; not read by the plot scripts), sampling constants, geometry (`PAGE_W_IN=6.5`, point conversions, `BOX_*_IN` box layout), and `apply_rc` (Latin Modern Sans rcParams -- registered from `utils/fonts/`, text saved as vector paths). Each plot script exposes render functions writing one panel SVG into `panels/` and returning its size in inches; nothing below 10 pt except the row-1 type axis (`FS_TYPE`).

`plot_transition_map.py` -> `panels/A_transition_map.svg`. Transposed alluvial: time flows top->bottom over 6 levels, the `N_TOP_TYPES` (=90) most abundant types spread across the wide axis, ordered by `sankey._minimize_edge_distance`, ribbons by `sankey._make_ribbon`. Germ layer = colour; node area ~ sqrt(observed_count).

`plot_metrics.py` -> `panels/B_metrics.svg`. Six horizontal-lollipop mini-panels (one per row-2 metric), lineage inpainting on top, all lower = better, point estimates only. The linkage average drops the `t8.5_to_t8.0` and `t8.0_to_t7.5` tags. Germ-layer crossing plots the pooled per-link row (tag `""`), not a mean of the `bin_*` rows. Method names label the leftmost panel only.

`plot_context_scaling.py` -> `panels/{C_context_box,D_context_forward,E_context_box_celltype,F_context_forward_celltype}.svg` (drawn on the figure as panels c / e / d / f respectively -- the filename letters are internal), `panels/G_context_legend.svg`, `panels/{T_anc,T_desc}.svg`. The four row-3 column panels plus the shared legend strip and two box titles. The two average panels (`render_box` = panel c, `render_forward` = panel e) share `_overall_panel`: teal model mean +/- SEM trend + dotted horizontal GENOT reference lines from `context_scaling_genot.csv`. Ancestral x = `time_to_mrca` binned into `S.CONTEXT_BIN_EDGES` (6 uniform 0.25-wide bins over [0, 1.5], points at the bin centres; 0.1-wide bins leave the sparse band just above 0 too thin for the per-type cut). Descendant x = `n_cousins`, a plain integer count TRUNCATED at `FWD_MAX_COUSINS` = 8; rows past the cap are dropped. The average and per-type panels of a box share one axis exactly. The two per-type panels (`render_box_celltype` = panel d, `render_forward_celltype` = panel f) use `_celltype_panel`: a grey per-type mean trend for each fate outside `S.CONTEXT_EXCLUDE_TYPES` (blood / extraembryonic) with >= `S.CONTEXT_MIN_LEAVES` leaves in `celltype_abundance.csv` AND >= `S.CELLTYPE_MIN_PER_POINT` rows at every point on the axis, highlighting the two with the steepest (most / least negative) energy-distance slope, fitted on continuous `time_to_mrca` ancestrally and on the truncated cousin count on the descendant side. The two highlighted fates carry mean +/- SEM bars (the grey trends carry none). Their legend labels are abbreviated by `_short` (`LEGEND_PHRASES` then `LEGEND_WORDS`, word-boundary substitution so parenthesised region names shorten too) to fit inside the 1.45 in column. `render_legend` -> G; `render_box_title` -> T_anc / T_desc (tightly-cropped text-path SVGs).

`assemble.py` -> `imgs/fig3.{svg,pdf,png}`. Re-renders all panels, then nests them as positioned nested `<svg>` viewports in one master SVG laid out in points (per-panel id prefixing so glyph / clip defs can't collide). Rows 1 and 2 are stacked (tighter `ROW_GAP_AB_IN` between them); row 3 is two rounded outline boxes (`_box_border`, a stroked path with a top-edge gap holding the box title) each wrapping an (average | per type) pair, with the shared legend below. Sizing from the `BOX_*_IN` / `ROW_GAP_*_IN` constants. Bold lowercase panel letters a-f (`_letter_svg`, `PANEL_LETTER_FS=16`) are placed last, in reading order, each at its panel's top-left corner: a/b share the page-left x (so they stack) and c-f share the row-3 panel-top baseline (each rendered in an identical fixed box so the glyphs line up exactly).

## run order
    python -m analysis.figures.summary_qc.compute_assignment_metrics
    python -m analysis.figures.summary_qc.compute_transition_map
    python -m analysis.figures.summary_qc.compute_distortion              # slow, GPU (torch + jax)
    python -m analysis.figures.summary_qc.compute_hox
    python -m analysis.figures.summary_qc.compute_celltype_abundance       # panels d/f: rare-fate cut (cpu, ~1 min)
    python -m analysis.figures.summary_qc.compute_context_scaling          # panels c/d: ancestral cycle consistency
    python -m analysis.figures.summary_qc.compute_context_scaling_forward  # panels e/f: forward dual (slow, GPU)
    python -m analysis.figures.summary_qc.compute_context_scaling_genot    # c+e reference lines (GPU, jax)
    python -m analysis.figures.summary_qc.assemble                         # renders panels/ + fig3.{svg,pdf,png}
