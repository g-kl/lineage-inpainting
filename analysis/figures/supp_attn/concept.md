# concept (attention supplement)

this supplement shows (1) what the learned \gamma are, (2) how the attention weights change over flow time and (3) how attention is paid to observed vs. generated tokens.

## panels

**a** the learned bias scale. lollipop chart per head.

**b** the share of a masked node's attention row that lands on the observed leaves, over flow time. one curve per head.

**c** an example attention matrix and how it evolves over flow time, for one head (`style.EXAMPLE_HEAD`, picked by eye; see below).

# implementation

compute is split from plotting. `compute_attention.py` writes small caches into numerical_results/, and `plot_panels.py` + `assemble.py` read only those and write into imgs/. both folders are gitignored. `style.py` holds the paths, palette, type sizes and panel geometry.

the model is the one the main figures use: `EXPERIMENT_DIR` and `STEP` are imported from `analysis/figures/summary_qc/style.py`, not copied. splitter only (`BridgeDeltaTreeFlow`, 4 layers x 4 heads, ema weights, VP-SDE bridge). attention is read by wrapping each layer's `nn.MultiheadAttention` so it returns its weights, then running the ordinary inference rollout: `split_layer_batch`, 16 reverse-SDE steps, observed leaves clamped as anchors and everything above them masked.

rollout step $k$ runs the network at interpolant time $s = 1 - k/16$, the bridge's diffusion clock ($s = 1$ noise, $s = 0$ data), so the 16 steps cover $s = 1$ down to $s = 1/16$. panel b's $s$ axis is inverted so the rollout reads left to right, as in panel c.

128 subtrees of 24 to 96 leaves, rooted at internal nodes between E7 and E9.5, are drawn with seed 0. each subtree graph is built in dfs order from its root, so the node order, and with it the rollout noise, is the same on every run.

## compute

`compute_attention.py` writes into numerical_results/. one gpu, a few minutes, most of it loading the dataset.

- `scales.csv`: gamma per (layer, head), the model's `attention_bias_scale`. gamma multiplies the patristic distance `pat(i, j) = t_i + t_j - 2 t_mrca(i, j)` and is added to the attention logit before the softmax, so it is in logits per day of branch length.
- `rollout.csv`: per (s, layer, head), the share of a masked node's attention row that lands on the observed leaves, averaged over every masked query row of the 128 subtrees. `uniform` is the share uniform attention would give.
- `example.npz`: the first drawn subtree with 40 to 64 leaves. its attention at every step for every (layer, head), node times, observed mask and s.

## plot + assemble

`plot_panels.py` writes imgs/panels/{gamma,share,example}.svg.

- a: gamma sorted, coloured by layer.
- b: one curve per (layer, head), dashed line at the uniform share.
- c: layer 2 head 2 (`style.EXAMPLE_HEAD`), picked by eye as the head whose masked key columns are dark for every query row at $s = 1$ and that still changes over the rollout. at rollout steps 0, 5, 10, 15 (`style.STEPS`). rows are queries, columns are keys, both ordered by node time so the observed cells are the block on the right. colour is `log2(attention / uniform)`, uniform being 1/n on the n-node subtree.

`assemble.py` re-renders the panels and nests them into imgs/supp_attn.{svg,pdf,png}, a and b side by side over c, with `utils/svg_assembly.py`. panel letters from `utils/figure_labels.panel_letter_svg`.

## imports

- `analysis/figures/summary_qc/style.py`: `EXPERIMENT_DIR`, `STEP`, `base_dir`
- `datasets/tree_record.py`: `_leaf_counts`, `collate_subtrees`
- `splitters/bridge_tree_flow.py`: the model and `split_layer_batch`. `splitters/bridges/vpsde.py`: the rollout's $s$ grid
- `utils/figure_labels.py`: Latin Modern Sans, panel letters. `utils/svg_assembly.py`: nesting and the svg/pdf/png export

## data sources

- model: `outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none_57ab441ec3fcecf81d545755db5ddb09/`, `config.yaml` and `split_ema_step{STEP}.pt`
- trees and observed leaf states: the run's `MouseEmbryoDataset` (`qiu24_integrated_hvgs_and_scvi.h5ad`, `X_scvi`), instantiated from its config

## run order (lt conda env, from the repo root)

    python -m analysis.figures.supp_attn.compute_attention
    python -m analysis.figures.supp_attn.assemble

only assemble needs rerunning for cosmetics.
