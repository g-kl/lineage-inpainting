"""the held-out clone split, paths, and a dataset with the held-out clones removed.

the 11 clones were drawn once with np.random.default_rng(0): for each stage E7.5-E9.5,
two replicates without replacement and one clone from each, plus one E10.0 clone
(E10.0 is a single embryo). held-out share of each stage's leaves: E7.5 0.33,
E8.0 0.11, E8.5 0.20, E9.0 0.14, E9.5 0.34, E10.0 0.02 (0.23 overall).

    python -m analysis.figures.supp_holdout.common    # load once, run the leak asserts
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from datasets.mouse_embryo import MouseEmbryoDataset

HOLDOUT = [
    "E7.5-R2-C2", "E7.5-R3-C1",
    "E8.0-R1-C3", "E8.0-R3-C2",
    "E8.5-R2-C2", "E8.5-R3-C3",
    "E9.0-R2-C2", "E9.0-R3-C2",
    "E9.5-R1-C1", "E9.5-R3-C2",
    "E10.0-R1-C6",
]

RESULTS = os.path.join(ROOT, "analysis", "results", "hold_out_metrics")
OUTPUTS = os.path.join(ROOT, "outputs")
RUN_NAME = "hold_out_vpsde"     # main.py writes outputs/hold_out_vpsde_<hash>/
STEP = 20000

# genot variants trained on the train split: name -> (paper script, line, replacement).
# the paper's own settings are eps 5e-2 and fused penalty 1.0 (alpha = 1 / (1 + fp)).
GENOT_VARIANTS = {
    "genot_cbfgw_fp0.1": ("baselines/genot_fgw_mm.py", "FUSED_PENALTY = 1.0", "FUSED_PENALTY = 0.1"),
    "genot_cbfgw_fp1": ("baselines/genot_fgw_mm.py", "FUSED_PENALTY = 1.0", "FUSED_PENALTY = 1.0"),
    "genot_cbfgw_fp10": ("baselines/genot_fgw_mm.py", "FUSED_PENALTY = 1.0", "FUSED_PENALTY = 10.0"),
    "genot_sinkhorn_eps0.01": ("baselines/genot_sinkhorn.py", "EPSILON = 5e-2", "EPSILON = 1e-2"),
    "genot_sinkhorn_eps0.05": ("baselines/genot_sinkhorn.py", "EPSILON = 5e-2", "EPSILON = 5e-2"),
    "genot_sinkhorn_eps0.2": ("baselines/genot_sinkhorn.py", "EPSILON = 5e-2", "EPSILON = 2e-1"),
}

# one evaluate.py task each. *_all are the paper's models (trained on every tree).
METHODS = ["vpsde", "vpsde_all", *GENOT_VARIANTS,
           "genot_sinkhorn_all", "genot_cbfgw_all"]
PARITY_METHODS = ["vpsde_all", "genot_sinkhorn_all", "genot_cbfgw_all"]


class HeldOutMouseEmbryoDataset(MouseEmbryoDataset):
    """MouseEmbryoDataset without the held-out clones: no records, trees, leaf states or bank cells."""

    def __init__(self, holdout_clones, **kwargs):
        assert sorted(holdout_clones) == sorted(HOLDOUT), holdout_clones
        super().__init__(**kwargs)
        held = set(holdout_clones)
        assert held <= set(self.trees), held - set(self.trees)
        held_leaves = {n for t in held for n in self.trees[t] if self.trees[t].out_degree(n) == 0}
        n_cells, n_leaf_states = self.states.n_obs, len(self.leaf_states)

        # the aligner's marginal reference and the nn pairing bank both read states +
        # reference_features (utils/reference.py), so the cells go too, not just the records
        keep = ~self.states.obs["clone"].isin(held).to_numpy()
        self.records = {k: v for k, v in self.records.items() if k not in held}
        self.trees = {k: v for k, v in self.trees.items() if k not in held}
        self.states = self.states[keep]
        self.reference_features = self.reference_features[keep]
        self.name_to_idx = {n: i for i, n in enumerate(self.states.obs_names)}
        kept = set(self.states.obs_names)
        self.leaf_states = {k: v for k, v in self.leaf_states.items() if k in kept}

        assert not held & set(self.records) and not held & set(self.trees)
        assert not self.states.obs["clone"].isin(held).any()
        assert held_leaves.isdisjoint(kept) and held_leaves.isdisjoint(self.leaf_states)
        assert len(self.reference_features) == self.states.n_obs
        # nothing but held-out leaves left leaf_states
        assert n_leaf_states - len(self.leaf_states) == len(held_leaves), \
            (n_leaf_states, len(self.leaf_states), len(held_leaves))
        print(f"held out {len(held)} clones: {len(held_leaves)} leaves, "
              f"{n_cells - self.states.n_obs} of {n_cells} cells removed from the bank, "
              f"{len(self.records)} training trees", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    HeldOutMouseEmbryoDataset(HOLDOUT)
