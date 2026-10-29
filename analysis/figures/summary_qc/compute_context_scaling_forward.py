"""row-3 cache for panels e/f: forward context scaling (more cousins -> better descendants).

The heavy computation (stratified internal focal nodes, pooled cross-section
frontiers, split-down rollouts) lives in the canonical script
analysis/pipelines/context/context_scaling_forward_layer.py. This script runs it with
its output in this figure's numerical_results/. We only compute if the cache is
missing.

For a focal node v we predict its descendants by splitting down from a complete
cross-section layer above it; the layer crosses v's lineage at one node f_v and the
rest are v's same-level cousins. The figure bins by how many cousins the layer
reveals (n_cousins = n_frontier - 1) and shows the energy distance falling as more
cousin context is added.

columns of interest (see the canonical script): n_cousins (context = cousins the
layer reveals), energy_distance (recovered vs. true descendants), time_to_mrca
(layer position = v_time - f_time), desc_cell_type (dominant terminal fate).

writes numerical_results/context_scaling_forward.csv.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from analysis.figures.summary_qc import style as S

CSV_NAME = "context_scaling_forward.csv"


def main():
    out = os.path.join(S.NR, CSV_NAME)
    if os.path.exists(out):
        print(f"{out} exists; nothing to do (delete it to recompute)")
        return
    print(f"{out} missing; running the canonical computation directly into "
          f"numerical_results "
          f"(analysis/pipelines/context/context_scaling_forward_layer.py)...",
          flush=True)
    from analysis.pipelines.context import context_scaling_forward_layer as fwd
    fwd.main(out_dir=S.NR, csv_name=CSV_NAME, plots=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
