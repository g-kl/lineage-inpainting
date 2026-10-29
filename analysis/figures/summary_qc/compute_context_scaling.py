"""row-3 cache: context scaling of subtree cycle consistency.

The heavy computation (stratified subtrees, leaf->root->leaf rollouts) lives in the
canonical script analysis/pipelines/context/context_scaling_cycle_consistency.py. This
script runs it with its output in this figure's numerical_results/. We only compute if
the cache is missing.

columns of interest (see the canonical script): time_to_mrca (context size),
energy_distance (recovered vs. true leaf set), cell_type (the predicted ROOT/MRCA
state at the root's early time), and leaf_cell_type (the focal leaf's ground-truth
terminal fate, used by the figure to stratify by terminal cell type).

writes numerical_results/context_scaling.csv.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from analysis.figures.summary_qc import style as S

CSV_NAME = "context_scaling.csv"


def main():
    out = os.path.join(S.NR, CSV_NAME)
    if os.path.exists(out):
        print(f"{out} exists; nothing to do (delete it to recompute)")
        return
    print(f"{out} missing; running the canonical computation directly into "
          f"numerical_results (analysis/pipelines/context/context_scaling_cycle_consistency.py)...",
          flush=True)
    from analysis.pipelines.context import context_scaling_cycle_consistency as csc
    csc.main(out_dir=S.NR, csv_name=CSV_NAME, plots=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
