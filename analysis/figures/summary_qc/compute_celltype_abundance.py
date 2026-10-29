"""row-3 cache: how many leaves each terminal fate has.

The row-3 per-type panels stratify by terminal fate, but the samplers behind both
context_scaling caches draw a fixed PER_TYPE_TARGET focals per fate, so a fate's row
count says nothing about how common it is -- a fate with 300 leaves in the whole tree
set and one with 300,000 both come out at ~100 focals. The per-type panels need the
real abundance to drop the rare labels, whose trends are fitted on a handful of
distinct clones no matter how many rows they contribute.

Cheap and CPU-only: walk the tree topology, label every leaf from the subtype table,
count. ~1 min, no model and no expression matrix.

writes numerical_results/celltype_abundance.csv (cell_type, n_leaves).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd

from analysis.figures.summary_qc import style as S

DATA = os.path.join(S.base_dir, "..", "data", "embryos")
TREES = os.path.join(DATA, "final_topology.h5td")
TYPE_INFO = os.path.join(DATA, "final_type_info_v2_lower.csv")
SUBTYPES = os.path.join(DATA, "final_subtype_v2_lower.csv")
CSV_NAME = "celltype_abundance.csv"


def main():
    out = os.path.join(S.NR, CSV_NAME)
    if os.path.exists(out):
        print(f"{out} exists; nothing to do (delete it to recompute)")
        return
    import treedata

    ti = pd.read_csv(TYPE_INFO)
    sub = pd.read_csv(SUBTYPES, index_col=0)["cell_subtype"].str.lower()
    ct = sub.map(dict(zip(ti.cell_subtype, ti.cell_type)))

    td = treedata.read_h5td(TREES)
    leaves = []
    for name in td.obst:
        g = td.obst[name]
        leaves += [n for n in g.nodes() if g.out_degree(n) == 0]
    counts = ct.reindex(pd.Index(leaves).unique()).value_counts()
    counts.rename_axis("cell_type").rename("n_leaves").to_csv(out)
    print(f"wrote {out}: {len(counts)} fates, "
          f"{counts.min()}-{counts.max()} leaves", flush=True)


if __name__ == "__main__":
    main()
